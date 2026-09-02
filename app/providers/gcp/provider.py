"""The Google Cloud provider: a confidential G4 VM in London, behind the contract.

What is different from RunPod, and why it is shaped this way:

  * The compute unit is a Compute Engine instance identified by (zone, name);
    the id the driver carries is "zone/name".
  * There is no public proxy. The VM has NO external IP. Reaching its ports is
    the provider's job (ensure_access): IAP tunnels to loopback by default
    (Phase 4), or the internal address when the client shares the VPC.
  * Persistence is a Hyperdisk attached with autoDelete=false; a ZONAL disk
    pins the VM to its zone, a REGIONAL (HA) one lets create rotate zones.
  * Secrets never enter instance metadata. They are synced to Secret Manager
    (replicated only in the region) before each create, and the VM's own
    identity fetches them at boot.
  * The platform kills the VM itself after max_run_hours — a backstop behind
    podlink's idle watchdog, covering the VM and nothing else.
  * POD DOWN deletes by default (clearest cost boundary); `stop` is opt-in.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from ... import stack as stack_contract
from . import bootstrap
from .api import AdcTokenSource, GcpApi, GcpApiError
from .config import GcpConfig, from_env

# Instance ids the driver hands back: "<zone>/<name>", both RFC1035-ish.
_INSTANCE_ID_RE = re.compile(r"^[a-z][a-z0-9-]{1,40}/[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
# Label values: lowercase, [a-z0-9_-], max 63.
_LABEL_RE = re.compile(r"[^a-z0-9_-]")

STATE_PATH = Path(__file__).resolve().parents[3] / "pod_state.gcp.json"
DATA_DEVICE = "podlink-data"        # deviceName of the persistent disk => /dev/disk/by-id/google-podlink-data
SCRATCH_DEVICE = "podlink-scratch"  # volume-less mode


def _label(value: str) -> str:
    return _LABEL_RE.sub("-", value.lower())[:63]


class GcpProvider:
    """Drives one confidential GPU VM for app/driver.py."""

    name = "gcp"
    create_error_types = (GcpApiError,)

    def __init__(self, api: GcpApi | None = None) -> None:
        self.cfg: GcpConfig = from_env()
        self._api = api                     # injected in tests; built by authenticate()
        self._internal_ip: str | None = None   # resolved by ensure_access in internal mode

    # --- configuration -----------------------------------------------------

    def reload_config(self) -> None:
        self.cfg = from_env()               # raises on a bad value; the caller rolls back
        self._internal_ip = None

    def snapshot_fields(self) -> dict:
        return {
            "provider": self.name,
            "persistence_configured": self.persistence_configured(),
            "persistence_id": self.cfg.data_disk or None,
            "persistence_label": "Hyperdisk",
            "persistence_off_hint": f"none — scratch disk ({self.cfg.volume_gb} GB) destroyed on POD DOWN",
            "llm_model_id": self.cfg.stack["llm_model_id"],
        }

    def preflight(self) -> list[tuple[str, str]]:
        """What would stop a launch on GCP, checked without spending anything."""
        c, rows = self.cfg, []
        rows.append(("ok" if c.project else "fail",
                     f"PODLINK_GCP_PROJECT = {c.project}" if c.project else "PODLINK_GCP_PROJECT not set"))
        try:
            AdcTokenSource().token()
            rows.append(("ok", "Application Default Credentials present"))
        except ImportError:
            rows.append(("fail", "google-auth not installed — pip install -r requirements-gcp.txt"))
        except Exception as e:  # noqa: BLE001 — the message is the finding
            rows.append(("fail", f"credentials: {e}"))
        rows.append(("ok" if c.boot_image else "fail",
                     f"boot image: {c.boot_image}" if c.boot_image
                     else "PODLINK_GCP_BOOT_IMAGE not set (the golden image — see pod_image/gcp)"))
        rows.append(("ok" if c.image else "fail",
                     f"container image: {c.image}" if c.image else "PODLINK_GCP_IMAGE / PODLINK_IMAGE not set"))
        if c.image and not c.image.split("/", 1)[0].endswith("pkg.dev"):
            rows.append(("warn", "container image is not in Artifact Registry — the VM will pull cross-region "
                                 "and needs registry credentials it does not have"))
        rows.append(("info", f"zones: {', '.join(c.zones)} · {c.machine_type} · "
                             f"{'confidential (SEV + GPU TEE)' if c.confidential else 'NOT confidential'} · "
                             f"{c.provisioning_model.lower()}"))
        if c.data_disk:
            rows.append(("ok", f"persistence: {c.data_disk_scope} Hyperdisk {c.data_disk}"
                               + (" — zone pinned to " + c.zones[0] if not c.rotates_zones else " — zone rotation on")))
        else:
            rows.append(("warn", f"persistence: none — scratch disk ({c.volume_gb} GB) destroyed on POD DOWN; "
                                 "weights and the container image re-download each launch"))
        rows.append(("ok" if c.kms_key else "warn",
                     "disks: customer-managed key (CMEK)" if c.kms_key
                     else "disks: Google-managed keys — set PODLINK_GCP_KMS_KEY for CMEK"))
        rows.append(("ok" if c.service_account else "warn",
                     f"VM identity: {c.service_account}" if c.service_account
                     else "VM identity: the project's DEFAULT compute service account — over-privileged; "
                          "set PODLINK_GCP_SERVICE_ACCOUNT"))
        rows.append(("ok" if c.max_run_hours else "warn",
                     f"kill switch: VM self-deletes after {c.max_run_hours:g} h" if c.max_run_hours
                     else "kill switch OFF (PODLINK_GCP_MAX_RUN_HOURS=0)"))
        rows.append(("ok" if c.cost_per_hr is not None else "warn",
                     f"cost meter: ${c.cost_per_hr:.2f}/hr from the profile" if c.cost_per_hr is not None
                     else "cost meter blank — set PODLINK_GCP_COST_PER_HR (GCP reports no rate on the instance)"))
        if c.access == "iap":
            rows.append(("ok" if shutil.which("gcloud") else "fail",
                         "gcloud present (IAP tunnels)" if shutil.which("gcloud")
                         else "gcloud not on PATH — required for IAP tunnels (or set PODLINK_GCP_ACCESS=internal)"))
        else:
            rows.append(("info", "access: internal IP — health tiles need this host on the VPC"))
        return rows

    # --- authentication ----------------------------------------------------

    def authenticate(self) -> None:
        if not self.cfg.project:
            raise RuntimeError("PODLINK_GCP_PROJECT is not set")
        if self._api is None:
            self._api = GcpApi(self.cfg.project, AdcTokenSource())
        self._api.project = self.cfg.project   # a profile switch may have changed it

    @property
    def api(self) -> GcpApi:
        if self._api is None:
            raise RuntimeError("provider not authenticated")
        return self._api

    # --- inventory ---------------------------------------------------------

    def valid_instance_id(self, instance_id: str) -> bool:
        return bool(_INSTANCE_ID_RE.match(instance_id))

    @staticmethod
    def _split(instance_id: str) -> tuple[str, str]:
        zone, name = instance_id.split("/", 1)
        return zone, name

    @staticmethod
    def _zone_of(instance: dict) -> str:
        return (instance.get("zone") or "").rsplit("/", 1)[-1]

    def list_instances(self) -> list[dict]:
        """WHITELISTED rows across the configured zones — never the raw record,
        which carries metadata (the cloud-init) and service-account details."""
        rows = []
        for zone in self.cfg.zones:
            for inst in self.api.list_instances(zone):
                rows.append({
                    "id": f"{zone}/{inst.get('name')}",
                    "name": inst.get("name"),
                    "status": inst.get("status"),
                    "gpu": (inst.get("guestAccelerators") or [{}])[0].get("acceleratorType", "").rsplit("/", 1)[-1]
                           or inst.get("machineType", "").rsplit("/", 1)[-1],
                    "cost_per_hr": self.cfg.cost_per_hr,
                })
        return rows

    def find_existing(self) -> dict | None:
        for zone in self.cfg.zones:
            inst = self.api.get_instance(zone, self.cfg.instance_name)
            if inst is not None:
                return inst
        return None

    def get_instance(self, instance_id: str) -> dict | None:
        zone, name = self._split(instance_id)
        return self.api.get_instance(zone, name)

    def instance_id(self, instance: dict) -> str:
        return f"{self._zone_of(instance)}/{instance['name']}"

    def status_of(self, instance: dict) -> str | None:
        return instance.get("status") if instance else None

    def is_running(self, instance: dict) -> bool:
        return bool(instance) and instance.get("status") == "RUNNING"

    def is_up(self, instance: dict) -> bool:
        # GCE has no "container is up" signal; RUNNING is as far as the platform
        # sees. The driver's readiness gate probes the services from here.
        return self.is_running(instance)

    def cost_per_hr(self, instance: dict) -> float | None:
        return self.cfg.cost_per_hr          # from the profile; the API reports no rate

    # --- lifecycle ---------------------------------------------------------

    def resume(self, instance: dict) -> None:
        self.api.start_instance(self._zone_of(instance), instance["name"])

    def prepare_create(self, session) -> dict | None:
        """Nothing to look up ahead of time on GCP; the context only tracks
        whether the secrets have been synced yet (create_once does that on the
        first attempt, because the driver hands the secret values to
        create_once, not here)."""
        if session.cancel.is_set():
            return None
        return {"synced": False}

    def create_once(self, ctx: dict, secrets: dict, attempt: int) -> dict:
        """ONE create attempt in the zone this attempt maps to.

        Ordering matters: the secrets reach Secret Manager before the VM exists,
        so its boot script never races an empty version. An instance of our
        name already present and TERMINATED (a previous `stop`) is STARTED
        instead of re-created — that is the stop/start lifecycle's resume.
        """
        cfg = self.cfg
        if not ctx.get("synced"):
            for key, secret_name in cfg.secret_names.items():
                if self.api.secret_latest(secret_name) != secrets[key]:   # only add a version on change
                    self.api.secret_put(secret_name, secrets[key], cfg.region)
            ctx["synced"] = True
        existing = self.find_existing()
        if existing is not None:
            if existing.get("status") == "TERMINATED":          # stopped earlier -> start it
                zone = self._zone_of(existing)
                self.api.start_instance(zone, existing["name"])
                return self.api.get_instance(zone, existing["name"]) or existing
            raise GcpApiError(f"an instance named {cfg.instance_name} already exists "
                              f"({existing.get('status')}) — POD DOWN it first", code="ALREADY_EXISTS")
        zone = cfg.zone_for(attempt)
        return self.api.insert_instance(zone, self.instance_body(zone))

    def is_retryable_create_error(self, exc: Exception) -> bool:
        return isinstance(exc, GcpApiError) and exc.retryable

    def create_phase(self, ctx: dict, attempt: int, retries: int) -> str:
        return (f"creating {self.cfg.machine_type} in {self.cfg.zone_for(attempt)}"
                f"{' (confidential)' if self.cfg.confidential else ''} — attempt {attempt}/{retries}")

    def capacity_note(self, attempt: int, retries: int, delay: int) -> str:
        nxt = self.cfg.zone_for(attempt + 1)
        where = f"trying {nxt} next" if self.cfg.rotates_zones else f"zone pinned to {nxt} by the zonal disk"
        return f"no {self.cfg.machine_type} capacity — attempt {attempt}/{retries}; {where} in {delay}s"

    def create_exhausted_message(self, retries: int, delay: int) -> str:
        return (f"no capacity for {self.cfg.machine_type} in {', '.join(self.cfg.zones)} after {retries} "
                f"attempts (~{retries * delay // 60} min). Check the quota page, try flex-start, or "
                f"try again later — press POD UP to keep trying.")

    def terminate(self, instance_id: str) -> None:
        zone, name = self._split(instance_id)
        if self.cfg.down_action == "stop":
            self.api.stop_instance(zone, name)      # boot disk kept (bills); faster next start
        else:
            self.api.delete_instance(zone, name)    # data disk survives (autoDelete=false)

    # --- access lifecycle ---------------------------------------------------

    def ensure_access(self, instance_id: str) -> None:
        if self.cfg.access == "internal":
            inst = self.get_instance(instance_id)
            nics = (inst or {}).get("networkInterfaces") or []
            ip = nics[0].get("networkIP") if nics else None
            if not ip:
                raise RuntimeError("instance has no internal address yet")
            self._internal_ip = ip
            return
        # Phase 4: the supervised IAP tunnel manager lands here. Until then a
        # launch in iap mode fails loudly at this exact point rather than
        # pretending the ports are reachable.
        raise RuntimeError("PODLINK_GCP_ACCESS=iap: the IAP tunnel manager is not implemented yet "
                           "(Phase 4) — use PODLINK_GCP_ACCESS=internal from a host on the VPC")

    def release_access(self) -> None:
        self._internal_ip = None             # nothing to close until Phase 4's tunnels

    # --- endpoints + local state -------------------------------------------

    def service_urls(self, instance_id: str) -> dict[str, str]:
        """Loopback tunnel ports (iap) or the VM's internal address (internal).
        Pure: the internal IP is whatever ensure_access last resolved, and a
        placeholder host before that — never a lookup."""
        if self.cfg.access == "internal":
            host = self._internal_ip or "pending.internal"
            return {svc: f"http://{host}:{port}" for svc, port in stack_contract.SERVICE_PORTS.items()}
        return {svc: f"http://127.0.0.1:{port}" for svc, port in self.cfg.local_ports.items()}

    def stack_config(self) -> dict:
        return dict(self.cfg.stack, image=self.cfg.image or None)

    def client_env(self, instance_id: str, embedding_dim: int | None) -> str:
        return stack_contract.client_env_block(self.service_urls(instance_id), self.cfg.stack, embedding_dim)

    def persistence_configured(self) -> bool:
        return bool(self.cfg.data_disk)

    def write_state(self, instance_id: str) -> None:
        state = {"instance_id": instance_id, "project": self.cfg.project,
                 "service_urls": self.service_urls(instance_id),
                 "models": {"llm": self.cfg.stack["llm_model_id"], "served_as": self.cfg.stack["llm_served_name"],
                            "embedder": self.cfg.stack["embed_model_id"], "reranker": self.cfg.stack["rerank_model_id"]},
                 "image": self.cfg.image}
        STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")

    def clear_state(self) -> None:
        try:
            STATE_PATH.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 — best-effort, never fatal to a stop
            pass

    def state_file_instance_id(self) -> str | None:
        if not STATE_PATH.exists():
            return None
        try:
            return json.loads(STATE_PATH.read_text()).get("instance_id")
        except Exception:  # noqa: BLE001
            return None

    # --- the create payload ------------------------------------------------

    def instance_body(self, zone: str) -> dict:
        """The instances.insert body. Every security property lives here, and
        tests/test_gcp_provider.py asserts each one — change with care."""
        cfg = self.cfg
        project = cfg.project
        disk_type = f"zones/{zone}/diskTypes/hyperdisk-balanced"    # the ONLY boot type G4 accepts
        boot = {
            "boot": True, "autoDelete": True,
            "initializeParams": {"sourceImage": cfg.boot_image, "diskType": disk_type,
                                 "diskSizeGb": str(cfg.boot_disk_gb)},
        }
        if cfg.kms_key:
            boot["initializeParams"]["diskEncryptionKey"] = {"kmsKeyName": cfg.kms_key}
        if cfg.data_disk:                       # persistent: attach, never auto-delete
            source = (f"projects/{project}/regions/{cfg.region}/disks/{cfg.data_disk}"
                      if cfg.data_disk_scope == "regional"
                      else f"projects/{project}/zones/{zone}/disks/{cfg.data_disk}")
            data = {"boot": False, "autoDelete": False, "source": source, "deviceName": DATA_DEVICE}
            if cfg.kms_key:
                data["diskEncryptionKey"] = {"kmsKeyName": cfg.kms_key}
            device = DATA_DEVICE
        else:                                   # volume-less: scratch dies with the VM
            data = {"boot": False, "autoDelete": True, "deviceName": SCRATCH_DEVICE,
                    "initializeParams": {"diskType": disk_type, "diskSizeGb": str(cfg.volume_gb)}}
            if cfg.kms_key:
                data["initializeParams"]["diskEncryptionKey"] = {"kmsKeyName": cfg.kms_key}
            device = SCRATCH_DEVICE

        scheduling = {
            "onHostMaintenance": "TERMINATE",          # required for GPUs and for Confidential VM
            "automaticRestart": False,
            "provisioningModel": cfg.provisioning_model,
        }
        if cfg.provisioning_model == "SPOT":
            scheduling["preemptible"] = True
            scheduling["instanceTerminationAction"] = "DELETE"
        if cfg.max_run_hours:
            scheduling["maxRunDuration"] = {"seconds": str(int(cfg.max_run_hours * 3600))}
            scheduling["instanceTerminationAction"] = "DELETE"   # the platform-side kill switch

        body = {
            "name": cfg.instance_name,
            "machineType": f"zones/{zone}/machineTypes/{cfg.machine_type}",
            "disks": [boot, data],
            "networkInterfaces": [{"subnetwork": f"regions/{cfg.region}/subnetworks/{cfg.subnet}"}],  # NO accessConfigs => no external IP
            "scheduling": scheduling,
            "shieldedInstanceConfig": {"enableSecureBoot": True, "enableVtpm": True,
                                       "enableIntegrityMonitoring": True},
            "metadata": {"items": [
                {"key": "user-data", "value": bootstrap.render(cfg, device)},
                {"key": "enable-oslogin", "value": "TRUE"},          # IAM-gated SSH, no metadata keys
                {"key": "block-project-ssh-keys", "value": "TRUE"},
            ]},
            "labels": {"managed-by": "podlink",
                       "podlink-profile": _label(os.environ.get("PODLINK_PROFILE") or "base"),
                       "podlink-llm": _label(cfg.stack["llm_model_id"].rsplit("/", 1)[-1])},
            "tags": {"items": [cfg.network_tag]},                   # firewall target for the IAP range
        }
        if cfg.service_account:
            body["serviceAccounts"] = [{"email": cfg.service_account,
                                        "scopes": ["https://www.googleapis.com/auth/cloud-platform"]}]
        if cfg.confidential:
            body["confidentialInstanceConfig"] = {"enableConfidentialCompute": True,
                                                  "confidentialInstanceType": "SEV"}
        return body
