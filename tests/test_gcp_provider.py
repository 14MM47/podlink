"""Offline tests for the GCP provider — no project, no credentials, no network.

The provider talks to two REST APIs through one small client (app/providers/
gcp/api.py). These tests replace that client with an in-memory fake and drive
the provider exactly as app/driver.py does, so the create payload, the retry
classification, the zone rotation and the terminate paths are all checked
before a single VM exists.

The assertions that matter most are the security properties of the create
payload. Each one is a decision that was made on purpose; if a test here fails,
read the comment on the assertion before changing the code.

Run: python3 tests/test_gcp_provider.py
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- fake secrets (the driver reads bearer + HF token through app.vendored) --
fake_secrets = types.ModuleType("_secrets")
fake_secrets.runpod_api_key = lambda: "rk_dummy"
fake_secrets.hf_token = lambda: "hf_dummy"
fake_secrets.bearer_token = lambda: "bearer_dummy"
sys.modules["_secrets"] = fake_secrets

from app import driver as rd, providers                       # noqa: E402
from app.providers.gcp import api as api_mod, provider as prov_mod   # noqa: E402
from app.providers.gcp.api import GcpApiError                 # noqa: E402
from app.providers.gcp.provider import GcpProvider            # noqa: E402
from app.session import PodSession                           # noqa: E402

rd.POLL_S = 0.05
SECRETS = {"bearer": "bearer_dummy", "hf": "hf_dummy"}

BASE_ENV = {
    "PODLINK_PROVIDER": "gcp",
    "PODLINK_GCP_PROJECT": "proj-1",
    "PODLINK_GCP_BOOT_IMAGE": "projects/proj-1/global/images/family/podlink-g4",
    "PODLINK_GCP_IMAGE": "europe-west2-docker.pkg.dev/proj-1/podlink/ragline-pod:2026-07b",
    "PODLINK_GCP_DATA_DISK": "podlink-data",
    "PODLINK_GCP_KMS_KEY": "projects/proj-1/locations/europe-west2/keyRings/kr/cryptoKeys/disks",
    "PODLINK_GCP_SERVICE_ACCOUNT": "podlink-vm@proj-1.iam.gserviceaccount.com",
    "PODLINK_GCP_COST_PER_HR": "5.40",
}


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise SystemExit(f"FAILED: {name}")


class FakeApi:
    """In-memory stand-in for GcpApi: records every call, scripts failures."""

    def __init__(self):
        self.project = "proj-1"
        self.instances: dict = {}        # (zone, name) -> record
        self.secrets: dict = {}          # name -> value
        self.calls: list = []            # ("kind", ...)
        self.insert_errors: list = []    # GcpApiError to raise on successive inserts
        self.inserted: list = []         # (zone, body)

    def get_instance(self, zone, name):
        self.calls.append(("get", zone, name))
        return self.instances.get((zone, name))

    def list_instances(self, zone):
        return [i for (z, _n), i in self.instances.items() if z == zone]

    def insert_instance(self, zone, body):
        self.inserted.append((zone, body))
        self.calls.append(("insert", zone))
        if self.insert_errors:
            raise self.insert_errors.pop(0)
        inst = {"name": body["name"], "zone": f"projects/proj-1/zones/{zone}", "status": "RUNNING",
                "networkInterfaces": [{"networkIP": "10.0.0.5"}]}
        self.instances[(zone, body["name"])] = inst
        return inst

    def delete_instance(self, zone, name):
        self.calls.append(("delete", zone, name))
        self.instances.pop((zone, name), None)

    def stop_instance(self, zone, name):
        self.calls.append(("stop", zone, name))
        self.instances[(zone, name)]["status"] = "TERMINATED"

    def start_instance(self, zone, name):
        self.calls.append(("start", zone, name))
        self.instances[(zone, name)]["status"] = "RUNNING"

    def secret_latest(self, name):
        self.calls.append(("secret_get", name))
        return self.secrets.get(name)

    log_sink_destination = "logging.googleapis.com/projects/proj-1/locations/europe-west2/buckets/podlink-europe-west2"

    def get_log_sink(self, name="_Default"):
        self.calls.append(("log_sink", name))
        return {"name": name, "destination": self.log_sink_destination}

    def secret_put(self, name, value, location):
        self.calls.append(("secret_put", name, location))
        self.secrets[name] = value


@contextlib.contextmanager
def gcp(**overrides):
    """A GcpProvider on a FakeApi, under BASE_ENV (+overrides), installed as the
    registry's active provider so app.driver drives it. Restores everything."""
    saved_env = {k: os.environ.get(k) for k in set(BASE_ENV) | set(overrides)}
    saved_active = (providers._ACTIVE, providers._ACTIVE_NAME)
    os.environ.update(BASE_ENV)
    for k, v in overrides.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        # Construction can raise (config validation is part of what is tested);
        # the env must be restored either way, so it lives inside the try.
        api = FakeApi()
        p = GcpProvider(api=api)
        if p.cfg.project:                   # a missing-project case tests preflight, not auth
            p.authenticate()
        providers._ACTIVE, providers._ACTIVE_NAME = p, "gcp"
        yield p, api
    finally:
        providers._ACTIVE, providers._ACTIVE_NAME = saved_active
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --- the create payload ----------------------------------------------------

def test_create_payload_security_properties():
    with gcp() as (p, api):
        body = p.instance_body("europe-west2-b")
        dump = json.dumps(body)
        # No external IP: an accessConfig on the NIC is exactly what would give it one.
        check("no accessConfigs (no external IP)", "accessConfigs" not in dump)
        # In-use encryption: SEV VM + the GPU TEE that comes with confidential G4.
        check("confidential compute on, SEV",
              body["confidentialInstanceConfig"] == {"enableConfidentialCompute": True,
                                                     "confidentialInstanceType": "SEV"})
        sch = body["scheduling"]
        check("host maintenance TERMINATE (GPU + CVM requirement)", sch["onHostMaintenance"] == "TERMINATE")
        check("no automatic restart", sch["automaticRestart"] is False)
        # The platform-side kill switch: 8h default, and DELETE not STOP.
        check("maxRunDuration 8h + DELETE",
              sch["maxRunDuration"] == {"seconds": "28800"} and sch["instanceTerminationAction"] == "DELETE")
        boot, data = body["disks"]
        # G4 accepts no Persistent Disk at all — Hyperdisk Balanced is the only boot type.
        check("boot disk is hyperdisk-balanced", boot["initializeParams"]["diskType"].endswith("hyperdisk-balanced"))
        check("boot disk CMEK", boot["initializeParams"]["diskEncryptionKey"]["kmsKeyName"].endswith("/disks"))
        # The persistent disk must outlive the VM — that is the whole point of it.
        check("data disk autoDelete=false", data["autoDelete"] is False)
        check("data disk is the zonal Hyperdisk", data["source"].endswith("zones/europe-west2-b/disks/podlink-data"))
        check("data disk CMEK", data["diskEncryptionKey"]["kmsKeyName"].endswith("/disks"))
        check("data disk device name", data["deviceName"] == "podlink-data")
        check("shielded VM on", body["shieldedInstanceConfig"]["enableSecureBoot"] is True)
        md = {i["key"]: i["value"] for i in body["metadata"]["items"]}
        check("OS Login on, project SSH keys blocked",
              md["enable-oslogin"] == "TRUE" and md["block-project-ssh-keys"] == "TRUE")
        check("user-data is cloud-config", md["user-data"].startswith("#cloud-config"))
        check("dedicated service account, cloud-platform scope",
              body["serviceAccounts"][0]["email"].startswith("podlink-vm@"))
        check("labels for cost attribution", body["labels"]["managed-by"] == "podlink")
        check("network tag for the IAP firewall rule", body["tags"]["items"] == ["podlink"])
        check("machine type g4-standard-48 in the zone",
              body["machineType"] == "zones/europe-west2-b/machineTypes/g4-standard-48")


def test_create_never_puts_secrets_in_the_payload():
    # The secret VALUES reach the VM via Secret Manager at boot, never via
    # metadata (readable by anyone with compute.instances.get).
    with gcp() as (p, api):
        ctx = p.prepare_create(PodSession())
        p.create_once(ctx, SECRETS, 1)
        zone, body = api.inserted[0]
        dump = json.dumps(body)
        check("no secret values anywhere in the insert body",
              "bearer_dummy" not in dump and "hf_dummy" not in dump)
        kinds = [c[0] for c in api.calls]
        check("secrets synced to Secret Manager BEFORE the insert",
              kinds.index("secret_put") < kinds.index("insert"))
        check("secrets replicated only in the region",
              all(c[2] == "europe-west2" for c in api.calls if c[0] == "secret_put"))
        check("secret names use the prefix",
              set(api.secrets) == {"podlink-bearer", "podlink-hf-token"})
        # A second create with unchanged secrets adds no new versions.
        api.calls.clear(); api.instances.clear()
        p.create_once(p.prepare_create(PodSession()), SECRETS, 1)
        check("unchanged secrets are not re-put", not any(c[0] == "secret_put" for c in api.calls))


def test_spot_and_volume_less_variants():
    with gcp(PODLINK_GCP_PROVISIONING="spot", PODLINK_GCP_DATA_DISK=None, PODLINK_VOLUME_GB="60") as (p, api):
        body = p.instance_body("europe-west2-c")
        sch = body["scheduling"]
        check("SPOT provisioning", sch["provisioningModel"] == "SPOT" and sch.get("preemptible") is True)
        scratch = body["disks"][1]
        check("volume-less: scratch disk dies with the VM", scratch["autoDelete"] is True)
        check("scratch sized from PODLINK_VOLUME_GB", scratch["initializeParams"]["diskSizeGb"] == "60")
        check("scratch device name", scratch["deviceName"] == "podlink-scratch")
        check("persistence_configured False => POD DOWN is destructive", p.persistence_configured() is False)
        check("snapshot says Hyperdisk / none", p.snapshot_fields()["persistence_id"] is None)


def test_no_kill_switch_when_disabled():
    with gcp(PODLINK_GCP_MAX_RUN_HOURS="0") as (p, api):
        sch = p.instance_body("europe-west2-b")["scheduling"]
        check("no maxRunDuration when 0", "maxRunDuration" not in sch)


# --- placement + retries ------------------------------------------------------

def test_zone_placement_rules():
    with gcp() as (p, api):                                  # zonal disk
        check("zonal disk pins every attempt to the first zone",
              [p.cfg.zone_for(n) for n in (1, 2, 3)] == ["europe-west2-b"] * 3 and not p.cfg.rotates_zones)
        check("capacity note says the zone is pinned", "pinned" in p.capacity_note(1, 5, 15))
    with gcp(PODLINK_GCP_DATA_DISK_SCOPE="regional") as (p, api):
        check("regional disk rotates b, c, b",
              [p.cfg.zone_for(n) for n in (1, 2, 3)] == ["europe-west2-b", "europe-west2-c", "europe-west2-b"])
        check("regional disk source is the region path",
              p.instance_body("europe-west2-c")["disks"][1]["source"].endswith("regions/europe-west2/disks/podlink-data"))
    with gcp(PODLINK_GCP_DATA_DISK=None) as (p, api):
        check("volume-less rotates too", p.cfg.rotates_zones)


def test_retry_classification():
    stockout = GcpApiError("no capacity", code="ZONE_RESOURCE_POOL_EXHAUSTED")
    quota = GcpApiError("quota", code="QUOTA_EXCEEDED")
    forbidden = GcpApiError("forbidden", http_status=403, code="forbidden")
    ratelimit = GcpApiError("slow down", http_status=429)
    with gcp() as (p, api):
        check("stockout is retryable", p.is_retryable_create_error(stockout))
        check("quota is NOT retryable", not p.is_retryable_create_error(quota))
        check("403 is NOT retryable", not p.is_retryable_create_error(forbidden))
        check("429 is retryable", p.is_retryable_create_error(ratelimit))
        check("a non-API exception is not ours", not p.is_retryable_create_error(RuntimeError("x")))


def test_driver_retries_rotating_zones_on_stockout():
    # Through the REAL driver loop: attempt 1 (zone b) hits a stockout, attempt 2
    # lands in zone c. Regional disk so rotation is allowed.
    saved = rd._sleep_or_cancel
    rd._sleep_or_cancel = lambda session, secs: False
    try:
        with gcp(PODLINK_GCP_DATA_DISK_SCOPE="regional") as (p, api):
            api.insert_errors = [GcpApiError("stockout", code="ZONE_RESOURCE_POOL_EXHAUSTED")]
            s = PodSession(); s.try_begin_start(None)
            inst = rd._create_with_retries(s)
            check("second attempt succeeded", inst is not None and inst["status"] == "RUNNING")
            check("attempts went b then c", [z for z, _ in api.inserted] == ["europe-west2-b", "europe-west2-c"])
            check("event feed names the next zone",
                  any("europe-west2-c next" in m for _, _c, m in s.events))
            check("secrets synced exactly once across attempts",
                  sum(1 for c in api.calls if c[0] == "secret_put") == 2)
    finally:
        rd._sleep_or_cancel = saved


def test_driver_surfaces_quota_error_without_retry():
    saved = rd._sleep_or_cancel
    rd._sleep_or_cancel = lambda session, secs: False
    try:
        with gcp() as (p, api):
            api.insert_errors = [GcpApiError("quota", code="QUOTA_EXCEEDED")]
            s = PodSession(); s.try_begin_start(None)
            raised = False
            try:
                rd._create_with_retries(s)
            except GcpApiError as e:
                raised = e.code == "QUOTA_EXCEEDED"
            check("quota error surfaces after ONE attempt", raised and len(api.inserted) == 1)
    finally:
        rd._sleep_or_cancel = saved


def test_create_resumes_a_stopped_instance_and_refuses_a_live_duplicate():
    with gcp() as (p, api):
        api.instances[("europe-west2-b", "podlink")] = {"name": "podlink", "status": "TERMINATED",
                                                       "zone": "projects/proj-1/zones/europe-west2-b"}
        inst = p.create_once(p.prepare_create(PodSession()), SECRETS, 1)
        check("TERMINATED instance is STARTED, not re-created",
              ("start", "europe-west2-b", "podlink") in api.calls and not api.inserted)
        check("returned instance is RUNNING", inst["status"] == "RUNNING")
        raised = None
        try:
            p.create_once(p.prepare_create(PodSession()), SECRETS, 1)
        except GcpApiError as e:
            raised = e
        check("a RUNNING duplicate is refused, non-retryable",
              raised is not None and raised.code == "ALREADY_EXISTS" and not raised.retryable)


# --- terminate ------------------------------------------------------------------

def test_terminate_delete_then_verify():
    with gcp() as (p, api):
        api.instances[("europe-west2-b", "podlink")] = {"name": "podlink", "status": "RUNNING",
                                                       "zone": "projects/proj-1/zones/europe-west2-b"}
        p.terminate("europe-west2-b/podlink")
        check("delete issued", ("delete", "europe-west2-b", "podlink") in api.calls)
        check("gone => get_instance None", p.get_instance("europe-west2-b/podlink") is None)
        check("driver verifies terminate", rd._verify_terminated(PodSession(), "europe-west2-b/podlink") is True)


def test_terminate_stop_action():
    with gcp(PODLINK_GCP_DOWN_ACTION="stop") as (p, api):
        api.instances[("europe-west2-b", "podlink")] = {"name": "podlink", "status": "RUNNING",
                                                       "zone": "projects/proj-1/zones/europe-west2-b"}
        p.terminate("europe-west2-b/podlink")
        check("stop issued instead of delete", ("stop", "europe-west2-b", "podlink") in api.calls
              and not any(c[0] == "delete" for c in api.calls))
        check("stopped instance is not running", not p.is_running(p.get_instance("europe-west2-b/podlink")))
        check("driver verifies the stop", rd._verify_terminated(PodSession(), "europe-west2-b/podlink") is True)


# --- access + endpoints ------------------------------------------------------------

def test_access_modes():
    with gcp(PODLINK_GCP_ACCESS="internal") as (p, api):
        api.instances[("europe-west2-b", "podlink")] = {"name": "podlink", "status": "RUNNING",
                                                       "zone": "projects/proj-1/zones/europe-west2-b",
                                                       "networkInterfaces": [{"networkIP": "10.1.2.3"}]}
        check("internal: placeholder host before ensure_access",
              "pending.internal" in p.service_urls("europe-west2-b/podlink")["llm"])
        p.ensure_access("europe-west2-b/podlink")
        urls = p.service_urls("europe-west2-b/podlink")
        check("internal: VPC address after ensure_access",
              urls == {"llm": "http://10.1.2.3:8000", "embedder": "http://10.1.2.3:8080",
                       "reranker": "http://10.1.2.3:8081"})
        p.release_access()
        check("release forgets the address", "pending.internal" in p.service_urls("europe-west2-b/podlink")["llm"])
        env = p.client_env("europe-west2-b/podlink", 4096)
        check("client env bearer is a placeholder", "<your pod_bearer_token>" in env and "bearer_dummy" not in env)
    with gcp() as (p, api):                                  # iap (default)
        check("iap: loopback tunnel ports",
              p.service_urls("europe-west2-b/podlink") == {"llm": "http://127.0.0.1:18000",
                                                            "embedder": "http://127.0.0.1:18080",
                                                            "reranker": "http://127.0.0.1:18081"})


class FakeTunnels:
    """Stands in for TunnelManager: records construction, ensure/release, and
    hands back scripted events."""
    made: list = []

    def __init__(self, project, zone, instance, ports):
        self.project, self.zone, self.instance, self.ports = project, zone, instance, ports
        self.instance_id = f"{zone}/{instance}"
        self.ensured = 0
        self.released = 0
        self.events = [f"tunnel: llm -> 127.0.0.1:{ports['llm'][1]} up"]
        FakeTunnels.made.append(self)

    def ensure(self):
        self.ensured += 1

    def release(self):
        self.released += 1

    def drain_events(self):
        out, self.events = self.events, []
        return out


def test_iap_access_supervises_tunnels_through_the_factory():
    FakeTunnels.made.clear()
    with gcp() as (p, api):
        p._tunnel_factory = FakeTunnels
        p.ensure_access("europe-west2-b/podlink")
        t = FakeTunnels.made[0]
        check("tunnels built for the instance in its zone/project",
              (t.project, t.zone, t.instance) == ("proj-1", "europe-west2-b", "podlink"))
        check("service ports mapped remote -> local",
              t.ports == {"llm": (8000, 18000), "embedder": (8080, 18080), "reranker": (8081, 18081)})
        check("ensure() called", t.ensured == 1)
        check("access events surface to the driver", p.access_events() == ["tunnel: llm -> 127.0.0.1:18000 up"])
        check("drained once", p.access_events() == [])
        p.ensure_access("europe-west2-b/podlink")
        check("same instance: re-ensured, not rebuilt", len(FakeTunnels.made) == 1 and t.ensured == 2)
        p.ensure_access("europe-west2-c/podlink")
        check("different instance: old released, new built",
              t.released == 1 and len(FakeTunnels.made) == 2 and FakeTunnels.made[1].zone == "europe-west2-c")
        p.release_access()
        check("release_access releases and forgets", FakeTunnels.made[1].released == 1 and p._tunnels is None)
        check("release is safe with nothing open", p.release_access() is None and p.access_events() == [])


def test_driver_drains_tunnel_events_on_both_start_paths():
    # The adopt path never creates, so the tunnel events must be drained there too.
    FakeTunnels.made.clear()
    saved = rd.egress_logger.client
    import contextlib as _cl

    @_cl.contextmanager
    def ok_client(timeout=15.0):
        class _C:
            def get(self, url, headers=None):
                class R: status_code = 200
                return R()
        yield _C()
    rd.egress_logger.client = ok_client
    try:
        with gcp() as (p, api):
            p._tunnel_factory = FakeTunnels
            api.instances[("europe-west2-b", "podlink")] = {"name": "podlink", "status": "RUNNING",
                                                           "zone": "projects/proj-1/zones/europe-west2-b"}
            s = PodSession(); s.try_begin_start(None)      # Auto -> adopts the running instance
            rd.start(s)
            check("adopt path reached RUNNING through the tunnels", s.snapshot()["state"] == "RUNNING")
            check("tunnel 'up' event landed in the health feed",
                  any(c == "health" and m.startswith("tunnel:") for _, c, m in s.events))
            s.try_begin_stop(); rd.stop(s)
            check("stop released the tunnels", FakeTunnels.made[0].released == 1)
    finally:
        rd.egress_logger.client = saved
        prov_mod.STATE_PATH.unlink(missing_ok=True)


def test_instance_id_shape_and_status_predicates():
    with gcp() as (p, api):
        check("valid zone/name id", p.valid_instance_id("europe-west2-b/podlink"))
        check("rejects a bare name", not p.valid_instance_id("podlink"))
        check("rejects injection-y ids", not p.valid_instance_id("europe-west2-b/pod link;rm"))
        running = {"name": "podlink", "status": "RUNNING", "zone": ".../zones/europe-west2-c"}
        check("instance_id is zone/name", p.instance_id(running) == "europe-west2-c/podlink")
        check("is_running/is_up RUNNING", p.is_running(running) and p.is_up(running))
        check("STAGING is not running", not p.is_running({"status": "STAGING"}))
        check("cost from the profile", p.cost_per_hr(running) == 5.40)


# --- bootstrap + preflight + registry ------------------------------------------------

def test_bootstrap_script_has_no_secrets():
    with gcp() as (p, api):
        doc = json.loads(prov_mod.bootstrap.render(p.cfg, "podlink-data").split("\n", 1)[1])
        script = doc["write_files"][0]["content"]
        check("mounts the named device", "google-podlink-data" in script)
        check("docker data-root on the disk", '"data-root": "/workspace/docker"' in script)
        check("fetches both secrets by name", "podlink-bearer" in script and "podlink-hf-token" in script)
        check("env file on tmpfs", "/run/podlink/env" in script)
        check("runs the configured image with GPUs", "--gpus all" in script and p.cfg.image in script)
        check("logs into Artifact Registry with the VM token", "docker login -u oauth2accesstoken" in script)
        check("model config passed through", "LLM_MODEL_ID=" in script and "MAX_MODEL_LEN=" in script)
        check("no secret values in the script", "bearer_dummy" not in script and "hf_dummy" not in script)


def test_preflight_reports_missing_credentials_and_config():
    class NoAdc:
        def token(self):
            raise RuntimeError("no Application Default Credentials — run gcloud ...")
    saved = prov_mod.AdcTokenSource
    prov_mod.AdcTokenSource = NoAdc
    try:
        with gcp(PODLINK_GCP_KMS_KEY=None, PODLINK_GCP_SERVICE_ACCOUNT=None) as (p, api):
            rows = dict((m.split(":")[0], lvl) for lvl, m in p.preflight())
            check("missing ADC is a hard failure", rows.get("credentials") == "fail")
            check("strict: Google-managed keys FAIL", any(k.startswith("disks") and v == "fail" for k, v in rows.items()))
            check("strict: default compute SA FAILS", any(k.startswith("VM identity") and v == "fail" for k, v in rows.items()))
        with gcp(PODLINK_GCP_KMS_KEY=None, PODLINK_GCP_SERVICE_ACCOUNT=None,
                 PODLINK_GCP_HARDENING="relaxed") as (p, api):
            rows = dict((m.split(":")[0], lvl) for lvl, m in p.preflight())
            check("relaxed: the same gaps are warnings",
                  any(k.startswith("disks") and v == "warn" for k, v in rows.items())
                  and any(k.startswith("VM identity") and v == "warn" for k, v in rows.items()))
        with gcp(PODLINK_GCP_PROJECT=None) as (p, api):
            check("missing project is a hard failure",
                  any(lvl == "fail" and "PODLINK_GCP_PROJECT" in m for lvl, m in p.preflight()))
    finally:
        prov_mod.AdcTokenSource = saved


def test_residency_is_enforced_in_config():
    # No org policy exists on a standalone project, so the config refuses zones
    # outside the allowed regions instead of quietly creating a VM in Iowa.
    raised = ""
    try:
        with gcp(PODLINK_GCP_ZONES="us-central1-a"):
            pass
    except ValueError as e:
        raised = str(e)
    check("zones outside PODLINK_GCP_ALLOWED_REGIONS are refused", "outside" in raised and "us-central1" in raised)
    with gcp(PODLINK_GCP_ZONES="europe-west4-a", PODLINK_GCP_ALLOWED_REGIONS="europe-west2,europe-west4") as (p, api):
        check("an explicitly allowed second region is accepted", p.cfg.region == "europe-west4")
    with gcp() as (p, api):
        check("in-region Artifact Registry image recognised", p.cfg.image_in_region)
    with gcp(PODLINK_GCP_IMAGE="ghcr.io/x/y:1") as (p, api):
        check("ghcr image is not in-region", not p.cfg.image_in_region)
    with gcp(PODLINK_GCP_IMAGE="us-central1-docker.pkg.dev/proj-1/r/y:1") as (p, api):
        check("out-of-region Artifact Registry is not in-region either", not p.cfg.image_in_region)


def test_strict_hardening_refuses_an_unhardened_create():
    # instance_body() is the enforcement point: preflight can be skipped, this cannot.
    for missing, override, expect in (
        ("CMEK", {"PODLINK_GCP_KMS_KEY": None}, "KMS_KEY"),
        ("dedicated SA", {"PODLINK_GCP_SERVICE_ACCOUNT": None}, "SERVICE_ACCOUNT"),
        ("in-region image", {"PODLINK_GCP_IMAGE": "ghcr.io/x/y:1"}, "in-region"),
    ):
        with gcp(**override) as (p, api):
            raised = ""
            try:
                p.instance_body("europe-west2-b")
            except ValueError as e:
                raised = str(e)
            check(f"strict refuses to build without {missing}", expect in raised)
    with gcp(PODLINK_GCP_KMS_KEY=None, PODLINK_GCP_SERVICE_ACCOUNT=None, PODLINK_GCP_IMAGE="ghcr.io/x/y:1",
             PODLINK_GCP_HARDENING="relaxed") as (p, api):
        body = p.instance_body("europe-west2-b")
        check("relaxed builds the body (Google-managed keys, default SA)",
              "diskEncryptionKey" not in json.dumps(body) and "serviceAccounts" not in body)
        check("relaxed still has no external IP and is still confidential",
              "accessConfigs" not in json.dumps(body) and "confidentialInstanceConfig" in body)


def test_preflight_checks_the_log_sink_is_regional():
    class Tok:
        def token(self): return "tok"
    saved = prov_mod.AdcTokenSource
    prov_mod.AdcTokenSource = Tok
    try:
        with gcp() as (p, api):
            rows = p.preflight()
            logs = [(lvl, m) for lvl, m in rows if m.startswith("logs:")]
            check("regional _Default sink is ok", logs and logs[0][0] == "ok")
            api.log_sink_destination = "logging.googleapis.com/projects/proj-1/locations/global/buckets/_Default"
            logs = [(lvl, m) for lvl, m in p.preflight() if m.startswith("logs:")]
            check("global _Default sink FAILS under strict", logs and logs[0][0] == "fail")
            check("hardening row names the posture", any("hardening: strict" in m for _, m in rows))
        with gcp(PODLINK_GCP_HARDENING="relaxed") as (p, api):
            api.log_sink_destination = "logging.googleapis.com/projects/proj-1/locations/global/buckets/_Default"
            logs = [(lvl, m) for lvl, m in p.preflight() if m.startswith("logs:")]
            check("global _Default sink is a warning under relaxed", logs and logs[0][0] == "warn")
    finally:
        prov_mod.AdcTokenSource = saved


def test_registry_builds_gcp_provider():
    saved = (providers._ACTIVE, providers._ACTIVE_NAME, os.environ.get("PODLINK_PROVIDER"))
    try:
        os.environ["PODLINK_PROVIDER"] = "gcp"
        providers._ACTIVE = None
        check("registry builds GcpProvider for PODLINK_PROVIDER=gcp", providers.active().name == "gcp")
        fields = providers.active().snapshot_fields()
        required = {"provider", "persistence_configured", "persistence_id", "persistence_label", "llm_model_id"}
        check("snapshot_fields complete", required <= set(fields))
    finally:
        providers._ACTIVE, providers._ACTIVE_NAME = saved[0], saved[1]
        if saved[2] is None:
            os.environ.pop("PODLINK_PROVIDER", None)
        else:
            os.environ["PODLINK_PROVIDER"] = saved[2]


# --- the REST client's error handling (fake httpx) ------------------------------------

class _Resp:
    def __init__(self, status, body=None):
        self.status_code, self._b = status, body
        self.content = b"x" if body is not None else b""
    def json(self):
        return self._b


def test_api_error_parsing_and_operation_wait():
    script = []                                          # responses in order

    @contextlib.contextmanager
    def fake_client(timeout=30.0):
        class _C:
            def request(self, method, url, headers=None, json=None, params=None):
                script.append((method, url))
                return responses.pop(0)
        yield _C()

    saved = api_mod.egress_logger.client
    api_mod.egress_logger.client = fake_client
    try:
        class Tok:
            def token(self): return "tok"
        api = api_mod.GcpApi("proj-1", Tok())
        responses = [_Resp(403, {"error": {"message": "denied", "errors": [{"reason": "forbidden"}]}})]
        raised = None
        try:
            api.get_instance("europe-west2-b", "podlink")
        except GcpApiError as e:
            raised = e
        check("403 -> GcpApiError with status + reason",
              raised.http_status == 403 and raised.code == "forbidden" and not raised.retryable)
        responses = [_Resp(404, {"error": {"message": "nope"}})]
        check("404 -> None for get_instance", api.get_instance("europe-west2-b", "podlink") is None)
        # insert: operation RUNNING, then wait returns DONE with a stockout error.
        responses = [_Resp(200, {"name": "op1", "status": "RUNNING"}),
                     _Resp(200, {"name": "op1", "status": "DONE",
                                 "error": {"errors": [{"code": "ZONE_RESOURCE_POOL_EXHAUSTED", "message": "no cap"}]}})]
        raised = None
        try:
            api.insert_instance("europe-west2-b", {"name": "podlink"})
        except GcpApiError as e:
            raised = e
        check("operation error surfaces with its code, retryable",
              raised.code == "ZONE_RESOURCE_POOL_EXHAUSTED" and raised.retryable)
        check("waited via the server-side wait endpoint", script[-1][1].endswith("/operations/op1/wait"))
        # secret_put creates the secret on 404 then adds the version.
        responses = [_Resp(404, {"error": {"message": "no secret"}}), _Resp(200, {}), _Resp(200, {})]
        api.secret_put("podlink-bearer", "v", "europe-west2")
        check("secret created in-region then versioned",
              script[-2][1].endswith("/secrets") and script[-1][1].endswith(":addVersion"))
    finally:
        api_mod.egress_logger.client = saved


if __name__ == "__main__":
    print("gcp provider tests:")
    test_create_payload_security_properties()
    test_create_never_puts_secrets_in_the_payload()
    test_spot_and_volume_less_variants()
    test_no_kill_switch_when_disabled()
    test_zone_placement_rules()
    test_retry_classification()
    test_driver_retries_rotating_zones_on_stockout()
    test_driver_surfaces_quota_error_without_retry()
    test_create_resumes_a_stopped_instance_and_refuses_a_live_duplicate()
    test_terminate_delete_then_verify()
    test_terminate_stop_action()
    test_access_modes()
    test_iap_access_supervises_tunnels_through_the_factory()
    test_driver_drains_tunnel_events_on_both_start_paths()
    test_instance_id_shape_and_status_predicates()
    test_bootstrap_script_has_no_secrets()
    test_preflight_reports_missing_credentials_and_config()
    test_residency_is_enforced_in_config()
    test_strict_hardening_refuses_an_unhardened_create()
    test_preflight_checks_the_log_sink_is_regional()
    test_registry_builds_gcp_provider()
    test_api_error_parsing_and_operation_wait()
    print("all gcp provider tests passed.")
