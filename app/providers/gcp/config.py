"""GCP provider settings, read from PODLINK_GCP_* (plus the shared PODLINK_* stack).

Read live by GcpProvider.reload_config() so a profile switch re-bakes them.
Every value has a documented default; the only required ones are the project
and the boot image. See the README's configuration table.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from ... import stack as stack_contract

DEFAULT_ZONES = ("europe-west2-b", "europe-west2-c")   # London's two G4 zones
DEFAULT_MACHINE_TYPE = "g4-standard-48"                 # 1x RTX PRO 6000 96 GB


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError as e:
        raise ValueError(f"{name} must be an integer") from e


def _float_or_none(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be a number") from e


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = os.environ.get(name, default).strip().lower() or default
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)} (got {value!r})")
    return value


@dataclass(frozen=True)
class GcpConfig:
    project: str
    zones: tuple[str, ...]
    region: str
    machine_type: str
    instance_name: str
    boot_image: str            # e.g. projects/<p>/global/images/family/podlink-g4 (the golden image)
    boot_disk_gb: int
    data_disk: str             # '' => volume-less: a pod-scoped scratch disk destroyed on POD DOWN
    data_disk_scope: str       # zonal | regional  (regional = Hyperdisk Balanced HA, enables zone rotation)
    volume_gb: int             # scratch size in volume-less mode
    confidential: bool         # AMD SEV + NVIDIA GPU TEE
    provisioning_model: str    # STANDARD | SPOT
    max_run_hours: float       # platform-side kill switch; 0 = none
    down_action: str           # delete | stop
    access: str                # iap | internal
    local_ports: dict          # service -> loopback port the IAP tunnels bind
    service_account: str       # VM identity (email); '' => the project's default compute SA
    subnet: str
    network_tag: str           # firewall target for the IAP range
    kms_key: str               # CMEK key resource name; '' => Google-managed keys
    secret_prefix: str         # Secret Manager names: <prefix>-bearer, <prefix>-hf-token
    cost_per_hr: float | None  # for the meter — GCP does not report a rate on the instance
    image: str                 # container image (the Artifact Registry mirror)
    stack: dict                # the image's env contract (app/stack.py)
    # Residency + hardening posture. These exist because the project starts
    # STANDALONE (no organization, so no org policies) and is migrated into an
    # Assured Workloads folder later: until then, everything the platform would
    # enforce has to be enforced HERE, or it is not enforced at all.
    allowed_regions: tuple[str, ...]   # every resource must be in one of these
    hardening: str             # strict (default): CMEK + dedicated SA + in-region image are REQUIRED
                               # relaxed: the same checks downgrade to warnings (experiments only)

    @property
    def strict(self) -> bool:
        return self.hardening == "strict"

    @property
    def image_in_region(self) -> bool:
        """True when the container image is served from Artifact Registry in an
        allowed region (host looks like <region>-docker.pkg.dev)."""
        host = self.image.split("/", 1)[0] if "/" in self.image else ""
        return host.endswith("-docker.pkg.dev") and host[: -len("-docker.pkg.dev")] in self.allowed_regions

    @property
    def secret_names(self) -> dict[str, str]:
        return {"bearer": f"{self.secret_prefix}-bearer", "hf": f"{self.secret_prefix}-hf-token"}

    def zone_for(self, attempt: int) -> str:
        """The zone for create attempt N (1-based).

        A ZONAL data disk lives in exactly one zone, so the VM must too: the
        first configured zone, every attempt. A regional disk or a volume-less
        launch may rotate through all configured zones on stockout.
        """
        if self.data_disk and self.data_disk_scope == "zonal":
            return self.zones[0]
        return self.zones[(attempt - 1) % len(self.zones)]

    @property
    def rotates_zones(self) -> bool:
        return len(self.zones) > 1 and not (self.data_disk and self.data_disk_scope == "zonal")


def from_env() -> GcpConfig:
    """Build the config from the environment, validating the parts that would
    otherwise fail deep inside a create call."""
    zones = tuple(z.strip() for z in os.environ.get("PODLINK_GCP_ZONES", ",".join(DEFAULT_ZONES)).split(",")
                  if z.strip())
    if not zones:
        raise ValueError("PODLINK_GCP_ZONES must name at least one zone")
    region = zones[0].rsplit("-", 1)[0]                      # europe-west2-b -> europe-west2
    if any(z.rsplit("-", 1)[0] != region for z in zones):
        raise ValueError("PODLINK_GCP_ZONES must all be in one region (a regional disk spans one region)")
    allowed = tuple(r.strip() for r in os.environ.get("PODLINK_GCP_ALLOWED_REGIONS", "europe-west2").split(",")
                    if r.strip())
    if region not in allowed:                                 # residency is enforced in code, not by policy
        raise ValueError(f"PODLINK_GCP_ZONES are in {region}, outside PODLINK_GCP_ALLOWED_REGIONS "
                         f"({', '.join(allowed)}) — UK residency is enforced here until the org migration")
    ports_raw = os.environ.get("PODLINK_GCP_LOCAL_PORTS", "18000,18080,18081").split(",")
    if len(ports_raw) != 3:
        raise ValueError("PODLINK_GCP_LOCAL_PORTS needs three ports: llm,embedder,reranker")
    try:
        local_ports = dict(zip(("llm", "embedder", "reranker"), (int(p) for p in ports_raw)))
    except ValueError as e:
        raise ValueError("PODLINK_GCP_LOCAL_PORTS must be integers") from e
    stack = stack_contract.from_env()
    return GcpConfig(
        project=os.environ.get("PODLINK_GCP_PROJECT", "").strip(),
        zones=zones,
        region=region,
        machine_type=os.environ.get("PODLINK_GCP_MACHINE_TYPE", DEFAULT_MACHINE_TYPE).strip(),
        instance_name=os.environ.get("PODLINK_POD_NAME", "podlink").strip(),
        boot_image=os.environ.get("PODLINK_GCP_BOOT_IMAGE", "").strip(),
        boot_disk_gb=_int("PODLINK_GCP_BOOT_DISK_GB", 100),
        data_disk=os.environ.get("PODLINK_GCP_DATA_DISK", "").strip(),
        data_disk_scope=_choice("PODLINK_GCP_DATA_DISK_SCOPE", "zonal", ("zonal", "regional")),
        volume_gb=_int("PODLINK_VOLUME_GB", 50),
        confidential=_flag("PODLINK_GCP_CONFIDENTIAL", "1"),
        provisioning_model=_choice("PODLINK_GCP_PROVISIONING", "standard", ("standard", "spot")).upper(),
        max_run_hours=float(os.environ.get("PODLINK_GCP_MAX_RUN_HOURS", "8") or 0),
        down_action=_choice("PODLINK_GCP_DOWN_ACTION", "delete", ("delete", "stop")),
        access=_choice("PODLINK_GCP_ACCESS", "iap", ("iap", "internal")),
        local_ports=local_ports,
        service_account=os.environ.get("PODLINK_GCP_SERVICE_ACCOUNT", "").strip(),
        subnet=os.environ.get("PODLINK_GCP_SUBNET", "default").strip(),
        network_tag=os.environ.get("PODLINK_GCP_NETWORK_TAG", "podlink").strip(),
        kms_key=os.environ.get("PODLINK_GCP_KMS_KEY", "").strip(),
        secret_prefix=os.environ.get("PODLINK_GCP_SECRET_PREFIX", "podlink").strip(),
        cost_per_hr=_float_or_none("PODLINK_GCP_COST_PER_HR"),
        image=(os.environ.get("PODLINK_GCP_IMAGE", "").strip() or stack["image"] or ""),
        stack=stack,
        allowed_regions=allowed,
        hardening=_choice("PODLINK_GCP_HARDENING", "strict", ("strict", "relaxed")),
    )
