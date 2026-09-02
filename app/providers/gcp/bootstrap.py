"""The cloud-init the VM boots with: mount the data disk, run the container.

Rendered into instance metadata as `user-data`. Instance metadata is readable
by anyone with compute.instances.get on the project, so NOTHING secret goes in
here: the script fetches the bearer and HF token from Secret Manager at boot,
using the VM's own service account, into a tmpfs env file the container reads.
The golden image (pod_image/gcp) supplies the NVIDIA driver, the container
toolkit and Docker; this script supplies only what changes per launch.

Two persistence modes, mirroring the RunPod provider:
  * a named Hyperdisk attached as device `podlink-data` (survives POD DOWN) —
    Docker's data-root AND the HF cache live on it, so neither the 38 GB image
    nor the weights are pulled again on a warm boot;
  * a pod-scoped scratch disk `podlink-scratch` (destroyed with the VM).
Either way the disk is formatted on first use and mounted at /workspace.
"""
from __future__ import annotations

import json
import shlex

from ... import stack as stack_contract

MOUNT = "/workspace"
ENV_FILE = "/run/podlink/env"           # tmpfs: secrets never touch the persistent disk


def render(cfg, device_name: str) -> str:
    """The #cloud-config document for this launch. Contains no secret values."""
    ports = " ".join(f"-p {p}:{p}" for p in stack_contract.SERVICE_PORTS.values())
    # Everything the image reads except the two secrets, which the script appends.
    static_env = stack_contract.container_env(cfg.stack, bearer="", hf="")
    static_lines = "\n".join(f"{k}={v}" for k, v in static_env.items()
                             if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "VLLM_API_KEY", "TEI_API_KEY"))
    names = cfg.secret_names
    registry_host = cfg.image.split("/", 1)[0] if "/" in cfg.image else ""
    login = (f"docker login -u oauth2accesstoken --password-stdin {shlex.quote(registry_host)} <<< \"$TOKEN\""
             if registry_host.endswith("pkg.dev") else "true  # public image: no registry login")

    boot_sh = f"""#!/bin/bash
# podlink boot — rendered by app/providers/gcp/bootstrap.py; no secrets here.
set -euo pipefail
DEV=/dev/disk/by-id/google-{device_name}
MNT={MOUNT}
# 1. Persistent (or scratch) disk -> /workspace. Format only when blank.
if ! blkid "$DEV" >/dev/null 2>&1; then mkfs.ext4 -F -L podlink "$DEV"; fi
mkdir -p "$MNT" && mountpoint -q "$MNT" || mount -o discard,defaults "$DEV" "$MNT"
mkdir -p "$MNT/docker" "$MNT/hf"
# 2. Docker's data-root on the disk: the container image survives POD DOWN too.
mkdir -p /etc/docker
printf '%s' '{{"data-root": "{MOUNT}/docker"}}' > /etc/docker/daemon.json
systemctl restart docker
# 3. Secrets from Secret Manager, via the VM's own identity, into tmpfs.
MD=http://metadata.google.internal/computeMetadata/v1
TOKEN=$(curl -sf -H 'Metadata-Flavor: Google' "$MD/instance/service-accounts/default/token" \\
        | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
fetch() {{ curl -sf -H "Authorization: Bearer $TOKEN" \\
  "https://secretmanager.googleapis.com/v1/projects/{cfg.project}/secrets/$1/versions/latest:access" \\
  | python3 -c 'import sys,json,base64;print(base64.b64decode(json.load(sys.stdin)["payload"]["data"]).decode())'; }}
mkdir -p /run/podlink && chmod 700 /run/podlink
umask 077
BEARER=$(fetch {names['bearer']})
HF=$(fetch {names['hf']})
cat > {ENV_FILE} <<EOF
{static_lines}
HF_TOKEN=$HF
HUGGING_FACE_HUB_TOKEN=$HF
VLLM_API_KEY=$BEARER
TEI_API_KEY=$BEARER
HF_HOME={MOUNT}/hf
EOF
unset BEARER HF
# 4. Pull (from the in-region registry) and run the SAME image every cloud uses.
{login}
docker rm -f podlink >/dev/null 2>&1 || true
docker run -d --name podlink --restart unless-stopped --gpus all \\
  --env-file {ENV_FILE} -v "$MNT:{MOUNT}" {ports} {shlex.quote(cfg.image)}
"""
    doc = {
        "write_files": [{"path": "/opt/podlink/boot.sh", "permissions": "0755", "content": boot_sh}],
        "runcmd": [["bash", "/opt/podlink/boot.sh"]],
    }
    # cloud-init accepts JSON as #cloud-config; JSON avoids YAML quoting traps
    # around the embedded shell.
    return "#cloud-config\n" + json.dumps(doc, indent=1)
