"""Provision (or resume) the Phase 0 RunPod pod. Idempotent.

If a pod named POD_NAME already exists, prints its state and exits 0.
Otherwise creates a new Secure Cloud pod serving Qwen 2.5 32B AWQ via vLLM
on RTX 5090 (with A6000 fallback when 5090 stock is exhausted).

Writes phase0/pod_state.json on success. The bearer token is NEVER written
here — it stays in ~/.config/podlink/pod_bearer_token.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import runpod
from rich import print as rprint
from rich.prompt import Confirm

import _secrets

POD_NAME = "podlink"
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct-AWQ"
# Pinned tag, not :latest. TODO before first prospect demo: replace with a
# digest pin (`vllm/vllm-openai@sha256:<digest>`) using `docker inspect` on a
# locally pulled image. Pin tag set 2026-05-12.
# Updated 2026-05-17: v0.20.2 requires CUDA 13.0 which RunPod's current driver
# fleet doesn't support (host error: "unsatisfied condition: cuda>=13.0").
# Dropped to v0.9.2 (CUDA 12.4) which RunPod hosts accept. Amend Phase0_Plan.md
# accordingly before next prospect demo.
IMAGE = "vllm/vllm-openai:v0.9.2"
EXPOSED_PORT = "8000/http"  # HTTP port mode -> RunPod proxy URL with auto TLS
CONTAINER_DISK_GB = 30
VOLUME_GB = 30
VOLUME_MOUNT = "/workspace"
MAX_MODEL_LEN = 8192

GPU_PREFERENCES = [
    "NVIDIA GeForce RTX 5090",  # primary
    "NVIDIA RTX A6000",         # architect-approved fallback (48 GB Ampere)
]

STATE_PATH = Path(__file__).resolve().parents[1] / "pod_state.json"


def find_existing() -> dict | None:
    for p in runpod.get_pods():
        if p.get("name") == POD_NAME:
            return p
    return None


def build_docker_args() -> str:
    # IMPORTANT: do NOT pass --api-key on the command line.
    # vLLM reads VLLM_API_KEY from the environment natively; passing it as an
    # argv would make the token visible in `ps aux` inside the pod (the
    # RunPod web terminal cannot be disabled, so any shell session there
    # would see the token).
    # vllm/vllm-openai images (>= v0.9) set ENTRYPOINT to
    # `python -m vllm.entrypoints.openai.api_server`, so we pass only the
    # flags here. Older images had a shell entrypoint and required the full
    # python invocation; check the image's Dockerfile before changing.
    return (
        f"--model {MODEL_ID} "
        "--quantization awq_marlin "
        "--host 0.0.0.0 --port 8000 "
        f"--max-model-len {MAX_MODEL_LEN} "
        "--gpu-memory-utilization 0.92"
    )


def create_pod(gpu_type_id: str, bearer: str, hf: str) -> dict:
    rprint(f"[bold cyan]Creating pod[/] on [bold]{gpu_type_id}[/] …")
    pod = runpod.create_pod(
        name=POD_NAME,
        image_name=IMAGE,
        gpu_type_id=gpu_type_id,
        cloud_type="SECURE",
        gpu_count=1,
        volume_in_gb=VOLUME_GB,
        container_disk_in_gb=CONTAINER_DISK_GB,
        volume_mount_path=VOLUME_MOUNT,
        ports=EXPOSED_PORT,
        env={
            "HF_TOKEN": hf,
            "VLLM_API_KEY": bearer,
            "HUGGING_FACE_HUB_TOKEN": hf,  # some images read this name instead
        },
        docker_args=build_docker_args(),
        support_public_ip=False,  # all traffic goes via RunPod's HTTPS proxy
        start_ssh=False,           # SSH disabled; reduces attack surface
    )
    return pod


def try_create() -> dict:
    bearer = _secrets.bearer_token()
    hf = _secrets.hf_token()
    last_err = None
    for gpu_id in GPU_PREFERENCES:
        try:
            return create_pod(gpu_id, bearer, hf)
        except Exception as e:
            # Avoid printing exception payload — SDK exceptions can include
            # request context (potentially the API key in some versions).
            rprint(f"[yellow]GPU {gpu_id} unavailable[/] "
                   f"({type(e).__name__}). Check the RunPod dashboard for details.")
            last_err = e
            if gpu_id != GPU_PREFERENCES[-1]:
                if not Confirm.ask(f"Fall back to next GPU option?", default=True):
                    break
    raise RuntimeError(f"All GPU options exhausted. Last error: {last_err}")


def derive_proxy_url(pod_id: str) -> str:
    # Format documented at docs.runpod.io/pods/configuration/expose-ports
    return f"https://{pod_id}-8000.proxy.runpod.net"


def wait_for_running(pod_id: str, timeout_s: int = 900) -> dict:
    rprint("[bold]Waiting for pod to reach RUNNING…[/] (model weight pull can take 5–15 min)")
    deadline = time.time() + timeout_s
    last_status = None
    while time.time() < deadline:
        pod = runpod.get_pod(pod_id)
        status = (pod.get("desiredStatus") or pod.get("lastStatusChange") or
                  pod.get("runtime", {}).get("uptimeInSeconds"))
        if status != last_status:
            rprint(f"  status: {pod.get('desiredStatus')} runtime={pod.get('runtime')}")
            last_status = status
        if pod.get("desiredStatus") == "RUNNING" and pod.get("runtime"):
            return pod
        time.sleep(15)
    sys.exit(f"Pod {pod_id} did not reach RUNNING within {timeout_s}s")


def write_state(pod: dict, gpu_type_id: str) -> None:
    pod_id = pod["id"]
    state = {
        "pod_id": pod_id,
        "name": pod.get("name"),
        "proxy_url": derive_proxy_url(pod_id),
        "model": MODEL_ID,
        "gpu_type": gpu_type_id,
        "image": IMAGE,
        "created_at": pod.get("lastStatusChange") or pod.get("createdAt"),
    }
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")
    rprint(f"[green]Wrote[/] {STATE_PATH}")
    rprint(f"[bold]Proxy URL:[/] {state['proxy_url']}")


def main() -> None:
    runpod.api_key = _secrets.runpod_api_key()

    existing = find_existing()
    if existing:
        pod_id = existing["id"]
        status = existing.get("desiredStatus")
        rprint(f"[yellow]Pod {POD_NAME!r} already exists[/]: id={pod_id} status={status}")
        if status != "RUNNING":
            if Confirm.ask("Resume it?", default=True):
                rprint("[cyan]Resuming…[/]")
                # resume_pod signature varies; SDK requires gpu_count
                runpod.resume_pod(pod_id, gpu_count=1)
                wait_for_running(pod_id)
        # rewrite state file so we always have fresh URL on disk
        pod = runpod.get_pod(pod_id)
        # gpu_type comes back nested in 'machine'/'gpuTypeId' depending on SDK version
        gpu_type = (pod.get("machine", {}).get("gpuTypeId")
                    or pod.get("gpuTypeId") or "unknown")
        write_state({**pod, "id": pod_id}, gpu_type)
        return

    pod = try_create()
    pod_id = pod["id"]
    rprint(f"[green]Pod created:[/] id={pod_id}")
    pod = wait_for_running(pod_id)
    gpu_type = (pod.get("machine", {}).get("gpuTypeId")
                or pod.get("gpuTypeId") or GPU_PREFERENCES[0])
    write_state(pod, gpu_type)
    rprint("[bold green]Pod is up.[/] vLLM may still be loading model weights — "
           "check logs in the RunPod dashboard, or run auth_checks.py and retry "
           "until /v1/models returns 200.")


if __name__ == "__main__":
    main()
