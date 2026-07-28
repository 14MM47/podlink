"""Provision (or resume) the podlink RunPod pod. Idempotent.

If a pod named POD_NAME already exists, prints its state and exits 0.
Otherwise creates a new Secure Cloud pod on a single RTX Pro 6000 (96 GB,
Blackwell) running a bundled RAG inference stack from ONE image:

    :8000  vLLM        LLM (OpenAI-compatible /v1)
    :8080  TEI         embedder  (/v1/embeddings, /health)
    :8081  TEI         reranker  (/rerank, /health)

The image itself (vLLM + both TEI + supervisor) lives in ../pod_image and must be
built and pushed first — see pod_image/README.md. Model weights are pulled on the
pod's first boot into the /workspace volume, not baked into the image.

Writes pod_state.json on success. The bearer token is NEVER written here — it
stays in ~/.config/podlink/pod_bearer_token.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import runpod
from runpod.error import QueryError
from rich import print as rprint

import _secrets

POD_NAME = os.environ.get("PODLINK_POD_NAME", "podlink")

# RunPod's on-demand finder lands on a random Secure host; some lack free
# disk/resources and reject the pod ("This machine does not have the resources…")
# or momentarily report "no longer any instances available" — intermittently, even
# when the card is available (the SAME config succeeds on the next host). A failed
# create allocates nothing and bills nothing, so we just retry until a host takes it.
CREATE_RETRIES = 15         # attempts before giving up (~ CREATE_RETRIES * RETRY_DELAY s)
RETRY_DELAY = 15            # seconds between create attempts
_RETRYABLE_CREATE_ERRORS = (
    "does not have the resources",
    "no longer any instances",
    "instances available",
)

# --- the bundled multi-service image (build + push from ../pod_image) ----------
# Set PODLINK_IMAGE to the tag you `docker push` (no default — it's your image).
# The image bundles an LLM + embedder + reranker under supervisord for a RAG
# pipeline. podlink must NOT pass docker_args at create time, so the image's own
# CMD runs. Account-specific values are read from the environment, never source,
# so this repo stays generic.
IMAGE = os.environ.get("PODLINK_IMAGE", "").strip()

# If the image is PRIVATE, RunPod needs registry pull-creds on EVERY pull, and the
# SDK's create_pod cannot attach them directly — only a TEMPLATE carries a
# containerRegistryAuthId. ensure_template() bundles this credential id (from RunPod
# → Settings → Container Registry Auth) into a template the pod deploys from. Leave
# PODLINK_REGISTRY_AUTH_ID empty for a PUBLIC image (no auth needed).
CONTAINER_REGISTRY_AUTH_ID = os.environ.get("PODLINK_REGISTRY_AUTH_ID", "").strip()
TEMPLATE_NAME = os.environ.get("PODLINK_TEMPLATE_NAME", "podlink-pod")

# --- models the pod serves (sensible RAG defaults; override via env) -----------
# These defaults are a balanced 96 GB profile; point PODLINK_*_MODEL_ID at your own.
LLM_MODEL_ID = os.environ.get("PODLINK_LLM_MODEL_ID", "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ")
EMBED_MODEL_ID = os.environ.get("PODLINK_EMBED_MODEL_ID", "Qwen/Qwen3-Embedding-8B")
RERANK_MODEL_ID = os.environ.get("PODLINK_RERANK_MODEL_ID", "BAAI/bge-reranker-v2-m3")
# vLLM's --served-model-name. A client's model field must match this EXACTLY or
# requests 404. Must equal what the image serves; the image wrapper reads
# LLM_SERVED_NAME from env, and podlink passes it through.
LLM_SERVED_NAME = os.environ.get("PODLINK_LLM_SERVED_NAME", "llm")
# vLLM --quantization. Empty auto-detects — correct for a compressed-tensors AWQ
# checkpoint (forcing "awq_marlin" conflicts and vLLM refuses to start). For a
# gpt-oss LLM set "mxfp4"; NEVER "nvfp4" on sm_120.
LLM_QUANT = os.environ.get("PODLINK_LLM_QUANT", "")

# --- the three service ports, exposed via RunPod's HTTPS proxy ----------------
SERVICE_PORTS = {"llm": 8000, "embedder": 8080, "reranker": 8081}
# HTTP port mode -> each port gets a proxy URL with auto TLS.
EXPOSED_PORT = ",".join(f"{p}/http" for p in SERVICE_PORTS.values())

# --- pod sizing ---------------------------------------------------------------
CONTAINER_DISK_GB = 55    # must hold the bundled image (~38 GB on the 24.04/CUDA-12.9
                          # base) + vLLM's torch_compile_cache (~1-2 GB, on container
                          # disk, not the volume) + scratch. 40 GB was too tight.
# Persistent HF-cache storage mounted at /workspace. Two modes:
#   * NETWORK VOLUME (preferred) — set NETWORK_VOLUME_ID to a volume created in
#     RunPod (Storage -> Network Volumes). It SURVIVES terminate, so the up/down
#     cycle is terminate/recreate: create finds ANY host (reliable — no host-pinning),
#     and the ~36 GB of weights are kept on the volume (no re-download). Trade-off:
#     the volume is REGION-LOCKED to its data center, so the pod only deploys where
#     the volume lives — pick a region with RTX PRO 6000 stock; RunPod pins the pod
#     to that DC automatically when a networkVolumeId is given.
#   * DATA VOLUME (fallback, NETWORK_VOLUME_ID left empty) — a pod-scoped volume of
#     VOLUME_GB that is DESTROYED on terminate (weights re-download) and forces the
#     fragile stop/resume model (resume fails when the pinned host has no free GPU:
#     "not enough free GPUs on the host machine").
# Supplied via the environment so the infra id is NOT committed into source and
# the CLI and the web app read one shared value. Empty => Data Volume fallback
# (weights are DESTROYED on terminate). Export PODLINK_NETWORK_VOLUME_ID before
# launching podlink (or in its service env) to enable terminate-safe persistence.
NETWORK_VOLUME_ID = os.environ.get("PODLINK_NETWORK_VOLUME_ID", "").strip()
# Only used when NETWORK_VOLUME_ID is empty (Data Volume). Env-overridable so a
# volume-less profile can size the pod-scoped scratch to its weights (a ~76 GB
# big-LLM stack needs ~120, the default 30B stack fits in 50).
VOLUME_GB = int(os.environ.get("PODLINK_VOLUME_GB", "50"))
VOLUME_MOUNT = "/workspace"
# vLLM sizing — env-overridable like the model ids (the image wrappers already
# read MAX_MODEL_LEN / GPU_MEMORY_UTILIZATION from the pod env, so no image
# rebuild is needed to retune). Raise GPU share only alongside a smaller
# embedder: at 0.70 the two TEI services fit; a big-LLM profile (e.g. a ~66 GB
# Int4 MoE + a 4B embedder) wants ~0.85.
MAX_MODEL_LEN = int(os.environ.get("PODLINK_MAX_MODEL_LEN", "32768"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("PODLINK_GPU_MEMORY_UTILIZATION", "0.70"))
# SSH is handy for first-boot debug / pre-warm but opens an extra surface on every
# pod. On by default (preserves debugging during bring-up); set PODLINK_START_SSH=0
# to deploy without it once the image is trusted.
START_SSH = os.environ.get("PODLINK_START_SSH", "1").strip().lower() not in ("0", "false", "no", "")

# The specs never publish RunPod's exact GPU type id for the RTX Pro 6000, so we
# resolve it live (resolve_gpu_id) by matching this substring against the catalog.
GPU_MATCH = "RTX PRO 6000"

STATE_PATH = Path(__file__).resolve().parents[1] / "pod_state.json"
TEMPLATE_STATE_PATH = Path(__file__).resolve().parents[1] / "template_state.json"


def find_existing() -> dict | None:
    for p in runpod.get_pods():
        if p.get("name") == POD_NAME:
            return p
    return None


def ensure_template() -> str:
    """Return a RunPod template id that bundles the private IMAGE with its registry
    credential, so the pod can pull it.

    Why a template: the SDK's create_pod has no registry-auth parameter — only a
    template carries containerRegistryAuthId (see the module note on
    CONTAINER_REGISTRY_AUTH_ID). The pod then deploys with template_id.

    Idempotency: there is no list-templates API, and create_template always makes a
    NEW template, so we cache the id in template_state.json and reuse it. We
    recreate only if IMAGE or the credential changed (a stale template would point
    at the old image/cred). Delete template_state.json to force a fresh one.

    Secrets are NOT put in the template (env=[] by default) — the bearer/HF token
    stay in the pod-level env, so nothing sensitive lands in a persistent template.
    """
    if not IMAGE:
        raise RuntimeError("PODLINK_IMAGE is not set — set it to your pushed image tag "
                           "(e.g. in ~/.config/podlink/podlink.conf; see the README).")
    if TEMPLATE_STATE_PATH.exists():
        st = json.loads(TEMPLATE_STATE_PATH.read_text())
        if (st.get("template_id")
                and st.get("image") == IMAGE
                and st.get("registry_auth_id") == CONTAINER_REGISTRY_AUTH_ID):
            rprint(f"[cyan]Reusing template[/] {st['template_id']} "
                   f"(from {TEMPLATE_STATE_PATH.name})")
            return st["template_id"]
        rprint("[yellow]template_state.json is stale (IMAGE or credential changed) "
               "— creating a new template.[/]")

    # Template names are unique per RunPod account and there is no delete-by-name
    # here, so a fixed name collides the moment a template for an older image
    # exists (create fails, POD UP dies at "ensuring pod template"). Suffix with
    # the image tag so every image revision gets its own name.
    tag = IMAGE.rsplit(":", 1)[-1] if ":" in IMAGE else "latest"
    slug = "".join(c if c.isalnum() or c == "-" else "-" for c in tag)
    template_name = f"{TEMPLATE_NAME}-{slug}"

    rprint(f"[bold cyan]Creating template[/] {template_name!r} for image {IMAGE} …")
    # docker_start_cmd omitted -> the mutation sends dockerArgs "" -> the image's
    # own CMD (supervisord) runs, launching all three services. Ports/disk mirror
    # the pod so the template is self-consistent; the pod re-specifies them anyway.
    tmpl_kwargs = dict(
        name=template_name,
        image_name=IMAGE,
        container_disk_in_gb=CONTAINER_DISK_GB,
        ports=EXPOSED_PORT,
        is_serverless=False,
    )
    if CONTAINER_REGISTRY_AUTH_ID:            # private image only; omit for a public one
        tmpl_kwargs["registry_auth_id"] = CONTAINER_REGISTRY_AUTH_ID
    tmpl = runpod.create_template(**tmpl_kwargs)
    template_id = tmpl["id"]
    TEMPLATE_STATE_PATH.write_text(json.dumps({
        "template_id": template_id,
        "name": tmpl.get("name"),
        "image": IMAGE,
        "registry_auth_id": CONTAINER_REGISTRY_AUTH_ID,
    }, indent=2) + "\n")
    rprint(f"[green]Created template[/] {template_id} → wrote {TEMPLATE_STATE_PATH.name}")
    return template_id


def resolve_gpu_id(match: str = GPU_MATCH) -> str:
    """Resolve the RunPod gpu_type_id for a FULL 96 GB RTX PRO 6000.

    RunPod lists several RTX PRO 6000 SKUs and the naive "first substring match"
    picked the wrong one: the *Max-Q Workstation Edition* (a desktop variant
    rarely stocked in Secure Cloud) → deploys hit "no instances available". The
    catalog also has 24/48 GB **MIG slices** (too small for our stack) and a
    *Workstation Edition*. The datacenter SKU that Secure Cloud actually stocks is
    the **Server Edition** (displayName "RTX PRO 6000").

    So among entries matching `match`, we drop MIG slices and anything under a
    full 96 GB card, then prefer the Server Edition, and return its gpu_type_id.
    """
    needle = match.upper().replace(" ", "")
    candidates = []
    for g in runpod.get_gpus():                              # live GPU catalog
        gid = g.get("id") or ""
        label = g.get("displayName") or gid
        if needle not in (gid + " " + label).upper().replace(" ", ""):
            continue
        if "MIG" in gid.upper():                             # skip 24/48 GB MIG slices
            continue
        if (g.get("memoryInGb") or 0) < 90:                  # require a full 96 GB card
            continue
        candidates.append(gid)
    if not candidates:
        raise RuntimeError(
            f"No full-96GB GPU type matched {match!r}. Inspect the catalog with "
            f"runpod.get_gpus()."
        )
    # Prefer the Secure-stocked datacenter SKU (Server Edition) over the
    # workstation variants; stable tiebreak by id.
    candidates.sort(key=lambda i: (0 if "SERVER EDITION" in i.upper() else 1, i))
    return candidates[0]


def is_retryable_create_error(e: Exception) -> bool:
    """True when a create failure is the transient host-capacity lottery — safe to
    retry (a failed create allocates nothing). See _RETRYABLE_CREATE_ERRORS."""
    return any(s in str(e) for s in _RETRYABLE_CREATE_ERRORS)


def _pod_env(bearer: str, hf: str) -> dict:
    """The pod's runtime env — secrets + model config, read by the image wrappers.
    Secrets live here (pod env), never in the persistent template."""
    env = {
        # weight-pull token, seen by all three services in the container
        "HF_TOKEN": hf,
        "HUGGING_FACE_HUB_TOKEN": hf,     # some loaders read this name instead
        # vLLM reads its API key from env (never argv -> not visible in ps)
        "VLLM_API_KEY": bearer,
        # Same bearer gates the two TEI services: their :8080/:8081 endpoints are
        # reachable over RunPod's PUBLIC proxy, so they must not be keyless. The
        # wrappers export this as TEI's native API_KEY env (not argv).
        "TEI_API_KEY": bearer,
        # which models to serve + how to size vLLM (read by the image wrappers)
        "LLM_MODEL_ID": LLM_MODEL_ID,
        "LLM_SERVED_NAME": LLM_SERVED_NAME,   # vLLM --served-model-name (client model must match)
        "EMBED_MODEL_ID": EMBED_MODEL_ID,
        "RERANK_MODEL_ID": RERANK_MODEL_ID,
        "LLM_QUANT": LLM_QUANT,
        "MAX_MODEL_LEN": str(MAX_MODEL_LEN),
        "GPU_MEMORY_UTILIZATION": str(GPU_MEMORY_UTILIZATION),
    }
    # Optional tuning passthroughs — only sent when set, so the image defaults
    # stay authoritative otherwise.
    for src, dst in (
        ("PODLINK_VLLM_EXTRA_ARGS", "VLLM_EXTRA_ARGS"),          # extra vllm serve flags
        ("PODLINK_PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"),  # allocator tuning
    ):
        val = os.environ.get(src, "").strip()
        if val:
            env[dst] = val
    return env


def create_pod_once(gpu_type_id: str, bearer: str, hf: str, template_id: str) -> dict:
    """ONE create attempt from the bundled image. Raises QueryError (retryable via
    is_retryable_create_error, or a real error) on failure; returns the pod on
    success. Callers own the retry policy — the CLI create_pod loop below, and the
    web driver's cancel-aware, UI-reporting loop. No docker_args (that would
    override the image CMD). template_id carries the private-registry credential."""
    # Persistent storage: a Network Volume (survives terminate; region-locked) when
    # NETWORK_VOLUME_ID is set, else a pod-scoped Data Volume (destroyed on terminate).
    create_kwargs = dict(
        name=POD_NAME,
        image_name=IMAGE,
        template_id=template_id,          # carries the ghcr pull credential
        gpu_type_id=gpu_type_id,
        cloud_type="SECURE",
        gpu_count=1,
        container_disk_in_gb=CONTAINER_DISK_GB,
        volume_mount_path=VOLUME_MOUNT,
        ports=EXPOSED_PORT,               # "8000/http,8080/http,8081/http"
        env=_pod_env(bearer, hf),
        support_public_ip=False,          # all traffic via RunPod's HTTPS proxy
        start_ssh=START_SSH,              # SSH for first-boot debug (PODLINK_START_SSH=0 to disable)
    )
    if NETWORK_VOLUME_ID:
        create_kwargs["network_volume_id"] = NETWORK_VOLUME_ID  # persists across terminate
    else:
        create_kwargs["volume_in_gb"] = VOLUME_GB               # pod-scoped, lost on terminate
    # runpod 1.7.13's create_pod does `print(f"raw_response: {raw_response}")`, and
    # raw_response echoes the pod env — HF_TOKEN and the VLLM/TEI bearer — to stdout.
    # Capture and discard that stdout so the secrets never leak; we use the return value.
    with contextlib.redirect_stdout(io.StringIO()):
        return runpod.create_pod(**create_kwargs)


def create_pod(gpu_type_id: str, bearer: str, hf: str, template_id: str | None = None) -> dict:
    """CLI path: resolve the template, then retry the host-selection lottery, printing
    progress to the terminal. The web driver does NOT use this — it runs its own
    cancel-aware loop over create_pod_once so retries are visible in the UI."""
    if template_id is None:
        template_id = ensure_template()
    last_err: QueryError | None = None
    for attempt in range(1, CREATE_RETRIES + 1):
        rprint(f"[bold cyan]Creating pod[/] on [bold]{gpu_type_id}[/] "
               f"(template {template_id}) — attempt {attempt}/{CREATE_RETRIES} …")
        try:
            return create_pod_once(gpu_type_id, bearer, hf, template_id)
        except QueryError as e:
            if not is_retryable_create_error(e):
                raise                         # a real error (bad spec, auth, …) — surface it
            last_err = e
            if attempt < CREATE_RETRIES:
                rprint(f"[yellow]  no host with capacity yet — retrying in "
                       f"{RETRY_DELAY}s[/] ({str(e)[:70]})")
                time.sleep(RETRY_DELAY)
    raise RuntimeError(
        f"No Secure host accepted the pod after {CREATE_RETRIES} attempts. "
        f"Last error: {last_err}. Try again later, pin a data_center_id, or trim disk."
    )


def try_create() -> dict:
    """CLI helper: resolve the RTX Pro 6000 id and create the pod (no GPU fallback
    — the image + models are sized for the 96 GB card; a smaller GPU would OOM)."""
    bearer = _secrets.bearer_token()
    hf = _secrets.hf_token()
    gpu_id = resolve_gpu_id()
    try:
        return create_pod(gpu_id, bearer, hf)   # create_pod ensures the template
    except Exception as e:
        # Avoid printing the exception payload — SDK exceptions can embed request
        # context (potentially the API key in some versions).
        rprint(f"[yellow]Create failed on {gpu_id}[/] ({type(e).__name__}). "
               f"Check the RunPod dashboard for details.")
        raise


def derive_proxy_url(pod_id: str, port: int = 8000) -> str:
    """RunPod HTTPS proxy URL for a given exposed port.
    Format: docs.runpod.io/pods/configuration/expose-ports"""
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def service_urls(pod_id: str) -> dict[str, str]:
    """The three proxy base URLs keyed by service name (llm/embedder/reranker)."""
    return {name: derive_proxy_url(pod_id, port) for name, port in SERVICE_PORTS.items()}


def wait_for_running(pod_id: str, timeout_s: int = 900) -> dict:
    rprint("[bold]Waiting for pod to reach RUNNING…[/] (model weight pull can take several min)")
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
    urls = service_urls(pod_id)
    state = {
        "pod_id": pod_id,
        "name": pod.get("name"),
        "proxy_url": urls["llm"],          # primary (kept for back-compat)
        "service_urls": urls,              # all three: llm / embedder / reranker
        "models": {
            "llm": LLM_MODEL_ID,
            "served_as": LLM_SERVED_NAME,
            "embedder": EMBED_MODEL_ID,
            "reranker": RERANK_MODEL_ID,
        },
        "gpu_type": gpu_type_id,
        "image": IMAGE,
        "created_at": pod.get("lastStatusChange") or pod.get("createdAt"),
    }
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")
    rprint(f"[green]Wrote[/] {STATE_PATH}")
    for name, url in urls.items():
        rprint(f"[bold]{name} URL:[/] {url}")


def main() -> None:
    runpod.api_key = _secrets.runpod_api_key()

    existing = find_existing()
    if existing and existing.get("desiredStatus") == "RUNNING":
        # A live pod already serves — adopt it and refresh the URLs on disk.
        pod_id = existing["id"]
        rprint(f"[yellow]Pod {POD_NAME!r} already RUNNING[/]: id={pod_id}")
        pod = runpod.get_pod(pod_id)
        gpu_type = (pod.get("machine", {}).get("gpuTypeId")
                    or pod.get("gpuTypeId") or "unknown")
        write_state({**pod, "id": pod_id}, gpu_type)
        return
    if existing:
        # Exists but not RUNNING: a leftover (crashed, or reaping after a
        # terminate). We never resume — the lifecycle is terminate/recreate — so
        # create a fresh pod; the Network Volume keeps the weights, and RunPod
        # reaps the dead one.
        rprint(f"[yellow]Pod {POD_NAME!r} exists but is "
               f"{existing.get('desiredStatus')}[/] — creating a fresh pod "
               f"(terminate/recreate lifecycle).")

    pod = try_create()
    pod_id = pod["id"]
    rprint(f"[green]Pod created:[/] id={pod_id}")
    pod = wait_for_running(pod_id)
    gpu_type = (pod.get("machine", {}).get("gpuTypeId")
                or pod.get("gpuTypeId") or GPU_MATCH)
    write_state(pod, gpu_type)
    rprint("[bold green]Pod is up.[/] The three services may still be loading model "
           "weights — check logs in the RunPod dashboard, or poll /v1/models (8000) "
           "and /health (8080, 8081) until each returns 200.")


if __name__ == "__main__":
    main()
