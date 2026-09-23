"""The RunPod provider: podlink's original cloud, behind the Provider contract.

This is the provider class. Package-level hooks that must work before a
provider instance exists (env normalisation, the launcher include) live in
__init__.py alongside it.

Everything here was previously inline in the driver or the server; the logic is
unchanged, only relocated. The vendored pod_control/pod_up.py stays the single
source of truth for RunPod's shape (image, template, ports, sizing, volume), and
this class is the adapter between it and app/driver.py.

Lifecycle: by default POD DOWN *terminates* and the weights live on a Network
Volume. PODLINK_LIFECYCLE=stop (driver-owned) makes POD DOWN *stop* instead, so
POD UP can resume the pod without re-pulling the image. A stopped RunPod pod is
host-pinned: resume fails with "not enough free GPUs on the host machine" when
someone else has rented that GPU, so the driver retries, then falls back to
terminate + create. This provider supplies the pieces: stop, is_stopped,
is_retryable_resume_error, gpu_count, matches_stack.
"""
from __future__ import annotations

import importlib  # re-bake pod_up's import-time constants after a profile switch
import json       # read pod_state.json as a last-resort id source
import re         # validate a client-supplied pod id
import sys        # reach the already-imported pod_up module for reload

# Importing vendored puts pod_control/ on sys.path, which `import pod_up` below
# relies on — so it must come first.
from ...vendored import _secrets, read_secret
from ... import stack as stack_contract  # cloud-neutral image env contract + client .env block

import runpod                        # noqa: E402  RunPod SDK
from runpod.error import QueryError  # noqa: E402  raised by create_pod on the capacity lottery

import pod_up  # noqa: E402  vendored: constants + find_existing/create_pod_once/…

# RunPod pod ids are short lowercase-alnum strings; validated before any SDK call
# so a malformed/injected target can't reach runpod.get_pod.
_POD_ID_RE = re.compile(r"^[a-z0-9]{6,40}$")


# Env keys that carry secrets: excluded from the stack fingerprint (never compared,
# never logged).
_SECRET_ENV = frozenset({"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "VLLM_API_KEY", "TEI_API_KEY"})

# Every NON-secret key pod_up._pod_env can send (base + optional passthroughs).
# The stack fingerprint compares only these: RunPod adds keys of its own to a
# pod's env (PUBLIC_KEY, the account SSH key — seen live 2026-09-23) and drops
# keys whose value is empty (LLM_QUANT=""), so a whole-env comparison judged
# every stopped pod stale and resume never ran. tests/test_lifecycle.py pins this
# set to what _pod_env actually emits.
_CONFIG_ENV = frozenset({
    "LLM_MODEL_ID", "LLM_SERVED_NAME", "EMBED_MODEL_ID", "RERANK_MODEL_ID", "LLM_QUANT",
    "MAX_MODEL_LEN", "GPU_MEMORY_UTILIZATION", "RERANK_BACKEND",
    "VLLM_EXTRA_ARGS", "PYTORCH_CUDA_ALLOC_CONF", "RERANK_GPU_MEMORY_UTILIZATION",
    "RERANK_VLLM_EXTRA_ARGS", "RERANK_MAX_MODEL_LEN", "RERANK_WAIT_FOR_EMBEDDER_S",
})


def _env_dict(env) -> dict[str, str]:
    """The pod record's env as a dict — the API returns ["K=V", ...] (or a dict)."""
    if isinstance(env, dict):
        return {str(k): str(v) for k, v in env.items()}
    out: dict[str, str] = {}
    for item in env or []:
        if isinstance(item, str) and "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
        elif isinstance(item, dict) and "key" in item:      # {key, value} shape
            out[str(item["key"])] = str(item.get("value", ""))
    return out


class RunPodProvider:
    """Drives RunPod Secure Cloud pods for app/driver.py."""

    name = "runpod"
    #: create_pod_once raises QueryError both for the transient host-capacity
    #: lottery and for real errors; is_retryable_create_error tells them apart.
    create_error_types = (QueryError,)
    #: RunPod pods can be stopped and resumed (host-pinned; see module docstring).
    supports_stop = True
    #: resume_pod raises QueryError for "no free GPU on the host" and real errors
    #: alike; is_retryable_resume_error tells them apart.
    resume_error_types = (QueryError,)

    # --- configuration -----------------------------------------------------

    def reload_config(self) -> None:
        """Re-execute pod_up so its import-time constants re-read the environment.

        importlib.reload re-runs the module in its existing module object, so
        this module's `pod_up` reference (and any other holder's) sees the new
        constants without re-importing. pod_up's module level is pure constant
        assignment — no side effects — which is what makes this safe.
        """
        importlib.reload(sys.modules["pod_up"])

    def snapshot_fields(self) -> dict:
        """Deploy flags the UI renders on every frame (all process constants)."""
        return {
            "provider": self.name,
            "persistence_configured": self.persistence_configured(),
            "persistence_id": pod_up.NETWORK_VOLUME_ID or None,
            "persistence_label": "Network Volume",
            "persistence_off_hint": "none — Data-Volume mode (weights not persisted)",
            "llm_model_id": pod_up.LLM_MODEL_ID,      # which stack this launch serves
        }

    # --- authentication ----------------------------------------------------

    def authenticate(self) -> None:
        """Point the SDK at the local API key (and register it for redaction)."""
        runpod.api_key = read_secret(_secrets.runpod_api_key)

    # --- inventory ---------------------------------------------------------

    def valid_instance_id(self, instance_id: str) -> bool:
        return bool(_POD_ID_RE.match(instance_id))

    def list_instances(self) -> list[dict]:
        """A WHITELISTED list of the account's pods for the selector.

        Only safe scalar fields are returned — never the raw pod dict, which can
        embed `env` (VLLM_API_KEY / HF_TOKEN). Adding fields here is a security
        decision: never surface `env`, ports, or anything credential-bearing.
        """
        pods = []
        for p in runpod.get_pods():                          # enumerate every pod
            machine = p.get("machine") or {}                 # gpu type nests here
            pods.append({
                "id": p.get("id"),
                "name": p.get("name"),
                "status": p.get("desiredStatus"),
                "gpu": machine.get("gpuTypeId") or p.get("gpuTypeId"),
                "cost_per_hr": p.get("costPerHr"),
            })
        return pods

    def find_existing(self) -> dict | None:
        """The pod named POD_NAME, or None (RUNNING preferred, then EXITED —
        see pod_up.find_existing)."""
        return pod_up.find_existing()

    def get_instance(self, instance_id: str) -> dict | None:
        return runpod.get_pod(instance_id)

    def instance_id(self, instance: dict) -> str:
        return instance["id"]

    def status_of(self, instance: dict) -> str | None:
        return instance.get("desiredStatus") if instance else None

    def is_running(self, instance: dict) -> bool:
        """Desired state is RUNNING — the pod may still be pulling/booting."""
        return bool(instance) and instance.get("desiredStatus") == "RUNNING"

    def is_up(self, instance: dict) -> bool:
        """RUNNING *and* carrying a runtime — the container is actually up."""
        return self.is_running(instance) and bool(instance.get("runtime"))

    def cost_per_hr(self, instance: dict) -> float | None:
        if not instance or instance.get("costPerHr") is None:
            return None
        try:
            return float(instance["costPerHr"])
        except (TypeError, ValueError):
            return None

    # --- lifecycle ---------------------------------------------------------

    def resume(self, instance: dict) -> None:
        runpod.resume_pod(self.instance_id(instance), gpu_count=instance.get("gpuCount") or 1)

    def stop(self, instance_id: str) -> None:
        """Release the GPU but keep the pod (and its cached image) on its host."""
        runpod.stop_pod(instance_id)

    def is_stopped(self, instance: dict) -> bool:
        """EXITED = stopped by us or by the user, resumable. A terminated pod
        vanishes from the API instead of reporting a status."""
        return bool(instance) and instance.get("desiredStatus") == "EXITED"

    def is_retryable_resume_error(self, exc: Exception) -> bool:
        """Only 'the host's GPU is taken' is worth retrying — verified live
        2026-09-23: "There are not enough free GPUs on the host machine to start
        this pod." Anything else (pod gone, bad state) falls back at once."""
        return isinstance(exc, QueryError) and "not enough free gpus" in str(exc).lower()

    def gpu_count(self, instance: dict) -> int | None:
        count = (instance or {}).get("gpuCount")
        return int(count) if isinstance(count, (int, float)) else None

    def matches_stack(self, instance: dict) -> bool:
        """Compare the stopped pod's image + non-secret env with what a create
        would send now. Every non-secret key on EITHER side must agree, so an
        optional passthrough present on only one side (e.g. RERANK_BACKEND) is a
        mismatch too. The secret values are never read or compared."""
        return bool(instance) and not self.stack_diff(instance)

    def stack_diff(self, instance: dict) -> list[str]:
        """Setting names that differ: the image, plus podlink's own config keys
        (_CONFIG_ENV) on either side. Keys RunPod adds (PUBLIC_KEY, …) are
        ignored, and an empty value equals a missing key, because RunPod drops
        empty env values when it stores the pod.

        RunPod stores env values as the GraphQL string literals parse, i.e. the
        raw values _pod_env builds (the create path's _gql_escape_env escaping is
        undone by the parser), so raw-vs-raw comparison is correct.
        """
        if not instance:
            return ["instance"]
        diff = [] if instance.get("imageName") == pod_up.IMAGE else ["image"]
        want = {k: str(v) for k, v in pod_up._pod_env("", "").items() if k in _CONFIG_ENV}
        have = {k: v for k, v in _env_dict(instance.get("env")).items() if k in _CONFIG_ENV}
        diff += sorted(k for k in set(want) | set(have) if (want.get(k) or "") != (have.get(k) or ""))
        return diff

    def prepare_create(self, session) -> dict | None:
        """Resolve the GPU type id and the registry-cred template.

        Both are live lookups that can take a moment, so each is announced as a
        phase and cancel is checked between them — POD DOWN pressed here should
        not have to wait for a template round-trip.
        """
        if session.cancel.is_set():                      # Down pressed before we resolve
            return None
        session.update(phase="resolving RTX Pro 6000 GPU id")   # live catalog lookup
        gpu_id = pod_up.resolve_gpu_id()                 # exact RunPod gpu_type_id
        if session.cancel.is_set():
            return None
        session.update(phase="ensuring pod template")    # registry-cred template (cached)
        template_id = pod_up.ensure_template()
        return {"gpu_id": gpu_id, "template_id": template_id}

    def create_once(self, ctx: dict, secrets: dict, attempt: int) -> dict:
        """ONE create attempt from the bundled image.

        The secrets travel as pod env (that is where the image's wrappers read
        them); pod_up keeps them out of the persistent template.
        """
        return pod_up.create_pod_once(ctx["gpu_id"], secrets["bearer"], secrets["hf"],
                                      ctx["template_id"])

    def is_retryable_create_error(self, exc: Exception) -> bool:
        """True for the transient host-capacity lottery (a failed create bills nothing)."""
        return pod_up.is_retryable_create_error(exc)

    def create_phase(self, ctx: dict, attempt: int, retries: int) -> str:
        return f"creating pod on {ctx['gpu_id']} — attempt {attempt}/{retries}"

    def capacity_note(self, attempt: int, retries: int, delay: int) -> str:
        return f"no host with capacity yet — attempt {attempt}/{retries}; retrying in {delay}s"

    def create_exhausted_message(self, retries: int, delay: int) -> str:
        return (f"no Secure host in the volume's region accepted the pod after {retries} "
                f"attempts (~{retries * delay // 60} min). RTX PRO 6000 capacity is transient — "
                f"press POD UP to keep trying, or try again later.")

    def terminate(self, instance_id: str) -> None:
        """Release the GPU and the pod (weights persist on the Network Volume)."""
        runpod.terminate_pod(instance_id)

    # --- access lifecycle ---------------------------------------------------
    # RunPod publishes every exposed port on its own HTTPS proxy, so the URLs
    # from service_urls() are reachable the moment the pod is up — there is no
    # local path for podlink to open or close. Both hooks are deliberate no-ops.

    def ensure_access(self, instance_id: str) -> None:
        """No-op: the proxy URLs need nothing opened locally."""

    def access_events(self) -> list[str]:
        """No-op: there is no access process to report on."""
        return []

    def release_access(self) -> None:
        """No-op: nothing was opened, so nothing leaks."""

    # --- endpoints + local state -------------------------------------------

    def service_urls(self, instance_id: str) -> dict[str, str]:
        """The three RunPod HTTPS-proxy base URLs, keyed llm/embedder/reranker."""
        return pod_up.service_urls(instance_id)

    def stack_config(self) -> dict:
        """What this launch will serve — read by the stack test and the preflight."""
        return {
            "image": pod_up.IMAGE or None,
            "llm_model_id": pod_up.LLM_MODEL_ID,
            "llm_served_name": pod_up.LLM_SERVED_NAME,   # client's `model` must match exactly
            "embed_model_id": pod_up.EMBED_MODEL_ID,
            "rerank_model_id": pod_up.RERANK_MODEL_ID,
            "max_model_len": pod_up.MAX_MODEL_LEN,
            "gpu_memory_utilization": pod_up.GPU_MEMORY_UTILIZATION,
            "rerank_backend": pod_up.RERANK_BACKEND,     # decides the stack test's rerank call
        }

    def client_env(self, instance_id: str, embedding_dim: int | None) -> str:
        """The RAG-client .env block for this pod: proxy URLs + model names, with the
        bearer left as a PLACEHOLDER (never the real secret — it must not reach the
        browser). Paste your pod_bearer_token where marked. EMBEDDING_DIMENSIONS is
        left for 'Test stack' to detect, since it depends on what the embedder serves."""
        # Delegates to the shared block (as the GCP provider does) so the two
        # clouds can never emit different client configs.
        return stack_contract.client_env_block(
            self.service_urls(instance_id), self.stack_config(), embedding_dim)

    def preflight(self) -> list[tuple[str, str]]:
        """RunPod-specific launch checks: the API key, the image, the volume mode.

        The neutral checks (bearer + HF token, the venv) are the launcher's; this
        is only what would fail or surprise on THIS cloud.
        """
        rows: list[tuple[str, str]] = []
        try:
            read_secret(_secrets.runpod_api_key)
            rows.append(("ok", "runpod_api_key readable (0600)"))
        except RuntimeError as e:                        # missing / mis-permissioned / empty
            rows.append(("fail", f"runpod_api_key: {e.__cause__ or e}"))
        if pod_up.IMAGE:
            rows.append(("ok", f"PODLINK_IMAGE = {pod_up.IMAGE}"))
        else:
            rows.append(("fail", "PODLINK_IMAGE not set (required — see the README)"))
        rows.append(("info", "registry auth: set (private image)" if pod_up.CONTAINER_REGISTRY_AUTH_ID
                     else "registry auth: none (public image)"))
        if pod_up.NETWORK_VOLUME_ID:
            rows.append(("ok", f"persistence: Network Volume {pod_up.NETWORK_VOLUME_ID}"))
        else:
            rows.append(("warn", f"persistence: none — Data-Volume ({pod_up.VOLUME_GB} GB) is destroyed "
                                 "on POD DOWN; weights re-download each launch"))
        return rows

    def persistence_configured(self) -> bool:
        """True when a Network Volume is set (weights persist across terminate).

        When False, POD DOWN's terminate DESTROYS the ~36 GB of downloaded weights,
        so the server refuses without an explicit confirm and the UI warns first.
        """
        return bool(pod_up.NETWORK_VOLUME_ID)

    def write_state(self, instance_id: str) -> None:
        """Write pod_state.json for a fully-up pod, matching the CLI's contract."""
        pod = runpod.get_pod(instance_id)                     # refresh the pod record
        gpu_type = (pod.get("machine", {}).get("gpuTypeId")   # SDK nests gpu type here…
                    or pod.get("gpuTypeId") or "unknown")     # …or here, depending on version
        pod_up.write_state({**pod, "id": instance_id}, gpu_type)

    def clear_state(self) -> None:
        """Remove pod_state.json after a confirmed terminate.

        The disk fallback in the driver's id resolution reads this file; leaving
        it after the pod is destroyed lets a later Down hand terminate_pod an
        already-dead id, which flips the UI to a spurious ERROR. Best-effort —
        cleanup never fails Down.
        """
        try:
            pod_up.STATE_PATH.unlink(missing_ok=True)   # gone-or-not, end up with no file
        except Exception:  # noqa: BLE001 — cleanup is best-effort, never fatal to a stop
            pass

    def state_file_instance_id(self) -> str | None:
        """The pod id recorded on disk, or None (missing, unreadable or partial)."""
        if not pod_up.STATE_PATH.exists():
            return None
        try:
            return json.loads(pod_up.STATE_PATH.read_text()).get("pod_id")
        except Exception:  # noqa: BLE001                # corrupt/partial state file
            return None
