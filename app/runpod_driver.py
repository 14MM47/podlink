"""Non-interactive orchestration over the vendored pod_control/ scripts.

The original pod_up.py / pod_down.py are CLI tools: they prompt with
rich.Confirm and quit via sys.exit — both fatal to a webapp. We keep edits to
the vendored copies minimal (see pod_control/PROVENANCE.md), so instead we
import their reusable, non-interactive pieces (constants, find_existing,
create_pod, derive_proxy_url, write_state) and re-implement the control loop
here with:

  * automatic GPU fallback (no prompt),
  * cooperative cancellation on every poll tick,
  * immediate pod-id capture the instant a pod is created, and
  * a verified stop that confirms GPU billing has actually ended.

The two entry points, `start(session)` and `stop(session)`, each run in their
own background thread.
"""
from __future__ import annotations  # allow `dict | None` etc. annotations

import json          # read pod_state.json as a last-resort id source
import sys           # to mutate sys.path for the vendored imports
import time          # wall-clock deadlines and inter-poll sleeps
from pathlib import Path  # build the pod_control directory path

# Make the vendored pod_control modules importable with their original sibling
# imports (`import _secrets`, `import egress_logger`) intact.
POD_CONTROL_DIR = Path(__file__).resolve().parents[1] / "pod_control"  # …/podlink/pod_control
if str(POD_CONTROL_DIR) not in sys.path:              # avoid duplicate entries on reload
    sys.path.insert(0, str(POD_CONTROL_DIR))          # front of path so our copy wins

import runpod            # noqa: E402  RunPod SDK (imported after sys.path tweak)
import _secrets          # noqa: E402  vendored secret reader (0600/ownership checked)
import egress_logger     # noqa: E402  vendored audited httpx client
import pod_up            # noqa: E402  vendored: constants + find_existing/create_pod/…

from .session import PodSession, State  # our state machine types

# How long to wait, in seconds, for each phase before declaring failure.
RUNNING_TIMEOUT_S = 900   # RunPod allocation + container boot
READY_TIMEOUT_S = 900     # vLLM model-weight load until /v1/models == 200
STOP_VERIFY_TIMEOUT_S = 180  # max wait to confirm the pod left RUNNING
POLL_S = 10               # inter-poll sleep (interruptible by cancel)


def _sleep_or_cancel(session: PodSession, seconds: float) -> bool:
    """Sleep up to `seconds`, returning True immediately if cancel is raised."""
    return session.cancel.wait(seconds)   # Event.wait returns True the moment it's set


def _read_secret(getter) -> str:
    """Read a secret and register it for verbatim redaction in the egress log.

    The vendored getter calls sys.exit() (raising SystemExit — a BaseException)
    when a secret is missing, mis-permissioned, or empty. SystemExit slips past
    the `except Exception` handlers in start()/stop() and the /pods route, which
    would silently kill the stop worker (session stuck STOPPING while the pod
    keeps billing) or crash the server. Convert it to a normal RuntimeError so
    those handlers catch it and surface a recoverable error.
    """
    try:
        value = getter()                    # vendored getter; sys.exit on missing/bad
    except SystemExit as e:                  # missing / mis-permissioned / empty secret
        raise RuntimeError("secret unavailable — check ~/.config/podlink") from e
    egress_logger.register_secret(value)    # strip this exact value from any log line
    return value


def list_pods() -> list[dict]:
    """Return a WHITELISTED list of the account's pods for the selector.

    Only safe scalar fields are returned — never the raw pod dict, which can
    embed `env` (VLLM_API_KEY / HF_TOKEN). Adding fields here is a security
    decision: never surface `env`, ports, or anything credential-bearing.
    """
    runpod.api_key = _read_secret(_secrets.runpod_api_key)   # authenticate the SDK
    pods = []
    for p in runpod.get_pods():                              # enumerate every pod
        machine = p.get("machine") or {}                     # gpu type nests here
        pods.append({
            "id": p.get("id"),
            "name": p.get("name"),
            "status": p.get("desiredStatus"),
            "gpu": machine.get("gpuTypeId") or p.get("gpuTypeId"),
            "cost_per_hr": p.get("costPerHr"),
        })
    return pods


def network_volume_configured() -> bool:
    """True when a RunPod Network Volume is set (weights persist across terminate).

    When False, POD DOWN's terminate DESTROYS the ~36 GB of downloaded weights, so
    the web UI warns before terminating — mirroring the CLI pod_down.py prompt.
    """
    return bool(pod_up.NETWORK_VOLUME_ID)


# ---------------------------------------------------------------------------
# Pod Up
# ---------------------------------------------------------------------------

def start(session: PodSession) -> None:
    """Provision, resume, or adopt the pod, then wait until it's up.

    Assumes the session is already in STARTING (set atomically by the request
    handler). If a specific pod was selected (session.target_pod_id) it is
    adopted directly; otherwise Auto creates/resumes the podlink pod. On cancel
    it returns quietly and lets the stop worker own the state; on failure it
    flips the session to ERROR.
    """
    try:
        runpod.api_key = _read_secret(_secrets.runpod_api_key)        # authenticate the SDK
        target = session.target_pod_id                    # a specific pod to adopt, or None

        if target:                                        # ---- adopt the selected pod ----
            pod = runpod.get_pod(target)                  # fetch the chosen pod
            if not pod:                                   # it was deleted since listing
                raise RuntimeError("selected pod no longer exists")
            pod_id = target                               # id already recorded by the session
            session.update(phase="adopting selected pod")
            if pod.get("desiredStatus") != "RUNNING":     # stopped -> resume it
                session.update(phase="resuming selected pod")
                runpod.resume_pod(pod_id, gpu_count=pod.get("gpuCount") or 1)
            if not _wait_for_running(session, pod_id):    # poll until RUNNING (or cancel)
                return
            # An arbitrary pod may not serve /v1/models, so RUNNING is 'up' here;
            # skip the readiness probe and write_state (both assume the vLLM pod).
            session.commit_running()                      # -> RUNNING (unless a stop won)
            return

        # ---- Auto: adopt a RUNNING podlink pod, else create a fresh one ----
        # The lifecycle is terminate/recreate (see stop()), so we never resume. A
        # non-RUNNING pod named `podlink` here is a leftover — crashed, or still
        # being reaped after a terminate. Resuming it would error (a terminating
        # pod can't resume), and a quick DOWN->UP would then fail instead of just
        # making a new pod. So we adopt ONLY a RUNNING pod and otherwise create
        # fresh; RunPod reaps the dead one.
        existing = pod_up.find_existing()                 # is a pod already named podlink?
        if existing is not None and existing.get("desiredStatus") == "RUNNING":
            pod_id = existing["id"]                       # adopt the live pod
            # Capture the id before anything can block — closes the billing race.
            session.update(pod_id=pod_id, phase="adopting running pod")
        else:                                             # none, or a dead/reaping leftover
            pod = _create_with_fallback(session)          # resolve RTX Pro 6000 + create
            if pod is None:                               # cancel arrived during create
                return                                    # let the stop worker take over
            pod_id = pod["id"]                            # id of the freshly created pod
            session.update(pod_id=pod_id, phase="pod created")  # capture id immediately

        if not _wait_for_running(session, pod_id):        # poll until RUNNING (or cancel)
            return                                        # cancelled mid-wait

        llm_url = pod_up.derive_proxy_url(pod_id, pod_up.SERVICE_PORTS["llm"])  # primary URL
        session.update(proxy_url=llm_url,
                       phase="waiting for all three services to become healthy")
        if not _wait_for_all_ready(session, pod_id):       # LLM + embedder + reranker (or cancel)
            return                                        # cancelled mid-wait

        # Persist state only once safely up, matching the scripts' contract.
        pod = runpod.get_pod(pod_id)                      # refresh the pod record
        gpu_type = (pod.get("machine", {}).get("gpuTypeId")   # SDK nests gpu type here…
                    or pod.get("gpuTypeId") or "unknown")     # …or here, depending on version
        pod_up.write_state({**pod, "id": pod_id}, gpu_type)   # write pod_state.json

        if not session.commit_running():                 # flip to RUNNING unless a stop won
            return                                        # a stop overtook us at the finish line
    except Exception as e:  # noqa: BLE001 — surface any failure to the UI
        # Type only — str(e) from the SDK/httpx can embed request context
        # (URLs, headers, the RunPod API key). The phase field already names
        # the failing step; full detail stays server-side / in the dashboard.
        session.update(state=State.ERROR, phase="error during start",
                       error=f"{type(e).__name__} — check the RunPod dashboard for details")


def _create_with_fallback(session: PodSession) -> dict | None:
    """Resolve the RTX Pro 6000 id and create the pod (no GPU fallback).

    The bundled image + models are sized for the 96 GB card, so a smaller GPU
    would OOM rather than help — we target one card and let any create error
    surface to start()'s handler.
    """
    bearer = _read_secret(_secrets.bearer_token)         # -> pod env VLLM_API_KEY
    hf = _read_secret(_secrets.hf_token)                 # -> weight-pull token (all 3 services)
    if session.cancel.is_set():                          # Down pressed before we resolve
        return None
    session.update(phase="resolving RTX Pro 6000 GPU id")  # live catalog lookup
    gpu_id = pod_up.resolve_gpu_id()                     # exact RunPod gpu_type_id
    if session.cancel.is_set():                          # Down pressed during the lookup
        return None
    session.update(phase=f"creating pod on {gpu_id}")    # progress text
    return pod_up.create_pod(gpu_id, bearer, hf)         # blocking SDK call; returns pod dict


def _wait_for_running(session: PodSession, pod_id: str) -> bool:
    """Poll until desiredStatus == RUNNING with a runtime, or cancel/timeout."""
    deadline = time.time() + RUNNING_TIMEOUT_S            # absolute give-up time
    while time.time() < deadline:                        # loop until deadline
        if session.cancel.is_set():                      # Down pressed mid-provision
            return False                                 # bail; stop worker owns state
        pod = runpod.get_pod(pod_id)                     # fetch current pod record
        status = pod.get("desiredStatus") if pod else None   # e.g. "RUNNING"/"CREATED"
        session.update(phase=f"waiting for RUNNING… ({status})")  # progress text
        if pod and status == "RUNNING" and pod.get("runtime"):    # container is actually up
            return True                                  # ready to check vLLM next
        if _sleep_or_cancel(session, POLL_S):            # wait, but wake early on cancel
            return False                                 # cancelled during the sleep
    raise RuntimeError(f"pod {pod_id} did not reach RUNNING within {RUNNING_TIMEOUT_S}s")


def _wait_for_all_ready(session: PodSession, pod_id: str) -> bool:
    """Poll all three services until each returns 200 (or cancel/timeout).

    "Pod ready" = LLM /v1/models AND embedder /health AND reranker /health, per
    the ragline spec. All three services are gated by the same bearer (their
    ports are on RunPod's public proxy), so every probe carries it. Each service
    is dropped from the poll set once healthy, and the phase text reports which
    are still coming up.
    """
    bearer = _read_secret(_secrets.bearer_token)         # gates all three services
    auth = {"Authorization": f"Bearer {bearer}"}         # same header for each probe
    urls = pod_up.service_urls(pod_id)                   # {llm,embedder,reranker: base URL}
    probes = {                                           # service -> (url, headers)
        "llm":      (f"{urls['llm']}/v1/models",   auth),
        "embedder": (f"{urls['embedder']}/health", auth),
        "reranker": (f"{urls['reranker']}/health", auth),
    }
    ready: set[str] = set()                              # services confirmed 200
    deadline = time.time() + READY_TIMEOUT_S             # absolute give-up time
    while time.time() < deadline:                        # loop until deadline
        if session.cancel.is_set():                      # Down pressed while weights load
            return False                                 # bail to the stop worker
        for name, (url, headers) in probes.items():      # probe each not-yet-ready service
            if name in ready:                            # already up — skip
                continue
            try:
                with egress_logger.client(timeout=15.0) as c:   # audited httpx client
                    r = c.get(url, headers=headers)      # probe the endpoint
                if r.status_code == 200:                 # this service is serving
                    ready.add(name)
            except Exception:  # noqa: BLE001            # connection refused while booting
                pass                                     # keep waiting on this one
        if len(ready) == len(probes):                    # all three healthy
            return True                                  # start is complete
        pending = [n for n in probes if n not in ready]  # what's still loading
        session.update(phase=f"waiting for services… "
                             f"(ready: {sorted(ready) or ['none']}; pending: {pending})")
        if _sleep_or_cancel(session, POLL_S):            # wait, waking early on cancel
            return False                                 # cancelled during the sleep
    pending = [n for n in probes if n not in ready]      # timed out — name the stragglers
    raise RuntimeError(f"pod RUNNING but services not all healthy within "
                       f"{READY_TIMEOUT_S}s (pending: {pending})")


# ---------------------------------------------------------------------------
# Pod Down — the clean kill
# ---------------------------------------------------------------------------

def stop(session: PodSession) -> None:
    """Terminate the pod (GPU + pod released; weights persist on the Network
    Volume) and VERIFY it actually left RUNNING.

    Terminate, not stop: a stopped pod is host-pinned and can fail to resume when
    that host has no free GPU. Works whether the pod is still provisioning or fully
    RUNNING. Never trusts pod_state.json alone — resolves the pod id from live
    memory or by name so a pod created before the state file existed is still caught.
    """
    try:
        runpod.api_key = _read_secret(_secrets.runpod_api_key)       # authenticate the SDK

        pod_id = _resolve_pod_id(session)                # find the pod however we can
        if pod_id is None:                               # genuinely nothing exists to terminate
            session.update(state=State.IDLE, phase="no pod found — nothing to terminate",
                           pod_id=None, proxy_url=None)   # settle back to IDLE
            return

        session.update(pod_id=pod_id, phase=f"terminating pod {pod_id}")  # progress text
        runpod.terminate_pod(pod_id)                     # release the GPU (weights persist on the Network Volume)

        if _verify_terminated(session, pod_id):          # confirm it left RUNNING / vanished
            _clear_state_file()                          # drop pod_state.json so a stale id can't resurface
            session.update(state=State.IDLE, phase="terminated — GPU released",
                           pod_id=None, proxy_url=None)   # safe: back to IDLE
        else:                                            # could NOT confirm — do not lie
            session.update(
                state=State.ERROR,                       # loud error state, buttons stay live
                phase="TERMINATE NOT VERIFIED — check RunPod dashboard immediately",
                error="terminate_pod issued but pod did not leave RUNNING within "
                      f"{STOP_VERIFY_TIMEOUT_S}s; it may still be charging.",
            )
    except Exception as e:  # noqa: BLE001                # surface any terminate failure
        # Type only — never interpolate str(e); it may leak request context/secrets.
        session.update(state=State.ERROR, phase="error during terminate",
                       error=f"{type(e).__name__} — check the RunPod dashboard for details")


def _resolve_pod_id(session: PodSession) -> str | None:
    """Find the pod id, most-authoritative source first.

    1. In-memory id captured the instant create_pod returned.
    2. Live lookup by name (catches a pod created just as Down was pressed —
       retried because the create call may still be returning).
    3. pod_state.json on disk (only exists after a fully successful start).
    """
    if session.pod_id:                                   # (1) fastest, most reliable
        return session.pod_id
    for _ in range(5):                                   # (2) retry to beat the create race
        found = pod_up.find_existing()                   # enumerate pods, match by name
        if found is not None:                            # a matching pod now exists
            return found["id"]
        time.sleep(2)                                    # let an in-flight create surface
    if pod_up.STATE_PATH.exists():                       # (3) disk fallback
        try:
            return json.loads(pod_up.STATE_PATH.read_text()).get("pod_id")  # read saved id
        except Exception:  # noqa: BLE001                # corrupt/partial state file
            return None
    return None                                          # nothing found anywhere


def _clear_state_file() -> None:
    """Remove pod_state.json after a confirmed terminate.

    The disk fallback in _resolve_pod_id reads this file; leaving it after the
    pod is destroyed lets a later Down hand terminate_pod an already-dead id,
    which flips the UI to a spurious ERROR. Best-effort — cleanup never fails Down.
    """
    try:
        pod_up.STATE_PATH.unlink(missing_ok=True)        # gone-or-not, end up with no file
    except Exception:  # noqa: BLE001 — cleanup is best-effort, never fatal to a stop
        pass


def _verify_terminated(session: PodSession, pod_id: str) -> bool:
    """Confirm the pod left RUNNING (or is gone) — proof the GPU has been released.

    terminate_pod already succeeded, so a subsequent get_pod may return None OR
    raise (the pod is being deleted and is no longer retrievable). Both mean 'not
    running', so we treat either as terminated rather than reporting a false
    failure on the get_pod call.
    """
    deadline = time.time() + STOP_VERIFY_TIMEOUT_S       # absolute give-up time
    while time.time() < deadline:                        # poll until confirmed or timeout
        try:
            pod = runpod.get_pod(pod_id)                 # fetch current record
        except Exception:  # noqa: BLE001                # pod already deleted / not retrievable
            return True                                  # terminate was accepted => gone
        if pod is None:                                  # pod gone entirely
            return True                                  # => definitely not billing GPU
        status = pod.get("desiredStatus")                # e.g. "EXITED"/"STOPPED"
        session.update(phase=f"verifying termination… ({status})")  # progress text
        if status and status != "RUNNING":               # left RUNNING => GPU released
            return True
        time.sleep(3)                                    # brief pause before re-checking
    return False                                         # could not confirm within the window
