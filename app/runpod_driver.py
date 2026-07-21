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
    """Read a secret and register it for verbatim redaction in the egress log."""
    value = getter()                        # vendored getter; may sys.exit if missing
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

        # ---- Auto: create or resume the podlink vLLM pod ----
        existing = pod_up.find_existing()                 # is a pod already named podlink?
        if existing is not None:                          # yes — adopt it instead of duplicating
            pod_id = existing["id"]                       # its RunPod id
            # Capture the id before anything can block — closes the billing race.
            session.update(pod_id=pod_id, phase="adopting existing pod")
            if existing.get("desiredStatus") != "RUNNING":  # it's stopped -> resume it
                session.update(phase="resuming stopped pod")
                runpod.resume_pod(pod_id, gpu_count=1)    # SDK requires gpu_count
        else:                                             # no pod yet — create one
            pod = _create_with_fallback(session)          # try 5090, then A6000
            if pod is None:                               # cancel arrived during create
                return                                    # let the stop worker take over
            pod_id = pod["id"]                            # id of the freshly created pod
            session.update(pod_id=pod_id, phase="pod created")  # capture id immediately

        if not _wait_for_running(session, pod_id):        # poll until RUNNING (or cancel)
            return                                        # cancelled mid-wait

        proxy_url = pod_up.derive_proxy_url(pod_id)        # https://<id>-8000.proxy.runpod.net
        session.update(proxy_url=proxy_url, phase="waiting for vLLM to serve model")
        if not _wait_for_ready(session, proxy_url):        # poll /v1/models until 200 (or cancel)
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
    """Try each GPU in preference order automatically (no interactive prompt)."""
    bearer = _read_secret(_secrets.bearer_token)                     # vLLM API key (goes into pod env)
    hf = _read_secret(_secrets.hf_token)                             # Hugging Face token for weight pull
    last_err: Exception | None = None                    # remember the final failure
    for gpu_id in pod_up.GPU_PREFERENCES:                # e.g. ["RTX 5090", "RTX A6000"]
        if session.cancel.is_set():                      # user hit Down before we created
            return None                                  # abort the create loop
        try:
            session.update(phase=f"creating pod on {gpu_id}")  # progress text
            return pod_up.create_pod(gpu_id, bearer, hf)  # blocking SDK call; returns pod dict
        except Exception as e:  # noqa: BLE001            # this GPU type unavailable etc.
            last_err = e                                 # keep the error…
            session.update(phase=f"{gpu_id} unavailable ({type(e).__name__}); trying next")
    raise RuntimeError(f"All GPU options exhausted. Last error: {last_err}")  # none worked


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


def _wait_for_ready(session: PodSession, proxy_url: str) -> bool:
    """Poll {proxy}/v1/models until 200 (vLLM finished loading), via the audit client."""
    bearer = _read_secret(_secrets.bearer_token)                     # bearer for the Authorization header
    url = f"{proxy_url}/v1/models"                        # OpenAI-compatible models endpoint
    headers = {"Authorization": f"Bearer {bearer}"}      # authenticate the probe
    deadline = time.time() + READY_TIMEOUT_S             # absolute give-up time
    while time.time() < deadline:                        # loop until deadline
        if session.cancel.is_set():                      # Down pressed while weights load
            return False                                 # bail to the stop worker
        try:
            with egress_logger.client(timeout=15.0) as c:    # audited httpx client (logs egress)
                r = c.get(url, headers=headers)          # probe the endpoint
            if r.status_code == 200:                     # vLLM is serving the model
                return True                              # start is complete
            session.update(phase=f"waiting for vLLM… (HTTP {r.status_code})")  # e.g. 401/503
        except Exception:  # noqa: BLE001                 # connection refused while booting
            session.update(phase="waiting for vLLM… (connecting)")  # progress text
        if _sleep_or_cancel(session, POLL_S):            # wait, waking early on cancel
            return False                                 # cancelled during the sleep
    raise RuntimeError("pod RUNNING but /v1/models never returned 200")  # boot never finished


# ---------------------------------------------------------------------------
# Pod Down — the clean kill
# ---------------------------------------------------------------------------

def stop(session: PodSession) -> None:
    """Stop the pod (GPU billing off, volume kept) and VERIFY it actually stopped.

    Works whether the pod is still provisioning or fully RUNNING. Never trusts
    pod_state.json alone — resolves the pod id from live memory or by name so a
    pod created before the state file existed is still caught.
    """
    try:
        runpod.api_key = _read_secret(_secrets.runpod_api_key)       # authenticate the SDK

        pod_id = _resolve_pod_id(session)                # find the pod however we can
        if pod_id is None:                               # genuinely nothing exists to stop
            session.update(state=State.IDLE, phase="no pod found — nothing to stop",
                           pod_id=None, proxy_url=None)   # settle back to IDLE
            return

        session.update(pod_id=pod_id, phase=f"stopping pod {pod_id}")  # progress text
        runpod.stop_pod(pod_id)                          # end GPU billing (volume preserved)

        if _verify_stopped(session, pod_id):             # confirm it actually left RUNNING
            session.update(state=State.IDLE, phase="stopped — GPU billing ended",
                           pod_id=None, proxy_url=None)   # safe: back to IDLE
        else:                                            # could NOT confirm — do not lie
            session.update(
                state=State.ERROR,                       # loud error state, buttons stay live
                phase="STOP NOT VERIFIED — check RunPod dashboard immediately",
                error="stop_pod issued but pod did not leave RUNNING within "
                      f"{STOP_VERIFY_TIMEOUT_S}s; it may still be charging.",
            )
    except Exception as e:  # noqa: BLE001                # surface any stop failure
        # Type only — never interpolate str(e); it may leak request context/secrets.
        session.update(state=State.ERROR, phase="error during stop",
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


def _verify_stopped(session: PodSession, pod_id: str) -> bool:
    """Confirm the pod left RUNNING (or vanished) — proof GPU billing has ceased."""
    deadline = time.time() + STOP_VERIFY_TIMEOUT_S       # absolute give-up time
    while time.time() < deadline:                        # poll until confirmed or timeout
        pod = runpod.get_pod(pod_id)                     # fetch current record
        if pod is None:                                  # pod gone entirely
            return True                                  # => definitely not billing GPU
        status = pod.get("desiredStatus")                # e.g. "EXITED"/"STOPPED"
        session.update(phase=f"verifying stop… ({status})")  # progress text
        if status and status != "RUNNING":               # left RUNNING => GPU billing ended
            return True
        time.sleep(3)                                    # brief pause before re-checking
    return False                                         # could not confirm within the window
