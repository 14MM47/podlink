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
import os            # read the auto-terminate window from the environment
import sys           # to mutate sys.path for the vendored imports
import time          # wall-clock deadlines and inter-poll sleeps
from pathlib import Path  # build the pod_control directory path

# Make the vendored pod_control modules importable with their original sibling
# imports (`import _secrets`, `import egress_logger`) intact.
POD_CONTROL_DIR = Path(__file__).resolve().parents[1] / "pod_control"  # …/podlink/pod_control
if str(POD_CONTROL_DIR) not in sys.path:              # avoid duplicate entries on reload
    sys.path.insert(0, str(POD_CONTROL_DIR))          # front of path so our copy wins

import runpod            # noqa: E402  RunPod SDK (imported after sys.path tweak)
from runpod.error import QueryError  # noqa: E402  raised by create_pod on the capacity lottery
import _secrets          # noqa: E402  vendored secret reader (0600/ownership checked)
import egress_logger     # noqa: E402  vendored audited httpx client
import pod_up            # noqa: E402  vendored: constants + find_existing/create_pod/…

from .session import PodSession, State, _default_services  # state machine + fresh tiles

# How long to wait, in seconds, for each phase before declaring failure.
RUNNING_TIMEOUT_S = 900   # RunPod allocation + container boot


def _env_seconds(name: str, default: int) -> int:
    """Read a non-negative seconds value from the environment (bad/negative -> default)."""
    try:
        return max(0, int(os.environ.get(name, str(default)) or default))
    except ValueError:
        return default


# Readiness is a soft wait: a pod that is RUNNING is billing whether or not vLLM is
# listening yet, so giving up early only hides the pod from the console. After
# READY_WARN_S the feed says so every READY_WARN_S and the wait continues; only the
# hard READY_TIMEOUT_S (0 = never) raises. A 122B volume-less boot takes ~16 min.
READY_WARN_S = _env_seconds("PODLINK_READY_WARN_S", 900)
READY_TIMEOUT_S = _env_seconds("PODLINK_READY_TIMEOUT_S", 3600)
STOP_VERIFY_TIMEOUT_S = 180  # max wait to confirm the pod left RUNNING
POLL_S = 10               # inter-poll sleep (interruptible by cancel)


def _auto_terminate_minutes() -> int:
    """Idle safety window in minutes from PODLINK_AUTO_TERMINATE_MIN (0 = off).

    Read live (not import-time) so the value can't get baked into a stale module
    and so tests can set it per-case. Non-numeric/negative -> disabled.
    """
    try:
        return max(0, int(os.environ.get("PODLINK_AUTO_TERMINATE_MIN", "0") or "0"))
    except ValueError:
        return 0


def _arm_billing(session: PodSession) -> None:
    """Start the cost clock and (if configured) the auto-terminate deadline.

    Called the instant a pod id is captured — RunPod bills from pod creation, so
    the meter and the idle timer both start there.
    """
    now = time.time()
    fields = {"billing_started_at": now}
    minutes = _auto_terminate_minutes()
    if minutes > 0:                                      # arm the idle safety timer
        fields["auto_terminate_at"] = now + minutes * 60
    session.update(**fields)


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
            _arm_billing(session)                         # start the cost meter + idle timer
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
        elif existing is not None and pod_up.LIFECYCLE == "stop":
            # stop lifecycle: the pod was stopped, not terminated — resume it in place
            # (no image pull). Retried, because the host may have no free GPU right now.
            pod_id = existing["id"]
            session.update(pod_id=pod_id, phase="resuming stopped pod")
            resumed = _resume_with_retries(session, pod_id)
            if resumed is None:                           # cancel arrived mid-retry
                return
            if not resumed:                               # host never freed a GPU: recreate
                session.add_event("resume exhausted — terminating the stuck pod and creating "
                                  "a fresh one (weights persist on the Network Volume)", "system")
                session.update(phase="terminating stuck pod before recreate")
                runpod.terminate_pod(pod_id)
                if not _wait_until_gone(session, pod_id):
                    return
                session.update(pod_id=None)
                pod = _create_with_fallback(session)
                if pod is None:
                    return
                pod_id = pod["id"]
                session.update(pod_id=pod_id, phase="pod created")
        else:                                             # none, or a dead/reaping leftover
            pod = _create_with_fallback(session)          # resolve the configured GPU + create
            if pod is None:                               # cancel arrived during create
                return                                    # let the stop worker take over
            pod_id = pod["id"]                            # id of the freshly created pod
            session.update(pod_id=pod_id, phase="pod created")  # capture id immediately
        _arm_billing(session)                             # start the cost meter + idle timer

        if not _wait_for_running(session, pod_id):        # poll until RUNNING (or cancel)
            return                                        # cancelled mid-wait

        llm_url = pod_up.derive_proxy_url(pod_id, pod_up.SERVICE_PORTS["llm"])  # primary URL
        session.update(proxy_url=llm_url,
                       phase=f"waiting for all {len(pod_up.SERVICES)} services to become healthy")
        if not _wait_for_all_ready(session, pod_id):       # every configured service (or cancel)
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
        session.add_event(f"error during start: {type(e).__name__}", "system")
        hint = (" — the pod may still be RUNNING and billing: the health watch keeps probing "
                "and recovers automatically; POD UP re-adopts it, POD DOWN stops it"
                if session.pod_id else "")
        session.update(state=State.ERROR, phase="error during start",
                       error=f"{type(e).__name__} — check the RunPod dashboard for details"
                             f"{hint}")


def _create_retries() -> int:
    """How many create attempts before giving up. Env-tunable; default 40 (~10 min
    at the 15s delay). Because POD DOWN can cancel mid-loop, a generous default is
    safe — it rides out a transient EU-RO capacity shortage instead of failing at 15."""
    try:
        return max(1, int(os.environ.get("PODLINK_CREATE_RETRIES", "40")))
    except ValueError:
        return 40


def _create_retry_delay() -> int:
    """Seconds between create attempts (env-tunable, default 15)."""
    try:
        return max(5, int(os.environ.get("PODLINK_CREATE_RETRY_DELAY", "15")))
    except ValueError:
        return 15


def _create_with_fallback(session: PodSession) -> dict | None:
    """Resolve the RTX Pro 6000 id and create the pod (no GPU fallback).

    The bundled image + models are sized for the 96 GB card, so a smaller GPU
    would OOM rather than help — we target one card. The host-selection lottery is
    retried HERE (not in the vendored create_pod) so every attempt is reported to
    the UI event feed and each inter-attempt wait is cancel-aware — POD DOWN
    interrupts the wait immediately.
    """
    bearer = _read_secret(_secrets.bearer_token)         # -> pod env VLLM_API_KEY
    hf = _read_secret(_secrets.hf_token)                 # -> weight-pull token (all 3 services)
    if session.cancel.is_set():                          # Down pressed before we resolve
        return None
    session.update(phase="resolving RTX Pro 6000 GPU id")  # live catalog lookup
    gpu_id = pod_up.resolve_gpu_id()                     # exact RunPod gpu_type_id
    if session.cancel.is_set():
        return None
    session.update(phase="ensuring pod template")        # registry-cred template (cached)
    template_id = pod_up.ensure_template()
    retries, delay = _create_retries(), _create_retry_delay()
    for attempt in range(1, retries + 1):
        if session.cancel.is_set():                      # Down pressed between attempts
            return None
        session.update(phase=f"creating pod on {gpu_id} — attempt {attempt}/{retries}")
        try:
            return pod_up.create_pod_once(gpu_id, bearer, hf, template_id)  # one attempt
        except QueryError as e:
            if not pod_up.is_retryable_create_error(e):  # real error (bad spec/auth) — surface it
                raise
            session.add_event(                           # visible in the UI feed
                f"no host with capacity yet — attempt {attempt}/{retries}; retrying in {delay}s")
            if attempt < retries and _sleep_or_cancel(session, delay):  # cancel-aware wait
                return None
    raise RuntimeError(
        f"no Secure host in the volume's region accepted the pod after {retries} "
        f"attempts (~{retries * delay // 60} min). RTX PRO 6000 capacity is transient — "
        f"press POD UP to keep trying, or try again later.")


def _resume_retries() -> int:
    try:
        return max(1, int(os.environ.get("PODLINK_RESUME_RETRIES", str(pod_up.RESUME_RETRIES))))
    except ValueError:
        return 40


def _resume_retry_delay() -> int:
    try:
        return max(5, int(os.environ.get("PODLINK_RESUME_RETRY_DELAY", str(pod_up.RESUME_RETRY_DELAY))))
    except ValueError:
        return 15


def _resume_with_retries(session: PodSession, pod_id: str) -> bool | None:
    """stop lifecycle: resume a stopped pod on its original host, retrying while that
    host has no free GPU. True = RUNNING; False = every attempt failed (caller falls back
    to terminate + create); None = cancelled by POD DOWN. Each attempt and wait is
    reported to the feed and cancel-aware, like the create lottery."""
    retries, delay = _resume_retries(), _resume_retry_delay()
    for attempt in range(1, retries + 1):
        if session.cancel.is_set():
            return None
        session.update(phase=f"resuming pod {pod_id} — attempt {attempt}/{retries}")
        try:
            runpod.resume_pod(pod_id, gpu_count=1)
            pod = runpod.get_pod(pod_id) or {}
            if pod.get("desiredStatus") == "RUNNING":
                session.add_event(f"resumed on the original host (attempt {attempt}) — no image pull",
                                  "lifecycle")
                return True
        except Exception as e:  # noqa: BLE001 — "not enough free GPUs on the host" and kin
            session.add_event(f"host has no free GPU yet ({type(e).__name__}) — attempt "
                              f"{attempt}/{retries}; retrying in {delay}s")
        if attempt < retries and _sleep_or_cancel(session, delay):
            return None
    return False


def _wait_until_gone(session: PodSession, pod_id: str) -> bool:
    """After a terminate, wait until RunPod no longer reports the pod (or it left RUNNING),
    so a fresh create with the same name does not collide. Cancel-aware."""
    deadline = time.time() + STOP_VERIFY_TIMEOUT_S
    while time.time() < deadline:
        if session.cancel.is_set():
            return False
        try:
            pod = runpod.get_pod(pod_id)
        except Exception:  # noqa: BLE001
            return True
        if not pod or pod.get("desiredStatus") != "RUNNING":
            return True
        if _sleep_or_cancel(session, 3):
            return False
    raise RuntimeError(f"pod {pod_id} did not leave RUNNING after terminate")


def _wait_for_running(session: PodSession, pod_id: str) -> bool:
    """Poll until desiredStatus == RUNNING with a runtime, or cancel/timeout."""
    deadline = time.time() + RUNNING_TIMEOUT_S            # absolute give-up time
    while time.time() < deadline:                        # loop until deadline
        if session.cancel.is_set():                      # Down pressed mid-provision
            return False                                 # bail; stop worker owns state
        pod = runpod.get_pod(pod_id)                     # fetch current pod record
        status = pod.get("desiredStatus") if pod else None   # e.g. "RUNNING"/"CREATED"
        if pod and pod.get("costPerHr") is not None:     # capture the hourly rate for the meter
            try:
                session.update(cost_per_hr=float(pod["costPerHr"]))
            except (TypeError, ValueError):
                pass
        session.update(phase=f"waiting for RUNNING… ({status})")  # progress text
        if pod and status == "RUNNING" and pod.get("runtime"):    # container is actually up
            return True                                  # ready to check vLLM next
        if _sleep_or_cancel(session, POLL_S):            # wait, but wake early on cancel
            return False                                 # cancelled during the sleep
    raise RuntimeError(f"pod {pod_id} did not reach RUNNING within {RUNNING_TIMEOUT_S}s")


def _service_probes(pod_id: str, bearer: str) -> dict:
    """Map each configured service to its (health-url, auth-headers). Every service is
    gated by the same bearer (their ports are on RunPod's public proxy). The set and
    the health paths come from PODLINK_SERVICES (pod_up.SERVICES)."""
    auth = {"Authorization": f"Bearer {bearer}"}
    urls = pod_up.service_urls(pod_id)                   # {name: base URL}
    return {name: (f"{urls[name]}{path}", auth)
            for name, (_port, path) in pod_up.SERVICES.items()}


def _probe_service(url: str, headers: dict) -> bool:
    """One audited GET; True iff it returned 200 (connection refused => False)."""
    try:
        with egress_logger.client(timeout=15.0) as c:    # audited httpx client
            return c.get(url, headers=headers).status_code == 200
    except Exception:  # noqa: BLE001                    # refused/timeout while booting
        return False


def _apply_health(session: PodSession, statuses: dict) -> None:
    """Write per-service statuses onto the session, logging each transition to the
    event feed (so the tiles and the streamed feed stay in sync)."""
    old = session.services                               # last-known statuses
    for name, st in statuses.items():
        if old.get(name) != st:                          # a service changed state
            session.add_event(f"{name}: {st}", "health")  # health-panel feed entry
    session.update(services=dict(statuses))              # refresh the tiles


def probe_health_once(session: PodSession, pod_id: str) -> None:
    """Probe every configured service once and update the health tiles — used by the
    background poller while RUNNING (healthy | down)."""
    bearer = _read_secret(_secrets.bearer_token)
    probes = _service_probes(pod_id, bearer)
    statuses = {name: ("healthy" if _probe_service(url, headers) else "down")
                for name, (url, headers) in probes.items()}
    _apply_health(session, statuses)


def _timed_post(url: str, headers: dict, body: dict) -> tuple:
    """One audited POST. Returns (ok, latency_ms, status_or_None, json_or_None)."""
    t0 = time.time()
    try:
        with egress_logger.client(timeout=60.0) as c:   # audited; real inference can be slow
            r = c.post(url, headers=headers, json=body)
        ms = int((time.time() - t0) * 1000)
        try:
            data = r.json()
        except Exception:  # noqa: BLE001                # non-JSON body
            data = None
        return (r.status_code == 200, ms, r.status_code, data)
    except Exception:  # noqa: BLE001                    # connection error / timeout
        return (False, int((time.time() - t0) * 1000), None, None)


def test_stack(session: PodSession) -> None:
    """Fire a REAL completion + embedding + rerank at the stock services, recording
    pass/fail + latency (and the embedding dimension) into session.test_result.

    Uses the same endpoints a RAG client will: vLLM OpenAI-compat
    /v1/chat/completions, TEI OpenAI-compat /v1/embeddings, and TEI native /rerank.
    Only the services present in PODLINK_SERVICES are exercised (an extended image's
    extra services are covered by the health probes, not this test).
    Runs in a background thread (launched by /pod/test)."""
    try:
        pod_id = session.pod_id
        if not pod_id:
            session.update(test_running=False, test_result={"error": "no pod running"})
            return
        bearer = _read_secret(_secrets.bearer_token)     # gates all three services
        auth = {"Authorization": f"Bearer {bearer}"}
        urls = pod_up.service_urls(pod_id)
        session.update(test_running=True)
        session.add_event("stack test started", "system")
        services: dict = {}

        dim = None
        # 1) LLM — OpenAI chat completion (model must equal vLLM --served-model-name).
        if "llm" in urls:
            ok, ms, status, data = _timed_post(
                f"{urls['llm']}/v1/chat/completions", auth,
                {"model": pod_up.LLM_SERVED_NAME,
                 "messages": [{"role": "user", "content": "ping"}],
                 "max_tokens": 1, "temperature": 0})
            services["llm"] = {"ok": ok, "latency_ms": ms,
                               "detail": "completion ok" if ok else f"HTTP {status}"}
            session.add_event(f"stack test — llm {'ok' if ok else 'FAIL'} {ms}ms", "health")

        # 2) Embedder — OpenAI embeddings; the vector length is the served dimension.
        if "embedder" in urls:
            ok, ms, status, data = _timed_post(
                f"{urls['embedder']}/v1/embeddings", auth,
                {"model": pod_up.EMBED_MODEL_ID, "input": "hello world"})
            try:
                dim = len(data["data"][0]["embedding"])
            except Exception:  # noqa: BLE001                # unexpected shape
                pass
            services["embedder"] = {"ok": bool(ok and dim), "latency_ms": ms,
                                    "detail": f"{dim}-dim" if dim else f"HTTP {status}"}
            session.add_event(
                f"stack test — embedder {'ok' if ok else 'FAIL'} {ms}ms"
                f"{f' ({dim}-dim)' if dim else ''}", "health")

        # 3) Reranker — TEI native /rerank (query + candidate texts).
        if "reranker" in urls:
            ok, ms, status, data = _timed_post(
                f"{urls['reranker']}/rerank", auth,
                {"query": "what does podlink do",
                 "texts": ["podlink controls a RunPod GPU pod", "an unrelated sentence"]})
            top = None
            try:
                top = round(max(x["score"] for x in data), 3)  # TEI returns [{index, score}, ...]
            except Exception:  # noqa: BLE001
                pass
            services["reranker"] = {"ok": ok, "latency_ms": ms,
                                    "detail": f"top score {top}" if top is not None else (f"HTTP {status}" if not ok else "ok")}
            session.add_event(f"stack test — reranker {'ok' if ok else 'FAIL'} {ms}ms", "health")

        if not services:
            session.update(test_running=False,
                           test_result={"error": "no llm/embedder/reranker service in PODLINK_SERVICES"})
            return
        all_ok = all(s["ok"] for s in services.values())
        session.update(test_running=False,
                       test_result={"services": services, "embedding_dim": dim, "all_ok": all_ok})
        session.add_event(f"stack test {'PASSED' if all_ok else 'had failures'}", "system")
    except Exception as e:  # noqa: BLE001               # never let the test thread die silently
        session.add_event(f"stack test error: {type(e).__name__}", "system")
        session.update(test_running=False, test_result={"error": type(e).__name__})


def stuck_service_hint(ready: set[str], pending: list[str]) -> str | None:
    """When the LLM answers but another service never listens, that service has almost
    certainly failed to start (CUDA allocation while vLLM loaded, or a weight download
    error) and supervisor may have given up on it. Say so, and where to look."""
    if "llm" not in ready or not pending or "llm" in pending:
        return None
    names = " and ".join(pending)
    if all(p in ("embedder", "reranker") for p in pending):
        return (f"{names} still not listening while the LLM is up: this is usually a failed "
                "start (VRAM taken by vLLM, or a weight download error), not a slow load — "
                "check the container log in the RunPod dashboard for text-embeddings-router; "
                "POD DOWN then POD UP restarts the stack (pod images built after 2026-09-16 "
                "start the TEI services before vLLM)")
    return (f"{names} still not listening while the LLM is up: this is usually a failed "
            "start (VRAM already taken, a missing model, or a wrapper error), not a slow "
            "load — check the container log in the RunPod dashboard, or from the pod's web "
            "terminal run `supervisorctl -c /etc/podlink/supervisord.conf status` and "
            f"`tail -200 <service> stderr`; POD DOWN then POD UP restarts the stack")


def _wait_for_all_ready(session: PodSession, pod_id: str) -> bool:
    """Poll every configured service until each returns 200 (or cancel/timeout).

    "Pod ready" = every PODLINK_SERVICES health path answers 200 (stock: LLM
    /v1/models AND embedder /health AND reranker /health). Each
    service is dropped from the poll set once healthy; the per-service tiles and
    the phase text report which are still coming up.
    """
    bearer = _read_secret(_secrets.bearer_token)         # gates all three services
    probes = _service_probes(pod_id, bearer)             # service -> (url, headers)
    ready: set[str] = set()                              # services confirmed 200
    started = time.time()
    next_warn = started + READY_WARN_S if READY_WARN_S else None   # soft: warn, keep waiting
    deadline = started + READY_TIMEOUT_S if READY_TIMEOUT_S else None  # hard: 0 = never
    while deadline is None or time.time() < deadline:   # loop until the hard deadline
        if session.cancel.is_set():                      # Down pressed while weights load
            return False                                 # bail to the stop worker
        for name, (url, headers) in probes.items():      # probe each not-yet-ready service
            if name in ready:                            # already up — skip
                continue
            if _probe_service(url, headers):             # this service is serving
                ready.add(name)
        # Reflect per-service health onto the tiles (healthy vs still pending).
        _apply_health(session, {n: ("healthy" if n in ready else "pending") for n in probes})
        if len(ready) == len(probes):                    # every service healthy
            session.update(phase="all services healthy")
            return True                                  # start is complete
        pending = [n for n in probes if n not in ready]  # what's still loading
        minutes = int((time.time() - started) // 60)
        session.update(phase=f"waiting for services… {minutes} min "
                             f"(ready: {sorted(ready) or ['none']}; pending: {pending})")
        if next_warn is not None and time.time() >= next_warn:   # soft deadline passed
            session.add_event(f"services still pending after {minutes} min: {pending} — the pod "
                              "is RUNNING and billing; still waiting (POD DOWN to stop)",
                              "system")
            hint = stuck_service_hint(ready, pending)        # a likely cause, if one stands out
            if hint:
                session.add_event(hint, "system")
            next_warn += READY_WARN_S                     # repeat the reminder
        if not _pod_still_running(pod_id):               # the pod itself went away
            raise RuntimeError(f"pod left RUNNING while services were pending: {pending}")
        if _sleep_or_cancel(session, POLL_S):            # wait, waking early on cancel
            return False                                 # cancelled during the sleep
    pending = [n for n in probes if n not in ready]      # hard timeout — name the stragglers
    raise RuntimeError(f"pod RUNNING but services not all healthy within "
                       f"{READY_TIMEOUT_S}s (pending: {pending})")


def _pod_still_running(pod_id: str) -> bool:
    """True unless RunPod positively reports the pod is no longer RUNNING. A failed
    lookup counts as still running: never abandon a pod on a transient API error."""
    try:
        pod = runpod.get_pod(pod_id)
    except Exception:  # noqa: BLE001
        return True
    if not pod:
        return False
    return pod.get("desiredStatus") in (None, "RUNNING")


def recover_if_healthy(session: PodSession, pod_id: str) -> bool:
    """Health-watch hook for the ERROR state: the pod is still ours and still billing, so
    keep probing; the moment all three services answer, flip to RUNNING.

    Returns True when the session recovered. A stop that has begun wins.
    """
    probe_health_once(session, pod_id)                   # tiles reflect reality even in ERROR
    if session.state != State.ERROR:                     # someone else moved us on
        return False
    if any(st != "healthy" for st in session.services.values()):
        return False
    llm_url = pod_up.derive_proxy_url(pod_id, pod_up.SERVICE_PORTS["llm"])
    if not session.commit_running():                     # STOPPING/cancel arrived first
        return False
    session.update(error=None, proxy_url=llm_url,
                   phase="running — recovered: all services healthy")
    session.add_event("recovered — all services healthy; state is RUNNING", "system")
    try:                                                 # best effort, matches start()
        pod = runpod.get_pod(pod_id) or {}
        gpu_type = (pod.get("machine", {}).get("gpuTypeId") or pod.get("gpuTypeId") or "unknown")
        pod_up.write_state({**pod, "id": pod_id}, gpu_type)
    except Exception:  # noqa: BLE001
        pass
    return True


def adopt_running_on_startup(session: PodSession) -> None:
    """Console start: if a RUNNING pod with our name already exists (a previous console
    lost it, or was restarted), adopt it so the panel shows the truth instead of IDLE.

    Nothing is created here; if there is no live pod this is a no-op. Runs the normal
    start() path, whose Auto branch adopts a RUNNING pod by name.
    """
    if os.environ.get("PODLINK_ADOPT_ON_START", "1") == "0":
        return
    try:
        runpod.api_key = _read_secret(_secrets.runpod_api_key)
        existing = pod_up.find_existing()
    except Exception as e:  # noqa: BLE001 — never block console start on a lookup failure
        session.add_event(f"startup pod lookup failed: {type(e).__name__}", "system")
        return
    if existing is None or existing.get("desiredStatus") != "RUNNING":
        return
    if not session.try_begin_start(None):                # only from IDLE/ERROR
        return
    session.add_event(f"adopting running pod {existing['id']} found at console start",
                      "system")
    start(session)


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
                           pod_id=None, proxy_url=None,   # settle back to IDLE
                           billing_started_at=None, cost_per_hr=None, auto_terminate_at=None,
                           services=_default_services())
            return

        if pod_up.LIFECYCLE == "stop":
            # stop lifecycle: keep the container on its host (small disk charge, no GPU
            # charge); POD UP resumes it without an image pull. pod_state.json is kept.
            session.update(pod_id=pod_id, phase=f"stopping pod {pod_id}")
            runpod.stop_pod(pod_id)
            verified, done_phase = _verify_terminated(session, pod_id), "stopped — GPU released, container kept on host"
        else:
            session.update(pod_id=pod_id, phase=f"terminating pod {pod_id}")  # progress text
            runpod.terminate_pod(pod_id)                 # release the GPU (weights persist on the Network Volume)
            verified, done_phase = _verify_terminated(session, pod_id), "terminated — GPU released"
            if verified:
                _clear_state_file()                      # drop pod_state.json so a stale id can't resurface

        if verified:                                     # confirmed it left RUNNING / vanished
            session.update(state=State.IDLE, phase=done_phase,
                           pod_id=None, proxy_url=None,   # safe: back to IDLE
                           billing_started_at=None, cost_per_hr=None, auto_terminate_at=None,  # stop the meter
                           services=_default_services(),  # clear the health tiles
                           test_result=None, test_running=False)  # clear the stack-test result

        else:                                            # could NOT confirm — do not lie
            session.update(
                state=State.ERROR,                       # loud error state, buttons stay live
                phase="TERMINATE NOT VERIFIED — check RunPod dashboard immediately",
                error="terminate_pod issued but pod did not leave RUNNING within "
                      f"{STOP_VERIFY_TIMEOUT_S}s; it may still be charging.",
            )
    except Exception as e:  # noqa: BLE001                # surface any terminate failure
        # Type only — never interpolate str(e); it may leak request context/secrets.
        session.add_event(f"error during terminate: {type(e).__name__}", "system")
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
