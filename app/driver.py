"""Provider-neutral orchestration of the pod lifecycle.

The original pod_up.py / pod_down.py are CLI tools: they prompt with
rich.Confirm and quit via sys.exit — both fatal to a webapp. So the control loop
lives here instead, with:

  * cooperative cancellation on every poll tick,
  * immediate instance-id capture the instant one is created,
  * a create-retry loop that reports every attempt to the UI, and
  * a verified stop that confirms GPU billing has actually ended.

Nothing in this module knows which cloud it is driving. Everything cloud-shaped
— creating, finding, describing, reaching and destroying an instance — goes
through the Provider contract in app/providers/base.py, so a second cloud is a
new provider module rather than a fork of this loop.

The two entry points, `start(session)` and `stop(session)`, each run in their
own background thread.
"""
from __future__ import annotations  # allow `dict | None` etc. annotations

import os            # read the auto-terminate window from the environment
import time          # wall-clock deadlines and inter-poll sleeps

from .providers import active as active_provider   # the configured cloud
from .session import PodSession, State, SERVICES   # our state machine types + service names
from .vendored import _secrets, egress_logger, read_secret  # local secrets + audited client

# How long to wait, in seconds, for each phase before declaring failure.
RUNNING_TIMEOUT_S = 900   # instance allocation + container boot
READY_TIMEOUT_S = 900     # vLLM model-weight load until /v1/models == 200
STOP_VERIFY_TIMEOUT_S = 180  # max wait to confirm the instance left RUNNING
STOP_VERIFY_POLL_S = 3    # pause between termination re-checks
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

    Called the instant an instance id is captured — billing runs from creation,
    so the meter and the idle timer both start there.
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


def _release_access(session: PodSession, provider) -> None:
    """Close the provider's local access path, swallowing any failure.

    Teardown must never be what fails a stop: the pod is already gone, and a
    stuck tunnel process is a smaller problem than a session wedged in STOPPING
    with the meter still running. The contract says release_access never raises;
    this is the belt to that braces.
    """
    try:
        provider.release_access()
    except Exception:  # noqa: BLE001 — a leaked local process is not worth failing on
        session.add_event("could not close the provider access path", "system")


def list_pods() -> list[dict]:
    """The account's instances for the selector (whitelisted fields only)."""
    provider = active_provider()
    provider.authenticate()                              # authenticate the SDK
    return provider.list_instances()


def running_instance_id() -> str | None:
    """The id of a RUNNING podlink instance on the active cloud, or None.

    A live-account check, not a session one: it asks the cloud, so it catches
    an instance this server process never knew about (created before a
    restart, or by the CLI). Used to refuse a cloud switch that would leave a
    billing instance behind on a provider the UI has stopped watching. Raises
    if the cloud cannot be asked — the caller decides what that means.
    """
    provider = active_provider()
    provider.authenticate()
    existing = provider.find_existing()
    if existing is not None and provider.is_running(existing):
        return provider.instance_id(existing)
    return None


def persistence_configured() -> bool:
    """True when storage survives POD DOWN, so terminate is non-destructive.

    When False, POD DOWN DESTROYS the ~36 GB of downloaded weights, so the
    server refuses without an explicit confirm and the web UI warns first.
    """
    return active_provider().persistence_configured()


# ---------------------------------------------------------------------------
# Pod Up
# ---------------------------------------------------------------------------

def start(session: PodSession) -> None:
    """Provision, resume, or adopt the instance, then wait until it's up.

    Assumes the session is already in STARTING (set atomically by the request
    handler). If a specific instance was selected (session.target_pod_id) it is
    adopted directly; otherwise Auto creates/adopts the podlink instance. On
    cancel it returns quietly and lets the stop worker own the state; on failure
    it flips the session to ERROR.
    """
    try:
        provider = active_provider()
        provider.authenticate()                           # load cloud credentials
        target = session.target_pod_id                    # a specific pod to adopt, or None

        if target:                                        # ---- adopt the selected pod ----
            instance = provider.get_instance(target)      # fetch the chosen instance
            if not instance:                              # it was deleted since listing
                raise RuntimeError("selected pod no longer exists")
            pod_id = target                               # id already recorded by the session
            session.update(phase="adopting selected pod")
            _arm_billing(session)                         # start the cost meter + idle timer
            if not provider.is_running(instance):         # stopped -> resume it
                session.update(phase="resuming selected pod")
                provider.resume(instance)
            if not _wait_for_running(session, pod_id):    # poll until RUNNING (or cancel)
                return
            provider.ensure_access(pod_id)                # open the local path, if any
            # An arbitrary instance may not serve /v1/models, so RUNNING is 'up'
            # here; skip the readiness probe and write_state (both assume the
            # podlink stack).
            session.commit_running()                      # -> RUNNING (unless a stop won)
            return

        # ---- Auto: adopt a RUNNING podlink instance, else create a fresh one ----
        # The lifecycle is terminate/recreate (see stop()), so we never resume. A
        # non-RUNNING instance with podlink's name here is a leftover — crashed, or
        # still being reaped after a terminate. Resuming it would error (a
        # terminating pod can't resume), and a quick DOWN->UP would then fail
        # instead of just making a new one. So we adopt ONLY a running instance
        # and otherwise create fresh; the cloud reaps the dead one.
        existing = provider.find_existing()               # is one already named podlink?
        if existing is not None and provider.is_running(existing):
            pod_id = provider.instance_id(existing)       # adopt the live instance
            # Capture the id before anything can block — closes the billing race.
            session.update(pod_id=pod_id, phase="adopting running pod")
        else:                                             # none, or a dead/reaping leftover
            instance = _create_with_retries(session)      # prepare + create (cancel-aware)
            if instance is None:                          # cancel arrived during create
                return                                    # let the stop worker take over
            pod_id = provider.instance_id(instance)       # id of the freshly created instance
            session.update(pod_id=pod_id, phase="pod created")  # capture id immediately
        _arm_billing(session)                             # start the cost meter + idle timer

        if not _wait_for_running(session, pod_id):        # poll until RUNNING (or cancel)
            return                                        # cancelled mid-wait

        # Open the local access path before anything reads a URL. This runs on
        # the adopt branch above as well as here — an adopted instance is never
        # created, so create_once() is the one place this must NOT live.
        session.update(phase="opening the access path")
        provider.ensure_access(pod_id)

        urls = provider.service_urls(pod_id)              # llm/embedder/reranker base URLs
        session.update(proxy_url=urls["llm"],             # primary URL
                       phase="waiting for all three services to become healthy")
        if not _wait_for_all_ready(session, pod_id):      # LLM + embedder + reranker (or cancel)
            return                                        # cancelled mid-wait

        # Persist state only once safely up, matching the scripts' contract.
        provider.write_state(pod_id)

        if not session.commit_running():                 # flip to RUNNING unless a stop won
            return                                        # a stop overtook us at the finish line
    except Exception as e:  # noqa: BLE001 — surface any failure to the UI
        # A failed launch owns its own teardown: no stop worker is coming, so
        # anything ensure_access opened would leak. (The cancel returns above
        # deliberately do NOT release — a cancel means POD DOWN is already
        # running, and stop() closes the path once the pod is actually gone.)
        _release_access(session, active_provider())
        # Type only — str(e) from an SDK/httpx can embed request context
        # (URLs, headers, the API key). The phase field already names the
        # failing step; full detail stays server-side / in the cloud console.
        session.add_event(f"error during start: {type(e).__name__}", "system")
        session.update(state=State.ERROR, phase="error during start",
                       error=f"{type(e).__name__} — check the provider console for details")


def _create_retries() -> int:
    """How many create attempts before giving up. Env-tunable; default 40 (~10 min
    at the 15s delay). Because POD DOWN can cancel mid-loop, a generous default is
    safe — it rides out a transient capacity shortage instead of failing at 15."""
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


def _create_with_retries(session: PodSession) -> dict | None:
    """Create the instance, riding out the capacity lottery. None if cancelled.

    The retry loop lives HERE rather than in a provider so every attempt is
    reported to the UI event feed and each inter-attempt wait is cancel-aware —
    POD DOWN interrupts the wait immediately. The provider decides only what one
    attempt is, and which failures are worth retrying: a stockout is transient,
    but a quota or permission error must surface at once rather than hide behind
    ten minutes of retries.
    """
    provider = active_provider()
    bearer = read_secret(_secrets.bearer_token)          # -> the stack's API key
    hf = read_secret(_secrets.hf_token)                  # -> weight-pull token (all 3 services)
    if session.cancel.is_set():                          # Down pressed before we resolve
        return None
    ctx = provider.prepare_create(session)               # pre-create lookups (cancel-aware)
    if ctx is None:                                      # cancelled during preparation
        return None
    secrets = {"bearer": bearer, "hf": hf}
    retries, delay = _create_retries(), _create_retry_delay()
    for attempt in range(1, retries + 1):
        if session.cancel.is_set():                      # Down pressed between attempts
            return None
        session.update(phase=provider.create_phase(ctx, attempt, retries))
        try:
            return provider.create_once(ctx, secrets)    # one attempt
        except provider.create_error_types as e:
            if not provider.is_retryable_create_error(e):  # real error — surface it
                raise
            session.add_event(provider.capacity_note(attempt, retries, delay))  # UI feed
            if attempt < retries and _sleep_or_cancel(session, delay):  # cancel-aware wait
                return None
    raise RuntimeError(provider.create_exhausted_message(retries, delay))


def _wait_for_running(session: PodSession, pod_id: str) -> bool:
    """Poll until the instance is up with a live container, or cancel/timeout."""
    provider = active_provider()
    deadline = time.time() + RUNNING_TIMEOUT_S            # absolute give-up time
    while time.time() < deadline:                        # loop until deadline
        if session.cancel.is_set():                      # Down pressed mid-provision
            return False                                 # bail; stop worker owns state
        instance = provider.get_instance(pod_id)         # fetch current record
        status = provider.status_of(instance) if instance else None   # e.g. "RUNNING"
        rate = provider.cost_per_hr(instance) if instance else None   # for the meter
        if rate is not None:                             # capture the hourly rate
            session.update(cost_per_hr=rate)
        session.update(phase=f"waiting for RUNNING… ({status})")  # progress text
        if instance and provider.is_up(instance):        # container is actually up
            return True                                  # ready to check the services next
        if _sleep_or_cancel(session, POLL_S):            # wait, but wake early on cancel
            return False                                 # cancelled during the sleep
    raise RuntimeError(f"pod {pod_id} did not reach RUNNING within {RUNNING_TIMEOUT_S}s")


def _service_probes(pod_id: str, bearer: str) -> dict:
    """Map each service to its (health-url, auth-headers). All three are gated by
    the same bearer, since their ports are reachable over the provider's ingress."""
    auth = {"Authorization": f"Bearer {bearer}"}
    urls = active_provider().service_urls(pod_id)        # {llm,embedder,reranker: base URL}
    return {
        "llm":      (f"{urls['llm']}/v1/models",   auth),
        "embedder": (f"{urls['embedder']}/health", auth),
        "reranker": (f"{urls['reranker']}/health", auth),
    }


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
    """Probe all three services once and update the health tiles — used by the
    background poller while RUNNING (healthy | down)."""
    bearer = read_secret(_secrets.bearer_token)
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
    """Fire a REAL completion + embedding + rerank at the three services, recording
    pass/fail + latency (and the embedding dimension) into session.test_result.

    Uses the same endpoints a RAG client will: vLLM OpenAI-compat
    /v1/chat/completions, TEI OpenAI-compat /v1/embeddings, and TEI native /rerank.
    Runs in a background thread (launched by /pod/test)."""
    try:
        pod_id = session.pod_id
        if not pod_id:
            session.update(test_running=False, test_result={"error": "no pod running"})
            return
        provider = active_provider()
        stack = provider.stack_config()                  # served name + embed model id
        bearer = read_secret(_secrets.bearer_token)      # gates all three services
        auth = {"Authorization": f"Bearer {bearer}"}
        urls = provider.service_urls(pod_id)
        session.update(test_running=True)
        session.add_event("stack test started", "system")
        services: dict = {}

        # 1) LLM — OpenAI chat completion (model must equal vLLM --served-model-name).
        ok, ms, status, data = _timed_post(
            f"{urls['llm']}/v1/chat/completions", auth,
            {"model": stack["llm_served_name"],
             "messages": [{"role": "user", "content": "ping"}],
             "max_tokens": 1, "temperature": 0})
        services["llm"] = {"ok": ok, "latency_ms": ms,
                           "detail": "completion ok" if ok else f"HTTP {status}"}
        session.add_event(f"stack test — llm {'ok' if ok else 'FAIL'} {ms}ms", "health")

        # 2) Embedder — OpenAI embeddings; the vector length is the served dimension.
        ok, ms, status, data = _timed_post(
            f"{urls['embedder']}/v1/embeddings", auth,
            {"model": stack["embed_model_id"], "input": "hello world"})
        dim = None
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
        ok, ms, status, data = _timed_post(
            f"{urls['reranker']}/rerank", auth,
            {"query": "what does podlink do",
             "texts": ["podlink controls a GPU pod", "an unrelated sentence"]})
        top = None
        try:
            top = round(max(x["score"] for x in data), 3)  # TEI returns [{index, score}, ...]
        except Exception:  # noqa: BLE001
            pass
        services["reranker"] = {"ok": ok, "latency_ms": ms,
                                "detail": f"top score {top}" if top is not None else (f"HTTP {status}" if not ok else "ok")}
        session.add_event(f"stack test — reranker {'ok' if ok else 'FAIL'} {ms}ms", "health")

        all_ok = all(s["ok"] for s in services.values())
        session.update(test_running=False,
                       test_result={"services": services, "embedding_dim": dim, "all_ok": all_ok})
        session.add_event(f"stack test {'PASSED' if all_ok else 'had failures'}", "system")
    except Exception as e:  # noqa: BLE001               # never let the test thread die silently
        session.add_event(f"stack test error: {type(e).__name__}", "system")
        session.update(test_running=False, test_result={"error": type(e).__name__})


def _wait_for_all_ready(session: PodSession, pod_id: str) -> bool:
    """Poll all three services until each returns 200 (or cancel/timeout).

    "Pod ready" = LLM /v1/models AND embedder /health AND reranker /health. Each
    service is dropped from the poll set once healthy; the per-service tiles and
    the phase text report which are still coming up.
    """
    bearer = read_secret(_secrets.bearer_token)          # gates all three services
    probes = _service_probes(pod_id, bearer)             # service -> (url, headers)
    ready: set[str] = set()                              # services confirmed 200
    deadline = time.time() + READY_TIMEOUT_S             # absolute give-up time
    while time.time() < deadline:                        # loop until deadline
        if session.cancel.is_set():                      # Down pressed while weights load
            return False                                 # bail to the stop worker
        for name, (url, headers) in probes.items():      # probe each not-yet-ready service
            if name in ready:                            # already up — skip
                continue
            if _probe_service(url, headers):             # this service is serving
                ready.add(name)
        # Reflect per-service health onto the tiles (healthy vs still pending).
        _apply_health(session, {n: ("healthy" if n in ready else "pending") for n in probes})
        if len(ready) == len(probes):                    # all three healthy
            session.update(phase="all services healthy")
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
    """Terminate the instance (GPU released) and VERIFY it actually left RUNNING.

    Terminate, not stop: a stopped pod can be host-pinned and fail to resume when
    that host has no free GPU. Works whether the instance is still provisioning or
    fully RUNNING. Never trusts the on-disk state file alone — resolves the id from
    live memory or by name, so an instance created before the file existed is still
    caught.
    """
    provider = None                                      # for the finally below
    try:
        provider = active_provider()
        provider.authenticate()                          # load cloud credentials

        pod_id = _resolve_pod_id(session, provider)      # find the instance however we can
        if pod_id is None:                               # genuinely nothing exists to terminate
            session.update(state=State.IDLE, phase="no pod found — nothing to terminate",
                           pod_id=None, proxy_url=None,   # settle back to IDLE
                           billing_started_at=None, cost_per_hr=None, auto_terminate_at=None,
                           services={n: "unknown" for n in SERVICES})
            return

        session.update(pod_id=pod_id, phase=f"terminating pod {pod_id}")  # progress text
        provider.terminate(pod_id)                       # release the GPU

        if _verify_terminated(session, pod_id):          # confirm it left RUNNING / vanished
            provider.clear_state()                       # drop the state file so a stale id can't resurface
            session.update(state=State.IDLE, phase="terminated — GPU released",
                           pod_id=None, proxy_url=None,   # safe: back to IDLE
                           billing_started_at=None, cost_per_hr=None, auto_terminate_at=None,  # stop the meter
                           services={n: "unknown" for n in SERVICES},  # clear the health tiles
                           test_result=None, test_running=False)  # clear the stack-test result

        else:                                            # could NOT confirm — do not lie
            session.update(
                state=State.ERROR,                       # loud error state, buttons stay live
                phase="TERMINATE NOT VERIFIED — check the provider console immediately",
                error="terminate issued but the pod did not leave RUNNING within "
                      f"{STOP_VERIFY_TIMEOUT_S}s; it may still be charging.",
            )
    except Exception as e:  # noqa: BLE001                # surface any terminate failure
        # Type only — never interpolate str(e); it may leak request context/secrets.
        session.add_event(f"error during terminate: {type(e).__name__}", "system")
        session.update(state=State.ERROR, phase="error during terminate",
                       error=f"{type(e).__name__} — check the provider console for details")
    finally:
        # Every path out of a stop closes the access path — the confirmed
        # terminate, the nothing-to-terminate early return, and the error
        # branches. The pod is gone or unreachable in all of them, so an open
        # tunnel is pure leak.
        if provider is not None:
            _release_access(session, provider)


def _resolve_pod_id(session: PodSession, provider) -> str | None:
    """Find the instance id, most-authoritative source first.

    1. In-memory id captured the instant create returned.
    2. Live lookup by name (catches an instance created just as Down was pressed —
       retried because the create call may still be returning).
    3. The on-disk state file (only exists after a fully successful start).
    """
    if session.pod_id:                                   # (1) fastest, most reliable
        return session.pod_id
    for _ in range(5):                                   # (2) retry to beat the create race
        found = provider.find_existing()                 # enumerate instances, match by name
        if found is not None:                            # a matching instance now exists
            return provider.instance_id(found)
        time.sleep(2)                                    # let an in-flight create surface
    return provider.state_file_instance_id()             # (3) disk fallback, or None


def _verify_terminated(session: PodSession, pod_id: str) -> bool:
    """Confirm the instance left RUNNING (or is gone) — proof the GPU is released.

    terminate already succeeded, so a subsequent lookup may return None OR raise
    (the instance is being deleted and is no longer retrievable). Both mean 'not
    running', so we treat either as terminated rather than reporting a false
    failure on the lookup call.
    """
    provider = active_provider()
    deadline = time.time() + STOP_VERIFY_TIMEOUT_S       # absolute give-up time
    while time.time() < deadline:                        # poll until confirmed or timeout
        try:
            instance = provider.get_instance(pod_id)     # fetch current record
        except Exception:  # noqa: BLE001                # already deleted / not retrievable
            return True                                  # terminate was accepted => gone
        if instance is None:                             # instance gone entirely
            return True                                  # => definitely not billing GPU
        status = provider.status_of(instance)            # e.g. "EXITED"/"STOPPED"
        session.update(phase=f"verifying termination… ({status})")  # progress text
        if status and not provider.is_running(instance):  # left RUNNING => GPU released
            return True
        time.sleep(STOP_VERIFY_POLL_S)                   # brief pause before re-checking
    return False                                         # could not confirm within the window
