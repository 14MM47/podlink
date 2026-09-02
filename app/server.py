"""FastAPI server for podlink — serves the two-button UI and drives the pod.

Security posture (see README): bound to 127.0.0.1 only, and state-changing
POSTs require a per-process token in the X-Podlink-Token header. Because a
cross-origin browser page can neither read GET /config (no CORS header) nor set
a custom request header without a preflight we reject, this blocks browser-based
CSRF that could otherwise start a pod and run up GPU charges. It cannot defend
against a malicious process already running as this user — nothing can.
"""
from __future__ import annotations  # postponed annotation evaluation

import asyncio                       # sleep between SSE pushes
import json                          # serialise snapshots for SSE frames
import os                            # active-profile name from the launch env
import secrets as pysecrets          # cryptographic token + constant-time compare
import threading                     # run driver work off the request thread
import time                          # auto-terminate watchdog clock
from pathlib import Path             # locate the static/ directory

from fastapi import Body, FastAPI, Header, HTTPException, Request   # web framework primitives
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # response types

from . import driver                 # start()/stop() entry points (provider-neutral)
from . import profiles as profiles_conf  # profile discovery + runtime switching
from . import providers              # the configured cloud
from .session import PodSession, State  # the shared state machine

app = FastAPI(title="podlink", docs_url=None, redoc_url=None)  # no public API docs pages

SESSION = PodSession()               # the single, process-wide pod session
# Per-process token; regenerated every launch so a stale token can't act.
TOKEN = pysecrets.token_urlsafe(24)  # unguessable per-run secret
STATIC_DIR = Path(__file__).resolve().parent / "static"  # …/app/static


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    """Forbid framing so a malicious page can't clickjack POD UP/DOWN.

    Without this, an external site could iframe the localhost UI and overlay a
    transparent trick-click on the buttons — the framed page's own JS fetches
    the token, so the per-process token gate alone would not stop it.
    """
    response = await call_next(request)                       # run the route
    response.headers["X-Frame-Options"] = "DENY"              # legacy anti-framing
    response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"  # modern
    return response


def _require_token(x_podlink_token: str | None) -> None:
    """Reject any state-changing request without the matching header token."""
    if not x_podlink_token or not pysecrets.compare_digest(x_podlink_token, TOKEN):
        raise HTTPException(status_code=403, detail="missing or invalid token")  # constant-time


def _launch(target) -> None:
    """Run a driver entry point in a daemon background thread."""
    threading.Thread(target=target, args=(SESSION,), daemon=True).start()  # non-blocking


_WATCH_INTERVAL_S = 15               # how often the idle watchdog checks the deadline
_HEALTH_INTERVAL_S = 30              # how often to re-probe the three services while RUNNING


def _health_watch() -> None:
    """Background poller: re-probe the three services while the pod is RUNNING so
    the health tiles reflect ongoing state (a service that dies shows as down).

    Only runs in RUNNING — bring-up already probes in _wait_for_all_ready. Each
    pass is 3 audited GETs, so the interval is kept modest to bound egress-log
    growth.
    """
    while True:
        time.sleep(_HEALTH_INTERVAL_S)
        try:
            if SESSION.state == State.RUNNING and SESSION.pod_id:
                driver.probe_health_once(SESSION, SESSION.pod_id)
        except Exception:  # noqa: BLE001 — a watchdog must never die on a transient error
            pass


def _auto_terminate_watch() -> None:
    """Background watchdog: terminate the pod once its auto-terminate deadline passes.

    Guards against a forgotten pod billing indefinitely. The deadline is armed by
    the driver when a pod is created (PODLINK_AUTO_TERMINATE_MIN) and cleared by
    /pod/keepalive. We only fire while a pod is actually up/coming up, and go
    through the same try_begin_stop + stop path as a manual POD DOWN.
    """
    while True:
        time.sleep(_WATCH_INTERVAL_S)
        try:
            deadline = SESSION.auto_terminate_at
            if (deadline and time.time() >= deadline
                    and SESSION.state in (State.STARTING, State.RUNNING)):
                if SESSION.try_begin_stop():          # atomic: enter STOPPING + cancel
                    SESSION.add_event("idle auto-terminate — deadline reached", "system")
                    SESSION.update(phase="idle auto-terminate — deadline reached")
                    _launch(driver.stop)              # terminate + verify in the background
        except Exception:  # noqa: BLE001 — a watchdog must never die on a transient error
            pass


def _client_env(pod_id: str) -> str:
    """The RAG-client .env block for this pod, from the active provider.

    The bearer is left as a PLACEHOLDER — it must never reach the browser.
    EMBEDDING_DIMENSIONS comes from a stack-test detection when one has run,
    since it depends on what the embedder actually serves.
    """
    dim = (SESSION.test_result or {}).get("embedding_dim")   # None until 'Test stack' runs
    return providers.active().client_env(pod_id, dim)


def _snapshot() -> dict:
    """Session snapshot plus the process-constant deploy flags the UI needs.

    The provider adds its own fields — chiefly `persistence_configured`, which
    tells the UI that POD DOWN (a terminate) will destroy the downloaded weights
    when no persistent storage is set. These are process constants, so emitting
    them on every SSE frame is cheap and the dedupe in /events still works.
    """
    provider = providers.active()
    snap = SESSION.snapshot()                                 # base state + button flags
    snap.update(provider.snapshot_fields())                   # provider/persistence/model
    snap["active_profile"] = os.environ.get("PODLINK_PROFILE") or None  # start.sh --profile
    # Service endpoints for the UI's link row — cheap, pure string building, and
    # no wider exposure than pod_id itself (this route is localhost-only).
    snap["service_urls"] = provider.service_urls(SESSION.pod_id) if SESSION.pod_id else None
    return snap


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")   # serve the two-button page


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(STATIC_DIR / "app.js")       # serve the frontend script


@app.get("/config")
def config() -> JSONResponse:
    # No CORS header => cross-origin JS cannot read this token.
    return JSONResponse({"token": TOKEN})            # same-origin page reads it


@app.get("/status")
def status() -> JSONResponse:
    return JSONResponse(_snapshot())                 # state + button flags + deploy flags


@app.get("/pods")
def pods(x_podlink_token: str | None = Header(default=None)) -> JSONResponse:
    _require_token(x_podlink_token)                   # gated: triggers an authed cloud call
    try:
        return JSONResponse({"pods": driver.list_pods()})  # whitelisted fields only
    except Exception as e:                            # noqa: BLE001
        # Type only in detail — str(e) could embed the API key.
        raise HTTPException(status_code=502, detail=f"pod list failed: {type(e).__name__}")


@app.get("/profiles")
def profiles_list(x_podlink_token: str | None = Header(default=None)) -> JSONResponse:
    """The available stack profiles (names only — conf contents never leave the server)."""
    _require_token(x_podlink_token)                  # gated: enumerates local config
    return JSONResponse({
        "profiles": profiles_conf.list_profiles(),
        "active": os.environ.get("PODLINK_PROFILE") or None,
    })


@app.post("/profile/select")
def profile_select(profile: str | None = Body(default=None, embed=True),
                   x_podlink_token: str | None = Header(default=None)) -> JSONResponse:
    """Switch the stack profile for the NEXT pod launch (None/"" = base conf only).

    Only allowed while no pod exists: a switch swaps the process PODLINK_* env
    and re-bakes pod_up's constants, which would desync the volume guard, the
    served-name health probes, and cost attribution for a pod already up or on
    its way up. STARTING/RUNNING/STOPPING therefore 409, as does ERROR with a
    lingering pod id (e.g. an unverified terminate).
    """
    _require_token(x_podlink_token)                  # CSRF/token gate
    profile = profile or None                        # normalise "" -> None (base)
    if profile is not None:
        if not profiles_conf.valid_name(profile):    # reject path fragments etc.
            raise HTTPException(status_code=400, detail="invalid profile name")
        if not (profiles_conf.PROFILE_DIR / f"{profile}.conf").is_file():
            raise HTTPException(status_code=404, detail="unknown profile")
    if SESSION.state not in (State.IDLE, State.ERROR) or SESSION.pod_id:
        raise HTTPException(status_code=409,
                            detail="profile switch only available while no pod exists")
    try:
        profiles_conf.apply(profile)                 # swap env + reload pod_up constants
    except Exception as e:  # noqa: BLE001 — apply() already rolled back
        # Type only — a conf-parse/reload error message could embed local paths.
        raise HTTPException(status_code=400, detail=f"profile apply failed: {type(e).__name__}")
    SESSION.add_event(f"profile switched to {profile or 'base (no profile)'}", "system")
    return JSONResponse(_snapshot())                 # active_profile/model/volume now updated


@app.post("/pod/up")
def pod_up(target: str | None = Body(default=None, embed=True),
           x_podlink_token: str | None = Header(default=None)) -> JSONResponse:
    _require_token(x_podlink_token)                   # CSRF/token gate
    target = target or None                           # normalise "" -> None (Auto)
    # Validate the client-supplied id against the provider's own id shape, before
    # it can reach any SDK call.
    if target is not None and not providers.active().valid_instance_id(target):
        raise HTTPException(status_code=400, detail="invalid pod id format")
    if not SESSION.try_begin_start(target):          # atomically enter STARTING (None = Auto)
        # Not in a state where Up is allowed (already starting/running/stopping).
        raise HTTPException(status_code=409, detail="pod up not available in current state")
    _launch(driver.start)                            # provision in the background
    return JSONResponse(_snapshot())                 # echo the new state


@app.post("/pod/down")
def pod_down(confirm: bool = Body(default=False, embed=True),
             x_podlink_token: str | None = Header(default=None)) -> JSONResponse:
    _require_token(x_podlink_token)                   # CSRF/token gate
    # Server-side destructive-action guard. With no persistent storage, terminate
    # DESTROYS the downloaded weights. The browser shows a confirm dialog, but
    # that JS is bypassable (curl, a script, a stale tab), so the server itself
    # refuses to terminate unless the caller explicitly confirms. With storage
    # configured, terminate is non-destructive and no confirmation is required.
    if not driver.persistence_configured() and not confirm:
        raise HTTPException(
            status_code=428,                          # Precondition Required
            detail='no persistent storage configured — POD DOWN will destroy the '
                   'downloaded model weights; resend with {"confirm": true} to proceed',
        )
    if not SESSION.try_begin_stop():                 # atomically enter STOPPING + cancel
        raise HTTPException(status_code=409, detail="pod down not available in current state")
    _launch(driver.stop)                             # terminate + verify in the background
    return JSONResponse(_snapshot())                 # echo the new state


@app.post("/pod/keepalive")
def pod_keepalive(x_podlink_token: str | None = Header(default=None)) -> JSONResponse:
    """Disarm the idle auto-terminate timer ('keep this pod alive')."""
    _require_token(x_podlink_token)                  # CSRF/token gate
    SESSION.update(auto_terminate_at=None)           # cancel the pending auto-terminate
    return JSONResponse(_snapshot())                 # echo the new state


@app.get("/pod/env")
def client_env(x_podlink_token: str | None = Header(default=None)) -> JSONResponse:
    """The RAG-client .env block for the running pod (bearer left as a placeholder)."""
    _require_token(x_podlink_token)                  # token gate (consistency)
    pod_id = SESSION.pod_id
    if not pod_id:
        raise HTTPException(status_code=409, detail="no pod running")
    return JSONResponse({"env": _client_env(pod_id)})


@app.post("/pod/test")
def pod_test(x_podlink_token: str | None = Header(default=None)) -> JSONResponse:
    """Run a real completion + embedding + rerank against the pod, in the background."""
    _require_token(x_podlink_token)                  # CSRF/token gate
    if SESSION.state != State.RUNNING or not SESSION.pod_id:
        raise HTTPException(status_code=409, detail="pod not running")
    if SESSION.test_running:
        raise HTTPException(status_code=409, detail="a stack test is already running")
    _launch(driver.test_stack)                       # fire the three probes off-thread
    return JSONResponse(_snapshot())


# Background daemon threads for the process lifetime: idle-safety watchdog and
# the continuous service-health poller.
threading.Thread(target=_auto_terminate_watch, daemon=True).start()
threading.Thread(target=_health_watch, daemon=True).start()


@app.get("/events")
async def events(request: Request) -> StreamingResponse:
    """Server-Sent Events: push a fresh snapshot ~1/s (and immediately on change)."""
    async def gen():                                 # async generator of SSE frames
        last = None                                  # last snapshot we sent
        while True:                                  # stream until client disconnects
            if await request.is_disconnected():      # browser closed the tab/stream
                break                                # end the generator
            snap = _snapshot()                       # current state + deploy flags
            if snap != last:                         # only emit on change (cheap dedupe)
                yield f"data: {json.dumps(snap)}\n\n"  # SSE frame format
                last = snap                          # remember what we sent
            await asyncio.sleep(1.0)                  # poll cadence
    return StreamingResponse(gen(), media_type="text/event-stream")  # SSE content type
