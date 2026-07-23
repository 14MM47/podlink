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
import re                            # validate the client-supplied pod id
import secrets as pysecrets          # cryptographic token + constant-time compare
import threading                     # run driver work off the request thread
from pathlib import Path             # locate the static/ directory

from fastapi import Body, FastAPI, Header, HTTPException, Request   # web framework primitives
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # response types

from . import runpod_driver          # start()/stop() entry points
from .session import PodSession      # the shared state machine

app = FastAPI(title="podlink", docs_url=None, redoc_url=None)  # no public API docs pages

SESSION = PodSession()               # the single, process-wide pod session
# Per-process token; regenerated every launch so a stale token can't act.
TOKEN = pysecrets.token_urlsafe(24)  # unguessable per-run secret
STATIC_DIR = Path(__file__).resolve().parent / "static"  # …/app/static
# RunPod pod ids are short lowercase-alnum strings; validate before any SDK call
# so a malformed/injected target can't reach runpod.get_pod.
_POD_ID_RE = re.compile(r"^[a-z0-9]{6,40}$")


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


def _snapshot() -> dict:
    """Session snapshot plus process-constant deploy flags the UI needs.

    Adds `network_volume_configured` so the UI can warn that POD DOWN (a terminate)
    will destroy the downloaded weights when no Network Volume is set. It's a
    process constant, so emitting it on every SSE frame is cheap and the dedupe
    in /events still works.
    """
    snap = SESSION.snapshot()                                 # base state + button flags
    snap["network_volume_configured"] = runpod_driver.network_volume_configured()
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
    _require_token(x_podlink_token)                   # gated: triggers an authed RunPod call
    try:
        return JSONResponse({"pods": runpod_driver.list_pods()})  # whitelisted fields only
    except Exception as e:                            # noqa: BLE001
        # Type only in detail — str(e) could embed the API key.
        raise HTTPException(status_code=502, detail=f"pod list failed: {type(e).__name__}")


@app.post("/pod/up")
def pod_up(target: str | None = Body(default=None, embed=True),
           x_podlink_token: str | None = Header(default=None)) -> JSONResponse:
    _require_token(x_podlink_token)                   # CSRF/token gate
    target = target or None                           # normalise "" -> None (Auto)
    if target is not None and not _POD_ID_RE.match(target):  # reject a malformed id
        raise HTTPException(status_code=400, detail="invalid pod id format")
    if not SESSION.try_begin_start(target):          # atomically enter STARTING (None = Auto)
        # Not in a state where Up is allowed (already starting/running/stopping).
        raise HTTPException(status_code=409, detail="pod up not available in current state")
    _launch(runpod_driver.start)                     # provision in the background
    return JSONResponse(_snapshot())                 # echo the new state


@app.post("/pod/down")
def pod_down(confirm: bool = Body(default=False, embed=True),
             x_podlink_token: str | None = Header(default=None)) -> JSONResponse:
    _require_token(x_podlink_token)                   # CSRF/token gate
    # Server-side destructive-action guard. With no Network Volume, terminate
    # DESTROYS the downloaded weights. The browser shows a confirm dialog, but
    # that JS is bypassable (curl, a script, a stale tab), so the server itself
    # refuses to terminate unless the caller explicitly confirms. With a volume
    # configured, terminate is non-destructive and no confirmation is required.
    if not runpod_driver.network_volume_configured() and not confirm:
        raise HTTPException(
            status_code=428,                          # Precondition Required
            detail='no Network Volume configured — POD DOWN will destroy the '
                   'downloaded model weights; resend with {"confirm": true} to proceed',
        )
    if not SESSION.try_begin_stop():                 # atomically enter STOPPING + cancel
        raise HTTPException(status_code=409, detail="pod down not available in current state")
    _launch(runpod_driver.stop)                      # terminate + verify in the background
    return JSONResponse(_snapshot())                 # echo the new state


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
