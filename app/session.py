"""Process-global pod session: a small thread-safe state machine.

One PodSession instance holds the entire live state of the pod for the single
local user. Every field mutation and every state-transition decision goes
through the internal lock so the FastAPI request threads and the background
worker thread never see a torn state.

Button enablement is derived purely from `state`, so the frontend never has to
make policy decisions — it just renders `up_enabled` / `down_enabled`.
"""
from __future__ import annotations  # allow `str | None` annotations on older runtimes

import enum       # for the State enumeration below
import threading  # Lock + Event for thread-safe coordination
import time        # uptime / cost / auto-terminate countdown maths


class State(str, enum.Enum):           # str-mixin so `.value` JSON-serialises cleanly
    IDLE = "IDLE"          # nothing running; Pod Up available, Pod Down greyed
    STARTING = "STARTING"  # provisioning/booting; Pod Down live (the safety window)
    RUNNING = "RUNNING"    # pod up + vLLM serving; Pod Down live
    STOPPING = "STOPPING"  # stop in flight; both buttons greyed
    ERROR = "ERROR"        # something failed; both buttons live so the user can recover


SERVICES = ("llm", "embedder", "reranker")   # the three services whose health we track
_MAX_EVENTS = 60                              # ring-buffer cap for the status event feed


def _default_services() -> dict:
    return {name: "unknown" for name in SERVICES}


class PodSession:
    def __init__(self) -> None:                 # construct the single shared session
        self._lock = threading.Lock()           # guards every read/write below
        self.state: State = State.IDLE          # start with no pod
        self.pod_id: str | None = None          # the instance id once known
        self.target_pod_id: str | None = None    # user-selected pod to adopt (None = Auto)
        self.proxy_url: str | None = None        # https proxy URL to the pod once known
        self.phase: str = "idle"                # human-readable progress line for the UI
        self.error: str | None = None            # last error message, if any
        # Cost meter: GPU $/hr (from the pod record) and when billing began (pod
        # creation). session_cost is derived live in snapshot() from these.
        self.cost_per_hr: float | None = None    # pod's GPU hourly rate, once known
        self.billing_started_at: float | None = None  # epoch when the pod started billing
        # Idle safety: epoch after which the watchdog auto-terminates the pod, or
        # None when disarmed / no auto-terminate configured.
        self.auto_terminate_at: float | None = None
        # Per-service health for the UI tiles: llm/embedder/reranker -> one of
        # unknown | pending | healthy | down.
        self.services: dict = _default_services()
        # Stack test: last result from the "Test stack" button (real completion +
        # embedding + rerank probes), and whether one is currently running.
        self.test_result: dict | None = None
        self.test_running: bool = False
        # Streamed status feed: a ring buffer of (epoch, category, message).
        # Categories split the feed into UI panels by source/function:
        #   lifecycle — provisioning/teardown (phase changes, create retries)
        #   health    — per-service up/down transitions
        #   system    — errors, idle auto-terminate, safety notices
        self.events: list = []
        # Set by a Pod Down request; the start worker polls this and bails out.
        self.cancel = threading.Event()          # cross-thread "stop now" signal

    # --- transition gates (atomic check-and-set) --------------------------

    def try_begin_start(self, target: str | None = None) -> bool:
        """Move IDLE/ERROR -> STARTING. Returns False if a start isn't allowed.

        `target` is a specific pod id to adopt, or None for Auto (create/resume
        the podlink pod). A concrete target is recorded as pod_id immediately so
        an instant POD DOWN can resolve and stop it before the worker records it.
        """
        with self._lock:                                    # atomic w.r.t. other threads
            if self.state not in (State.IDLE, State.ERROR):  # only start from a settled state
                return False                                # reject (server answers 409)
            self.state = State.STARTING                     # enter the provisioning window
            self.phase = "initialising"                     # reset progress text
            self.error = None                               # clear any stale error
            self.target_pod_id = target                     # remember the selection
            self.pod_id = target                            # id if adopting; None if Auto/create
            self.proxy_url = None                            # no proxy URL yet
            self.cost_per_hr = None                          # reset the cost meter for the new run
            self.billing_started_at = None                   # billing clock starts at pod creation
            self.auto_terminate_at = None                    # re-armed by the driver once a pod exists
            self.services = _default_services()              # fresh health tiles for the new pod
            self.test_result = None                          # clear any prior stack-test result
            self.test_running = False
            self.cancel.clear()                             # ensure a fresh (un-cancelled) run
            return True                                     # caller may launch the worker

    def try_begin_stop(self) -> bool:
        """Move STARTING/RUNNING/ERROR -> STOPPING and raise the cancel flag."""
        with self._lock:                                              # atomic transition
            if self.state not in (State.STARTING, State.RUNNING, State.ERROR):
                return False                                          # nothing to stop
            self.state = State.STOPPING                               # both buttons grey now
            self.phase = "stopping — resolving pod"                  # progress text
            # Tell the start worker (if any) to stop polling and exit.
            self.cancel.set()                                        # raise the cancel signal
            return True                                              # caller may launch stop

    def commit_running(self) -> bool:
        """Final start step: RUNNING only if a stop hasn't overtaken us."""
        with self._lock:                                     # atomic final check
            if self.cancel.is_set() or self.state == State.STOPPING:  # a Down arrived first
                return False                                 # don't clobber the stop
            self.state = State.RUNNING                       # pod is genuinely serving
            self.phase = "running — model serving"           # progress text
            return True                                      # start succeeded

    # --- generic field updates -------------------------------------------

    def update(self, **fields) -> None:
        """Set one or more fields under the lock (phase, pod_id, error, …).

        A changed `phase` is auto-appended to the status event feed, so the feed
        is a timestamped progress history without instrumenting every call site.
        """
        with self._lock:                          # keep writes atomic vs snapshot()
            new_phase = fields.get("phase")
            if new_phase is not None and new_phase != self.phase:
                self._record_event(new_phase, "lifecycle")  # provisioning timeline entry
            for key, value in fields.items():      # apply each supplied field
                setattr(self, key, value)          # e.g. self.phase = "…"

    def add_event(self, message: str, category: str = "lifecycle") -> None:
        """Append a categorised status event (health transition, system notice) to the feed."""
        with self._lock:
            self._record_event(message, category)

    def _record_event(self, message: str, category: str) -> None:
        """Append (now, category, message) to the ring buffer. Caller holds the lock."""
        self.events.append((time.time(), category, message))
        if len(self.events) > _MAX_EVENTS:
            del self.events[:-_MAX_EVENTS]         # keep only the most recent

    # --- read side --------------------------------------------------------

    def snapshot(self) -> dict:
        """Immutable view for /status and SSE, including derived button flags."""
        with self._lock:                          # read all fields consistently
            state = self.state                    # local copy for the flag maths below
            now = time.time()
            # Cost meter — live only while a pod exists (STARTING..STOPPING). Once
            # IDLE the pod is gone, so uptime/cost read as None (blank in the UI).
            live = (state in (State.STARTING, State.RUNNING, State.STOPPING)
                    or (state == State.ERROR and self.pod_id is not None))
            uptime_s = int(now - self.billing_started_at) if (live and self.billing_started_at) else None
            session_cost = (round(self.cost_per_hr * uptime_s / 3600.0, 4)
                            if (self.cost_per_hr and uptime_s) else None)
            # Auto-terminate countdown — only meaningful while the pod is up/coming up.
            armed = state in (State.STARTING, State.RUNNING)
            auto_in = (max(0, int(self.auto_terminate_at - now))
                       if (armed and self.auto_terminate_at) else None)
            return {
                "state": state.value,             # e.g. "RUNNING"
                "phase": self.phase,              # progress line
                "pod_id": self.pod_id,            # may be None
                "proxy_url": self.proxy_url,       # may be None
                "error": self.error,              # may be None
                # Pod Up is pressable only when nothing is in flight.
                "up_enabled": state in (State.IDLE, State.ERROR),
                # Pod Down is pressable from the instant Up starts, through
                # RUNNING, and in ERROR (so a stuck pod can always be killed).
                "down_enabled": state in (State.STARTING, State.RUNNING, State.ERROR),
                # Cost meter (derived, live).
                "cost_per_hr": self.cost_per_hr,          # $/hr or None
                "uptime_s": uptime_s,                     # seconds billing, or None
                "session_cost_usd": session_cost,         # cost_per_hr * uptime, or None
                # Idle auto-terminate countdown.
                "auto_terminate_in_s": auto_in,           # seconds until auto-off, or None
                # Per-service health tiles + the streamed status event feed.
                "services": dict(self.services),          # llm/embedder/reranker -> status
                "test_result": self.test_result,          # last stack-test result, or None
                "test_running": self.test_running,        # a stack test is in flight
                "events": [{"t": t, "cat": c, "msg": m}   # recent feed, split by category in the UI
                           for t, c, m in self.events[-40:]],
            }
