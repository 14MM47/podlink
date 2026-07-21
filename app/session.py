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


class State(str, enum.Enum):           # str-mixin so `.value` JSON-serialises cleanly
    IDLE = "IDLE"          # nothing running; Pod Up available, Pod Down greyed
    STARTING = "STARTING"  # provisioning/booting; Pod Down live (the safety window)
    RUNNING = "RUNNING"    # pod up + vLLM serving; Pod Down live
    STOPPING = "STOPPING"  # stop in flight; both buttons greyed
    ERROR = "ERROR"        # something failed; both buttons live so the user can recover


class PodSession:
    def __init__(self) -> None:                 # construct the single shared session
        self._lock = threading.Lock()           # guards every read/write below
        self.state: State = State.IDLE          # start with no pod
        self.pod_id: str | None = None          # RunPod pod id once known
        self.target_pod_id: str | None = None    # user-selected pod to adopt (None = Auto)
        self.proxy_url: str | None = None        # https proxy URL to the pod once known
        self.phase: str = "idle"                # human-readable progress line for the UI
        self.error: str | None = None            # last error message, if any
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
        """Set one or more fields under the lock (phase, pod_id, error, …)."""
        with self._lock:                          # keep writes atomic vs snapshot()
            for key, value in fields.items():      # apply each supplied field
                setattr(self, key, value)          # e.g. self.phase = "…"

    # --- read side --------------------------------------------------------

    def snapshot(self) -> dict:
        """Immutable view for /status and SSE, including derived button flags."""
        with self._lock:                          # read all fields consistently
            state = self.state                    # local copy for the flag maths below
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
            }
