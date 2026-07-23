"""Stub-based smoke tests for the three-service driver logic.

The RunPod SDK isn't installed in CI/dev (see requirements.txt), and the live
API can't be hit here, so we inject a fake `runpod` + fake `_secrets` into
sys.modules and stub the audited egress client. That lets us exercise the two
riskiest pieces of new logic without a real pod:

  * resolve_gpu_id() — RTX Pro 6000 catalog lookup
  * _wait_for_all_ready() — the LLM + embedder + reranker readiness gate

Run: python3 tests/test_driver_smoke.py   (plain asserts, no pytest needed)
"""
from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # import the `app` package


# --- fake runpod SDK --------------------------------------------------------
fake_runpod = types.ModuleType("runpod")
fake_runpod.api_key = None
fake_runpod._gpus = [                              # what resolve_gpu_id sees
    # resolve_gpu_id requires memoryInGb >= 90 and skips MIG slices, so entries
    # carry a realistic memoryInGb (the 96 GB card qualifies; the 5090 does not).
    {"id": "NVIDIA GeForce RTX 5090", "displayName": "RTX 5090", "memoryInGb": 32},
    {"id": "NVIDIA RTX PRO 6000 Blackwell WE", "displayName": "RTX PRO 6000 Blackwell Workstation Edition", "memoryInGb": 96},
]
fake_runpod.get_gpus = lambda: fake_runpod._gpus
fake_runpod.get_pods = lambda: []
fake_runpod.get_pod = lambda pod_id: {"id": pod_id, "desiredStatus": "RUNNING", "runtime": {"x": 1}}
fake_runpod.create_pod = lambda **kw: {"id": "newpod123", **kw}
fake_runpod.resume_pod = lambda pod_id, **kw: None
fake_runpod.stop_pod = lambda pod_id: None
fake_runpod.terminate_pod = lambda pod_id: None    # POD DOWN now terminates (weights on the volume)
sys.modules["runpod"] = fake_runpod

# pod_up.py does `from runpod.error import QueryError` at import time (create-retry
# lottery), so the fake SDK must expose that submodule or the driver import fails.
fake_runpod_error = types.ModuleType("runpod.error")
class QueryError(Exception):                        # noqa: E742 — mirrors the real class name
    pass
fake_runpod_error.QueryError = QueryError
fake_runpod.error = fake_runpod_error
sys.modules["runpod.error"] = fake_runpod_error

# --- fake secrets -----------------------------------------------------------
fake_secrets = types.ModuleType("_secrets")
fake_secrets.runpod_api_key = lambda: "rk_dummy"
fake_secrets.hf_token = lambda: "hf_dummy"
fake_secrets.bearer_token = lambda: "bearer_dummy"
sys.modules["_secrets"] = fake_secrets

# Now safe to import the driver (it self-bootstraps pod_control onto sys.path).
from app import runpod_driver as rd                 # noqa: E402
from app.session import PodSession, State            # noqa: E402

# Make the readiness loop fast for tests.
rd.READY_TIMEOUT_S = 3
rd.POLL_S = 0.05


# --- stub the audited egress client with scriptable per-URL responses -------
class _Resp:
    def __init__(self, status): self.status_code = status


def install_fake_client(status_for):
    """status_for(url) -> int. Patch rd.egress_logger.client to use it."""
    @contextlib.contextmanager
    def _client(timeout=15.0):
        class _C:
            def get(self, url, headers=None):
                return _Resp(status_for(url))
        yield _C()
    rd.egress_logger.client = _client


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise SystemExit(f"FAILED: {name}")


# --- tests ------------------------------------------------------------------
def test_resolve_gpu_id_matches_rtx_pro_6000():
    gid = rd.pod_up.resolve_gpu_id()
    check("resolve_gpu_id -> RTX Pro 6000 id",
          gid == "NVIDIA RTX PRO 6000 Blackwell WE")


def test_resolve_gpu_id_raises_when_absent():
    saved = fake_runpod._gpus
    fake_runpod._gpus = [{"id": "A100", "displayName": "A100 80GB"}]
    try:
        raised = False
        try:
            rd.pod_up.resolve_gpu_id()
        except RuntimeError:
            raised = True
        check("resolve_gpu_id raises when no match", raised)
    finally:
        fake_runpod._gpus = saved


def test_all_ready_true_when_all_200():
    install_fake_client(lambda url: 200)
    s = PodSession()
    s.try_begin_start(None)
    check("_wait_for_all_ready True when all 3 healthy",
          rd._wait_for_all_ready(s, "pod1") is True)


def test_all_ready_waits_for_stragglers_then_times_out():
    # embedder + reranker are 200, LLM never becomes ready -> timeout naming 'llm'.
    install_fake_client(lambda url: 200 if "/health" in url else 503)
    s = PodSession()
    s.try_begin_start(None)
    raised_msg = ""
    try:
        rd._wait_for_all_ready(s, "pod1")
    except RuntimeError as e:
        raised_msg = str(e)
    check("_wait_for_all_ready times out on the straggler",
          "pending" in raised_msg and "llm" in raised_msg)


def test_all_ready_cancels_promptly():
    install_fake_client(lambda url: 503)             # nothing healthy
    s = PodSession()
    s.try_begin_start(None)
    s.cancel.set()                                   # Down pressed
    check("_wait_for_all_ready returns False on cancel",
          rd._wait_for_all_ready(s, "pod1") is False)


def test_create_aborts_on_cancel_before_create():
    s = PodSession()
    s.try_begin_start(None)
    s.cancel.set()
    check("_create_with_fallback returns None when cancelled",
          rd._create_with_fallback(s) is None)


def test_stop_terminates_and_lands_idle():
    # POD DOWN calls runpod.terminate_pod (not stop_pod), verifies the pod left
    # RUNNING, clears the state file, and settles to IDLE. Covers the refactor.
    terminated = {}
    saved_term = fake_runpod.terminate_pod
    saved_get = fake_runpod.get_pod
    saved_state = rd.pod_up.STATE_PATH
    scratch = Path("/tmp/podlink_smoke_state.json")     # never touch the real pod_state.json
    scratch.write_text('{"pod_id": "podX"}')            # something for _clear_state_file to remove
    fake_runpod.terminate_pod = lambda pod_id: terminated.__setitem__("id", pod_id)
    fake_runpod.get_pod = lambda pod_id: {"id": pod_id, "desiredStatus": "EXITED"}  # verify passes at once
    rd.pod_up.STATE_PATH = scratch
    try:
        s = PodSession()
        s.try_begin_start(None)      # -> STARTING
        s.pod_id = "podX"            # known id so _resolve_pod_id short-circuits
        s.try_begin_stop()           # -> STOPPING (raises cancel)
        rd.stop(s)
        snap = s.snapshot()
        check("stop() calls terminate_pod with the pod id", terminated.get("id") == "podX")
        check("stop() lands IDLE after verifying it left RUNNING", snap["state"] == "IDLE")
        check("stop() clears pod_state.json after terminate", not scratch.exists())
    finally:
        fake_runpod.terminate_pod = saved_term
        fake_runpod.get_pod = saved_get
        rd.pod_up.STATE_PATH = saved_state
        scratch.unlink(missing_ok=True)


def test_read_secret_converts_systemexit():
    # The vendored secret readers sys.exit() (SystemExit, a BaseException) on a
    # missing/bad secret; _read_secret must convert it to a catchable RuntimeError
    # so the driver's `except Exception` handlers recover instead of the worker
    # dying with the session stuck STOPPING and the pod still billing.
    def missing():
        raise SystemExit("Missing secret: ~/.config/podlink/runpod_api_key")
    outcome = "no-raise"
    try:
        rd._read_secret(missing)
    except RuntimeError:
        outcome = "RuntimeError"
    except SystemExit:
        outcome = "SystemExit"
    check("_read_secret converts SystemExit -> RuntimeError", outcome == "RuntimeError")


def test_verify_terminated_true_when_get_pod_raises():
    # After terminate_pod succeeds, get_pod may raise (pod already deleted); that
    # must read as 'terminated', not a false failure.
    saved_get = fake_runpod.get_pod
    def boom(pod_id):
        raise RuntimeError("pod not found")
    fake_runpod.get_pod = boom
    try:
        check("_verify_terminated True when get_pod raises",
              rd._verify_terminated(PodSession(), "podX") is True)
    finally:
        fake_runpod.get_pod = saved_get


def test_cost_meter_derives_from_billing_and_rate():
    # With a rate and a billing start 1h ago, the snapshot should report ~1h
    # uptime and ~ the hourly rate as the session cost.
    import time as _t
    s = PodSession()
    s.try_begin_start(None)                          # -> STARTING (a 'live' state)
    s.update(cost_per_hr=2.0, billing_started_at=_t.time() - 3600)
    snap = s.snapshot()
    check("uptime_s ~ 3600", snap["uptime_s"] is not None and abs(snap["uptime_s"] - 3600) <= 2)
    check("session_cost_usd ~ 2.00", abs(snap["session_cost_usd"] - 2.0) <= 0.01)
    # Once IDLE the meter blanks (pod is gone).
    s.state = State.IDLE
    idle = s.snapshot()
    check("cost meter blank when IDLE", idle["uptime_s"] is None and idle["session_cost_usd"] is None)


def test_arm_billing_sets_auto_terminate_when_configured():
    import os as _os
    saved = _os.environ.get("PODLINK_AUTO_TERMINATE_MIN")
    _os.environ["PODLINK_AUTO_TERMINATE_MIN"] = "30"
    try:
        s = PodSession()
        rd._arm_billing(s)
        snap = s.snapshot() if False else None       # snapshot needs a live state; check field directly
        check("billing_started_at set", s.billing_started_at is not None)
        check("auto_terminate_at ~ now+30m",
              s.auto_terminate_at is not None and 1750 <= (s.auto_terminate_at - s.billing_started_at) <= 1810)
    finally:
        if saved is None:
            _os.environ.pop("PODLINK_AUTO_TERMINATE_MIN", None)
        else:
            _os.environ["PODLINK_AUTO_TERMINATE_MIN"] = saved


def test_arm_billing_no_auto_terminate_when_disabled():
    import os as _os
    saved = _os.environ.get("PODLINK_AUTO_TERMINATE_MIN")
    _os.environ.pop("PODLINK_AUTO_TERMINATE_MIN", None)   # default 0 = off
    try:
        s = PodSession()
        rd._arm_billing(s)
        check("no auto-terminate deadline when disabled", s.auto_terminate_at is None)
        check("billing still armed", s.billing_started_at is not None)
    finally:
        if saved is not None:
            _os.environ["PODLINK_AUTO_TERMINATE_MIN"] = saved


def test_phase_change_records_event():
    # A changed phase auto-appends to the streamed event feed; an unchanged one
    # does not.
    s = PodSession()
    s.update(phase="creating pod")
    s.update(phase="creating pod")            # same -> no new event
    s.update(phase="all services healthy")
    msgs = [m for _, m in s.events]
    check("phase changes recorded once each",
          msgs == ["creating pod", "all services healthy"])


def test_apply_health_updates_tiles_and_logs_transitions():
    s = PodSession()
    rd._apply_health(s, {"llm": "pending", "embedder": "healthy", "reranker": "pending"})
    rd._apply_health(s, {"llm": "healthy", "embedder": "healthy", "reranker": "down"})
    snap_services = s.services
    check("tiles reflect latest statuses",
          snap_services == {"llm": "healthy", "embedder": "healthy", "reranker": "down"})
    # Transitions logged: llm pending, embedder healthy, reranker pending (first pass),
    # then llm healthy, reranker down (embedder unchanged -> not re-logged).
    msgs = [m for _, m in s.events]
    check("health transitions logged, unchanged not re-logged",
          msgs == ["llm: pending", "embedder: healthy", "reranker: pending",
                   "llm: healthy", "reranker: down"])


def test_probe_health_once_marks_healthy_and_down():
    # Stub the egress client so llm/embedder are 200 and reranker is 503.
    install_fake_client(lambda url: 503 if "8081" in url else 200)
    s = PodSession()
    rd.probe_health_once(s, "podX")
    check("llm healthy", s.services["llm"] == "healthy")
    check("embedder healthy", s.services["embedder"] == "healthy")
    check("reranker down (503)", s.services["reranker"] == "down")


def test_create_retries_then_succeeds_and_logs():
    # A retryable capacity error retries and is reported to the event feed; a later
    # attempt that succeeds returns the pod. _sleep_or_cancel is stubbed so the test
    # doesn't actually wait.
    from runpod.error import QueryError as QE
    calls = {"n": 0}
    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise QE("There are no longer any instances available with the requested specifications")
        return {"id": "pod-ok"}
    saved = (rd.pod_up.create_pod_once, rd.pod_up.ensure_template,
             rd.pod_up.resolve_gpu_id, rd._sleep_or_cancel)
    rd.pod_up.create_pod_once = flaky
    rd.pod_up.ensure_template = lambda: "tmpl1"
    rd.pod_up.resolve_gpu_id = lambda: "gpuX"
    rd._sleep_or_cancel = lambda session, secs: False   # instant, not cancelled
    try:
        s = PodSession(); s.try_begin_start(None)
        pod = rd._create_with_fallback(s)
        check("retries then succeeds on the 3rd attempt", pod == {"id": "pod-ok"} and calls["n"] == 3)
        check("retry attempts logged to the event feed",
              any("no host with capacity" in m for _, m in s.events))
    finally:
        (rd.pod_up.create_pod_once, rd.pod_up.ensure_template,
         rd.pod_up.resolve_gpu_id, rd._sleep_or_cancel) = saved


def test_create_non_retryable_error_surfaces():
    # A non-capacity QueryError (bad spec/auth) must NOT be retried — it surfaces.
    from runpod.error import QueryError as QE
    saved = (rd.pod_up.create_pod_once, rd.pod_up.ensure_template, rd.pod_up.resolve_gpu_id)
    rd.pod_up.create_pod_once = lambda *a, **k: (_ for _ in ()).throw(QE("invalid gpu spec"))
    rd.pod_up.ensure_template = lambda: "t"
    rd.pod_up.resolve_gpu_id = lambda: "g"
    try:
        s = PodSession(); s.try_begin_start(None)
        raised = False
        try:
            rd._create_with_fallback(s)
        except QE:
            raised = True
        check("non-retryable create error surfaces (not retried)", raised)
    finally:
        (rd.pod_up.create_pod_once, rd.pod_up.ensure_template, rd.pod_up.resolve_gpu_id) = saved


if __name__ == "__main__":
    print("driver smoke tests:")
    test_resolve_gpu_id_matches_rtx_pro_6000()
    test_resolve_gpu_id_raises_when_absent()
    test_all_ready_true_when_all_200()
    test_all_ready_waits_for_stragglers_then_times_out()
    test_all_ready_cancels_promptly()
    test_create_aborts_on_cancel_before_create()
    test_stop_terminates_and_lands_idle()
    test_read_secret_converts_systemexit()
    test_verify_terminated_true_when_get_pod_raises()
    test_cost_meter_derives_from_billing_and_rate()
    test_arm_billing_sets_auto_terminate_when_configured()
    test_arm_billing_no_auto_terminate_when_disabled()
    test_phase_change_records_event()
    test_apply_health_updates_tiles_and_logs_transitions()
    test_probe_health_once_marks_healthy_and_down()
    test_create_retries_then_succeeds_and_logs()
    test_create_non_retryable_error_surfaces()
    print("all driver smoke tests passed.")
