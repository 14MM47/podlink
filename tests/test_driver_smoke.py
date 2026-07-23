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
from app.session import PodSession                   # noqa: E402

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
    print("all driver smoke tests passed.")
