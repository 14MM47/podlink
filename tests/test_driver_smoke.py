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


def install_fake_post(fn):
    """fn(url, body) -> (status, json). Patch rd.egress_logger.client for POST (test_stack)."""
    class _JResp:
        def __init__(self, st, d): self.status_code = st; self._d = d
        def json(self): return self._d
    @contextlib.contextmanager
    def _client(timeout=60.0):
        class _C:
            def post(self, url, headers=None, json=None):
                return _JResp(*fn(url, json))
            def get(self, url, headers=None):
                return _Resp(200)
        yield _C()
    rd.egress_logger.client = _client


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise SystemExit(f"FAILED: {name}")


# --- tests ------------------------------------------------------------------
def test_gql_escape_env_makes_json_values_safe():
    # runpod 1.7.13 interpolates env values into GraphQL unescaped; compact-JSON
    # values (VLLM_EXTRA_ARGS) must be pre-escaped or create_pod dies instantly.
    raw = {"VLLM_EXTRA_ARGS": '--limit-mm-per-prompt {"image":1,"video":0}',
           "PLAIN": "no-quotes", "BACKSLASH": "a\\b"}
    esc = rd.pod_up._gql_escape_env(raw)
    check("quotes escaped for GraphQL",
          esc["VLLM_EXTRA_ARGS"] == '--limit-mm-per-prompt {\\"image\\":1,\\"video\\":0}')
    check("plain values untouched", esc["PLAIN"] == "no-quotes")
    check("backslashes escaped first", esc["BACKSLASH"] == "a\\\\b")
    check("escaped values survive the SDK's f-string as valid GraphQL",
          all('"' not in v.replace('\\"', "").replace("\\\\", "") for v in esc.values()))


def test_create_pod_once_sends_escaped_env():
    import os
    os.environ["PODLINK_VLLM_EXTRA_ARGS"] = '--x {"a":1}'
    try:
        pod = rd.pod_up.create_pod_once("gpu-id", "bearer", "hf", "tmpl-id")
        check("create_pod received escaped env",
              pod["env"]["VLLM_EXTRA_ARGS"] == '--x {\\"a\\":1}')
    finally:
        del os.environ["PODLINK_VLLM_EXTRA_ARGS"]


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


def test_all_ready_warns_and_keeps_waiting_past_the_soft_deadline():
    # Soft deadline passes while the LLM is still loading: the feed gets a reminder,
    # the wait continues, and it still succeeds once the service answers.
    calls = {"n": 0}
    def status_for(url):
        if "/v1/models" in url:
            calls["n"] += 1
            return 200 if calls["n"] > 6 else 503
        return 200
    install_fake_client(status_for)
    saved_warn, saved_hard = rd.READY_WARN_S, rd.READY_TIMEOUT_S
    rd.READY_WARN_S, rd.READY_TIMEOUT_S = 0.05, 30
    s = PodSession()
    s.try_begin_start(None)
    try:
        ok = rd._wait_for_all_ready(s, "pod1")
    finally:
        rd.READY_WARN_S, rd.READY_TIMEOUT_S = saved_warn, saved_hard
    msgs = [m for (_, cat, m) in s.events if cat == "system"]
    check("_wait_for_all_ready keeps waiting past the soft deadline and succeeds", ok is True)
    check("soft deadline posts a 'still pending' reminder that names the pod as billing",
          any("still pending" in m and "billing" in m for m in msgs))


def test_soft_deadline_names_a_stuck_tei_service_when_the_llm_is_up():
    # The LLM answers but the embedder never listens: the reminder is followed by a hint
    # that this is a failed start, not a slow load, and where to look.
    install_fake_client(lambda url: 503 if "8080" in url else 200)
    saved_warn, saved_hard = rd.READY_WARN_S, rd.READY_TIMEOUT_S
    rd.READY_WARN_S, rd.READY_TIMEOUT_S = 0.05, 0.4
    s = PodSession()
    s.try_begin_start(None)
    try:
        try:
            rd._wait_for_all_ready(s, "pod1")
        except RuntimeError:
            pass
    finally:
        rd.READY_WARN_S, rd.READY_TIMEOUT_S = saved_warn, saved_hard
    msgs = [m for (_, cat, m) in s.events if cat == "system"]
    check("stuck embedder while llm is up posts the failed-start hint",
          any("embedder" in m and "failed start" in m and "container log" in m for m in msgs))
    check("no hint when the LLM itself is the straggler",
          rd.stuck_service_hint({"embedder", "reranker"}, ["llm"]) is None)


def test_all_ready_raises_when_the_pod_itself_leaves_running():
    install_fake_client(lambda url: 503)
    saved_get = fake_runpod.get_pod
    fake_runpod.get_pod = lambda pod_id: {"id": pod_id, "desiredStatus": "EXITED"}
    s = PodSession()
    s.try_begin_start(None)
    msg = ""
    try:
        rd._wait_for_all_ready(s, "pod1")
    except RuntimeError as e:
        msg = str(e)
    finally:
        fake_runpod.get_pod = saved_get
    check("_wait_for_all_ready raises promptly when RunPod says the pod left RUNNING",
          "left RUNNING" in msg)


def test_recover_if_healthy_flips_error_to_running():
    saved_state = rd.pod_up.STATE_PATH
    scratch = Path("/tmp/podlink_smoke_recover_state.json")
    rd.pod_up.STATE_PATH = scratch
    try:
        s = PodSession()
        s.try_begin_start(None)
        s.update(pod_id="podLost", state=State.ERROR, error="RuntimeError — gave up")
        install_fake_client(lambda url: 503)
        check("not healthy yet: stays ERROR",
              rd.recover_if_healthy(s, "podLost") is False and s.state == State.ERROR)
        install_fake_client(lambda url: 200)
        check("all healthy: recovers to RUNNING",
              rd.recover_if_healthy(s, "podLost") is True and s.state == State.RUNNING)
        snap = s.snapshot()
        check("recovery clears the error and sets the proxy url",
              snap["error"] is None and snap["proxy_url"] and "podLost" in snap["proxy_url"])
        check("recovery is logged to the feed",
              any("recovered" in m for (_, c, m) in s.events if c == "system"))
    finally:
        rd.pod_up.STATE_PATH = saved_state
        scratch.unlink(missing_ok=True)


def test_adopt_running_on_startup_adopts_our_running_pod():
    import os
    saved_pods, saved_state = fake_runpod.get_pods, rd.pod_up.STATE_PATH
    scratch = Path("/tmp/podlink_smoke_adopt_state.json")
    rd.pod_up.STATE_PATH = scratch
    install_fake_client(lambda url: 200)
    try:
        fake_runpod.get_pods = lambda: [{"id": "podLive", "name": rd.pod_up.POD_NAME,
                                         "desiredStatus": "RUNNING", "runtime": {"x": 1}}]
        s = PodSession()
        rd.adopt_running_on_startup(s)
        check("console start adopts the RUNNING pod by name",
              s.state == State.RUNNING and s.pod_id == "podLive")
        fake_runpod.get_pods = lambda: []
        s2 = PodSession()
        rd.adopt_running_on_startup(s2)
        check("no live pod: stays IDLE and creates nothing", s2.state == State.IDLE)
        os.environ["PODLINK_ADOPT_ON_START"] = "0"
        fake_runpod.get_pods = lambda: [{"id": "podLive", "name": rd.pod_up.POD_NAME,
                                         "desiredStatus": "RUNNING", "runtime": {"x": 1}}]
        s3 = PodSession()
        rd.adopt_running_on_startup(s3)
        check("PODLINK_ADOPT_ON_START=0 disables adoption", s3.state == State.IDLE)
    finally:
        os.environ.pop("PODLINK_ADOPT_ON_START", None)
        fake_runpod.get_pods = saved_pods
        rd.pod_up.STATE_PATH = saved_state
        scratch.unlink(missing_ok=True)


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
    msgs = [m for _, _c, m in s.events]
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
    msgs = [m for _, _c, m in s.events]
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
              any("no host with capacity" in m for _, _c, m in s.events))
    finally:
        (rd.pod_up.create_pod_once, rd.pod_up.ensure_template,
         rd.pod_up.resolve_gpu_id, rd._sleep_or_cancel) = saved


def test_stack_probes_all_three_and_detects_dim():
    # test_stack POSTs a real completion/embedding/rerank; success records pass +
    # latency and detects the embedding dimension from the vector length.
    def responder(url, body):
        if "chat/completions" in url:
            return (200, {"choices": [{"message": {"content": "pong"}}]})
        if "embeddings" in url:
            return (200, {"data": [{"embedding": [0.0] * 4096}]})
        if "rerank" in url:
            return (200, [{"index": 0, "score": 0.91}, {"index": 1, "score": 0.02}])
        return (404, None)
    install_fake_post(responder)
    s = PodSession(); s.try_begin_start(None); s.state = State.RUNNING; s.pod_id = "podABC"
    rd.test_stack(s)
    tr = s.test_result
    check("stack test marks all_ok", tr["all_ok"] is True)
    check("stack test detects embedding dim (4096)", tr["embedding_dim"] == 4096)
    check("stack test parses reranker top score", "0.91" in tr["services"]["reranker"]["detail"])
    check("stack test clears test_running", s.test_running is False)


def test_stack_flags_service_failure():
    def responder(url, body):
        if "embeddings" in url:
            return (503, None)                       # embedder down
        if "chat/completions" in url:
            return (200, {"choices": [{}]})
        if "rerank" in url:
            return (200, [{"index": 0, "score": 0.5}])
        return (404, None)
    install_fake_post(responder)
    s = PodSession(); s.try_begin_start(None); s.state = State.RUNNING; s.pod_id = "podABC"
    rd.test_stack(s)
    check("failing embedder marked not-ok", s.test_result["services"]["embedder"]["ok"] is False)
    check("all_ok False when a service fails", s.test_result["all_ok"] is False)


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
    test_gql_escape_env_makes_json_values_safe()
    test_create_pod_once_sends_escaped_env()
    test_resolve_gpu_id_matches_rtx_pro_6000()
    test_resolve_gpu_id_raises_when_absent()
    test_all_ready_true_when_all_200()
    test_all_ready_waits_for_stragglers_then_times_out()
    test_all_ready_warns_and_keeps_waiting_past_the_soft_deadline()
    test_all_ready_raises_when_the_pod_itself_leaves_running()
    test_recover_if_healthy_flips_error_to_running()
    test_adopt_running_on_startup_adopts_our_running_pod()
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
    test_stack_probes_all_three_and_detects_dim()
    test_stack_flags_service_failure()
    test_create_non_retryable_error_surfaces()
    print("all driver smoke tests passed.")
