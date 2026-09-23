"""Offline tests for the stop lifecycle (PODLINK_LIFECYCLE=stop) — no RunPod calls.

Reuses the fake RunPod SDK that tests/test_driver_smoke.py installs at import,
and drives it with a small stateful pod store so stop / resume / terminate /
create act on the same records a real account would. What these pin down:

  * the default lifecycle is unchanged (terminate);
  * POD DOWN under stop stops (keeps the pod and pod_state.json) and verifies;
  * POD UP resumes a stopped pod, retrying ONLY "not enough free GPUs on the
    host", then terminates it and creates fresh when the retries run out;
  * a non-capacity resume error, a stack-fingerprint mismatch, or a resume
    that comes back with 0 GPUs never leaves the user on a stale/useless pod;
  * POD DOWN cancels mid-retry; "Terminate instead" works from IDLE; a failed
    stop falls back to terminate (never leaves a pod billing);
  * matches_stack ignores secrets but catches config drift on either side;
  * preflight's lifecycle rows.

Run: python3 tests/test_lifecycle.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_driver_smoke as T                      # noqa: E402  installs the fake SDK + imports app

rd, rp, fake = T.rd, T.rp, T.fake_runpod
PodSession, State, check = T.PodSession, T.State, T.check
QueryError = T.QueryError
NO_GPU = "There are not enough free GPUs on the host machine to start this pod."
SCRATCH = Path("/tmp/podlink_lifecycle_state.json")


class Store:
    """A tiny stateful RunPod account: pods keyed by id, mutated by the fakes."""

    def __init__(self):
        self.pods: dict[str, dict] = {}
        self.calls: list[tuple[str, str]] = []
        self.resume_script: list = []              # per-call outcome: "ok" | "zero" | Exception
        self.stop_raises: Exception | None = None
        self.created = 0

    def add(self, pod_id, status, **extra):
        # Env as RunPod really stores it (seen live 2026-09-23): empty values
        # dropped, and the account's PUBLIC_KEY added.
        pod = {"id": pod_id, "name": rp.pod_up.POD_NAME, "desiredStatus": status, "gpuCount": 1,
               "imageName": rp.pod_up.IMAGE, "env": _runpod_stored_env(), "costPerHr": 2.09, **extra}
        self.pods[pod_id] = pod
        return pod

    # --- SDK fakes ---
    def get_pods(self):
        return list(self.pods.values())

    def get_pod(self, pod_id):
        pod = self.pods.get(pod_id)
        if pod is None:
            return None
        out = dict(pod)
        if pod["desiredStatus"] == "RUNNING":
            out["runtime"] = {"uptimeInSeconds": 5}
        return out

    def resume_pod(self, pod_id, **kw):
        self.calls.append(("resume", pod_id))
        outcome = self.resume_script.pop(0) if self.resume_script else "ok"
        if isinstance(outcome, Exception):
            raise outcome
        self.pods[pod_id]["desiredStatus"] = "RUNNING"
        self.pods[pod_id]["gpuCount"] = 0 if outcome == "zero" else 1

    def stop_pod(self, pod_id):
        self.calls.append(("stop", pod_id))
        if self.stop_raises:
            raise self.stop_raises
        self.pods[pod_id]["desiredStatus"] = "EXITED"

    def terminate_pod(self, pod_id):
        self.calls.append(("terminate", pod_id))
        self.pods.pop(pod_id, None)

    def create_pod_once(self, *a, **k):
        self.created += 1
        pod = self.add(f"fresh{self.created}", "RUNNING")
        self.calls.append(("create", pod["id"]))
        return pod

    def verbs(self):
        return [v for v, _ in self.calls]


def _pod_env_list(**overrides) -> list[str]:
    """The env a create would send now, as the API returns it (["K=V", ...])."""
    # The live secrets the fake reader returns — a stopped pod created now
    # carries exactly these (matches_stack compares their digests).
    env = {**rp.pod_up._pod_env(T.fake_secrets.bearer_token(), T.fake_secrets.hf_token()), **overrides}
    return [f"{k}={v}" for k, v in env.items()]


def _runpod_stored_env(**overrides) -> list[str]:
    """_pod_env_list as RunPod echoes it back: empty values gone, PUBLIC_KEY added."""
    return [e for e in _pod_env_list(**overrides) if not e.endswith("=")] + ["PUBLIC_KEY=ssh-ed25519 AAAA… me"]


_SAVED = {}


def setup(lifecycle="stop", retries="3"):
    """Install a fresh store + env for one test; returns the store."""
    store = Store()
    for name in ("get_pods", "get_pod", "resume_pod", "stop_pod", "terminate_pod"):
        _SAVED.setdefault(name, getattr(fake, name))
        setattr(fake, name, getattr(store, name))
    for name in ("create_pod_once", "ensure_template", "resolve_gpu_id", "STATE_PATH"):
        _SAVED.setdefault("pod_up." + name, getattr(rp.pod_up, name))
    rp.pod_up.create_pod_once = store.create_pod_once
    rp.pod_up.ensure_template = lambda: "tmpl1"
    rp.pod_up.resolve_gpu_id = lambda: "gpuX"
    rp.pod_up.STATE_PATH = SCRATCH
    _SAVED.setdefault("delay", rd._resume_retry_delay)
    rd._resume_retry_delay = lambda: 0              # no real sleeps between attempts
    _SAVED.setdefault("poll", rd.STOP_VERIFY_POLL_S)
    rd.STOP_VERIFY_POLL_S = 0.01
    _SAVED.setdefault("gpu_poll", rd.GPU_SETTLE_POLL_S)
    rd.GPU_SETTLE_POLL_S = 0                        # no real sleeps while re-reading the GPU count
    os.environ["PODLINK_LIFECYCLE"] = lifecycle
    os.environ["PODLINK_RESUME_RETRIES"] = retries
    T._healthy_pod_fixture(SCRATCH)                 # all three services answer 200
    rp.pod_up.STATE_PATH = SCRATCH
    SCRATCH.unlink(missing_ok=True)
    return store


def teardown():
    for key, val in _SAVED.items():
        if key.startswith("pod_up."):
            setattr(rp.pod_up, key[7:], val)
        elif key == "delay":
            rd._resume_retry_delay = val
        elif key == "poll":
            rd.STOP_VERIFY_POLL_S = val
        elif key == "gpu_poll":
            rd.GPU_SETTLE_POLL_S = val
        else:
            setattr(fake, key, val)
    _SAVED.clear()
    for var in ("PODLINK_LIFECYCLE", "PODLINK_RESUME_RETRIES"):
        os.environ.pop(var, None)
    SCRATCH.unlink(missing_ok=True)


def _start() -> PodSession:
    s = PodSession()
    s.try_begin_start(None)                         # Auto
    rd.start(s)
    return s


def _events(s: PodSession) -> str:
    return "\n".join(e["msg"] for e in s.snapshot()["events"])


# --- tests ---------------------------------------------------------------------


def test_default_lifecycle_is_terminate():
    setup(lifecycle="")
    try:
        check("default lifecycle = terminate", rd.lifecycle() == "terminate")
        os.environ["PODLINK_LIFECYCLE"] = "bogus"
        check("unknown value -> terminate", rd.lifecycle() == "terminate")
    finally:
        teardown()


def test_down_stops_and_keeps_state():
    store = setup()
    try:
        store.add("pod1", "RUNNING")
        SCRATCH.write_text('{"pod_id": "pod1"}')
        s = PodSession(); s.try_begin_start(None); s.pod_id = "pod1"; s.state = State.RUNNING
        s.try_begin_stop()
        rd.stop(s)
        snap = s.snapshot()
        check("stop lifecycle: stop_pod called, not terminate", store.verbs() == ["stop"])
        check("stop lifecycle: lands IDLE", snap["state"] == "IDLE")
        check("stop lifecycle: phase says POD UP resumes", "POD UP resumes" in snap["phase"])
        check("stop lifecycle: pod kept (EXITED)", store.pods["pod1"]["desiredStatus"] == "EXITED")
        check("stop lifecycle: pod_state.json kept", SCRATCH.exists())
    finally:
        teardown()


def test_up_resumes_first_try():
    store = setup()
    try:
        store.add("pod1", "EXITED")
        s = _start()
        check("resume: RUNNING", s.snapshot()["state"] == "RUNNING")
        check("resume: same pod, no create", s.pod_id == "pod1" and store.created == 0)
        check("resume: one resume call", store.verbs() == ["resume"])
        check("resume: feed says no image pull", "no image pull" in _events(s))
    finally:
        teardown()


def test_up_resumes_after_capacity_failures():
    store = setup(retries="5")
    try:
        store.add("pod1", "EXITED")
        store.resume_script = [QueryError(NO_GPU), QueryError(NO_GPU), "ok"]
        s = _start()
        check("retry: RUNNING on the original pod", s.snapshot()["state"] == "RUNNING" and s.pod_id == "pod1")
        check("retry: three resume calls, no create", store.verbs() == ["resume"] * 3 and store.created == 0)
        check("retry: capacity notes in the feed", _events(s).count("host's GPU is taken") == 2)
    finally:
        teardown()


def test_up_falls_back_when_retries_run_out():
    store = setup(retries="3")
    try:
        store.add("pod1", "EXITED")
        store.resume_script = [QueryError(NO_GPU)] * 3
        s = _start()
        check("exhausted: RUNNING on a fresh pod", s.snapshot()["state"] == "RUNNING" and s.pod_id == "fresh1")
        check("exhausted: 3 resumes, terminate, create",
              store.verbs() == ["resume", "resume", "resume", "terminate", "create"])
        check("exhausted: stopped pod gone", "pod1" not in store.pods)
        check("exhausted: state file rewritten for the fresh pod",
              '"fresh1"' in SCRATCH.read_text() and '"pod1"' not in SCRATCH.read_text())
        check("exhausted: feed explains the fallback", "no free GPU after 3 attempts" in _events(s))
    finally:
        teardown()


def test_non_capacity_error_falls_back_at_once():
    store = setup(retries="40")
    try:
        store.add("pod1", "EXITED")
        store.resume_script = [QueryError("pod is being migrated")]
        s = _start()
        check("permanent error: one resume only, then terminate + create",
              store.verbs() == ["resume", "terminate", "create"])
        check("permanent error: RUNNING on fresh pod", s.pod_id == "fresh1")
        check("permanent error: str(e) not leaked to the feed", "migrated" not in _events(s))
    finally:
        teardown()


def test_zero_gpu_resume_is_stopped_and_retried():
    store = setup(retries="3")
    try:
        store.add("pod1", "EXITED")
        store.resume_script = ["zero", "ok"]
        s = _start()
        check("0 GPUs: stopped again, then resumed", store.verbs() == ["resume", "stop", "resume"])
        check("0 GPUs: RUNNING on the original pod", s.pod_id == "pod1" and s.snapshot()["state"] == "RUNNING")
        check("0 GPUs: feed says so", "0 GPUs" in _events(s))
    finally:
        teardown()


def test_stack_mismatch_recreates_without_resuming():
    store = setup()
    try:
        store.add("pod1", "EXITED", imageName="ghcr.io/x/old:1")
        s = _start()
        check("mismatch: never resumed", "resume" not in store.verbs())
        check("mismatch: terminate + create", store.verbs() == ["terminate", "create"])
        check("mismatch: feed names the reason", "stack changed" in _events(s))
        check("mismatch: feed names the differing setting, not its value",
              "(image)" in _events(s) and "old:1" not in _events(s))

    finally:
        teardown()


def test_rotated_bearer_recreates_and_never_logs_the_secret():
    store = setup()
    try:
        store.add("pod1", "EXITED", env=_runpod_stored_env(VLLM_API_KEY="leaked-key", TEI_API_KEY="leaked-key"))
        s = _start()
        check("rotation: never resumed", "resume" not in store.verbs())
        check("rotation: terminate + create", store.verbs() == ["terminate", "create"])
        check("rotation: feed names 'bearer'", "(bearer)" in _events(s))
        check("rotation: no secret text in the feed",
              "leaked-key" not in _events(s) and T.fake_secrets.bearer_token() not in _events(s))
    finally:
        teardown()


def test_cancel_mid_retry():
    store = setup(retries="40")
    try:
        store.add("pod1", "EXITED")
        s = PodSession(); s.try_begin_start(None)

        def resume_then_cancel(pod_id, **kw):
            store.calls.append(("resume", pod_id))
            s.try_begin_stop()                      # POD DOWN pressed during the retry wait
            raise QueryError(NO_GPU)
        fake.resume_pod = resume_then_cancel
        rd.start(s)
        check("cancel: stopped after one attempt, no create", store.verbs() == ["resume"] and store.created == 0)
        check("cancel: session left to the stop worker", s.snapshot()["state"] == "STOPPING")
        rd.stop(s)                                  # the stop worker: pod already stopped
        check("cancel: stop worker settles IDLE without re-stopping",
              s.snapshot()["state"] == "IDLE" and store.verbs() == ["resume"])
    finally:
        teardown()


def test_terminate_instead_from_idle():
    store = setup()
    try:
        store.add("pod1", "EXITED")
        s = PodSession()
        check("plain POD DOWN not allowed from IDLE", s.try_begin_stop() is False)
        check("terminate-instead allowed from IDLE", s.try_begin_stop(terminate=True) is True)
        rd.stop(s)
        check("terminate-instead: stopped pod terminated", store.verbs() == ["terminate"])
        check("terminate-instead: IDLE, phase says terminated",
              s.snapshot()["state"] == "IDLE" and "terminated" in s.snapshot()["phase"])
    finally:
        teardown()


def test_failed_stop_terminates_instead():
    store = setup()
    try:
        store.add("pod1", "RUNNING")
        store.stop_raises = QueryError("cannot stop")
        s = PodSession(); s.try_begin_start(None); s.pod_id = "pod1"; s.state = State.RUNNING
        s.try_begin_stop()
        rd.stop(s)
        check("failed stop: terminate issued", store.verbs() == ["stop", "terminate"])
        check("failed stop: IDLE, no pod billing", s.snapshot()["state"] == "IDLE" and "pod1" not in store.pods)
    finally:
        teardown()


def test_terminate_lifecycle_never_resumes():
    store = setup(lifecycle="terminate")
    try:
        store.add("pod1", "EXITED")
        s = _start()
        check("terminate lifecycle: EXITED leftover ignored, fresh create",
              "resume" not in store.verbs() and s.pod_id == "fresh1")
    finally:
        teardown()


def test_matches_stack_rules():
    setup()
    try:
        p = rp.PROVIDER()
        base = {"imageName": rp.pod_up.IMAGE, "env": _pod_env_list()}
        check("fingerprint: identical config matches", p.matches_stack(base))
        # A rotated secret is a change (a resumed pod would keep the old key and
        # 401 every request) — reported by LABEL only, never value or digest.
        rotated = p.stack_diff({**base, "env": _pod_env_list(VLLM_API_KEY="old-leaked", TEI_API_KEY="old-leaked")})
        check("fingerprint: rotated bearer -> ['bearer']", rotated == ["bearer"])
        check("fingerprint: rotated HF token -> ['hf_token']",
              p.stack_diff({**base, "env": _pod_env_list(HF_TOKEN="old", HUGGING_FACE_HUB_TOKEN="old")}) == ["hf_token"])
        check("fingerprint: bearer on only one key still differs",
              p.stack_diff({**base, "env": _pod_env_list(TEI_API_KEY="old-leaked")}) == ["bearer"])
        # Review: a secret key MISSING from the pod (not just different) is a change…
        no_bearer = [e for e in _pod_env_list() if not e.startswith(("VLLM_API_KEY=", "TEI_API_KEY="))]
        check("fingerprint: bearer key missing on the pod -> ['bearer']",
              p.stack_diff({**base, "env": no_bearer}) == ["bearer"])
        # …while a local secret that can't be read is skipped, not reported.
        from app.providers.runpod import provider as rpp
        saved_checks = rpp._SECRET_CHECKS

        def unreadable():
            raise SystemExit("Missing secret: pod_bearer_token")   # what _secrets does
        rpp._SECRET_CHECKS = (("bearer", unreadable, ("VLLM_API_KEY", "TEI_API_KEY")),) + saved_checks[1:]
        try:
            check("fingerprint: unreadable local bearer is skipped (no label, no crash)",
                  p.stack_diff({**base, "env": _pod_env_list(VLLM_API_KEY="x", TEI_API_KEY="x")}) == [])
        finally:
            rpp._SECRET_CHECKS = saved_checks
        check("fingerprint: diff never carries secret text",
              not any("old-leaked" in d or T.fake_secrets.bearer_token() in d for d in rotated))
        check("fingerprint: image drift -> mismatch", not p.matches_stack({**base, "imageName": "other:1"}))
        check("fingerprint: model drift -> mismatch",
              not p.matches_stack({**base, "env": _pod_env_list(RERANK_MODEL_ID="other/model")}))
        check("fingerprint: extra key on the pod -> mismatch",
              not p.matches_stack({**base, "env": _pod_env_list() + ["RERANK_BACKEND=vllm"]}))
        # Seen live 2026-09-23: RunPod drops empty values (LLM_QUANT="") and adds
        # its own keys (PUBLIC_KEY) — neither may count as a stack change.
        runpod_view = [e for e in _pod_env_list() if e != "LLM_QUANT="] + ["PUBLIC_KEY=ssh-ed25519 AAAA… me"]
        check("fingerprint: RunPod's dropped-empty + added PUBLIC_KEY still match",
              p.matches_stack({**base, "env": runpod_view}) and p.stack_diff({**base, "env": runpod_view}) == [])
        check("fingerprint: a real quant change still mismatches",
              p.stack_diff({**base, "env": _pod_env_list(LLM_QUANT="awq_marlin")}) == ["LLM_QUANT"])
        check("fingerprint: dict-shaped env accepted",
              p.matches_stack({**base, "env": dict(e.split("=", 1) for e in _pod_env_list())}))
        check("retryable only for the no-free-GPU error",
              p.is_retryable_resume_error(QueryError(NO_GPU))
              and not p.is_retryable_resume_error(QueryError("pod not found"))
              and not p.is_retryable_resume_error(RuntimeError(NO_GPU)))
    finally:
        teardown()


def test_preflight_lifecycle_rows():
    from app import preflight
    setup()
    try:
        prov = rp.PROVIDER()
        levels = lambda: [lvl for lvl, _ in preflight._lifecycle_rows(prov)]  # noqa: E731
        os.environ["PODLINK_LIFECYCLE"] = "terminate"
        check("preflight: terminate ok", levels() == ["ok"])
        os.environ["PODLINK_LIFECYCLE"] = "pause"
        check("preflight: bad value fails", levels() == ["fail"])
        os.environ["PODLINK_LIFECYCLE"] = "stop"
        saved_vol = rp.pod_up.NETWORK_VOLUME_ID
        rp.pod_up.NETWORK_VOLUME_ID = "vol1"
        check("preflight: stop + volume ok", levels() == ["ok"])
        rp.pod_up.NETWORK_VOLUME_ID = ""
        check("preflight: stop without volume warns", levels() == ["ok", "warn"])
        rp.pod_up.NETWORK_VOLUME_ID = saved_vol

        class NoStop:
            name, supports_stop = "other", False
        check("preflight: stop on a provider without it fails",
              [lvl for lvl, _ in preflight._lifecycle_rows(NoStop())] == ["fail"])
    finally:
        teardown()


def test_pod_down_during_successful_resume_stops_the_pod():
    # Review HIGH: resume() succeeds while POD DOWN arrives. The stop worker can
    # see the pod still EXITED (the cloud flips status async) and settle IDLE, so
    # the start worker must stop what it just resumed — never leave it billing.
    store = setup()
    try:
        store.add("pod1", "EXITED")
        s = PodSession(); s.try_begin_start(None)
        real_resume = store.resume_pod

        def resume_while_down_pressed(pod_id, **kw):
            real_resume(pod_id, **kw)               # resume is accepted: pod RUNNING
            s.try_begin_stop()                      # ... and POD DOWN lands in the same window
        fake.resume_pod = resume_while_down_pressed
        rd.start(s)
        check("race: start worker stopped the resumed pod", store.verbs() == ["resume", "stop"])
        check("race: pod ends EXITED (no GPU billing)", store.pods["pod1"]["desiredStatus"] == "EXITED")
        check("race: feed explains it", "POD DOWN during resume" in _events(s))
        rd.stop(s)                                  # the stop worker then settles cleanly
        check("race: console IDLE, pod still stopped",
              s.snapshot()["state"] == "IDLE" and store.pods["pod1"]["desiredStatus"] == "EXITED")
    finally:
        teardown()


def test_transient_zero_gpu_count_is_not_a_failed_resume():
    # The GPU count can lag a resume: 0 on the first read, 1 shortly after. That
    # must not be treated as a 0-GPU resume (which would stop it again).
    store = setup()
    try:
        store.add("pod1", "EXITED")
        reads = {"n": 0}
        real_get = store.get_pod

        def lagging_get(pod_id):
            pod = real_get(pod_id)
            if pod and pod["desiredStatus"] == "RUNNING" and reads["n"] < 2:
                reads["n"] += 1
                pod["gpuCount"] = 0                 # not attached yet
            return pod
        fake.get_pod = lagging_get
        s = _start()
        check("lag: no re-stop, one resume", store.verbs() == ["resume"])
        check("lag: RUNNING on the original pod", s.pod_id == "pod1" and s.snapshot()["state"] == "RUNNING")
    finally:
        teardown()


def test_terminate_from_idle_refuses_a_running_pod():
    # Review LOW: from IDLE the console isn't tracking any pod, so Terminate
    # instead may only remove a STOPPED one — a running same-name pod is left alone.
    store = setup()
    try:
        store.add("pod1", "RUNNING")
        s = PodSession()
        s.try_begin_stop(terminate=True)
        rd.stop(s)
        check("idle terminate: running pod untouched", store.verbs() == [] and "pod1" in store.pods)
        check("idle terminate: tells the user to adopt it first",
              s.snapshot()["state"] == "IDLE" and "POD UP to adopt" in s.snapshot()["phase"])
    finally:
        teardown()


def test_stop_flags_reset_on_start():
    s = PodSession()
    s.try_begin_stop(terminate=True)                # from IDLE: both flags set
    check("flags set by terminate-from-idle", s.force_terminate and s.stop_from_idle)
    s.state = State.IDLE
    s.try_begin_start(None)
    check("flags cleared by the next start", not s.force_terminate and not s.stop_from_idle)


def test_config_key_set_covers_everything_pod_env_can_send():
    # _CONFIG_ENV must list every non-secret key _pod_env can emit, or a changed
    # optional setting would be invisible to the fingerprint. Set every optional
    # passthrough (and a non-default reranker backend) and compare key sets.
    from app.providers.runpod import provider as rpp
    extra = {"PODLINK_VLLM_EXTRA_ARGS": "x", "PODLINK_PYTORCH_CUDA_ALLOC_CONF": "x",
             "PODLINK_RERANK_GPU_MEMORY_UTILIZATION": "0.1", "PODLINK_RERANK_VLLM_EXTRA_ARGS": "x",
             "PODLINK_RERANK_MAX_MODEL_LEN": "1", "PODLINK_RERANK_WAIT_FOR_EMBEDDER_S": "1",
             "PODLINK_VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0"}
    saved_backend = rp.pod_up.RERANK_BACKEND
    os.environ.update(extra)
    rp.pod_up.RERANK_BACKEND = "vllm"
    try:
        emitted = set(rp.pod_up._pod_env("b", "h")) - rpp._SECRET_ENV
        check("_CONFIG_ENV == every non-secret key _pod_env can send", emitted == set(rpp._CONFIG_ENV))
    finally:
        for k in extra:
            os.environ.pop(k, None)
        rp.pod_up.RERANK_BACKEND = saved_backend


if __name__ == "__main__":
    test_default_lifecycle_is_terminate()
    test_down_stops_and_keeps_state()
    test_up_resumes_first_try()
    test_up_resumes_after_capacity_failures()
    test_up_falls_back_when_retries_run_out()
    test_non_capacity_error_falls_back_at_once()
    test_zero_gpu_resume_is_stopped_and_retried()
    test_stack_mismatch_recreates_without_resuming()
    test_cancel_mid_retry()
    test_terminate_instead_from_idle()
    test_failed_stop_terminates_instead()
    test_terminate_lifecycle_never_resumes()
    test_matches_stack_rules()
    test_preflight_lifecycle_rows()
    test_pod_down_during_successful_resume_stops_the_pod()
    test_transient_zero_gpu_count_is_not_a_failed_resume()
    test_terminate_from_idle_refuses_a_running_pod()
    test_stop_flags_reset_on_start()
    test_config_key_set_covers_everything_pod_env_can_send()
    test_rotated_bearer_recreates_and_never_logs_the_secret()
    print("all lifecycle tests passed.")
