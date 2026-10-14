"""Stub-based smoke tests for the three-service driver logic.

The RunPod SDK isn't installed in CI/dev (see requirements.txt), and the live
API can't be hit here, so we inject a fake `runpod` + fake `_secrets` into
sys.modules and stub the audited egress client. That lets us exercise the two
riskiest pieces of new logic without a real pod:

  * resolve_gpu_id() — RTX Pro 6000 catalog lookup
  * _wait_for_all_ready() — the LLM + embedder + reranker readiness gate

The driver itself is provider-neutral, so these tests drive it through the
RunPod provider (`rp`) — which is what `providers.active()` returns unless
PODLINK_PROVIDER says otherwise.

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
    {"id": "NVIDIA GeForce RTX 3090", "displayName": "RTX 3090", "memoryInGb": 24},
    {"id": "NVIDIA GeForce RTX 3090 Ti", "displayName": "RTX 3090 Ti", "memoryInGb": 24},
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

# Now safe to import the driver (app.vendored bootstraps pod_control onto sys.path).
from app import driver as rd                        # noqa: E402  provider-neutral orchestration
from app import providers, vendored                 # noqa: E402  provider registry + secret reader
from app.providers import runpod as rp              # noqa: E402  the RunPod provider (owns pod_up)
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
    esc = rp.pod_up._gql_escape_env(raw)
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
        pod = rp.pod_up.create_pod_once("gpu-id", "bearer", "hf", "tmpl-id")
        check("create_pod received escaped env",
              pod["env"]["VLLM_EXTRA_ARGS"] == '--x {\\"a\\":1}')
    finally:
        del os.environ["PODLINK_VLLM_EXTRA_ARGS"]


def test_resolve_gpu_id_matches_rtx_pro_6000():
    gid = rp.pod_up.resolve_gpu_id()
    check("resolve_gpu_id -> RTX Pro 6000 id",
          gid == "NVIDIA RTX PRO 6000 Blackwell WE")


def test_resolve_gpu_id_honours_match_and_min_vram():
    # A profile can target another card: explicit args, and the module constants
    # re-baked from PODLINK_GPU_MATCH / _GPU_MIN_VRAM_GB on a provider reload.
    import importlib
    import os as _os
    check("explicit match + min vram -> the 24 GB card, plain SKU before Ti",
          rp.pod_up.resolve_gpu_id("RTX 3090", 24) == "NVIDIA GeForce RTX 3090")
    check("default min vram (90) rejects the 3090 even when matched",
          _raises(lambda: rp.pod_up.resolve_gpu_id("RTX 3090")))
    _os.environ.update({"PODLINK_GPU_MATCH": "RTX 3090", "PODLINK_GPU_MIN_VRAM_GB": "24",
                        "PODLINK_GPU_COUNT": "2"})
    try:
        importlib.reload(rp.pod_up)
        check("env re-bakes GPU_MATCH/GPU_MIN_VRAM_GB/GPU_COUNT",
              (rp.pod_up.GPU_MATCH, rp.pod_up.GPU_MIN_VRAM_GB, rp.pod_up.GPU_COUNT) == ("RTX 3090", 24, 2))
        check("resolve_gpu_id() reads the re-baked constants",
              rp.pod_up.resolve_gpu_id() == "NVIDIA GeForce RTX 3090")
        pod = rp.pod_up.create_pod_once("gpu-id", "bearer", "hf", "tmpl-id")
        check("create_pod asks for GPU_COUNT gpus", pod["gpu_count"] == 2)
        step = {"gpu_id": "g", "rung": rp.pod_up.ladder()[0], "of": 1}
        check("create phase names the gpu count",
              rp.PROVIDER().create_phase(step, 1, 3).startswith("2x RTX 3090 — creating pod on g"))
    finally:
        for k in ("PODLINK_GPU_MATCH", "PODLINK_GPU_MIN_VRAM_GB", "PODLINK_GPU_COUNT"):
            _os.environ.pop(k, None)
        importlib.reload(rp.pod_up)
    check("defaults restored: RTX Pro 6000, 90 GB, 1 gpu",
          (rp.pod_up.GPU_MATCH, rp.pod_up.GPU_MIN_VRAM_GB, rp.pod_up.GPU_COUNT) == ("RTX PRO 6000", 90, 1))
    pod = rp.pod_up.create_pod_once("gpu-id", "bearer", "hf", "tmpl-id")
    check("create_pod default gpu_count is 1", pod["gpu_count"] == 1)


def _raises(fn) -> bool:
    try:
        fn()
    except RuntimeError:
        return True
    return False


def test_gpu_pinning_passthrough_and_fingerprint():
    # PODLINK_{LLM,EMBED,RERANK}_GPUS reach the container as LLM_GPUS/EMBED_GPUS/
    # RERANK_GPUS (both the neutral contract and pod_up), only when set, and a
    # changed pinning is a stack-fingerprint mismatch (resume would boot the
    # wrong layout).
    import os as _os
    from app import stack
    from app.providers.runpod import provider as rpp
    base_env = stack.container_env(stack.from_env(), "B", "H")
    check("no pinning env when unset", not {"LLM_GPUS", "EMBED_GPUS", "RERANK_GPUS"} & set(base_env))
    _os.environ.update({"PODLINK_LLM_GPUS": "0", "PODLINK_EMBED_GPUS": "1", "PODLINK_RERANK_GPUS": "1"})
    try:
        env = stack.container_env(stack.from_env(), "B", "H")
        check("pinning sent", (env.get("LLM_GPUS"), env.get("EMBED_GPUS"), env.get("RERANK_GPUS")) == ("0", "1", "1"))
        check("pinning: container env keys == pod_up's", set(env) == set(rp.pod_up._pod_env("B", "H")))
        check("pinning keys are fingerprinted", {"LLM_GPUS", "EMBED_GPUS", "RERANK_GPUS"} <= rpp._CONFIG_ENV)
        have = rp.pod_up._pod_env("B", "H")
        inst = {"imageName": rp.pod_up.IMAGE, "env": [{"key": k, "value": v} for k, v in have.items()]}
        _os.environ["PODLINK_LLM_GPUS"] = "1"                      # switch the LLM to the other GPU
        check("changed pinning shows in the stack diff", "LLM_GPUS" in rp.PROVIDER().stack_diff(inst))
    finally:
        for k in ("PODLINK_LLM_GPUS", "PODLINK_EMBED_GPUS", "PODLINK_RERANK_GPUS"):
            _os.environ.pop(k, None)


def test_gpu_ladder_parsing_and_default():
    # Unset = one rung from the single-card constants; set = ordered rungs with
    # optional per-rung retries; malformed = ValueError naming the rung.
    import importlib
    import os as _os
    single = rp.pod_up.ladder()
    check("no ladder -> one rung from GPU_MATCH/COUNT/volume",
          len(single) == 1 and single[0].match == "RTX PRO 6000" and single[0].count == 1
          and single[0].volume_id == rp.pod_up.NETWORK_VOLUME_ID and single[0].retries is None)
    rungs = rp.pod_up.parse_ladder("RTX 3090|24|2|none|1 ; RTX PRO 4500|30|2|vol123")
    check("two rungs parsed in order",
          [(r.index, r.match, r.min_vram_gb, r.count, r.volume_id, r.retries) for r in rungs]
          == [(1, "RTX 3090", 24, 2, "", 1), (2, "RTX PRO 4500", 30, 2, "vol123", None)])
    check("rung labels", [r.label for r in rungs] == ["2x RTX 3090", "2x RTX PRO 4500"])
    for bad in ("RTX 3090|24|2", "RTX 3090|x|2|none", "|24|2|none", " ; ", "RTX 3090|24|0|none"):
        check(f"malformed ladder {bad!r} raises", _raises_value(lambda: rp.pod_up.parse_ladder(bad)))
    _os.environ["PODLINK_GPU_LADDER"] = "RTX 3090|24|2|none|1 ; RTX 5090|30|2|none"
    try:
        importlib.reload(rp.pod_up)
        check("env ladder re-baked", [r.match for r in rp.pod_up.ladder()] == ["RTX 3090", "RTX 5090"])
        check("stack_config reports the ladder",
              [r["label"] for r in rp.PROVIDER().stack_config()["ladder"]] == ["2x RTX 3090", "2x RTX 5090"])
        check("persistence: a volume-less rung means not configured", rp.PROVIDER().persistence_configured() is False)
    finally:
        _os.environ.pop("PODLINK_GPU_LADDER", None)
        importlib.reload(rp.pod_up)


def _raises_value(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def test_memory_budgets_derive_per_card_settings():
    # Absolute GB budgets -> exact KV bytes on any card, and per-card fractions
    # when the card's memory is known; nothing changes when unset.
    import os as _os
    from app import stack
    base = dict(stack.container_env(stack.from_env(), "B", "H"))
    _os.environ.update({"PODLINK_LLM_KV_GB": "4.5", "PODLINK_RERANK_MEMORY_GB": "2.9",
                        "PODLINK_LLM_MEMORY_GB": "22.8"})
    try:
        env24 = stack.container_env(stack.from_env(), "B", "H", card_memory_gb=24)
        env32 = stack.container_env(stack.from_env(), "B", "H", card_memory_gb=32)
        kv = f"--kv-cache-memory-bytes {int(4.5 * 2**30)}"
        check("kv bytes appended to the vLLM args", env24["VLLM_EXTRA_ARGS"].endswith(kv)
              and env32["VLLM_EXTRA_ARGS"].endswith(kv))
        check("llm fraction derived per card",
              (env24["GPU_MEMORY_UTILIZATION"], env32["GPU_MEMORY_UTILIZATION"]) == ("0.950", "0.713"))
        check("reranker fraction derived per card",
              (env24["RERANK_GPU_MEMORY_UTILIZATION"], env32["RERANK_GPU_MEMORY_UTILIZATION"]) == ("0.121", "0.091"))
        env_none = stack.container_env(stack.from_env(), "B", "H")
        check("unknown card: kv bytes still applied, fractions left alone",
              env_none["VLLM_EXTRA_ARGS"].endswith(kv) and env_none["GPU_MEMORY_UTILIZATION"] == base["GPU_MEMORY_UTILIZATION"])
        check("pod_up applies the same rule",
              rp.pod_up._pod_env("B", "H", 32)["GPU_MEMORY_UTILIZATION"] == "0.713"
              and rp.pod_up._pod_env("B", "H", 32)["VLLM_EXTRA_ARGS"].endswith(kv))
        check("budget keys are already fingerprinted",
              {"VLLM_EXTRA_ARGS", "GPU_MEMORY_UTILIZATION", "RERANK_GPU_MEMORY_UTILIZATION"}
              <= set(__import__("app.providers.runpod.provider", fromlist=["x"])._CONFIG_ENV))
    finally:
        for k in ("PODLINK_LLM_KV_GB", "PODLINK_RERANK_MEMORY_GB", "PODLINK_LLM_MEMORY_GB"):
            _os.environ.pop(k, None)
    check("unset budgets: env unchanged", stack.container_env(stack.from_env(), "B", "H", 32) == base)


def test_ladder_walk_probes_then_creates_on_the_first_rung_with_stock():
    # Rung 1 (1 attempt) probes empty -> skipped without a create; rung 2 probes
    # stock -> create (capacity error once, then success). Feed names the rung.
    import importlib
    import os as _os
    _os.environ["PODLINK_GPU_LADDER"] = "RTX 3090|24|2|none|1 ; RTX PRO 4500|30|2|none"
    saved = (fake_runpod._gpus, rp.pod_up.ensure_template, rp.pod_up.STATE_PATH, rd.egress_logger.client)
    fake_runpod._gpus = saved[0] + [{"id": "NVIDIA RTX PRO 4500 Blackwell", "displayName": "RTX PRO 4500", "memoryInGb": 32}]
    probes, creates = [], []
    scratch = Path("/tmp/podlink_ladder_state.json")
    try:
        importlib.reload(rp.pod_up)
        rp.pod_up.ensure_template = lambda: "tmpl1"
        rp.pod_up.STATE_PATH = scratch
        stock = {"NVIDIA GeForce RTX 3090": False, "NVIDIA RTX PRO 4500 Blackwell": True}
        rp.pod_up.probe_stock = lambda gpu_id, count, volume_id="", min_vram_gb=1: probes.append((gpu_id, count)) or stock[gpu_id]

        def fake_create(gpu_id, bearer, hf, template_id, rung=None, card_memory_gb=None):
            creates.append((gpu_id, rung.count, card_memory_gb))
            if len(creates) == 1:
                raise QueryError("no longer any instances available", "q")
            return {"id": "ladderpod", "gpu_count": rung.count}
        rp.pod_up.create_pod_once = fake_create
        saved_sleep = rd._sleep_or_cancel
        rd._sleep_or_cancel = lambda session, seconds: session.cancel.is_set()   # no real waits
        s = PodSession()
        s.try_begin_start(None)
        inst = rd._create_with_retries(s)
        check("created on rung 2", inst and inst["id"] == "ladderpod")
        check("rung 1 probed once, never created", probes[0] == ("NVIDIA GeForce RTX 3090", 2)
              and all(c[0] != "NVIDIA GeForce RTX 3090" for c in creates))
        check("rung 2 created with its count + card memory", creates[-1] == ("NVIDIA RTX PRO 4500 Blackwell", 2, 32))
        feed = " | ".join(m for _, _, m in s.events)
        check("feed: rung 1 empty, moved on", "rung 1/2: 2x RTX 3090: no Secure stock" in feed
              and "trying the next rung" in feed)
        check("feed: created on rung 2", "pod created on rung 2/2: 2x RTX PRO 4500" in feed)
        # Everything empty -> exhausted message names both rungs, no create calls.
        probes.clear(); creates.clear()
        stock["NVIDIA RTX PRO 4500 Blackwell"] = False
        _os.environ["PODLINK_CREATE_RETRIES"] = "2"
        s2 = PodSession()
        s2.try_begin_start(None)
        msg = ""
        try:
            rd._create_with_retries(s2)
        except RuntimeError as e:
            msg = str(e)
        check("all rungs empty -> exhausted, no create attempted",
              creates == [] and "2x RTX 3090 / 2x RTX PRO 4500" in msg)
        check("last rung used its retries", [p for p in probes if p[0].endswith("4500 Blackwell")].__len__() == 2)
    finally:
        _os.environ.pop("PODLINK_GPU_LADDER", None)
        _os.environ.pop("PODLINK_CREATE_RETRIES", None)
        fake_runpod._gpus = saved[0]
        importlib.reload(rp.pod_up)
        rp.pod_up.ensure_template, rp.pod_up.STATE_PATH = saved[1], saved[2]
        rd._sleep_or_cancel = saved_sleep
        scratch.unlink(missing_ok=True)


def test_community_cloud_knob():
    # PODLINK_CLOUD_TYPE=COMMUNITY: the create asks for Community hosts (with the
    # CUDA driver filter), probes answer "unknown" without touching the API (the
    # catalog publishes no Community stock), messages say Community, and the
    # preflight warns about the exposure + fails a rung that names a volume.
    import importlib
    import os as _os
    _os.environ.update({"PODLINK_CLOUD_TYPE": "community", "PODLINK_ALLOWED_CUDA_VERSIONS": "12.9, 13.0",
                        "PODLINK_GPU_LADDER": "RTX 3090|24|2|none ; RTX 3090|24|2|vol123"})
    saved_gql = sys.modules.get("runpod.api.graphql")
    try:
        importlib.reload(rp.pod_up)
        check("env re-bakes CLOUD_TYPE (upper-cased) + the CUDA filter",
              (rp.pod_up.CLOUD_TYPE, rp.pod_up.ALLOWED_CUDA_VERSIONS) == ("COMMUNITY", ["12.9", "13.0"]))
        pod = rp.pod_up.create_pod_once("gpu-id", "bearer", "hf", "tmpl-id")
        check("create_pod asks for COMMUNITY + the driver filter",
              pod["cloud_type"] == "COMMUNITY" and pod["allowed_cuda_versions"] == ["12.9", "13.0"])
        # No GraphQL module is installed in the fake SDK: a Secure probe would fail
        # closed to None inside its try, so prove the short-circuit by the absence
        # of an import attempt.
        gql = types.ModuleType("runpod.api.graphql")
        gql.run_graphql_query = lambda q: (_ for _ in ()).throw(AssertionError("Community probe hit the API"))
        sys.modules["runpod.api.graphql"] = gql
        check("Community probe is 'unknown' without an API call",
              rp.pod_up.probe_stock("NVIDIA GeForce RTX 3090", 2) is None)
        p = rp.PROVIDER()
        check("stock scope word", p.stock_scope == "Community")
        check("exhausted message says Community", p.create_exhausted_message(3, 15).startswith("no Community host"))
        rows = p.preflight()
        levels = {m: lvl for lvl, m in rows}
        check("preflight warns about Community exposure",
              any(lvl == "warn" and m.startswith("cloud: COMMUNITY") for lvl, m in rows))
        check("preflight lists the CUDA filter", any(m == "host CUDA filter: 12.9, 13.0" and lvl == "ok" for lvl, m in rows))
        check("a volume rung fails on Community",
              any(lvl == "fail" and "vol123" in m and "Community" in m for lvl, m in rows))
        check("no stock line on Community", not any(m.startswith("stock now") for m in levels))
        check("stack_config carries the cloud", p.stack_config()["cloud_type"] == "COMMUNITY")
        _os.environ["PODLINK_CLOUD_TYPE"] = "spot"
        bad = False
        try:
            importlib.reload(rp.pod_up)
        except ValueError:
            bad = True
        check("an unknown cloud type is rejected at import", bad)
    finally:
        for k in ("PODLINK_CLOUD_TYPE", "PODLINK_ALLOWED_CUDA_VERSIONS", "PODLINK_GPU_LADDER"):
            _os.environ.pop(k, None)
        if saved_gql is None:
            sys.modules.pop("runpod.api.graphql", None)
        else:
            sys.modules["runpod.api.graphql"] = saved_gql
        importlib.reload(rp.pod_up)
    check("defaults restored: Secure, no CUDA filter",
          (rp.pod_up.CLOUD_TYPE, rp.pod_up.ALLOWED_CUDA_VERSIONS) == ("SECURE", []))
    pod = rp.pod_up.create_pod_once("gpu-id", "bearer", "hf", "tmpl-id")
    check("Secure create carries no CUDA filter", pod["cloud_type"] == "SECURE" and "allowed_cuda_versions" not in pod)
    p = rp.PROVIDER()
    check("Secure preflight has the info row", ("info", "cloud: Secure") in p.preflight())


def test_ladder_fingerprint_is_rung_aware():
    # A stopped pod on rung 2 must be compared with rung 2's derived env (its
    # card's fractions), not rung 1's; a pod on no rung is a mismatch.
    import importlib
    import os as _os
    _os.environ["PODLINK_GPU_LADDER"] = "RTX 3090|24|2|none|1 ; RTX PRO 4500|30|2|none"
    _os.environ["PODLINK_RERANK_MEMORY_GB"] = "2.9"
    saved = fake_runpod._gpus
    fake_runpod._gpus = saved + [{"id": "NVIDIA RTX PRO 4500 Blackwell", "displayName": "RTX PRO 4500", "memoryInGb": 32}]
    try:
        importlib.reload(rp.pod_up)
        p = rp.PROVIDER()
        # The fake secrets' values, so the digest comparison sees them unchanged.
        env32 = rp.pod_up._pod_env("bearer_dummy", "hf_dummy", 32)
        inst = {"imageName": rp.pod_up.IMAGE, "gpuCount": 2,
                "machine": {"gpuTypeId": "NVIDIA RTX PRO 4500 Blackwell"},
                "env": [{"key": k, "value": v} for k, v in env32.items() if v]}
        check("rung-2 pod with rung-2 env matches", p.stack_diff(inst) == [])
        inst24 = dict(inst, env=[{"key": k, "value": v}
                                 for k, v in rp.pod_up._pod_env("bearer_dummy", "hf_dummy", 24).items() if v])
        check("rung-2 pod carrying 24 GB fractions is stale", "RERANK_GPU_MEMORY_UTILIZATION" in p.stack_diff(inst24))
        stray = dict(inst, machine={"gpuTypeId": "NVIDIA A100"})
        check("pod on no rung -> 'gpu' in the diff", "gpu" in p.stack_diff(stray))
    finally:
        _os.environ.pop("PODLINK_GPU_LADDER", None)
        _os.environ.pop("PODLINK_RERANK_MEMORY_GB", None)
        fake_runpod._gpus = saved
        importlib.reload(rp.pod_up)


def test_preflight_gpu_rows():
    # Pinned GPU indices and any --tensor-parallel-size must fit gpu_count; a
    # multi-GPU instance with neither gets a warning; no gpu_count = no rows.
    from app import preflight
    levels = lambda st: [lvl for lvl, _ in preflight._gpu_rows(st)]  # noqa: E731
    check("no gpu_count -> no rows", levels({"llm_gpus": "7"}) == [])
    check("single gpu, nothing pinned -> no rows", levels({"gpu_count": 1}) == [])
    check("pinned within range -> ok", levels({"gpu_count": 2, "llm_gpus": "0", "embed_gpus": "1"}) == ["ok", "ok"])
    check("index beyond the instance -> fail", levels({"gpu_count": 1, "embed_gpus": "1"}) == ["fail"])
    check("non-integer index -> fail", levels({"gpu_count": 2, "llm_gpus": "zero"}) == ["fail"])
    check("tp fits -> ok", levels({"gpu_count": 2, "vllm_extra_args": "--tensor-parallel-size 2 --x"}) == ["ok"])
    check("tp= form parsed", levels({"gpu_count": 2, "vllm_extra_args": "--tensor-parallel-size=2"}) == ["ok"])
    check("tp too big -> fail", levels({"gpu_count": 1, "vllm_extra_args": "--tensor-parallel-size 2"}) == ["fail"])
    check("multi-gpu, nothing pinned, no tp -> warn", levels({"gpu_count": 2}) == ["warn"])


def test_resolve_gpu_id_raises_when_absent():
    saved = fake_runpod._gpus
    fake_runpod._gpus = [{"id": "A100", "displayName": "A100 80GB"}]
    try:
        raised = False
        try:
            rp.pod_up.resolve_gpu_id()
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
    check("boot records each service and the all-healthy moment",
          {"healthy_llm", "healthy_embedder", "healthy_reranker", "services_healthy"} <= set(s.boot))


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
    saved_state = rp.pod_up.STATE_PATH
    scratch = Path("/tmp/podlink_smoke_recover_state.json")
    rp.pod_up.STATE_PATH = scratch
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
        check("a recovered start closes its boot timeline", s.snapshot()["boot_s"] is not None)
        snap = s.snapshot()
        check("recovery clears the error and sets the proxy url",
              snap["error"] is None and snap["proxy_url"] and "podLost" in snap["proxy_url"])
        check("recovery is logged to the feed",
              any("recovered" in m for (_, c, m) in s.events if c == "system"))
    finally:
        rp.pod_up.STATE_PATH = saved_state
        scratch.unlink(missing_ok=True)


def test_adopt_running_on_startup_adopts_our_running_pod():
    import os
    saved_pods, saved_state = fake_runpod.get_pods, rp.pod_up.STATE_PATH
    scratch = Path("/tmp/podlink_smoke_adopt_state.json")
    rp.pod_up.STATE_PATH = scratch
    install_fake_client(lambda url: 200)
    try:
        fake_runpod.get_pods = lambda: [{"id": "podLive", "name": rp.pod_up.POD_NAME,
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
        fake_runpod.get_pods = lambda: [{"id": "podLive", "name": rp.pod_up.POD_NAME,
                                         "desiredStatus": "RUNNING", "runtime": {"x": 1}}]
        s3 = PodSession()
        rd.adopt_running_on_startup(s3)
        check("PODLINK_ADOPT_ON_START=0 disables adoption", s3.state == State.IDLE)
    finally:
        os.environ.pop("PODLINK_ADOPT_ON_START", None)
        fake_runpod.get_pods = saved_pods
        rp.pod_up.STATE_PATH = saved_state
        scratch.unlink(missing_ok=True)


def test_pod_still_running_never_abandons_on_doubt():
    saved = fake_runpod.get_pod
    try:
        fake_runpod.get_pod = lambda pod_id: {"id": pod_id}              # no status at all
        check("a record with no status counts as still running", rd._pod_still_running("p") is True)
        def _boom(pod_id):
            raise RuntimeError("api blip")
        fake_runpod.get_pod = _boom
        check("a failed lookup counts as still running", rd._pod_still_running("p") is True)
        fake_runpod.get_pod = lambda pod_id: None
        check("a missing instance is not running", rd._pod_still_running("p") is False)
        fake_runpod.get_pod = lambda pod_id: {"id": pod_id, "desiredStatus": "EXITED"}
        check("a definite non-running status is not running", rd._pod_still_running("p") is False)
    finally:
        fake_runpod.get_pod = saved


def test_recover_settles_idle_when_the_instance_is_gone():
    rec = _AccessRecorder()
    saved = (fake_runpod.get_pod, rp.pod_up.STATE_PATH)
    scratch = Path("/tmp/podlink_smoke_recover_gone.json")
    scratch.write_text("{}")
    rp.pod_up.STATE_PATH = scratch
    fake_runpod.get_pod = lambda pod_id: None           # the instance no longer exists
    try:
        s = PodSession()
        s.try_begin_start(None)
        s.update(pod_id="podGone", state=State.ERROR, error="RuntimeError — gave up",
                 billing_started_at=1.0, cost_per_hr=2.0, auto_terminate_at=9e9)
        check("gone instance: not a recovery", rd.recover_if_healthy(s, "podGone") is False)
        snap = s.snapshot()
        check("gone instance: session settles to IDLE with the error cleared",
              snap["state"] == "IDLE" and snap["error"] is None and s.pod_id is None)
        check("gone instance: meter and idle timer stopped",
              snap["uptime_s"] is None and s.auto_terminate_at is None)
        check("gone instance: access released, never re-ensured",
              rec.released == 1 and rec.opened == [])
        check("gone instance: state file cleared", not scratch.exists())
        check("gone instance: explained in the feed",
              any("no longer running" in m for (_, c, m) in s.events if c == "system"))
    finally:
        fake_runpod.get_pod, rp.pod_up.STATE_PATH = saved
        scratch.unlink(missing_ok=True)
        rec.restore()


def test_recover_does_not_clobber_a_session_that_moved_on():
    saved = fake_runpod.get_pod
    fake_runpod.get_pod = lambda pod_id: None
    try:
        s = PodSession()
        s.try_begin_start(None)
        s.update(pod_id="podOld", state=State.ERROR, error="x")
        s.try_begin_start(None)                          # user pressed POD UP meanwhile
        rd.recover_if_healthy(s, "podOld")
        check("a new start is not settled to IDLE by a stale recovery",
              s.state == State.STARTING)
    finally:
        fake_runpod.get_pod = saved


def test_recover_authenticates_before_any_provider_call():
    p = providers.active()
    calls = []
    p.authenticate = lambda: calls.append("auth")
    saved = fake_runpod.get_pod
    fake_runpod.get_pod = lambda pod_id: (calls.append("get"), {"id": pod_id, "desiredStatus": "RUNNING",
                                                                "runtime": {"x": 1}})[1]
    install_fake_client(lambda url: 503)
    try:
        s = PodSession()
        s.try_begin_start(None)
        s.update(pod_id="podA", state=State.ERROR, error="x")
        rd.recover_if_healthy(s, "podA")
        check("recovery authenticates first", calls[:2] == ["auth", "get"])
        def _noauth():
            raise RuntimeError("no key")
        p.authenticate = _noauth
        check("no credentials: not recovered, no exception",
              rd.recover_if_healthy(s, "podA") is False and s.state == State.ERROR)
    finally:
        del p.authenticate
        fake_runpod.get_pod = saved


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
    check("_create_with_retries returns None when cancelled",
          rd._create_with_retries(s) is None)


def test_provider_selection():
    # The driver resolves its cloud from PODLINK_PROVIDER; the default is RunPod,
    # and an unknown name fails loudly at selection rather than mid-launch.
    import os as _os
    check("default provider is runpod", providers.active().name == "runpod")
    saved = _os.environ.get("PODLINK_PROVIDER")
    _os.environ["PODLINK_PROVIDER"] = "nope"
    try:
        raised = False
        try:
            providers.active()
        except ValueError:
            raised = True
        check("unknown PODLINK_PROVIDER raises", raised)
    finally:
        if saved is None:
            _os.environ.pop("PODLINK_PROVIDER", None)
        else:
            _os.environ["PODLINK_PROVIDER"] = saved
    check("provider restored after the bad name", providers.active().name == "runpod")


def test_provider_snapshot_fields_are_complete():
    # The UI renders these generically, so a provider that omits one silently
    # breaks the persistence banner or the POD DOWN guard.
    fields = providers.active().snapshot_fields()
    required = {"provider", "persistence_configured", "persistence_id",
                "persistence_label", "llm_model_id"}
    check("snapshot_fields carries every UI key", required <= set(fields))
    check("persistence_configured is a bool", isinstance(fields["persistence_configured"], bool))


class _AccessRecorder:
    """Record ensure_access / release_access calls on the live provider.

    Instance attributes shadow the class methods, so restore() deletes them and
    lets the real (no-op) RunPod implementations take over again.
    """

    def __init__(self):
        self.provider = providers.active()
        self.opened = []
        self.released = 0
        self.provider.ensure_access = lambda instance_id: self.opened.append(instance_id)
        self.provider.release_access = lambda: setattr(self, "released", self.released + 1)

    def restore(self):
        for attr in ("ensure_access", "release_access"):
            try:
                delattr(self.provider, attr)
            except AttributeError:
                pass


def _healthy_pod_fixture(scratch):
    """Fakes for a pod that is RUNNING with all three services answering 200."""
    install_fake_client(lambda url: 200)
    rp.pod_up.STATE_PATH = scratch          # never touch the real pod_state.json


def test_is_running_and_is_up_are_not_the_same_predicate():
    # These drive different decisions: is_running gates resume/adopt, is_up gates
    # the readiness wait. Collapsing them would make podlink call a pod 'up' the
    # instant RunPod says RUNNING — before the container actually exists.
    p = providers.active()
    booting = {"id": "podX", "desiredStatus": "RUNNING"}                    # no runtime yet
    serving = {"id": "podX", "desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 5}}
    exited = {"id": "podX", "desiredStatus": "EXITED"}
    check("is_running True while still booting", p.is_running(booting) is True)
    check("is_up False until a runtime exists", p.is_up(booting) is False)
    check("is_up True once a runtime exists", p.is_up(serving) is True)
    check("is_running False when EXITED", p.is_running(exited) is False)


def test_wait_for_running_ignores_running_without_a_runtime():
    # The driver must keep waiting on RUNNING-without-runtime, not treat it as up.
    saved = (fake_runpod.get_pod, rd.RUNNING_TIMEOUT_S)
    fake_runpod.get_pod = lambda pod_id: {"id": pod_id, "desiredStatus": "RUNNING"}
    rd.RUNNING_TIMEOUT_S = 0.3
    try:
        s = PodSession()
        s.try_begin_start(None)
        timed_out = False
        try:
            rd._wait_for_running(s, "podX")
        except RuntimeError:
            timed_out = True
        check("_wait_for_running keeps waiting without a runtime", timed_out)
    finally:
        fake_runpod.get_pod, rd.RUNNING_TIMEOUT_S = saved


def test_verify_terminated_keeps_polling_on_unknown_status():
    # An absent/empty status is NOT proof the GPU was released — it must keep
    # polling and ultimately report failure rather than claim a clean terminate.
    saved = (fake_runpod.get_pod, rd.STOP_VERIFY_TIMEOUT_S, rd.STOP_VERIFY_POLL_S)
    fake_runpod.get_pod = lambda pod_id: {"id": pod_id, "desiredStatus": None}
    rd.STOP_VERIFY_TIMEOUT_S, rd.STOP_VERIFY_POLL_S = 0.2, 0.01
    try:
        check("_verify_terminated False when the status is unknown",
              rd._verify_terminated(PodSession(), "podX") is False)
    finally:
        fake_runpod.get_pod, rd.STOP_VERIFY_TIMEOUT_S, rd.STOP_VERIFY_POLL_S = saved


def test_start_opens_access_on_the_adopt_path():
    # An ADOPTED instance is never created, so an access path opened inside
    # create_once() would never exist for it. This is the regression guard.
    rec = _AccessRecorder()
    saved = (rp.pod_up.find_existing, rp.pod_up.STATE_PATH)
    scratch = Path("/tmp/podlink_access_adopt.json")
    rp.pod_up.find_existing = lambda: {"id": "adopted1", "desiredStatus": "RUNNING",
                                       "runtime": {"uptimeInSeconds": 9}}
    _healthy_pod_fixture(scratch)
    try:
        s = PodSession()
        s.try_begin_start(None)                      # Auto — no explicit target
        rd.start(s)
        check("adopt path opened the access path", rec.opened == ["adopted1"])
        check("adopt path reached RUNNING", s.snapshot()["state"] == "RUNNING")
    finally:
        rp.pod_up.find_existing, rp.pod_up.STATE_PATH = saved
        scratch.unlink(missing_ok=True)
        rec.restore()


def test_start_opens_access_on_the_create_path():
    rec = _AccessRecorder()
    saved = (rp.pod_up.find_existing, rp.pod_up.create_pod_once, rp.pod_up.ensure_template,
             rp.pod_up.resolve_gpu_id, rp.pod_up.STATE_PATH)
    scratch = Path("/tmp/podlink_access_create.json")
    rp.pod_up.find_existing = lambda: None           # nothing to adopt -> create
    rp.pod_up.resolve_gpu_id = lambda *a, **k: "gpuX"
    rp.pod_up.ensure_template = lambda: "tmpl1"
    rp.pod_up.create_pod_once = lambda *a, **k: {"id": "created1"}
    _healthy_pod_fixture(scratch)
    try:
        s = PodSession()
        s.try_begin_start(None)
        rd.start(s)
        check("create path opened the access path", rec.opened == ["created1"])
    finally:
        (rp.pod_up.find_existing, rp.pod_up.create_pod_once, rp.pod_up.ensure_template,
         rp.pod_up.resolve_gpu_id, rp.pod_up.STATE_PATH) = saved
        scratch.unlink(missing_ok=True)
        rec.restore()


def test_start_and_recovery_hand_off_the_client_env():
    # Both "all healthy" paths call clientenv.handoff(session, provider, pod_id,
    # test_stack) after the session is RUNNING — the create path and the
    # ERROR->RUNNING recovery. The handoff itself is unit-tested in test_clientenv.
    calls = []
    saved_handoff = rd.clientenv.handoff
    rd.clientenv.handoff = lambda s, p, pod_id, ts: calls.append((pod_id, s.state, ts is rd.test_stack))
    saved = (rp.pod_up.find_existing, rp.pod_up.create_pod_once, rp.pod_up.ensure_template,
             rp.pod_up.resolve_gpu_id, rp.pod_up.STATE_PATH)
    scratch = Path("/tmp/podlink_handoff_state.json")
    rp.pod_up.find_existing = lambda: None
    rp.pod_up.resolve_gpu_id = lambda *a, **k: "gpuX"
    rp.pod_up.ensure_template = lambda: "tmpl1"
    rp.pod_up.create_pod_once = lambda *a, **k: {"id": "createdH"}
    _healthy_pod_fixture(scratch)
    try:
        s = PodSession()
        s.try_begin_start(None)
        rd.start(s)
        check("create path hands off once RUNNING, with the real stack test",
              calls == [("createdH", State.RUNNING, True)])
        s2 = PodSession()
        s2.try_begin_start(None)
        s2.update(pod_id="podLostH", state=State.ERROR, error="x")
        rd.recover_if_healthy(s2, "podLostH")
        check("recovery path hands off too", calls[-1] == ("podLostH", State.RUNNING, True))
    finally:
        rd.clientenv.handoff = saved_handoff
        (rp.pod_up.find_existing, rp.pod_up.create_pod_once, rp.pod_up.ensure_template,
         rp.pod_up.resolve_gpu_id, rp.pod_up.STATE_PATH) = saved
        scratch.unlink(missing_ok=True)


def test_failed_start_before_an_instance_closes_the_access_path():
    # No stop worker follows a failed start, so start() owns the teardown —
    # otherwise a tunnel process outlives the launch that opened it.
    rec = _AccessRecorder()
    saved = (rp.pod_up.find_existing, rp.pod_up.ensure_template)
    rp.pod_up.find_existing = lambda: None           # nothing to adopt -> prepare/create
    def _boom():
        raise RuntimeError("template lookup failed")
    rp.pod_up.ensure_template = _boom                # fails before any instance exists
    try:
        s = PodSession()
        s.try_begin_start(None)
        rd.start(s)
        check("failed start with no instance closed the access path", rec.released == 1)
        check("failed start lands in ERROR", s.snapshot()["state"] == "ERROR")
    finally:
        rp.pod_up.find_existing, rp.pod_up.ensure_template = saved
        rec.restore()


def test_failed_start_with_an_instance_keeps_the_access_path():
    # The instance outlived the failed start and is still billing: the health watch
    # keeps probing it to recover, so its access path (a tunnel, on some clouds)
    # must stay open until stop() releases it.
    rec = _AccessRecorder()
    saved = (rp.pod_up.find_existing, rd.READY_TIMEOUT_S)
    rp.pod_up.find_existing = lambda: {"id": "doomed1", "desiredStatus": "RUNNING",
                                       "runtime": {"uptimeInSeconds": 1}}
    install_fake_client(lambda url: 503)             # services never come up
    rd.READY_TIMEOUT_S = 0.2
    try:
        s = PodSession()
        s.try_begin_start(None)
        rd.start(s)
        check("failed start with an instance opened and kept the access path",
              rec.opened == ["doomed1"] and rec.released == 0)
        snap = s.snapshot()
        check("failed start lands in ERROR with the instance id kept",
              snap["state"] == "ERROR" and s.pod_id == "doomed1")
        check("the error says the instance may still be billing", "billing" in snap["error"])
    finally:
        rp.pod_up.find_existing, rd.READY_TIMEOUT_S = saved
        rec.restore()


def test_recover_if_healthy_reopens_the_access_path_before_probing():
    # Probes of a tunnelled service read "down" until the path exists, so recovery
    # must ensure it first — and a path that won't open is "not recovered yet".
    rec = _AccessRecorder()
    order = []
    rec.provider.ensure_access = lambda instance_id: order.append(("ensure", instance_id))
    install_fake_client(lambda url: (order.append(("probe", url)), 200)[1])
    saved_state = rp.pod_up.STATE_PATH
    scratch = Path("/tmp/podlink_smoke_recover_access.json")
    rp.pod_up.STATE_PATH = scratch
    try:
        s = PodSession()
        s.try_begin_start(None)
        s.update(pod_id="podT", state=State.ERROR, error="RuntimeError — gave up")
        check("recovery succeeds once the path is up and all answer",
              rd.recover_if_healthy(s, "podT") is True and s.state == State.RUNNING)
        check("ensure_access ran before the first probe",
              order and order[0] == ("ensure", "podT")
              and any(k == "probe" for k, _ in order[1:]))
        def _no_path(instance_id):
            raise RuntimeError("tunnel refused")
        rec.provider.ensure_access = _no_path
        s2 = PodSession()
        s2.try_begin_start(None)
        s2.update(pod_id="podT", state=State.ERROR, error="x")
        check("access path down: stays ERROR, no exception",
              rd.recover_if_healthy(s2, "podT") is False and s2.state == State.ERROR)
    finally:
        rp.pod_up.STATE_PATH = saved_state
        scratch.unlink(missing_ok=True)
        rec.restore()


def test_stop_closes_the_access_path():
    rec = _AccessRecorder()
    saved = (fake_runpod.get_pod, rp.pod_up.STATE_PATH)
    scratch = Path("/tmp/podlink_access_stop.json")
    fake_runpod.get_pod = lambda pod_id: {"id": pod_id, "desiredStatus": "EXITED"}
    rp.pod_up.STATE_PATH = scratch
    try:
        s = PodSession()
        s.try_begin_start(None)
        s.pod_id = "podX"
        s.try_begin_stop()
        rd.stop(s)
        check("stop closed the access path", rec.released == 1)
        check("stop still landed IDLE", s.snapshot()["state"] == "IDLE")
    finally:
        fake_runpod.get_pod, rp.pod_up.STATE_PATH = saved
        scratch.unlink(missing_ok=True)
        rec.restore()


def test_stack_defaults_agree_with_pod_up():
    # app/stack.py is the cloud-neutral copy of the image's env contract; the
    # vendored pod_up.py keeps its own (minimal-edit policy). They must not drift.
    from app import stack
    import os as _os
    for var in ("PODLINK_LLM_MODEL_ID", "PODLINK_EMBED_MODEL_ID", "PODLINK_RERANK_MODEL_ID",
                "PODLINK_LLM_SERVED_NAME", "PODLINK_MAX_MODEL_LEN", "PODLINK_GPU_MEMORY_UTILIZATION"):
        assert var not in _os.environ, f"{var} set in the test env — defaults not observable"
    check("stack defaults == pod_up defaults", (
        stack.DEFAULT_LLM_MODEL_ID == rp.pod_up.LLM_MODEL_ID
        and stack.DEFAULT_EMBED_MODEL_ID == rp.pod_up.EMBED_MODEL_ID
        and stack.DEFAULT_RERANK_MODEL_ID == rp.pod_up.RERANK_MODEL_ID
        and stack.DEFAULT_LLM_SERVED_NAME == rp.pod_up.LLM_SERVED_NAME
        and stack.DEFAULT_MAX_MODEL_LEN == rp.pod_up.MAX_MODEL_LEN
        and stack.DEFAULT_GPU_MEMORY_UTILIZATION == rp.pod_up.GPU_MEMORY_UTILIZATION))
    check("stack ports == pod_up ports", stack.SERVICE_PORTS == rp.pod_up.SERVICE_PORTS)
    env = stack.container_env(stack.from_env(), "B", "H")
    check("container env keys == pod_up's", set(env) == set(rp.pod_up._pod_env("B", "H")))


def test_stop_terminates_and_lands_idle():
    # POD DOWN calls runpod.terminate_pod (not stop_pod), verifies the pod left
    # RUNNING, clears the state file, and settles to IDLE. Covers the refactor.
    terminated = {}
    saved_term = fake_runpod.terminate_pod
    saved_get = fake_runpod.get_pod
    saved_state = rp.pod_up.STATE_PATH
    scratch = Path("/tmp/podlink_smoke_state.json")     # never touch the real pod_state.json
    scratch.write_text('{"pod_id": "podX"}')            # something for _clear_state_file to remove
    fake_runpod.terminate_pod = lambda pod_id: terminated.__setitem__("id", pod_id)
    fake_runpod.get_pod = lambda pod_id: {"id": pod_id, "desiredStatus": "EXITED"}  # verify passes at once
    rp.pod_up.STATE_PATH = scratch
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
        rp.pod_up.STATE_PATH = saved_state
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
        vendored.read_secret(missing)
    except RuntimeError:
        outcome = "RuntimeError"
    except SystemExit:
        outcome = "SystemExit"
    check("read_secret converts SystemExit -> RuntimeError", outcome == "RuntimeError")


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
    saved = (rp.pod_up.create_pod_once, rp.pod_up.ensure_template,
             rp.pod_up.resolve_gpu_id, rd._sleep_or_cancel)
    rp.pod_up.create_pod_once = flaky
    rp.pod_up.ensure_template = lambda: "tmpl1"
    rp.pod_up.resolve_gpu_id = lambda *a, **k: "gpuX"
    rd._sleep_or_cancel = lambda session, secs: False   # instant, not cancelled
    try:
        s = PodSession(); s.try_begin_start(None)
        pod = rd._create_with_retries(s)
        check("retries then succeeds on the 3rd attempt", pod == {"id": "pod-ok"} and calls["n"] == 3)
        check("retry attempts logged to the event feed",
              any("no host with capacity" in m for _, _c, m in s.events))
    finally:
        (rp.pod_up.create_pod_once, rp.pod_up.ensure_template,
         rp.pod_up.resolve_gpu_id, rd._sleep_or_cancel) = saved


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
    saved = (rp.pod_up.create_pod_once, rp.pod_up.ensure_template, rp.pod_up.resolve_gpu_id)
    rp.pod_up.create_pod_once = lambda *a, **k: (_ for _ in ()).throw(QE("invalid gpu spec"))
    rp.pod_up.ensure_template = lambda: "t"
    rp.pod_up.resolve_gpu_id = lambda *a, **k: "g"
    try:
        s = PodSession(); s.try_begin_start(None)
        raised = False
        try:
            rd._create_with_retries(s)
        except QE:
            raised = True
        check("non-retryable create error surfaces (not retried)", raised)
    finally:
        (rp.pod_up.create_pod_once, rp.pod_up.ensure_template, rp.pod_up.resolve_gpu_id) = saved


# --- reranker backend (tei | vllm on :8081) ---------------------------------


def test_rerank_backend_default_agrees_with_pod_up():
    # The tei default is duplicated in pod_up (minimal-edit vendored copy); pin it.
    from app import stack
    import os as _os
    assert "PODLINK_RERANK_BACKEND" not in _os.environ, "backend set in the test env"
    check("rerank backend default == pod_up's", stack.DEFAULT_RERANK_BACKEND == rp.pod_up.RERANK_BACKEND)
    check("stack.from_env() reports tei by default", stack.from_env()["rerank_backend"] == "tei")


def test_tei_backend_env_and_client_block_unchanged():
    # A tei stack must produce exactly the pre-option env + client block: no
    # RERANK_* keys in the container, no RERANKER_API_FORMAT line for the client.
    from app import stack
    st = stack.from_env()
    env = stack.container_env(st, "B", "H")
    check("tei: no RERANK_* passthroughs in container env",
          not any(k.startswith("RERANK_") and k != "RERANK_MODEL_ID" for k in env))
    urls = {"llm": "L", "embedder": "E", "reranker": "R"}
    check("tei: client block has no RERANKER_API_FORMAT",
          "RERANKER_API_FORMAT" not in stack.client_env_block(urls, st, 4096))
    check("tei: runpod client block has no RERANKER_API_FORMAT",
          "RERANKER_API_FORMAT" not in rp.PROVIDER().client_env("podX", 4096))


def test_vllm_backend_env_passthroughs_and_client_format():
    # vllm backend: RERANK_BACKEND + the optional sizing/flag passthroughs reach
    # the container (both the neutral contract and pod_up), and the client block
    # tells the RAG app to speak the Cohere-style format.
    import os as _os
    from app import stack
    keys = {"PODLINK_RERANK_BACKEND": "VLLM",               # case-insensitive
            "PODLINK_RERANK_GPU_MEMORY_UTILIZATION": "0.12",
            "PODLINK_RERANK_VLLM_EXTRA_ARGS": "--foo=1",
            "PODLINK_RERANK_MAX_MODEL_LEN": "8192",
            "PODLINK_RERANK_WAIT_FOR_EMBEDDER_S": "600",
            "PODLINK_VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0"}
    saved_backend = rp.pod_up.RERANK_BACKEND
    _os.environ.update(keys)
    rp.pod_up.RERANK_BACKEND = "vllm"                      # import-time constant in pod_up
    try:
        st = stack.from_env()
        env = stack.container_env(st, "B", "H")
        check("vllm: backend normalised to lower case", st["rerank_backend"] == "vllm")
        check("vllm: RERANK_BACKEND sent", env.get("RERANK_BACKEND") == "vllm")
        check("vllm: reranker gpu share sent", env.get("RERANK_GPU_MEMORY_UTILIZATION") == "0.12")
        check("vllm: reranker extra args sent", env.get("RERANK_VLLM_EXTRA_ARGS") == "--foo=1")
        check("vllm: reranker max len sent", env.get("RERANK_MAX_MODEL_LEN") == "8192")
        check("vllm: reranker embedder wait sent", env.get("RERANK_WAIT_FOR_EMBEDDER_S") == "600")
        check("cuda-graph profiling switch sent", env.get("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS") == "0")
        check("vllm: container env keys == pod_up's", set(env) == set(rp.pod_up._pod_env("B", "H")))
        urls = {"llm": "L", "embedder": "E", "reranker": "R"}
        check("vllm: client block says cohere",
              "RERANKER_API_FORMAT=cohere" in stack.client_env_block(urls, st, 4096))
        check("vllm: runpod client block says cohere",
              "RERANKER_API_FORMAT=cohere" in rp.PROVIDER().client_env("podX", 4096))
        check("vllm: runpod stack_config reports the backend",
              rp.PROVIDER().stack_config()["rerank_backend"] == "vllm")
    finally:
        for k in keys:
            _os.environ.pop(k, None)
        rp.pod_up.RERANK_BACKEND = saved_backend


def test_stack_test_uses_v1_rerank_for_vllm_backend():
    # With the vllm backend the stack test must hit the bearer-guarded /v1/rerank
    # with a `documents` body and parse results[].relevance_score.
    seen = {}
    def responder(url, body):
        if "chat/completions" in url:
            return (200, {"choices": [{"message": {"content": "pong"}}]})
        if "embeddings" in url:
            return (200, {"data": [{"embedding": [0.0] * 4096}]})
        if "rerank" in url:
            seen["url"], seen["body"] = url, body
            return (200, {"results": [{"index": 0, "relevance_score": 0.87},
                                      {"index": 1, "relevance_score": 0.01}]})
        return (404, None)
    install_fake_post(responder)
    saved_backend = rp.pod_up.RERANK_BACKEND
    rp.pod_up.RERANK_BACKEND = "vllm"
    try:
        s = PodSession(); s.try_begin_start(None); s.state = State.RUNNING; s.pod_id = "podABC"
        rd.test_stack(s)
    finally:
        rp.pod_up.RERANK_BACKEND = saved_backend
    check("vllm stack test posts to /v1/rerank", seen.get("url", "").endswith("/v1/rerank"))
    check("vllm stack test sends documents, not texts",
          "documents" in seen.get("body", {}) and "texts" not in seen.get("body", {}))
    check("vllm stack test parses relevance_score", "0.87" in s.test_result["services"]["reranker"]["detail"])
    check("vllm stack test all_ok", s.test_result["all_ok"] is True)


def test_preflight_stack_rows():
    # Bad backend = hard fail; vllm + preset model = ok; vllm + unknown model with
    # no extra args = warn; tei = ok.
    from app import preflight
    levels = lambda st: [lvl for lvl, _ in preflight._stack_rows(st)]  # noqa: E731
    check("preflight fails an unknown backend", levels({"rerank_backend": "onnx"}) == ["fail"])
    check("preflight ok for tei", levels({"rerank_backend": "tei"}) == ["ok"])
    check("preflight ok for vllm + Qwen3-Reranker preset",
          levels({"rerank_backend": "vllm", "rerank_model_id": "Qwen/Qwen3-Reranker-4B"}) == ["ok"])
    check("preflight warns vllm + un-preset model without extra args",
          levels({"rerank_backend": "vllm", "rerank_model_id": "zeroentropy/zerank-2"}) == ["ok", "warn"])
    check("preflight ok vllm + un-preset model WITH extra args",
          levels({"rerank_backend": "vllm", "rerank_model_id": "zeroentropy/zerank-2",
                  "rerank_vllm_extra_args": "--convert=classify"}) == ["ok"])


def test_runpod_client_block_pinned():
    # The RunPod provider now delegates to stack.client_env_block; pin the exact
    # text a user copies so neither path can drift (tei, then vllm).
    base = "\n".join([
        "LLM_BASE_URL=https://podX-8000.proxy.runpod.net/v1",
        "LLM_MODEL=llm",
        "LLM_API_KEY=<your pod_bearer_token>",
        "EMBEDDING_BASE_URL=https://podX-8080.proxy.runpod.net/v1",
        "EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B",
        "EMBEDDING_DIMENSIONS=4096",
        "EMBEDDING_API_KEY=<your pod_bearer_token>",
        "RERANKER_PROVIDER=api",
        "RERANKER_BASE_URL=https://podX-8081.proxy.runpod.net",
        "RERANKER_API_KEY=<your pod_bearer_token>",
    ])
    tail = "\nKG_EXTRACTION_CONCURRENCY=10"
    saved_backend = rp.pod_up.RERANK_BACKEND
    try:
        rp.pod_up.RERANK_BACKEND = "tei"
        check("runpod tei block exact", rp.PROVIDER().client_env("podX", 4096) == base + tail)
        rp.pod_up.RERANK_BACKEND = "vllm"
        check("runpod vllm block exact",
              rp.PROVIDER().client_env("podX", 4096) == base + "\nRERANKER_API_FORMAT=cohere" + tail)
        check("runpod block without dim keeps the detect hint",
              "# EMBEDDING_DIMENSIONS=  <- run 'Test stack'" in rp.PROVIDER().client_env("podX", None))
    finally:
        rp.pod_up.RERANK_BACKEND = saved_backend


if __name__ == "__main__":
    print("driver smoke tests:")
    test_gql_escape_env_makes_json_values_safe()
    test_create_pod_once_sends_escaped_env()
    test_resolve_gpu_id_matches_rtx_pro_6000()
    test_resolve_gpu_id_honours_match_and_min_vram()
    test_gpu_pinning_passthrough_and_fingerprint()
    test_gpu_ladder_parsing_and_default()
    test_memory_budgets_derive_per_card_settings()
    test_ladder_walk_probes_then_creates_on_the_first_rung_with_stock()
    test_community_cloud_knob()
    test_ladder_fingerprint_is_rung_aware()
    test_preflight_gpu_rows()
    test_resolve_gpu_id_raises_when_absent()
    test_all_ready_true_when_all_200()
    test_all_ready_waits_for_stragglers_then_times_out()
    test_all_ready_warns_and_keeps_waiting_past_the_soft_deadline()
    test_soft_deadline_names_a_stuck_tei_service_when_the_llm_is_up()
    test_all_ready_raises_when_the_pod_itself_leaves_running()
    test_recover_if_healthy_flips_error_to_running()
    test_adopt_running_on_startup_adopts_our_running_pod()
    test_pod_still_running_never_abandons_on_doubt()
    test_recover_settles_idle_when_the_instance_is_gone()
    test_recover_does_not_clobber_a_session_that_moved_on()
    test_recover_authenticates_before_any_provider_call()
    test_all_ready_cancels_promptly()
    test_create_aborts_on_cancel_before_create()
    test_provider_selection()
    test_provider_snapshot_fields_are_complete()
    test_stack_defaults_agree_with_pod_up()
    test_is_running_and_is_up_are_not_the_same_predicate()
    test_wait_for_running_ignores_running_without_a_runtime()
    test_verify_terminated_keeps_polling_on_unknown_status()
    test_start_opens_access_on_the_adopt_path()
    test_start_opens_access_on_the_create_path()
    test_start_and_recovery_hand_off_the_client_env()
    test_failed_start_before_an_instance_closes_the_access_path()
    test_failed_start_with_an_instance_keeps_the_access_path()
    test_recover_if_healthy_reopens_the_access_path_before_probing()
    test_stop_closes_the_access_path()
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
    test_rerank_backend_default_agrees_with_pod_up()
    test_tei_backend_env_and_client_block_unchanged()
    test_vllm_backend_env_passthroughs_and_client_format()
    test_stack_test_uses_v1_rerank_for_vllm_backend()
    test_preflight_stack_rows()
    test_runpod_client_block_pinned()
    print("all driver smoke tests passed.")
