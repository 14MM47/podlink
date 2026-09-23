"""Offline tests for the pod image's launch scripts (pod_image/start-*.sh).

Runs the real bash scripts with stub `vllm`, `text-embeddings-router` and `curl`
binaries first on PATH, so nothing is served or downloaded. The stubs print the
argv and the relevant env they receive, which is what the tests assert on:

  * a non-integer wait (VLLM_WAIT_FOR_TEI_S / RERANK_WAIT_FOR_EMBEDDER_S) warns
    and falls back to 1200 s instead of silently skipping the wait;
  * every vLLM process binds 127.0.0.1 on its internal port (the public port
    belongs to the auth proxy), while TEI keeps its public port;
  * the reranker script fails closed on a missing key or unknown backend;
  * TEI runs with LOG_LEVEL=warn, so its startup line (which prints api_key)
    never reaches the logs;
  * both vLLM scripts put their compile caches under a per-service tree on the
    volume, keep explicit values, fall back cleanly when the base isn't
    writable, and clear a cache left by a start that never became healthy.

Needs bash (present on the CI runner and every dev box). No network.

Run: python3 tests/test_pod_scripts.py
"""
from __future__ import annotations

import os
import stat
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "pod_image"

# Stub: print "<name> ARGV:" then one argv element per line, bracketed, then the
# env vars the scripts are meant to set (LOG_LEVEL for TEI, caches for vLLM).
_STUB = ('#!/bin/bash\necho "{name} ARGV:"\nfor a in "$@"; do printf "[%s]\\n" "$a"; done\n'
         'echo "ENV LOG_LEVEL=${{LOG_LEVEL-<unset>}}"\n'
         'echo "ENV VLLM_CACHE_ROOT=${{VLLM_CACHE_ROOT-<unset>}}"\n'
         'echo "ENV TRITON_CACHE_DIR=${{TRITON_CACHE_DIR-<unset>}}"\n')


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise SystemExit(f"FAILED: {name}")


def _run(script: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    """Run pod_image/<script> with stubs first on PATH; curl always succeeds."""
    with tempfile.TemporaryDirectory() as d:
        for name in ("vllm", "text-embeddings-router"):
            p = Path(d, name)
            p.write_text(_STUB.format(name=name))
            p.chmod(p.stat().st_mode | stat.S_IEXEC)
        # curl exits 0 => every health wait is satisfied on its first probe.
        curl = Path(d, "curl")
        curl.write_text("#!/bin/bash\nexit 0\n")
        curl.chmod(curl.stat().st_mode | stat.S_IEXEC)
        # One /health probe at most, so the cache helper's detached watcher can't
        # outlive the test by long (the real default is an hour of probes).
        full = {"PATH": f"{d}:{os.environ['PATH']}", "PODLINK_CACHE_HEALTH_TRIES": "1", **env}
        return subprocess.run(["bash", str(IMAGE / script)], env=full,
                              capture_output=True, text=True, timeout=30)


def test_vllm_non_integer_wait_falls_back():
    r = _run("start-vllm.sh", {"LLM_MODEL_ID": "m", "VLLM_WAIT_FOR_TEI_S": "off",
                               "TEI_API_KEY": "k"})
    check("start-vllm: runs to exec", r.returncode == 0 and "vllm ARGV:" in r.stdout)
    check("start-vllm: non-integer wait warns + uses 1200",
          "VLLM_WAIT_FOR_TEI_S='off' is not integer seconds; using 1200" in r.stderr)
    check("start-vllm: health wait ran (TEI probes satisfied)", "healthy after" in r.stdout)


def test_vllm_binds_loopback_internal_port():
    r = _run("start-vllm.sh", {"LLM_MODEL_ID": "m", "VLLM_WAIT_FOR_TEI_S": "0"})
    check("start-vllm: --host 127.0.0.1", "[--host]\n[127.0.0.1]" in r.stdout)
    check("start-vllm: --port 18000", "[--port]\n[18000]" in r.stdout)
    check("start-vllm: never binds 0.0.0.0", "0.0.0.0" not in r.stdout)


def test_reranker_vllm_non_integer_wait_falls_back_and_binds_loopback():
    r = _run("start-reranker.sh", {"RERANK_BACKEND": "vllm", "RERANK_MODEL_ID": "Qwen/Qwen3-Reranker-4B",
                                   "VLLM_API_KEY": "k", "RERANK_WAIT_FOR_EMBEDDER_S": "soon"})
    check("start-reranker: runs to exec", r.returncode == 0 and "vllm ARGV:" in r.stdout)
    check("start-reranker: non-integer wait warns + uses 1200",
          "RERANK_WAIT_FOR_EMBEDDER_S='soon' is not integer seconds; using 1200" in r.stderr)
    check("start-reranker(vllm): --host 127.0.0.1", "[--host]\n[127.0.0.1]" in r.stdout)
    check("start-reranker(vllm): --port 18081", "[--port]\n[18081]" in r.stdout)
    check("start-reranker(vllm): Qwen3 preset applied", "[--chat-template]" in r.stdout
          and '"is_original_qwen3_reranker":true' in r.stdout)


def test_reranker_tei_keeps_public_port():
    r = _run("start-reranker.sh", {"RERANK_MODEL_ID": "BAAI/bge-reranker-v2-m3", "TEI_API_KEY": "k"})
    check("start-reranker(tei): TEI on public 8081",
          "text-embeddings-router ARGV:" in r.stdout and "[--port]\n[8081]" in r.stdout)


def test_reranker_fails_closed():
    r = _run("start-reranker.sh", {"RERANK_BACKEND": "vllm", "RERANK_MODEL_ID": "x"})
    check("start-reranker: no VLLM_API_KEY -> exit 1, no exec",
          r.returncode == 1 and "vllm ARGV:" not in r.stdout)
    r = _run("start-reranker.sh", {"RERANK_BACKEND": "onnx", "RERANK_MODEL_ID": "x"})
    check("start-reranker: unknown backend -> exit 1", r.returncode == 1 and "must be 'tei' or 'vllm'" in r.stderr)


def test_tei_logs_at_warn_so_the_key_never_prints():
    # TEI's startup INFO line prints its args INCLUDING api_key; LOG_LEVEL=warn
    # (TEI ignores RUST_LOG — verified against the 1.9 router) suppresses it.
    r = _run("start-embedder.sh", {"EMBED_MODEL_ID": "m", "TEI_API_KEY": "k"})
    check("embedder: TEI started with LOG_LEVEL=warn", "ENV LOG_LEVEL=warn" in r.stdout)
    r = _run("start-reranker.sh", {"RERANK_MODEL_ID": "m", "TEI_API_KEY": "k"})
    check("reranker(tei): TEI started with LOG_LEVEL=warn", "ENV LOG_LEVEL=warn" in r.stdout)
    r = _run("start-embedder.sh", {"EMBED_MODEL_ID": "m", "TEI_API_KEY": "k", "LOG_LEVEL": "info"})
    check("embedder: an explicit LOG_LEVEL wins", "ENV LOG_LEVEL=info" in r.stdout)


def test_vllm_compile_caches_on_the_volume():
    with tempfile.TemporaryDirectory() as base:
        for script, env in (("start-vllm.sh", {"LLM_MODEL_ID": "m", "VLLM_WAIT_FOR_TEI_S": "0"}),
                            ("start-reranker.sh", {"RERANK_BACKEND": "vllm", "RERANK_MODEL_ID": "x",
                                                   "VLLM_API_KEY": "k", "RERANK_WAIT_FOR_EMBEDDER_S": "0"})):
            svc = script.removesuffix(".sh")        # per-service tree: <base>/<service>/…
            r = _run(script, {**env, "PODLINK_CACHE_BASE": base})
            check(f"{script}: VLLM_CACHE_ROOT under <base>/{svc}/",
                  f"ENV VLLM_CACHE_ROOT={base}/{svc}/vllm-" in r.stdout)
            check(f"{script}: TRITON_CACHE_DIR under <base>/{svc}/",
                  f"ENV TRITON_CACHE_DIR={base}/{svc}/triton-" in r.stdout)
            r = _run(script, {**env, "PODLINK_CACHE_BASE": base, "VLLM_CACHE_ROOT": "/explicit"})
            check(f"{script}: an explicit VLLM_CACHE_ROOT wins", "ENV VLLM_CACHE_ROOT=/explicit" in r.stdout)
            r = _run(script, {**env, "PODLINK_CACHE_BASE": "/proc/podlink-not-writable"})
            check(f"{script}: unwritable base keeps the defaults (and still starts)",
                  r.returncode == 0 and "ENV VLLM_CACHE_ROOT=<unset>" in r.stdout
                  and "not writable" in r.stderr)
        # The TEI reranker path must be untouched: TEI execs before the helper runs.
        r = _run("start-reranker.sh", {"RERANK_MODEL_ID": "m", "TEI_API_KEY": "k", "PODLINK_CACHE_BASE": base})
        check("start-reranker.sh (tei): no vLLM cache env set",
              "text-embeddings-router ARGV:" in r.stdout and "ENV VLLM_CACHE_ROOT=<unset>" in r.stdout
              and "ENV TRITON_CACHE_DIR=<unset>" in r.stdout)


def test_cache_self_heals_after_a_start_that_never_became_healthy():
    # A marker left by a start that never reached /health means its cache may be
    # half-written: the next start must delete that service's cache and rebuild.
    import time
    with tempfile.TemporaryDirectory() as base:
        env = {"LLM_MODEL_ID": "m", "VLLM_WAIT_FOR_TEI_S": "0", "PODLINK_CACHE_BASE": base}
        root = Path(base, "start-vllm")
        r = _run("start-vllm.sh", env)                       # first start: picks the cache dir
        # vLLM (not the helper) creates the dir at runtime; the stub doesn't, so
        # take the path the helper exported and create it as vLLM would.
        vdir = Path(next(line.split("=", 1)[1] for line in r.stdout.splitlines()
                         if line.startswith("ENV VLLM_CACHE_ROOT=")))
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "artifact.bin").write_text("half-written")     # simulate a partial artifact
        other = Path(base, "start-reranker", "vllm-x")
        other.mkdir(parents=True)
        (other / "keep.bin").write_text("other service")
        (root / ".start-in-progress").touch()                  # previous start never healthy
        r = _run("start-vllm.sh", env)
        check("self-heal: says it is clearing the cache", "clearing its compile cache" in r.stderr)
        check("self-heal: this service's stale artifact is gone", not (vdir / "artifact.bin").exists())
        check("self-heal: the other service's cache is untouched", (other / "keep.bin").exists())
        # The stub curl answers /health at once, so the watcher clears the marker.
        for _ in range(20):
            if not (root / ".start-in-progress").exists():
                break
            time.sleep(0.1)
        check("marker cleared once /health answers", not (root / ".start-in-progress").exists())


def test_api_key_in_extra_args_is_refused():
    # vLLM logs non-default CLI args unredacted, so a key must never be on argv.
    r = _run("start-vllm.sh", {"LLM_MODEL_ID": "m", "VLLM_WAIT_FOR_TEI_S": "0",
                               "VLLM_EXTRA_ARGS": "--foo 1 --api-key=abc"})
    check("start-vllm: --api-key in extra args -> exit 1, no exec",
          r.returncode == 1 and "vllm ARGV:" not in r.stdout and "--api-key" in r.stderr)
    r = _run("start-reranker.sh", {"RERANK_BACKEND": "vllm", "RERANK_MODEL_ID": "x", "VLLM_API_KEY": "k",
                                   "RERANK_WAIT_FOR_EMBEDDER_S": "0", "RERANK_VLLM_EXTRA_ARGS": "--api-key abc"})
    check("start-reranker: --api-key in extra args -> exit 1, no exec",
          r.returncode == 1 and "vllm ARGV:" not in r.stdout)
    r = _run("start-vllm.sh", {"LLM_MODEL_ID": "m", "VLLM_WAIT_FOR_TEI_S": "0",
                               "VLLM_EXTRA_ARGS": "--api-keys-are-not-this-flag"})
    check("start-vllm: other flags still pass", r.returncode == 0)
    r = _run("start-vllm.sh", {"LLM_MODEL_ID": "m", "VLLM_WAIT_FOR_TEI_S": "0",
                               "VLLM_EXTRA_ARGS": "--api_key abc"})
    check("start-vllm: --api_key (underscore form) also refused", r.returncode == 1)


if __name__ == "__main__":
    test_vllm_non_integer_wait_falls_back()
    test_vllm_binds_loopback_internal_port()
    test_reranker_vllm_non_integer_wait_falls_back_and_binds_loopback()
    test_reranker_tei_keeps_public_port()
    test_reranker_fails_closed()
    test_tei_logs_at_warn_so_the_key_never_prints()
    test_vllm_compile_caches_on_the_volume()
    test_cache_self_heals_after_a_start_that_never_became_healthy()
    test_api_key_in_extra_args_is_refused()
    print("all pod script tests passed.")
