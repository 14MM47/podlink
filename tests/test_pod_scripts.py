"""Offline tests for the pod image's launch scripts (pod_image/start-*.sh).

Runs the real bash scripts with stub `vllm`, `text-embeddings-router` and `curl`
binaries first on PATH, so nothing is served or downloaded. The stubs print the
argv they receive, which is what the tests assert on:

  * a non-integer wait (VLLM_WAIT_FOR_TEI_S / RERANK_WAIT_FOR_EMBEDDER_S) warns
    and falls back to 1200 s instead of silently skipping the wait;
  * every vLLM process binds 127.0.0.1 on its internal port (the public port
    belongs to the auth proxy), while TEI keeps its public port;
  * the reranker script fails closed on a missing key or unknown backend.

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

# Stub: print "<name> ARGV:" then one argv element per line, bracketed.
_STUB = '#!/bin/bash\necho "{name} ARGV:"\nfor a in "$@"; do printf "[%s]\\n" "$a"; done\n'


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
        full = {"PATH": f"{d}:{os.environ['PATH']}", **env}
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


if __name__ == "__main__":
    test_vllm_non_integer_wait_falls_back()
    test_vllm_binds_loopback_internal_port()
    test_reranker_vllm_non_integer_wait_falls_back_and_binds_loopback()
    test_reranker_tei_keeps_public_port()
    test_reranker_fails_closed()
    print("all pod script tests passed.")
