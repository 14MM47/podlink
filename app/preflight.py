"""`./start.sh --check`: the launch preflight, in Python so providers own their half.

Prints what a launch WOULD deploy and what would stop it, spending nothing.
The neutral checks live here — the two secrets every provider's stack needs,
and the stack the active provider reports it will serve. Everything cloud-shaped
(API keys, image registries, volume modes, quotas) comes from the provider's
own `preflight()` rows.

Exit status: 1 if any row is a hard failure, else 0 — so start.sh can gate on
it. Run directly: `python -m app.preflight`.
"""
from __future__ import annotations

import os
import re
import sys

from . import clientenv, providers
from . import stack as stack_contract
from .vendored import _secrets, read_secret

_MARK = {"ok": "\033[32m  ✓\033[0m", "warn": "\033[33m  !\033[0m",
         "fail": "\033[31m  ✗\033[0m", "info": "   ·"}


# Same rule as pod_image/authproxy.py _TOKEN_RE (a test pins them equal): the
# in-pod auth proxy embeds the bearer in nginx.conf and refuses any other shape,
# which would leave the pod with no public LLM port. Catch it before billing.
_BEARER_RE = re.compile(r"[A-Za-z0-9._~+/=-]{16,}")


def _neutral_rows() -> list[tuple[str, str]]:
    """The secrets every stack needs, regardless of cloud."""
    rows: list[tuple[str, str]] = []
    for label, getter in (("pod_bearer_token", _secrets.bearer_token),
                          ("hf_token", _secrets.hf_token)):
        try:
            value = read_secret(getter)
            rows.append(("ok", f"{label} readable (0600)"))
        except RuntimeError as e:                    # missing / mis-permissioned / empty
            rows.append(("fail", f"{label}: {e.__cause__ or e}"))
            continue
        # Never print the value — only whether it has the shape the proxy accepts.
        if label == "pod_bearer_token" and not _BEARER_RE.fullmatch(value):
            rows.append(("fail", "pod_bearer_token must be >=16 chars of [A-Za-z0-9._~+/=-] "
                                 "(the in-pod auth proxy refuses anything else)"))
    return rows


# Rerankers start-reranker.sh has a built-in vllm preset for (hf-overrides + template).
_VLLM_RERANK_PRESETS = ("Qwen/Qwen3-Reranker-",)

# `--tensor-parallel-size N` / `--tensor-parallel-size=N` / `-tp N` in the extra vLLM args.
_TP_RE = re.compile(r"(?:--tensor-parallel-size|-tp)(?:=|\s+)(\d+)")
_PIN_KEYS = (("llm_gpus", "PODLINK_LLM_GPUS", "LLM"),
             ("embed_gpus", "PODLINK_EMBED_GPUS", "embedder"),
             ("rerank_gpus", "PODLINK_RERANK_GPUS", "reranker"))


def _gpu_rows(stack: dict) -> list[tuple[str, str]]:
    """Multi-GPU sanity: every pinned GPU index and any tensor-parallel size must
    fit the instance's GPU count, or the service dies at boot on a billing pod
    (CUDA_VISIBLE_DEVICES=1 on a one-GPU pod = no device). Only when the provider
    reports a gpu_count; a multi-GPU pod with neither pinning nor TP gets a warning,
    since every service would then land on GPU 0 and the others sit idle.
    """
    count = stack.get("gpu_count")
    if not isinstance(count, int) or count < 1:
        return []
    rows: list[tuple[str, str]] = []
    pinned = False
    for key, var, label in _PIN_KEYS:
        val = (stack.get(key) or "").strip()
        if not val:
            continue
        pinned = True
        try:
            idx = [int(x) for x in val.split(",")]
        except ValueError:
            rows.append(("fail", f"{var}={val!r}: must be comma-separated GPU indices"))
            continue
        bad = [i for i in idx if i < 0 or i >= count]
        if bad:
            rows.append(("fail", f"{var}={val!r}: GPU index {bad[0]} does not exist on a "
                                 f"{count}-GPU instance (indices 0..{count - 1})"))
        else:
            rows.append(("ok", f"{label} pinned to GPU {val}"))
    tp = _TP_RE.search(stack.get("vllm_extra_args") or "")
    if tp and int(tp.group(1)) > count:
        rows.append(("fail", f"--tensor-parallel-size {tp.group(1)} needs more GPUs than the "
                             f"instance has ({count})"))
    elif tp:
        rows.append(("ok", f"LLM tensor-parallel across {tp.group(1)} GPUs"))
    if count > 1 and not pinned and not tp:
        rows.append(("warn", f"{count}-GPU instance with no per-service pinning (PODLINK_*_GPUS) "
                             "and no --tensor-parallel-size: every service lands on GPU 0"))
    return rows


def _stack_rows(stack: dict) -> list[tuple[str, str]]:
    """Checks on the stack itself: the reranker backend value and its model fit."""
    backend = stack.get("rerank_backend") or stack_contract.DEFAULT_RERANK_BACKEND
    if backend not in stack_contract.RERANK_BACKENDS:
        # The image would refuse to start the reranker; catch it before billing.
        return [("fail", f"PODLINK_RERANK_BACKEND={backend!r}: must be one of "
                         f"{', '.join(stack_contract.RERANK_BACKENDS)}")]
    if backend != "vllm":
        return [("ok", "reranker backend: tei")]
    rows = [("ok", "reranker backend: vllm (client needs RERANKER_API_FORMAT=cohere)")]
    model = stack.get("rerank_model_id") or ""
    # A model with no preset and no extra flags would load as a plain LM and
    # either fail or score garbage — warn rather than fail (flags may be baked in).
    if not model.startswith(_VLLM_RERANK_PRESETS) and not stack.get("rerank_vllm_extra_args"):
        rows.append(("warn", f"vllm reranker {model!r} has no built-in preset and "
                             "PODLINK_RERANK_VLLM_EXTRA_ARGS is empty"))
    return rows


def _lifecycle_rows(provider) -> list[tuple[str, str]]:
    """PODLINK_LIFECYCLE: what POD DOWN does, and whether this provider can do it."""
    wanted = os.environ.get("PODLINK_LIFECYCLE", "").strip().lower() or "terminate"
    if wanted not in ("terminate", "stop"):
        return [("fail", f"PODLINK_LIFECYCLE={wanted!r}: must be terminate or stop")]
    if wanted == "terminate":
        return [("ok", "lifecycle: terminate (POD DOWN destroys the pod; POD UP creates fresh)")]
    if not getattr(provider, "supports_stop", False):
        # driver.lifecycle() would silently fall back to terminate — say so loudly.
        return [("fail", f"PODLINK_LIFECYCLE=stop is not supported by provider {provider.name!r}")]
    from .driver import _resume_retries, _resume_retry_delay
    rows = [("ok", f"lifecycle: stop (POD DOWN stops; POD UP resumes, {_resume_retries()} tries "
                   f"x {_resume_retry_delay()}s, then terminates + creates fresh)")]
    if not provider.persistence_configured():
        rows.append(("warn", "lifecycle stop without persistent storage: a fallback recreate "
                             "re-downloads the weights"))
    return rows


def run(out=sys.stdout) -> int:
    """Print the report; return the exit status (1 on any hard failure)."""
    def line(s: str = "") -> None:
        print(s, file=out)

    try:
        provider = providers.active()
    except Exception as e:  # noqa: BLE001 — a bad provider name / missing SDK is the finding
        line(f"{_MARK['fail']} provider: {type(e).__name__}: {e}")
        return 1

    stack = provider.stack_config()
    line("Stack config:")
    line(f"   profile:  {os.environ.get('PODLINK_PROFILE') or '<none — base conf only>'}")
    line(f"   provider: {provider.name}")
    line(f"   image:    {stack.get('image') or '<not set>'}")
    line(f"   llm:      {stack.get('llm_model_id')}  (served as {stack.get('llm_served_name')})")
    line(f"   embedder: {stack.get('embed_model_id')}")
    line(f"   reranker: {stack.get('rerank_model_id')}  "
         f"({stack.get('rerank_backend') or stack_contract.DEFAULT_RERANK_BACKEND})")
    line(f"   max model len: {stack.get('max_model_len')} · gpu share: {stack.get('gpu_memory_utilization')}")
    if stack.get("gpu_count"):
        line(f"   gpu:      {stack['gpu_count']} x {stack.get('gpu_match') or '<provider default>'}"
             f" (>= {stack.get('gpu_min_vram_gb')} GB)")
    line()

    rows = (_neutral_rows() + _stack_rows(stack) + _gpu_rows(stack) + _lifecycle_rows(provider)
            + clientenv.preflight_rows() + list(provider.preflight()))
    line("Checks:")
    for level, msg in rows:
        line(f"{_MARK.get(level, '   ?')} {msg}")
    failed = any(level == "fail" for level, _ in rows)
    line()
    line("\033[31mPreflight FAILED — fix the ✗ rows above before launching.\033[0m" if failed
         else "\033[32mPreflight OK — run ./start.sh (no --check) to launch.\033[0m")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
