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
import sys

from . import providers
from .vendored import _secrets, read_secret

_MARK = {"ok": "\033[32m  ✓\033[0m", "warn": "\033[33m  !\033[0m",
         "fail": "\033[31m  ✗\033[0m", "info": "   ·"}


def _neutral_rows() -> list[tuple[str, str]]:
    """The secrets every stack needs, regardless of cloud."""
    rows: list[tuple[str, str]] = []
    for label, getter in (("pod_bearer_token", _secrets.bearer_token),
                          ("hf_token", _secrets.hf_token)):
        try:
            read_secret(getter)
            rows.append(("ok", f"{label} readable (0600)"))
        except RuntimeError as e:                    # missing / mis-permissioned / empty
            rows.append(("fail", f"{label}: {e.__cause__ or e}"))
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
    line(f"   reranker: {stack.get('rerank_model_id')}")
    line(f"   max model len: {stack.get('max_model_len')} · gpu share: {stack.get('gpu_memory_utilization')}")
    line()

    rows = _neutral_rows() + list(provider.preflight())
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
