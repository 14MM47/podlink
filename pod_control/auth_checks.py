"""Phase 0 auth verification: positive + two negatives.

Exits non-zero on any unexpected outcome so this script can gate the exit demo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from rich import print as rprint

import _secrets
import egress_logger

STATE_PATH = Path(__file__).resolve().parents[1] / "pod_state.json"


def load_state() -> dict:
    if not STATE_PATH.exists():
        sys.exit(f"No pod state at {STATE_PATH}; run pod_up.py first.")
    return json.loads(STATE_PATH.read_text())


def check(label: str, url: str, headers: dict, expect_status: int) -> bool:
    with egress_logger.client(timeout=30.0) as c:
        try:
            r = c.get(url, headers=headers)
        except Exception as e:
            rprint(f"[red]✗ {label}[/]: request failed: {e}")
            return False
    ok = r.status_code == expect_status
    marker = "[green]✓[/]" if ok else "[red]✗[/]"
    rprint(f"{marker} {label}: expected {expect_status}, got {r.status_code}")
    return ok


def main() -> None:
    state = load_state()
    bearer = _secrets.bearer_token()
    url = f"{state['proxy_url']}/v1/models"

    results = [
        check("positive (correct bearer)",
              url, {"Authorization": f"Bearer {bearer}"}, 200),
        check("negative (wrong bearer)",
              url, {"Authorization": "Bearer wrongtoken"}, 401),
        check("negative (no auth header)",
              url, {}, 401),
    ]

    if all(results):
        rprint("\n[bold green]All auth checks passed.[/]")
        sys.exit(0)
    else:
        rprint("\n[bold red]Auth checks FAILED.[/] Pod may still be loading weights; "
               "wait 1–2 minutes and retry. If failures persist, inspect the pod logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
