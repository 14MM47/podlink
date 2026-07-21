"""Stop (do not terminate) the Phase 0 pod.

Stopped pods bill only for volume storage (~£0.005/hr). Model weights survive,
so next pod_up.py resume skips the download.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import runpod
from rich import print as rprint
from rich.prompt import Confirm

import _secrets

STATE_PATH = Path(__file__).resolve().parents[1] / "pod_state.json"


def main() -> None:
    if not STATE_PATH.exists():
        sys.exit(f"No pod state at {STATE_PATH}; run pod_up.py first.")

    runpod.api_key = _secrets.runpod_api_key()
    state = json.loads(STATE_PATH.read_text())
    pod_id = state["pod_id"]

    pod = runpod.get_pod(pod_id)
    current = pod.get("desiredStatus")
    rprint(f"Pod {pod_id} current status: [bold]{current}[/]")

    if current != "RUNNING":
        rprint("[yellow]Pod not in RUNNING state; nothing to stop.[/]")
        return

    if not Confirm.ask("Stop the pod (volume preserved)?", default=True):
        rprint("Aborted.")
        return

    runpod.stop_pod(pod_id)
    rprint(f"[green]Stop request sent for {pod_id}.[/]")
    rprint("Volume storage continues to bill at the static rate; GPU billing ceases.")


if __name__ == "__main__":
    main()
