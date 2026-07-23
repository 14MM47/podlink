"""Terminate the pod (fully release the GPU).

With the Network Volume model, "down" TERMINATES rather than stops: terminate
releases the GPU cleanly (no host-pinning), and the ~36 GB of model weights
persist on the Network Volume, so the next pod_up.py re-creates and reuses them
without re-downloading. (If NETWORK_VOLUME_ID is empty — the Data Volume fallback
— terminate DOES destroy the weights; they re-download on next up.)

Stopping is deliberately not used: a stopped pod is pinned to its original host,
and resuming fails when that host has no free GPU
("not enough free GPUs on the host machine to start this pod").
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import runpod
from rich import print as rprint
from rich.prompt import Confirm

import _secrets
from pod_up import NETWORK_VOLUME_ID

STATE_PATH = Path(__file__).resolve().parents[1] / "pod_state.json"


def main() -> None:
    if not STATE_PATH.exists():
        sys.exit(f"No pod state at {STATE_PATH}; run pod_up.py first.")

    runpod.api_key = _secrets.runpod_api_key()
    state = json.loads(STATE_PATH.read_text())
    pod_id = state["pod_id"]

    pod = runpod.get_pod(pod_id)
    if pod is None:
        rprint(f"[yellow]Pod {pod_id} no longer exists; nothing to terminate.[/]")
        return
    rprint(f"Pod {pod_id} current status: [bold]{pod.get('desiredStatus')}[/]")

    weights_note = (
        "weights persist on the Network Volume for a fast next up"
        if NETWORK_VOLUME_ID
        else "NO Network Volume set — weights will be LOST and re-download on next up"
    )
    # default=False: terminate is destructive (releases the pod, and without a
    # Network Volume destroys the weights), so a bare Enter must NOT terminate.
    if not Confirm.ask(f"Terminate the pod ({weights_note})?", default=False):
        rprint("Aborted.")
        return

    runpod.terminate_pod(pod_id)
    rprint(f"[green]Terminate request sent for {pod_id}.[/] GPU released; "
           f"{weights_note}.")


if __name__ == "__main__":
    main()
