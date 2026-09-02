"""The RunPod provider package.

Layout:
  provider.py — RunPodProvider, the class app/driver.py drives
  launch.sh   — sourced by start.sh before the venv exists: the interactive
                Network Volume id prompt (needs a TTY, so it stays in bash)
  __init__.py — this file: hooks that must work BEFORE a provider instance
                exists, plus the re-exports the rest of the app imports

Nothing outside this package may know RunPod's shape; tests/test_siloing.py
enforces that by grepping the generic core.
"""
from __future__ import annotations

import re
from pathlib import Path

from .provider import RunPodProvider, pod_up  # noqa: F401  re-exported

#: The class the registry instantiates for PODLINK_PROVIDER=runpod.
PROVIDER = RunPodProvider

# The launcher saves the pasted Network Volume id here (it is infra, not a
# secret, but kept under the config dir for tidiness). Module-level so tests can
# point it at a temp directory.
VOL_FILE = Path.home() / ".config" / "podlink" / "network_volume_id"

# start.sh's explicit "deliberately no volume" sentinel.
_NONE_RE = re.compile(r"^none$", re.IGNORECASE)


def normalise_env(env: dict[str, str]) -> dict[str, str]:
    """Resolve PODLINK_NETWORK_VOLUME_ID the way start.sh does, for a profile switch.

    Explicit `none` => Data-Volume mode (empty); an explicit id is used as-is;
    unset falls back to the saved id file. There is no interactive prompt here —
    unset with no file is empty. Mirrors app/providers/runpod/launch.sh, so the
    console's profile switch and a fresh launch agree on the volume.
    """
    vol = env.get("PODLINK_NETWORK_VOLUME_ID")
    if vol is None and VOL_FILE.is_file():
        vol = VOL_FILE.read_text()
    vol = (vol or "").strip()
    env["PODLINK_NETWORK_VOLUME_ID"] = "" if _NONE_RE.match(vol) else vol
    return env


__all__ = ["PROVIDER", "RunPodProvider", "VOL_FILE", "normalise_env", "pod_up"]
