"""Read secrets from ~/.config/podlink/ with explicit permission + ownership checks."""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

SECRETS_DIR = Path.home() / ".config" / "podlink"


def read_secret(name: str) -> str:
    path = SECRETS_DIR / name
    if not path.exists():
        sys.exit(
            f"Missing secret: {path}\n"
            f"Create it with mode 0600 before running this script."
        )

    # Open first, then fstat — eliminates the TOCTOU race between stat() and read().
    fd = os.open(str(path), os.O_RDONLY)
    try:
        st = os.fstat(fd)
        if st.st_uid != os.getuid():
            sys.exit(
                f"Secret {path} is owned by uid {st.st_uid}, not the current user "
                f"({os.getuid()}). Refusing to read."
            )
        mode = stat.S_IMODE(st.st_mode)
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            sys.exit(
                f"Secret {path} is mode {oct(mode)}; must be 0600. "
                f"Fix with: chmod 0600 {path}"
            )
        # 4 KB is generous for any secret we store here (token = 64 hex chars).
        data = os.read(fd, 4096).decode().strip()
    finally:
        os.close(fd)

    if not data:
        sys.exit(f"Secret {path} is empty.")
    return data


def runpod_api_key() -> str:
    return read_secret("runpod_api_key")


def hf_token() -> str:
    return read_secret("hf_token")


def bearer_token() -> str:
    return read_secret("pod_bearer_token")
