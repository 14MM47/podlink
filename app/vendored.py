"""Import shim for the vendored pod_control/ modules that are provider-neutral.

The pod_control/ scripts are CLI tools with sibling imports (`import _secrets`,
`import egress_logger`), so they only import cleanly when pod_control/ itself is
on sys.path. Doing that insert here — once, in one place — keeps the path hack
out of the driver and the provider modules, which both need these two:

  * _secrets      — the 0600/ownership-checked local secret reader
  * egress_logger — the audited httpx client every outbound call goes through

Both are about *this machine's* secrets and outbound traffic, not about any
particular cloud, so they sit below the provider seam. The cloud-specific
vendored script (pod_up) is imported by its own provider package instead.
"""
from __future__ import annotations

import sys                # to mutate sys.path for the vendored imports
from pathlib import Path  # build the pod_control directory path

# …/podlink/pod_control — front of path so our vendored copy wins.
POD_CONTROL_DIR = Path(__file__).resolve().parents[1] / "pod_control"
if str(POD_CONTROL_DIR) not in sys.path:              # avoid duplicate entries on reload
    sys.path.insert(0, str(POD_CONTROL_DIR))

import _secrets       # noqa: E402  vendored secret reader (0600/ownership checked)
import egress_logger  # noqa: E402  vendored audited httpx client


def read_secret(getter) -> str:
    """Read a secret and register it for verbatim redaction in the egress log.

    The vendored getter calls sys.exit() (raising SystemExit — a BaseException)
    when a secret is missing, mis-permissioned, or empty. SystemExit slips past
    the `except Exception` handlers in start()/stop() and the /pods route, which
    would silently kill the stop worker (session stuck STOPPING while the pod
    keeps billing) or crash the server. Convert it to a normal RuntimeError so
    those handlers catch it and surface a recoverable error.
    """
    try:
        value = getter()                    # vendored getter; sys.exit on missing/bad
    except SystemExit as e:                  # missing / mis-permissioned / empty secret
        raise RuntimeError("secret unavailable — check ~/.config/podlink") from e
    egress_logger.register_secret(value)    # strip this exact value from any log line
    return value


__all__ = ["POD_CONTROL_DIR", "_secrets", "egress_logger", "read_secret"]
