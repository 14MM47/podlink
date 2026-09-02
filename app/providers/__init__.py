"""Provider selection: which cloud this launch drives.

`PODLINK_PROVIDER` is read from the environment like every other setting, which
means a profile conf picks the cloud exactly the way it already picks models and
sizing — no new concept in the UI, and `--profile` at launch selects it too.

Each provider is a package under app/providers/<name>/ exposing:

  PROVIDER        — the class implementing app/providers/base.py
  normalise_env   — optional: fix up a PODLINK_* env before it is applied
  launch.sh       — optional: sourced by start.sh before the venv exists

Provider packages are imported lazily, inside the factory, so a launch never
has to have another cloud's SDK installed. This module and base.py are the seam:
the only files outside a provider package allowed to name a cloud.
"""
from __future__ import annotations

import importlib
import os
from types import ModuleType

from .base import Provider

DEFAULT_PROVIDER = "runpod"          # what a config that says nothing gets
KNOWN_PROVIDERS = ("runpod",)        # extended as providers land

# The live provider instance and the name it was built for. Swapped wholesale by
# reload_active(); two plain global assignments, so a concurrent snapshot read
# sees a whole provider object either way, never a half-built one.
_ACTIVE: Provider | None = None
_ACTIVE_NAME: str | None = None


def _normalise_name(raw: str | None) -> str:
    return (raw or DEFAULT_PROVIDER).strip().lower()


def provider_name() -> str:
    """The configured provider name, normalised."""
    return _normalise_name(os.environ.get("PODLINK_PROVIDER"))


def _module(name: str) -> ModuleType:
    """Import the provider package called `name`, or fail with a clear message."""
    if name not in KNOWN_PROVIDERS:
        raise ValueError(
            f"unknown PODLINK_PROVIDER {name!r} — known providers: {', '.join(KNOWN_PROVIDERS)}")
    return importlib.import_module(f"{__name__}.{name}")   # imports that cloud's SDK


def _build(name: str) -> Provider:
    """Construct the provider called `name`."""
    return _module(name).PROVIDER()


def active() -> Provider:
    """The provider for the current configuration, built once and reused."""
    global _ACTIVE, _ACTIVE_NAME
    name = provider_name()
    if _ACTIVE is None or name != _ACTIVE_NAME:      # first call, or the config changed
        _ACTIVE, _ACTIVE_NAME = _build(name), name
    return _ACTIVE


def reload_active() -> None:
    """Re-bake the active provider from the current environment.

    Called after a profile switch has swapped the process env. A profile that
    changes PODLINK_PROVIDER gets a whole new provider; otherwise the existing
    one re-reads its settings. Raises if the new configuration is unusable —
    the caller restores the old environment and calls this again.
    """
    global _ACTIVE, _ACTIVE_NAME
    name = provider_name()
    if _ACTIVE is None or name != _ACTIVE_NAME:      # different cloud entirely
        _ACTIVE, _ACTIVE_NAME = _build(name), name
        return
    _ACTIVE.reload_config()                          # same cloud, new settings


def normalise_env(env: dict[str, str]) -> dict[str, str]:
    """Let the provider named IN `env` fix up that env before it is applied.

    Runs before a provider instance exists (the env may be switching clouds),
    so it dispatches on the package-level hook rather than the instance. A
    provider with nothing to normalise simply omits the hook.
    """
    hook = getattr(_module(_normalise_name(env.get("PODLINK_PROVIDER"))), "normalise_env", None)
    return hook(env) if hook else env


__all__ = ["Provider", "active", "reload_active", "provider_name", "normalise_env",
           "DEFAULT_PROVIDER", "KNOWN_PROVIDERS"]
