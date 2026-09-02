"""Runtime profile discovery + switching for the web console.

start.sh applies a profile by *sourcing* ~/.config/podlink/profiles/<name>.conf
into the server's environment before launch. This module lets the console switch
profiles WITHOUT a relaunch:

  * conf files are PARSED (shlex), never executed — no shell ever runs here;
  * only `export PODLINK_*=value` lines are honoured; every other line
    (comments, other keys, arbitrary shell) is ignored;
  * switching rebuilds the process PODLINK_* env from a startup baseline plus
    the chosen profile's exports, then re-bakes the active provider so its
    settings re-read the new environment. A profile that names a different
    PODLINK_PROVIDER switches cloud outright.

The caller (the /profile/select route) must only switch while no pod work is in
flight — re-baking the provider under a live driver thread would let one launch
read half-old, half-new settings.
"""
from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

from . import providers

# Same layout start.sh uses. Module-level (not function-local) so tests can
# point them at a temp directory.
CONFIG_DIR = Path.home() / ".config" / "podlink"
BASE_CONF = CONFIG_DIR / "podlink.conf"
PROFILE_DIR = CONFIG_DIR / "profiles"

# Profile names come from the client — no path separators, no traversal.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_KEY_RE = re.compile(r"^PODLINK_[A-Z0-9_]+$")


def valid_name(name: str) -> bool:
    """True when `name` is a safe profile name (and not a path fragment)."""
    return bool(_NAME_RE.match(name))


def list_profiles() -> list[str]:
    """Names of the available profile confs (sorted; unsafe names excluded)."""
    if not PROFILE_DIR.is_dir():
        return []
    return sorted(p.stem for p in PROFILE_DIR.glob("*.conf") if valid_name(p.stem))


def parse_conf(path: Path) -> dict[str, str]:
    """The PODLINK_* exports of a conf file, parsed — never executed.

    Accepts the subset of bash that start.sh's confs actually use: one
    `export KEY=VALUE` per line, double/single quoting and backslash escapes
    (shlex mirrors POSIX sh here, so a quoted PODLINK_VLLM_EXTRA_ARGS with
    embedded \" survives intact), `#` comments. Anything else — other keys,
    multi-word commands, malformed lines — is skipped, so a hostile line in a
    conf can at worst set a PODLINK_ variable, exactly as sourcing it could.
    """
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text().splitlines():
        try:
            tokens = shlex.split(raw, comments=True)
        except ValueError:  # unbalanced quote — skip the line, not the file
            continue
        if tokens and tokens[0] == "export":
            tokens = tokens[1:]
        if len(tokens) != 1 or "=" not in tokens[0]:
            continue  # not a simple KEY=VALUE line
        key, value = tokens[0].split("=", 1)
        if _KEY_RE.match(key):
            out[key] = value
    return out


def _snapshot_podlink_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k.startswith("PODLINK_")}


def _compute_base_env() -> dict[str, str]:
    """The PODLINK_* env as if the server had been launched with NO profile.

    The startup env is base conf ∪ launch profile (profile wins) ∪ whatever the
    user exported by hand. Backing the launch profile's keys out and re-parsing
    the base conf over the remainder recovers base values the profile overrode,
    while keeping hand-exported extras (e.g. PODLINK_AUTO_TERMINATE_MIN).
    """
    env = _snapshot_podlink_env()
    launch = env.pop("PODLINK_PROFILE", None)
    if launch and valid_name(launch):
        for key in parse_conf(PROFILE_DIR / f"{launch}.conf"):
            env.pop(key, None)
    env.update(parse_conf(BASE_CONF))
    return env


BASE_ENV = _compute_base_env()  # captured once, at server startup


def effective_env(profile: str | None) -> dict[str, str]:
    """The full PODLINK_* env for `profile` (None = base conf only)."""
    env = dict(BASE_ENV)
    if profile:
        env.update(parse_conf(PROFILE_DIR / f"{profile}.conf"))
        env["PODLINK_PROFILE"] = profile  # surfaced in the console UI
    else:
        env.pop("PODLINK_PROFILE", None)
    # Cloud-specific fix-ups (e.g. resolving a saved storage id the way the
    # launcher would) belong to the provider the env names, not here.
    return providers.normalise_env(env)


def _swap_env(env: dict[str, str]) -> None:
    """Make `env` the process's complete PODLINK_* environment."""
    for key in _snapshot_podlink_env():
        if key not in env:
            del os.environ[key]
    os.environ.update(env)


def apply(profile: str | None) -> None:
    """Switch the process env to `profile` and re-bake the active provider.

    The provider re-reads every PODLINK_* setting from the environment, so each
    holder of a reference sees the new configuration without re-importing. A
    profile naming a different PODLINK_PROVIDER swaps the provider itself.

    On a failure (e.g. a non-numeric PODLINK_VOLUME_GB in a conf, or an unknown
    provider name) the previous env is restored and the provider re-baked again
    so process state stays consistent, then the error propagates for the route
    to surface.
    """
    before = _snapshot_podlink_env()
    _swap_env(effective_env(profile))
    try:
        providers.reload_active()
    except Exception:
        _swap_env(before)
        providers.reload_active()          # re-bake the previous configuration
        raise
