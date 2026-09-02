"""Siloing guard: nothing outside a provider package may name a cloud.

podlink is one app with pluggable clouds. That stays true only if the generic
core — the driver, state machine, server, launcher, UI — never learns which
cloud it is on. Muddle arrives one "just this once" at a time, so this test
turns the rule into CI: it greps the generic core for cloud vocabulary and
fails on any hit, naming the file and line.

Allowed to name clouds: app/providers/<name>/ (the provider itself),
app/providers/__init__.py and base.py (the seam — the registry must list
providers, and the contract uses them as examples), pod_control/ (vendored
RunPod CLI scripts, tracked in PROVENANCE.md), docs, and tests.

Run: python3 tests/test_siloing.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The generic core. Add a file here when it is provider-neutral by design;
# never remove one to make the test pass.
GENERIC_CORE = [
    "app/driver.py",
    "app/stack.py",
    "app/session.py",
    "app/server.py",
    "app/profiles.py",
    "app/preflight.py",
    "app/vendored.py",
    "app/static/app.js",
    "app/static/index.html",
    "start.sh",
    "run.sh",
]

# Vocabulary that betrays a cloud. Word-bounded so "gcp" doesn't match inside
# an unrelated identifier, case-insensitive so comments are caught too.
CLOUD_WORDS = [
    r"runpod", r"network[ _-]?volume", r"data[ _-]?volume",
    r"gcp", r"google", r"gcloud", r"hyperdisk", r"\biap\b", r"compute engine",
    r"aws", r"ec2", r"azure",
]
_PATTERN = re.compile("|".join(f"(?:{w})" for w in CLOUD_WORDS), re.IGNORECASE)


def scan() -> list[str]:
    """Return 'path:line: text' for every cloud mention in the generic core."""
    hits: list[str] = []
    for rel in GENERIC_CORE:
        path = ROOT / rel
        if not path.is_file():
            hits.append(f"{rel}: listed in GENERIC_CORE but missing")
            continue
        for n, text in enumerate(path.read_text().splitlines(), 1):
            if _PATTERN.search(text):
                hits.append(f"{rel}:{n}: {text.strip()}")
    return hits


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise SystemExit(f"FAILED: {name}")


def test_generic_core_names_no_cloud():
    hits = scan()
    for h in hits:
        print(f"        {h}")
    check("generic core names no cloud (see hits above if any)", not hits)


def test_every_known_provider_is_a_package_with_a_class():
    sys.path.insert(0, str(ROOT))
    from app import providers
    for name in providers.KNOWN_PROVIDERS:
        pkg = ROOT / "app" / "providers" / name
        check(f"provider '{name}' is a package", (pkg / "__init__.py").is_file())
        mod = providers._module(name)
        check(f"provider '{name}' exposes PROVIDER", hasattr(mod, "PROVIDER"))


if __name__ == "__main__":
    print("siloing tests:")
    test_generic_core_names_no_cloud()
    test_every_known_provider_is_a_package_with_a_class()
    print("all siloing tests passed.")
