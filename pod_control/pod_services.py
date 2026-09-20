"""The pod's service spec: which services the bundled image runs, on which ports, and
how podlink checks each one's health.

podlink drives ONE image that co-hosts several services on one GPU. The stock image
runs three (vLLM :8000, TEI embedder :8080, TEI reranker :8081); an extended image may
add more (a second vLLM, ASR, TTS, an app gateway). Rather than hard-wiring the three,
everything that enumerates services — the exposed ports, the readiness gate, the
health tiles, the client .env block — reads this one spec:

    PODLINK_SERVICES="llm:8000:/v1/models,embedder:8080:/health,reranker:8081:/health"

One `name:port:health-path` entry per service. `name` is the tile label and the key in
every status payload (`[a-z][a-z0-9_]*`); `port` is exposed through RunPod's HTTPS proxy
(`<pod>-<port>.proxy.runpod.net`); `health-path` is what podlink GETs (with the bearer)
and expects a 200 from. Unset/empty => the stock three, so existing installs and
profiles are unaffected.

This module has no runpod/SDK imports so both the vendored pod_up.py and the web app's
session module can use it.
"""
from __future__ import annotations

import os
import re

DEFAULT_SPEC = "llm:8000:/v1/models,embedder:8080:/health,reranker:8081:/health"
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def parse_services(spec: str | None = None) -> dict[str, tuple[int, str]]:
    """Parse a services spec into an ordered {name: (port, health_path)} map.

    `spec` defaults to $PODLINK_SERVICES, and an unset/blank spec means DEFAULT_SPEC.
    Raises ValueError on a malformed entry — a profile typo must fail preflight, not
    deploy a pod whose tiles never go green.
    """
    if spec is None:
        spec = os.environ.get("PODLINK_SERVICES", "")
    spec = spec.strip() or DEFAULT_SPEC
    out: dict[str, tuple[int, str]] = {}
    seen_ports: set[int] = set()
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"PODLINK_SERVICES entry {item!r}: want name:port:/health-path")
        name, port_s, path = (p.strip() for p in parts)
        if not _NAME_RE.match(name):
            raise ValueError(f"PODLINK_SERVICES entry {item!r}: bad service name {name!r} "
                             "(lowercase letters, digits, underscore; must start with a letter)")
        if name in out:
            raise ValueError(f"PODLINK_SERVICES: duplicate service name {name!r}")
        try:
            port = int(port_s)
        except ValueError:
            raise ValueError(f"PODLINK_SERVICES entry {item!r}: port {port_s!r} is not an integer") from None
        if not 1 <= port <= 65535:
            raise ValueError(f"PODLINK_SERVICES entry {item!r}: port {port} out of range")
        if port in seen_ports:
            raise ValueError(f"PODLINK_SERVICES: port {port} listed twice")
        if not path.startswith("/"):
            raise ValueError(f"PODLINK_SERVICES entry {item!r}: health path must start with '/'")
        out[name] = (port, path)
        seen_ports.add(port)
    if not out:
        raise ValueError("PODLINK_SERVICES parsed to no services")
    return out


def service_names(spec: str | None = None) -> tuple[str, ...]:
    """Just the service names, in spec order (tile order in the console)."""
    return tuple(parse_services(spec))
