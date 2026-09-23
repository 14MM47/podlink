#!/usr/bin/env python3
"""Authenticating reverse proxy for the pod's vLLM ports — renders nginx.conf, then execs nginx.

Why: vLLM's --api-key only guards paths under /v1, /v2 and /inference
(v0.25.1 serve/utils/server_utils.py GUARDED_PREFIX). Its root routes — /rerank,
/score, /pooling, /classify on a pooling server; /tokenize, /detokenize on any
server — answer WITHOUT the bearer, and every container port is reachable over
the cloud's public HTTPS proxy. So each vLLM process binds to 127.0.0.1 on an
internal port, and nginx owns the public port: every path except /health needs
`Authorization: Bearer <token>`, else 401. vLLM keeps its own --api-key behind
this as a second layer on /v1.

Which public ports are fronted:
  :8000 -> 127.0.0.1:18000   always (the LLM)
  :8081 -> 127.0.0.1:18081   only when RERANK_BACKEND=vllm (TEI already gates
                             every route with its own API_KEY, so a tei reranker
                             keeps :8081 itself)

The token comes from VLLM_API_KEY (env, never argv) and is written only into a
0600 config under the container's /run/podlink (not the persistent
/workspace volume). Fails closed: no token, or a token with
characters that could break out of the nginx string literal, means no proxy —
and so no public LLM port.

Stdlib only (runs on the image's system python). Importable for tests:
render() is pure.
"""
from __future__ import annotations

import os
import re
import sys

# Public port -> internal loopback port the vLLM process binds.
LLM_PORTS = (8000, 18000)
RERANK_PORTS = (8081, 18081)

CONF_DIR = "/run/podlink"
CONF_PATH = f"{CONF_DIR}/nginx.conf"

# Bearer charset: RFC 6750 b64token characters. Anything else (quotes, spaces,
# semicolons, braces, backslashes) could escape the quoted map key below.
_TOKEN_RE = re.compile(r"[A-Za-z0-9._~+/=-]{16,}")


class ProxyConfigError(ValueError):
    """The proxy cannot be configured safely — refuse to start it."""


def validate_token(token: str) -> str:
    """Return the token if it is safe to embed in nginx.conf, else raise."""
    if not token:
        raise ProxyConfigError("VLLM_API_KEY is empty; refusing to start an unauthenticated proxy")
    if not _TOKEN_RE.fullmatch(token):
        raise ProxyConfigError(
            "VLLM_API_KEY must be >=16 chars of [A-Za-z0-9._~+/=-]; refusing to embed it in nginx.conf")
    return token


def _regex_literal(token: str) -> str:
    """The token as a PCRE literal. Of the allowed charset only '.' and '+' are
    regex metacharacters; the rest ([A-Za-z0-9_~/=-] outside a class) are literal."""
    return token.replace(".", r"\.").replace("+", r"\+")


def listeners_for(rerank_backend: str) -> list[tuple[int, int]]:
    """The (public, internal) port pairs to front for this stack."""
    pairs = [LLM_PORTS]
    # Only a vLLM reranker needs fronting; TEI gates all of its own routes.
    if rerank_backend == "vllm":
        pairs.append(RERANK_PORTS)
    return pairs


def render(token: str, listeners: list[tuple[int, int]], user: str = "www-data") -> str:
    """nginx.conf text: one server per listener, bearer-gated except /health."""
    token = validate_token(token)
    servers = []
    for public, internal in listeners:
        upstream = f"http://127.0.0.1:{internal}"
        servers.append(f"""
    server {{
        listen {public};
        # Readiness probes stay open, exactly as vLLM serves /health today.
        location = /health {{
            proxy_pass {upstream};
        }}
        # Everything else — including vLLM's unguarded root routes — needs the bearer.
        location / {{
            if ($podlink_bearer_ok = 0) {{
                return 401;
            }}
            proxy_pass {upstream};
        }}
    }}""")
    return f"""# Rendered by /opt/podlink/authproxy.py at container start. Contains the
# bearer — mode 0600 in the container's own /run (never the persistent
# /workspace volume). Do not copy off the pod.
user {user};
worker_processes 2;
daemon off;
pid {CONF_DIR}/nginx.pid;
error_log stderr warn;

events {{
    worker_connections 1024;
}}

http {{
    access_log off;
    server_tokens off;

    # 1 when the Authorization header is exactly our bearer, else 0. A `~` regex
    # key, NOT a plain string key: nginx matches map strings case-INsensitively,
    # which would accept a case-flipped token. `~` is case-sensitive.
    map $http_authorization $podlink_bearer_ok {{
        default 0;
        "~^Bearer {_regex_literal(token)}$" 1;
    }}

    # Long prompts (64K context) and long streamed generations.
    client_max_body_size 64m;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_set_header Host $host;
    # Stream tokens (SSE) straight through instead of buffering the response.
    proxy_buffering off;
    # Request bodies stream too: buffering would spill any body >16 KB (prompts
    # carrying document text) to temp files on the container disk. Framing
    # ambiguity is not a concern: single tenant, behind the cloud's HTTPS proxy.
    proxy_request_buffering off;
    # Never let an upstream response trigger an nginx internal redirect, which
    # would re-enter a location without the client having passed the gate.
    proxy_ignore_headers X-Accel-Redirect;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
{"".join(servers)}
}}
"""


def write_conf(text: str, path: str = CONF_PATH) -> None:
    """Write the config 0600 in a 0700 directory (it contains the bearer)."""
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    # O_TRUNC + explicit mode: a restart rewrites the file, never widens it.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.chmod(path, 0o600)


def main() -> int:
    """Render from env, write the config, and replace this process with nginx."""
    try:
        text = render(os.environ.get("VLLM_API_KEY", ""),
                      listeners_for(os.environ.get("RERANK_BACKEND", "tei").strip().lower()))
    except ProxyConfigError as e:
        # Never echo the token; the message names only the rule it broke.
        print(f"[authproxy] ERROR: {e}", file=sys.stderr)
        return 1
    write_conf(text)
    print("[authproxy] config written; starting nginx", file=sys.stderr)
    os.execvp("nginx", ["nginx", "-c", CONF_PATH])
    return 0  # unreachable: execvp replaces the process


if __name__ == "__main__":
    sys.exit(main())
