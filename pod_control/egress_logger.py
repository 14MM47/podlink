"""Append-only JSONL audit of every outbound HTTP call.

Contract: if a request was not made through `client()`, it did not happen.
That's how the 'single egress' demo claim stays honest from day one.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "egress.jsonl"

# Redact anything that looks like our bearer token (64 hex chars) from any
# audit-log string field. Defence-in-depth — exception messages from httpx or
# downstream libs can occasionally include header values.
_TOKEN_RE = re.compile(r"[A-Fa-f0-9]{64}")

# Literal secret values registered at runtime (via register_secret). Stripped
# verbatim so a token that ISN'T 64-hex (JWT, base64, short) is still redacted —
# the regex above is only a format-guess backstop.
_SECRETS: set[str] = set()


def register_secret(value: str) -> None:
    """Register a literal secret string to strip from every audit-log field."""
    if value:                       # ignore empty/None
        _SECRETS.add(value)


def _redact(s: str) -> str:
    for secret in _SECRETS:         # literal, format-independent redaction first
        s = s.replace(secret, "[REDACTED]")
    return _TOKEN_RE.sub("[REDACTED]", s)  # then the 64-hex backstop


def _tls_version(resp: httpx.Response, scheme: str) -> str | None:
    """Best-effort TLS version extraction from the underlying socket.

    httpx exposes the network stream via response.extensions; if the SSL
    object is reachable we can read the negotiated protocol (e.g. 'TLSv1.3').
    When the introspection path is unavailable (mock transports, proxies that
    hide the socket), fall back to the scheme-derived flag so the field is
    never silently null on a known-https call.
    """
    try:
        stream = resp.extensions.get("network_stream")
        if stream is not None:
            ssl_obj = stream.get_extra_info("ssl_object")
            if ssl_obj is not None and hasattr(ssl_obj, "version"):
                v = ssl_obj.version()
                if v:
                    return v
    except Exception:
        pass
    return "https" if scheme == "https" else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


class AuditedClient:
    """Thin httpx.Client wrapper that audits every request."""

    def __init__(self, timeout: float = 60.0):
        self._client = httpx.Client(timeout=timeout, follow_redirects=False)

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        parsed = urlparse(url)
        t0 = time.perf_counter()
        record = {
            "ts": _now_iso(),
            "method": method,
            "dst_host": parsed.hostname,
            "dst_port": parsed.port or (443 if parsed.scheme == "https" else 80),
            "scheme": parsed.scheme,
            "tls_version": None,
            "path": parsed.path,
            "stream": kwargs.get("stream", False),
        }
        try:
            resp = self._client.request(method, url, **kwargs)
            record["status"] = resp.status_code
            record["bytes_recv"] = len(resp.content) if not kwargs.get("stream") else None
            record["tls_version"] = _tls_version(resp, parsed.scheme)
            record["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            _append(record)
            return resp
        except Exception as e:
            record["status"] = None
            record["error"] = _redact(f"{type(e).__name__}: {e}")
            record["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            _append(record)
            raise

    def stream(self, method: str, url: str, **kwargs):
        """Streaming context manager that audits start + end of the stream."""
        parsed = urlparse(url)
        t0 = time.perf_counter()
        record = {
            "ts": _now_iso(),
            "method": method,
            "dst_host": parsed.hostname,
            "dst_port": parsed.port or (443 if parsed.scheme == "https" else 80),
            "scheme": parsed.scheme,
            "tls_version": None,
            "path": parsed.path,
            "stream": True,
        }
        return _StreamCtx(self._client, method, url, kwargs, record, t0, parsed.scheme)

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AuditedClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class _StreamCtx:
    def __init__(self, client, method, url, kwargs, record, t0, scheme):
        self._client = client
        self._method = method
        self._url = url
        self._kwargs = kwargs
        self._record = record
        self._t0 = t0
        self._scheme = scheme
        self._resp_ctx = None
        self._resp = None

    def __enter__(self) -> httpx.Response:
        self._resp_ctx = self._client.stream(self._method, self._url, **self._kwargs)
        self._resp = self._resp_ctx.__enter__()
        self._record["status"] = self._resp.status_code
        self._record["tls_version"] = _tls_version(self._resp, self._scheme)
        return self._resp

    def __exit__(self, exc_type, exc, tb):
        self._record["duration_ms"] = round((time.perf_counter() - self._t0) * 1000, 1)
        if exc_type is not None:
            self._record["error"] = _redact(f"{exc_type.__name__}: {exc}")
        _append(self._record)
        return self._resp_ctx.__exit__(exc_type, exc, tb)


def client(timeout: float = 60.0) -> AuditedClient:
    return AuditedClient(timeout=timeout)
