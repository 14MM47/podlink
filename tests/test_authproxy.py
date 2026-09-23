"""Offline tests for the in-pod auth proxy renderer (pod_image/authproxy.py).

authproxy.py renders the nginx.conf that fronts the vLLM ports: vLLM's own
--api-key only guards /v1, /v2 and /inference, so nginx must demand the bearer
on EVERY other path (except /health). These tests prove the rendered config's
shape — the gate, the /health exemption, the loopback upstreams, which ports are
fronted per reranker backend — plus the fail-closed token rules, the 0600 write,
and that the in-image rule matches podlink's preflight rule.

A live nginx run of the same config is a manual check (see CHANGELOG); nothing
here needs nginx, docker or network.

Run: python3 tests/test_authproxy.py
"""
from __future__ import annotations

import importlib.util
import os
import stat
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                      # import the `app` package

# pod_image/ is not a package (it is copied into the image as-is), so load the
# renderer by path.
_spec = importlib.util.spec_from_file_location("authproxy", ROOT / "pod_image" / "authproxy.py")
authproxy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(authproxy)

TOKEN = "a" * 32 + "B9._~+/=-"                    # every allowed character class


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise SystemExit(f"FAILED: {name}")


def _raises(fn) -> bool:
    """True when fn() raises ProxyConfigError."""
    try:
        fn()
    except authproxy.ProxyConfigError:
        return True
    return False


def test_listeners_per_backend():
    # :8000 is always fronted; :8081 only for a vLLM reranker (TEI gates itself).
    check("tei: only the LLM port", authproxy.listeners_for("tei") == [(8000, 18000)])
    check("vllm: LLM + reranker ports",
          authproxy.listeners_for("vllm") == [(8000, 18000), (8081, 18081)])
    check("unknown backend: LLM only (reranker script refuses it anyway)",
          authproxy.listeners_for("onnx") == [(8000, 18000)])


def test_rendered_config_gates_everything_but_health():
    conf = authproxy.render(TOKEN, authproxy.listeners_for("vllm"))
    # The bearer map: exact header match -> 1, anything else -> 0.
    check("map defaults to not-authorised", "default 0;" in conf)
    # Case-sensitive, anchored regex key with '.' and '+' escaped (a plain string
    # key would match case-insensitively — found live against nginx).
    escaped = TOKEN.replace(".", "\\.").replace("+", "\\+")
    check("map key is a case-sensitive anchored regex", f'"~^Bearer {escaped}$" 1;' in conf)
    check("no case-insensitive (~*) or plain-string bearer key",
          "~*" not in conf and f'"Bearer {TOKEN}"' not in conf)
    # One server per public port, each proxying to its loopback upstream.
    check("listens on public 8000", "listen 8000;" in conf)
    check("listens on public 8081", "listen 8081;" in conf)
    check("upstream 18000 is loopback", "proxy_pass http://127.0.0.1:18000;" in conf)
    check("upstream 18081 is loopback", "proxy_pass http://127.0.0.1:18081;" in conf)
    check("no public 0.0.0.0 upstream", "0.0.0.0" not in conf)
    # Per server: exactly one exact-match /health exemption and one gated catch-all.
    check("one /health exemption per server", conf.count("location = /health {") == 2)
    check("one gated catch-all per server", conf.count("location / {") == 2)
    check("catch-all returns 401 without the bearer",
          conf.count("if ($podlink_bearer_ok = 0) {\n                return 401;") == 2)
    # Streaming + long-generation settings.
    check("response buffering off (SSE streams)", "proxy_buffering off;" in conf)
    check("long read timeout", "proxy_read_timeout 3600s;" in conf)
    check("upstream X-Accel-Redirect ignored", "proxy_ignore_headers X-Accel-Redirect;" in conf)
    check("foreground for supervisord", "daemon off;" in conf)
    check("no server version leak", "server_tokens off;" in conf)
    check("no access log (keeps request lines out of logs)", "access_log off;" in conf)


def test_tei_render_leaves_8081_alone():
    conf = authproxy.render(TOKEN, authproxy.listeners_for("tei"))
    check("tei: 8081 not fronted", "listen 8081;" not in conf and "18081" not in conf)


def test_token_rules_fail_closed():
    ok = lambda t: authproxy.render(t, [(8000, 18000)])  # noqa: E731
    check("empty token refused", _raises(lambda: ok("")))
    check("short token refused", _raises(lambda: ok("abc123")))
    for bad in ('x' * 20 + '"', 'x' * 20 + ' 1; }', 'x' * 20 + ';', 'x' * 20 + '\\',
                'x' * 20 + '{', 'x' * 20 + '$host', 'x' * 20 + '\n'):
        check(f"token with {bad[20:]!r} refused", _raises(lambda b=bad: ok(b)))
    check("64-hex token accepted", "listen 8000;" in ok("0123456789abcdef" * 4))


def test_write_conf_is_0600():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub", "nginx.conf")
        authproxy.write_conf("x", path)
        mode = stat.S_IMODE(os.stat(path).st_mode)
        dmode = stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode)
        check("config written 0600", mode == 0o600)
        check("config dir 0700", dmode == 0o700)
        # A rewrite never widens the mode, even if someone chmod'ed it open.
        os.chmod(path, 0o644)
        authproxy.write_conf("y", path)
        check("rewrite restores 0600", stat.S_IMODE(os.stat(path).st_mode) == 0o600)


def test_main_fails_closed_without_key():
    saved = os.environ.pop("VLLM_API_KEY", None)
    try:
        # main() must return 1 (supervisor retries, port stays closed), not exec nginx.
        check("main() exits 1 with no VLLM_API_KEY", authproxy.main() == 1)
    finally:
        if saved is not None:
            os.environ["VLLM_API_KEY"] = saved


def test_preflight_rule_matches_image_rule():
    # podlink's preflight must reject exactly what the in-pod proxy rejects, or a
    # launch passes preflight and then boots with no public LLM port.
    from app import preflight
    check("preflight bearer regex == authproxy token regex",
          preflight._BEARER_RE.pattern == authproxy._TOKEN_RE.pattern)


if __name__ == "__main__":
    test_listeners_per_backend()
    test_rendered_config_gates_everything_but_health()
    test_tei_render_leaves_8081_alone()
    test_token_rules_fail_closed()
    test_write_conf_is_0600()
    test_main_fails_closed_without_key()
    test_preflight_rule_matches_image_rule()
    print("all authproxy tests passed.")
