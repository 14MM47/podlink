"""Route + guard tests using FastAPI's TestClient. No live RunPod calls are made —
every check hits a guard that returns before any SDK call, or reads local state.

Run: python3 tests/test_server.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # import the `app` package

from fastapi.testclient import TestClient   # noqa: E402
from app import server                       # noqa: E402
from app.session import State                # noqa: E402

client = TestClient(server.app)
TOK = client.get("/config").json()["token"]
H = {"X-Podlink-Token": TOK}
S = server.SESSION


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise SystemExit(f"FAILED: {name}")


def _reset():
    S.state = State.IDLE
    S.pod_id = None
    S.test_result = None
    S.update(auto_terminate_at=None)


def test_config_and_status():
    check("/config returns a token", isinstance(TOK, str) and len(TOK) > 10)
    snap = client.get("/status").json()
    check("/status has state + button flags", {"state", "up_enabled", "down_enabled"} <= set(snap))
    check("/status carries the deploy flag", "network_volume_configured" in snap)
    check("/status carries the llm model id",
          snap.get("llm_model_id") == server.runpod_driver.pod_up.LLM_MODEL_ID)


def test_status_active_profile():
    import os

    old = os.environ.pop("PODLINK_PROFILE", None)
    try:
        check("no profile -> active_profile null",
              client.get("/status").json()["active_profile"] is None)
        os.environ["PODLINK_PROFILE"] = "agentic"
        check("profile env -> active_profile echoed",
              client.get("/status").json()["active_profile"] == "agentic")
    finally:
        os.environ.pop("PODLINK_PROFILE", None)
        if old is not None:
            os.environ["PODLINK_PROFILE"] = old


def test_token_gate():
    check("/pods without token -> 403", client.get("/pods").status_code == 403)
    check("/pod/keepalive without token -> 403", client.post("/pod/keepalive").status_code == 403)
    check("/pod/test without token -> 403", client.post("/pod/test").status_code == 403)
    check("/pod/env without token -> 403", client.get("/pod/env").status_code == 403)


def test_pod_up_validation():
    _reset()
    check("/pod/up bad target -> 400", client.post("/pod/up", headers=H, json={"target": "bad id!"}).status_code == 400)


def test_down_guard():
    _reset()
    server.runpod_driver.pod_up.NETWORK_VOLUME_ID = ""            # no volume
    check("down, no volume, no confirm -> 428",
          client.post("/pod/down", headers=H, json={}).status_code == 428)
    check("down, no volume, confirm -> 409 (IDLE rejects)",
          client.post("/pod/down", headers=H, json={"confirm": True}).status_code == 409)
    server.runpod_driver.pod_up.NETWORK_VOLUME_ID = "vol_x"       # volume set
    check("down, volume set, no confirm -> 409 (guard skipped, IDLE rejects)",
          client.post("/pod/down", headers=H, json={}).status_code == 409)
    server.runpod_driver.pod_up.NETWORK_VOLUME_ID = ""


def test_keepalive_clears_deadline():
    _reset()
    S.state = State.RUNNING
    S.update(auto_terminate_at=__import__("time").time() + 600)
    check("countdown present", client.get("/status").json()["auto_terminate_in_s"] is not None)
    check("keepalive -> 200", client.post("/pod/keepalive", headers=H).status_code == 200)
    check("keepalive cleared the deadline", client.get("/status").json()["auto_terminate_in_s"] is None)
    _reset()


def test_env_and_test_guards():
    _reset()
    check("/pod/env with no pod -> 409", client.get("/pod/env", headers=H).status_code == 409)
    check("/pod/test not running -> 409", client.post("/pod/test", headers=H).status_code == 409)
    # With a pod up, the client-config block is generated correctly.
    S.pod_id = "abc123def"; S.state = State.RUNNING
    env = client.get("/pod/env", headers=H).json()["env"]
    check("env has the live pod URLs", "abc123def-8000.proxy.runpod.net" in env)
    check("env uses the served name", f"LLM_MODEL={server.runpod_driver.pod_up.LLM_SERVED_NAME}" in env)
    check("bearer is a placeholder, not a secret", "<your pod_bearer_token>" in env)
    _reset()


if __name__ == "__main__":
    print("server tests:")
    test_config_and_status()
    test_status_active_profile()
    test_token_gate()
    test_pod_up_validation()
    test_down_guard()
    test_keepalive_clears_deadline()
    test_env_and_test_guards()
    print("all server tests passed.")
