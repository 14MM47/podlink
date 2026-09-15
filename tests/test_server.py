"""Route + guard tests using FastAPI's TestClient. No live RunPod calls are made —
every check hits a guard that returns before any SDK call, or reads local state.

Run: python3 tests/test_server.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["PODLINK_ADOPT_ON_START"] = "0"     # no live RunPod lookup when the app imports
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


def test_error_with_known_pod_keeps_cost_meter_live():
    _reset()
    S.state = State.ERROR
    S.pod_id = "podErr"
    S.update(billing_started_at=1.0, cost_per_hr=2.0)
    snap = client.get("/status").json()
    check("ERROR + pod id: uptime still reported", snap["uptime_s"] is not None)
    check("ERROR + pod id: both buttons live", snap["up_enabled"] and snap["down_enabled"])
    _reset()
    S.state = State.ERROR
    snap = client.get("/status").json()
    check("ERROR without a pod: no meter", snap["uptime_s"] is None)
    _reset()


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


def test_profile_parsing_and_switching():
    import tempfile

    from app import profiles as pconf

    with tempfile.TemporaryDirectory() as td:
        pdir = Path(td) / "profiles"
        pdir.mkdir()
        (pdir / "alt.conf").write_text(
            "# a comment line\n"
            "export PODLINK_LLM_MODEL_ID=test/alt-model   # inline comment\n"
            'export PODLINK_VLLM_EXTRA_ARGS="--limit-mm-per-prompt {\\"image\\":1} --enforce-eager"\n'
            "export NOT_PODLINK=1\n"
            "rm -rf /tmp/whatever   # arbitrary shell — must be ignored, never executed\n"
            "export PODLINK_NETWORK_VOLUME_ID=none\n")

        # --- parser: quoting survives, comments stripped, foreign keys dropped ---
        parsed = pconf.parse_conf(pdir / "alt.conf")
        check("parser: plain value + inline comment", parsed["PODLINK_LLM_MODEL_ID"] == "test/alt-model")
        check("parser: quoted extra-args intact",
              parsed["PODLINK_VLLM_EXTRA_ARGS"] == '--limit-mm-per-prompt {"image":1} --enforce-eager')
        check("parser: non-PODLINK key dropped", "NOT_PODLINK" not in parsed)
        check("parser: shell line ignored", len(parsed) == 3)

        # --- routes, against a temp config dir (no saved volume id, no base conf) ---
        saved = (pconf.PROFILE_DIR, pconf.BASE_CONF, pconf.VOL_FILE, dict(pconf.BASE_ENV))
        pconf.PROFILE_DIR = pdir
        pconf.BASE_CONF = Path(td) / "podlink.conf"
        pconf.VOL_FILE = Path(td) / "network_volume_id"
        try:
            _reset()
            check("/profiles without token -> 403", client.get("/profiles").status_code == 403)
            listing = client.get("/profiles", headers=H).json()
            check("/profiles lists the conf", listing["profiles"] == ["alt"])
            check("select without token -> 403", client.post("/profile/select").status_code == 403)
            check("select bad name -> 400",
                  client.post("/profile/select", headers=H, json={"profile": "../evil"}).status_code == 400)
            check("select unknown profile -> 404",
                  client.post("/profile/select", headers=H, json={"profile": "nope"}).status_code == 404)

            S.state = State.RUNNING
            S.pod_id = "abc123def"
            check("select while pod up -> 409",
                  client.post("/profile/select", headers=H, json={"profile": "alt"}).status_code == 409)
            S.state = State.ERROR                    # lingering pod id (unverified terminate)
            check("select in ERROR with pod id -> 409",
                  client.post("/profile/select", headers=H, json={"profile": "alt"}).status_code == 409)

            _reset()
            r = client.post("/profile/select", headers=H, json={"profile": "alt"})
            check("select -> 200", r.status_code == 200)
            check("snapshot shows the profile", r.json()["active_profile"] == "alt")
            check("pod_up re-baked the model id",
                  server.runpod_driver.pod_up.LLM_MODEL_ID == "test/alt-model")
            check("volume sentinel 'none' -> empty", server.runpod_driver.pod_up.NETWORK_VOLUME_ID == "")
            check("snapshot llm follows the switch",
                  client.get("/status").json()["llm_model_id"] == "test/alt-model")

            r = client.post("/profile/select", headers=H, json={})
            check("select base -> 200, profile cleared", r.json()["active_profile"] is None)
            check("model id back to the baseline",
                  server.runpod_driver.pod_up.LLM_MODEL_ID != "test/alt-model")
        finally:
            pconf.PROFILE_DIR, pconf.BASE_CONF, pconf.VOL_FILE = saved[0], saved[1], saved[2]
            pconf.BASE_ENV = saved[3]
            client.post("/profile/select", headers=H, json={})  # re-bake from the real baseline
            _reset()


if __name__ == "__main__":
    print("server tests:")
    test_error_with_known_pod_keeps_cost_meter_live()
    test_config_and_status()
    test_status_active_profile()
    test_token_gate()
    test_pod_up_validation()
    test_down_guard()
    test_keepalive_clears_deadline()
    test_env_and_test_guards()
    test_profile_parsing_and_switching()
    print("all server tests passed.")
