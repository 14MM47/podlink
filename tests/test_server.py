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
from app.providers import runpod as rp       # noqa: E402  the active provider (owns pod_up)
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


def test_auto_terminate_covers_error_with_a_pod():
    launched = []
    saved = server._launch
    server._launch = lambda target: launched.append(target)
    past = __import__("time").time() - 1
    try:
        _reset()
        S.state = State.ERROR
        S.pod_id = "podErr"
        S.update(auto_terminate_at=past)
        check("ERROR + pod past the deadline: auto-terminate fires",
              server._auto_terminate_tick() is True and S.state == State.STOPPING
              and launched == [server.driver.stop])
        _reset()
        S.state = State.ERROR
        S.update(auto_terminate_at=past)
        check("ERROR without a pod: nothing to terminate",
              server._auto_terminate_tick() is False and S.state == State.ERROR)
        _reset()
        S.state = State.RUNNING
        S.pod_id = "podRun"
        S.update(auto_terminate_at=past + 3600)
        check("before the deadline: no terminate", server._auto_terminate_tick() is False)
    finally:
        server._launch = saved
        _reset()


def test_config_and_status():
    check("/config returns a token", isinstance(TOK, str) and len(TOK) > 10)
    snap = client.get("/status").json()
    check("/status has state + button flags", {"state", "up_enabled", "down_enabled"} <= set(snap))
    check("/status carries the deploy flag", "persistence_configured" in snap)
    check("/status names the active provider", snap.get("provider") == "runpod")
    check("/status carries the llm model id",
          snap.get("llm_model_id") == rp.pod_up.LLM_MODEL_ID)


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
    rp.pod_up.NETWORK_VOLUME_ID = ""            # no volume
    check("down, no volume, no confirm -> 428",
          client.post("/pod/down", headers=H, json={}).status_code == 428)
    check("down, no volume, confirm -> 409 (IDLE rejects)",
          client.post("/pod/down", headers=H, json={"confirm": True}).status_code == 409)
    rp.pod_up.NETWORK_VOLUME_ID = "vol_x"       # volume set
    check("down, volume set, no confirm -> 409 (guard skipped, IDLE rejects)",
          client.post("/pod/down", headers=H, json={}).status_code == 409)
    rp.pod_up.NETWORK_VOLUME_ID = ""


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
    snap = client.get("/status").json()
    check("snapshot exposes the provider's service URLs",
          snap["service_urls"]["llm"].endswith("abc123def-8000.proxy.runpod.net"))
    env = client.get("/pod/env", headers=H).json()["env"]
    check("env has the live pod URLs", "abc123def-8000.proxy.runpod.net" in env)
    check("env uses the served name", f"LLM_MODEL={rp.pod_up.LLM_SERVED_NAME}" in env)
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
        saved = (pconf.PROFILE_DIR, pconf.BASE_CONF, rp.VOL_FILE, dict(pconf.BASE_ENV))
        pconf.PROFILE_DIR = pdir
        pconf.BASE_CONF = Path(td) / "podlink.conf"
        rp.VOL_FILE = Path(td) / "network_volume_id"     # the provider owns the saved-id file
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
            check("provider re-baked the model id",
                  rp.pod_up.LLM_MODEL_ID == "test/alt-model")
            check("volume sentinel 'none' -> empty", rp.pod_up.NETWORK_VOLUME_ID == "")
            check("snapshot llm follows the switch",
                  client.get("/status").json()["llm_model_id"] == "test/alt-model")

            r = client.post("/profile/select", headers=H, json={})
            check("select base -> 200, profile cleared", r.json()["active_profile"] is None)
            check("model id back to the baseline",
                  rp.pod_up.LLM_MODEL_ID != "test/alt-model")
        finally:
            pconf.PROFILE_DIR, pconf.BASE_CONF, rp.VOL_FILE = saved[0], saved[1], saved[2]
            pconf.BASE_ENV = saved[3]
            client.post("/profile/select", headers=H, json={})  # re-bake from the real baseline
            _reset()


def test_cross_cloud_switch_guard():
    """Switching to a DIFFERENT cloud asks the outgoing one for a running pod first.

    Registers a throwaway provider package so the registry can actually switch,
    then drives /profile/select through the three outcomes: outgoing cloud busy
    -> 409; outgoing cloud unverifiable -> 409; outgoing cloud idle -> switch.
    """
    import sys
    import tempfile
    import types

    from app import driver, profiles as pconf, providers

    class FakeProvider:                      # the minimum the switch + snapshot touch
        name = "fakecloud"
        def reload_config(self): pass
        def authenticate(self): pass
        def find_existing(self): return None
        def is_running(self, inst): return False
        def instance_id(self, inst): return inst["id"]
        def persistence_configured(self): return True
        def service_urls(self, i): return {"llm": "", "embedder": "", "reranker": ""}
        def stack_config(self): return {"llm_served_name": "x", "embed_model_id": "y"}
        def snapshot_fields(self):
            return {"provider": self.name, "persistence_configured": True, "persistence_id": "d1",
                    "persistence_label": "Disk", "persistence_off_hint": "", "llm_model_id": "m"}

    fake_pkg = types.ModuleType("app.providers.fakecloud")
    fake_pkg.PROVIDER = FakeProvider
    sys.modules["app.providers.fakecloud"] = fake_pkg
    saved_known = providers.KNOWN_PROVIDERS
    providers.KNOWN_PROVIDERS = saved_known + ("fakecloud",)

    live = rp.PROVIDER.authenticate, rp.PROVIDER.find_existing   # class-level, restored below
    with tempfile.TemporaryDirectory() as td:
        pdir = Path(td) / "profiles"; pdir.mkdir()
        (pdir / "other.conf").write_text("export PODLINK_PROVIDER=fakecloud\n")
        saved = (pconf.PROFILE_DIR, pconf.BASE_CONF, rp.VOL_FILE, dict(pconf.BASE_ENV))
        pconf.PROFILE_DIR, pconf.BASE_CONF, rp.VOL_FILE = pdir, Path(td) / "podlink.conf", Path(td) / "vol"
        try:
            _reset()
            check("starts on runpod", providers.active().name == "runpod")

            # 1) outgoing cloud has a RUNNING podlink pod -> refuse.
            rp.PROVIDER.authenticate = lambda self: None
            rp.PROVIDER.find_existing = lambda self: {"id": "ghost1", "desiredStatus": "RUNNING"}
            r = client.post("/profile/select", headers=H, json={"profile": "other"})
            check("switch refused while the outgoing cloud has a running pod", r.status_code == 409)
            check("refusal names the pod", "ghost1" in r.json()["detail"])
            check("still on runpod after refusal", providers.active().name == "runpod")

            # 2) outgoing cloud cannot be asked -> refuse (fail closed).
            def boom(self): raise RuntimeError("secret unavailable")
            rp.PROVIDER.authenticate = boom
            r = client.post("/profile/select", headers=H, json={"profile": "other"})
            check("switch refused when the outgoing cloud is unverifiable", r.status_code == 409)
            check("refusal says it could not verify", "could not verify" in r.json()["detail"])

            # 3) outgoing cloud idle -> switch goes through, badge data follows.
            rp.PROVIDER.authenticate = lambda self: None
            rp.PROVIDER.find_existing = lambda self: None
            r = client.post("/profile/select", headers=H, json={"profile": "other"})
            check("switch allowed when the outgoing cloud is idle", r.status_code == 200)
            check("snapshot names the new provider", r.json()["provider"] == "fakecloud")
            check("driver now drives the new provider", driver.running_instance_id() is None)

            # Same-cloud switch (fakecloud -> fakecloud via base? no: base is runpod) —
            # switching BACK asks fakecloud, which is idle, so it succeeds.
            r = client.post("/profile/select", headers=H, json={})
            check("switch back to base -> runpod", r.status_code == 200 and r.json()["provider"] == "runpod")
        finally:
            rp.PROVIDER.authenticate, rp.PROVIDER.find_existing = live
            pconf.PROFILE_DIR, pconf.BASE_CONF, rp.VOL_FILE = saved[0], saved[1], saved[2]
            pconf.BASE_ENV = saved[3]
            providers.KNOWN_PROVIDERS = saved_known
            sys.modules.pop("app.providers.fakecloud", None)
            client.post("/profile/select", headers=H, json={})   # re-bake from the real baseline
            _reset()


if __name__ == "__main__":
    print("server tests:")
    test_error_with_known_pod_keeps_cost_meter_live()
    test_auto_terminate_covers_error_with_a_pod()
    test_config_and_status()
    test_status_active_profile()
    test_token_gate()
    test_pod_up_validation()
    test_down_guard()
    test_keepalive_clears_deadline()
    test_env_and_test_guards()
    test_profile_parsing_and_switching()
    test_cross_cloud_switch_guard()
    print("all server tests passed.")
