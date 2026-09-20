"""State-machine + snapshot tests for PodSession — no SDK, no network, no server.

Run: python3 tests/test_session.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # import the `app` package

from app.session import PodSession, State, SERVICES   # noqa: E402


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise SystemExit(f"FAILED: {name}")


def test_start_transition():
    s = PodSession()
    check("fresh session is IDLE", s.state is State.IDLE)
    check("try_begin_start from IDLE -> True", s.try_begin_start(None) is True)
    check("state is STARTING", s.state is State.STARTING)
    check("start again while STARTING -> False", s.try_begin_start(None) is False)


def test_start_records_target_and_resets():
    s = PodSession()
    s.cost_per_hr = 9.9
    s.services = {n: "healthy" for n in SERVICES}
    s.try_begin_start("podABC")
    check("target recorded as pod_id", s.pod_id == "podABC")
    check("cost meter reset on start", s.cost_per_hr is None)
    check("services reset to unknown", all(v == "unknown" for v in s.services.values()))


def test_services_follow_the_spec_env():
    import os
    from app.session import service_names
    check("stock service names", service_names() == ("llm", "embedder", "reranker"))
    os.environ["PODLINK_SERVICES"] = "llm:8000:/v1/models,tts:8091:/health"
    try:
        check("service_names reads the env at call time", service_names() == ("llm", "tts"))
        s = PodSession()
        s.try_begin_start(None)
        check("a fresh start lays out tiles for the new set", list(s.services) == ["llm", "tts"])
    finally:
        del os.environ["PODLINK_SERVICES"]


def test_stop_transition_and_cancel():
    s = PodSession()
    check("stop from IDLE -> False", s.try_begin_stop() is False)
    s.try_begin_start(None)
    check("stop from STARTING -> True", s.try_begin_stop() is True)
    check("state is STOPPING", s.state is State.STOPPING)
    check("cancel flag raised", s.cancel.is_set())


def test_commit_running_guarded_by_cancel():
    s = PodSession()
    s.try_begin_start(None)
    check("commit_running -> True when not cancelled", s.commit_running() is True)
    check("state RUNNING", s.state is State.RUNNING)
    s2 = PodSession()
    s2.try_begin_start(None)
    s2.cancel.set()
    check("commit_running -> False when a stop won the race", s2.commit_running() is False)


def test_snapshot_button_flags():
    s = PodSession()
    snap = s.snapshot()
    for k in ("state", "phase", "up_enabled", "down_enabled", "services", "events",
              "cost_per_hr", "auto_terminate_in_s", "test_result"):
        check(f"snapshot has {k}", k in snap)
    check("IDLE: up enabled, down disabled", snap["up_enabled"] and not snap["down_enabled"])
    s.try_begin_start(None)
    snap = s.snapshot()
    check("STARTING: up disabled, down enabled", (not snap["up_enabled"]) and snap["down_enabled"])


def test_phase_events_and_categories():
    s = PodSession()
    s.update(phase="creating pod")
    s.update(phase="creating pod")          # unchanged -> no new event
    s.update(phase="running")
    s.add_event("llm: healthy", "health")
    s.add_event("boom", "system")
    cats = {c for _t, c, _m in s.events}
    check("phase changes recorded once each",
          [m for _t, _c, m in s.events if _c == "lifecycle"] == ["creating pod", "running"])
    check("health + system categories present", {"health", "system"} <= cats)


def test_cost_meter_and_auto_terminate_derived():
    s = PodSession()
    s.try_begin_start(None); s.state = State.RUNNING
    s.update(cost_per_hr=2.0, billing_started_at=time.time() - 1800, auto_terminate_at=time.time() + 300)
    snap = s.snapshot()
    check("uptime ~1800s", abs(snap["uptime_s"] - 1800) <= 3)
    check("session cost ~$1.00", abs(snap["session_cost_usd"] - 1.0) <= 0.02)
    check("auto-terminate countdown ~300s", abs(snap["auto_terminate_in_s"] - 300) <= 3)


if __name__ == "__main__":
    print("session tests:")
    test_start_transition()
    test_start_records_target_and_resets()
    test_services_follow_the_spec_env()
    test_stop_transition_and_cancel()
    test_commit_running_guarded_by_cancel()
    test_snapshot_button_flags()
    test_phase_events_and_categories()
    test_cost_meter_and_auto_terminate_derived()
    print("all session tests passed.")
