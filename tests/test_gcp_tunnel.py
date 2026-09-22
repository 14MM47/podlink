"""Offline tests for the IAP tunnel manager — no gcloud, no sockets.

`spawn` and `port_open` are injected: a FakeProc stands in for the gcloud
process and a scripted port table stands in for the loopback socket. That is
enough to prove the behaviours that matter — readiness is proven by the port,
a dead process fails the launch with its stderr, a busy port is refused rather
than adopted, the supervisor restarts a dropped tunnel and gives up after the
budget, release is idempotent, and the argv gcloud would receive is loopback-
bound and shell-free.

Run: python3 tests/test_gcp_tunnel.py
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.providers.gcp import tunnel as tm   # noqa: E402
from app.providers.gcp.tunnel import TunnelError, TunnelManager   # noqa: E402

PORTS = {"llm": (8000, 18000), "embedder": (8080, 18080), "reranker": (8081, 18081)}


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise SystemExit(f"FAILED: {name}")


class FakeProc:
    """A gcloud stand-in: alive until told to die; records signals."""

    def __init__(self, stderr=""):
        self.exit = None
        self.stderr = io.StringIO(stderr)
        self.signals = []

    def poll(self):
        return self.exit

    def terminate(self):
        self.signals.append("term"); self.exit = -15

    def kill(self):
        self.signals.append("kill"); self.exit = -9

    def wait(self, timeout=None):
        return self.exit


class Harness:
    """Scripts ports + processes for one TunnelManager."""

    def __init__(self, *, listen_after_spawn=True):
        self.open_ports: set = set()
        self.procs: dict = {}           # local_port -> FakeProc
        self.spawned: list = []         # _Tunnel objects, in order
        self.listen_after_spawn = listen_after_spawn
        self.next_stderr = ""

    def spawn(self, t):
        proc = FakeProc(self.next_stderr)
        self.procs[t.local_port] = proc
        self.spawned.append(t)
        if self.listen_after_spawn:
            self.open_ports.add(t.local_port)     # "gcloud is listening"
        return proc

    def port_open(self, port):
        return port in self.open_ports

    def manager(self, **kw):
        return TunnelManager("proj-1", "europe-west2-b", "podlink", PORTS,
                             spawn=self.spawn, port_open=self.port_open,
                             ready_timeout=kw.pop("ready_timeout", 1.0),
                             check_interval=kw.pop("check_interval", 0.05), **kw)


def test_ensure_starts_three_loopback_tunnels_and_reports():
    h = Harness()
    m = h.manager()
    m.ensure()
    check("three tunnels spawned", [t.service for t in h.spawned] == ["llm", "embedder", "reranker"])
    check("all alive", all(m.alive().values()))
    ev = m.drain_events()
    check("each reports up on loopback",
          len(ev) == 3 and all("127.0.0.1" in e and e.endswith("up") for e in ev))
    check("drain empties the queue", m.drain_events() == [])
    m.ensure()
    check("ensure is idempotent (no respawn)", len(h.spawned) == 3)
    m.release()
    check("release terminates every process", all(p.signals == ["term"] for p in h.procs.values()))
    check("release reports", any("closed" in e for e in m.drain_events()))
    m.release()
    check("release is idempotent", True)


def test_dead_process_fails_ensure_with_stderr():
    h = Harness(listen_after_spawn=False)
    h.next_stderr = "ERROR: (gcloud.compute.start-iap-tunnel) Permission denied: iap.tunnelInstances.accessViaIAP\n"

    def spawn_dead(t):
        p = h.spawn(t); p.exit = 1; return p
    m = TunnelManager("proj-1", "europe-west2-b", "podlink", PORTS, spawn=spawn_dead,
                      port_open=h.port_open, ready_timeout=1.0)
    raised = ""
    try:
        m.ensure()
    except TunnelError as e:
        raised = str(e)
    check("a process that exits before listening fails the launch", "exited before listening" in raised)
    check("the failure carries gcloud's first stderr line", "accessViaIAP" in raised)


def test_never_listens_times_out_and_kills():
    h = Harness(listen_after_spawn=False)
    m = h.manager(ready_timeout=0.3)
    raised = ""
    try:
        m.ensure()
    except TunnelError as e:
        raised = str(e)
    check("a tunnel that never listens times out", "did not start listening" in raised)
    check("the stuck process was terminated", h.procs[18000].signals == ["term"])


def test_busy_local_port_is_refused_not_adopted():
    h = Harness()
    h.open_ports.add(18000)                     # something already there
    m = h.manager()
    raised = ""
    try:
        m.ensure()
    except TunnelError as e:
        raised = str(e)
    check("a port already in use is refused", "already in use" in raised and "18000" in raised)
    check("nothing was spawned onto it", h.spawned == [])


def test_supervisor_restarts_a_dropped_tunnel():
    h = Harness()
    m = h.manager(check_interval=0.05)
    m.ensure(); m.drain_events()
    h.procs[18080].exit = 1                      # the embedder tunnel dies
    h.open_ports.discard(18080)
    deadline = time.time() + 3
    while time.time() < deadline and not (h.procs.get(18080) and h.procs[18080].exit is None
                                          and len(h.spawned) == 4):
        time.sleep(0.05)
    ev = m.drain_events()
    check("the dropped tunnel was respawned", len(h.spawned) == 4 and h.spawned[3].service == "embedder")
    check("drop and restore were reported",
          any("embedder tunnel dropped" in e for e in ev) and any("embedder" in e and "restored" in e for e in ev))
    check("the others were left alone", h.spawned[3].local_port == 18080)
    m.release()


def test_ensure_racing_the_supervisor_spawns_once():
    # The health watch's recovery calls ensure() while the supervisor may be
    # mid-restart of the same dropped tunnel: exactly one respawn, no spurious
    # "already in use" error, no orphaned process.
    import threading
    h = Harness()
    m = h.manager(check_interval=0.02)
    m.ensure(); m.drain_events()
    in_spawn = threading.Event()
    def slow_spawn(t):
        in_spawn.set()
        time.sleep(0.3)                          # the supervisor is inside _start
        return h.spawn(t)
    m._spawn = slow_spawn
    h.procs[18080].exit = 1
    h.open_ports.discard(18080)
    check("supervisor began the restart", in_spawn.wait(2))
    err = None
    try:
        m.ensure()                               # concurrent ensure from another thread
    except Exception as e:  # noqa: BLE001
        err = e
    check("concurrent ensure raised nothing", err is None)
    check("the dropped tunnel was respawned exactly once",
          [t.service for t in h.spawned[3:]] == ["embedder"])
    m.release()


def test_supervisor_gives_up_after_the_budget():
    h = Harness()
    m = h.manager(check_interval=0.02)
    m.ensure(); m.drain_events()
    saved = tm.MAX_RESTARTS
    tm.MAX_RESTARTS = 2
    try:
        # Every respawn dies instantly (port never opens, process exits).
        def spawn_dying(t):
            p = h.spawn(t); p.exit = 1; h.open_ports.discard(t.local_port); return p
        m._spawn = spawn_dying
        h.procs[18000].exit = 1; h.open_ports.discard(18000)
        deadline = time.time() + 3
        while time.time() < deadline and not any("given up" in e for e in list(m._events)):
            time.sleep(0.05)
        ev = m.drain_events()
        check("restarts are budgeted", any("given up after 2 restarts" in e for e in ev))
        check("failure reasons were reported", any("restart failed" in e for e in ev))
    finally:
        tm.MAX_RESTARTS = saved
        m.release()


def test_gcloud_argv_is_loopback_bound_and_shell_free():
    calls = []

    class P:
        def __init__(self, argv, **kw): calls.append((argv, kw)); self._e = None
        def poll(self): return self._e
        def terminate(self): self._e = -15
        def kill(self): self._e = -9
        def wait(self, timeout=None): return self._e
        stderr = io.StringIO("")

    saved = tm.subprocess.Popen
    tm.subprocess.Popen = P
    try:
        m = TunnelManager("proj-1", "europe-west2-b", "podlink", {"llm": (8000, 18000)},
                          port_open=lambda port: bool(calls), gcloud="/usr/bin/gcloud", ready_timeout=1.0)
        m.ensure()
        argv, kw = calls[0]
        check("argv is a list (no shell)", isinstance(argv, list) and "shell" not in kw)
        check("gcloud compute start-iap-tunnel <instance> <port>",
              argv[:5] == ["/usr/bin/gcloud", "compute", "start-iap-tunnel", "podlink", "8000"])
        check("bound to 127.0.0.1 only", "--local-host-port=127.0.0.1:18000" in argv)
        check("zone + project pinned, non-interactive",
              "--zone=europe-west2-b" in argv and "--project=proj-1" in argv and "--quiet" in argv)
        check("stdin closed, stdout discarded", kw["stdin"] is tm.subprocess.DEVNULL and kw["stdout"] is tm.subprocess.DEVNULL)
        m.release()
    finally:
        tm.subprocess.Popen = saved


def test_missing_gcloud_is_a_clear_error():
    saved = tm.shutil.which
    tm.shutil.which = lambda name: None
    try:
        m = TunnelManager("proj-1", "europe-west2-b", "podlink", {"llm": (8000, 18000)},
                          port_open=lambda port: False, ready_timeout=1.0)
        raised = ""
        try:
            m.ensure()
        except TunnelError as e:
            raised = str(e)
        check("no gcloud -> TunnelError naming the fix", "gcloud is not on PATH" in raised and "internal" in raised)
    finally:
        tm.shutil.which = saved


if __name__ == "__main__":
    print("gcp tunnel tests:")
    test_ensure_starts_three_loopback_tunnels_and_reports()
    test_dead_process_fails_ensure_with_stderr()
    test_never_listens_times_out_and_kills()
    test_busy_local_port_is_refused_not_adopted()
    test_supervisor_restarts_a_dropped_tunnel()
    test_ensure_racing_the_supervisor_spawns_once()
    test_supervisor_gives_up_after_the_budget()
    test_gcloud_argv_is_loopback_bound_and_shell_free()
    test_missing_gcloud_is_a_clear_error()
    print("all gcp tunnel tests passed.")
