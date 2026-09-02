"""Supervised IAP tunnels: the local path to a VM that has no external IP.

Google's Identity-Aware Proxy forwards a TCP port on the VM to a port on this
machine, over TLS to Google's edge, authorised by the caller's IAM identity.
The sanctioned client is `gcloud compute start-iap-tunnel`; there is no
supported library path, so this module supervises one gcloud process per
service port:

  * bound to 127.0.0.1 only — never a routable address;
  * started by ensure() and proven ready by a real connect to the local port
    (gcloud's "Listening on port" line is not waited on — the socket is);
  * watched by a daemon thread that restarts a tunnel whose process has
    exited (IAP drops idle sessions after an hour; gcloud usually reconnects
    itself, but a process that dies is respawned here), reporting each
    transition as an event the driver drains into the feed;
  * torn down by release(), which is idempotent and never raises.

Limits worth knowing: IAP is not a bulk data plane — Google rate-limits it and
says so — so a heavy re-ingest belongs on the VPC-internal path, not here.

Dependencies are injectable (spawn, port_open) so the tests run with no gcloud
and no network.
"""
from __future__ import annotations

import shutil
import socket
import subprocess
import threading
import time
from typing import Callable

LOOPBACK = "127.0.0.1"
READY_TIMEOUT_S = 30.0       # a tunnel that is not listening by then is not coming
CHECK_INTERVAL_S = 5.0       # supervisor poll cadence
MAX_RESTARTS = 20            # per tunnel, per ensure(); then give up and say so


def _port_open(port: int, host: str = LOOPBACK, timeout: float = 0.5) -> bool:
    """True when something accepts a TCP connection on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class TunnelError(RuntimeError):
    """A tunnel could not be established (gcloud missing, port busy, IAP refused…)."""


class _Tunnel:
    """One service port's tunnel: its process (if any) and its restart tally."""

    def __init__(self, service: str, remote_port: int, local_port: int) -> None:
        self.service = service
        self.remote_port = remote_port
        self.local_port = local_port
        self.proc = None
        self.restarts = 0

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


class TunnelManager:
    """Three supervised gcloud IAP tunnels to one instance."""

    def __init__(self, project: str, zone: str, instance: str, tunnels: dict[str, tuple[int, int]], *,
                 spawn: Callable | None = None, port_open: Callable | None = None,
                 gcloud: str | None = None, ready_timeout: float = READY_TIMEOUT_S,
                 check_interval: float = CHECK_INTERVAL_S) -> None:
        """`tunnels` maps service -> (remote_port, local_port)."""
        self.project, self.zone, self.instance = project, zone, instance
        self.instance_id = f"{zone}/{instance}"
        self._tunnels = [_Tunnel(svc, r, lo) for svc, (r, lo) in tunnels.items()]
        self._spawn = spawn or self._spawn_gcloud
        self._port_open = port_open or _port_open
        self._gcloud = gcloud
        self._ready_timeout = ready_timeout
        self._check_interval = check_interval
        self._events: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._watcher: threading.Thread | None = None

    # --- public ------------------------------------------------------------

    def ensure(self) -> None:
        """Start every tunnel that is not already up, prove each is listening,
        and start the supervisor. Idempotent: alive tunnels are left alone."""
        for t in self._tunnels:
            if t.alive:
                continue
            self._start(t, first=True)
        if self._watcher is None or not self._watcher.is_alive():
            self._stop.clear()
            self._watcher = threading.Thread(target=self._watch, name="podlink-iap-watch", daemon=True)
            self._watcher.start()

    def release(self) -> None:
        """Stop the supervisor and every tunnel process. Never raises."""
        self._stop.set()
        for t in self._tunnels:
            self._kill(t)
        self._event("tunnels closed")

    def alive(self) -> dict[str, bool]:
        return {t.service: t.alive for t in self._tunnels}

    def drain_events(self) -> list[str]:
        """Messages since the last drain (the driver feeds them to the UI)."""
        with self._lock:
            out, self._events = self._events, []
        return out

    # --- internals ---------------------------------------------------------

    def _event(self, msg: str) -> None:
        with self._lock:
            self._events.append(f"tunnel: {msg}")

    def _spawn_gcloud(self, t: _Tunnel):
        exe = self._gcloud or shutil.which("gcloud")
        if not exe:
            raise TunnelError("gcloud is not on PATH — required for IAP tunnels "
                              "(or set PODLINK_GCP_ACCESS=internal)")
        # argv list, never a shell; the zone/name were validated upstream, and
        # the local bind is loopback by construction.
        argv = [exe, "compute", "start-iap-tunnel", self.instance, str(t.remote_port),
                f"--local-host-port={LOOPBACK}:{t.local_port}",
                f"--zone={self.zone}", f"--project={self.project}", "--quiet"]
        return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True)

    def _start(self, t: _Tunnel, *, first: bool) -> None:
        if self._port_open(t.local_port):
            # Something already listens here: a tunnel from a podlink that
            # crashed, or an unrelated service. Neither is ours to adopt.
            raise TunnelError(f"local port {t.local_port} ({t.service}) is already in use — a stale "
                              f"tunnel from an earlier run, or another service; free it or change "
                              f"PODLINK_GCP_LOCAL_PORTS")
        t.proc = self._spawn(t)
        deadline = time.time() + self._ready_timeout
        while time.time() < deadline:
            if t.proc.poll() is not None:                    # died before listening
                raise TunnelError(f"{t.service} tunnel exited before listening: {self._stderr_line(t)}")
            if self._port_open(t.local_port):
                self._event(f"{t.service} -> {LOOPBACK}:{t.local_port} {'up' if first else 'restored'}")
                return
            time.sleep(0.2)
        self._kill(t)
        raise TunnelError(f"{t.service} tunnel did not start listening on {t.local_port} within "
                          f"{self._ready_timeout:.0f}s")

    def _watch(self) -> None:
        """Respawn a tunnel whose process has exited, until release() or the
        restart budget is spent. Never raises out of the thread."""
        while not self._stop.wait(self._check_interval):
            for t in self._tunnels:
                if t.alive or self._stop.is_set():
                    continue
                if t.restarts >= MAX_RESTARTS:
                    continue                                 # already reported below
                t.restarts += 1
                self._event(f"{t.service} tunnel dropped ({self._stderr_line(t)}) — restarting "
                            f"({t.restarts}/{MAX_RESTARTS})")
                try:
                    self._start(t, first=False)
                except Exception as e:  # noqa: BLE001 — keep supervising the others
                    self._event(f"{t.service} restart failed: {e}")
                    if t.restarts >= MAX_RESTARTS:
                        self._event(f"{t.service} tunnel given up after {MAX_RESTARTS} restarts — "
                                    f"POD DOWN and POD UP to re-establish")

    def _kill(self, t: _Tunnel) -> None:
        proc, t.proc = t.proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 — escalate
                proc.kill()
                proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 — a process we cannot signal is not worth failing a stop for
            pass

    @staticmethod
    def _stderr_line(t: _Tunnel) -> str:
        """The first useful stderr line from a dead process, truncated — for a
        human, never a log of the whole stream."""
        try:
            if t.proc is not None and t.proc.stderr is not None:
                for line in t.proc.stderr:
                    line = line.strip()
                    if line:
                        return line[:160]
        except Exception:  # noqa: BLE001
            pass
        return "no output"
