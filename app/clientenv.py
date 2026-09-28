"""Client-env handoff: hand the RAG client its live config the moment a pod is proven up.

Every POD UP changes the service URLs (a recreated instance has a new id), so the
client app needs its `.env` re-pointed and restarted each time. The console's
"Copy .env" button covers the manual path; this module is the automatic one, so a
profile switch is just POD DOWN → pick profile → POD UP.

Configured per profile (all optional; nothing happens when the file is unset):

  PODLINK_CLIENT_ENV_FILE   where to write the client block. Written 0600 via a
                            temp file + rename, WITH the real bearer — this is the
                            server side; the browser's /pod/env keeps its placeholder.
  PODLINK_CLIENT_ENV_EXTRA  `KEY=VALUE;KEY2=VALUE2` appended to the block (an
                            existing key in the block is overridden), e.g. a
                            per-stack `KG_EXTRACTION_CONCURRENCY=2`.
  PODLINK_CLIENT_ENV_HOOK   absolute path run as `[hook, env_file]` (argv, no shell)
                            after a successful write — the client app's own
                            "apply + restart" script. Only its exit code is
                            reported: its output may contain the secrets.

The write is gated on the stack test passing (a real completion + embedding +
rerank), which also yields the served embedding dimension. A failing test leaves
the file — and so the client — untouched, and says so in the event feed.
"""
from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import time
from pathlib import Path

from .vendored import _secrets, read_secret

PLACEHOLDER = "<your pod_bearer_token>"   # what stack.client_env_block emits for the bearer
HOOK_TIMEOUT_S = 120


def settings() -> dict:
    """The three knobs, read live (a profile switch re-exports them)."""
    return {
        "file": os.environ.get("PODLINK_CLIENT_ENV_FILE", "").strip(),
        "extra": os.environ.get("PODLINK_CLIENT_ENV_EXTRA", "").strip(),
        "hook": os.environ.get("PODLINK_CLIENT_ENV_HOOK", "").strip(),
    }


def configured() -> bool:
    return bool(settings()["file"])


def parse_extra(extra: str) -> list[tuple[str, str]]:
    """`K=V;K2=V2` -> [(K, V), …]. Empty items are skipped; a bad item raises."""
    pairs: list[tuple[str, str]] = []
    for item in extra.split(";"):
        item = item.strip()
        if not item:
            continue
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key or not key.replace("_", "").isalnum():
            raise ValueError(f"PODLINK_CLIENT_ENV_EXTRA item {item!r} is not KEY=VALUE")
        pairs.append((key, value.strip()))
    return pairs


def render(block: str, bearer: str, extra: str = "") -> str:
    """The client block with the real bearer and the profile's extra lines applied.

    An extra whose key already has a line in the block replaces that line (so a
    per-stack value overrides the block's default); new keys are appended.
    """
    lines = block.replace(PLACEHOLDER, bearer).splitlines()
    for key, value in parse_extra(extra):
        prefix = f"{key}="
        for i, line in enumerate(lines):
            if line.startswith(prefix):
                lines[i] = f"{key}={value}"
                break
        else:
            lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def write_file(path: str, text: str) -> Path:
    """Write atomically, owner-only (0600), never leaving a partial or world-readable file."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def validate_hook(hook: str) -> str | None:
    """Why this hook must not run, or None when it may.

    An absolute, existing, regular, executable file owned by this user and not
    writable by anyone else — a profile line must not be able to run something
    another account planted.
    """
    if not hook:
        return None
    if not os.path.isabs(hook):
        return "hook path must be absolute"
    try:
        st = os.stat(hook)
    except OSError as e:
        return f"hook not found: {type(e).__name__}"
    if not stat.S_ISREG(st.st_mode):
        return "hook is not a regular file"
    if st.st_uid != os.getuid():
        return "hook is not owned by this user"
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return "hook is writable by group/others"
    if not os.access(hook, os.X_OK):
        return "hook is not executable"
    return None


def run_hook(hook: str, env_file: Path, timeout: float = HOOK_TIMEOUT_S) -> tuple[int | None, str | None]:
    """Run `[hook, env_file]`; return (exit code, error kind). Output is discarded —
    it may echo the secrets — so only the exit status reaches the event feed."""
    reason = validate_hook(hook)
    if reason:
        return None, reason
    try:
        proc = subprocess.run([hook, str(env_file)], stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None, f"timed out after {int(timeout)}s"
    except OSError as e:
        return None, f"could not run: {type(e).__name__}"
    return proc.returncode, None


def preflight_rows() -> list[tuple[str, str]]:
    """What `start.sh --check` says about the handoff: off, or file dir + extra + hook."""
    cfg = settings()
    if not cfg["file"]:
        return [("info", "client env handoff: off (PODLINK_CLIENT_ENV_FILE unset)")]
    rows: list[tuple[str, str]] = []
    target = Path(cfg["file"]).expanduser()
    if target.exists() and not target.is_file():
        rows.append(("fail", f"client env file {target} exists and is not a file"))
    elif not os.access(target.parent if target.parent.exists() else target.parent.parent, os.W_OK):
        rows.append(("fail", f"client env file: cannot write under {target.parent}"))
    else:
        rows.append(("ok", f"client env file: {target} (written after a passing stack test)"))
    try:
        n = len(parse_extra(cfg["extra"]))
        if n:
            rows.append(("ok", f"client env extra: {n} line(s)"))
    except ValueError as e:
        rows.append(("fail", str(e)))
    if cfg["hook"]:
        reason = validate_hook(cfg["hook"])
        rows.append(("fail", f"client env hook: {reason}") if reason
                    else ("ok", f"client env hook: {cfg['hook']}"))
    else:
        rows.append(("info", "client env hook: none (file is written, nothing restarts the client)"))
    return rows


def handoff(session, provider, pod_id: str, test_stack) -> None:
    """Run after a pod is up (start and recovery paths). Never raises.

    Order: stack test → (only if it passed) render + write → hook. Every step lands
    in the event feed and in session.client_env for the UI.
    """
    cfg = settings()
    if not cfg["file"]:
        return
    try:
        session.add_event("client env: running the stack test before handoff", "system")
        test_stack(session)                                   # sets session.test_result
        result = session.test_result or {}
        if not result.get("all_ok"):
            session.add_event("client env: stack test did not pass — client left untouched", "system")
            session.update(client_env={"status": "skipped", "at": time.time(),
                                       "detail": "stack test failed"})
            return
        dim = result.get("embedding_dim")
        bearer = read_secret(_secrets.bearer_token)
        text = render(provider.client_env(pod_id, dim), bearer, cfg["extra"])
        target = write_file(cfg["file"], text)
        session.add_event(f"client env: written to {target}", "system")
        status = {"status": "written", "at": time.time(), "path": str(target), "hook": None}
        if cfg["hook"]:
            rc, err = run_hook(cfg["hook"], target)
            if err:
                status["hook"] = f"not run: {err}"
                session.add_event(f"client env hook: not run — {err}", "system")
            elif rc == 0:
                status["hook"] = "ok"
                session.add_event("client env hook: exit 0 — client re-pointed", "system")
            else:
                status["hook"] = f"exit {rc}"
                session.add_event(f"client env hook: exit {rc} — check the client's own logs", "system")
        session.update(client_env=status)
    except Exception as e:  # noqa: BLE001 — a handoff failure must never take the session down
        session.add_event(f"client env: error: {type(e).__name__}", "system")
        session.update(client_env={"status": "error", "at": time.time(), "detail": type(e).__name__})
