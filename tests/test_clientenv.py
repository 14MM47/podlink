"""app/clientenv.py — the automatic client-env handoff.

Pure-Python checks (no cloud, no live pod): rendering with the real bearer and
the profile's extra lines, the 0600 atomic write, the hook validation rules, the
hook runner (argv, exit code, timeout, output discarded), the preflight rows, and
the end-to-end handoff() gate: nothing is written unless the stack test passed.
"""
import os
import stat
import sys
import tempfile
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# handoff() reads the bearer through the vendored secrets reader; stub it so no
# real secret file is touched.
fake_secrets = types.ModuleType("_secrets")
fake_secrets.runpod_api_key = lambda: "rk_dummy"
fake_secrets.hf_token = lambda: "hf_dummy"
fake_secrets.bearer_token = lambda: "bearer-secret-value-0123456789"
sys.modules["_secrets"] = fake_secrets

from app import clientenv  # noqa: E402
from app.session import PodSession  # noqa: E402

FAILED = []


def check(name, cond):
    print(("  PASS  " if cond else "  FAIL  ") + name)
    if not cond:
        FAILED.append(name)


BLOCK = "\n".join([
    "LLM_BASE_URL=https://x-8000.example/v1",
    "LLM_API_KEY=<your pod_bearer_token>",
    "EMBEDDING_API_KEY=<your pod_bearer_token>",
    "KG_EXTRACTION_CONCURRENCY=10",
])


def test_render_fills_bearer_and_applies_extra():
    out = clientenv.render(BLOCK, "B3AR3R", "KG_EXTRACTION_CONCURRENCY=2; NEW_KEY=v ;")
    lines = out.splitlines()
    check("every placeholder replaced by the bearer",
          "<your" not in out and lines.count("LLM_API_KEY=B3AR3R") == 1 and "EMBEDDING_API_KEY=B3AR3R" in lines)
    check("extra overrides an existing key in place",
          lines.index("KG_EXTRACTION_CONCURRENCY=2") == 3 and "KG_EXTRACTION_CONCURRENCY=10" not in lines)
    check("extra appends a new key", lines[-1] == "NEW_KEY=v")
    check("trailing newline", out.endswith("\n"))
    check("no extra = block unchanged", clientenv.render(BLOCK, "b").count("\n") == 4)
    bad = False
    try:
        clientenv.parse_extra("NOEQUALS")
    except ValueError:
        bad = True
    check("malformed extra raises", bad)
    bad = False
    try:
        clientenv.parse_extra("BAD-KEY=1")
    except ValueError:
        bad = True
    check("non-identifier key raises", bad)


def test_write_file_is_atomic_and_private():
    with tempfile.TemporaryDirectory() as d:
        target = clientenv.write_file(os.path.join(d, "sub", "client.env"), "A=1\n")
        check("file written under a created parent", target.read_text() == "A=1\n")
        check("mode 0600", stat.S_IMODE(target.stat().st_mode) == 0o600)
        clientenv.write_file(str(target), "A=2\n")
        check("overwrite replaces content", target.read_text() == "A=2\n")
        check("no temp files left behind", sorted(os.listdir(target.parent)) == ["client.env"])


def test_validate_hook_rules():
    with tempfile.TemporaryDirectory() as d:
        hook = Path(d) / "hook.sh"
        hook.write_text("#!/bin/sh\nexit 0\n")
        check("relative path refused", clientenv.validate_hook("hook.sh") is not None)
        check("missing file refused", clientenv.validate_hook(os.path.join(d, "nope")) is not None)
        check("directory refused", clientenv.validate_hook(d) is not None)
        hook.chmod(0o644)
        check("non-executable refused", "executable" in clientenv.validate_hook(str(hook)))
        hook.chmod(0o777)
        check("group/other-writable refused", "writable" in clientenv.validate_hook(str(hook)))
        hook.chmod(0o755)
        check("owned, exec, not group-writable -> allowed", clientenv.validate_hook(str(hook)) is None)
        check("empty hook = nothing to validate", clientenv.validate_hook("") is None)


def test_run_hook_argv_exit_code_and_timeout():
    with tempfile.TemporaryDirectory() as d:
        marker = Path(d) / "seen"
        hook = Path(d) / "hook.sh"
        hook.write_text(f'#!/bin/sh\necho "$1" > "{marker}"\necho SECRET-OUTPUT\nexit 3\n')
        hook.chmod(0o700)
        rc, err = clientenv.run_hook(str(hook), Path("/tmp/the.env"))
        check("hook gets the env file as argv[1]", marker.read_text().strip() == "/tmp/the.env")
        check("exit code returned, no error", (rc, err) == (3, None))
        slow = Path(d) / "slow.sh"
        slow.write_text("#!/bin/sh\nsleep 5\n")
        slow.chmod(0o700)
        t0 = time.time()
        rc, err = clientenv.run_hook(str(slow), Path("/tmp/x"), timeout=0.3)
        check("timeout reported", rc is None and "timed out" in err and time.time() - t0 < 3)
        rc, err = clientenv.run_hook("relative.sh", Path("/tmp/x"))
        check("invalid hook not run", rc is None and "absolute" in err)


def _env(**kw):
    for k in ("PODLINK_CLIENT_ENV_FILE", "PODLINK_CLIENT_ENV_EXTRA", "PODLINK_CLIENT_ENV_HOOK"):
        os.environ.pop(k, None)
    os.environ.update(kw)


class _Provider:
    def client_env(self, pod_id, dim):
        return BLOCK.replace("x-8000", f"{pod_id}-8000") + f"\nEMBEDDING_DIMENSIONS={dim}"


def test_handoff_is_a_noop_when_unconfigured():
    _env()
    s = PodSession()
    calls = []
    clientenv.handoff(s, _Provider(), "pod1", lambda sess: calls.append("test"))
    check("unconfigured: stack test not even run", calls == [] and s.client_env is None)


def test_handoff_skips_write_when_stack_test_fails():
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "client.env")
        _env(PODLINK_CLIENT_ENV_FILE=target)
        s = PodSession()

        def failing_test(sess):
            sess.update(test_result={"all_ok": False, "embedding_dim": 4096})
        clientenv.handoff(s, _Provider(), "pod1", failing_test)
        check("failed stack test: file not written", not os.path.exists(target))
        check("failed stack test: status skipped", s.client_env and s.client_env["status"] == "skipped")
        check("failed stack test: feed says so",
              any("did not pass" in m for _, _, m in s.events))


def test_handoff_writes_and_runs_hook_on_pass():
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "client.env")
        marker = Path(d) / "hooked"
        hook = Path(d) / "apply.sh"
        hook.write_text(f'#!/bin/sh\ncp "$1" "{marker}"\nexit 0\n')
        hook.chmod(0o700)
        _env(PODLINK_CLIENT_ENV_FILE=target, PODLINK_CLIENT_ENV_EXTRA="KG_EXTRACTION_CONCURRENCY=2",
             PODLINK_CLIENT_ENV_HOOK=str(hook))
        s = PodSession()

        def passing_test(sess):
            sess.update(test_result={"all_ok": True, "embedding_dim": 4096})
        clientenv.handoff(s, _Provider(), "podZ", passing_test)
        text = Path(target).read_text()
        check("file written with the live pod URL", "podZ-8000" in text)
        check("file carries the REAL bearer", "bearer-secret-value-0123456789" in text and "<your" not in text)
        check("detected dimension included", "EMBEDDING_DIMENSIONS=4096" in text)
        check("extra applied", "KG_EXTRACTION_CONCURRENCY=2" in text and "CONCURRENCY=10" not in text)
        check("file is 0600", stat.S_IMODE(os.stat(target).st_mode) == 0o600)
        check("hook ran with the file", marker.exists() and marker.read_text() == text)
        check("status written + hook ok",
              s.client_env["status"] == "written" and s.client_env["hook"] == "ok"
              and s.client_env["path"] == target)
        check("feed never contains the bearer",
              not any("bearer-secret-value" in m for _, _, m in s.events))
        # A failing hook is reported by exit code only.
        hook.write_text("#!/bin/sh\necho leaking-secret\nexit 7\n")
        s2 = PodSession()
        clientenv.handoff(s2, _Provider(), "podZ", passing_test)
        check("hook failure -> exit code in status", s2.client_env["hook"] == "exit 7")
        check("hook failure -> exit code in feed, no output",
              any("exit 7" in m for _, _, m in s2.events)
              and not any("leaking" in m for _, _, m in s2.events))
        # A hook that fails validation is not run; the file is still written.
        hook.chmod(0o666)
        s3 = PodSession()
        clientenv.handoff(s3, _Provider(), "podZ", passing_test)
        check("invalid hook -> file written, hook 'not run'",
              s3.client_env["status"] == "written" and s3.client_env["hook"].startswith("not run"))
    _env()


def test_handoff_never_raises():
    _env(PODLINK_CLIENT_ENV_FILE="/proc/definitely/not/writable/client.env")
    s = PodSession()
    clientenv.handoff(s, _Provider(), "p", lambda sess: sess.update(test_result={"all_ok": True}))
    check("write failure -> status error, no exception", s.client_env["status"] == "error")
    _env()


def test_preflight_rows():
    _env()
    check("off when unset", [lvl for lvl, _ in clientenv.preflight_rows()] == ["info"])
    with tempfile.TemporaryDirectory() as d:
        _env(PODLINK_CLIENT_ENV_FILE=os.path.join(d, "c.env"))
        check("file dir writable, no hook -> ok + info",
              [lvl for lvl, _ in clientenv.preflight_rows()] == ["ok", "info"])
        _env(PODLINK_CLIENT_ENV_FILE=os.path.join(d, "c.env"), PODLINK_CLIENT_ENV_EXTRA="oops")
        check("bad extra -> fail", "fail" in [lvl for lvl, _ in clientenv.preflight_rows()])
        _env(PODLINK_CLIENT_ENV_FILE=os.path.join(d, "c.env"), PODLINK_CLIENT_ENV_HOOK="rel.sh")
        check("bad hook -> fail", [lvl for lvl, _ in clientenv.preflight_rows()][-1] == "fail")
        hook = Path(d) / "ok.sh"
        hook.write_text("#!/bin/sh\n")
        hook.chmod(0o700)
        _env(PODLINK_CLIENT_ENV_FILE=os.path.join(d, "c.env"), PODLINK_CLIENT_ENV_HOOK=str(hook),
             PODLINK_CLIENT_ENV_EXTRA="A=1;B=2")
        check("good file + extra + hook -> all ok",
              [lvl for lvl, _ in clientenv.preflight_rows()] == ["ok", "ok", "ok"])
    _env()


def test_session_snapshot_carries_client_env():
    s = PodSession()
    check("snapshot has client_env (None)", s.snapshot()["client_env"] is None)
    s.update(client_env={"status": "written"})
    check("snapshot reflects it", s.snapshot()["client_env"] == {"status": "written"})
    s.try_begin_start(None)
    check("a new start clears it", s.client_env is None)


if __name__ == "__main__":
    test_render_fills_bearer_and_applies_extra()
    test_write_file_is_atomic_and_private()
    test_validate_hook_rules()
    test_run_hook_argv_exit_code_and_timeout()
    test_handoff_is_a_noop_when_unconfigured()
    test_handoff_skips_write_when_stack_test_fails()
    test_handoff_writes_and_runs_hook_on_pass()
    test_handoff_never_raises()
    test_preflight_rows()
    test_session_snapshot_carries_client_env()
    if FAILED:
        print(f"\n{len(FAILED)} client-env check(s) FAILED: {FAILED}")
        sys.exit(1)
    print("\nall client-env tests passed.")
