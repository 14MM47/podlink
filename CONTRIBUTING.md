# Contributing to podlink

Thanks for your interest! podlink is a small, focused tool — a local web console
for a RAG inference pod on RunPod. Contributions that keep it small, secure, and
readable are very welcome.

## Dev setup

```bash
git clone https://github.com/14MM47/podlink && cd podlink
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt ruff
```

You do **not** need a RunPod account or a GPU to develop or run the tests — the
tests stub the SDK and never make live calls.

## Running the checks

```bash
ruff check .
python tests/test_driver_smoke.py    # driver logic (stubbed SDK)
python tests/test_session.py         # state machine + snapshot
python tests/test_server.py          # routes + guards (FastAPI TestClient)
```

CI runs exactly these on every push and PR.

## Guidelines

- **Keep secrets out of the browser and out of logs.** Never interpolate `str(e)`
  from the SDK/httpx into user-facing text; use `type(e).__name__`. Never add a new
  outbound call outside `egress_logger.client()`.
- **Preserve the security posture:** localhost bind, the token gate on
  state-changing routes, the destructive-action guard on POD DOWN.
- **No hardcoded account/stack values** — everything account- or deployment-specific
  is read from `PODLINK_*` env vars (see the README's Configuration section).
- **Match the house style:** line-by-line comments, `ruff` clean, tests for new
  behavior. The state machine lives in `app/session.py`; the RunPod orchestration in
  `app/runpod_driver.py`; the vendored pod scripts in `pod_control/` (see its
  `PROVENANCE.md` — treat that directory as a unit).

## Pull requests

Open an issue first for anything non-trivial. Keep PRs focused, describe the change
and how you tested it, and make sure CI is green. By contributing you agree your
work is licensed under the project's MIT license.
