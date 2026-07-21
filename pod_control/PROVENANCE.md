# pod_control — vendored pod-control modules

Low-level RunPod control scripts that podlink wraps. Treat this directory as a
unit: podlink's own orchestration (state machine, non-interactive driver, web
server) lives in `app/`, not here. If you change infra assumptions (pod name,
GPU list, secrets location), change them here and re-test the driver.

Local modification: `egress_logger.py` gained `register_secret()` so the driver
can register literal secret values for verbatim redaction in the audit log
(the original only redacted 64-hex-looking tokens).

| File | Role |
|------|------|
| `pod_up.py`      | Create/resume the `podlink` RunPod pod; waits for RUNNING; writes `pod_state.json`. |
| `pod_down.py`    | Stop (not terminate) the pod — ends GPU billing, keeps volume. |
| `_secrets.py`    | Read secrets from `~/.config/podlink/` with 0600 + ownership checks. |
| `auth_checks.py` | Positive/negative auth probes against the pod. |
| `egress_logger.py` | Audited httpx client + JSONL egress log. |
| `exit_demo.py`   | Streaming completion + perf record. |
| `log_manual.py`  | Manual egress-audit entry. |
