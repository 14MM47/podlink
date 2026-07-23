# pod_control — vendored pod-control modules

Low-level RunPod control scripts that podlink wraps. Treat this directory as a
unit: podlink's own orchestration (state machine, non-interactive driver, web
server) lives in `app/`, not here. If you change infra assumptions (pod name,
GPU list, secrets location), change them here and re-test the driver.

Local modifications:
- `egress_logger.py` gained `register_secret()` so the driver can register literal
  secret values for verbatim redaction in the audit log (the original only redacted
  64-hex-looking tokens).
- `pod_up.py` / `pod_down.py` switched the lifecycle from **stop/resume** to
  **terminate/recreate** backed by a RunPod **Network Volume**. A stopped pod is
  pinned to its original host and fails to resume when that host has no free GPU;
  terminate releases the GPU cleanly and the Network Volume keeps the weights.
  `pod_up.py` reads the volume id from `PODLINK_NETWORK_VOLUME_ID` (empty => a
  pod-scoped Data Volume that is destroyed on terminate).

| File | Role |
|------|------|
| `pod_up.py`      | Create/resume the `podlink` RunPod pod; waits for RUNNING; writes `pod_state.json`. Attaches the Network Volume when `PODLINK_NETWORK_VOLUME_ID` is set. |
| `pod_down.py`    | Terminate the pod — releases the GPU; weights persist on the Network Volume (or are lost if none is set). |
| `_secrets.py`    | Read secrets from `~/.config/podlink/` with 0600 + ownership checks. |
| `auth_checks.py` | Positive/negative auth probes against the pod. |
| `egress_logger.py` | Audited httpx client + JSONL egress log. |
| `exit_demo.py`   | Streaming completion + perf record. |
| `log_manual.py`  | Manual egress-audit entry. |
