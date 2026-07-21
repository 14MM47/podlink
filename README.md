# podlink

A standalone local webapp that reduces the podlink Phase 0 pod lifecycle to **two
buttons**: **POD UP** and **POD DOWN**. It wraps the original pod-control scripts
(vendored verbatim in `pod_control/`) so you never touch the RunPod dashboard or
a terminal to bring the GPU pod up or take it down.

## Behaviour

| Button | State it's live in | What it does |
|--------|--------------------|--------------|
| **POD UP** | IDLE / ERROR | Creates or resumes the `podlink` pod (RTX 5090 → A6000 fallback), waits for `RUNNING`, then waits until vLLM answers `/v1/models` with `200`. |
| **POD DOWN** | STARTING / RUNNING / ERROR | Cancels any in-flight start, **stops** the pod (GPU billing ends, model-weight volume kept), and **verifies** the pod left `RUNNING` before reporting safe. |

Pod Down is greyed out until Pod Up is pressed. The instant Pod Up starts, Pod Up
greys out and Pod Down goes live — and stays live through the **entire**
provisioning window, so you can kill the pod cleanly at any point.

### Why Pod Down is safe anywhere

RunPod bills at the full GPU rate from the moment a pod is **created**, not from
`RUNNING` — and the original `pod_down.py` relied on `pod_state.json`, which
`pod_up.py` writes only after the pod is fully up. podlink closes that gap: it
captures the pod id the instant the pod is created and, failing that, finds the
pod **by name**, so Pod Down can always locate and stop the pod even mid-boot
before any state file exists. It then polls until the pod has actually left
`RUNNING` before it tells you billing has stopped; if it can't confirm, it shows
a loud `STOP NOT VERIFIED` error rather than a false "safe".

## Prerequisites

Secrets, each mode `0600` and owned by you, in `~/.config/podlink/`:

- `runpod_api_key`
- `hf_token`
- `pod_bearer_token`

## Run

```bash
pip install -r requirements.txt
./run.sh                     # serves http://127.0.0.1:8765
```

## Security

- Bound to `127.0.0.1` only — never exposed on the network.
- State-changing POSTs require a per-process token (`X-Podlink-Token`), which
  blocks browser-based CSRF from other sites. This does **not** protect against a
  malicious process already running as your user.
- Secrets never reach the browser; only the RunPod driver reads them.

## Layout

```
podlink/
├─ pod_control/   # vendored pod-control scripts (see PROVENANCE.md)
└─ app/
   ├─ session.py         # thread-safe pod state machine
   ├─ runpod_driver.py   # non-interactive start/stop over pod_control
   ├─ server.py          # FastAPI routes + SSE
   └─ static/            # two-button UI
```
