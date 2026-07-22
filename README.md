# podlink

A standalone local webapp that reduces the podlink pod lifecycle to **two
buttons**: **POD UP** and **POD DOWN**. It wraps the original pod-control scripts
(vendored verbatim in `pod_control/`) so you never touch the RunPod dashboard or
a terminal to bring the GPU pod up or take it down.

POD UP provisions the **ragline three-service stack** on a single **RTX Pro 6000
(96 GB, Blackwell)** from one bundled image (see `pod_image/`):

| Port | Service | Readiness gate |
|------|---------|----------------|
| 8000 | vLLM — LLM (chat + KG extraction), served as `ragline-llm` | `GET /v1/models` → 200 |
| 8080 | TEI — embedder | `GET /health` → 200 |
| 8081 | TEI — reranker | `GET /health` → 200 |

## Behaviour

| Button | State it's live in | What it does |
|--------|--------------------|--------------|
| **POD UP** | IDLE / ERROR | Resolves the RTX Pro 6000 GPU id (live, via `runpod.get_gpus()`), creates or resumes the `podlink` pod, waits for `RUNNING`, then waits until **all three** services are healthy (LLM `/v1/models` + both TEI `/health` return `200`). |
| **POD DOWN** | STARTING / RUNNING / ERROR | Cancels any in-flight start, **stops** the whole pod (GPU billing ends, model-weight volume kept), and **verifies** the pod left `RUNNING` before reporting safe. |

Pod Down is greyed out until Pod Up is pressed. The instant Pod Up starts, Pod Up
greys out and Pod Down goes live — and stays live through the **entire**
provisioning window, so you can kill the pod cleanly at any point.

### Target selection

Above the buttons, a **Target** dropdown chooses what Pod Up acts on:

- **Auto** (default) — create or resume the `podlink` vLLM pod and wait until it
  serves `/v1/models` (the original behaviour).
- **A specific pod** — pick any pod on your RunPod account to adopt; Pod Up
  resumes it (if stopped) and waits for `RUNNING` (no vLLM readiness probe, since
  an arbitrary pod may not serve that endpoint).

The selected pod's id is recorded the instant Pod Up is pressed, so Pod Down
stops **that** pod — including if you hit Down immediately. Pod Down always
**stops** (keeps the volume) regardless of which pod is selected. The list shows
name, status, GPU, and hourly cost, and only ever exposes those fields — never a
pod's environment (which can hold API keys).

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

podlink reads three secrets from `~/.config/podlink/`. Each file must be owned by
you and mode `0600` — `_secrets.py` refuses to read anything more permissive.

| File | What it holds |
|------|---------------|
| `runpod_api_key`   | RunPod API key (drives create / resume / stop / GPU lookup). |
| `hf_token`         | Hugging Face token for the model-weight pull — injected container-wide, so **all three** services (LLM + both TEI) use it. |
| `pod_bearer_token` | vLLM API key (`VLLM_API_KEY`); also used to probe `/v1/models` and doubles as ragline's `LLM_API_KEY`. |

**No other secrets are needed.** TEI (embedder/reranker) is keyless internally.
If you use a private image registry, its pull credentials live in **RunPod**
(Settings → Container Registry Auth), never in `~/.config/podlink/`.

Set them up once:

```bash
mkdir -p ~/.config/podlink
chmod 700 ~/.config/podlink

# Write each secret (printf avoids a trailing newline). Replace the placeholders.
printf '%s' 'YOUR_RUNPOD_API_KEY'   > ~/.config/podlink/runpod_api_key
printf '%s' 'YOUR_HF_TOKEN'         > ~/.config/podlink/hf_token
printf '%s' 'YOUR_POD_BEARER_TOKEN' > ~/.config/podlink/pod_bearer_token

chmod 600 ~/.config/podlink/*
```

Tip: to keep the values out of your shell history, prefix each command with a
space (if `HISTCONTROL=ignorespace`) or paste them into an editor instead.

Verify:

```bash
ls -l ~/.config/podlink            # each file should show -rw------- (0600)
```

## Pod image (build once)

POD UP pulls a single bundled image that runs all three services under
`supervisord`. Build and push it before your first pod up, then set `IMAGE` in
`pod_control/pod_up.py`. Full instructions — including the Blackwell `sm_120`
base-image pins to validate — are in [`pod_image/README.md`](pod_image/README.md).

## Run

```bash
pip install -r requirements.txt
./run.sh                     # serves http://127.0.0.1:8765
```

## Deploying to the RTX Pro 6000

1. **Build & push** the bundled image (`pod_image/`); set `IMAGE` in `pod_control/pod_up.py`.
2. **Confirm the model pins** in `pod_up.py` (`LLM_MODEL_ID` / `EMBED_MODEL_ID` /
   `RERANK_MODEL_ID` / `LLM_QUANT`) exist on HF with a Blackwell-compatible quant
   (FP8 checkpoint, or AWQ-Marlin W4A16 — **never NVFP4** on `sm_120`).
3. **Ensure the three secrets** are in place (above).
4. **POD UP** — first boot resolves the GPU id, creates the pod, and downloads
   weights to the `/workspace` volume (a one-time cost; later resumes are fast).
   The `/workspace` volume is sized (`VOLUME_GB`, ~150 GB) to hold the persistent
   HF cache so weights aren't re-downloaded each boot.
5. **Wire ragline** — the UI shows the three proxy URLs (also saved in
   `pod_state.json`). Point ragline's `.env` at them:

   ```dotenv
   LLM_BASE_URL=https://<pod-id>-8000.proxy.runpod.net/v1
   LLM_MODEL=ragline-llm            # must match vLLM --served-model-name
   LLM_API_KEY=<pod_bearer_token>
   EMBEDDING_BASE_URL=https://<pod-id>-8080.proxy.runpod.net/v1
   EMBEDDING_MODEL=BAAI/bge-m3
   EMBEDDING_DIMENSIONS=1024
   EMBEDDING_API_KEY=<pod_bearer_token>    # TEI is key-gated (public proxy)
   RERANKER_PROVIDER=api
   RERANKER_BASE_URL=https://<pod-id>-8081.proxy.runpod.net   # root, no /rerank
   RERANKER_API_KEY=<pod_bearer_token>     # TEI is key-gated (public proxy)
   KG_EXTRACTION_CONCURRENCY=10
   ```

   The embedder/reranker ports are on RunPod's **public** proxy, so podlink gates
   them with the same bearer as the LLM — ragline must send it (above) or its
   embedding/rerank calls get 401.

## Tests

```bash
python3 tests/test_driver_smoke.py   # stub-based; no RunPod SDK / GPU needed
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
├─ pod_image/     # bundled 3-service image: Dockerfile + supervisor + wrappers
├─ tests/         # stub-based driver smoke tests (no SDK/GPU needed)
└─ app/
   ├─ session.py         # thread-safe pod state machine
   ├─ runpod_driver.py   # non-interactive start/stop over pod_control
   ├─ server.py          # FastAPI routes + SSE
   └─ static/            # two-button UI
```
