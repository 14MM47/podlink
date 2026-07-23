# podlink

A standalone **local web console** for a GPU inference pod on [RunPod](https://runpod.io).
It reduces the pod lifecycle to two buttons — **POD UP** / **POD DOWN** — and surfaces
everything you'd otherwise chase through the RunPod dashboard: provisioning progress,
per-service health, live cost, and the config block your app needs. Bound to
`127.0.0.1` only.

podlink drives one **bundled RAG inference image** (see [`pod_image/`](pod_image/)) that
co-hosts three OpenAI/TEI-compatible services on a single GPU:

| Port | Service | Readiness gate |
|------|---------|----------------|
| 8000 | vLLM — LLM, served as `$LLM_SERVED_NAME` (default `llm`) | `GET /v1/models` → 200 |
| 8080 | TEI — embedder | `GET /health` → 200 |
| 8081 | TEI — reranker | `GET /health` → 200 |

The image, models, and served name are all configurable (see [Configuration](#configuration)),
so podlink is a generic controller for any such three-service RAG pod — not tied to a
particular stack.

![podlink console](podlinkscreen.png)

## The console

Beyond the two buttons, the UI is a live status console (dark "frosted tactical" skin;
all motion respects `prefers-reduced-motion`):

- **POD UP / POD DOWN** — POD DOWN **terminates** (see below), and is live from the
  instant POD UP starts through the whole boot, so you can kill a pod cleanly anytime.
- **Live cost meter** — uptime × the pod's `$/hr`, ticking every second.
- **Idle auto-terminate** — an optional watchdog that terminates a forgotten pod after
  `PODLINK_AUTO_TERMINATE_MIN`; the UI shows a countdown + a **Keep alive** button.
- **Per-service health tiles** — LLM / embedder / reranker, re-probed continuously.
- **Streamed status panels** — Provisioning & timeline (with per-step deltas), Service
  health, and System, split by source; every create-retry and phase shows here.
- **Test stack** — fires a real completion + embedding + rerank at the pod, reporting
  pass/fail + latency and detecting the embedding dimension.
- **Copy .env** — the exact client config block for the running pod (bearer left as a
  placeholder), ready to paste. See [wiring a client](#wiring-a-client).
- **Network Volume** line + a red banner if none is configured (POD DOWN would then
  destroy weights).

## Terminate, not stop — and the Network Volume

POD DOWN **terminates** the pod rather than stopping it. A *stopped* pod is pinned to
its original host and can fail to resume when that host has no free GPU
(*"not enough free GPUs on the host machine"*); terminate always releases the GPU
cleanly and the next POD UP creates a fresh pod on **any** host.

So weights survive a terminate, podlink attaches a pre-created RunPod **Network Volume**
(region-locked, survives terminate) at `/workspace`, where the image's
`HF_HOME=/workspace/hf` cache lives. Create it once in **RunPod → Storage → Network
Volumes**, sized **75–100 GB**, in a **data center that stocks your GPU** (the volume
pins the pod to its region). A warm POD UP re-pulls the image but **skips the weight
download** — the payoff of the volume.

> **Cost:** a Network Volume bills storage 24/7 even with no pod running
> (~$4–7/month for 75 GB). When the pod is down there is no GPU cost.

If no volume id is set, podlink falls back to a pod-scoped **Data Volume** that is
**destroyed on terminate** (weights re-download next up). In that mode the UI shows a
warning banner and POD DOWN requires an explicit confirmation.

## Quick start

```bash
# 1. one-time: three secrets (Prerequisites) + a stack config (Configuration)
# 2. one-time: create a Network Volume, note its id
./start.sh          # prompts for the Network Volume id (saved), then serves
                    # http://127.0.0.1:8765
./start.sh --check  # preflight only: config + venv/deps + secrets, no launch
```

`start.sh` sources `~/.config/podlink/podlink.conf` (if present) for your stack
config, resolves the Network Volume id (env → saved file → prompt), sets up the venv,
and hands off to the hardened `run.sh` (localhost bind).

## Prerequisites

podlink reads three secrets from `~/.config/podlink/`, each owned by you and mode
`0600` (`_secrets.py` refuses anything more permissive):

| File | What it holds |
|------|---------------|
| `runpod_api_key`   | RunPod API key (create / terminate / GPU lookup). |
| `hf_token`         | Hugging Face token for the weight pull — seen by all three services. |
| `pod_bearer_token` | vLLM + TEI API key; gates all three services; also your client's `*_API_KEY`. |

```bash
mkdir -p ~/.config/podlink && chmod 700 ~/.config/podlink
printf '%s' 'YOUR_RUNPOD_API_KEY'   > ~/.config/podlink/runpod_api_key
printf '%s' 'YOUR_HF_TOKEN'         > ~/.config/podlink/hf_token
printf '%s' 'YOUR_POD_BEARER_TOKEN' > ~/.config/podlink/pod_bearer_token
chmod 600 ~/.config/podlink/*
```

## Configuration

Everything account- or stack-specific is read from the environment — nothing is baked
into source. Put your values in `~/.config/podlink/podlink.conf` (a shell file of
`export`s that `start.sh` sources), or export them yourself:

```bash
# ~/.config/podlink/podlink.conf  (example)
export PODLINK_IMAGE=ghcr.io/you/rag-pod:latest      # your pushed image (required)
export PODLINK_REGISTRY_AUTH_ID=<runpod-cred-id>     # only for a PRIVATE image
export PODLINK_LLM_SERVED_NAME=llm                   # vLLM --served-model-name
export PODLINK_NETWORK_VOLUME_ID=<volume-id>         # terminate-safe weight persistence
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `PODLINK_IMAGE` | *(required)* | The pushed image tag podlink deploys. |
| `PODLINK_REGISTRY_AUTH_ID` | *(empty)* | RunPod Container-Registry-Auth id for a **private** image; empty ⇒ public. |
| `PODLINK_LLM_MODEL_ID` / `PODLINK_EMBED_MODEL_ID` / `PODLINK_RERANK_MODEL_ID` | balanced 96 GB defaults | HF repos the image serves. |
| `PODLINK_LLM_SERVED_NAME` | `llm` | vLLM `--served-model-name`; a client's model field must match. |
| `PODLINK_LLM_QUANT` | *(empty)* | vLLM `--quantization` (empty auto-detects). |
| `PODLINK_NETWORK_VOLUME_ID` | *(empty)* | Network Volume id. Empty ⇒ Data-Volume fallback (weights destroyed on terminate). `start.sh` prompts/saves this. |
| `PODLINK_POD_NAME` / `PODLINK_TEMPLATE_NAME` | `podlink` / `podlink-pod` | RunPod pod + template names. |
| `PODLINK_AUTO_TERMINATE_MIN` | `0` (off) | Idle auto-terminate window, minutes. |
| `PODLINK_CREATE_RETRIES` / `PODLINK_CREATE_RETRY_DELAY` | `40` / `15` | Host-capacity retry attempts and delay. |
| `PODLINK_START_SSH` | `1` (on) | Enable SSH on the pod for first-boot debug. |

## Pod image (build once)

POD UP pulls one bundled image running all three services under `supervisord`. Build
and push it, then set `PODLINK_IMAGE`. Full instructions (including the Blackwell
`sm_120` base-image pins) are in [`pod_image/README.md`](pod_image/README.md).

## Wiring a client

Bring the pod up, wait until all three tiles are green, and click **Copy .env** — it
emits the block below with the live pod URLs and the detected `EMBEDDING_DIMENSIONS`,
the bearer left as a placeholder for you to paste from `~/.config/podlink/pod_bearer_token`:

```dotenv
LLM_BASE_URL=https://<pod-id>-8000.proxy.runpod.net/v1
LLM_MODEL=<PODLINK_LLM_SERVED_NAME>          # must match vLLM --served-model-name
LLM_API_KEY=<your pod_bearer_token>
EMBEDDING_BASE_URL=https://<pod-id>-8080.proxy.runpod.net/v1
EMBEDDING_MODEL=<PODLINK_EMBED_MODEL_ID>
EMBEDDING_DIMENSIONS=<detected by Test stack>
EMBEDDING_API_KEY=<your pod_bearer_token>    # TEI is key-gated (public proxy)
RERANKER_PROVIDER=api
RERANKER_BASE_URL=https://<pod-id>-8081.proxy.runpod.net   # root, no /rerank
RERANKER_API_KEY=<your pod_bearer_token>     # TEI is key-gated (public proxy)
```

The pod id (and URLs) **changes on every POD UP** (terminate/recreate), so this is a
per-session block. All three services need the bearer — blank keys → 401 on
embedder/reranker. Switching the embedder changes the vector dimension, so recreate
your vector collection and re-ingest.

## Security

- Bound to `127.0.0.1` only — never exposed on the network.
- State-changing POSTs require a per-process token (`X-Podlink-Token`), blocking
  browser-based CSRF. POD DOWN has a server-side destructive-action guard (refuses to
  terminate without a Network Volume unless the caller confirms). Neither defends
  against a malicious process already running as your user.
- Secrets never reach the browser; only the RunPod driver reads them, and error
  messages carry the exception *type* only (never `str(e)`, which can embed keys).
- The SDK's env-echoing `create_pod` stdout is captured and discarded so secrets don't
  leak to logs.

## Tests

```bash
./.venv/bin/python tests/test_driver_smoke.py   # driver logic (stubbed SDK)
./.venv/bin/python tests/test_session.py        # state machine + snapshot
./.venv/bin/python tests/test_server.py         # routes + guards (FastAPI TestClient)
```

## Layout

```
podlink/
├─ start.sh        # launcher (sources config, resolves volume id, then run.sh)
├─ run.sh          # hardened uvicorn launch (localhost bind, no arg pass-through)
├─ pod_control/    # vendored pod-control scripts (see PROVENANCE.md)
├─ pod_image/      # bundled 3-service image: Dockerfile + supervisor + wrappers
├─ tests/          # smoke / session / server tests (no live SDK or GPU needed)
└─ app/
   ├─ session.py         # thread-safe pod state machine + snapshot
   ├─ runpod_driver.py   # non-interactive start/terminate + health/test over pod_control
   ├─ server.py          # FastAPI routes + SSE + watchdogs
   └─ static/            # the console UI (index.html + app.js)
```

## License

MIT — see [LICENSE](LICENSE).
