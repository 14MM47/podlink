# podlink bundled pod image

One Docker image that runs **three co-resident services** on a single RTX Pro 6000
(96 GB, Blackwell) — a bundled LLM + embedder + reranker stack for a RAG pipeline:

| Port | Service | Endpoints podlink gates on |
|------|---------|----------------------------|
| 8000 | vLLM (LLM), served as `$LLM_SERVED_NAME` (default `llm`), behind the authproxy | `GET /v1/models` → 200 |
| 8080 | TEI embedder | `GET /health` → 200 |
| 8081 | reranker — TEI (default) or vLLM pooling (`RERANK_BACKEND=vllm`, behind the authproxy) | `GET /health` → 200 |

`supervisord` supervises all three, starting the two TEI services first; `start-vllm.sh`
waits for both `/health` endpoints (up to `VLLM_WAIT_FOR_TEI_S`, default 1200 s, `0` to
skip) before vLLM profiles its KV cache, so the embedder owns its VRAM before vLLM
sizes itself. From the RunPod web terminal, `supervisorctl -c /etc/podlink/supervisord.conf status|tail -200 <service> stderr|restart <service>` inspects and restarts a service in place. Model weights are **not** baked in — they
download on first boot into `$HF_HOME=/workspace/hf` (the persistent volume), so
the download is a one-time cost and later resumes are fast.

## Base images (already pinned)

The two `ARG`s at the top of `Dockerfile` are pinned to Blackwell-capable builds —
override them only to bump versions:

- `VLLM_IMAGE` — `vllm/vllm-openai:v0.25.1-cu129-ubuntu2404` (CUDA 12.9, Ubuntu
  24.04, `sm_120`). Use the `-cu129-ubuntu2404` variant, **not** the bare
  `v0.25.1` tag (Ubuntu 22.04 / glibc 2.35, which the copied TEI binary needs 24.04
  to satisfy). podlink's older `v0.9.2` (CUDA 12.4) does **not** support `sm_120`.
- `TEI_IMAGE` — `ghcr.io/huggingface/text-embeddings-inference:120-1.9` (the sm_120
  build). Bump the version as needed.

## Build & push

```bash
cd podlink/pod_image

# Pick your registry path. RunPod must be able to pull it.
IMAGE=ghcr.io/<you>/rag-pod:latest

docker build -t "$IMAGE" .
docker push "$IMAGE"
```

Then point podlink at it: `export PODLINK_IMAGE="<that value>"`.

- **Private registry?** Add the pull credentials in RunPod (Settings → Container
  Registry Auth) so the pod can pull. Those creds live in RunPod, never in
  `~/.config/podlink/`.

## What the container expects at runtime (injected by podlink)

podlink's `create_pod` passes these as env — you don't set them here:

| Env | Purpose |
|-----|---------|
| `LLM_MODEL_ID` / `EMBED_MODEL_ID` / `RERANK_MODEL_ID` | the three HF repos to serve |
| `LLM_SERVED_NAME` | vLLM `--served-model-name` (a client's model field must match; default `llm`) |
| `LLM_QUANT` | vLLM `--quantization` (e.g. `awq_marlin`); leave empty for an FP8 checkpoint |
| `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION` | vLLM sizing (defaults 32768 / 0.70) |
| `EMBED_MAX_BATCH_TOKENS` | TEI embedder `--max-batch-tokens` (default 4096; keeps warm-up inside the memory vLLM leaves) |
| `VLLM_WAIT_FOR_TEI_S` | how long `start-vllm.sh` waits for both TEI `/health` endpoints before starting vLLM (default 1200, `0` = don't wait) |
| `VLLM_API_KEY` | vLLM bearer (read natively by vLLM; never on argv) |
| `TEI_API_KEY` | gates both TEI services; the wrappers `export API_KEY=$TEI_API_KEY` so TEI reads it from env (not argv). Same value as the vLLM bearer. |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | weight-pull token, seen by all three services |
| `RERANK_BACKEND` | `tei` (default) or `vllm` — which server `start-reranker.sh` runs on :8081 |
| `RERANK_GPU_MEMORY_UTILIZATION`, `RERANK_MAX_MODEL_LEN` | vLLM reranker sizing (defaults 0.12 / 4096); `vllm` backend only. Lower the LLM's `GPU_MEMORY_UTILIZATION` to make room |
| `RERANK_VLLM_EXTRA_ARGS` | extra `vllm serve` flags for rerankers without a built-in preset (space-split). `Qwen/Qwen3-Reranker-*` has a preset (hf-overrides + `templates/qwen3_reranker.jinja`) |
| `RERANK_WAIT_FOR_EMBEDDER_S` | how long the vLLM reranker waits for the embedder's `/health` before profiling memory (default 1200, `0` = don't wait) |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | vLLM's own switch, passed through from `PODLINK_VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`. `0` skips CUDA-graph memory profiling (~39 s on the 30B LLM) in **every** vLLM process in the container (the LLM and a vLLM reranker), so their graphs (~0.65 + ~0.3 GB) sit outside each `--gpu-memory-utilization` budget — only for stacks with GPU headroom. Unset by default |
| `PODLINK_CACHE_BASE` | where the vLLM processes keep their compile caches (default `/workspace/cache`, the persistent volume): `<base>/<service>/{vllm,triton}-<vllm version>/`, one tree per service so they never share files. An explicit `VLLM_CACHE_ROOT` / `TRITON_CACHE_DIR` wins; unwritable base = container-disk defaults. A start that never reached `/health` leaves a marker, and the next start clears that service's cache (self-heal). Old-version trees are not pruned — after an image upgrade, `rm -rf /workspace/cache/*/*-<old version>` from the web terminal reclaims the space |
| `LOG_LEVEL` | TEI log level, default `warn` in both TEI wrappers: TEI's INFO startup line prints its args **including `api_key`**. (TEI reads `LOG_LEVEL`, not `RUST_LOG`.) |

**Auth proxy (nginx, `authproxy.py`):** vLLM's `--api-key` guards only `/v1`,
`/v2` and `/inference` paths (v0.25.1 `serve/utils/server_utils.py`); its root
`/tokenize`, `/detokenize`, `/rerank`, `/score`, `/pooling`, `/classify` answer
without the bearer. So every vLLM process binds `127.0.0.1` (LLM `:18000`, vLLM
reranker `:18081`) and nginx owns the public port, returning **401** on every path
except `/health` unless `Authorization: Bearer <VLLM_API_KEY>` matches exactly
(case-sensitive). It runs first under supervisord, renders its config from
`VLLM_API_KEY` into `/run/podlink/nginx.conf` (0600), and **fails closed**: an empty
key, or one that is not >=16 chars of `[A-Za-z0-9._~+/=-]`, means no proxy and so
no public LLM port (podlink's preflight checks the same rule). TEI ports are not
fronted — TEI gates all of its own routes. Responses stream through unbuffered.

`--served-model-name` comes from `LLM_SERVED_NAME` (podlink passes it from
`PODLINK_LLM_SERVED_NAME`); a client's `LLM_MODEL` must match it.

## Validate on the real card before locking `IMAGE`

The build can succeed and still fail at runtime on Blackwell. On a live RTX Pro 6000:

1. `curl -H "Authorization: Bearer $VLLM_API_KEY" :8000/v1/models` lists your served model.
2. `curl -H "Authorization: Bearer $TEI_API_KEY" :8080/health` and `:8081/health`
   return 200 — and confirm a request **without** the key is rejected (401), i.e.
   the pinned TEI build honors the `API_KEY` env var. If it does not, the ports
   would be publicly open; switch the wrappers to `--api-key` on argv instead.
3. Confirm the LLM quant path works — **FP8 checkpoint** or **AWQ-Marlin (W4A16)**.
   Do **not** use NVFP4: the native MoE NVFP4 path is broken on `sm_120`.

## If the TEI binary won't run

Copying `text-embeddings-router` out of the TEI image assumes its CUDA/glibc deps
are satisfied by the vLLM base. If it errors on start (missing `.so`, glibc
mismatch), fall back to building TEI from source in stage B for `sm_120`
(`runtime_compute_cap=120` per TEI's `Dockerfile-cuda`) instead of copying the
binary.
