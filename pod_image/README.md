# podlink bundled pod image

One Docker image that runs **three co-resident services** on a single RTX Pro 6000
(96 GB, Blackwell) — a bundled LLM + embedder + reranker stack for a RAG pipeline:

| Port | Service | Endpoints podlink gates on |
|------|---------|----------------------------|
| 8000 | vLLM (LLM), served as `$LLM_SERVED_NAME` (default `llm`) | `GET /v1/models` → 200 |
| 8080 | TEI embedder | `GET /health` → 200 |
| 8081 | TEI reranker | `GET /health` → 200 |

`supervisord` supervises all three. Model weights are **not** baked in — they
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
| `VLLM_API_KEY` | vLLM bearer (read natively by vLLM; never on argv) |
| `TEI_API_KEY` | gates both TEI services; the wrappers `export API_KEY=$TEI_API_KEY` so TEI reads it from env (not argv). Same value as the vLLM bearer. |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | weight-pull token, seen by all three services |

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
