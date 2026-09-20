# Changelog

All notable changes to podlink are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com); versions follow SemVer.

## [Unreleased]

### Added
- **Multi-service pods**: `PODLINK_SERVICES="name:port:/health-path,…"` declares
  the services the image runs (default: the stock three). Exposed ports, the
  readiness gate, the health tiles, `/status.services_spec` and **Copy .env**
  (one `<NAME>_BASE_URL` per extra service) all follow it; `./start.sh --check`
  validates it. The tiles are now rendered from the snapshot instead of fixed markup.
- `PODLINK_GPU_MATCH` / `PODLINK_GPU_MIN_VRAM_GB` (default `RTX PRO 6000` / `90`):
  target another card from a profile. `resolve_gpu_id()` reads them at call time so
  a console profile switch is honoured.
- `PODLINK_CONTAINER_DISK_GB` (default `55`) for larger extended images.
- `PODLINK_POD_ENV_<KEY>=v` generic pod-env passthrough (lands as `<KEY>=v`); the
  secrets and stock model keys podlink sets cannot be overridden this way.
- `profiles/chat.conf`: a committed example profile — an eight-service voice
  harness pod (fast + think + helper vLLMs, TEI ×2, ASR, TTS, gateway) on an H200.
- **Profile switching from the console**: a profile dropdown above the pod
  selector (`GET /profiles`, `POST /profile/select`) switches the active stack
  for the NEXT POD UP without relaunching start.sh. Confs are parsed (shlex,
  `export PODLINK_*` lines only), never executed; switching is allowed only
  while no pod exists, and pod_up's constants are re-baked via module reload.
- `PODLINK_ADOPT_ON_START` (default on): a console start adopts a RUNNING pod with
  our name, so a restarted console shows the real state.

### Fixed
- **Pod image start order** (`pod_image/`): the two TEI services now start before
  vLLM, and `start-vllm.sh` waits for both `/health` endpoints (bounded by
  `VLLM_WAIT_FOR_TEI_S`, default 20 min) before vLLM profiles its KV cache. This
  removes the first-start OOM seen when the embedder was still allocating as vLLM
  measured free memory, and the opposite race where vLLM took the GPU first and the
  embedder could never allocate. `startretries=50` on all three programs so a few
  early failures no longer leave a service FATAL (one tile pending forever while the
  pod bills). Rebuild and push the image for this to take effect.
- Embedder warm-up no longer OOMs beside a loaded vLLM: `start-embedder.sh` passes
  `--max-batch-tokens` (`EMBED_MAX_BATCH_TOKENS`, default 4096; TEI's default 16384
  warm-up batch needed more activation memory than the ~2 GB left after the weights).
- `supervisorctl` works inside the pod (`unix_http_server` / `rpcinterface` /
  `supervisorctl` sections were missing, so `status`, `tail` and `restart <service>`
  failed with ".ini file does not include supervisorctl section"). A service that
  failed to start can now be restarted in place from the web terminal.
- The readiness feed names the likely cause when the LLM answers but a TEI service
  never listens (failed start, check the container log, POD DOWN/UP).
- A start no longer abandons a billing pod. The readiness wait (pod RUNNING but a
  service not yet answering) used to raise after a fixed 15 min, which is shorter
  than a volume-less 122B boot; the console then showed ERROR / "no pod running"
  while RunPod kept billing. It now warns in the feed every `PODLINK_READY_WARN_S`
  and keeps waiting, gives up only at `PODLINK_READY_TIMEOUT_S` (default 60 min,
  `0` = never), and raises promptly if RunPod reports the pod left RUNNING.
- ERROR with a known pod keeps the health watch probing, keeps the cost meter live,
  and recovers to RUNNING automatically once all three services answer.
- The error line says the pod may still be running and how to recover (POD UP
  re-adopts, POD DOWN stops).

## [0.2.0] — 2026-07-28

### Added
- **Profiles**: `./start.sh --profile <name>` (or `PODLINK_PROFILE`) overlays
  `~/.config/podlink/profiles/<name>.conf` on the base conf, so one podlink
  switches between whole stacks per launch. Active profile + LLM shown in the
  console (`/status`: `active_profile`, `llm_model_id`) and in `--check`.
- `PODLINK_NETWORK_VOLUME_ID=none` sentinel — explicitly volume-less launch that
  skips the saved-id/prompt fallback (Data-Volume mode, weights re-download).
- `PODLINK_VOLUME_GB` — size the pod-scoped Data Volume to the stack's weights.
- `PODLINK_MAX_MODEL_LEN` / `PODLINK_GPU_MEMORY_UTILIZATION` env overrides
  (previously hard-coded constants; image wrappers already read them).
- `PODLINK_VLLM_EXTRA_ARGS` — extra `vllm serve` flags via the image wrapper
  (image ≥ `2026-07b`), e.g. bounded-multimodal caps or `--language-model-only`.
- `PODLINK_PYTORCH_CUDA_ALLOC_CONF` — allocator tuning passthrough
  (e.g. `expandable_segments:True`).
- `--check` now prints the active profile, model ids, and vLLM sizing.

### Fixed
- Template recreation on an image change no longer collides with the account's
  existing template: names are suffixed with the image tag
  (`podlink-pod-<tag>`), fixing POD UP dying at "ensuring pod template".

## [0.1.0] — 2026-07-23

First public release.

### Added
- Two-button local web console (**POD UP** / **POD DOWN**) for a RunPod RAG
  inference pod (LLM + embedder + reranker in one image).
- **Terminate + Network Volume** lifecycle — POD DOWN terminates (no host-pinning);
  weights persist on a region-locked volume across recreate.
- Live status console: cost meter, idle auto-terminate watchdog, per-service health
  tiles, split status panels with a provisioning timeline, **Test stack** (real
  completion/embedding/rerank + dimension detection), and **Copy .env**.
- `start.sh` launcher — sources stack config, resolves the volume id, ensures the venv.
- Fully generic configuration via `PODLINK_*` env vars (image, registry auth, model
  ids, served name, quant, pod/template names, retries, auto-terminate, SSH).
- CI (ruff + three test suites); driver / session / server tests that need no live
  SDK or GPU.

### Security
- Localhost-only bind, per-process CSRF token on state-changing routes, a
  server-side destructive-action guard on POD DOWN, secret redaction (type-only
  error messages), and captured SDK stdout so env-echoed secrets don't reach logs.
