# Changelog

All notable changes to podlink are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com); versions follow SemVer.

## [Unreleased]

### Added
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
- Provider seam: the soft wait, ERROR recovery and startup adoption run through the
  provider contract, so they cover every cloud. A failed start that already has an
  instance now keeps its access path open (on GCP, the IAP tunnels) until POD DOWN,
  and recovery re-ensures that path before probing. Otherwise recovery would probe
  closed local ports and never succeed.
- Recovery never abandons an instance on doubt: a lookup that fails, or a record with
  no status, counts as still running (restores master's rule after the port).
- An instance in ERROR that the provider confirms is gone settles the console to
  IDLE (meter stopped, error cleared, access path closed), instead of sitting in
  ERROR with the meter running and, on GCP, re-opening tunnels to a dead VM.
- Idle auto-terminate also covers ERROR while an instance exists, so a pod left
  behind by a failed start is not exempt from the safety timer.
- GCP tunnels: tunnel start/kill is serialised, so the supervisor and a concurrent
  `ensure()` from recovery cannot spawn the same tunnel twice.
- Recovery loads provider credentials before its first cloud call.

### Known issues
- Startup adoption checks for a RUNNING instance, then runs the normal start path,
  which looks again. If the instance stops between the two lookups, console start
  creates a fresh instance instead of staying IDLE.

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
