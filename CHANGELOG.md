# Changelog

All notable changes to podlink are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com); versions follow SemVer.

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
- `--check` now prints the active profile, model ids, and vLLM sizing.

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
