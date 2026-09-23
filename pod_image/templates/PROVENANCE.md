# Vendored templates

| File | Source | Licence |
|------|--------|---------|
| `qwen3_reranker.jinja` | vLLM `v0.25.1`, `examples/pooling/score/template/qwen3_reranker.jinja` (https://github.com/vllm-project/vllm/blob/v0.25.1/examples/pooling/score/template/qwen3_reranker.jinja), copied unmodified 2026-09-23 | Apache-2.0 (vLLM project) |

Why vendored: the `vllm/vllm-openai` image ships the wheel, not the repo's
`examples/` tree, so the chat template `start-reranker.sh` passes to
`vllm serve --chat-template` for `Qwen/Qwen3-Reranker-*` must be copied in.
Re-check it against the matching vLLM tag whenever `VLLM_IMAGE` is bumped.
