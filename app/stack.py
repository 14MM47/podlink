"""The inference stack's env contract — what the bundled image expects, cloud-free.

The pod image (pod_image/) serves three services under supervisord and reads
its configuration from environment variables: which models, how vLLM is sized,
and the two secrets. That contract belongs to the IMAGE, not to any cloud, so
it lives here where every provider can build the container env from the same
PODLINK_* settings and the same defaults.

The vendored pod_control/pod_up.py carries its own copy of these defaults (it
is a standalone CLI and is edited minimally — see PROVENANCE.md);
tests/test_driver_smoke.py asserts the two agree so they cannot drift.
"""
from __future__ import annotations

import os

# The three service ports the image EXPOSEs, keyed by service name.
SERVICE_PORTS = {"llm": 8000, "embedder": 8080, "reranker": 8081}

# Balanced 96 GB defaults — identical to pod_up.py's.
DEFAULT_LLM_MODEL_ID = "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ"
DEFAULT_EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
DEFAULT_RERANK_MODEL_ID = "BAAI/bge-reranker-v2-m3"
DEFAULT_LLM_SERVED_NAME = "llm"
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_GPU_MEMORY_UTILIZATION = 0.70


def from_env() -> dict:
    """The stack this launch will serve, read live from PODLINK_* (with defaults)."""
    return {
        "image": os.environ.get("PODLINK_IMAGE", "").strip() or None,
        "llm_model_id": os.environ.get("PODLINK_LLM_MODEL_ID", DEFAULT_LLM_MODEL_ID),
        "llm_served_name": os.environ.get("PODLINK_LLM_SERVED_NAME", DEFAULT_LLM_SERVED_NAME),
        "embed_model_id": os.environ.get("PODLINK_EMBED_MODEL_ID", DEFAULT_EMBED_MODEL_ID),
        "rerank_model_id": os.environ.get("PODLINK_RERANK_MODEL_ID", DEFAULT_RERANK_MODEL_ID),
        "llm_quant": os.environ.get("PODLINK_LLM_QUANT", ""),
        "max_model_len": int(os.environ.get("PODLINK_MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN))),
        "gpu_memory_utilization": float(
            os.environ.get("PODLINK_GPU_MEMORY_UTILIZATION", str(DEFAULT_GPU_MEMORY_UTILIZATION))),
        "vllm_extra_args": os.environ.get("PODLINK_VLLM_EXTRA_ARGS", "").strip(),
        "pytorch_cuda_alloc_conf": os.environ.get("PODLINK_PYTORCH_CUDA_ALLOC_CONF", "").strip(),
    }


def container_env(stack: dict, bearer: str, hf: str) -> dict[str, str]:
    """The container's runtime env — secrets + model config, read by the image wrappers.

    Same keys pod_up._pod_env() emits, so the image behaves identically on
    every cloud. The optional tuning passthroughs are only sent when set, so the
    image defaults stay authoritative otherwise.
    """
    env = {
        "HF_TOKEN": hf,                              # weight-pull token, all three services
        "HUGGING_FACE_HUB_TOKEN": hf,                # some loaders read this name instead
        "VLLM_API_KEY": bearer,                      # vLLM reads its key from env, never argv
        "TEI_API_KEY": bearer,                       # the same bearer gates both TEI services
        "LLM_MODEL_ID": stack["llm_model_id"],
        "LLM_SERVED_NAME": stack["llm_served_name"],
        "EMBED_MODEL_ID": stack["embed_model_id"],
        "RERANK_MODEL_ID": stack["rerank_model_id"],
        "LLM_QUANT": stack["llm_quant"],
        "MAX_MODEL_LEN": str(stack["max_model_len"]),
        "GPU_MEMORY_UTILIZATION": str(stack["gpu_memory_utilization"]),
    }
    if stack["vllm_extra_args"]:
        env["VLLM_EXTRA_ARGS"] = stack["vllm_extra_args"]
    if stack["pytorch_cuda_alloc_conf"]:
        env["PYTORCH_CUDA_ALLOC_CONF"] = stack["pytorch_cuda_alloc_conf"]
    return env


def client_env_block(urls: dict[str, str], stack: dict, embedding_dim: int | None) -> str:
    """The RAG-client .env block: base URLs + model names, bearer as a PLACEHOLDER.

    Shared by every provider so the block a user copies is identical whatever
    cloud served it. The bearer must never be filled in — this string reaches
    the browser.
    """
    dim_line = (f"EMBEDDING_DIMENSIONS={embedding_dim}" if embedding_dim
                else "# EMBEDDING_DIMENSIONS=  <- run 'Test stack' to detect the served dimension")
    return "\n".join([
        f"LLM_BASE_URL={urls['llm']}/v1",
        f"LLM_MODEL={stack['llm_served_name']}",
        "LLM_API_KEY=<your pod_bearer_token>",
        f"EMBEDDING_BASE_URL={urls['embedder']}/v1",
        f"EMBEDDING_MODEL={stack['embed_model_id']}",
        dim_line,
        "EMBEDDING_API_KEY=<your pod_bearer_token>",
        "RERANKER_PROVIDER=api",
        f"RERANKER_BASE_URL={urls['reranker']}",
        "RERANKER_API_KEY=<your pod_bearer_token>",
        "KG_EXTRACTION_CONCURRENCY=10",
    ])
