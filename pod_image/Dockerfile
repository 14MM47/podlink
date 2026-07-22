# syntax=docker/dockerfile:1
#
# podlink bundled pod image — ONE container running THREE co-resident services on
# a single RTX Pro 6000 (96 GB, Blackwell / sm_120):
#
#   :8000  vLLM        — OpenAI-compatible LLM (chat + KG extraction)
#   :8080  TEI         — embedder  (/v1/embeddings, /health)
#   :8081  TEI         — reranker  (/rerank, /health)
#
# supervisord keeps all three alive. Model weights are NOT baked in — they are
# pulled on first boot into $HF_HOME (on the persistent /workspace volume), so the
# multi-GB download is a first-boot-only cost and later resumes are fast.
#
# See README.md in this folder for build/push and Blackwell validation notes.

# ---- stage A: source the sm_120 TEI router binary --------------------------
# The 120-* tag is the Blackwell (compute capability 12.0) build of TEI.
# BOTH base-image ARGs are declared here, BEFORE the first FROM, so they are in
# scope for every FROM below. (An ARG declared after a FROM is stage-scoped and a
# later FROM can't see it — that is why the earlier layout failed to build with
# "base name (${VLLM_IMAGE}) should not be blank".)
ARG TEI_IMAGE=ghcr.io/huggingface/text-embeddings-inference:120-1.9
ARG VLLM_IMAGE=vllm/vllm-openai:v0.25.1-cu129-ubuntu2404
FROM ${TEI_IMAGE} AS tei

# ---- stage B: vLLM base + TEI binary + supervisor --------------------------
# VLLM_IMAGE = v0.25.1-cu129-ubuntu2404: the Ubuntu 24.04 / CUDA 12.9.1 variant of
# v0.25.1. It MATCHES the TEI 120-1.9 runtime base (nvidia/cuda:12.9.1-runtime-
# ubuntu24.04) exactly — same glibc (2.39) and same CUDA 12.9 minor — so the TEI
# binary copied below actually runs.
#   The bare `vllm/vllm-openai:v0.25.1` tag is Ubuntu 22.04 / glibc 2.35, and the
#   24.04-built TEI binary dies on it with "GLIBC_2.38/2.39 not found" (both TEI
#   services crash-loop to FATAL). That mismatch is why THIS is the -ubuntu2404 tag.
#   Do NOT use the CUDA-13 `v0.25.1-ubuntu2404` tag: the TEI binary links CUDA 12.x
#   and a 12->13 bump risks breaking it. Stay on the cu129 variant.
# sm_120 (compute cap 12.0) is in this build's arch list — the same kernels the LLM
# already loads successfully.
FROM ${VLLM_IMAGE}

# TEI router binary — built for sm_120 / glibc 2.39 / CUDA 12.9, matching this base.
COPY --from=tei /usr/local/bin/text-embeddings-router /usr/local/bin/text-embeddings-router
# TEI runtime deps the vLLM image may not ship (libssl for HTTPS weight pulls, CA
# certs). Without them the router can fail to start or fail to reach the HF hub.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libssl3 ca-certificates && rm -rf /var/lib/apt/lists/*

# supervisor runs the three long-lived services inside the one container.
RUN pip install --no-cache-dir supervisor==4.2.5

# Service launch wrappers + supervisor config.
COPY start-vllm.sh start-embedder.sh start-reranker.sh /opt/podlink/
COPY supervisord.conf /etc/podlink/supervisord.conf
RUN chmod +x /opt/podlink/*.sh

# Keep the HF cache on the persistent /workspace volume (podlink mounts it there),
# so weights download once and survive stop/resume.
ENV HF_HOME=/workspace/hf

EXPOSE 8000 8080 8081

# The vLLM image's default ENTRYPOINT is `vllm serve`; clear it so our CMD wins.
# podlink must NOT pass docker_args at create time, or it would override this CMD.
ENTRYPOINT []
CMD ["supervisord", "-c", "/etc/podlink/supervisord.conf"]
