#!/usr/bin/env bash
# Multi-turn Search Tool-Calling ECHO-CA on BrowseComp-Plus: 1 node x 8 B cards, Qwen3-32B.
#
# Thin wrapper that pins the GPFS layout and the Blackwell/CUDA-13 environment,
# then hands off to the synchronous ECHO-CA launcher with NNODES=1.
# Mirrors run_bcp_grpo_1node_blackwell.sh so the two algorithms are comparable
# on a single node: same topology, batch sizes, and GPU retrieval.
#
# Run on the node itself:
#   bash examples/sglang_multiturn/run_bcp_echo_ca_1node_blackwell.sh
#
# Config-only check (no Ray, no GPU):
#   BCP_PREFLIGHT_ONLY=True bash examples/sglang_multiturn/run_bcp_echo_ca_1node_blackwell.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PACKAGE_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"

# ---- Environment (GPFS layout) ----
export VENV_PATH="${VENV_PATH:-${PACKAGE_ROOT}/venv_echo_blackwell}"
export MODEL_PATH="${MODEL_PATH:-${PACKAGE_ROOT}/model/Qwen3-32B}"
export RETRIEVER_MODEL_PATH="${RETRIEVER_MODEL_PATH:-${PACKAGE_ROOT}/model/Qwen3-Embedding-8B}"
export DATA_DIR="${DATA_DIR:-${PACKAGE_ROOT}/dataset/browsecomp-plus-context-folding}"
export RETRIEVER_DENSE_CACHE="${RETRIEVER_DENSE_CACHE:-${PACKAGE_ROOT}/browsecomp_dense_cache_tevatron.pkl}"
export RETRIEVER_MODE="${RETRIEVER_MODE:-dense}"
# GPU retrieval (CPU dense retrieval measured ~1.05 QPS and was the pipeline
# bottleneck); ~16 GiB on cuda:7 shared with the Megatron trainer. Fall back to
# RETRIEVER_DEVICE=cpu with RETRIEVER_TORCH_THREADS=32 if the trainer OOMs.
export RETRIEVER_DEVICE="${RETRIEVER_DEVICE:-cuda:7}"
export RETRIEVER_BATCH_SIZE="${RETRIEVER_BATCH_SIZE:-32}"
export RETRIEVER_MAX_CONCURRENT="${RETRIEVER_MAX_CONCURRENT:-32}"

# ---- Blackwell / CUDA 13 ----
export BCP_CUDA_MAJOR="${BCP_CUDA_MAJOR:-13}"
# Transformer Engine's flash backend is unavailable on Blackwell (sm_100).
export MEGATRON_ATTENTION_BACKEND="${MEGATRON_ATTENTION_BACKEND:-fused}"
export SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"
export SGLANG_MM_ATTENTION_BACKEND="${SGLANG_MM_ATTENTION_BACKEND:-flashinfer}"

# ---- Node topology: single node ----
# The shared node selector defaults to four nodes. For a one-node smoke test,
# select the first entry from the scheduler's trainer list (node 0) by default;
# when NODE_SLICE is nonzero, let the shared selector choose that slice.
export NNODES="${NNODES:-1}"
export NODES_PER_EXPERIMENT="${NODES_PER_EXPERIMENT:-1}"
export NODE_SLICE="${NODE_SLICE:-0}"
if [ -z "${TRAINER_IPS:-}" ] && [ -n "${PADDLE_TRAINERS:-}" ] && [ "${NODE_SLICE}" = "0" ]; then
    TRAINER_IPS="${PADDLE_TRAINERS%%,*}"
fi
export TRAINER_IPS

# ---- Parallelism: 8 GPUs on one node, no CP/PP ----
export ACTOR_TP="${ACTOR_TP:-8}"
export ACTOR_PP="${ACTOR_PP:-1}"
export ACTOR_CP="${ACTOR_CP:-1}"
export REF_TP="${REF_TP:-8}"
export REF_PP="${REF_PP:-1}"
export REF_CP="${REF_CP:-1}"

# ---- Rollout sizing ----
# This is SGLang's static (weights + KV-cache) pool fraction, not the Megatron
# training allocation. In hybrid mode free_cache_engine releases SGLang weights
# and KV cache before training. The 0.70 setting is intentionally aggressive on
# the 183 GiB cards; cuda:7 also hosts the ~16 GiB retriever. Drop to 0.60 first
# if SGLang initialization or the training transition runs out of memory.
export ROLLOUT_TP="${ROLLOUT_TP:-8}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.70}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-128}"
export AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-64}"

# ---- Data ----
export TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.paper.parquet}"
if [ -z "${VAL_FILE:-}" ]; then
    VAL_FILE="[${DATA_DIR}/test.easy.paper.labeled.parquet"
    VAL_FILE+=",${DATA_DIR}/test.medium.paper.labeled.parquet"
    VAL_FILE+=",${DATA_DIR}/test.hard.paper.labeled.parquet]"
fi
export VAL_FILE

# ---- Training sizes ----
# Aligned with the 2-node ECHO-CA Blackwell run and the single-node GRPO
# wrapper (32/8/8) so the algorithms stay comparable.
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-32b-bcp-echo-1node-32k-s5}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export N_RESP="${N_RESP:-8}"
# Run the initial validation pass by default for the single-node test. Override
# with VAL_BEFORE_TRAIN=False when a validation-free smoke run is intended.
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
# Validate before training, then every five optimizer steps. Override with
# TEST_FREQ=1/2/etc. for a denser debugging cadence.
export TEST_FREQ="${TEST_FREQ:-5}"

# ---- Sequence lengths ----
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"

# Keep config-only checks from overwriting the latest training log.
case "${BCP_PREFLIGHT_ONLY:-False}" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        export BCP_LOG_FILE="${BCP_LOG_FILE:-${PROJECT_DIR}/logs/${EXPERIMENT_NAME}.preflight.log}"
        ;;
esac

# The ECHO-CA launcher honours --bcp-experiment-name for its log/ckpt names.
exec bash "${PROJECT_DIR}/examples/sglang_multiturn/run_qwen3-32b_bcp_echo-ca_4node.sh" \
    --bcp-experiment-name "$EXPERIMENT_NAME" "$@"
