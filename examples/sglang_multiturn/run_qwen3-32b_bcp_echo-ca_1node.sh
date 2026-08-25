#!/usr/bin/env bash
# Single-node 8-GPU ECHO launcher for the packaged echo-0821 environment.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PACKAGE_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
LOCAL_IP="${ECHO_HEAD_IP:-${POD_IP:-$(hostname -I | awk '{print $1}')}}"
PROFILE="${ECHO_1NODE_PROFILE:-smoke}"

export VENV_PATH="${ECHO_VENV_PATH:-${PACKAGE_ROOT}/venv_echo_megatron}"
export ECHO_PYTHON_BIN="${ECHO_PYTHON_BIN:-${PACKAGE_ROOT}/python/cpython-3.10.16-linux-x86_64-gnu/bin/python3.10}"
export MODEL_PATH="${ECHO_MODEL_PATH:-/mnt/cfs_bj_mt/workspace/qiruyi/0-Models/Qwen3-32B}"
export RETRIEVER_MODEL_PATH="${ECHO_RETRIEVER_MODEL_PATH:-/mnt/cfs_bj_mt/workspace/qiruyi/0-Models/Qwen3-Embedding-8B}"
export DATA_DIR="${ECHO_DATA_DIR:-${PACKAGE_ROOT}/dataset/browsecomp-plus-context-folding}"
export RETRIEVER_CORPUS_FILE="${ECHO_RETRIEVER_CORPUS_FILE:-${PACKAGE_ROOT}/downloads/browsecomp-plus-corpus/data}"
export RETRIEVER_DENSE_CACHE="${ECHO_RETRIEVER_DENSE_CACHE:-${PACKAGE_ROOT}/browsecomp_dense_cache_tevatron.pkl}"
if [[ -z "${BCP_ENV_FILE:-}" && -f "${PROJECT_DIR}/ltp/secrets.env.local" ]]; then
    export BCP_ENV_FILE="${PROJECT_DIR}/ltp/secrets.env.local"
fi

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PACKAGE_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${PACKAGE_ROOT}/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PACKAGE_ROOT}/.cache/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PACKAGE_ROOT}/.cache/uv}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${PACKAGE_ROOT}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${PACKAGE_ROOT}/.cache/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${PACKAGE_ROOT}/.cache/cuda}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
unset TRANSFORMERS_CACHE
case "$PROFILE" in
    smoke|tuned-smoke) export WANDB_MODE="${ECHO_WANDB_MODE:-disabled}" ;;
    *)
        if [[ -n "${ECHO_WANDB_MODE:-}" ]]; then
            export WANDB_MODE="$ECHO_WANDB_MODE"
        fi
        ;;
esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export BCP_SKIP_REMOTE_CHECK="${BCP_SKIP_REMOTE_CHECK:-True}"
# Keep the default below the host's ephemeral port range (10000-61000).
export RETRIEVER_PORT="${ECHO_RETRIEVER_PORT:-8088}"
export RAY_NUM_CPUS="${ECHO_RAY_NUM_CPUS:-32}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}"
export SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"
export SGLANG_SAMPLING_BACKEND="${SGLANG_SAMPLING_BACKEND:-pytorch}"
export SGLANG_DISABLE_CUSTOM_ALL_REDUCE="${SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-True}"
export SGLANG_FORCE_NATIVE_CUSTOM_OPS="${SGLANG_FORCE_NATIVE_CUSTOM_OPS:-1}"

export NODES_PER_EXPERIMENT=1
export NNODES=1
export TRAINER_IPS="${LOCAL_IP}"
export HEAD_IP="${LOCAL_IP}"

export ACTOR_TP="${ACTOR_TP:-4}"
export ACTOR_PP="${ACTOR_PP:-1}"
export ACTOR_CP="${ACTOR_CP:-2}"
export REF_TP="${REF_TP:-4}"
export REF_PP="${REF_PP:-1}"
export REF_CP="${REF_CP:-2}"
export ROLLOUT_TP="${ROLLOUT_TP:-8}"

configure_tuned_profile() {
    export RAY_NUM_CPUS="${ECHO_RAY_NUM_CPUS:-64}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
    export N_RESP="${N_RESP:-2}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
    export WORKING_CONTEXT_LENGTH="${WORKING_CONTEXT_LENGTH:-6144}"
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-14336}"
    export MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-3072}"
    export MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-20}"
    export MAX_SUMMARY_ROUNDS="${MAX_SUMMARY_ROUNDS:-2}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
    export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
    export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-16}"
    export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-4096}"
    export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-12288}"
}

case "$PROFILE" in
    smoke)
        export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-32b-bcp-echo-ca-1node-smoke}"
        export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
        export N_RESP="${N_RESP:-2}"
        export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
        export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
        export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
        export WORKING_CONTEXT_LENGTH="${WORKING_CONTEXT_LENGTH:-6144}"
        export MAX_MODEL_LEN="${MAX_MODEL_LEN:-14336}"
        export MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-3072}"
        export MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-20}"
        export MAX_SUMMARY_ROUNDS="${MAX_SUMMARY_ROUNDS:-2}"
        export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-4}"
        export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-2}"
        export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-4096}"
        export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-12288}"
        export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
        export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
        export SAVE_FREQ="${SAVE_FREQ:--1}"
        export TEST_FREQ="${TEST_FREQ:--1}"
        export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
        export TRAINER_LOGGER="${TRAINER_LOGGER:-console}"
        ;;
    tuned-smoke)
        configure_tuned_profile
        export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-32b-bcp-echo-ca-1node-tuned-smoke}"
        export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
        export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
        export SAVE_FREQ="${SAVE_FREQ:--1}"
        export TEST_FREQ="${TEST_FREQ:--1}"
        export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
        export TRAINER_LOGGER="${TRAINER_LOGGER:-console}"
        ;;
    tuned)
        configure_tuned_profile
        export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-32b-bcp-echo-ca-1node-tuned-8k-s5}"
        export TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
        export SAVE_FREQ="${SAVE_FREQ:-50}"
        export TEST_FREQ="${TEST_FREQ:-25}"
        export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
        ;;
    full)
        export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-32b-bcp-echo-ca-1node-32k-s5}"
        ;;
    *)
        echo "[config] ERROR: ECHO_1NODE_PROFILE must be smoke, tuned-smoke, tuned, or full; got ${PROFILE}."
        exit 1
        ;;
esac

# Keep config-only checks from replacing the latest training log.
case "${BCP_PREFLIGHT_ONLY:-False}" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        export BCP_LOG_FILE="${BCP_LOG_FILE:-${PROJECT_DIR}/logs/${EXPERIMENT_NAME}.preflight.log}"
        ;;
esac

exec bash "${PROJECT_DIR}/examples/sglang_multiturn/run_qwen3-32b_bcp_echo-ca_4node.sh" "$@"
