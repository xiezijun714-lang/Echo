#!/bin/bash
# Multi-turn Search Tool-Calling GRPO on BrowseComp-Plus
# 4 nodes × 8 H100-80GB, Qwen3-32B, Megatron backend, SGLang rollout
#
# Physical context is capped by the local Qwen3-32B config at 40960.
#
# Run from project root: bash examples/sglang_multiturn/run_qwen3-32b_bcp_grpo_4node.sh

set -x
ulimit -n 65535
ulimit -u unlimited

# Log directory
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"
EXPERIMENT_NAME="qwen3-32b-bcp-grpo-4node-32k"
if [ "${1:-}" = "--bcp-experiment-name" ]; then
    if [ -z "${2:-}" ]; then
        echo "[config] ERROR: --bcp-experiment-name requires a value."
        exit 1
    fi
    EXPERIMENT_NAME="$2"
    shift 2
fi
LOG_FILE="${BCP_LOG_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}.log}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${PROJECT_DIR}/val_outputs/${EXPERIMENT_NAME}}"
exec > >(tee "$LOG_FILE") 2>&1

# ---- Environment ----
VENV_PATH="${VENV_PATH:?set VENV_PATH to your Python virtualenv directory}"
source "${PROJECT_DIR}/examples/sglang_multiturn/bcp_node_utils.sh"
select_bcp_nodes

source "${VENV_PATH}/bin/activate"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:$PATH"

# 强制使用 venv 内的 cuDNN
CUDNN_LIB="${VENV_PATH}/lib/python3.10/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"

export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"

ALL_IPS=("$HEAD_IP")
for ip in $WORKER_IPS; do ALL_IPS+=("$ip"); done
NNODES=4
NO_PROXY_LIST="127.0.0.1,localhost,$(IFS=,; echo "${ALL_IPS[*]}")"
export no_proxy="$NO_PROXY_LIST"
export NO_PROXY="$NO_PROXY_LIST"

# Required for Megatron
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
# Fix Ray OpenTelemetry segfault (getenv in grpc_core::GetEnv)
export RAY_enable_open_telemetry=0

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the policy model directory (e.g. Qwen3-32B)}"
MODEL_CONTEXT_LIMIT=${MODEL_CONTEXT_LIMIT:-40960}
DATA_DIR="${DATA_DIR:?set DATA_DIR to the BrowseComp-Plus processed dataset directory}"
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.paper.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/test.paper.parquet}
RETRIEVER_MODEL_PATH="${RETRIEVER_MODEL_PATH:?set RETRIEVER_MODEL_PATH to the embedding model directory (e.g. Qwen3-Embedding-8B)}"
RETRIEVER_CORPUS_FILE="${RETRIEVER_CORPUS_FILE:-${DATA_DIR}/corpus.parquet}"
RETRIEVER_DENSE_CACHE="${RETRIEVER_DENSE_CACHE:-${DATA_DIR}/browsecomp_dense_cache.pkl}"

# ---- Batch sizes ----
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
N_RESP=${N_RESP:-8}

# ---- Parallelism ----
ACTOR_TP=${ACTOR_TP:-8}
ACTOR_PP=${ACTOR_PP:-2}
ACTOR_VPP=null
ACTOR_CP=${ACTOR_CP:-2}

REF_TP=${REF_TP:-8}
REF_PP=${REF_PP:-2}
REF_VPP=null
REF_CP=${REF_CP:-2}

# Inference: TP=8, 4 replicas (1 per DP rank)
ROLLOUT_TP=${ROLLOUT_TP:-8}

# ---- Sequence lengths ----
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-16000}
MAX_PARALLEL_CALLS=${MAX_PARALLEL_CALLS:-5}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-8192}
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-32768}
TOKEN_BUDGET=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ACTOR_TOKEN_BUDGET_PER_GPU=$(((TOKEN_BUDGET + ACTOR_CP - 1) / ACTOR_CP))
REF_TOKEN_BUDGET_PER_GPU=$(((TOKEN_BUDGET + REF_CP - 1) / REF_CP))

TOTAL_GPUS=$((NNODES * 8))
ACTOR_MODEL_PARALLEL_SIZE=$((ACTOR_TP * ACTOR_PP * ACTOR_CP))
REF_MODEL_PARALLEL_SIZE=$((REF_TP * REF_PP * REF_CP))
if [ "$ACTOR_MODEL_PARALLEL_SIZE" -ne "$TOTAL_GPUS" ]; then
    echo "[config] ERROR: ACTOR_TP*ACTOR_PP*ACTOR_CP=${ACTOR_MODEL_PARALLEL_SIZE}, expected ${TOTAL_GPUS}."
    exit 1
fi
if [ "$REF_MODEL_PARALLEL_SIZE" -ne "$TOTAL_GPUS" ]; then
    echo "[config] ERROR: REF_TP*REF_PP*REF_CP=${REF_MODEL_PARALLEL_SIZE}, expected ${TOTAL_GPUS}."
    exit 1
fi
if [ "$TOKEN_BUDGET" -ge "$MAX_MODEL_LEN" ]; then
    echo "[config] ERROR: MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH=${TOKEN_BUDGET} must be < MAX_MODEL_LEN=${MAX_MODEL_LEN}."
    exit 1
fi
if [ "$MAX_MODEL_LEN" -gt "$MODEL_CONTEXT_LIMIT" ]; then
    echo "[config] ERROR: MAX_MODEL_LEN=${MAX_MODEL_LEN} exceeds model context limit=${MODEL_CONTEXT_LIMIT}."
    exit 1
fi

DATA_TOOL_CONFIG_OVERRIDES=(+data.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml")
if [ "${INJECT_DATA_TOOL_SCHEMAS:-True}" = "False" ]; then
    DATA_TOOL_CONFIG_OVERRIDES=(data.tool_config_path=null)
fi

# ---- Ray cluster ----
ray stop --force 2>/dev/null || true
for ip in $WORKER_IPS; do ssh "$ip" "ray stop --force" 2>/dev/null || true; done
sleep 2

ray start --head \
    --node-ip-address="$HEAD_IP" \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-gpus=8

for ip in $WORKER_IPS; do
    ssh "$ip" "export LD_LIBRARY_PATH='${VENV_PATH}/lib/python3.10/site-packages/nvidia/cudnn/lib:\$LD_LIBRARY_PATH' && export RAY_enable_open_telemetry=0 && source ${VENV_PATH}/bin/activate && ray start --address='${HEAD_IP}:6379' --num-gpus=8"
done

for i in $(seq 1 60); do
    NODE_COUNT=$(python3 -c \
        "import ray; ray.init(address='auto', ignore_reinit_error=True); print(len(ray.nodes()))" \
        2>/dev/null)
    if [ "${NODE_COUNT:-0}" -ge "$NNODES" ] 2>/dev/null; then
        echo "[ray] Cluster ready: $NODE_COUNT nodes."; break
    fi
    [ "$i" -eq 60 ] && { echo "[ray] ERROR: timeout."; ray status; exit 1; }
    sleep 2
done

# ---- Retrieval service ----
RETRIEVER_LOG="${LOG_DIR}/browsecomp_retriever.log"
RETRIEVER_CMD=("${VENV_PATH}/bin/python3"
    "$PROJECT_DIR/examples/sglang_multiturn/browsecomp_retrieval_server.py"
    --mode dense
    --model "${RETRIEVER_MODEL_PATH}"
    --device cpu
    --corpus_file "${RETRIEVER_CORPUS_FILE}"
    --host 0.0.0.0 --port 8000
    --batch_size 4
    --dense_cache "${RETRIEVER_DENSE_CACHE}")

fuser -k 8000/tcp 2>/dev/null || true
sleep 1

setsid "${RETRIEVER_CMD[@]}" > "$RETRIEVER_LOG" 2>&1 &
RETRIEVER_PID=$!

cleanup() {
    kill $WATCHDOG_PID 2>/dev/null
    kill $RETRIEVER_PID 2>/dev/null
    ray stop --force 2>/dev/null || true
    for ip in $WORKER_IPS; do ssh "$ip" "source ${VENV_PATH}/bin/activate && ray stop --force" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

for i in $(seq 1 1200); do
    curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1 && break
    if ! kill -0 $RETRIEVER_PID 2>/dev/null; then
        echo "[retriever] Server died. Log:"; cat "$RETRIEVER_LOG"; exit 1
    fi
    sleep 1
done

# ---- Retrieval server watchdog (restarts server if health check fails) ----
(
    while true; do
        sleep 30
        if ! curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
            fuser -k 8000/tcp 2>/dev/null || true
            sleep 2
            setsid "${RETRIEVER_CMD[@]}" >> "$RETRIEVER_LOG" 2>&1 &
            RETRIEVER_PID=$!
            for j in $(seq 1 120); do
                curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1 && break
                kill -0 $RETRIEVER_PID 2>/dev/null || break
                sleep 1
            done
        fi
    done
) &
WATCHDOG_PID=$!

# ---- Training ----
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/examples/sglang_multiturn/config" \
    --config-name='bcp_multiturn_megatron_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    "${DATA_TOOL_CONFIG_OVERRIDES[@]}" \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_TOKEN_BUDGET_PER_GPU} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.20 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP} \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ACTOR_TOKEN_BUDGET_PER_GPU} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.n=${N_RESP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=${MAX_MODEL_LEN} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=${SGLANG_CHUNKED_PREFILL_SIZE} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens=${SGLANG_MAX_PREFILL_TOKENS} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=${MAX_PARALLEL_CALLS} \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.multi_turn.format=${MULTI_TURN_FORMAT:-hermes} \
    +actor_rollout_ref.rollout.multi_turn.inject_tool_schemas=${INJECT_ROLLOUT_TOOL_SCHEMAS:-True} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${REF_TOKEN_BUDGET_PER_GPU} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP} \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${REF_VPP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP} \
    actor_rollout_ref.ref.megatron.param_offload=True \
    algorithm.use_kl_in_reward=False \
    +reward.custom_reward_function.path="$PROJECT_DIR/verl/utils/reward_score/bc_p_llm_judge.py" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_model.enable=False \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES} \
    trainer.project_name='echo' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CKPT_DIR:-${PROJECT_DIR}/ckpt}/${EXPERIMENT_NAME}" \
    trainer.logger='["console", "wandb"]' \
    trainer.save_freq=${SAVE_FREQ:-100} \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    +trainer.master_port_range='[31000,32000]' \
    +trainer.validation_data_dir="${VALIDATION_DATA_DIR}"
