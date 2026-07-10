#!/bin/bash
# Multi-turn Search Tool-Calling with SUPO on BrowseComp-Plus
# SUPO: Summarization augmented Policy Optimization
# 4 nodes × 8 H100-80GB, Qwen3-32B, Megatron backend, SGLang rollout
#
# SUPO enables handling tasks beyond working_context_length via periodic summarization
# - working_context_length (L): actual memory constraint per trajectory
# - max_model_len: SGLang physical context window; this model config caps it at 40960
#
# Run from project root: bash examples/sglang_multiturn/run_qwen3-32b_bcp_supo_4node.sh

set -x
set -euo pipefail
ulimit -n 65535
ulimit -u unlimited

# Log directory
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"
EXPERIMENT_NAME="qwen3-32b-bcp-supo-4node-32k"
if [ "${1:-}" = "--bcp-experiment-name" ]; then
    if [ -z "${2:-}" ]; then
        echo "[config] ERROR: --bcp-experiment-name requires a value."
        exit 1
    fi
    EXPERIMENT_NAME="$2"
    shift 2
fi
LOG_FILE="${BCP_LOG_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}.log}"
exec > >(tee "$LOG_FILE") 2>&1

# ---- Environment ----
VENV_PATH="${VENV_PATH:?set VENV_PATH to your Python virtualenv directory}"
source "${PROJECT_DIR}/examples/sglang_multiturn/bcp_node_utils.sh"
select_bcp_nodes

source "${VENV_PATH}/bin/activate"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:$PATH"

NVIDIA_SITE_PACKAGES="${VENV_PATH}/lib/python3.10/site-packages/nvidia"
VENV_CUDA_LIB_PATH="${NVIDIA_SITE_PACKAGES}/nvjitlink/lib:${NVIDIA_SITE_PACKAGES}/cusparse/lib:${NVIDIA_SITE_PACKAGES}/cublas/lib:${NVIDIA_SITE_PACKAGES}/cudnn/lib:${NVIDIA_SITE_PACKAGES}/cuda_runtime/lib"
export LD_LIBRARY_PATH="${VENV_CUDA_LIB_PATH}:${LD_LIBRARY_PATH:-}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"

export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"

ALL_IPS=("$HEAD_IP")
for ip in $WORKER_IPS; do ALL_IPS+=("$ip"); done
NNODES=${#ALL_IPS[@]}
NO_PROXY_LIST="127.0.0.1,localhost,$(IFS=,; echo "${ALL_IPS[*]}")"
export no_proxy="$NO_PROXY_LIST"
export NO_PROXY="$NO_PROXY_LIST"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export RAY_enable_open_telemetry=0
export RAY_raylet_start_wait_time_s=120
RAY_HEAD_PORT_ARGS="--dashboard-agent-grpc-port=28101 --dashboard-agent-listen-port=28102 --metrics-export-port=28103"
RAY_WORKER_PORT_ARGS="--dashboard-agent-grpc-port=28001 --dashboard-agent-listen-port=28002 --metrics-export-port=28003"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the policy model directory (e.g. Qwen3-32B)}"
MODEL_CONTEXT_LIMIT=40960
DATA_DIR="${DATA_DIR:?set DATA_DIR to the BrowseComp-Plus processed dataset directory}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.paper.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/test.paper.parquet}"
RETRIEVER_MODEL_PATH="${RETRIEVER_MODEL_PATH:?set RETRIEVER_MODEL_PATH to the embedding model directory (e.g. Qwen3-Embedding-8B)}"
RETRIEVER_CORPUS_FILE="${RETRIEVER_CORPUS_FILE:-${DATA_DIR}/corpus.parquet}"
RETRIEVER_DENSE_CACHE="${RETRIEVER_DENSE_CACHE:-${DATA_DIR}/browsecomp_dense_cache.pkl}"
TOOL_CONFIG_PATH="${PROJECT_DIR}/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml"

# ---- Batch sizes ----
TRAIN_BATCH_SIZE=32
N_RESP=8
ACTOR_PPO_MICRO_BSZ=2
LOG_PROB_MICRO_BSZ=2

# ---- Parallelism ----
# On 8x8 GPUs, keep one 32-GPU model-parallel group and use DP=2.
ACTOR_TP=8
ACTOR_PP=2
ACTOR_VPP=null
ACTOR_CP=2

REF_TP=8
REF_PP=2
REF_VPP=null
REF_CP=2

ROLLOUT_TP=8

# ---- Sequence lengths ----
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=32768
MAX_TOOL_RESPONSE_LENGTH=16000
MAX_PARALLEL_CALLS=5

# ---- SUPO Configuration ----
# working_context_length: single-segment threshold to trigger summarization
# max_model_len: SGLang KV cache size
# max_summary_rounds: how many times to summarize before marking as overlong
WORKING_CONTEXT_LENGTH=32768
MAX_SUMMARY_ROUNDS=5
# Qwen3-32B local config has max_position_embeddings=40960.
# Keep this <=40960 unless the model config/rope scaling is changed.
MAX_MODEL_LEN=40960
CONTEXT_COMPRESSION_METHOD="${CONTEXT_COMPRESSION_METHOD:-summary}"
ECHO_RECENT_TURNS="${ECHO_RECENT_TURNS:-3}"
SEMANTIC_SELECTION_TOPK="${SEMANTIC_SELECTION_TOPK:-5}"

ROLLOUT_GPU_MEMORY_UTILIZATION=0.35
ROLLOUT_MAX_NUM_SEQS=32
SGLANG_CHUNKED_PREFILL_SIZE=8192
SGLANG_MAX_PREFILL_TOKENS=32768
BCP_SUMMARY_INSTRUCTION=${BCP_SUMMARY_INSTRUCTION:-$'System:\nYour operational context is full. Generate a concise summary by populating the template below.\nThis summary will be your sole context for continuing this task. Be brief but ensure all critical data is present.\n\nRules:\n- Output exactly one <summary>...</summary> block.\n- Do not call any function/tool in this turn.\n- Do not include <think>, tool calls, markdown fences, or text outside the summary tags.\n\n<summary>\nMission Objective:\n- Original query: [State the user verbatim query.]\n- Verification checklist:\n  - [VERIFIED/PENDING]: [Checklist item]\n\nKey Findings:\n- Sources:\n  - [Critical verified fact with source docid]\n- Discrepancies:\n  - [Conflicting information or uncertainty]\n\nTactical Plan:\n- Promising leads:\n  - [Best remaining keywords, sources, or angles]\n- Known dead ends:\n  - [Queries or sources that proved useless]\n- Immediate next action:\n  - [Exact tool call or query to execute next]\n</summary>'}
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/ckpt/${EXPERIMENT_NAME}}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${PROJECT_DIR}/val_outputs/${EXPERIMENT_NAME}}"
BCP_RESUME_MODE="${BCP_RESUME_MODE:-disable}"

TOKEN_BUDGET=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
BASE_ACTOR_TOKEN_BUDGET_PER_GPU=$(((TOKEN_BUDGET + ACTOR_CP - 1) / ACTOR_CP))
BASE_REF_TOKEN_BUDGET_PER_GPU=$(((TOKEN_BUDGET + REF_CP - 1) / REF_CP))
TOKEN_BUDGET_SCALE_NUM=4
TOKEN_BUDGET_SCALE_DEN=3
ACTOR_TOKEN_BUDGET_PER_GPU=$(((BASE_ACTOR_TOKEN_BUDGET_PER_GPU * TOKEN_BUDGET_SCALE_NUM + TOKEN_BUDGET_SCALE_DEN - 1) / TOKEN_BUDGET_SCALE_DEN))
REF_TOKEN_BUDGET_PER_GPU=$(((BASE_REF_TOKEN_BUDGET_PER_GPU * TOKEN_BUDGET_SCALE_NUM + TOKEN_BUDGET_SCALE_DEN - 1) / TOKEN_BUDGET_SCALE_DEN))

TOTAL_GPUS=$((NNODES * 8))
ACTOR_MODEL_PARALLEL_SIZE=$((ACTOR_TP * ACTOR_PP * ACTOR_CP))
REF_MODEL_PARALLEL_SIZE=$((REF_TP * REF_PP * REF_CP))
if [ $((TOTAL_GPUS % ACTOR_MODEL_PARALLEL_SIZE)) -ne 0 ]; then
    echo "[config] ERROR: total GPUs=${TOTAL_GPUS} must be divisible by ACTOR_TP*ACTOR_PP*ACTOR_CP=${ACTOR_MODEL_PARALLEL_SIZE}."
    exit 1
fi
if [ $((TOTAL_GPUS % REF_MODEL_PARALLEL_SIZE)) -ne 0 ]; then
    echo "[config] ERROR: total GPUs=${TOTAL_GPUS} must be divisible by REF_TP*REF_PP*REF_CP=${REF_MODEL_PARALLEL_SIZE}."
    exit 1
fi
ACTOR_DP_SIZE=$((TOTAL_GPUS / ACTOR_MODEL_PARALLEL_SIZE))
REF_DP_SIZE=$((TOTAL_GPUS / REF_MODEL_PARALLEL_SIZE))
if [ "$TOKEN_BUDGET" -ge "$MAX_MODEL_LEN" ]; then
    echo "[config] ERROR: MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH=${TOKEN_BUDGET} must be < MAX_MODEL_LEN=${MAX_MODEL_LEN}."
    exit 1
fi
if [ "$MAX_MODEL_LEN" -gt "$MODEL_CONTEXT_LIMIT" ]; then
    echo "[config] ERROR: MAX_MODEL_LEN=${MAX_MODEL_LEN} exceeds model context limit=${MODEL_CONTEXT_LIMIT}."
    exit 1
fi

echo "[config] context: prompt=${MAX_PROMPT_LENGTH}, response=${MAX_RESPONSE_LENGTH}, working=${WORKING_CONTEXT_LENGTH}, max_model=${MAX_MODEL_LEN}"
echo "[config] compression: method=${CONTEXT_COMPRESSION_METHOD}, max_rounds=${MAX_SUMMARY_ROUNDS}, effective_context=$((WORKING_CONTEXT_LENGTH * (MAX_SUMMARY_ROUNDS + 1)))"
echo "[config] parallel: actor TP/PP/CP=${ACTOR_TP}/${ACTOR_PP}/${ACTOR_CP} DP=${ACTOR_DP_SIZE}, ref TP/PP/CP=${REF_TP}/${REF_PP}/${REF_CP} DP=${REF_DP_SIZE}, rollout TP=${ROLLOUT_TP}"
echo "[config] micro bsz: actor=${ACTOR_PPO_MICRO_BSZ}, log_prob=${LOG_PROB_MICRO_BSZ}"
echo "[config] token budget per GPU: actor=${ACTOR_TOKEN_BUDGET_PER_GPU} (base=${BASE_ACTOR_TOKEN_BUDGET_PER_GPU}), ref=${REF_TOKEN_BUDGET_PER_GPU} (base=${BASE_REF_TOKEN_BUDGET_PER_GPU})"

sed -i -E "s#http://[^/]+:8000/(retrieve|get_doc)#http://${HEAD_IP}:8000/\1#g" "$TOOL_CONFIG_PATH"

# ---- Ray cluster ----
ray stop --force 2>/dev/null || true
for ip in $WORKER_IPS; do ssh "$ip" "source ${VENV_PATH}/bin/activate && ray stop --force" 2>/dev/null || true; done
sleep 8

ray start --head \
    --node-ip-address="$HEAD_IP" \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-gpus=8 \
    ${RAY_HEAD_PORT_ARGS}

for ip in $WORKER_IPS; do
    ssh "$ip" "export LD_LIBRARY_PATH='${VENV_CUDA_LIB_PATH}:\$LD_LIBRARY_PATH' && export RAY_enable_open_telemetry=0 && source ${VENV_PATH}/bin/activate && ray start --address='${HEAD_IP}:6379' --node-ip-address='${ip}' --num-gpus=8 ${RAY_WORKER_PORT_ARGS}"
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
    if [[ -n "${WATCHDOG_PID:-}" ]]; then kill "$WATCHDOG_PID" 2>/dev/null || true; fi
    if [[ -n "${RETRIEVER_PID:-}" ]]; then kill "$RETRIEVER_PID" 2>/dev/null || true; fi
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


# ---- Training with SUPO ----
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/examples/sglang_multiturn/config" \
    --config-name='bcp_multiturn_megatron_grpo' \
    algorithm.adv_estimator=supo \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    +data.tool_config_path="$TOOL_CONFIG_PATH" \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BSZ} \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BSZ} \
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
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    +actor_rollout_ref.rollout.multi_turn.inject_tool_schemas=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml" \
    +actor_rollout_ref.rollout.multi_turn.enable_summarization=True \
    +actor_rollout_ref.rollout.multi_turn.context_compression_method=${CONTEXT_COMPRESSION_METHOD} \
    +actor_rollout_ref.rollout.multi_turn.max_summary_rounds=${MAX_SUMMARY_ROUNDS} \
    +actor_rollout_ref.rollout.multi_turn.working_context_length=${WORKING_CONTEXT_LENGTH} \
    +actor_rollout_ref.rollout.multi_turn.echo_recent_turns=${ECHO_RECENT_TURNS} \
    +actor_rollout_ref.rollout.multi_turn.semantic_selection_topk=${SEMANTIC_SELECTION_TOPK} \
    +actor_rollout_ref.rollout.multi_turn.summary_instruction="'${BCP_SUMMARY_INSTRUCTION}'" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BSZ} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${REF_TOKEN_BUDGET_PER_GPU} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP} \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${REF_VPP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP} \
    actor_rollout_ref.ref.megatron.param_offload=True \
    algorithm.use_kl_in_reward=False \
    +reward.custom_reward_function.path="$PROJECT_DIR/verl/utils/reward_score/bc_p_llm_judge.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES} \
    trainer.balance_batch=False \
    trainer.project_name='echo' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode="${BCP_RESUME_MODE}" \
    trainer.logger='["console", "wandb"]' \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    +trainer.master_port_range='[31000,32000]' \
    +trainer.validation_data_dir="${VALIDATION_DATA_DIR}"
