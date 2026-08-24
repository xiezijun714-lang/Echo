#!/bin/bash
# Multi-turn Search Tool-Calling with ECHO on BrowseComp-Plus
# ECHO: end-to-end turn selection + credit-token credit assignment
# 4 nodes × 8 H100-80GB, Qwen3-32B, Megatron backend, SGLang rollout
#
# - working_context_length (L): single-segment threshold that triggers turn selection
# - max_model_len: SGLang physical context window (capped at 40960 by this model config)
#
# Run from project root: bash examples/sglang_multiturn/run_qwen3-32b_bcp_echo-ca_4node.sh

set -euo pipefail
ulimit -n 65535
ulimit -u unlimited
ulimit -c 0

# Log directory
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
source "${PROJECT_DIR}/examples/sglang_multiturn/bcp_runtime.sh"
bcp_load_env "$PROJECT_DIR"
bcp_enable_shell_trace

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-32b-bcp-echo-ca-4node-32k-s5}"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${BCP_LOG_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}.log}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/ckpt/${EXPERIMENT_NAME}}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${PROJECT_DIR}/val_outputs/${EXPERIMENT_NAME}}"
exec > >(tee "$LOG_FILE") 2>&1

# ---- Environment ----
VENV_PATH="${VENV_PATH:?set VENV_PATH to your Python virtualenv directory}"
source "${PROJECT_DIR}/examples/sglang_multiturn/bcp_node_utils.sh"
select_bcp_nodes
bcp_validate_head_node "$HEAD_IP"

bcp_configure_python "$PROJECT_DIR" "$VENV_PATH"

CUDNN_LIB="${VENV_PATH}/lib/python3.10/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"

export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"

ALL_IPS=("$HEAD_IP")
for ip in $WORKER_IPS; do ALL_IPS+=("$ip"); done
NNODES=${NNODES:-${#ALL_IPS[@]}}
if [ "$NNODES" -ne "${#ALL_IPS[@]}" ]; then
    echo "[config] ERROR: NNODES=${NNODES} but configured IP count is ${#ALL_IPS[@]}: ${ALL_IPS[*]}"
    exit 1
fi
JUDGE_HOST="${BCP_JUDGE_API_BASE#*://}"
JUDGE_HOST="${JUDGE_HOST%%/*}"
JUDGE_HOST="${JUDGE_HOST%%:*}"
NO_PROXY_LIST="127.0.0.1,localhost,$(IFS=,; echo "${ALL_IPS[*]}")${JUDGE_HOST:+,${JUDGE_HOST}}${no_proxy:+,${no_proxy}}"
export no_proxy="$NO_PROXY_LIST"
export NO_PROXY="$NO_PROXY_LIST"
bcp_prepare_service_env

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export RAY_enable_open_telemetry=0

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the policy model directory (e.g. Qwen3-32B)}"
DATA_DIR="${DATA_DIR:?set DATA_DIR to the BrowseComp-Plus processed dataset directory}"
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.paper.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/test.paper.parquet}
RETRIEVER_MODEL_PATH="${RETRIEVER_MODEL_PATH:?set RETRIEVER_MODEL_PATH to the embedding model directory (e.g. Qwen3-Embedding-8B)}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
if [[ ! "$RETRIEVER_PORT" =~ ^[1-9][0-9]*$ ]] || [ "$RETRIEVER_PORT" -gt 65535 ]; then
    echo "[config] ERROR: RETRIEVER_PORT must be an integer from 1 to 65535; got ${RETRIEVER_PORT}."
    exit 1
fi
bcp_resolve_retriever_data "$DATA_DIR"
bcp_validate_training_paths "$PROJECT_DIR" "$VENV_PATH" "$MODEL_PATH" "$TRAIN_FILE" "$VAL_FILE" "$RETRIEVER_MODEL_PATH" qwen3
bcp_validate_remote_paths "$PROJECT_DIR" "$VENV_PATH" "$MODEL_PATH" "$TRAIN_FILE" "$VAL_FILE"

# ---- Batch sizes ----
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
N_RESP=${N_RESP:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
ACTOR_PPO_MICRO_BSZ=${ACTOR_PPO_MICRO_BSZ:-1}

# ---- Parallelism ----
ACTOR_TP=${ACTOR_TP:-8}
ACTOR_PP=${ACTOR_PP:-2}
ACTOR_VPP=null
ACTOR_CP=${ACTOR_CP:-2}

REF_TP=${REF_TP:-8}
REF_PP=${REF_PP:-2}
REF_VPP=null
REF_CP=${REF_CP:-2}

ROLLOUT_TP=${ROLLOUT_TP:-8}

# ---- Sequence lengths ----
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-16000}
MAX_PARALLEL_CALLS=${MAX_PARALLEL_CALLS:-5}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-100}

# ---- ECHO Configuration ----
# working_context_length: single-segment threshold to trigger turn selection
# max_model_len: SGLang KV cache size
# max_summary_rounds: how many times to compress before marking as overlong
WORKING_CONTEXT_LENGTH=${WORKING_CONTEXT_LENGTH:-32768}
ECHO_RECENT_TURNS=${ECHO_RECENT_TURNS:-3}
ECHO_CREDIT_METHOD=${ECHO_CREDIT_METHOD:-token}
ECHO_CREDIT_PENALTY_RATIO=${ECHO_CREDIT_PENALTY_RATIO:-0.0}
ECHO_POLICY_LOSS_MODE=${ECHO_POLICY_LOSS_MODE:-vanilla}
MAX_SUMMARY_ROUNDS=${MAX_SUMMARY_ROUNDS:-5}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}

ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-8}
SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-8192}
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-32768}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console", "wandb"]'}
TRAINING_LIMIT_OVERRIDES=()
if [ -n "${TOTAL_TRAINING_STEPS:-}" ]; then
    TRAINING_LIMIT_OVERRIDES+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi
ROLLOUT_ENGINE_OVERRIDES=()
if [ -n "${SGLANG_ATTENTION_BACKEND:-}" ]; then
    ROLLOUT_ENGINE_OVERRIDES+=("+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=${SGLANG_ATTENTION_BACKEND}")
fi
if [ -n "${SGLANG_SAMPLING_BACKEND:-}" ]; then
    ROLLOUT_ENGINE_OVERRIDES+=("+actor_rollout_ref.rollout.engine_kwargs.sglang.sampling_backend=${SGLANG_SAMPLING_BACKEND}")
fi
if [ -n "${SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-}" ]; then
    ROLLOUT_ENGINE_OVERRIDES+=("+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_custom_all_reduce=${SGLANG_DISABLE_CUSTOM_ALL_REDUCE}")
fi
BCP_SUMMARY_INSTRUCTION=${BCP_SUMMARY_INSTRUCTION:-$'System:\nYour operational context is full. Generate a concise summary by populating the template below.\nThis summary will be your sole context for continuing this task. Be brief but ensure all critical data is present.\n\nRules:\n- Output exactly one <summary>...</summary> block.\n- Do not call any function/tool in this turn.\n- Do not include <think>, tool calls, markdown fences, or text outside the summary tags.\n\n<summary>\nMission Objective:\n- Original query: [State the user verbatim query.]\n- Verification checklist:\n  - [VERIFIED/PENDING]: [Checklist item]\n\nKey Findings:\n- Sources:\n  - [Critical verified fact with source docid]\n- Discrepancies:\n  - [Conflicting information or uncertainty]\n\nTactical Plan:\n- Promising leads:\n  - [Best remaining keywords, sources, or angles]\n- Known dead ends:\n  - [Queries or sources that proved useless]\n- Immediate next action:\n  - [Exact tool call or query to execute next]\n</summary>'}

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
if [ "$MAX_MODEL_LEN" -gt 40960 ]; then
    echo "[config] ERROR: MAX_MODEL_LEN=${MAX_MODEL_LEN} exceeds local Qwen3-32B max_position_embeddings=40960."
    exit 1
fi
if [[ ! "$ROLLOUT_AGENT_NUM_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[config] ERROR: ROLLOUT_AGENT_NUM_WORKERS must be a positive integer; got ${ROLLOUT_AGENT_NUM_WORKERS}."
    exit 1
fi
if [ $(((TRAIN_BATCH_SIZE * N_RESP) % ROLLOUT_AGENT_NUM_WORKERS)) -ne 0 ]; then
    echo "[config] ERROR: TRAIN_BATCH_SIZE*N_RESP must be divisible by ROLLOUT_AGENT_NUM_WORKERS."
    exit 1
fi

echo "[config] context: prompt=${MAX_PROMPT_LENGTH}, response=${MAX_RESPONSE_LENGTH}, working=${WORKING_CONTEXT_LENGTH}, max_model=${MAX_MODEL_LEN}"
echo "[config] summary: max_summary_rounds=${MAX_SUMMARY_ROUNDS}, effective_context=$((WORKING_CONTEXT_LENGTH * (MAX_SUMMARY_ROUNDS + 1)))"
echo "[config] echo credit: method=${ECHO_CREDIT_METHOD}, penalty_ratio=${ECHO_CREDIT_PENALTY_RATIO}, policy_loss=${ECHO_POLICY_LOSS_MODE}"
echo "[config] parallel: actor TP/PP/CP=${ACTOR_TP}/${ACTOR_PP}/${ACTOR_CP}, ref TP/PP/CP=${REF_TP}/${REF_PP}/${REF_CP}, rollout TP=${ROLLOUT_TP}"
echo "[config] rollout: responses=${N_RESP}, agent_workers=${ROLLOUT_AGENT_NUM_WORKERS}, max_seqs=${ROLLOUT_MAX_NUM_SEQS}"
echo "[config] token budget per GPU: actor=${ACTOR_TOKEN_BUDGET_PER_GPU}, ref=${REF_TOKEN_BUDGET_PER_GPU}"

TOOL_CONFIG_TEMPLATE_PATH="${TOOL_CONFIG_TEMPLATE_PATH:-${PROJECT_DIR}/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml}"
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-${CKPT_DIR}/runtime/search_tool_config.yaml}"
bcp_prepare_tool_config "$TOOL_CONFIG_TEMPLATE_PATH" "$TOOL_CONFIG_PATH" "$HEAD_IP" "$RETRIEVER_PORT"

if bcp_preflight_complete verl.trainer.main_ppo "$PROJECT_DIR/examples/sglang_multiturn/config" bcp_multiturn_megatron_grpo; then
    exit 0
fi

RETRIEVER_PID=""
WATCHDOG_PID=""
RETRIEVER_PID_FILE="${CKPT_DIR}/runtime/retriever.pid"
: > "$RETRIEVER_PID_FILE"

cleanup() {
    local active_retriever_pid=""
    if [[ -n "${WATCHDOG_PID:-}" ]]; then
        kill "$WATCHDOG_PID" 2>/dev/null || true
        wait "$WATCHDOG_PID" 2>/dev/null || true
    fi
    active_retriever_pid="$(cat "$RETRIEVER_PID_FILE" 2>/dev/null || true)"
    if [[ "$active_retriever_pid" =~ ^[0-9]+$ ]]; then
        kill "$active_retriever_pid" 2>/dev/null || true
        wait "$active_retriever_pid" 2>/dev/null || true
    fi
    : > "$RETRIEVER_PID_FILE" 2>/dev/null || true
    "$RAY_BIN" stop --force 2>/dev/null || true
    for ip in $WORKER_IPS; do
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$ip" "source '${VENV_PATH}/bin/activate' && export VIRTUAL_ENV='${VENV_PATH}' && export PATH='${VENV_PATH}/bin':\$PATH && ${RAY_BIN} stop --force" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---- Ray cluster ----
"$RAY_BIN" stop --force 2>/dev/null || true
for ip in $WORKER_IPS; do
    ssh "$ip" "source '${VENV_PATH}/bin/activate' && export VIRTUAL_ENV='${VENV_PATH}' && export PATH='${VENV_PATH}/bin':\$PATH && ${RAY_BIN} stop --force" 2>/dev/null || true
done
sleep 2

RAY_CPU_ARGS=()
if [ -n "${RAY_NUM_CPUS:-}" ]; then
    if [[ ! "$RAY_NUM_CPUS" =~ ^[1-9][0-9]*$ ]]; then
        echo "[config] ERROR: RAY_NUM_CPUS must be a positive integer; got ${RAY_NUM_CPUS}."
        exit 1
    fi
    RAY_CPU_ARGS+=(--num-cpus="$RAY_NUM_CPUS")
fi

"$RAY_BIN" start --head \
    --disable-usage-stats \
    --node-ip-address="$HEAD_IP" \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-gpus=8 \
    "${RAY_CPU_ARGS[@]}"

for ip in $WORKER_IPS; do
    ssh "$ip" "export LD_LIBRARY_PATH='${VENV_PATH}/lib/python3.10/site-packages/nvidia/cudnn/lib':\$LD_LIBRARY_PATH && export PYTHONPATH='${PYTHONPATH}' && export RAY_enable_open_telemetry=0 && source '${VENV_PATH}/bin/activate' && export VIRTUAL_ENV='${VENV_PATH}' && export PATH='${VENV_PATH}/bin':\$PATH && ${RAY_BIN} start --address='${HEAD_IP}:6379' --node-ip-address='${ip}' --num-gpus=8"
done

for i in $(seq 1 60); do
    NODE_COUNT=$("$PYTHON_BIN" -c \
        "import ray; ray.init(address='${HEAD_IP}:6379', ignore_reinit_error=True); print(len(ray.nodes()))" \
        2>/dev/null)
    if [ "${NODE_COUNT:-0}" -ge "$NNODES" ] 2>/dev/null; then
        echo "[ray] Cluster ready: $NODE_COUNT nodes."; break
    fi
    [ "$i" -eq 60 ] && { echo "[ray] ERROR: timeout."; "$RAY_BIN" status --address="${HEAD_IP}:6379"; exit 1; }
    sleep 2
done

# ---- Retrieval service ----
RETRIEVER_LOG="${LOG_DIR}/browsecomp_retriever.log"
RETRIEVER_CMD=("${PYTHON_BIN}"
    "$PROJECT_DIR/examples/sglang_multiturn/browsecomp_retrieval_server.py"
    --mode dense
    --model "${RETRIEVER_MODEL_PATH}"
    --device cpu
    --corpus_file "${RETRIEVER_CORPUS_FILE}"
    --host 0.0.0.0 --port "$RETRIEVER_PORT"
    --batch_size 4
    --dense_cache "${RETRIEVER_DENSE_CACHE}")

bcp_stop_port_listeners "$RETRIEVER_PORT"
sleep 1
bcp_require_bindable_port "$PYTHON_BIN" "$RETRIEVER_PORT"

setsid "${RETRIEVER_CMD[@]}" > "$RETRIEVER_LOG" 2>&1 &
RETRIEVER_PID=$!
printf '%s\n' "$RETRIEVER_PID" > "$RETRIEVER_PID_FILE"
bcp_wait_for_retriever "$RETRIEVER_PID" "$RETRIEVER_LOG" "$RETRIEVER_PORT"

# ---- Retrieval server watchdog (restarts server if health check fails) ----
(
    while true; do
        sleep 30
        if ! curl -sf "http://127.0.0.1:${RETRIEVER_PORT}/health" > /dev/null 2>&1; then
            bcp_stop_port_listeners "$RETRIEVER_PORT"
            sleep 2
            setsid "${RETRIEVER_CMD[@]}" >> "$RETRIEVER_LOG" 2>&1 &
            RETRIEVER_PID=$!
            printf '%s\n' "$RETRIEVER_PID" > "$RETRIEVER_PID_FILE"
            for j in $(seq 1 120); do
                curl -sf "http://127.0.0.1:${RETRIEVER_PORT}/health" > /dev/null 2>&1 && break
                kill -0 $RETRIEVER_PID 2>/dev/null || break
                sleep 1
            done
        fi
    done
) &
WATCHDOG_PID=$!


# ---- Training with ECHO ----
cd "$PROJECT_DIR"
"$PYTHON_BIN" -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/examples/sglang_multiturn/config" \
    --config-name='bcp_multiturn_megatron_grpo' \
    algorithm.adv_estimator=supo \
    +algorithm.echo_credit_method=${ECHO_CREDIT_METHOD} \
    +algorithm.echo_credit_penalty_ratio=${ECHO_CREDIT_PENALTY_RATIO} \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BSZ} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_TOKEN_BUDGET_PER_GPU} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.20 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.policy_loss.loss_mode=${ECHO_POLICY_LOSS_MODE} \
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
    actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_AGENT_NUM_WORKERS} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=${MAX_MODEL_LEN} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=${SGLANG_CHUNKED_PREFILL_SIZE} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens=${SGLANG_MAX_PREFILL_TOKENS} \
    "${ROLLOUT_ENGINE_OVERRIDES[@]}" \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=${MAX_PARALLEL_CALLS} \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    +actor_rollout_ref.rollout.multi_turn.context_compression_method=echo_e2e \
    +actor_rollout_ref.rollout.multi_turn.enable_summarization=True \
    +actor_rollout_ref.rollout.multi_turn.max_summary_rounds=${MAX_SUMMARY_ROUNDS} \
    +actor_rollout_ref.rollout.multi_turn.working_context_length=${WORKING_CONTEXT_LENGTH} \
    +actor_rollout_ref.rollout.multi_turn.summary_instruction="'${BCP_SUMMARY_INSTRUCTION}'" \
    +actor_rollout_ref.rollout.multi_turn.echo_recent_turns=${ECHO_RECENT_TURNS} \
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
    +ray_kwargs.ray_init.address="${HEAD_IP}:6379" \
    "${BCP_RAY_ENV_OVERRIDES[@]}" \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES} \
    trainer.project_name='echo' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.logger="${TRAINER_LOGGER}" \
    trainer.save_freq=${SAVE_FREQ:-100} \
    trainer.test_freq=${TEST_FREQ:-2} \
    trainer.total_epochs=${TOTAL_EPOCHS:-5} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} \
    +trainer.master_port_range='[31000,32000]' \
    +trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
    "${TRAINING_LIMIT_OVERRIDES[@]}" \
    "$@"

echo "[train] Training command completed successfully."
