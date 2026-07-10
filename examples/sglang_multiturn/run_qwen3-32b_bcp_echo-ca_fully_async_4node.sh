#!/bin/bash
# Fully-async multi-turn Search Tool-Calling ECHO-CA on BrowseComp-Plus
# ECHO-CA: ECHO context compression with credit assignment on top of SUPO.
# 4 nodes x 8 H100-80GB, Qwen3-32B, Megatron backend, SGLang rollout.
# Default split: 2 train nodes + 2 rollout nodes.
#
# ECHO uses turn selection/context compression to continue beyond the working context length.
# Physical context is capped by the local Qwen3-32B config at 40960.
#
# Use NODE_SLICE=0 for machines 0/1/2/3, or NODE_SLICE=1 for 4/5/6/7.
#
# Run from project root: bash examples/sglang_multiturn/run_qwen3-32b_bcp_echo-ca_fully_async_4node.sh

set -x
set -euo pipefail
ulimit -n 65535
ulimit -u unlimited

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
EXPERIMENT_NAME="qwen3-32b-bcp-echo-ca-fully-async-4node-32k-s5"
if [ "${1:-}" = "--bcp-experiment-name" ]; then
    if [ -z "${2:-}" ]; then
        echo "[config] ERROR: --bcp-experiment-name requires a value."
        exit 1
    fi
    EXPERIMENT_NAME="$2"
    shift 2
fi

# Log directory
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${BCP_LOG_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}.log}"
exec > >(tee "$LOG_FILE") 2>&1

# ---- Environment ----
VENV_PATH="${VENV_PATH:?set VENV_PATH to your Python virtualenv directory}"

source "${PROJECT_DIR}/examples/sglang_multiturn/bcp_node_utils.sh"
select_bcp_nodes
LOCAL_IPS="${POD_IP:-} $(hostname -I 2>/dev/null || true)"
if [[ -n "${LOCAL_IPS// /}" && " ${LOCAL_IPS} " != *" ${HEAD_IP} "* ]]; then
    echo "[config] ERROR: current host IPs (${LOCAL_IPS}) do not include HEAD_IP=${HEAD_IP}; run this script on the slice head node."
    exit 1
fi

source "${VENV_PATH}/bin/activate"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:$PATH"
PYTHON_BIN="${VENV_PATH}/bin/python3"
RAY_BIN="${VENV_PATH}/bin/ray"

NVIDIA_SITE_PACKAGES="${VENV_PATH}/lib/python3.10/site-packages/nvidia"
VENV_CUDA_LIB_PATH="${NVIDIA_SITE_PACKAGES}/nvjitlink/lib:${NVIDIA_SITE_PACKAGES}/cusparse/lib:${NVIDIA_SITE_PACKAGES}/cublas/lib:${NVIDIA_SITE_PACKAGES}/cudnn/lib:${NVIDIA_SITE_PACKAGES}/cuda_runtime/lib"
export LD_LIBRARY_PATH="${VENV_CUDA_LIB_PATH}:${LD_LIBRARY_PATH:-}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"

export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"

ALL_IPS=("$HEAD_IP")
for ip in $WORKER_IPS; do ALL_IPS+=("$ip"); done
NNODES=${NNODES:-${#ALL_IPS[@]}}
if [ "$NNODES" -ne "${#ALL_IPS[@]}" ]; then
    echo "[config] ERROR: NNODES=${NNODES} but configured IP count is ${#ALL_IPS[@]}: ${ALL_IPS[*]}"
    exit 1
fi
NO_PROXY_LIST="127.0.0.1,localhost,$(IFS=,; echo "${ALL_IPS[*]}")"
export no_proxy="$NO_PROXY_LIST"
export NO_PROXY="$NO_PROXY_LIST"

# Required for Megatron
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export RAY_enable_open_telemetry=0
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-120}"
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}"
export RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL="${RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL:-minor}"
export VERL_DATAPROTO_SERIALIZATION_METHOD="${VERL_DATAPROTO_SERIALIZATION_METHOD:-numpy}"
export VERL_MASTER_PORT_RANGE="${VERL_MASTER_PORT_RANGE:-42000,43000}"
RAY_HEAD_PORT_ARGS="${RAY_HEAD_PORT_ARGS:---dashboard-agent-grpc-port=28101 --dashboard-agent-listen-port=28102 --metrics-export-port=28103}"
RAY_WORKER_PORT_ARGS="${RAY_WORKER_PORT_ARGS:---dashboard-agent-grpc-port=28001 --dashboard-agent-listen-port=28002 --metrics-export-port=28003}"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the policy model directory (e.g. Qwen3-32B)}"
MODEL_CONTEXT_LIMIT=${MODEL_CONTEXT_LIMIT:-40960}
USE_DIST_CKPT="${USE_DIST_CKPT:-False}"
DATA_DIR="${DATA_DIR:?set DATA_DIR to the BrowseComp-Plus processed dataset directory}"
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.paper.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/test.paper.parquet}
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/ckpt/${EXPERIMENT_NAME}}"
BCP_RESUME_MODE="${BCP_RESUME_MODE:-disable}"
RETRIEVER_MODEL_PATH="${RETRIEVER_MODEL_PATH:?set RETRIEVER_MODEL_PATH to the embedding model directory (e.g. Qwen3-Embedding-8B)}"
RETRIEVER_CORPUS_FILE="${RETRIEVER_CORPUS_FILE:-${DATA_DIR}/corpus.parquet}"
RETRIEVER_DENSE_CACHE="${RETRIEVER_DENSE_CACHE:-${DATA_DIR}/browsecomp_dense_cache.pkl}"
for required_path in "$MODEL_PATH" "$TRAIN_FILE" "$VAL_FILE" "$RETRIEVER_MODEL_PATH"; do
    if [ ! -e "$required_path" ]; then
        echo "[config] ERROR: required path does not exist on head node: ${required_path}"
        exit 1
    fi
done
if [ ! -e "$RETRIEVER_DENSE_CACHE" ] && [ ! -e "$RETRIEVER_CORPUS_FILE" ]; then
    echo "[config] ERROR: retriever needs either dense cache or corpus file on head node."
    echo "[config]        missing dense_cache=${RETRIEVER_DENSE_CACHE}"
    echo "[config]        missing corpus_file=${RETRIEVER_CORPUS_FILE}"
    exit 1
fi
if [ ! -e "$RETRIEVER_DENSE_CACHE" ]; then
    echo "[config] WARNING: dense cache missing; retriever will rebuild from ${RETRIEVER_CORPUS_FILE}."
fi

# ---- Batch sizes ----
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
N_RESP=${N_RESP:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
ACTOR_PPO_MICRO_BSZ=${ACTOR_PPO_MICRO_BSZ:-1}
LOG_PROB_MICRO_BSZ=${LOG_PROB_MICRO_BSZ:-1}

# ---- Parallelism ----
ACTOR_TP=${ACTOR_TP:-8}
ACTOR_PP=${ACTOR_PP:-1}
ACTOR_VPP=null
ACTOR_CP=${ACTOR_CP:-2}

REF_TP=${REF_TP:-8}
REF_PP=${REF_PP:-1}
REF_VPP=null
REF_CP=${REF_CP:-2}

# Inference: TP=8, one replica per rollout node by default.
ROLLOUT_TP=${ROLLOUT_TP:-8}

# ---- Fully async resource split ----
TRAINER_NNODES=${TRAINER_NNODES:-2}
TRAINER_GPUS_PER_NODE=${TRAINER_GPUS_PER_NODE:-8}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-$((NNODES - TRAINER_NNODES))}
ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-8}

detect_resume_step() {
    local latest_file="${CKPT_DIR}/latest_checkpointed_iteration.txt"
    if [ -f "$latest_file" ]; then
        tr -dc '0-9' < "$latest_file"
        return 0
    fi
    local latest_step=""
    latest_step=$(find "$CKPT_DIR" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sed -E 's#.*/global_step_([0-9]+)#\1#' | sort -n | tail -n 1)
    if [ -n "$latest_step" ]; then
        echo "$latest_step"
        return 0
    fi
    return 1
}

count_actor_shards_on_ip() {
    local ip="$1"
    local step="$2"
    local actor_dist_ckpt="${CKPT_DIR}/global_step_${step}/actor/dist_ckpt"
    if [ "$ip" = "$HEAD_IP" ]; then
        if [ -d "$actor_dist_ckpt" ]; then
            find "$actor_dist_ckpt" -maxdepth 1 -name '*.distcp' | wc -l
        else
            echo 0
        fi
    else
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$ip" \
            "if [ -d '$actor_dist_ckpt' ]; then find '$actor_dist_ckpt' -maxdepth 1 -name '*.distcp' | wc -l; else echo 0; fi" \
            2>/dev/null || echo 0
    fi
}

if [[ "${BCP_RESUME_MODE}" != "disable" && -z "${TRAINER_POOL_IPS:-}" && -z "${ROLLOUT_POOL_IPS:-}" ]]; then
    RESUME_STEP="$(detect_resume_step || true)"
    if [ -n "$RESUME_STEP" ]; then
        DETECTED_TRAINER_POOL_IPS=()
        for ip in "${TRAINER_IP_ARRAY[@]}"; do
            SHARD_COUNT="$(count_actor_shards_on_ip "$ip" "$RESUME_STEP")"
            echo "[resume] ckpt global_step_${RESUME_STEP}: actor shard count on ${ip} = ${SHARD_COUNT}"
            if [ "${SHARD_COUNT:-0}" -gt 0 ] 2>/dev/null; then
                DETECTED_TRAINER_POOL_IPS+=("$ip")
            fi
        done
        if [ "${#DETECTED_TRAINER_POOL_IPS[@]}" -eq "$TRAINER_NNODES" ]; then
            TRAINER_POOL_IPS="$(IFS=,; echo "${DETECTED_TRAINER_POOL_IPS[*]}")"
            DETECTED_ROLLOUT_POOL_IPS=()
            for ip in "${TRAINER_IP_ARRAY[@]}"; do
                in_trainer_pool=0
                for trainer_ip in "${DETECTED_TRAINER_POOL_IPS[@]}"; do
                    if [ "$ip" = "$trainer_ip" ]; then
                        in_trainer_pool=1
                        break
                    fi
                done
                if [ "$in_trainer_pool" -eq 0 ]; then
                    DETECTED_ROLLOUT_POOL_IPS+=("$ip")
                fi
            done
            ROLLOUT_POOL_IPS="$(IFS=,; echo "${DETECTED_ROLLOUT_POOL_IPS[*]}")"
            echo "[resume] auto placement from ckpt: trainer_pool=${TRAINER_POOL_IPS}, rollout_pool=${ROLLOUT_POOL_IPS}"
        else
            echo "[resume] WARNING: expected ${TRAINER_NNODES} actor ckpt nodes, detected ${#DETECTED_TRAINER_POOL_IPS[@]}; use default pool split."
        fi
    else
        echo "[resume] WARNING: resume enabled but no local checkpoint step detected in ${CKPT_DIR}; use default pool split."
    fi
fi

DEFAULT_TRAINER_POOL_IP_ARRAY=("${TRAINER_IP_ARRAY[@]:0:TRAINER_NNODES}")
DEFAULT_ROLLOUT_POOL_IP_ARRAY=("${TRAINER_IP_ARRAY[@]:TRAINER_NNODES:ROLLOUT_NNODES}")
TRAINER_POOL_IPS="${TRAINER_POOL_IPS:-$(IFS=,; echo "${DEFAULT_TRAINER_POOL_IP_ARRAY[*]}")}"
ROLLOUT_POOL_IPS="${ROLLOUT_POOL_IPS:-$(IFS=,; echo "${DEFAULT_ROLLOUT_POOL_IP_ARRAY[*]}")}"
IFS=',' read -r -a TRAINER_POOL_IP_ARRAY <<< "$TRAINER_POOL_IPS"
IFS=',' read -r -a ROLLOUT_POOL_IP_ARRAY <<< "$ROLLOUT_POOL_IPS"
if [ "${#TRAINER_POOL_IP_ARRAY[@]}" -ne "$TRAINER_NNODES" ]; then
    echo "[config] ERROR: TRAINER_POOL_IPS must have ${TRAINER_NNODES} IPs, got ${#TRAINER_POOL_IP_ARRAY[@]}: ${TRAINER_POOL_IPS}"
    exit 1
fi
if [ "${#ROLLOUT_POOL_IP_ARRAY[@]}" -ne "$ROLLOUT_NNODES" ]; then
    echo "[config] ERROR: ROLLOUT_POOL_IPS must have ${ROLLOUT_NNODES} IPs, got ${#ROLLOUT_POOL_IP_ARRAY[@]}: ${ROLLOUT_POOL_IPS}"
    exit 1
fi
for ip in "${TRAINER_POOL_IP_ARRAY[@]}" "${ROLLOUT_POOL_IP_ARRAY[@]}"; do
    if [[ -z "${SEEN_IPS[$ip]:-}" ]]; then
        echo "[config] ERROR: pool IP ${ip} is not in TRAINER_IPS=${TRAINER_IPS}"
        exit 1
    fi
done
export VERL_TRAINER_POOL_IPS="$TRAINER_POOL_IPS"
export VERL_ROLLOUT_POOL_IPS="$ROLLOUT_POOL_IPS"

# ---- Sequence lengths ----
# Keep rollout context defaults aligned with the synchronous BCP scripts; ACTOR_CP only affects trainer token budget.
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-16000}
MAX_PARALLEL_CALLS=${MAX_PARALLEL_CALLS:-5}

# ---- ECHO configuration ----
WORKING_CONTEXT_LENGTH=${WORKING_CONTEXT_LENGTH:-32768}
ECHO_RECENT_TURNS=${ECHO_RECENT_TURNS:-3}
ECHO_CREDIT_METHOD=${ECHO_CREDIT_METHOD:-token}
ECHO_CREDIT_PENALTY_RATIO=${ECHO_CREDIT_PENALTY_RATIO:-0.0}
ECHO_POLICY_LOSS_MODE=${ECHO_POLICY_LOSS_MODE:-vanilla}
MAX_SUMMARY_ROUNDS=${MAX_SUMMARY_ROUNDS:-5}
CONTEXT_COMPRESSION_METHOD=${CONTEXT_COMPRESSION_METHOD:-echo_e2e}
SEMANTIC_SELECTION_FULL_OBSERVATION=${SEMANTIC_SELECTION_FULL_OBSERVATION:-False}
SEMANTIC_SELECTION_FULL_OBSERVATION_MAX_CHARS=${SEMANTIC_SELECTION_FULL_OBSERVATION_MAX_CHARS:-4000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
BCP_SUMMARY_INSTRUCTION=${BCP_SUMMARY_INSTRUCTION:-$'System:\nYour operational context is full. Generate a concise summary by populating the template below.\nThis summary will be your sole context for continuing this task. Be brief but ensure all critical data is present.\n\nRules:\n- Output exactly one <summary>...</summary> block.\n- Do not call any function/tool in this turn.\n- Do not include <think>, tool calls, markdown fences, or text outside the summary tags.\n\n<summary>\nMission Objective:\n- Original query: [State the user verbatim query.]\n- Verification checklist:\n  - [VERIFIED/PENDING]: [Checklist item]\n\nKey Findings:\n- Sources:\n  - [Critical verified fact with source docid]\n- Discrepancies:\n  - [Conflicting information or uncertainty]\n\nTactical Plan:\n- Promising leads:\n  - [Best remaining keywords, sources, or angles]\n- Known dead ends:\n  - [Queries or sources that proved useless]\n- Immediate next action:\n  - [Exact tool call or query to execute next]\n</summary>'}

VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${PROJECT_DIR}/val_outputs/${EXPERIMENT_NAME}}"
mkdir -p "$CKPT_DIR" "$VALIDATION_DATA_DIR"
echo "[output] log=${LOG_FILE}"
echo "[output] ckpt=${CKPT_DIR}"
echo "[output] validation=${VALIDATION_DATA_DIR}"

ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-8192}
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-32768}
TOKEN_BUDGET=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ACTOR_TOKEN_BUDGET_PER_GPU=$(((TOKEN_BUDGET + ACTOR_CP - 1) / ACTOR_CP))
REF_TOKEN_BUDGET_PER_GPU=$(((TOKEN_BUDGET + REF_CP - 1) / REF_CP))

TOTAL_GPUS=$((NNODES * 8))
TRAINER_TOTAL_GPUS=$((TRAINER_NNODES * TRAINER_GPUS_PER_NODE))
ROLLOUT_TOTAL_GPUS=$((ROLLOUT_NNODES * ROLLOUT_GPUS_PER_NODE))
ACTOR_MODEL_PARALLEL_SIZE=$((ACTOR_TP * ACTOR_PP * ACTOR_CP))
REF_MODEL_PARALLEL_SIZE=$((REF_TP * REF_PP * REF_CP))
if [ "$ROLLOUT_NNODES" -lt 1 ]; then
    echo "[config] ERROR: ROLLOUT_NNODES=${ROLLOUT_NNODES}; need at least one rollout node."
    exit 1
fi
if [ $((TRAINER_TOTAL_GPUS + ROLLOUT_TOTAL_GPUS)) -gt "$TOTAL_GPUS" ]; then
    echo "[config] ERROR: trainer GPUs (${TRAINER_TOTAL_GPUS}) + rollout GPUs (${ROLLOUT_TOTAL_GPUS}) exceed cluster GPUs (${TOTAL_GPUS})."
    exit 1
fi
if [ "$ACTOR_MODEL_PARALLEL_SIZE" -gt "$TRAINER_TOTAL_GPUS" ] || [ $((TRAINER_TOTAL_GPUS % ACTOR_MODEL_PARALLEL_SIZE)) -ne 0 ]; then
    echo "[config] ERROR: actor MP=${ACTOR_MODEL_PARALLEL_SIZE} must divide trainer GPUs=${TRAINER_TOTAL_GPUS}."
    exit 1
fi
if [ "$REF_MODEL_PARALLEL_SIZE" -gt "$TRAINER_TOTAL_GPUS" ] || [ $((TRAINER_TOTAL_GPUS % REF_MODEL_PARALLEL_SIZE)) -ne 0 ]; then
    echo "[config] ERROR: ref MP=${REF_MODEL_PARALLEL_SIZE} must divide trainer GPUs=${TRAINER_TOTAL_GPUS}."
    exit 1
fi
if [ "$ROLLOUT_TOTAL_GPUS" -le 0 ] || [ $((ROLLOUT_TOTAL_GPUS % ROLLOUT_TP)) -ne 0 ]; then
    echo "[config] ERROR: rollout TP=${ROLLOUT_TP} must divide rollout GPUs=${ROLLOUT_TOTAL_GPUS}."
    exit 1
fi
if [ "$TOKEN_BUDGET" -ge "$MAX_MODEL_LEN" ]; then
    echo "[config] ERROR: MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH=${TOKEN_BUDGET} must be < MAX_MODEL_LEN=${MAX_MODEL_LEN}."
    exit 1
fi
if [ "$TOKEN_BUDGET" -le "$WORKING_CONTEXT_LENGTH" ]; then
    echo "[config] WARNING: MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH=${TOKEN_BUDGET} <= WORKING_CONTEXT_LENGTH=${WORKING_CONTEXT_LENGTH}; rollout may terminate before ECHO split triggers."
fi
if [ "$MAX_MODEL_LEN" -gt "$MODEL_CONTEXT_LIMIT" ]; then
    echo "[config] ERROR: MAX_MODEL_LEN=${MAX_MODEL_LEN} exceeds model context limit=${MODEL_CONTEXT_LIMIT}."
    exit 1
fi
ACTOR_DP_SIZE=$((TRAINER_TOTAL_GPUS / ACTOR_MODEL_PARALLEL_SIZE))
REF_DP_SIZE=$((TRAINER_TOTAL_GPUS / REF_MODEL_PARALLEL_SIZE))
echo "[config] context: prompt=${MAX_PROMPT_LENGTH}, response=${MAX_RESPONSE_LENGTH}, working=${WORKING_CONTEXT_LENGTH}, max_model=${MAX_MODEL_LEN}"
echo "[config] compression: method=${CONTEXT_COMPRESSION_METHOD}, max_rounds=${MAX_SUMMARY_ROUNDS}, effective_context=$((WORKING_CONTEXT_LENGTH * (MAX_SUMMARY_ROUNDS + 1)))"
echo "[config] semantic selection: full_observation=${SEMANTIC_SELECTION_FULL_OBSERVATION}, full_observation_max_chars=${SEMANTIC_SELECTION_FULL_OBSERVATION_MAX_CHARS}"
echo "[config] echo credit: method=${ECHO_CREDIT_METHOD}, penalty_ratio=${ECHO_CREDIT_PENALTY_RATIO}, policy_loss=${ECHO_POLICY_LOSS_MODE}"
echo "[config] checkpoint: use_dist_checkpointing=${USE_DIST_CKPT}"
echo "[config] entry=echo-ca_fully_async, nodes=${NNODES}, trainer GPUs=${TRAINER_TOTAL_GPUS}, rollout GPUs=${ROLLOUT_TOTAL_GPUS}"
echo "[config] resource pools: trainer_pool=${TRAINER_POOL_IPS}, rollout_pool=${ROLLOUT_POOL_IPS}"
echo "[config] parallel: actor TP/PP/CP=${ACTOR_TP}/${ACTOR_PP}/${ACTOR_CP} DP=${ACTOR_DP_SIZE}, ref TP/PP/CP=${REF_TP}/${REF_PP}/${REF_CP} DP=${REF_DP_SIZE}, rollout TP=${ROLLOUT_TP}"
echo "[config] token budget per GPU: actor=${ACTOR_TOKEN_BUDGET_PER_GPU}, ref=${REF_TOKEN_BUDGET_PER_GPU}"

# Use a per-experiment tool config so concurrent 4-node slices do not overwrite
# the shared template's retrieval host.
TOOL_CONFIG_TEMPLATE_PATH="${TOOL_CONFIG_TEMPLATE_PATH:-${PROJECT_DIR}/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml}"
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-${PROJECT_DIR}/examples/sglang_multiturn/config/tool_config/search_tool_config.${EXPERIMENT_NAME}.yaml}"
cp "$TOOL_CONFIG_TEMPLATE_PATH" "$TOOL_CONFIG_PATH"
sed -i -E "s#http://[^/]+:8000/(retrieve|get_doc)#http://${HEAD_IP}:8000/\1#g" "$TOOL_CONFIG_PATH"

DATA_TOOL_CONFIG_OVERRIDES=(+data.tool_config_path="$TOOL_CONFIG_PATH")
if [ "${INJECT_DATA_TOOL_SCHEMAS:-True}" = "False" ]; then
    DATA_TOOL_CONFIG_OVERRIDES=(data.tool_config_path=null)
fi

# ---- Ray cluster ----
"$RAY_BIN" stop --force 2>/dev/null || ray stop --force 2>/dev/null || true
for ip in $WORKER_IPS; do
    ssh "$ip" "source ${VENV_PATH}/bin/activate && ${RAY_BIN} stop --force 2>/dev/null || ray stop --force 2>/dev/null || true" 2>/dev/null || true
done
sleep 2

echo "[ray] Starting Ray head on $HEAD_IP ($NNODES nodes total) ..."
"$RAY_BIN" start --head \
    --node-ip-address="$HEAD_IP" \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-gpus=8 \
    ${RAY_HEAD_PORT_ARGS}

for ip in $WORKER_IPS; do
    echo "[ray] Starting worker on $ip ..."
    ssh "$ip" "unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES && export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' && export LD_LIBRARY_PATH='${VENV_CUDA_LIB_PATH}:\$LD_LIBRARY_PATH' && export RAY_enable_open_telemetry=0 && export RAY_raylet_start_wait_time_s='${RAY_raylet_start_wait_time_s}' && export RAY_ENABLE_UV_RUN_RUNTIME_ENV='${RAY_ENABLE_UV_RUN_RUNTIME_ENV}' && export RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL='${RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL}' && export VERL_DATAPROTO_SERIALIZATION_METHOD='${VERL_DATAPROTO_SERIALIZATION_METHOD}' && export VERL_MASTER_PORT_RANGE='${VERL_MASTER_PORT_RANGE}' && source ${VENV_PATH}/bin/activate && ${PYTHON_BIN} -V && ${RAY_BIN} start --address='${HEAD_IP}:6379' --node-ip-address='${ip}' --num-gpus=8 ${RAY_WORKER_PORT_ARGS}"
done

echo "[ray] Waiting for ${NNODES} nodes ..."
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
RETRIEVER_LOG="${LOG_DIR}/browsecomp_retriever.${EXPERIMENT_NAME}.log"
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
    "$RAY_BIN" stop --force 2>/dev/null || true
    for ip in $WORKER_IPS; do ssh "$ip" "source ${VENV_PATH}/bin/activate && ${RAY_BIN} stop --force" 2>/dev/null || true; done
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

# ---- Retrieval server watchdog ----
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
FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP=${FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP:-$((TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE))}
if [ "$FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP" -lt 1 ]; then
    FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP=1
fi
FULL_ASYNC_STALENESS_THRESHOLD=${FULL_ASYNC_STALENESS_THRESHOLD:-0.5}
FULL_ASYNC_REQUIRE_BATCHES=${FULL_ASYNC_REQUIRE_BATCHES:-1}
FULL_ASYNC_PARTIAL_ROLLOUT=${FULL_ASYNC_PARTIAL_ROLLOUT:-True}
FULL_ASYNC_TOTAL_ROLLOUT_STEPS=${FULL_ASYNC_TOTAL_ROLLOUT_STEPS:-1000000000}
FULL_ASYNC_TOTAL_TRAINING_STEPS=${FULL_ASYNC_TOTAL_TRAINING_STEPS:-$((FULL_ASYNC_TOTAL_ROLLOUT_STEPS / (PPO_MINI_BATCH_SIZE * FULL_ASYNC_REQUIRE_BATCHES * FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP)))}
if [ "$FULL_ASYNC_TOTAL_TRAINING_STEPS" -lt 1 ]; then
    FULL_ASYNC_TOTAL_TRAINING_STEPS=1
fi

echo "[train] Starting fully async ECHO-CA trainer: trigger_sync=${FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP}, staleness=${FULL_ASYNC_STALENESS_THRESHOLD}, partial=${FULL_ASYNC_PARTIAL_ROLLOUT}, optim_steps=${FULL_ASYNC_TOTAL_TRAINING_STEPS}"
"$PYTHON_BIN" -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path="$PROJECT_DIR/examples/sglang_multiturn/config" \
    --config-name='bcp_multiturn_megatron_grpo' \
    algorithm.adv_estimator=supo \
    +algorithm.echo_credit_method=${ECHO_CREDIT_METHOD} \
    +algorithm.echo_credit_penalty_ratio=${ECHO_CREDIT_PENALTY_RATIO} \
    data.train_batch_size=0 \
    +data.gen_batch_size=1 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    "${DATA_TOOL_CONFIG_OVERRIDES[@]}" \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.optim.total_training_steps=${FULL_ASYNC_TOTAL_TRAINING_STEPS} \
    actor_rollout_ref.actor.optim.lr_decay_steps=${FULL_ASYNC_TOTAL_TRAINING_STEPS} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BSZ} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_TOKEN_BUDGET_PER_GPU} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    +actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.clip_ratio=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.20 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.policy_loss.loss_mode=${ECHO_POLICY_LOSS_MODE} \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
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
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
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
    actor_rollout_ref.rollout.multi_turn.format=${MULTI_TURN_FORMAT:-hermes} \
    +actor_rollout_ref.rollout.multi_turn.inject_tool_schemas=${INJECT_ROLLOUT_TOOL_SCHEMAS:-True} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    +actor_rollout_ref.rollout.multi_turn.enable_summarization=True \
    +actor_rollout_ref.rollout.multi_turn.context_compression_method=${CONTEXT_COMPRESSION_METHOD} \
    +actor_rollout_ref.rollout.multi_turn.max_summary_rounds=${MAX_SUMMARY_ROUNDS} \
    +actor_rollout_ref.rollout.multi_turn.working_context_length=${WORKING_CONTEXT_LENGTH} \
    +actor_rollout_ref.rollout.multi_turn.summary_instruction="'${BCP_SUMMARY_INSTRUCTION}'" \
    +actor_rollout_ref.rollout.multi_turn.echo_recent_turns=${ECHO_RECENT_TURNS} \
    +actor_rollout_ref.rollout.multi_turn.semantic_selection_full_observation=${SEMANTIC_SELECTION_FULL_OBSERVATION} \
    +actor_rollout_ref.rollout.multi_turn.semantic_selection_full_observation_max_chars=${SEMANTIC_SELECTION_FULL_OBSERVATION_MAX_CHARS} \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_LOOP_WORKERS:-16} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BSZ} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${REF_TOKEN_BUDGET_PER_GPU} \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP} \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${REF_VPP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP} \
    actor_rollout_ref.ref.megatron.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=True \
    +reward.custom_reward_function.path="$PROJECT_DIR/verl/utils/reward_score/bc_p_llm_judge.py" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_model.enable=False \
    +ray_kwargs.ray_init.address="${HEAD_IP}:6379" \
    +ray_kwargs.ray_init.runtime_env.env_vars.RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL="'${RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.RAY_ENABLE_UV_RUN_RUNTIME_ENV="'${RAY_ENABLE_UV_RUN_RUNTIME_ENV}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.RAY_enable_open_telemetry="'${RAY_enable_open_telemetry}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_DATAPROTO_SERIALIZATION_METHOD="'${VERL_DATAPROTO_SERIALIZATION_METHOD}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MASTER_PORT_RANGE="'${VERL_MASTER_PORT_RANGE}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_TRAINER_POOL_IPS="'${TRAINER_POOL_IPS}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_ROLLOUT_POOL_IPS="'${ROLLOUT_POOL_IPS}'" \
    trainer.use_legacy_worker_impl=disable \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=${TRAINER_GPUS_PER_NODE} \
    trainer.nnodes=${TRAINER_NNODES} \
    +trainer.node_ips="'${TRAINER_POOL_IPS}'" \
    rollout.n_gpus_per_node=${ROLLOUT_GPUS_PER_NODE} \
    rollout.nnodes=${ROLLOUT_NNODES} \
    +rollout.node_ips="'${ROLLOUT_POOL_IPS}'" \
    rollout.n=${N_RESP} \
    rollout.total_rollout_steps=${FULL_ASYNC_TOTAL_ROLLOUT_STEPS} \
    trainer.project_name='echo' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode="${BCP_RESUME_MODE}" \
    trainer.logger='["console", "wandb"]' \
    trainer.save_freq=${SAVE_FREQ:-10} \
    trainer.test_freq=${FULL_ASYNC_TEST_FREQ:-5} \
    trainer.total_epochs=5 \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} \
    async_training.staleness_threshold=${FULL_ASYNC_STALENESS_THRESHOLD} \
    async_training.trigger_parameter_sync_step=${FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP} \
    async_training.require_batches=${FULL_ASYNC_REQUIRE_BATCHES} \
    async_training.partial_rollout=${FULL_ASYNC_PARTIAL_ROLLOUT} \
    +trainer.master_port_range='[31000,32000]' \
    +trainer.sync_local_checkpoint_metadata=${SYNC_LOCAL_CHECKPOINT_METADATA:-True} \
    +trainer.sync_local_checkpoint_hf=${SYNC_LOCAL_CHECKPOINT_HF:-True} \
    +trainer.validation_data_dir="${VALIDATION_DATA_DIR}"
