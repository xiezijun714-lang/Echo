#!/usr/bin/env bash

# Shared runtime setup for the BrowseComp-Plus training entrypoints.

bcp_is_true() {
    case "${1:-}" in
        1|true|TRUE|True|yes|YES|Yes|on|ON|On) return 0 ;;
        *) return 1 ;;
    esac
}

bcp_load_env() {
    local project_dir="$1"
    local env_file="${BCP_ENV_FILE:-${project_dir}/.env}"

    if [ ! -f "$env_file" ]; then
        return
    fi

    echo "[config] Loading environment from ${env_file}"
    local restore_allexport=0
    if [[ $- == *a* ]]; then
        restore_allexport=1
    fi
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    if [ "$restore_allexport" -eq 0 ]; then
        set +a
    fi
}

bcp_enable_shell_trace() {
    if bcp_is_true "${BCP_SHELL_TRACE:-False}"; then
        set -x
    fi
}

bcp_configure_python() {
    local project_dir="$1"
    local venv_path="$2"
    local python_bin="${ECHO_PYTHON_BIN:-${venv_path}/bin/python3}"
    local ray_bin="${ECHO_RAY_BIN:-${venv_path}/bin/ray}"

    if [ ! -x "$python_bin" ]; then
        echo "[config] ERROR: Python is missing or not executable: ${python_bin}"
        return 1
    fi
    if [ ! -x "$ray_bin" ]; then
        echo "[config] ERROR: Ray is missing or not executable: ${ray_bin}"
        return 1
    fi

    # Copied virtualenvs often retain a stale absolute path in bin/activate.
    # Reset these values after activation and always put this checkout first.
    # shellcheck disable=SC1091
    source "${venv_path}/bin/activate"
    export VIRTUAL_ENV="$venv_path"
    export PATH="$(dirname "$python_bin"):${venv_path}/bin:${PATH}"

    local python_version
    local venv_site_packages
    local transformer_engine_overlay
    python_version="$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    venv_site_packages="${venv_path}/lib/python${python_version}/site-packages"
    if [ ! -d "$venv_site_packages" ]; then
        echo "[config] ERROR: packaged site-packages do not match Python ${python_version}: ${venv_site_packages}"
        return 1
    fi
    transformer_engine_overlay="${ECHO_TRANSFORMER_ENGINE_OVERLAY:-${venv_path}/../overlays/transformer-engine-2.5.0-glibc236}"
    if [ -d "${transformer_engine_overlay}/transformer_engine" ]; then
        export PYTHONPATH="${project_dir}:${transformer_engine_overlay}:${venv_site_packages}${PYTHONPATH:+:${PYTHONPATH}}"
        echo "[config] Using Transformer Engine overlay: ${transformer_engine_overlay}"
    else
        export PYTHONPATH="${project_dir}:${venv_site_packages}${PYTHONPATH:+:${PYTHONPATH}}"
    fi

    local nvidia_site_packages="${venv_site_packages}/nvidia"
    local cuda_library_path=""
    local cuda_component
    for cuda_component in nvjitlink cusparse cublas cudnn cuda_runtime cuda_nvrtc; do
        if [ -d "${nvidia_site_packages}/${cuda_component}/lib" ]; then
            cuda_library_path+="${nvidia_site_packages}/${cuda_component}/lib:"
        fi
    done
    export LD_LIBRARY_PATH="${cuda_library_path}${venv_site_packages}/torch/lib:${LD_LIBRARY_PATH:-}"

    PYTHON_BIN="$python_bin"
    RAY_BIN="$ray_bin"
    export ECHO_PYTHON_BIN="$PYTHON_BIN" ECHO_RAY_BIN="$RAY_BIN"
    export PYTHON_BIN RAY_BIN PYTHONPATH

    local imported_verl
    imported_verl="$("$PYTHON_BIN" -c 'import pathlib, verl; print(pathlib.Path(verl.__file__).resolve())')" || {
        echo "[config] ERROR: failed to import verl with ${PYTHON_BIN}."
        return 1
    }
    case "$imported_verl" in
        "${project_dir}"/*) ;;
        *)
            echo "[config] ERROR: Python imported verl from ${imported_verl}, expected ${project_dir}/verl."
            return 1
            ;;
    esac
    echo "[config] Python ${python_version} will use ${imported_verl}"
}

bcp_validate_head_node() {
    local head_ip="$1"
    local local_ips="${POD_IP:-} $(hostname -I 2>/dev/null || true)"

    if [[ -n "${local_ips// /}" && " ${local_ips} " != *" ${head_ip} "* ]]; then
        echo "[config] ERROR: current host IPs (${local_ips}) do not include HEAD_IP=${head_ip}; run this script on the slice head node."
        return 1
    fi
}

bcp_resolve_retriever_data() {
    local data_dir="$1"

    RETRIEVER_CORPUS_FILE="${RETRIEVER_CORPUS_FILE:-${data_dir}/corpus.parquet}"
    RETRIEVER_DENSE_CACHE="${RETRIEVER_DENSE_CACHE:-${data_dir}/browsecomp_dense_cache.pkl}"

    # Preserve compatibility with the layout used by the runnable internal
    # scripts: <root>/dataset/browsecomp-plus-processed plus a cache at <root>.
    local legacy_cache="${data_dir}/../../browsecomp_dense_cache_tevatron.pkl"
    if [ ! -e "$RETRIEVER_DENSE_CACHE" ] && [ -e "$legacy_cache" ]; then
        RETRIEVER_DENSE_CACHE="$(cd "$(dirname "$legacy_cache")" && pwd)/$(basename "$legacy_cache")"
        echo "[config] Using legacy dense retrieval cache: ${RETRIEVER_DENSE_CACHE}"
    fi
    export RETRIEVER_CORPUS_FILE RETRIEVER_DENSE_CACHE
}

bcp_validate_training_paths() {
    local project_dir="$1"
    local venv_path="$2"
    local model_path="$3"
    local train_file="$4"
    local val_file="$5"
    local retriever_model_path="$6"
    local expected_model_type="$7"

    local required_command
    for required_command in ssh curl setsid tee; do
        if ! command -v "$required_command" >/dev/null 2>&1; then
            echo "[config] ERROR: required command is unavailable: ${required_command}"
            return 1
        fi
    done
    if ! command -v lsof >/dev/null 2>&1 && ! command -v fuser >/dev/null 2>&1; then
        echo "[config] ERROR: either lsof or fuser is required to manage port listeners"
        return 1
    fi

    local required_path
    for required_path in \
        "$project_dir" \
        "${project_dir}/verl/__init__.py" \
        "$PYTHON_BIN" \
        "${venv_path}/bin/ray" \
        "$model_path" \
        "${model_path}/config.json" \
        "$train_file" \
        "$val_file" \
        "$retriever_model_path"; do
        if [ ! -e "$required_path" ]; then
            echo "[config] ERROR: required path does not exist: ${required_path}"
            return 1
        fi
    done

    local actual_model_type
    if ! actual_model_type="$("$PYTHON_BIN" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("model_type", ""))' "${model_path}/config.json")"; then
        echo "[config] ERROR: failed to read model_type from ${model_path}/config.json"
        return 1
    fi
    if [ "$actual_model_type" != "$expected_model_type" ]; then
        echo "[config] ERROR: model_type=${actual_model_type:-<missing>} at ${model_path}; this launcher requires ${expected_model_type}."
        return 1
    fi

    if [ ! -e "$RETRIEVER_DENSE_CACHE" ] && [ ! -e "$RETRIEVER_CORPUS_FILE" ]; then
        echo "[config] ERROR: retriever needs either a dense cache or a corpus file."
        echo "[config]        missing dense_cache=${RETRIEVER_DENSE_CACHE}"
        echo "[config]        missing corpus_file=${RETRIEVER_CORPUS_FILE}"
        return 1
    fi
}

bcp_validate_remote_paths() {
    local project_dir="$1"
    local venv_path="$2"
    local model_path="$3"
    local train_file="$4"
    local val_file="$5"
    local python_bin="${ECHO_PYTHON_BIN:-${venv_path}/bin/python3}"

    if bcp_is_true "${BCP_SKIP_REMOTE_CHECK:-False}"; then
        echo "[preflight] Skipping remote SSH checks (BCP_SKIP_REMOTE_CHECK=True)."
        return
    fi

    local ip
    for ip in $WORKER_IPS; do
        local remote_status=0
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$ip" \
            "test -f '${model_path}/config.json' && test -x '${python_bin}' && test -x '${venv_path}/bin/ray' && test -f '${project_dir}/verl/__init__.py' && test -f '${train_file}' && test -f '${val_file}'" \
            2>/dev/null || remote_status=$?
        if [ "$remote_status" -eq 255 ]; then
            echo "[config] ERROR: cannot connect to node ${ip} with passwordless SSH."
            return 1
        fi
        if [ "$remote_status" -ne 0 ]; then
            echo "[config] ERROR: node ${ip} cannot access the model, venv, dataset, or current checkout."
            return 1
        fi
    done
    echo "[preflight] Remote paths are visible on all worker nodes."
}

bcp_validate_pool_partition() {
    local all_ips_csv="$1"
    local trainer_ips_csv="$2"
    local rollout_ips_csv="$3"
    local -a all_ips trainer_ips rollout_ips
    local ip
    declare -A allowed=()
    declare -A assigned=()

    IFS=',' read -r -a all_ips <<< "$all_ips_csv"
    IFS=',' read -r -a trainer_ips <<< "$trainer_ips_csv"
    IFS=',' read -r -a rollout_ips <<< "$rollout_ips_csv"
    for ip in "${all_ips[@]}"; do
        allowed[$ip]=1
    done
    for ip in "${trainer_ips[@]}" "${rollout_ips[@]}"; do
        if [[ -z "${allowed[$ip]:-}" ]]; then
            echo "[config] ERROR: resource-pool IP ${ip} is not in TRAINER_IPS=${all_ips_csv}."
            return 1
        fi
        if [[ -n "${assigned[$ip]:-}" ]]; then
            echo "[config] ERROR: resource-pool IP ${ip} is assigned more than once."
            return 1
        fi
        assigned[$ip]=1
    done
}

bcp_prepare_service_env() {
    BCP_JUDGE_API_BASE="${BCP_JUDGE_API_BASE:-}"
    BCP_JUDGE_MODEL="${BCP_JUDGE_MODEL:-DeepSeek-V4-Flash}"
    BCP_JUDGE_API_KEY_ENV="${BCP_JUDGE_API_KEY_ENV:-ONEAPI_KEY}"

    if [ -z "$BCP_JUDGE_API_BASE" ]; then
        echo "[config] ERROR: set BCP_JUDGE_API_BASE to an OpenAI-compatible /v1 endpoint."
        return 1
    fi
    if [[ ! "$BCP_JUDGE_API_KEY_ENV" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        echo "[config] ERROR: invalid BCP_JUDGE_API_KEY_ENV=${BCP_JUDGE_API_KEY_ENV}"
        return 1
    fi
    if [ -z "${!BCP_JUDGE_API_KEY_ENV:-}" ]; then
        echo "[config] ERROR: LLM judge key is missing. Set ${BCP_JUDGE_API_KEY_ENV}, or choose its name with BCP_JUDGE_API_KEY_ENV."
        return 1
    fi
    export "$BCP_JUDGE_API_KEY_ENV"

    if [ -z "${WANDB_API_KEY:-}" ] && [ -z "${WANDB_MODE:-}" ]; then
        WANDB_MODE=offline
    fi
    WANDB_API_KEY="${WANDB_API_KEY:-}"
    WANDB_ENTITY="${WANDB_ENTITY:-}"
    WANDB_MODE="${WANDB_MODE:-online}"
    WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
    WANDB_PROJECT="${WANDB_PROJECT:-echo}"
    http_proxy="${http_proxy:-}"
    https_proxy="${https_proxy:-}"
    HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
    HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
    no_proxy="${no_proxy:-}"
    NO_PROXY="${NO_PROXY:-$no_proxy}"
    export BCP_JUDGE_API_BASE BCP_JUDGE_MODEL BCP_JUDGE_API_KEY_ENV
    export WANDB_API_KEY WANDB_ENTITY WANDB_MODE WANDB_BASE_URL WANDB_PROJECT
    export http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

    local bcp_ray_env_vars
    bcp_ray_env_vars="PYTHONPATH,CUDA_DEVICE_MAX_CONNECTIONS,PYTHONUNBUFFERED,BCP_JUDGE_API_BASE,BCP_JUDGE_MODEL,BCP_JUDGE_API_KEY_ENV,BCP_JUDGE_MAX_TOKENS,BCP_JUDGE_TEMPERATURE,BCP_JUDGE_TOP_P,BCP_JUDGE_TOP_K,${BCP_JUDGE_API_KEY_ENV},WANDB_API_KEY,WANDB_ENTITY,WANDB_MODE,WANDB_BASE_URL,WANDB_PROJECT,http_proxy,https_proxy,HTTP_PROXY,HTTPS_PROXY,no_proxy,NO_PROXY,XDG_CACHE_HOME,HF_HOME,PIP_CACHE_DIR,UV_CACHE_DIR,FLASHINFER_WORKSPACE_BASE,TORCHINDUCTOR_CACHE_DIR,TORCHINDUCTOR_COMPILE_THREADS,CUDA_CACHE_PATH,SGLANG_FORCE_NATIVE_CUSTOM_OPS"
    VERL_RAY_RUNTIME_ENV_VARS="${VERL_RAY_RUNTIME_ENV_VARS:+${VERL_RAY_RUNTIME_ENV_VARS},}${bcp_ray_env_vars}"
    export VERL_RAY_RUNTIME_ENV_VARS

    # Kept as an empty array so existing launch commands need no special case.
    # Values are forwarded directly by get_ppo_ray_runtime_env(), keeping
    # credentials out of Hydra config dumps and process arguments.
    BCP_RAY_ENV_OVERRIDES=()
}

bcp_prepare_tool_config() {
    local template_path="$1"
    local output_path="$2"
    local retrieval_host="$3"
    local retrieval_port="${4:-8000}"

    if [ ! -f "$template_path" ]; then
        echo "[config] ERROR: tool config template does not exist: ${template_path}"
        return 1
    fi
    mkdir -p "$(dirname "$output_path")"
    cp "$template_path" "$output_path"
    sed -i -E "s#http://[^/]+:[0-9]+/(retrieve|get_doc)#http://${retrieval_host}:${retrieval_port}/\1#g" "$output_path"
    echo "[config] Tool config: ${output_path}"
}

bcp_stop_port_listeners() {
    local port="$1"
    local pids
    local pid
    local attempt

    if command -v lsof >/dev/null 2>&1; then
        pids="$(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    else
        pids="$(fuser -n tcp "$port" 2>/dev/null || true)"
    fi
    if [ -z "$pids" ]; then
        return 0
    fi
    echo "[retriever] Stopping existing listener(s) on port ${port}: ${pids//$'\n'/ }"
    for pid in $pids; do
        kill "$pid" 2>/dev/null || true
    done
    for ((attempt = 0; attempt < 25; attempt++)); do
        if command -v lsof >/dev/null 2>&1; then
            pids="$(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
        else
            pids="$(fuser -n tcp "$port" 2>/dev/null || true)"
        fi
        [ -z "$pids" ] && return 0
        sleep 0.2
    done
    for pid in $pids; do
        kill -KILL "$pid" 2>/dev/null || true
    done
}

bcp_require_bindable_port() {
    local python_bin="$1"
    local port="$2"

    "$python_bin" - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
try:
    sock.bind(("0.0.0.0", port))
except OSError as exc:
    print(f"[retriever] ERROR: port {port} is unavailable: {exc}", file=sys.stderr)
    raise SystemExit(1)
finally:
    sock.close()
PY
}

bcp_wait_for_retriever() {
    local retriever_pid="$1"
    local retriever_log="$2"
    local retriever_port="${3:-8000}"
    local timeout="${RETRIEVER_STARTUP_TIMEOUT_S:-1200}"
    local i

    if [[ ! "$timeout" =~ ^[1-9][0-9]*$ ]]; then
        echo "[config] ERROR: RETRIEVER_STARTUP_TIMEOUT_S must be a positive integer, got ${timeout}."
        return 1
    fi
    for ((i = 0; i < timeout; i++)); do
        if ! kill -0 "$retriever_pid" 2>/dev/null; then
            echo "[retriever] ERROR: server exited before becoming healthy. Log:"
            tail -n 200 "$retriever_log" 2>/dev/null || true
            return 1
        fi
        if curl -sf "http://127.0.0.1:${retriever_port}/health" >/dev/null 2>&1; then
            echo "[retriever] Server is healthy."
            return 0
        fi
        sleep 1
    done
    echo "[retriever] ERROR: health check timed out after ${timeout}s. Log:"
    tail -n 200 "$retriever_log" 2>/dev/null || true
    return 1
}

bcp_preflight_complete() {
    if ! bcp_is_true "${BCP_PREFLIGHT_ONLY:-False}"; then
        return 1
    fi

    local module_name="$1"
    local config_path="$2"
    local config_name="$3"
    echo "[preflight] Composing Hydra config through ${module_name} ..."
    "$PYTHON_BIN" -m "$module_name" \
        --config-path="$config_path" \
        --config-name="$config_name" \
        "${BCP_RAY_ENV_OVERRIDES[@]}" \
        --cfg job \
        --resolve \
        >/dev/null
    echo "[preflight] Local paths, imports, and base Hydra configuration are valid. No Ray or GPU process was started."
    return 0
}
