#!/bin/bash

BCP_DEFAULT_ALL_TRAINERS="${BCP_DEFAULT_ALL_TRAINERS:-}"
BCP_DEFAULT_SLICE0_TRAINERS="${BCP_DEFAULT_SLICE0_TRAINERS:-}"

select_bcp_nodes() {
    NODES_PER_EXPERIMENT="${NODES_PER_EXPERIMENT:-4}"
    if [[ ! "$NODES_PER_EXPERIMENT" =~ ^[0-9]+$ ]] || [ "$NODES_PER_EXPERIMENT" -ne 4 ]; then
        echo "[config] ERROR: BCP scripts are configured for 4 nodes; got NODES_PER_EXPERIMENT=${NODES_PER_EXPERIMENT}."
        exit 1
    fi

    TRAINER_IPS="${TRAINER_IPS:-}"
    if [[ -n "$TRAINER_IPS" ]]; then
        :
    elif [[ -n "${PADDLE_TRAINERS:-}" || -n "${NODE_SLICE:-}" ]]; then
        SOURCE_TRAINERS="${PADDLE_TRAINERS:-${BCP_DEFAULT_ALL_TRAINERS}}"
        IFS=',' read -r -a PADDLE_TRAINER_ARRAY <<< "$SOURCE_TRAINERS"
        if [ "${#PADDLE_TRAINER_ARRAY[@]}" -gt "$NODES_PER_EXPERIMENT" ]; then
            NODE_SLICE="${NODE_SLICE:-${DEFAULT_NODE_SLICE:-0}}"
            if [[ ! "$NODE_SLICE" =~ ^[0-9]+$ ]]; then
                echo "[config] ERROR: NODE_SLICE must be a non-negative integer, got ${NODE_SLICE}."
                exit 1
            fi
            SLICE_START=$((NODE_SLICE * NODES_PER_EXPERIMENT))
            if [ "$SLICE_START" -lt 0 ] || [ $((SLICE_START + NODES_PER_EXPERIMENT)) -gt "${#PADDLE_TRAINER_ARRAY[@]}" ]; then
                echo "[config] ERROR: NODE_SLICE=${NODE_SLICE}, NODES_PER_EXPERIMENT=${NODES_PER_EXPERIMENT} is out of range for PADDLE_TRAINERS (${#PADDLE_TRAINER_ARRAY[@]} nodes)."
                exit 1
            fi
            SELECTED_IPS=("${PADDLE_TRAINER_ARRAY[@]:SLICE_START:NODES_PER_EXPERIMENT}")
            TRAINER_IPS="$(IFS=,; echo "${SELECTED_IPS[*]}")"
        elif [ "${#PADDLE_TRAINER_ARRAY[@]}" -eq "$NODES_PER_EXPERIMENT" ]; then
            if [[ -n "${NODE_SLICE:-}" && "${NODE_SLICE}" != "0" ]]; then
                echo "[config] ERROR: NODE_SLICE=${NODE_SLICE} is out of range for a 4-node PADDLE_TRAINERS list."
                exit 1
            fi
            TRAINER_IPS="${SOURCE_TRAINERS}"
        else
            echo "[config] ERROR: PADDLE_TRAINERS has ${#PADDLE_TRAINER_ARRAY[@]} nodes; expected 4."
            exit 1
        fi
    else
        DEFAULT_TRAINER_IPS="${DEFAULT_TRAINER_IPS:-${BCP_DEFAULT_SLICE0_TRAINERS}}"
        TRAINER_IPS="${DEFAULT_TRAINER_IPS}"
    fi

    IFS=',' read -r -a TRAINER_IP_ARRAY <<< "$TRAINER_IPS"
    if [ "${#TRAINER_IP_ARRAY[@]}" -ne "$NODES_PER_EXPERIMENT" ]; then
        echo "[config] ERROR: expected ${NODES_PER_EXPERIMENT} IPs, got ${#TRAINER_IP_ARRAY[@]}: ${TRAINER_IPS}"
        exit 1
    fi

    declare -gA SEEN_IPS=()
    for ip in "${TRAINER_IP_ARRAY[@]}"; do
        if [[ -z "$ip" ]]; then
            echo "[config] ERROR: empty IP in TRAINER_IPS=${TRAINER_IPS}"
            exit 1
        fi
        if [[ -n "${SEEN_IPS[$ip]:-}" ]]; then
            echo "[config] ERROR: duplicate IP in TRAINER_IPS=${TRAINER_IPS}: ${ip}"
            exit 1
        fi
        SEEN_IPS[$ip]=1
    done

    HEAD_IP="${HEAD_IP:-${TRAINER_IP_ARRAY[0]}}"
    if [[ "$HEAD_IP" != "${TRAINER_IP_ARRAY[0]}" ]]; then
        echo "[config] ERROR: HEAD_IP=${HEAD_IP} must match the first TRAINER_IPS entry (${TRAINER_IP_ARRAY[0]}). Reorder TRAINER_IPS to choose a different head."
        exit 1
    fi
    WORKER_IPS="${WORKER_IPS:-${TRAINER_IP_ARRAY[*]:1}}"

    export TRAINER_IPS HEAD_IP WORKER_IPS NODES_PER_EXPERIMENT
    echo "[config] selected TRAINER_IPS=${TRAINER_IPS}"
    echo "[config] head=${HEAD_IP}, workers=${WORKER_IPS}"
}
