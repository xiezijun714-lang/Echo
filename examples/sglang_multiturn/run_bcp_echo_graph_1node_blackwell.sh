#!/usr/bin/env bash
# Multi-turn Search Tool-Calling ECHO-Graph on BrowseComp-Plus: 1 node x 8 B
# cards, Qwen3-32B, using the original Context-Folding split.
#
# This wrapper keeps the ECHO-CA rollout/selection path unchanged and selects
# the graph-based turn-memory credit assignment in the PPO algorithm. It is
# intentionally separate so graph experiments cannot silently reuse a token-
# credit checkpoint or experiment name.
#
# Run on node 0 of the allocated slice:
#   bash examples/sglang_multiturn/run_bcp_echo_graph_1node_blackwell.sh
#
# Config-only check (no Ray, no GPU):
#   BCP_PREFLIGHT_ONLY=True bash examples/sglang_multiturn/run_bcp_echo_graph_1node_blackwell.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PACKAGE_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"

# Pin the same GPFS layout and node-0 topology as the regular 1-node ECHO
# wrapper. Every value remains overrideable for smoke tests and tuning.
export DATA_DIR="${DATA_DIR:-${PACKAGE_ROOT}/dataset/browsecomp-plus-context-folding}"
export ECHO_CREDIT_METHOD="graph"
export ECHO_NEG_PENALTY_RATIO="${ECHO_NEG_PENALTY_RATIO:-0.0}"
export ECHO_GRAPH_GAMMA_TURN="${ECHO_GRAPH_GAMMA_TURN:-1.0}"
export ECHO_GRAPH_GAMMA_SEGMENT="${ECHO_GRAPH_GAMMA_SEGMENT:-0.9}"
export ECHO_GRAPH_AGGREGATION="${ECHO_GRAPH_AGGREGATION:-sum}"
# The graph baseline uses sum aggregation with a clip cap of 5. Current
# ablations vary turn-level gamma, negative dense penalty, and summary rounds.
export ECHO_GRAPH_CLIP_MAX="${ECHO_GRAPH_CLIP_MAX:-5}"
export MAX_SUMMARY_ROUNDS="${MAX_SUMMARY_ROUNDS:-5}"
if [ -z "${EXPERIMENT_NAME:-}" ]; then
    export EXPERIMENT_NAME="qwen3-32b-bcp-echo-graph-32k-s${MAX_SUMMARY_ROUNDS}-gturn${ECHO_GRAPH_GAMMA_TURN}-gseg${ECHO_GRAPH_GAMMA_SEGMENT}-neg${ECHO_NEG_PENALTY_RATIO}"
else
    export EXPERIMENT_NAME
fi
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export TEST_FREQ="${TEST_FREQ:-5}"

exec bash "${PROJECT_DIR}/examples/sglang_multiturn/run_bcp_echo_ca_1node_blackwell.sh" "$@"
