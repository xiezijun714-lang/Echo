#!/usr/bin/env bash
# Single-node Blackwell ECHO-v1 (token-credit ECHO-CA) on BrowseComp-Plus.

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export BCP_ALGORITHM="supo"
export BCP_CONTEXT_COMPRESSION_METHOD="echo_e2e"
export ECHO_CREDIT_METHOD="${ECHO_CREDIT_METHOD:-token}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-32b-bcp-echo-v1-1node-32k-s5}"
exec bash "${PROJECT_DIR}/scripts/run_bcp_echo_graph_1node_blackwell.sh" "$@"
