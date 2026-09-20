#!/usr/bin/env bash
# Single-node Blackwell GRPO baseline on BrowseComp-Plus.

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export BCP_ALGORITHM="grpo"
export BCP_CONTEXT_COMPRESSION_METHOD="${BCP_CONTEXT_COMPRESSION_METHOD:-summary}"
export ECHO_CREDIT_METHOD="none"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-32b-bcp-grpo-1node-32k-s5}"
exec bash "${PROJECT_DIR}/scripts/run_bcp_echo_graph_1node_blackwell.sh" "$@"
