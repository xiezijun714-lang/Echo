#!/usr/bin/env bash
# Single-node Blackwell ECHO-CA on BrowseComp-Plus.
#
# This is the provenance-guided token-credit ECHO baseline. The graph variant
# remains available through run_bcp_echo_graph_1node_blackwell.sh.

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export BCP_ALGORITHM="supo"
export BCP_CONTEXT_COMPRESSION_METHOD="echo_e2e"
export ECHO_CREDIT_METHOD="${ECHO_CREDIT_METHOD:-token}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-32b-bcp-echo-ca-1node-32k-s5}"
exec bash "${PROJECT_DIR}/examples/sglang_multiturn/run_bcp_echo_graph_1node_blackwell.sh" "$@"
