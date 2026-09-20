# BrowseComp-Plus Multi-Turn RL (ECHO / GRPO)

This directory contains the BrowseComp-Plus (BCP) multi-turn search tool-calling
training scripts for ECHO and the GRPO baseline. See the repository root
[`README.md`](../../README.md) for the full method description, environment setup,
and required environment variables.

## Scripts

- `run_bcp_echo_graph_1node_blackwell.sh` — ECHO-Graph on one Blackwell node
- `run_bcp_echo_ca_1node_blackwell.sh` — ECHO-CA with token credit on one Blackwell node
- `run_bcp_grpo_1node_blackwell.sh` — GRPO baseline on one Blackwell node

## Usage

Export the required paths (`VENV_PATH`, `MODEL_PATH`, `RETRIEVER_MODEL_PATH`,
`DATA_DIR`; see root README), then run from the project root:

```bash
bash examples/sglang_multiturn/run_bcp_echo_graph_1node_blackwell.sh
```

A dense retrieval service over the BrowseComp-Plus corpus
(`browsecomp_retrieval_server.py`) is started automatically and health-checked by
each script. Ablations reuse these scripts and are toggled through environment
variables documented at the top of each script.
