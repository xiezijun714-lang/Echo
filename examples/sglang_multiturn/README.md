# BrowseComp-Plus Multi-Turn RL (ECHO / GRPO)

This directory contains the BrowseComp-Plus (BCP) multi-turn search tool-calling
training scripts for ECHO and the GRPO baseline. See the repository root
[`README.md`](../../README.md) for the full method description, environment setup,
and required environment variables.

## Scripts

- `run_qwen3-32b_bcp_echo-ca_4node.sh` — ECHO (synchronous)
- `run_qwen3-32b_bcp_echo-ca_fully_async_4node.sh` — ECHO (fully async)
- `run_qwen3-32b_bcp_grpo_4node.sh` — GRPO baseline (synchronous)
- `run_qwen3-32b_bcp_grpo_fully_async_4node.sh` — GRPO baseline (fully async)
- `run_qwen3-30b-a3b_bcp_echo-ca_fully_async_4node.sh` — ECHO on the MoE backbone
- `run_qwen3-30b-a3b_bcp_grpo_fully_async_4node.sh` — GRPO on the MoE backbone

## Usage

Export the required paths (`VENV_PATH`, `MODEL_PATH`, `RETRIEVER_MODEL_PATH`,
`DATA_DIR`; see root README), then run from the project root:

```bash
bash examples/sglang_multiturn/run_qwen3-32b_bcp_echo-ca_fully_async_4node.sh
```

A dense retrieval service over the BrowseComp-Plus corpus
(`browsecomp_retrieval_server.py`) is started automatically and health-checked by
each script. Ablations reuse these scripts and are toggled through environment
variables documented at the top of each script.

