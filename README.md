<div align="center">

# ECHO: Prune to Act, Trace to Learn with Selective Turn Memory in Agentic RL

</div>

ECHO is a selective turn-memory framework for traceable context reconstruction in
agentic reinforcement learning, built on top of [verl](https://github.com/volcengine/verl).

Long-horizon language agents must act under bounded contexts while learning from
sparse outcome rewards. Common context-management methods make long rollouts
feasible by truncating history or folding multiple turns into rolling summaries,
but these transformations lose source-level addressability: once historical
observations are collapsed, outcome-based RL has no explicit route for assigning
delayed credit back to the original evidence turns that later decisions
conditioned on.

ECHO addresses this with two coupled ideas:

- **Prune to Act** — each completed turn is compressed into a *source-indexed*
  memory; distant history is kept as a non-collapsing memory set, and a bounded
  policy context is reconstructed by selecting relevant memories together with
  recent interactions.
- **Trace to Learn** — the selected source indices are reused as provenance
  routes for delayed credit assignment, so learning rewards the final segment and
  the useful historical memory tokens instead of all generated tokens.

<div align="center">
<img src="assets/main_experiment.png" width="100%" alt="Main results on BrowseComp-Plus">
</div>

On BrowseComp-Plus, ECHO reaches **43.4%** held-out accuracy, outperforming GRPO
(28.9%) and the rolling-summary baseline SUPO (36.1%), while using fewer turns and
lower trajectory volume than SUPO. The trained policy also improves zero-shot
generalization across multi-objective QA, code generation, and deep
information-seeking benchmarks on both dense (Qwen3-32B) and MoE (Qwen3-30B-A3B)
backbones.

## Method

<div align="center">
<img src="assets/echo_ca.png" width="98%" alt="ECHO credit assignment">
</div>

ECHO separates local turn compression from global context reconstruction. After
each completed tool-use turn, the policy summarizes only that turn into a compact,
source-indexed memory clue. When the working context reaches the compression
threshold, ECHO reconstructs a bounded context by keeping the most recent turns
and letting the policy select relevant historical memories. The selected source
indices form provenance routes that guide credit assignment during the GRPO-style
update.

## Installation

Reference environment used for the paper:

- Python 3.10, CUDA 12.8
- GPU: NVIDIA H800 (80GB), 8 GPUs per node
- PyTorch 2.7.1+cu128, sglang 0.4.10, megatron-core 0.13.2
- transformer_engine 2.5.0, flash_attn 2.7.4.post1, flashinfer 0.2.6.post1

The CUDA-compiled stack (torch / sglang / megatron-core / transformer_engine /
flash_attn / flashinfer / apex) must be installed in order against your CUDA
toolkit — see [`requirements-cuda.txt`](requirements-cuda.txt) for the pinned
versions and suggested install order. After the CUDA stack is in place:

```bash
pip install -r requirements.txt
pip install -e .
```

## Data preparation

ECHO is trained and evaluated on
[BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus). Build the prompt
parquet files with:

```bash
python3 examples/data_preprocess/bcp_paper_prompt.py
```

This produces `train.paper.parquet` and `test.paper.parquet` under your dataset
directory. A dense retrieval service over the BrowseComp-Plus corpus is launched
automatically by the training scripts
(`examples/sglang_multiturn/browsecomp_retrieval_server.py`).

## Training

The scripts target a 4-node × 8×H800 setup with the Megatron backend and SGLang
rollout. Before launching, export the required paths (scripts fail fast if these
are unset):

```bash
export VENV_PATH=/path/to/your/venv               # Python virtualenv
export MODEL_PATH=/path/to/Qwen3-32B              # policy model
export RETRIEVER_MODEL_PATH=/path/to/Qwen3-Embedding-8B
export DATA_DIR=/path/to/browsecomp-plus-processed  # contains *.paper.parquet

# Optional services / accounts
export WANDB_API_KEY=...   export WANDB_ENTITY=...   # if using Weights & Biases
export ONEAPI_KEY=...                                # LLM-judge reward API key
```

For multi-node runs, the node list is resolved by `bcp_node_utils.sh` from
`TRAINER_IPS` (or the cluster-provided `PADDLE_TRAINERS`).

Available scripts in `examples/sglang_multiturn/`:

- `run_qwen3-32b_bcp_echo-ca_4node.sh` — ECHO (synchronous)
- `run_qwen3-32b_bcp_echo-ca_fully_async_4node.sh` — ECHO (fully async)
- `run_qwen3-32b_bcp_grpo_4node.sh` — GRPO baseline (synchronous)
- `run_qwen3-32b_bcp_grpo_fully_async_4node.sh` — GRPO baseline (fully async)
- `run_qwen3-30b-a3b_bcp_echo-ca_fully_async_4node.sh` — ECHO on the MoE backbone
- `run_qwen3-30b-a3b_bcp_grpo_fully_async_4node.sh` — GRPO on the MoE backbone

Run from the project root, e.g.:

```bash
bash examples/sglang_multiturn/run_qwen3-32b_bcp_echo-ca_fully_async_4node.sh
```

Ablation variants (e.g. different context-compression methods or credit-assignment
settings) reuse the same core scripts and are selected through environment
variables — for example `CONTEXT_COMPRESSION_METHOD`, `ECHO_CREDIT_METHOD`,
`ECHO_RECENT_TURNS`, and `WORKING_CONTEXT_LENGTH`. See the top of each script for
the full list of tunable variables.

## Results

<div align="center">
<img src="assets/component_ablation.png" width="49%" alt="Component ablation">
<img src="assets/credit_assignment_ablation.png" width="49%" alt="Credit assignment ablation">
</div>

<div align="center">
<img src="assets/moe_experiment_adjusted.png" width="60%" alt="MoE experiment">
</div>

## Acknowledgements

ECHO is built on [verl](https://github.com/volcengine/verl) (Volcano Engine
Reinforcement Learning for LLMs). We thank the verl team and community.

## Citation

```bibtex
@article{echo,
  title  = {ECHO: Prune to Act, Trace to Learn with Selective Turn Memory in Agentic RL},
  author = {Xie, Zijun and others},
  year   = {2026}
}
```
