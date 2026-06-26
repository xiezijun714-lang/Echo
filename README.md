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

> **Figure 1 — Training dynamics on BrowseComp-Plus** (ECHO = purple, GRPO =
> orange, SUPO = green), under an identical backbone, verifier, rollout budget,
> and sampling configuration. **Left:** held-out pass@1 accuracy over training.
> **Middle:** average tool-use turns per rollout. **Right:** trajectory volume
> (total generated tokens) per rollout. ECHO traces the upper-left frontier —
> rising accuracy *without* the turn and volume growth seen for SUPO.

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
```

**LLM-judge reward.** The reward function (`verl/utils/reward_score/bc_p_llm_judge.py`)
scores answers with an OpenAI-compatible chat-completions endpoint. Point it at any
compatible API (OpenAI, DeepSeek, a self-hosted vLLM/SGLang server, etc.) via:

```bash
export BCP_JUDGE_API_BASE="https://api.openai.com/v1"   # any OpenAI-compatible base URL
export BCP_JUDGE_MODEL="gpt-4o-mini"                     # judge model name on that endpoint
export BCP_JUDGE_API_KEY_ENV="OPENAI_API_KEY"           # name of the env var holding the key
export OPENAI_API_KEY="sk-..."                          # the key itself (name must match above)
```

The script reads the key from the environment variable named by
`BCP_JUDGE_API_KEY_ENV` (default `ONEAPI_KEY`) and fails fast if it is unset, so
there are no hardcoded credentials. Use your own provider and key.

For multi-node runs, the node list is resolved by `bcp_node_utils.sh` from
`TRAINER_IPS` (or the cluster-provided `PADDLE_TRAINERS`).

Available scripts in `examples/sglang_multiturn/`:

- `run_qwen3-32b_bcp_echo-ca_4node.sh` — ECHO (synchronous)
- `run_qwen3-32b_bcp_echo-ca_fully_async_4node.sh` — ECHO (fully async)
- `run_qwen3-32b_bcp_grpo_4node.sh` — GRPO baseline (synchronous)
- `run_qwen3-32b_bcp_grpo_fully_async_4node.sh` — GRPO baseline (fully async)
- `run_qwen3-30b-a3b_bcp_echo-ca_fully_async_4node.sh` — ECHO on the MoE backbone
- `run_qwen3-30b-a3b_bcp_grpo_fully_async_4node.sh` — GRPO on the MoE backbone
- `run_qwen3-32b_bcp_supo_4node.sh` — SUPO rolling-summary baseline (synchronous)

Run from the project root, e.g.:

```bash
bash examples/sglang_multiturn/run_qwen3-32b_bcp_echo-ca_fully_async_4node.sh
```

### Reproducing ablations

Ablation variants reuse the same core scripts and are toggled through environment
variables (see the top of each script for the full list). Key knobs:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONTEXT_COMPRESSION_METHOD` | `echo_e2e` | Context reconstruction strategy: `echo_e2e` (learned selection), `semantic_selection` (static top-k retrieval), `truncate` (left-truncation), `summarize` (SUPO rolling summary) |
| `ECHO_CREDIT_METHOD` | `token` | Credit routing: `token` (provenance-guided, ECHO), `traj` (trajectory-level), `none` (dense, all tokens) |
| `ECHO_RECENT_TURNS` | `3` | Number of most-recent turns always kept during reconstruction |
| `WORKING_CONTEXT_LENGTH` | `32768` | Single-segment token threshold that triggers compression |
| `MAX_SUMMARY_ROUNDS` | `5` | Max compression rounds before a rollout is marked overlong |
| `SEMANTIC_SELECTION_FULL_OBSERVATION` | `False` | When using `semantic_selection`, retrieve full observations instead of compact findings |
| `ECHO_CREDIT_PENALTY_RATIO` | `0.0` | Down-weight (vs. 1.0 for credited tokens) applied to non-credited tokens |

Examples reproducing paper ablations (all on top of the ECHO async script):

```bash
# Full ECHO (paper main): learned selection + provenance-guided token credit
bash examples/sglang_multiturn/run_qwen3-32b_bcp_echo-ca_fully_async_4node.sh

# Ablation: static semantic top-k retrieval instead of learned selection
CONTEXT_COMPRESSION_METHOD=semantic_selection \
  bash examples/sglang_multiturn/run_qwen3-32b_bcp_echo-ca_fully_async_4node.sh

# Ablation: semantic top-k retrieving full observations (not compact findings)
CONTEXT_COMPRESSION_METHOD=semantic_selection SEMANTIC_SELECTION_FULL_OBSERVATION=True \
  bash examples/sglang_multiturn/run_qwen3-32b_bcp_echo-ca_fully_async_4node.sh

# Ablation: w/o traceable credit assignment (dense credit on all tokens)
ECHO_CREDIT_METHOD=none \
  bash examples/sglang_multiturn/run_qwen3-32b_bcp_echo-ca_fully_async_4node.sh

# SUPO baseline (rolling summarization) — synchronous script
bash examples/sglang_multiturn/run_qwen3-32b_bcp_supo_4node.sh
```

## Results

### Ablations

<div align="center">
<img src="assets/component_ablation.png" width="49%" alt="Memory component ablation">
<img src="assets/credit_assignment_ablation.png" width="49%" alt="Credit assignment ablation">
</div>

> **Left — Memory component ablation.** Held-out accuracy vs. training, comparing
> ECHO's learned source selection against static semantic top-k retrieval, and
> compact last-turn findings against full observations. Learned selection is the
> main driver of accuracy; semantic top-k stays compact but plateaus lower, and
> full observations add no gain over compact findings.
>
> **Right — Credit assignment ablation.** ECHO's provenance-guided token credit
> vs. dense credit (w/o traceable CA, rewards all tokens) and all-turn importance
> weighting. Dense credit lowers accuracy and stability; all-turn weighting
> further inflates turn counts. Traceable credit gives the best accuracy/stability.

### MoE backbone

<div align="center">
<img src="assets/moe_experiment_adjusted.png" width="60%" alt="MoE experiment">
</div>

> **Transfer to the sparse MoE backbone (Qwen3-30B-A3B).** Held-out accuracy over
> training for ECHO vs. GRPO, showing the method's gains are not specific to the
> dense backbone.

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
