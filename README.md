<div align="center">

# 🔊 ECHO: Prune to Act, Trace to Learn with Selective Turn Memory in Agentic RL

<p>
  <a href="https://arxiv.org/abs/2606.31650"><img src="https://img.shields.io/badge/arXiv-2606.31650-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/b1Ack714/echo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Hugging%20Face-yellow" alt="Hugging Face Dataset"></a>
</p>

</div>

**ECHO** is a selective turn-memory framework for traceable context reconstruction in
agentic reinforcement learning, built on top of [verl](https://github.com/volcengine/verl).

Long-horizon language agents must act under bounded contexts while learning from
sparse outcome rewards. Common context-management methods make long rollouts
feasible by truncating history or folding multiple turns into rolling summaries,
but these transformations can weaken or remove source-level addressability. In
particular, once historical observations are collapsed into a single state,
outcome-based RL has no explicit route for assigning delayed credit back to the
original evidence turns that later decisions conditioned on.

ECHO addresses this with two coupled ideas:

- ✂️ **Prune to Act** — each completed turn is compressed into a *source-indexed*
  memory and retained in a persistent, non-collapsing archive. A bounded policy
  context is reconstructed by selecting relevant memories together with recent
  interactions.
- 🔍 **Trace to Learn** — the final reconstruction's selected source indices are
  reused as provenance routes, so positive outcome credit is routed to the final
  segment and the historical tokens that the policy actually reused.

<div align="center">
<img src="assets/main_experiment.png" width="100%" alt="Main results on BrowseComp-Plus">
</div>

> **Figure 1 — Training dynamics on BrowseComp-Plus** (ECHO = purple, GRPO =
> orange, SUPO = green), under the same backbone, tool environment, verifier,
> rollout group size, and 32k working-context budget. **Left:** held-out pass@1
> accuracy over training.
> **Middle:** average tool-use turns per rollout. **Right:** trajectory volume
> (total generated tokens) per rollout. ECHO traces the upper-left frontier —
> rising accuracy *without* the turn and volume growth seen for SUPO.

On BrowseComp-Plus, ECHO reaches **43.4%** held-out accuracy, outperforming GRPO
(28.9%) and the rolling-summary baseline SUPO (36.1%), while using fewer turns and
lower trajectory volume than SUPO. The trained policy also improves zero-shot
generalization across multi-objective QA, code generation, and deep
information-seeking benchmarks on both dense (Qwen3-32B-Instruct) and MoE
(Qwen3-30B-A3B-Instruct) backbones.

---

## 🧠 Method

ECHO has two coupled stages: it **prunes** distant history into selectable,
source-indexed memory so the policy can keep acting under a bounded context, and
it **traces** the final reconstruction's source indices through the update so
positive credit flows only along the evidence path the policy actually reused.

### 💡 Motivation

| History transformation | Managed state | Reconstruction form | Source trace | Examples |
| --- | --- | --- | --- | --- |
| Append-only | $`H_j^{\mathrm{pre}}`$ | $`H_j^{\mathrm{pre}} \oplus H_{j,t}^{\mathrm{loc}}`$ | Explicit | Vanilla prompting |
| Deletion / pruning | $`C_j \subseteq H_j^{\mathrm{pre}}`$ | $`\mathrm{Render}(C_j, H_{j,t}^{\mathrm{loc}}; B)`$ | Partial / lost | Sliding window<sup>R</sup>, Agent-Omit<sup>A</sup> |
| Edited memory state | $`B_j`$ | $`\mathrm{Render}(B_j, H_{j,t}^{\mathrm{loc}}; B)`$ | Action-traced | MemAct<sup>A</sup>, Memory-R1<sup>A</sup> |
| Collapsed folding | $`z_j=C(z_{j-1},\sigma_{j-1})`$ | $`\mathrm{Render}(z_j) \oplus H_{j,t}^{\mathrm{loc}}`$ | Collapsed | SUPO<sup>R</sup>, FoldGRPO<sup>A</sup> |
| **Selective turn memory** | $`\mathcal{E}_j=\{e_i\}_{i \le K_{j-1}}`$ | $`\mathrm{Render}(\mathcal{E}_j[\hat I_j]) \oplus H_{j,t}^{\mathrm{loc}}`$ | **Source-indexed** | **ECHO<sup>R</sup>** |

> Methods are grouped by how they transform completed history. Superscripts
> describe the separate reconstruction trigger: **R** is rule-triggered and
> **A** is action-triggered. “Source trace” asks whether reconstructed context
> remains addressable at the level of the original environment turns.
>
> After each completed tool-use turn, the same policy writes a compact finding and
> appends its source index, compact action, and finding to a persistent,
> non-collapsing archive. At a reconstruction boundary, the policy selects useful
> records and combines them with automatically retained recent turns. Selection
> changes only the bounded active view; unselected records remain available at
> later boundaries. The selected source indices are used both to reconstruct
> context and to route delayed credit during learning.

<div align="center">
<img src="assets/motivation_training_diagnostics.png" width="98%" alt="Training diagnostics: summarization induces turn proliferation and volume growth">
</div>

> Training diagnostics on long-horizon search. Summarization-based context
> management enables longer rollouts, but also drives turn proliferation, longer
> responses, higher generation time, and inflated trajectory volume — motivating a
> reconstruction scheme that stays compact while remaining source-traceable.

### 🎯 Credit Assignment

<div align="center">
<img src="assets/echo_ca.png" width="98%" alt="Provenance-guided credit assignment in ECHO">
</div>

> **Overview of ECHO.** GRPO and SUPO apply the signed trajectory advantage
> *densely* to all trainable response tokens (in SUPO, including rolling-summary
> tokens), so the outcome reward does not distinguish useful evidence from
> redundant searches. ECHO instead uses a **final-trace approximation**: source
> turns selected into the final reconstructed context define the historical tokens
> eligible for credit. Its token-level mask keeps **(i)** every trainable token in
> the final saved segment, **(ii)** trainable assistant/action tokens from those
> selected source turns, **(iii)** their memory-finding tokens, and **(iv)**
> generated memory-selection spans. The final saved segment begins after the last
> context reconstruction (or at rollout start if no reconstruction occurs) and
> includes assistant reasoning and tool-call tokens through the terminal response.
> Only a rollout with positive group-relative advantage reinforces this trace;
> zero- or negative-advantage rollouts receive no ECHO update. Dense signed credit
> over all response tokens is retained only as the "w/o traceable CA" ablation.

---

## 🛠️ Installation

Reference environment used for the paper:

- 🐍 Python 3.10, CUDA 12.8
- 🖥️ GPU: NVIDIA H800 (80GB), 8 GPUs per node
- ⚙️ PyTorch 2.7.1+cu128, sglang 0.4.10, megatron-core 0.13.2
- transformer_engine 2.5.0, flash_attn 2.7.4.post1, flashinfer 0.2.6.post1

The CUDA-compiled stack (torch / sglang / megatron-core / transformer_engine /
flash_attn / flashinfer / apex) must be installed in order against your CUDA
toolkit — see [`requirements-cuda.txt`](requirements-cuda.txt) for the pinned
versions and suggested install order. After the CUDA stack is in place:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 📦 Data Preparation

ECHO is trained and evaluated on
[BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus).

You can download the preprocessed ECHO data from
[Hugging Face](https://huggingface.co/datasets/b1Ack714/echo):

```bash
huggingface-cli download b1Ack714/echo \
  --repo-type dataset \
  --local-dir /path/to/browsecomp-plus-processed
```

Then point the training scripts at that directory:

```bash
export DATA_DIR=/path/to/browsecomp-plus-processed
```

Alternatively, build the prompt parquet files from a local BrowseComp-Plus copy
with:

```bash
python3 examples/data_preprocess/bcp_paper_prompt.py --data-dir "$DATA_DIR"
```

This reads `train.parquet` and `test.parquet` and writes `train.paper.parquet`
and `test.paper.parquet` under `DATA_DIR`. Training also requires either the
official corpus at `$DATA_DIR/corpus.parquet` or a prebuilt dense cache at
`$DATA_DIR/browsecomp_dense_cache.pkl`. To build the cache ahead of time:

```bash
python3 examples/sglang_multiturn/build_embed_index.py \
  --corpus_file "$DATA_DIR/corpus.parquet" \
  --model /path/to/Qwen3-Embedding-8B \
  --output "$DATA_DIR/browsecomp_dense_cache.pkl"
```

If the corpus exists but the cache does not, the training script builds the
cache when it launches the dense retrieval service
(`examples/sglang_multiturn/browsecomp_retrieval_server.py`).

---

## 🚀 Training

The scripts target a 4-node × 8×H800 setup with the Megatron backend and SGLang
rollout. Before launching, export the required paths (scripts fail fast if these
are unset):

```bash
export VENV_PATH=/path/to/your/venv               # Python virtualenv
export MODEL_PATH=/path/to/Qwen3-32B              # policy model
export RETRIEVER_MODEL_PATH=/path/to/Qwen3-Embedding-8B
export DATA_DIR=/path/to/browsecomp-plus-processed  # contains *.paper.parquet
export TRAINER_IPS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4

# Weights & Biases: configure online logging, or use WANDB_MODE=offline
export WANDB_API_KEY=...   export WANDB_ENTITY=...
# export WANDB_MODE=offline
```

Launch from the first IP in `TRAINER_IPS`. The four nodes must have passwordless
SSH connectivity and consistent access to the virtual environment, model, and
data paths. Cluster environments may provide the node list through
`PADDLE_TRAINERS` instead.

**⚖️ LLM-judge reward.** The reward function (`verl/utils/reward_score/bc_p_llm_judge.py`)
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

📄 Available scripts in `examples/sglang_multiturn/`:

| Script | Description |
| --- | --- |
| `run_qwen3-32b_bcp_echo-ca_4node.sh` | ECHO (synchronous) |
| `run_qwen3-32b_bcp_echo-ca_fully_async_4node.sh` | ECHO (fully async) |
| `run_qwen3-32b_bcp_grpo_4node.sh` | GRPO baseline (synchronous) |
| `run_qwen3-32b_bcp_grpo_fully_async_4node.sh` | GRPO baseline (fully async) |
| `run_qwen3-30b-a3b_bcp_echo-ca_fully_async_4node.sh` | ECHO on the MoE backbone |
| `run_qwen3-30b-a3b_bcp_grpo_fully_async_4node.sh` | GRPO on the MoE backbone |
| `run_qwen3-32b_bcp_supo_4node.sh` | SUPO rolling-summary baseline (synchronous) |

Run from the project root, e.g.:

```bash
bash examples/sglang_multiturn/run_qwen3-32b_bcp_echo-ca_fully_async_4node.sh
```

### 🔬 Reproducing Ablations

Ablation variants reuse the same core scripts and are toggled through environment
variables (see the top of each script for the full list). Key knobs:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONTEXT_COMPRESSION_METHOD` | `echo_e2e` | Context reconstruction strategy: `echo_e2e` (learned selection), `semantic_selection` (static top-k retrieval), `truncate` (left-truncation), `summary` (SUPO rolling summary) |
| `ECHO_CREDIT_METHOD` | `token` | Credit routing: `token` (provenance-guided ECHO), `traj` (coarser selected-segment credit), `none` (dense signed credit over all response tokens) |
| `ECHO_RECENT_TURNS` | `3` | Number of most-recent turns always kept during reconstruction |
| `WORKING_CONTEXT_LENGTH` | `32768` | Single-segment token threshold that triggers compression |
| `MAX_SUMMARY_ROUNDS` | `5` | Max compression rounds before a rollout is marked overlong |
| `SEMANTIC_SELECTION_FULL_OBSERVATION` | `False` | When using `semantic_selection`, retrieve full observations instead of compact findings |
| `ECHO_CREDIT_PENALTY_RATIO` | `0.0` | Down-weight (vs. 1.0 for credited tokens) applied to non-credited tokens |

Examples reproducing paper ablations (all on top of the ECHO async script):

```bash
# ✅ Full ECHO (paper main): learned selection + provenance-guided token credit
bash examples/sglang_multiturn/run_qwen3-32b_bcp_echo-ca_fully_async_4node.sh

# 🔁 Ablation: static semantic top-k retrieval instead of learned selection
CONTEXT_COMPRESSION_METHOD=semantic_selection \
  bash examples/sglang_multiturn/run_qwen3-32b_bcp_echo-ca_fully_async_4node.sh

# 🔁 Ablation: semantic top-k retrieving full observations (not compact findings)
CONTEXT_COMPRESSION_METHOD=semantic_selection SEMANTIC_SELECTION_FULL_OBSERVATION=True \
  bash examples/sglang_multiturn/run_qwen3-32b_bcp_echo-ca_fully_async_4node.sh

# 🔁 Ablation: w/o traceable credit assignment (dense credit on all tokens)
ECHO_CREDIT_METHOD=none \
  bash examples/sglang_multiturn/run_qwen3-32b_bcp_echo-ca_fully_async_4node.sh

# 📊 SUPO baseline (rolling summarization) — synchronous script
bash examples/sglang_multiturn/run_qwen3-32b_bcp_supo_4node.sh
```

---

## 📊 Results

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
> vs. dense credit (w/o traceable CA, rewards all response tokens) and turn-level
> importance sampling. Dense credit lowers accuracy and stability; turn-level IS
> becomes unstable as turn counts and trajectory volume grow. Traceable credit
> gives the best accuracy and stability.

### 🧩 MoE Backbone

<div align="center">
<img src="assets/moe_experiment.png" width="60%" alt="MoE experiment">
</div>

> **Transfer to the sparse MoE backbone (Qwen3-30B-A3B-Instruct).** ECHO, GRPO,
> and SUPO use the same controlled setup. GRPO reaches 22.9% held-out accuracy;
> SUPO's turn count and trajectory volume grow rapidly while accuracy falls to
> 13.3% by step 50; ECHO remains stable and finishes around 35.0%. The same
> ranking on a sparse backbone shows that ECHO's gains are not specific to the
> dense model.

### ⏱️ Training Efficiency

<div align="center">
<img src="assets/training_efficiency_comparison.png" width="98%" alt="Training efficiency comparison across GRPO, ECHO, and SUPO">
</div>

ECHO incurs early overhead because the policy writes a finding after each
completed tool turn and generates selection actions at reconstruction boundaries.
Later in training, however, its bounded selected view limits turn proliferation;
step, generation, and tool-call time remain lower than SUPO while accuracy is
higher.

### 🌐 Zero-shot Generalization

Without benchmark-specific further training, the BrowseComp-Plus-trained policy
is evaluated in the same agent-loop framework across three out-of-domain
families: Multi-Objective QA (2–16 objectives), Code Generation (CodeGym,
LoCoBench-Agent), and Deep Information Seeking (GAIA, HLE, Frames). These results
measure zero-shot agentic transfer rather than bare-model generation. **Bold** =
best, _underline_ = second-best. CA = credit assignment. Column abbreviations:
**MO-Avg** = Multi-Objective QA average; **LoCo** = LoCoBench-Agent.

**Backbone: Qwen3-32B-Instruct**

| Method | 2-obj | 4-obj | 8-obj | 16-obj | MO-Avg | CodeGym | LoCo | GAIA | HLE | Frames | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GRPO | 38.6 | 39.8 | 35.8 | 29.0 | 35.8 | 32.8 | 67.7 | _25.2_ | 8.8 | 24.8 | 33.6 |
| SUPO | 40.9 | 36.4 | 36.4 | 34.7 | 37.1 | 35.4 | 68.1 | _25.2_ | 9.2 | 26.8 | 34.8 |
| **ECHO** | **47.7** | _45.5_ | **41.5** | **36.1** | **42.7** | **41.4** | **70.4** | **29.1** | **11.4** | **39.1** | **40.2** |
| ECHO w/ Top-K retrieval | **47.7** | 42.0 | _39.2_ | 27.8 | _39.2_ | _40.7_ | 69.3 | **29.1** | _10.6_ | _37.3_ | _38.2_ |
| ECHO w/ Top-K retrieval & w/o turn summary | 40.9 | 44.3 | 35.8 | _35.5_ | 39.1 | 40.3 | 69.5 | 23.3 | 10.0 | 31.3 | 36.8 |
| ECHO w/o traceable CA | _45.5_ | 42.0 | _39.2_ | 22.7 | 37.4 | 38.1 | 68.2 | _25.2_ | 8.8 | 30.8 | 35.6 |
| ECHO w/o Traceable CA & Turn-level IS | _45.5_ | **47.7** | 35.2 | 27.0 | 38.8 | 34.6 | _70.1_ | 23.3 | 9.4 | 32.2 | 36.1 |

**Backbone: Qwen3-30B-A3B-Instruct**

| Method | 2-obj | 4-obj | 8-obj | 16-obj | MO-Avg | CodeGym | LoCo | GAIA | HLE | Frames | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GRPO | _27.3_ | 26.1 | _27.3_ | 16.2 | 24.2 | 20.3 | _65.7_ | _23.3_ | 7.8 | _19.1_ | 25.9 |
| SUPO | 25.0 | _30.7_ | _27.3_ | _18.2_ | _25.3_ | _27.3_ | 65.1 | _23.3_ | _8.0_ | 17.0 | _26.9_ |
| **ECHO** | **34.1** | **36.4** | **30.1** | **18.8** | **29.9** | **29.7** | **66.8** | **24.3** | **9.2** | **25.0** | **30.5** |

ECHO achieves the best dense-backbone average at **40.2%**, compared with 33.6%
for GRPO and 34.8% for SUPO. On the MoE backbone, it reaches **30.5%**, compared
with 26.9% for SUPO. Replacing learned selection with Top-K retrieval lowers the
dense average to 38.2%, while removing traceable credit lowers it to 35.6%.

---

## 🙏 Acknowledgements

ECHO is built on [verl](https://github.com/volcengine/verl) (Volcano Engine
Reinforcement Learning for LLMs). We thank the verl team and community.

---

## 📝 Citation

```bibtex
@article{xie2026echo,
  title  = {ECHO: Prune to Act, Trace to Learn with Selective Turn Memory in Agentic RL},
  author = {Xie, Zijun and Zheng, Binbin and Gong, Enlei and Liu, Jihua and
            You, Yuyang and Liu, Lingfeng and Tang, Jiayao and Zhao, Guanqun and
            Hu, Aoqi and Chen, Zeyu},
  journal = {arXiv preprint arXiv:2606.31650},
  year   = {2026}
}
```
