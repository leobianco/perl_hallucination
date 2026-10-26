<div align="center">

# Hallucination Reduction with PE-RL and Synthetic Data

**Parameter-efficient RLAIF that teaches LLMs to stay faithful to their context,<br/>
trained on synthetic hallucinations and run end-to-end by an automated research orchestrator.**

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers · TRL · PEFT](https://img.shields.io/badge/Transformers%20%C2%B7%20TRL%20%C2%B7%20PEFT-FFD21E?logo=huggingface&logoColor=black)
![DeepSpeed](https://img.shields.io/badge/DeepSpeed-ZeRO--2%20%2F%203-0A66C2)
![vLLM](https://img.shields.io/badge/vLLM-inference-30A2FF)
![Weights & Biases](https://img.shields.io/badge/W%26B-sweeps-FFBE00?logo=weightsandbiases&logoColor=black)
![Gemini](https://img.shields.io/badge/Gemini-autorater-8E75B2?logo=googlegemini&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

[Overview](#overview) •
[Method](#method) •
[Auto-PERL](#auto-perl-the-campaign-orchestrator) •
[Under the hood](#under-the-hood) •
[Quickstart](#quickstart) •
[Documentation](#documentation)

</div>

<br/>

<p align="center">
  <img src="figures/AUTOPERL_PICS/finished.png" width="90%" alt="Auto-PERL Mission Control at the end of a completed campaign"/>
</p>
<p align="center"><sub><b>Auto-PERL Mission Control</b> at the end of a campaign: every stage was swept, its winner selected, published and handed to the next stage, with no human in the loop.</sub></p>

---

## Overview

This project studies the reduction of hallucinations via RLAIF with *synthetic* data. More precisely, we use Parameter-Efficient Reinforcement Learning [PE-RL](https://arxiv.org/abs/2403.10704).

Grounded generation (rewriting, question answering, summarization) requires a model to say only what its context supports. Classic RLHF attacks this with human preference labels and full-model reinforcement learning, and both are expensive. This repository replaces both: the reward model learns from **synthetic hallucinations**, and **every stage trains LoRA adapters**, from supervised fine-tuning to reinforcement learning.

**Highlights**

- **Complete RLAIF stack, LoRA end to end.** Supervised fine-tuning → reward model → RLOO policy optimization, built on 🤗 Transformers, TRL, PEFT, DeepSpeed and vLLM. Gemma 4 by default, any Hugging Face chat model via `--base-model`.
- **Synthetic preference data.** Hallucinations are injected by structured perturbation or written by an LLM, so the reward model can be trained without human labels. Human-labelled ("organic") data is supported for comparison.
- **Auto-PERL orchestrator.** One command runs a whole research campaign: a Bayesian W&B sweep per stage, automatic winner selection, publication to the Hugging Face Hub, hand-off to the next stage, evaluation and a written report.
- **Evaluation built to catch reward hacking.** A calibrated Gemini autorater, a writing-quality rubric judge, ROUGE, BERTScore, perplexity and diversity metrics, scored over a grid of decoding temperatures with confidence intervals.
- **Made for long, unattended runs.** Resumable campaign state that survives preemption and SSH drops, size-aware DeepSpeed ZeRO configuration, and a tmux-native live dashboard.
- **Baselines included.** [SCOPE](BASELINES_SCOPE.md) and [SSFO](BASELINES_SSFO.md), two self-supervised DPO methods, are implemented for comparison.

**Tasks**

| Task | Description |
| :--- | :--- |
| `npov` | Neutral Point of View rewriting (Wikipedia NPOV edits) |
| `bosch` | Factuality / hallucination reduction on Bosch reports |
| `ragtruth-qa` | RAG factuality, question-answering split of [RAGTruth](https://arxiv.org/abs/2401.00396) |
| `ragtruth-summarization` | RAG factuality, summarization split |
| `ragtruth` | All RAGTruth sub-tasks |

---

## Method

| | Stage | What is trained | Training signal |
| :---: | :--- | :--- | :--- |
| 1 | **SFT** | LoRA adapter on the writer policy | Next-token cross-entropy on task demonstrations |
| 2 | **Reward model** | LoRA sequence classifier | Faithful vs. hallucinated responses (organic or synthetic negatives) |
| 3 | **PE-RL** | LoRA policy, initialised from the SFT model | RLOO on the reward model's score, KL-regularised towards the SFT reference |
| 4 | **Evaluation** | n/a | vLLM generations scored by a Gemini judge, a rubric judge and reference metrics |

The policy is rewarded with the reward model's **log-odds** rather than its probability, with a multiplier $\alpha \ge 1$ on negative rewards because hallucinations are rare events (see [NOTES.md](NOTES.md)):

$$
r_\alpha(x, y) =
\begin{cases}
\alpha\,(z_1 - z_0) & \text{if } z_1 - z_0 < 0,\\
z_1 - z_0 & \text{otherwise,}
\end{cases}
$$

where $z_1 - z_0$ is the logit difference between the *faithful* and the *hallucinated* class.

### Training dynamics

Example Weights & Biases curves from each stage of the pipeline (click a figure to enlarge it).

**Stage 1: supervised fine-tuning.** LoRA-SFT on `bosch`: training loss, evaluation loss, mean token accuracy and gradient norm.

<p align="center">
  <img src="figures/BOSCH_SFT_GRAPHS/trainloss.png" width="40%" alt="SFT training loss"/>
  <img src="figures/BOSCH_SFT_GRAPHS/evalloss.png" width="40%" alt="SFT evaluation loss"/>
  <br/>
  <img src="figures/BOSCH_SFT_GRAPHS/acc.png" width="40%" alt="SFT mean token accuracy"/>
  <img src="figures/BOSCH_SFT_GRAPHS/gradnorm.png" width="40%" alt="SFT gradient norm"/>
</p>

**Stage 2: reward model.** LoRA sequence classifier on `npov`, trained on synthetic hallucinations: evaluation ROC-AUC, training loss and gradient norm.

<p align="center">
  <img src="figures/NPOV_RM_SYN_GRAPHS/rm_syn_auc.png" width="32%" alt="Reward model evaluation ROC-AUC"/>
  <img src="figures/NPOV_RM_SYN_GRAPHS/rm_syn_loss.png" width="32%" alt="Reward model training loss"/>
  <img src="figures/NPOV_RM_SYN_GRAPHS/rm_syn_grad.png" width="32%" alt="Reward model gradient norm"/>
</p>

**Stage 3: PE-RL.** RLOO with a LoRA policy on `npov`, rewarded by a reward model trained on organic hallucinations: mean reward and KL divergence to the SFT reference policy.

<p align="center">
  <img src="figures/NPOV_PERL_ORG_GRAPHS/perl_org_reward.png" width="40%" alt="PE-RL mean reward"/>
  <img src="figures/NPOV_PERL_ORG_GRAPHS/perl_org_kl.png" width="40%" alt="PE-RL KL divergence to the reference policy"/>
</p>

---

## Auto-PERL: the campaign orchestrator

Rather than manually managing and chaining individual training scripts, you can run an end-to-end automated scientific campaign (SFT sweep $\to$ RM sweep $\to$ PE-RL sweep $\to$ Evaluation $\to$ Markdown Report) using the Auto-PERL orchestrator. Each stage runs its own hyperparameter sweep, picks its best checkpoint, publishes it and hands it to the next stage.

```mermaid
flowchart LR
    subgraph SFT["① SFT sweep"]
        direction TB
        s1["Bayesian trials<br/>lr · LoRA rank · α · epochs"] --> s2{{"Select winner<br/>lowest eval/loss"}} --> s3[("Publish to<br/>HF Hub")]
    end
    subgraph RM["② Reward-model sweep"]
        direction TB
        r1["Bayesian trials<br/>lr · LoRA rank · α · epochs"] --> r2{{"Select winner<br/>highest eval/roc_auc"}} --> r3[("Publish to<br/>HF Hub")]
    end
    subgraph PERL["③ PE-RL sweep · RLOO"]
        direction TB
        p1["Bayesian trials<br/>lr · KL β · temperature"] --> p2{{"Select winner<br/>converged mean reward"}} --> p3[("Publish to<br/>HF Hub")]
    end
    subgraph EVAL["④ Evaluation"]
        direction TB
        e1["vLLM generation<br/>SFT vs. PE-RL"] --> e2["Gemini autorater · rubric judge<br/>BERTScore · ROUGE · perplexity"] --> e3(["Markdown report<br/>+ web dashboard"])
    end
    SFT -- "policy init + KL reference" --> PERL
    RM -- "reward function" --> PERL
    PERL -- "best policy" --> EVAL
```

### 1. Configure: guided wizard or declarative YAML

`python3 scripts/run_campaign.py wizard` asks for the task, the stages, a trial budget and any upstream checkpoints. It then prints a review panel with a runtime estimate and the **equivalent non-interactive command**, and saves the configuration to `configs/` for reproducibility.

<p align="center">
  <img src="figures/AUTOPERL_PICS/setup3.png" width="85%" alt="Auto-PERL wizard campaign review panel"/>
</p>

<details>
<summary><b>See the wizard step by step</b></summary>
<br/>
<p align="center">
  <img src="figures/AUTOPERL_PICS/welcome.png" width="85%" alt="Wizard: task selection"/>
  <br/><br/>
  <img src="figures/AUTOPERL_PICS/setup.png" width="85%" alt="Wizard: stage selection"/>
  <br/><br/>
  <img src="figures/AUTOPERL_PICS/setup2.png" width="85%" alt="Wizard: trial budget preset"/>
</p>
</details>

| Preset | Trials (SFT / RM / PE-RL) | Eval samples | Intent |
| :--- | :---: | :---: | :--- |
| `smoke` | 1 / 1 / 1 | 20 | Wiring check (minutes) |
| `quick` | 5 / 5 / 3 | 200 | Fast signal (a few hours) |
| `standard` | 30 / 30 / 10 | 1000 | Project default |
| `thorough` | 60 / 60 / 20 | 2000 | Publication run |

### 2. Sweep: every stage is a Bayesian W&B sweep

Each stage registers a W&B sweep from `scripts/sweep_*.yaml`, runs exactly its trial budget, then seals the sweep so that no stray agent keeps spending GPU time.

<p align="center">
  <img src="figures/AUTOPERL_PICS/sweep.png" width="95%" alt="W&B sweep workspace with parameter importance and parallel coordinates"/>
</p>

| Stage | Searched hyperparameters | Ranked on |
| :--- | :--- | :--- |
| SFT | Learning rate, LoRA rank and α, epochs | `eval/loss` (minimize) |
| Reward model | Learning rate, LoRA rank and α, epochs | `eval/roc_auc` (maximize) |
| PE-RL | Learning rate, KL coefficient β, rollout temperature | `train/rewards/reward_fn/mean` (maximize) |

### 3. Select and promote: automatic checkpoint hand-off

When a sweep ends, the orchestrator takes over with no human in the loop:

1. **It ranks the trials with a curve-aware rule.** SFT and RM trials are scored at their best evaluation step (early stopping); PE-RL trials on the converged level of their reward, never on a single lucky batch.
2. **It re-trains the winner** with its exact configuration and checkpointing enabled, **publishes** it to the Hugging Face Hub (best and last checkpoints, plus a `checkpoints.json` manifest), and tags the run `best-<stage>` in W&B.
3. **It wires the winner into the next stage.** The SFT winner becomes PE-RL's initial policy and KL reference (`--sft_model_path`), the RM winner its reward function (`--reward_model_path`), and the PE-RL winner is evaluated against the SFT baseline. No repo ids to copy, no YAML to edit.

### 4. Monitor: Mission Control in tmux

Logs stream normally while a status block stays pinned to the bottom of the pane: the stage DAG, progress bars, best-so-far per stage, published artifacts and a live leaderboard of the running sweep. Single keystrokes control the run: `[a]` advance with the current best, `[p]` pause, `[s]` stop and publish, `[x]` abort, `[q]` detach. A second pane can follow along with `status --watch`.

<p align="center">
  <img src="figures/AUTOPERL_PICS/during.png" width="80%" alt="Auto-PERL Mission Control during the PE-RL sweep"/>
</p>

### 5. Report

When the last stage completes, the dashboard turns into the scorecard shown at the top of this page, and a Markdown report is written to `reports/` with the stage winners, their hyperparameters and evaluation tables with confidence intervals. A static web dashboard of all campaigns can also be published to a Hugging Face Space (see [`src/dashboard/README.md`](src/dashboard/README.md)).

---

## Under the hood

- **Reward = log-odds, not probability.** A confident classifier saturates its softmax near 1, which collapses RLOO's leave-one-out advantages. The policy is instead rewarded with the logit difference $z_1 - z_0$ (the Bradley–Terry implicit reward), with an asymmetric multiplier α for rare negative signals. → [NOTES.md](NOTES.md)
- **Rollouts sized for rare events.** Eight completions per prompt make it likely that a group contains a hallucination at all, since otherwise its advantage is exactly zero. The micro-batch is halved and gradient accumulation doubled, so peak memory and the effective batch stay unchanged. → [NOTES.md](NOTES.md)
- **Sweeps that cannot game their own objective.** α is pinned rather than searched, because it rescales the very metric the Bayesian optimizer maximizes. Greedy decoding is excluded from the rollout-temperature grid, because identical rollouts carry no advantage. → [`scripts/sweep_perl.yaml`](scripts/sweep_perl.yaml)
- **Honest model selection.** Curve-aware ranking per stage, best *and* last checkpoints published with a manifest, and headline metrics computed on a held-out test set that took no part in selection. → [orchestrator README](src/orchestrator/README.md)
- **Evaluation that catches reward hacking.** A policy that copies its context is perfectly "faithful", so a rubric judge grades fluency, non-repetition and non-extractiveness next to the calibrated hallucination judge. Every policy is scored over a decoding-temperature grid, with 95% confidence intervals. → [METRICS.md](METRICS.md)
- **Scales with the model.** The launcher is derived from the parameter count (gradient checkpointing from 6B, ZeRO-3 for PE-RL from 12B, CPU offload from 30B), and LoRA was made to compose with gradient checkpointing. → [Base models, GPUs and DeepSpeed](#base-models-gpus-and-deepspeed)
- **Survives the real world.** Atomic, resumable campaign state; a resumed sweep runs only the trials it still owes; SSH drops and VM preemption are survivable; and the VM can power itself off when a campaign ends. → [orchestrator README](src/orchestrator/README.md)
- **Zero lock-in.** The orchestrator lives in `src/orchestrator/` and drives the same entry points as the manual scripts. Remove it and everything else still works.

---

## Quickstart

Install the requirements (see [Installation](#installation)) and set the Gemini API key used by the synthetic-data generator and the autorater:

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"
```

Then run a campaign:

```bash
# 1. Pre-flight environment check (CUDA, W&B, HF tokens, disk space)
python3 scripts/run_campaign.py doctor --task npov

# 2. Guided interactive setup wizard
python3 scripts/run_campaign.py wizard

# 3. Launch with a budget preset (smoke, quick, standard, thorough)
python3 scripts/run_campaign.py run --task npov --preset standard

# 4. Monitor from a second tmux pane
python3 scripts/run_campaign.py status --task npov --watch

# 5. Seamlessly resume after VM preemption or disconnect
python3 scripts/run_campaign.py resume --task npov
```

### Resuming vs. restarting (read this if your config pins `name:`)

A campaign's `name` determines its state file
(`./checkpoints/<task>/<name>_state.json`). If your YAML pins `name:`, every
`run` targets the same campaign, and **`run` will not guess** whether you want
to continue it or start it over — guessing wrong either re-burns hours of GPU
time or throws away a finished campaign. Say which:

```bash
# Continue the existing campaign (same as `resume`).
python3 scripts/run_campaign.py run --task bosch --config configs/campaign_bosch.yaml --resume

# Start over. The old state file is archived, never deleted, under
# ./checkpoints/bosch/archive/.
python3 scripts/run_campaign.py run --task bosch --config configs/campaign_bosch.yaml --fresh
```

Without a flag, an interactive terminal prompts `[r]esume / [f]resh / [a]bort`;
a non-interactive one (piped output, `nohup`, CI) refuses with exit code `2`.
`--yes` does *not* answer this question — it only skips the launch
confirmation. Omit `name:` from your config and every launch is a new,
timestamped campaign, so none of this applies.

For complete documentation on the orchestrator, interactive dashboard hotkeys, declarative YAML configs, and remote VM/tmux setups, see [**`src/orchestrator/README.md`**](src/orchestrator/README.md).

---

## Manual Script Execution

If running scripts individually, here is a brief description about the usage of each script, roughly in the order that they should be executed.
Substitute `TASK` by `npov`, `bosch`, `ragtruth-qa`, or `ragtruth-summarization` (or `ragtruth`, which defaults to `ragtruth-qa`) to select the correct dataset. 
Every `.sh` script in the `/scripts` folder runs the corresponding Python entrypoint located under `src/` (for example `src/writer_sft.py`).
**Run the scripts from the project folder (not from within the `scripts` folder) so paths resolve correctly.**

Please, set the Gemini API key in order to use the LLM-generated synthetic data, as well as the autorater:
```
export GEMINI_API_KEY="your-key-here"
```

From the project folder, run
* `./scripts/data_processing.sh (TASK)`: processes the raw dataset and saves resulting datasets to the HuggingFace Hub. This includes creating prompts to be passed to the LLMs later, putting data in the right format, creating synthetic hallucinations, creating data for evaluation.
* `writer_sft.sh (TASK)`: LoRA-SFT the model indicated as a writer.
* `reward_model.sh (TASK)`: trains the reward model. Set one of the `ORGANIC`, `SYN_HALL_LLM`, or `SYN_HALL_STRUCT` parameters to true to choose what type of hallucinations to train on.
* `perl.sh (TASK)`: runs the PERL loop with the indicated reward model and SFT model as reference. Since these jobs are longer, there is a `SHUTDOWN` parameter that allows the VM to be automatically shut down once the job finishes. Set `CONTINUAL_EVAL=True` to also score the policy with the final evaluation's autorater judges at step 0 and every `EVAL_STEPS` steps (logged to the run's W&B page as `eval/*`); it is off by default because each evaluation step costs a vLLM generation pass plus judge calls. The orchestrator always turns it on for its PE-RL runs.
* `evaluator.sh (TASK) (MODE)`:
    * When `(MODE)` is set to `generate`, the model indicated by the `RUN_IDENTIFIER` parameter is used as a writer and generates completions for the prompts in `DATASET_PROMPTS`. This generation is done using vLLM.
    * When `(MODE)` is set to `score`, the `EVALUATOR_MODEL` scores the generations of the previous step, and a rate of hallucination is calculated using `THRESHOLD`. The evaluator uses `EVALUATOR_NUM_FEWSHOT` examples taken from `DATASET_LABELS` to help it classify the samples.
    * When `(MODE)` is set to `autoratereval`, the quality of the evaluator itself is evaluated. We ask for it to score the samples in `DATASET_LABELS`, and we return the best threshold along with the associated metrics.
    * When `(MODE)` is set to `reward_hacking_autorater_eval`, the reward-hacking rubric judge (configured as in `score`) grades healthy references from `{task}_rm_organic:test` and `{task}_sft:test`. Every flag is a false positive, so the report gives the judge's false-positive rate at `REWARD_HACKING_THRESHOLD` and the largest thresholds that keep it within 1%, 5% and 10%. Report only. A readable markdown report is written to `reports/reward_hacking_calibration/`; see [METRICS.md](METRICS.md#4-reward-hacking-rubric-autorater-writing-quality).

Notice that you can chain commands, *e.g.* `./scripts/evaluator.sh npov generate ; ./scripts/evaluator.sh npov score`.

### Evaluating the Last Checkpoint vs. Best Checkpoint

By default, training scripts (`perl.sh`, `writer_sft.sh`, `reward_model.sh`, `scope_dpo.sh`, `ssfo_dpo.sh`) save checkpoints as follows:
1. **Hugging Face Hub (`<user>/<run_name>`)**: Stores only the **best model** (the checkpoint that achieved the highest validation reward / metric during training).
2. **Local Disk (`./checkpoints/<task>/<method>/<user>/<run_name>/`)**: Stores:
   - The **best model** at the root of the directory and in `checkpoint-<BEST_STEP>/`.
   - The **final step checkpoint** (adapters from the very last training step) in `checkpoint-<LAST_STEP>/`.
   - *(Note for PE-RL runs)*: Each checkpoint directory also contains a `ref/` subfolder holding TRL's frozen reference policy LoRA adapter used for KL computation; `src.evaluator` automatically ignores `ref/` and loads the trained policy adapter.

To evaluate the **best model** (default):
- Set `RUN_IDENTIFIER="<user>/<run_name>"` (loads from Hugging Face Hub) or `RUN_IDENTIFIER="./checkpoints/<task>/perl/<user>/<run_name>"` (loads from local root).

To evaluate the **last checkpoint** (`checkpoint-<LAST_STEP>`) instead of the best:
- Point `RUN_IDENTIFIER` to the specific local `checkpoint-<LAST_STEP>` directory.
- Explicitly set `DATASET_WITH_COMPLETIONS` to a distinct short Hugging Face dataset name (since auto-generated dataset names from long local filesystem paths are truncated at 96 characters on Hugging Face Hub and could collide with evaluations of other checkpoints from the same run):

```bash
RUN_IDENTIFIER="./checkpoints/npov/perl/leobianco/<run_name>/checkpoint-220" \
DATASET_WITH_COMPLETIONS="leobianco/npov_eval_last_ckpt_220" \
./scripts/evaluator.sh npov generate

RUN_IDENTIFIER="./checkpoints/npov/perl/leobianco/<run_name>/checkpoint-220" \
DATASET_WITH_COMPLETIONS="leobianco/npov_eval_last_ckpt_220" \
./scripts/evaluator.sh npov score
```

---

## Base models, GPUs and DeepSpeed

The original experiments ran on 8 x L4 GPUs. For an efficient use of GPU
memory we employ ZeRO
[(link to paper)](https://arxiv.org/abs/1910.02054) through Hugging Face's
Accelerate integration of Microsoft's DeepSpeed. The configurations live in
`scripts/` (run `accelerate config` to set up your own environment, see
[Accelerate's documentation](https://huggingface.co/docs/transformers/en/deepspeed)).
All three ship with `num_processes: 2`, matching the 2 x A100-80GB box the
campaigns currently run on.

| Profile | File | Used for |
|---|---|---|
| `zero2` | `scripts/deepspeed_config.yaml` | everything below 12B, and SFT/RM at any size below 30B |
| `zero3` | `scripts/deepspeed_config_zero3.yaml` | the PE-RL stage from 12B up |
| `zero3_offload` | `scripts/deepspeed_config_zero3_offload.yaml` | from 30B up, or forced by hand |

### Running a campaign on another base model

```bash
python3 scripts/run_campaign.py run --task npov \
    --base-model "mistralai/Mistral-7B-Instruct-v0.3"
```

`--base-model` reaches the SFT, RM and PE-RL sweeps, the winners' retraining
and the evaluation, overriding the `--model_repo_id` pinned in every
`scripts/sweep_*.yaml`. The wizard (`run_campaign.py wizard`) asks for it too
and shows the resulting launcher in its review block. Nothing else needs
editing; `src/model_compat.py` handles the family differences (chat template,
turn terminator, missing pad token).

**What a large model changes automatically.** `src/orchestrator/accel.py`
reads the parameter count off the checkpoint's name. There are two separate
thresholds, because trading compute for activation memory starts paying long
before it is worth changing the distributed execution model.

At or above **6B** (`CHECKPOINT_THRESHOLD_B`):

* every training stage gets `--gradient_checkpointing True`;
* the PE-RL micro-batch is halved and gradient accumulation doubled, so the
  effective batch, and therefore the optimisation problem the sweep searches,
  is unchanged.

At or above **12B** (`ZERO3_THRESHOLD_B`):

* the PE-RL stage moves to ZeRO Stage 3, because it is the only stage holding
  two models at once - the policy and the reward model.

A 7B model therefore gets checkpointing but stays on ZeRO-2. On a 2x80 GB box
a 7B policy plus a 7B reward model is ~29 GB of bf16 weights, landing around
35-38 GB of 80 with LoRA - Stage 3 would spend 20-30% of PE-RL throughput to
reclaim memory that is not scarce, and `reward_fn` all-gathers the reward
model in full on every scored batch regardless, so part of the saving is
illusory.

Below 6B nothing changes at all, so the existing 4B results stay comparable.
Override the profile with `--deepspeed-profile {auto,zero2,zero3,zero3_offload}`
or a path; override the batch geometry with
`PERL_PER_DEVICE_TRAIN_BATCH_SIZE` / `PERL_GRADIENT_ACCUMULATION_STEPS` (an
explicit environment value always beats the automatic one). Hand-run scripts
do not consult the orchestrator, so tell them directly:

```bash
DEEPSPEED_CONFIG=scripts/deepspeed_config_zero3.yaml \
MODEL_REPO_ID=mistralai/Mistral-7B-Instruct-v0.3 ./scripts/perl.sh npov
```

> **Cost of Stage 3.** Parameters are all-gathered on every forward, and the
> reward model is gathered once per scored batch (TRL's
> `ds3_gather_for_generation` only covers the policy, so `reward_fn` in
> `src/pipelines.py` does it explicitly). Budget 20-30% more wall clock for
> the PE-RL stage than the estimate the wizard prints.

---

## Installation

The experiments were run using Python 3.11.

Please also install the necessary Python header files by installing the `python-dev` package.

Please also install the CUDA drivers following the [Google Cloud CUDA Driver Installation Guide](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu).

Install Python package requirements by first compiling `requirements.in` using `pip-tools`:
```
pip-compile requirements.in -o requirements.txt
```
then install them:
```
pip install -r requirements.txt
```

*Note:* `pip-compile` can be slow. For this reason we recommend using [uv](https://docs.astral.sh/uv/), a Python package manager written in Rust that is much faster. After installing it, run
```
uv pip compile requirements.in > requirements.txt
pip install -r requirements.txt
```

---

## Repository layout

```
.
├── scripts/                # Shell entry points, W&B sweep YAMLs, DeepSpeed configs, run_campaign.py
├── src/
│   ├── writer_sft.py       # Stage 1: LoRA-SFT of the writer
│   ├── reward_model.py     # Stage 2: reward model
│   ├── perl.py             # Stage 3: PE-RL (RLOO)
│   ├── evaluator.py        # Stage 4: generation, scoring and autorater calibration
│   ├── pipelines.py        # Training and evaluation pipelines shared by all stages
│   ├── metrics.py          # Evaluation metrics
│   ├── task_processors/    # NPOV, Bosch and RAGTruth data processing
│   ├── orchestrator/       # Auto-PERL: engine, stages, sweep control, CLI and live dashboard
│   └── dashboard/          # Static campaign dashboard (Hugging Face Space publisher)
├── figures/                # Figures used in this README
└── *.md                    # Design notes, metrics and baseline documentation
```

## Documentation

| Document | Contents |
| :--- | :--- |
| [`src/orchestrator/README.md`](src/orchestrator/README.md) | Auto-PERL user guide: presets, selection rules, hotkeys, resuming, tmux workflow |
| [`ORCHESTRATOR_IMPLEMENTATION.md`](ORCHESTRATOR_IMPLEMENTATION.md) | Orchestrator architecture and design decisions |
| [`METRICS.md`](METRICS.md) | Evaluation metrics, judges and confidence intervals |
| [`NOTES.md`](NOTES.md) | Reward shaping and RLOO design notes |
| [`BASELINES_SCOPE.md`](BASELINES_SCOPE.md) · [`BASELINES_SSFO.md`](BASELINES_SSFO.md) | SCOPE and SSFO baselines |
| [`src/dashboard/README.md`](src/dashboard/README.md) | Static campaign dashboard |

## References

- **PE-RL**: *Parameter Efficient Reinforcement Learning from Human Feedback*, 2024. [arXiv:2403.10704](https://arxiv.org/abs/2403.10704)
- **RLOO**: *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*, 2024. [arXiv:2402.14740](https://arxiv.org/abs/2402.14740)
- **LoRA**: *LoRA: Low-Rank Adaptation of Large Language Models*, 2021. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **ZeRO**: *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, 2019. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
- **RAGTruth**: *RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models*, 2024. [arXiv:2401.00396](https://arxiv.org/abs/2401.00396)
- **SCOPE**: *Self-supervised Framework for Improving Faithfulness in Conditional Text Generation*, ICLR 2025. [arXiv:2502.13674](https://arxiv.org/abs/2502.13674)
- **SSFO**: *Self-Supervised Faithfulness Optimization for Retrieval-Augmented Generation*, 2025. [arXiv:2508.17225](https://arxiv.org/abs/2508.17225)

## License

Released under the [Apache 2.0 License](LICENSE).

<p align="center"><sub>Built by <a href="https://github.com/leobianco">Leonardo Martins Bianco</a></sub></p>