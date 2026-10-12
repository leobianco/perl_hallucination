# Baselines: SCOPE Framework

This document describes the baseline methods implemented in this repository, primarily focusing on **SCOPE** (*Self-supervised Framework for Improving Faithfulness in Conditional Text Generation*, ICLR 2025: [arXiv:2502.13674](https://arxiv.org/abs/2502.13674)).

---

## Table of Contents
1. [Overview & Motivation](#overview--motivation)
2. [Methodology & Architecture](#methodology--architecture)
   - [Stage 1: Initial SFT Training ($\mathcal{D}_1$)](#stage-1-initial-sft-training-mathcald_1)
   - [Stage 2: Synthetic Preference Data Generation ($\mathcal{D}_2$)](#stage-2-synthetic-preference-data-generation-mathcald_2)
   - [Stage 3: Direct Preference Optimization (DPO)](#stage-3-direct-preference-optimization-dpo)
3. [Codebase Architecture & File Mapping](#codebase-architecture--file-mapping)
4. [Usage & Execution Guide](#usage--execution-guide)
   - [End-to-End Execution](#end-to-end-execution)
   - [Step-by-Step Execution](#step-by-step-execution)
5. [Hyperparameter Reference](#hyperparameter-reference)
6. [Comparison: SCOPE vs. PE-RL](#comparison-scope-vs-pe-rl)

---

## Overview & Motivation

In conditional text generation tasks (such as neutral point-of-view editing, technical Q&A, and grounded dialogue), supervised fine-tuning (SFT) can still produce unfaithful or hallucinated outputs when context constraints are subtle.

**SCOPE** addresses this by generating synthetic preference pairs entirely without human annotations or external LLM judges, then optimizing the policy using Direct Preference Optimization (DPO):

```
                                  SCOPE Pipeline Overview
  +-------------------------------------------------------------------------------+
  |  Full SFT Dataset (D)                                                         |
  |  |                                                                            |
  |  +---> D1 (50% Split) ---> Initial SFT Model (p_θ0)                           |
  |                                   |                                           |
  |  +---> D2 (50% Split)             v                                           |
  |         + Base Model (p_LM) ---> Noisy Decoding (Algorithm 1)                 |
  |                                   |                                           |
  |                                   v                                           |
  |                           Synthetic Preference Dataset                        |
  |                           (prompt, chosen=y+, rejected=y-)                    |
  |                                   |                                           |
  |                                   v                                           |
  |                           DPO Preference Tuning                               |
  |                           (TRL DPOTrainer on p_θ0)                            |
  +-------------------------------------------------------------------------------+
```

---

## Methodology & Architecture

### Data Partitioning
Given an SFT dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, SCOPE partitions the training data into two equal halves:
- **$\mathcal{D}_1$**: Used for training the initial supervised fine-tuned checkpoint $p_{\theta_0}$.
- **$\mathcal{D}_2$**: Used for generating synthetic dispreferred completions and running DPO preference optimization.

### Stage 1: Initial SFT Training ($\mathcal{D}_1$)
A pre-trained base model $p_{\text{LM}}$ (e.g., `google/gemma-4-E4B`) is fine-tuned on $\mathcal{D}_1$ using standard cross-entropy loss:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}_1} \left[ \sum_{t=1}^{|y|} \log p_\theta(y_t \mid y_{<t}, x) \right]$$

This yields the initial SFT policy $p_{\theta_0}$. In this repository, Stage 1 is executed using `SFTPipeline` with `--sft_data_fraction 0.5`.

### Stage 2: Synthetic Preference Data Generation ($\mathcal{D}_2$)
To construct dispreferred samples $y^-$, SCOPE applies **Noisy Decoding** (Algorithm 1 in Section 3 of the paper):

1. For each prompt $x \in \mathcal{D}_2$ with ground truth target $y^+$, the token-by-token generation proceeds autoregressively.
2. At each token step $t$:
   - Sample a binary mixture variable $\alpha_t \sim \text{Bernoulli}(\alpha)$, where $\alpha \in [0, 1]$ represents the noise level (default: $\alpha = 0.5$).
   - If $\alpha_t = 0$: sample next token from the context-conditioned SFT model $p_{\theta_0}(\cdot \mid y_{<t}^-, x)$.
   - If $\alpha_t = 1$: sample next token from the unconditional base pre-trained model $p_{\text{LM}}(\cdot \mid y_{<t}^-)$.
3. The resulting sequence $y^-$ maintains natural language fluency and style (from $p_{\text{LM}}$) while dropping context faithfulness and groundedness (introduced by $p_{\theta_0}$).
4. The preference triplet is constructed as $(x, y^+, y^-)$, where:
   - $y^+$ (**chosen**): Ground-truth target completion from $\mathcal{D}_2$.
   - $y^-$ (**rejected**): Unfaithful synthetic completion from noisy decoding.

The implementation in [`ScopeDataGenerationPipeline`](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/src/pipelines.py) supports three mixing strategies:
* `bernoulli` (Algorithm 1 default): $\alpha_t \sim \text{Bernoulli}(\alpha)$.
* `prob_mix`: $p(y_t) = (1 - \alpha) p_{\theta_0}(y_t) + \alpha p_{\text{LM}}(y_t)$.
* `logit_mix`: $\mathbf{z}(y_t) = (1 - \alpha) \mathbf{z}_{\theta_0}(y_t) + \alpha \mathbf{z}_{\text{LM}}(y_t)$.

### Stage 3: Direct Preference Optimization (DPO)
The initial SFT checkpoint $p_{\theta_0}$ is aligned on the preference dataset using DPO:

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y^+, y^-) \sim \mathcal{D}_2} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y^+ \mid x)}{\pi_{\text{ref}}(y^+ \mid x)} - \beta \log \frac{\pi_\theta(y^- \mid x)}{\pi_{\text{ref}}(y^- \mid x)} \right) \right]$$

where:
- The reference model is the frozen SFT model: $\pi_{\text{ref}} = p_{\theta_0}$.
- The policy $\pi_\theta$ is initialized from $p_{\theta_0}$.

#### Memory-Efficient LoRA Adapter Management
To avoid maintaining two full model instances in GPU memory during DPO:
1. The base model weights $W_{\text{base}}$ are loaded, and the Stage 1 SFT LoRA adapter $\Delta W_{\text{SFT}}$ is merged via `merge_and_unload()`:
   $$W_{\text{merged}} = W_{\text{base}} + \Delta W_{\text{SFT}} = p_{\theta_0}$$
2. A fresh LoRA adapter $\Delta W_{\text{DPO}}$ is attached to $W_{\text{merged}}$.
3. Setting `ref_model=None` in TRL's `DPOTrainer` disables the DPO adapter during reference log-probability evaluation, which evaluates $W_{\text{merged}} = p_{\theta_0} = \pi_{\text{ref}}$ with **zero extra VRAM overhead**.

---

## Codebase Architecture & File Mapping

| Component | File Path | Purpose |
| :--- | :--- | :--- |
| **Arguments** | [src/utils.py](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/src/utils.py) | Defines `ScriptArguments.sft_data_fraction` and `ScopeDataGenArguments`. |
| **Pipelines** | [src/pipelines.py](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/src/pipelines.py) | Contains `SFTPipeline` (split slicing), `ScopeDataGenerationPipeline` (Algorithm 1), and `DPOPipeline` (TRL DPOTrainer). |
| **Data Generation Entry** | [src/scope_data_generation.py](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/src/scope_data_generation.py) | CLI entry point for SCOPE synthetic preference dataset generation. |
| **DPO Entry** | [src/dpo.py](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/src/dpo.py) | CLI entry point for DPO preference training. |
| **Task Processor Helper** | [src/task_processors/base_task_processor.py](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/src/task_processors/base_task_processor.py) | `_make_scope_sft_splits` for deterministic $\mathcal{D}_1 / \mathcal{D}_2$ splitting. |
| **Data Gen Script** | [scripts/scope_data_generation.sh](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/scripts/scope_data_generation.sh) | Standalone script for synthetic dispreferred generation. |
| **DPO Script** | [scripts/scope_dpo.sh](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/scripts/scope_dpo.sh) | DeepSpeed/Accelerate launch script for DPO preference training. |
| **Unified Baseline Script** | [scripts/scope_baseline.sh](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/scripts/scope_baseline.sh) | Multi-stage runner executing SFT, Data Gen, and DPO end-to-end. |
| **Unit Tests** | [src/test_scope.py](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/src/test_scope.py) | Unit tests verifying splitting, noisy decoding, and DPO structures. |

---

## Usage & Execution Guide

All commands must be executed from the **root project directory**. Substitute `TASK` with `npov`, `bosch`, or `ragtruth`.

### End-to-End Execution
Run the complete 3-stage SCOPE baseline with a single command:
```bash
./scripts/scope_baseline.sh (TASK) all
```

### Step-by-Step Execution

#### 1. Stage 1: Train Initial SFT Model on Half Data ($\mathcal{D}_1$)
```bash
./scripts/scope_baseline.sh (TASK) sft
```
*Or using the underlying script directly:*
```bash
accelerate launch --config_file=scripts/deepspeed_config.yaml src/writer_sft.py \
  --task_name (TASK) \
  --dataset_repo_id (USER)/(TASK)_sft \
  --model_repo_id "google/gemma-4-E4B" \
  --sft_data_fraction 0.5 \
  --do_train True \
  --output_dir "./checkpoints/(TASK)/writer_sft_half"
```

#### 2. Stage 2: Generate Synthetic Preference Data ($\mathcal{D}_2$)
```bash
./scripts/scope_baseline.sh (TASK) generate_data
```
*Or using `scripts/scope_data_generation.sh`:*
```bash
./scripts/scope_data_generation.sh (TASK)
```

#### 3. Stage 3: Direct Preference Optimization (DPO)
```bash
./scripts/scope_baseline.sh (TASK) dpo
```
*Or using `scripts/scope_dpo.sh`:*
```bash
./scripts/scope_dpo.sh (TASK)
```

#### 4. Evaluate Checkpoints
Once trained, the SCOPE DPO checkpoint can be evaluated using the standard repository evaluation flow:
```bash
# Generate completions with the DPO model
./scripts/evaluator.sh (TASK) generate

# Score the generated completions against the autorater
./scripts/evaluator.sh (TASK) score
```

---

## Hyperparameter Reference

| Hyperparameter | Scope Parameter | Default Value | Description |
| :--- | :--- | :--- | :--- |
| $\alpha$ | `--alpha` | `0.5` | Bernoulli mixing parameter (noise rate for unfaithfulness). |
| Split Ratio | `--split_ratio` | `0.5` | Fraction of SFT data used for $\mathcal{D}_1$ vs. $\mathcal{D}_2$. |
| Sampling Mode | `--sampling_mode` | `bernoulli` | `bernoulli` (Algorithm 1), `prob_mix`, or `logit_mix`. |
| Gen Temperature | `--temperature` | `0.7` | Temperature during noisy decoding. |
| Gen Top-$p$ | `--top_p` | `0.9` | Nucleus sampling threshold. |
| Gen Top-$k$ | `--top_k` | `50` | Top-$k$ token filtering. |
| Max New Tokens | `--max_new_tokens` | `150` | Maximum generated tokens per completion. |
| DPO $\beta$ | `--beta` | `0.1` | Temperature parameter in the DPO objective. |
| DPO Learning Rate | `--learning_rate` | `5e-6` | Peak learning rate for policy optimization. |
| DPO Epochs | `--num_train_epochs` | `1` | Number of DPO training epochs over $\mathcal{D}_2$. |
| LoRA Rank ($r$) | `--lora_r` | `8` | Rank for low-rank adapter matrices. |
| LoRA Alpha | `--lora_alpha` | `16` | Scaling factor for LoRA updates. |

---

## Comparison: SCOPE vs. PE-RL

| Dimension | PE-RL (Our Approach) | SCOPE (Baseline) |
| :--- | :--- | :--- |
| **Optimization Method** | Reinforcement Learning via RLOO / PPO | Direct Preference Optimization (DPO) |
| **Reward Signal** | Fine-grained token/sequence classification Reward Model | Implicit reward from pairwise preference likelihood ratios |
| **Negative Data Source** | Synthetic hallucinations (structured / LLM perturbed) | Autoregressive noisy decoding mixing SFT and Base logits |
| **Data Split Requirement** | Full dataset used for RM + RL loops | Dataset divided into halves ($\mathcal{D}_1$ for SFT, $\mathcal{D}_2$ for DPO) |
| **Inference in Training** | Online sampling during RL rollouts | Offline generation prior to DPO training |
