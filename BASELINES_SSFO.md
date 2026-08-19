# Baselines: Self-Supervised Preference Optimization

This document describes the baseline methods implemented in this repository for reducing hallucinations and improving faithfulness in conditional text generation:

1. **SCOPE** (*Self-supervised Framework for Improving Faithfulness in Conditional Text Generation*, ICLR 2025: [arXiv:2502.13674](https://arxiv.org/abs/2502.13674))
2. **SSFO** (*Self-Supervised Faithfulness Optimization for Retrieval-Augmented Generation*, 2025: [arXiv:2508.17225](https://arxiv.org/abs/2508.17225))

---

## Table of Contents
1. [Overview & Motivation](#overview--motivation)
2. [Methodology: SCOPE](#methodology-scope)
   - [Stage 1: Initial SFT Training ($\mathcal{D}_1$)](#scope-stage-1-initial-sft-training-mathcald_1)
   - [Stage 2: Noisy Decoding Preference Generation ($\mathcal{D}_2$)](#scope-stage-2-noisy-decoding-preference-generation-mathcald_2)
   - [Stage 3: Direct Preference Optimization (DPO)](#scope-stage-3-direct-preference-optimization-dpo)
3. [Methodology: SSFO](#methodology-ssfo)
   - [Context Contrasting Principle](#ssfo-context-contrasting-principle)
   - [Stage 1: SFT Model Training](#ssfo-stage-1-sft-model-training)
   - [Stage 2: Paired Generation With vs. Without Context](#ssfo-stage-2-paired-generation-with-vs-without-context)
   - [Stage 3: Direct Preference Optimization (DPO)](#ssfo-stage-3-direct-preference-optimization-dpo)
4. [Codebase Architecture & File Mapping](#codebase-architecture--file-mapping)
5. [Usage & Execution Guide](#usage--execution-guide)
   - [Running SCOPE](#running-scope)
   - [Running SSFO](#running-ssfo)
6. [Hyperparameter Reference](#hyperparameter-reference)
7. [Comparison: PE-RL vs. SCOPE vs. SSFO](#comparison-pe-rl-vs-scope-vs-ssfo)

---

## Overview & Motivation

Both **SCOPE** and **SSFO** are self-supervised alignment baselines. They address faithfulness hallucinations by automatically creating pairwise preference data $(x, y^+, y^-)$ without requiring human annotations or external LLM judges, then aligning the policy model via Direct Preference Optimization (DPO).

```
                                      Self-Supervised Baselines
  ===================================================================================================

  [ SCOPE Pipeline ]
  Full SFT Data (D)
   ├──> D1 (50% Split) ---> Initial SFT Model (p_θ0)
   └──> D2 (50% Split)           │
          + Base Model (p_LM) ───┼──> Noisy Decoding (Algorithm 1)
                                 │      │
                                 │      └──> (prompt, chosen=y_gt, rejected=y_noisy)
                                 │              │
                                 └──────────────┴──> DPO Preference Tuning (on p_θ0)

  ---------------------------------------------------------------------------------------------------

  [ SSFO Pipeline ]
  SFT Data (D) ───────────> SFT Model (p_θ0)
                                 │
  Input (Query x, Context c) ────┼──> Conditioned Generation (x, c) ──> Chosen y+
                                 │
  Input (Query x only) ──────────┼──> Context-Free Generation (x)   ──> Rejected y-
                                 │      │
                                 │      └──> (prompt=(x,c), chosen=y+, rejected=y-)
                                 │              │
                                 └──────────────┴──> DPO Preference Tuning (on p_θ0)
  ===================================================================================================
```

---

## Methodology: SCOPE

### SCOPE Stage 1: Initial SFT Training ($\mathcal{D}_1$)
The dataset $\mathcal{D}$ is split into two halves: $\mathcal{D}_1$ and $\mathcal{D}_2$. A pre-trained base model $p_{\text{LM}}$ is fine-tuned on $\mathcal{D}_1$ to produce the initial SFT checkpoint $p_{\theta_0}$:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}_1} \left[ \sum_{t=1}^{|y|} \log p_\theta(y_t \mid y_{<t}, x) \right]$$

### SCOPE Stage 2: Noisy Decoding Preference Generation ($\mathcal{D}_2$)
To construct dispreferred completions $y^-$, SCOPE samples tokens by mixing the context-conditioned SFT model $p_{\theta_0}$ with the unconditional base model $p_{\text{LM}}$:

- At each token step $t$, sample $\alpha_t \sim \text{Bernoulli}(\alpha)$ (Algorithm 1, default $\alpha = 0.5$).
- If $\alpha_t = 0$: sample next token from $p_{\theta_0}(\cdot \mid y_{<t}^-, x)$ (maintains context groundedness).
- If $\alpha_t = 1$: sample next token from $p_{\text{LM}}(\cdot \mid y_{<t}^-)$ (preserves language quality while removing groundedness).
- Triplet: $(x, y^+, y^-)$ where $y^+$ is the ground-truth completion and $y^-$ is the noisy generated completion.

### SCOPE Stage 3: Direct Preference Optimization (DPO)
The policy is initialized from $p_{\theta_0}$ and optimized with DPO using $\pi_{\text{ref}} = p_{\theta_0}$.

---

## Methodology: SSFO

### SSFO Context Contrasting Principle
While SCOPE mixes logits per token with a base model, **SSFO** (*Self-Supervised Faithfulness Optimization*, arXiv:2508.17225) contrasts context-conditioned and context-free generations.

Given an input query $x$ and retrieved context $c$:
1. **Preferred completion ($y^+$)**: The fine-tuned SFT model generates conditioned on both query and retrieved context $(x, c)$:
   $$y^+ \sim p_{\theta_0}(\cdot \mid x, c)$$
   *(Optionally, the ground-truth reference completion can be used).*
2. **Dispreferred completion ($y^-$)**: The base instruction-tuned model $p_{\text{LM}}$ (or SFT model) generates conditioned *only* on the query $x$, withholding context $c$:
   $$y^- \sim p_{\text{LM}}(\cdot \mid x)$$
   Because the SFT adapter is heavily overfitted to context-conditioned templates (causing out-of-distribution mode collapse without context), generating $y^-$ with the base instruction-tuned model produces fluent, sensical sentences that rely purely on parametric memory (true contextual hallucinations relative to $c$).
3. **Preference Pair**:
   $$\text{Prompt: } (x, c) \quad\mid\quad \text{Chosen: } y^+ \quad\mid\quad \text{Rejected: } y^-$$

### SSFO Stage 1: SFT Model Training
Train or load the SFT checkpoint $p_{\theta_0}$ on the full task SFT training data.

### SSFO Stage 2: Paired Generation With vs. Without Context
[`SSFODataGenerationPipeline`](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/src/pipelines.py) extracts:
* **NPOV**:
  * Context-conditioned: User query + perspective arguments + prompt template.
  * Context-free: User query + prompt template (without perspective arguments).
* **Bosch**:
  * Context-conditioned: User question + car manual excerpt Context + prompt template.
  * Context-free: User question + prompt template (manual excerpt omitted).
* **RAGTruth / General**:
  * Context-conditioned: Context + Question + Answer prompt.
  * Context-free: Question + Answer prompt.

### SSFO Stage 3: Direct Preference Optimization (DPO)
The SFT model is aligned on the generated pairs using standard DPO loss:

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{( (x,c), y^+, y^- )} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y^+ \mid x, c)}{\pi_{\text{ref}}(y^+ \mid x, c)} - \beta \log \frac{\pi_\theta(y^- \mid x, c)}{\pi_{\text{ref}}(y^- \mid x, c)} \right) \right]$$

---

## Codebase Architecture & File Mapping

| Component | File Path | Description |
| :--- | :--- | :--- |
| **Dataclasses & Arguments** | [src/utils.py](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/src/utils.py) | `ScriptArguments.sft_data_fraction`, `ScopeDataGenArguments`, `SsfoDataGenArguments`. |
| **Pipelines** | [src/pipelines.py](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/src/pipelines.py) | `SFTPipeline`, `ScopeDataGenerationPipeline`, `SSFODataGenerationPipeline`, `DPOPipeline`. |
| **SCOPE Entry Point** | [src/scope_data_generation.py](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/src/scope_data_generation.py) | CLI runner for SCOPE noisy decoding. |
| **SSFO Entry Point** | [src/ssfo_data_generation.py](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/src/ssfo_data_generation.py) | CLI runner for SSFO context contrasting generation. |
| **DPO Entry Point** | [src/dpo.py](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/src/dpo.py) | Shared CLI runner for DPO preference training (used by both SCOPE and SSFO). |
| **SCOPE Scripts** | [scripts/scope_baseline.sh](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/scripts/scope_baseline.sh)<br>[scripts/scope_data_generation.sh](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/scripts/scope_data_generation.sh)<br>[scripts/scope_dpo.sh](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/scripts/scope_dpo.sh) | Shell launch scripts for SCOPE end-to-end, data generation, and DPO training. |
| **SSFO Scripts** | [scripts/ssfo_baseline.sh](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/scripts/ssfo_baseline.sh)<br>[scripts/ssfo_data_generation.sh](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/scripts/ssfo_data_generation.sh)<br>[scripts/ssfo_dpo.sh](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/scripts/ssfo_dpo.sh) | Shell launch scripts for SSFO end-to-end, data generation, and DPO training. |
| **Unit Tests** | [src/test_scope.py](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/src/test_scope.py)<br>[src/test_ssfo.py](file:///google/src/cloud/leobianco/new_perl/google3/experimental/users/leobianco/new_perl/src/test_ssfo.py) | Unit tests verifying prompt formatting, decoding simulation, and pipeline setups. |

---

## Usage & Execution Guide

All commands must be executed from the **root project directory**. Substitute `(TASK)` with `npov`, `bosch`, or `ragtruth`.

### Running SCOPE

#### End-to-End:
```bash
./scripts/scope_baseline.sh (TASK) all
```

#### By Stage:
```bash
./scripts/scope_baseline.sh (TASK) sft            # Stage 1: SFT on D1 (50%)
./scripts/scope_baseline.sh (TASK) generate_data  # Stage 2: Noisy decoding on D2
./scripts/scope_baseline.sh (TASK) dpo            # Stage 3: DPO preference training
```

---

### Running SSFO

#### End-to-End:
```bash
./scripts/ssfo_baseline.sh (TASK) all
```

#### By Stage:
```bash
./scripts/ssfo_baseline.sh (TASK) sft            # Stage 1: SFT training
./scripts/ssfo_baseline.sh (TASK) generate_data  # Stage 2: Context contrasting generation
./scripts/ssfo_baseline.sh (TASK) dpo            # Stage 3: DPO preference training
```

---

## Hyperparameter Reference

| Hyperparameter | Flag | SCOPE Default | SSFO Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| $\alpha$ (Noise rate) | `--alpha` | `0.5` | N/A | Bernoulli probability of sampling from base model $p_{\text{LM}}$. |
| Split Ratio | `--split_ratio` | `0.5` | N/A | Data partition fraction ($\mathcal{D}_1$ vs. $\mathcal{D}_2$). |
| Sampling Mode | `--sampling_mode` | `bernoulli` | N/A | `bernoulli`, `prob_mix`, or `logit_mix`. |
| Ground Truth Chosen | `--use_ground_truth_chosen` | `True` (fixed) | `False` (configurable) | Use ground truth target vs. context-conditioned generation. |
| Base Model for Rejected | `--use_base_model_for_rejected` | N/A | `True` (configurable) | Use base instruction model (no adapter) for context-free rejected generation. |
| Gen Temperature | `--temperature` | `0.7` | `0.7` | Temperature for autoregressive sampling. |
| Gen Top-$p$ | `--top_p` | `0.9` | `0.9` | Nucleus sampling threshold. |
| Gen Top-$k$ | `--top_k` | `50` | `50` | Top-$k$ vocabulary filtering. |
| Max New Tokens | `--max_new_tokens` | `150` | `150` | Maximum generated tokens per completion. |
| DPO $\beta$ | `--beta` | `0.1` | `0.1` | Temperature parameter in the DPO objective. |
| DPO Learning Rate | `--learning_rate` | `5e-6` | `5e-6` | Peak learning rate for DPO policy tuning. |
| DPO Epochs | `--num_train_epochs` | `1` | `1` | Number of DPO training epochs. |
| LoRA Rank ($r$) | `--lora_r` | `8` | `8` | LoRA adapter rank. |
| LoRA Alpha | `--lora_alpha` | `16` | `16` | LoRA scaling factor. |

---

## Comparison: PE-RL vs. SCOPE vs. SSFO

| Dimension | PE-RL (Our Approach) | SCOPE (Baseline) | SSFO (Baseline) |
| :--- | :--- | :--- | :--- |
| **Optimization Method** | Reinforcement Learning (RLOO / PPO) | Direct Preference Optimization (DPO) | Direct Preference Optimization (DPO) |
| **Reward Mechanism** | Token/sequence Reward Model score | Pairwise preference likelihood ratio | Pairwise preference likelihood ratio |
| **Negative Data Source** | Synthetic hallucinations (structured / LLM perturbed) | Noisy decoding (mixing SFT and base logits) | Context-free generation (withholding $c$) |
| **Base Model Needed in Gen?** | No | Yes (for $p_{\text{LM}}$ logit mixing) | Yes (base IT model for rejected $y^-$) |
| **Data Split Required** | Full dataset for RM + RL loops | Split dataset in half ($\mathcal{D}_1$ / $\mathcal{D}_2$) | Full dataset for SFT + generation |
| **Generation Efficiency** | Online sampling during RL rollouts | Dual-model decoding per token | Standard single-model decoding per prompt |
