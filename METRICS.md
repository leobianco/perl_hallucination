# Evaluation Metrics & Model Reference Guide

This document details the complete evaluation metrics suite implemented in this repository, the underlying models used for scoring (such as BERTScore, Perplexity, and the Autorater), and how to interpret and visualize the results.

---

## Table of Contents
1. [Overview & Evaluation Philosophy](#overview--evaluation-philosophy)
2. [Underlying Models Used](#underlying-models-used)
   - [BERTScore Model](#1-bertscore-model)
   - [Fluency / Perplexity Model](#2-fluency--perplexity-model)
   - [Autorater Model (Hallucination Scoring)](#3-autorater-model-hallucination-scoring)
3. [Full Metrics Catalog](#full-metrics-catalog)
   - [Reference Alignment Metrics](#reference-alignment-metrics)
   - [Generation Health Diagnostics](#generation-health-diagnostics)
   - [Fluency & Language Modeling](#fluency--language-modeling)
   - [Faithfulness & Hallucination Rate](#faithfulness--hallucination-rate)
4. [Storage, Artifacts & Weights & Biases (WandB)](#storage-artifacts--weights--biases-wandb)
5. [CLI & Script Configuration Reference](#cli--script-configuration-reference)

---

## Overview & Evaluation Philosophy

Evaluating alignment models (PE-RL, SCOPE, SSFO, SFT) solely on a single hallucination score is vulnerable to "reward hacking" (e.g. models learning to generate ultra-short, repetitive, or evasive outputs to artificially minimize hallucinations).

This codebase uses a **4-pillar evaluation framework**:
1. **Faithfulness**: Calibrated LLM autorater scoring.
2. **Context Grounding & Coverage**: ROUGE-1/2/L (Precision, Recall, F1) and BERTScore against the provided input context (e.g. source perspective arguments, manual excerpt, or documents).
3. **Generation Health**: Token/character length, Distinct-1/2 diversity, and n-gram repetition rates.
4. **Fluency**: Conditional perplexity under a neutral pre-trained language model.

---

## Underlying Models Used

### 1. Semantic Similarity / BERTScore Model
* **Default Model**: `sentence-transformers/all-MiniLM-L6-v2` (Fast, 80MB, safetensors, 0 warnings; or `bert-base-uncased` / `roberta-large`)
* **Configuration Flag**: `--bertscore_model "sentence-transformers/all-MiniLM-L6-v2"` (can be disabled via `--compute_bertscore False`)
* **Why all-MiniLM-L6-v2?**
  * Specially trained for semantic similarity and sentence embedding cosine correlation.
  * Super lightweight (80MB vs 1.4GB for roberta-large), processes 1,000 samples in <1s, and runs with 0 initialization warnings.
* **How It Works**:
  1. Computes contextual embeddings for both the generated completion and the source context (e.g. `perspective_1 + perspective_2`).
  2. Calculates normalized cosine similarity between the semantic embeddings.
  3. Returns the similarity score in $[0.0, 1.0]$.

---

### 2. Fluency / Perplexity Model
* **Default / Recommended Models**: `google/gemma-4-E4B-it`, `google/gemma-2b`, or `gpt2-xl`
* **Configuration Flags**: `--compute_perplexity True --fluency_model "google/gemma-4-E4B-it"`
* **Why a Neutral Base LM?**
  * Evaluates whether the fine-tuned or RL-aligned model produces grammatical, natural, and probable English text under a general language distribution.
  * Detects language degradation, gibberish loops, or unnatural phrasing that may arise during aggressive preference tuning or RL exploration.
* **How It Works (Conditional Perplexity)**:
  * To avoid penalizing valid completions for the difficulty of the prompt itself, we compute **conditional perplexity**:
    $$\text{PPL} = \exp\left( \frac{1}{|y|} \sum_{t=1}^{|y|} -\log p_{\text{LM}}(y_t \mid x, y_{<t}) \right)$$
  * The full sequence `[Prompt (x) + Completion (y)]` is tokenized, but all prompt tokens $x$ are masked with label `-100`. The cross-entropy loss is evaluated **strictly on the completion tokens $y$**.
  * A lower perplexity indicates higher fluency and natural likelihood under the base model.

---

### 3. Autorater Model (Hallucination Scoring)
* **Default Model**: `gemini-2.5-flash` (or `gemini-2.0-flash`) via Google GenAI Client (Vertex AI or AI Studio).
* **Open-Weights Fallback**: `google/gemma-4-26B-A4B-it` or `google/gemma-4-E4B-it` via Hugging Face CausalLM.
* **Configuration Flags**: `--evaluator_model "gemini-2.5-flash" --use_gemini True --threshold 0.991`
* **How It Works**:
  * Formats task-specific prompt with few-shot calibration examples.
  * Extracts the model's calibrated probability $P(\text{"No hallucination"})$ from token logprobs or structured output.
  * Applies the calibrated `--threshold` (e.g. 0.991) to classify each generation as faithful (`1`) vs. hallucinated (`0`).

---

### 4. Reward-Hacking Rubric Autorater (Writing Quality)
* **Default Model**: the same judge as the hallucination autorater (`--reward_hacking_model` overrides it).
* **Configuration Flags**: `--run_reward_hacking_autorater True --reward_hacking_num_fewshot 2 --reward_hacking_threshold 0.6`
* **Why it exists**: the hallucination autorater cannot see this failure. A policy that returns the retrieved context verbatim is, by construction, perfectly faithful to it, so the hallucination judge scores it as excellent - and RLOO, optimising exactly that reward, finds the policy. The symptoms are in the prose, not in the facts.
* **How It Works**:
  * One structured-JSON call per completion returns an integer in $[1, 5]$ for each of three dimensions: `fluency`, `non_repetition`, `non_extractiveness`.
  * Grades are rescaled to $[0, 1]$ ($1 \to 0.0$, $5 \to 1.0$, out-of-range clamped) and averaged unweighted into `reward_hacking_quality`.
  * `--autorater_num_samples k` draws $k$ independent grades per sample and takes their **mean** per dimension (not the median: the grades are already discretised to five points, so a median collapses back onto that grid). The per-dimension spread is recorded under `reward_hacking_audit_*`.
  * Few-shot demonstrations are built from the task's SFT split: each gold response is shown graded 5/5/5 next to a mechanically degenerated copy of *the same* response graded 1/1/1. Pairing them on identical content isolates writing quality as the only difference, so the judge cannot key the grade off topic or length.
* **Calibration status**: **uncalibrated.** There is no labelled reward-hacking set, so unlike `--threshold` (fitted by the `autoratereval` mode against human labels) `--reward_hacking_threshold` is a judgement call. Read `reward_hacking_quality` and the per-dimension means as the primary signal, and read `reward_hacking_rate` only as a *delta* between policies graded by the same judge.
  * To calibrate it later: score a contrast set of known-good responses (SFT gold) against known-hacked ones (the synthetic degenerations, or completions from a deliberately over-trained policy), then fit a threshold by ROC exactly as `EvaluationAutoraterPipeline` does for the hallucination judge. The synthetic degenerations are already available via `src.pipelines.degenerate_reward_hacked_response`.

---

## Full Metrics Catalog

| Metric | Category | Formula / Definition | Good Value | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`scores`** | Faithfulness | $P(\text{No Hallucination})$ | Higher ($\to 1.0$) | Raw calibrated autorater probability. |
| **`classifications`** | Faithfulness | $\mathbb{I}(\text{score} \ge \tau)$ | `1.0` (Faithful) | Binary classification after thresholding. |
| **`hallucination_rate`** | Faithfulness | $\frac{1}{N}\sum \mathbb{I}(\text{score} < \tau)$ | Lower ($\to 0.0$) | Overall run hallucination percentage. |
| **`reward_hacking_fluency`** | Writing Quality | Judge grade, rescaled to $[0, 1]$ | Higher ($\to 1.0$) | Is the text well-formed, grammatical prose? |
| **`reward_hacking_non_repetition`** | Writing Quality | Judge grade, rescaled to $[0, 1]$ | Higher ($\to 1.0$) | Is the text free of loops and restated facts? |
| **`reward_hacking_non_extractiveness`** | Writing Quality | Judge grade, rescaled to $[0, 1]$ | Higher ($\to 1.0$) | Is the text the model's own words rather than a copied span? |
| **`reward_hacking_quality`** | Writing Quality | Unweighted mean of the three dimensions | Higher ($\to 1.0$) | Headline writing-quality score. Read it *against* `hallucination_rate`. |
| **`reward_hacking_rate`** | Writing Quality | $\frac{1}{N}\sum \mathbb{I}(\text{quality} < \tau_{rh})$ | Lower ($\to 0.0$) | Fraction flagged as degenerate. **Uncalibrated**; trust the delta, not the level. |
| **`rouge1_precision`** | Context Grounding | $\frac{|\text{Overlap Unigrams}|}{|\text{Generation Unigrams}|}$ | Higher ($\to 1.0$) | Lexical grounding: fraction of generation from context. |
| **`rouge1_recall`** | Context Coverage | $\frac{|\text{Overlap Unigrams}|}{|\text{Context Unigrams}|}$ | Higher ($\to 1.0$) | Context coverage: fraction of context arguments preserved. |
| **`rouge1_f1`** | Context Alignment | Harmonic mean of $P_1$ & $R_1$ | Higher ($\to 1.0$) | Balanced unigram context grounding and coverage. |
| **`rouge2_precision`** | Context Grounding | Bigram lexical grounding | Higher ($\to 1.0$) | Bigram fraction originating from context. |
| **`rouge2_recall`** | Context Coverage | Bigram context coverage | Higher ($\to 1.0$) | Bigram fraction of context captured in output. |
| **`rouge2_f1`** | Context Alignment | Harmonic mean of $P_2$ & $R_2$ | Higher ($\to 1.0$) | Balanced bigram overlap with provided context. |
| **`rougeL_f1`** | Context Alignment | Longest Common Subsequence F1 | Higher ($\to 1.0$) | Structural sequence similarity with context. |
| **`bertscore_f1`** | Context Alignment | Sentence-embedding cosine sim | Higher ($\to 1.0$) | Semantic alignment between generation and context. |
| **`token_length`** | Generation Health | Count of generated word/subword tokens | Task-dependent | Verifies policy did not collapse in length. |
| **`char_length`** | Generation Health | Total character count | Task-dependent | Raw text length. |
| **`distinct_1`** | Diversity | $\frac{\|\text{Unique Unigrams}\|}{\|\text{Total Unigrams}\|}$ | Higher ($\to 1.0$) | Vocabulary diversity within the completion. |
| **`distinct_2`** | Diversity | $\frac{\|\text{Unique Bigrams}\|}{\|\text{Total Bigrams}\|}$ | Higher ($\to 1.0$) | Phrase diversity; catches 2-token loops. |
| **`repetition_rate`** | Diversity | $1.0 - \text{Distinct-}4$ | Lower ($\to 0.0$) | Fraction of repeated 4-grams in output. |
| **`perplexity`** | Fluency | $\exp(\mathcal{L}_{\text{CE}}(\text{comp} \mid \text{prompt}))$ | Lower ($\to 1.0$) | Natural language fluency under base LM. |

> The `reward_hacking_*` rows and the n-gram diversity rows overlap deliberately.
> `distinct_2` and `repetition_rate` are cheap and surface-level: they catch a
> token loop but score a fluent verbatim copy of the context as perfectly
> diverse. `reward_hacking_non_extractiveness` is the one that catches that,
> and it costs an API call.

---

## Storage, Artifacts & Weights & Biases (WandB)

### 1. Per-Sample Dataset Storage (Hugging Face Hub)
When scoring completes, the Hugging Face dataset is updated with all metric columns and pushed to:
`leobianco/eval_<RUN_IDENTIFIER>_gens_T<TEMP>_wfs<FEWSHOT>`

Columns stored:
```python
[
    "prompt",
    "completion",
    "scores",  # Autorater P(No)
    "classifications",  # Binary 0/1
    "char_length",  # Character count
    "token_length",  # Token count
    "distinct_1",  # Unigram uniqueness
    "distinct_2",  # Bigram uniqueness
    "repetition_rate",  # 4-gram repetition rate
    "rouge1_f1",  # ROUGE-1 F1
    "rouge2_f1",  # ROUGE-2 F1
    "rougeL_f1",  # ROUGE-L F1
    "bertscore_f1",  # BERTScore DeBERTa F1
    "perplexity",  # Conditional PPL (if enabled)
]
```

### 2. Local Summary Artifacts
Summary statistics (mean, std, median, min, max) for every metric are written to:
`logs/eval/<dataset_name>_summary.json`

### 3. Interactive Weights & Biases (WandB) Logging
Enable WandB logging by passing `LOG_TO_WANDB=True` or `--log_to_wandb True`:
* **Scalar Dashboard**: Logs `eval/hallucination_rate`, `eval/rougeL_f1_mean`, `eval/bertscore_f1_mean`, `eval/distinct_2_mean`, `eval/avg_token_length`.
* **Interactive `wandb.Table`**: Creates a searchable, sortable browser table containing all prompt-completion pairs alongside their individual ROUGE, BERTScore, autorater scores, and length.

---

## CLI & Script Configuration Reference

### Running Evaluation via Script
```bash
# Fast evaluation on 1,000 subsampled test samples (default):
./scripts/evaluator.sh npov generate
./scripts/evaluator.sh npov score

# Skip Gemini autorater (compute only ROUGE, BERTScore, length, diversity, fluency):
RUN_AUTORATER=False ./scripts/evaluator.sh npov score

# Full 10k evaluation for final results:
MAX_EVAL_SAMPLES=-1 ./scripts/evaluator.sh npov generate
MAX_EVAL_SAMPLES=-1 ./scripts/evaluator.sh npov score

# Scoring with Weights & Biases logging enabled:
LOG_TO_WANDB=True WANDB_PROJECT="my_perl_project" ./scripts/evaluator.sh npov score
```

### Running Evaluation Directly with Python
```bash
python3 -m src.evaluator \
    --task_name "npov" \
    --user "leobianco" \
    --dataset_with_completions "leobianco/eval_npov_gens" \
    --max_eval_samples 1000 \
    --run_autorater True \
    --evaluator_model "gemini-2.5-flash" \
    --use_gemini True \
    --threshold 0.991 \
    --compute_generation_metrics True \
    --compute_bertscore True \
    --bertscore_model "microsoft/deberta-v3-large" \
    --compute_perplexity False \
    --log_to_wandb True \
    --wandb_project "new_perl_eval"
```
