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
2. **Reference Alignment**: ROUGE-1/2/L and BERTScore F1 against ground-truth targets.
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
  1. Computes contextual embeddings for both the generated completion and the reference.
  2. Calculates normalized cosine similarity between the semantic embeddings.
  3. Returns the similarity score in $[0.0, 1.0]$.

---

### 2. Fluency / Perplexity Model
* **Default / Recommended Models**: `google/gemma-4-E2B-it`, `google/gemma-2b`, or `gpt2-xl`
* **Configuration Flags**: `--compute_perplexity True --fluency_model "google/gemma-4-E2B-it"`
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
* **Open-Weights Fallback**: `google/gemma-4-26B-A4B-it` or `google/gemma-4-E2B-it` via Hugging Face CausalLM.
* **Configuration Flags**: `--evaluator_model "gemini-2.5-flash" --use_gemini True --threshold 0.991`
* **How It Works**:
  * Formats task-specific prompt with few-shot calibration examples.
  * Extracts the model's calibrated probability $P(\text{"No hallucination"})$ from token logprobs or structured output.
  * Applies the calibrated `--threshold` (e.g. 0.991) to classify each generation as faithful (`1`) vs. hallucinated (`0`).

---

## Full Metrics Catalog

| Metric | Category | Formula / Definition | Good Value | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`scores`** | Faithfulness | $P(\text{No Hallucination})$ | Higher ($\to 1.0$) | Raw calibrated autorater probability. |
| **`classifications`** | Faithfulness | $\mathbb{I}(\text{score} \ge \tau)$ | `1.0` (Faithful) | Binary classification after thresholding. |
| **`hallucination_rate`** | Faithfulness | $\frac{1}{N}\sum \mathbb{I}(\text{score} < \tau)$ | Lower ($\to 0.0$) | Overall run hallucination percentage. |
| **`rouge1_f1`** | Reference Alignment | Unigram overlap F1 | Higher ($\to 1.0$) | Overlap of individual words with reference. |
| **`rouge2_f1`** | Reference Alignment | Bigram overlap F1 | Higher ($\to 1.0$) | Overlap of word pairs with reference. |
| **`rougeL_f1`** | Reference Alignment | Longest Common Subsequence F1 | Higher ($\to 1.0$) | Structural sequence similarity with reference. |
| **`bertscore_f1`** | Reference Alignment | DeBERTa token embedding cosine F1 | Higher ($\to 1.0$) | Semantic similarity robust to paraphrasing. |
| **`token_length`** | Generation Health | Count of generated word/subword tokens | Task-dependent | Verifies policy did not collapse in length. |
| **`char_length`** | Generation Health | Total character count | Task-dependent | Raw text length. |
| **`distinct_1`** | Diversity | $\frac{\|\text{Unique Unigrams}\|}{\|\text{Total Unigrams}\|}$ | Higher ($\to 1.0$) | Vocabulary diversity within the completion. |
| **`distinct_2`** | Diversity | $\frac{\|\text{Unique Bigrams}\|}{\|\text{Total Bigrams}\|}$ | Higher ($\to 1.0$) | Phrase diversity; catches 2-token loops. |
| **`repetition_rate`** | Diversity | $1.0 - \text{Distinct-}4$ | Lower ($\to 0.0$) | Fraction of repeated 4-grams in output. |
| **`perplexity`** | Fluency | $\exp(\mathcal{L}_{\text{CE}}(\text{comp} \mid \text{prompt}))$ | Lower ($\to 1.0$) | Natural language fluency under base LM. |

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
