"""Evaluation metrics module for generative language models.

Provides publication-grade reference alignment (ROUGE-1/2/L, BERTScore F1),
generation health (character and token length, Distinct-1/2, repetition rate),
fluency (conditional perplexity), and interactive Weights & Biases (WandB)
visualization and reporting.
"""

import json
import math
import os
import re
from typing import Any, Optional

try:
  import numpy as np
except ImportError:
  np = None

try:
  import torch
except ImportError:
  torch = None

try:
  from rouge_score import rouge_scorer
except ImportError:
  rouge_scorer = None

try:
  import wandb
except ImportError:
  wandb = None


def tokenize_words(text: str) -> list[str]:
  """Extracts word tokens from text using word boundary matching.

  Args:
      text: The input string.

  Returns:
      A list of lowercase word tokens.
  """
  if not text or not isinstance(text, str):
    return []
  return re.findall(r"\b\w+\b", text.lower())


def _lcs_length(seq1: list[str], seq2: list[str]) -> int:
  """Computes length of longest common subsequence."""
  m, n = len(seq1), len(seq2)
  dp = [0] * (n + 1)
  for i in range(1, m + 1):
    prev = 0
    for j in range(1, n + 1):
      temp = dp[j]
      if seq1[i - 1] == seq2[j - 1]:
        dp[j] = prev + 1
      else:
        dp[j] = max(dp[j], dp[j - 1])
      prev = temp
  return dp[n]


def _pure_python_rouge_f1(
    pred_tokens: list[str], ref_tokens: list[str], n: int = 1
) -> float:
  """Computes ROUGE-n F1 in pure Python."""
  if not pred_tokens or not ref_tokens:
    return 0.0
  if n == 0:  # ROUGE-L via LCS
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
      return 0.0
    prec = lcs / len(pred_tokens)
    rec = lcs / len(ref_tokens)
    return float(2 * prec * rec / (prec + rec))

  pred_ngrams = [
      tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)
  ]
  ref_ngrams = [
      tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
  ]
  if not pred_ngrams or not ref_ngrams:
    return 0.0

  pred_counts = {}
  for ng in pred_ngrams:
    pred_counts[ng] = pred_counts.get(ng, 0) + 1

  overlap = 0
  for ng in ref_ngrams:
    if pred_counts.get(ng, 0) > 0:
      overlap += 1
      pred_counts[ng] -= 1

  prec = overlap / len(pred_ngrams)
  rec = overlap / len(ref_ngrams)
  if prec + rec == 0:
    return 0.0
  return float(2 * prec * rec / (prec + rec))


def compute_rouge(
    predictions: list[str],
    references: list[str],
    use_stemmer: bool = True,
) -> dict[str, list[float]]:
  """Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores for prediction-reference pairs.

  Args:
      predictions: List of generated completion strings.
      references: List of ground-truth reference strings.
      use_stemmer: Whether to apply Porter stemmer in ROUGE evaluation.

  Returns:
      Dict mapping metric name to list of per-sample F1 scores:
      'rouge1_f1', 'rouge2_f1', 'rougeL_f1'.
  """
  if len(predictions) != len(references):
    raise ValueError(
        f"Length mismatch: {len(predictions)} predictions vs"
        f" {len(references)} references"
    )

  scorer = None
  if rouge_scorer is not None:
    try:
      scorer = rouge_scorer.RougeScorer(
          ["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer
      )
    except Exception:
      scorer = None

  r1_scores = []
  r2_scores = []
  rl_scores = []

  for pred, ref in zip(predictions, references):
    pred_clean = pred.strip() if pred else ""
    ref_clean = ref.strip() if ref else ""
    if not pred_clean or not ref_clean:
      r1_scores.append(0.0)
      r2_scores.append(0.0)
      rl_scores.append(0.0)
      continue

    if scorer is not None:
      score = scorer.score(ref_clean, pred_clean)
      r1_scores.append(float(score["rouge1"].fmeasure))
      r2_scores.append(float(score["rouge2"].fmeasure))
      rl_scores.append(float(score["rougeL"].fmeasure))
    else:
      pred_toks = tokenize_words(pred_clean)
      ref_toks = tokenize_words(ref_clean)
      r1_scores.append(_pure_python_rouge_f1(pred_toks, ref_toks, n=1))
      r2_scores.append(_pure_python_rouge_f1(pred_toks, ref_toks, n=2))
      rl_scores.append(_pure_python_rouge_f1(pred_toks, ref_toks, n=0))

  return {
      "rouge1_f1": r1_scores,
      "rouge2_f1": r2_scores,
      "rougeL_f1": rl_scores,
  }


# Prevent huggingface_hub / tqdm download overflow bug in Python 3.13
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: Optional[str] = None,
) -> list[float]:
  """Computes semantic embedding similarity / BERTScore between predictions and references.

  Supports sentence-transformers embeddings (fast cosine similarity on mean-pooled
  vectors) and standard transformer encoders (token-level greedy matching BERTScore).

  Args:
      predictions: List of generated completion strings.
      references: List of ground-truth reference strings.
      model_type: HuggingFace model identifier for contextual embeddings.
      batch_size: Batch size for embedding computation.
      device: Device string (e.g. 'cuda', 'cpu'). Auto-detected if None.

  Returns:
      List of per-sample similarity float values in [0.0, 1.0].
  """
  if len(predictions) != len(references):
    raise ValueError(
        f"Length mismatch: {len(predictions)} predictions vs"
        f" {len(references)} references"
    )
  if not predictions:
    return []
  if torch is None:
    return [0.0] * len(predictions)

  try:
    from transformers import AutoModel, AutoTokenizer, logging as hf_logging
  except ImportError:
    return [0.0] * len(predictions)

  if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

  # Load model cleanly without noisy weight-initialization warnings
  prev_verbosity = hf_logging.get_verbosity()
  hf_logging.set_verbosity_error()
  try:
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModel.from_pretrained(model_type)
    model.to(device)
    model.eval()
  except Exception as e:
    print(
        f"Notice: Semantic similarity model '{model_type}' could not be"
        f" loaded: {e}. Defaulting scores to 0.0."
    )
    return [0.0] * len(predictions)
  finally:
    hf_logging.set_verbosity(prev_verbosity)

  scores = []
  is_sentence_emb = any(
      k in model_type.lower() for k in ["sentence-transformers", "minilm", "bge", "mpnet"]
  )

  for i in range(0, len(predictions), batch_size):
    batch_preds = predictions[i : i + batch_size]
    batch_refs = references[i : i + batch_size]

    valid_pairs = []
    for idx, (p, r) in enumerate(zip(batch_preds, batch_refs)):
      p_str = p.strip() if (p and isinstance(p, str)) else ""
      r_str = r.strip() if (r and isinstance(r, str)) else ""
      if p_str and r_str:
        valid_pairs.append((idx, p_str, r_str))

    if not valid_pairs:
      scores.extend([0.0] * len(batch_preds))
      continue

    batch_scores = [0.0] * len(batch_preds)
    v_preds = [item[1] for item in valid_pairs]
    v_refs = [item[2] for item in valid_pairs]

    try:
      p_enc = tokenizer(
          v_preds,
          padding=True,
          truncation=True,
          max_length=512,
          return_tensors="pt",
      ).to(device)
      r_enc = tokenizer(
          v_refs,
          padding=True,
          truncation=True,
          max_length=512,
          return_tensors="pt",
      ).to(device)

      with torch.no_grad():
        p_out = model(**p_enc)
        r_out = model(**r_enc)

        if is_sentence_emb:
          p_mask = p_enc["attention_mask"].unsqueeze(-1).float()
          p_vec = (p_out.last_hidden_state * p_mask).sum(1) / torch.clamp(
              p_mask.sum(1), min=1e-9
          )
          p_vec = torch.nn.functional.normalize(p_vec, p=2, dim=-1)

          r_mask = r_enc["attention_mask"].unsqueeze(-1).float()
          r_vec = (r_out.last_hidden_state * r_mask).sum(1) / torch.clamp(
              r_mask.sum(1), min=1e-9
          )
          r_vec = torch.nn.functional.normalize(r_vec, p=2, dim=-1)

          sims = (p_vec * r_vec).sum(dim=-1).cpu().tolist()
          for (orig_idx, _, _), sim in zip(valid_pairs, sims):
            batch_scores[orig_idx] = max(0.0, min(1.0, float(sim)))
        else:
          for (orig_idx, _, _), p_emb, r_emb, p_m, r_m in zip(
              valid_pairs,
              p_out.last_hidden_state,
              r_out.last_hidden_state,
              p_enc["attention_mask"],
              r_enc["attention_mask"],
          ):
            p_valid = p_emb[p_m.bool()]
            r_valid = r_emb[r_m.bool()]
            if len(p_valid) > 2:
              p_valid = p_valid[1:-1]
            if len(r_valid) > 2:
              r_valid = r_valid[1:-1]
            p_norm = torch.nn.functional.normalize(p_valid, p=2, dim=-1)
            r_norm = torch.nn.functional.normalize(r_valid, p=2, dim=-1)
            sim = torch.matmul(p_norm, r_norm.transpose(0, 1))
            r_val = sim.max(dim=0).values.mean().item()
            p_val = sim.max(dim=1).values.mean().item()
            f1 = (
                (2.0 * p_val * r_val) / (p_val + r_val)
                if (p_val + r_val) > 0
                else 0.0
            )
            batch_scores[orig_idx] = max(0.0, min(1.0, float(f1)))
    except Exception:
      pass

    scores.extend(batch_scores)

  return scores


def compute_length_metrics(
    completions: list[str],
    tokenizer: Optional[Any] = None,
) -> dict[str, list[int]]:
  """Computes character and token length metrics for a list of completions.

  Args:
      completions: List of text completions.
      tokenizer: Optional Hugging Face tokenizer instance. If None, word-level
        tokenization is used.

  Returns:
      Dict with 'char_length' and 'token_length' lists.
  """
  char_lengths = []
  token_lengths = []

  for comp in completions:
    text = comp if comp is not None else ""
    char_lengths.append(len(text))
    if tokenizer is not None:
      tokens = tokenizer.encode(text, add_special_tokens=False)
      token_lengths.append(len(tokens))
    else:
      token_lengths.append(len(tokenize_words(text)))

  return {
      "char_length": char_lengths,
      "token_length": token_lengths,
  }


def compute_distinct_n(
    completions: list[str],
    n: int = 1,
) -> list[float]:
  """Computes Distinct-n metric (ratio of unique n-grams to total n-grams).

  Distinct-n = |Unique n-grams| / |Total n-grams|

  Args:
      completions: List of text completions.
      n: The n-gram size (e.g. 1 for unigrams, 2 for bigrams).

  Returns:
      List of per-sample Distinct-n float scores in [0.0, 1.0].
  """
  if n < 1:
    raise ValueError(f"n must be >= 1, got {n}")

  distinct_scores = []
  for comp in completions:
    tokens = tokenize_words(comp)
    if len(tokens) < n:
      distinct_scores.append(0.0 if not tokens else 1.0)
      continue

    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    total_ngrams = len(ngrams)
    if total_ngrams == 0:
      distinct_scores.append(0.0)
    else:
      unique_ngrams = len(set(ngrams))
      distinct_scores.append(float(unique_ngrams / total_ngrams))

  return distinct_scores


def compute_repetition_rate(
    completions: list[str],
    n: int = 4,
) -> list[float]:
  """Computes n-gram repetition rate (fraction of duplicate n-grams).

  Repetition Rate = 1.0 - Distinct-n = (|Total n-grams| - |Unique n-grams|) /
  |Total n-grams|

  Args:
      completions: List of text completions.
      n: The n-gram size (default 4 for 4-gram repetition).

  Returns:
      List of per-sample repetition rate float scores in [0.0, 1.0].
  """
  distinct_scores = compute_distinct_n(completions, n=n)
  return [float(1.0 - d) for d in distinct_scores]


def compute_conditional_perplexity(
    prompts: list[str],
    completions: list[str],
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    batch_size: int = 4,
    device: Optional[str] = None,
    max_length: int = 2048,
) -> list[float]:
  """Computes conditional perplexity of completions conditioned on prompts.

  Loss is evaluated strictly on the completion tokens with prompt tokens masked.
  PPL = exp( CrossEntropyLoss(completion | prompt) )

  Args:
      prompts: List of prompt strings.
      completions: List of completion strings.
      model: Pre-trained CausalLM model instance.
      tokenizer: Pre-trained tokenizer.
      batch_size: Batch size for evaluation.
      device: Device to place tensors on.
      max_length: Maximum sequence length.

  Returns:
      List of per-sample perplexity values (float).
  """
  if len(prompts) != len(completions):
    raise ValueError(
        f"Length mismatch: {len(prompts)} prompts vs {len(completions)}"
        " completions"
    )

  if model is None or tokenizer is None or torch is None:
    print(
        "Warning: Model, tokenizer, or torch not provided for perplexity"
        " computation. Returning 0.0."
    )
    return [0.0] * len(prompts)

  if device is None:
    device = next(model.parameters()).device

  model.eval()
  loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
  perplexities = []

  if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

  for i in range(0, len(prompts), batch_size):
    batch_prompts = prompts[i : i + batch_size]
    batch_completions = completions[i : i + batch_size]

    for p, c in zip(batch_prompts, batch_completions):
      p_clean = p if p else ""
      c_clean = c.strip() if c else ""

      if not c_clean:
        perplexities.append(0.0)
        continue

      prompt_ids = tokenizer.encode(
          p_clean, add_special_tokens=True, truncation=True, max_length=max_length
      )
      comp_ids = tokenizer.encode(
          c_clean, add_special_tokens=False, truncation=True, max_length=max_length
      )

      if not comp_ids:
        perplexities.append(0.0)
        continue

      full_ids = prompt_ids + comp_ids
      if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        prompt_len = max(0, len(full_ids) - len(comp_ids))
      else:
        prompt_len = len(prompt_ids)

      input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
      labels = torch.tensor([full_ids], dtype=torch.long, device=device)
      labels[:, :prompt_len] = -100

      with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_per_token = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        valid_mask = shift_labels.view(-1) != -100
        num_valid = valid_mask.sum().item()

        if num_valid == 0:
          perplexities.append(0.0)
        else:
          mean_loss = loss_per_token[valid_mask].mean().item()
          ppl = float(math.exp(min(mean_loss, 20.0)))
          perplexities.append(ppl)

  return perplexities


def compute_summary_statistics(
    scores: list[float | int],
) -> dict[str, float]:
  """Computes mean, standard deviation, median, min, and max for a score list.

  Args:
      scores: List of numeric values.

  Returns:
      Dict with 'mean', 'std', 'median', 'min', 'max'.
  """
  clean_scores = []
  for s in scores:
    if s is not None:
      try:
        val = float(s)
        if not math.isnan(val):
          clean_scores.append(val)
      except (ValueError, TypeError):
        continue

  if not clean_scores:
    return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}

  n = len(clean_scores)
  mean_val = float(sum(clean_scores) / n)
  variance = float(sum((x - mean_val) ** 2 for x in clean_scores) / n)
  std_val = float(math.sqrt(variance))

  sorted_scores = sorted(clean_scores)
  if n % 2 == 1:
    median_val = float(sorted_scores[n // 2])
  else:
    median_val = float((sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2.0)

  return {
      "mean": mean_val,
      "std": std_val,
      "median": median_val,
      "min": float(sorted_scores[0]),
      "max": float(sorted_scores[-1]),
  }


class GenerationMetricsEvaluator:
  """Orchestrator for evaluating full generation quality suites."""

  def __init__(
      self,
      bertscore_model: str = "sentence-transformers/all-MiniLM-L6-v2",
      compute_bertscore_metric: bool = True,
      compute_perplexity_metric: bool = False,
      fluency_model: Optional[Any] = None,
      fluency_tokenizer: Optional[Any] = None,
  ):
    """Initializes the GenerationMetricsEvaluator.

    Args:
        bertscore_model: Model name for BERTScore embedding computation.
        compute_bertscore_metric: Whether to compute BERTScore.
        compute_perplexity_metric: Whether to compute conditional perplexity.
        fluency_model: Optional CausalLM model for perplexity.
        fluency_tokenizer: Optional tokenizer for perplexity.
    """
    self.bertscore_model = bertscore_model
    self.compute_bertscore_metric = compute_bertscore_metric
    self.compute_perplexity_metric = compute_perplexity_metric
    self.fluency_model = fluency_model
    self.fluency_tokenizer = fluency_tokenizer

  def find_reference_column(self, column_names: list[str]) -> Optional[str]:
    """Identifies the reference/ground truth column from dataset column names.

    Args:
        column_names: List of column names in dataset.

    Returns:
        The matched reference column name, or None if not found.
    """
    candidates = [
        "npov_response",
        "reference",
        "target",
        "ground_truth",
        "chosen",
        "gold",
        "answer",
        "response",
    ]
    for c in candidates:
      if c in column_names:
        return c
    return None

  def evaluate_dataset(
      self,
      dataset: Any,
      prompt_column: str = "prompt",
      completion_column: str = "completion",
      reference_column: Optional[str] = None,
      tokenizer: Optional[Any] = None,
  ) -> tuple[Any, dict[str, Any]]:
    """Evaluates a dataset containing prompts and completions across all metrics.

    Args:
        dataset: Dataset instance or MockDataset.
        prompt_column: Column name for prompts.
        completion_column: Column name for generated completions.
        reference_column: Column name for ground-truth references. If None,
          auto-detected.
        tokenizer: Optional tokenizer for length calculation.

    Returns:
        Tuple of (updated dataset with metric columns, summary metrics dict).
    """
    if completion_column not in dataset.column_names:
      raise ValueError(
          f"Column '{completion_column}' not found in dataset columns:"
          f" {dataset.column_names}"
      )

    prompts = (
        list(dataset[prompt_column])
        if prompt_column in dataset.column_names
        else [""] * len(dataset)
    )
    completions = list(dataset[completion_column])

    # 1. Generation Health: Length metrics
    length_metrics = compute_length_metrics(completions, tokenizer=tokenizer)

    # 2. Generation Health: Diversity & Repetition
    distinct_1 = compute_distinct_n(completions, n=1)
    distinct_2 = compute_distinct_n(completions, n=2)
    repetition_4 = compute_repetition_rate(completions, n=4)

    metric_columns = {
        "char_length": length_metrics["char_length"],
        "token_length": length_metrics["token_length"],
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "repetition_rate": repetition_4,
    }

    # 3. Reference Alignment (ROUGE & BERTScore)
    ref_col = reference_column or self.find_reference_column(
        dataset.column_names
    )
    if ref_col is not None and ref_col in dataset.column_names:
      references = list(dataset[ref_col])
      rouge_dict = compute_rouge(completions, references)
      metric_columns["rouge1_f1"] = rouge_dict["rouge1_f1"]
      metric_columns["rouge2_f1"] = rouge_dict["rouge2_f1"]
      metric_columns["rougeL_f1"] = rouge_dict["rougeL_f1"]

      if self.compute_bertscore_metric:
        bs_scores = compute_bertscore(
            completions, references, model_type=self.bertscore_model
        )
        metric_columns["bertscore_f1"] = bs_scores

    # 4. Fluency (Conditional Perplexity)
    if (
        self.compute_perplexity_metric
        and self.fluency_model is not None
        and self.fluency_tokenizer is not None
    ):
      ppl_scores = compute_conditional_perplexity(
          prompts,
          completions,
          model=self.fluency_model,
          tokenizer=self.fluency_tokenizer,
      )
      metric_columns["perplexity"] = ppl_scores

    # 5. Add columns to dataset
    updated_dict = (
        dict(dataset.to_dict())
        if hasattr(dataset, "to_dict")
        else {col: list(dataset[col]) for col in dataset.column_names}
    )
    for col_name, values in metric_columns.items():
      updated_dict[col_name] = values

    try:
      from datasets import Dataset
      updated_dataset = Dataset.from_dict(updated_dict)
    except Exception:
      updated_dataset = dataset
      for col_name, values in metric_columns.items():
        if col_name in updated_dataset.column_names:
          updated_dataset = updated_dataset.remove_columns(col_name)
        updated_dataset = updated_dataset.add_column(col_name, values)

    # 6. Compute summary statistics
    summary = {}
    for col_name, values in metric_columns.items():
      stats = compute_summary_statistics(values)
      summary[f"{col_name}_mean"] = stats["mean"]
      summary[f"{col_name}_std"] = stats["std"]
      summary[f"{col_name}_median"] = stats["median"]

    return updated_dataset, summary

  def save_and_log_results(
      self,
      dataset: Any,
      summary: dict[str, Any],
      output_dir: str = "logs/eval",
      dataset_name: str = "eval_results",
      autorater_scores: Optional[list[float]] = None,
      threshold: float = 0.5,
      log_to_wandb: bool = False,
      wandb_project: str = "new_perl_eval",
      wandb_run_name: Optional[str] = None,
  ) -> str:
    """Saves evaluation summary locally and logs rich dashboards to WandB.

    Args:
        dataset: Evaluated dataset containing metric columns.
        summary: Aggregated summary metrics dictionary.
        output_dir: Local directory for summary JSON files.
        dataset_name: Identifier name for the dataset.
        autorater_scores: Optional list of autorater scores (P(No
          hallucination)).
        threshold: Threshold for classifying hallucination vs. faithful.
        log_to_wandb: Whether to initialize and log to WandB.
        wandb_project: WandB project name.
        wandb_run_name: WandB run name.

    Returns:
        Path to the saved local summary JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"{dataset_name}_summary.json")

    if autorater_scores:
      clean_autorater = [
          float(s) for s in autorater_scores if s is not None
      ]
      if clean_autorater:
        hallu_rate = float(
            sum(1.0 for s in clean_autorater if s < threshold)
            / len(clean_autorater)
        )
        faithful_rate = float(1.0 - hallu_rate)
        summary["hallucination_rate"] = hallu_rate
        summary["faithfulness_rate"] = faithful_rate
        autorater_stats = compute_summary_statistics(clean_autorater)
        summary["autorater_score_mean"] = autorater_stats["mean"]
        summary["autorater_score_std"] = autorater_stats["std"]

    with open(summary_path, "w") as f:
      json.dump(summary, f, indent=2)
    print(f"Summary metrics saved locally to: {summary_path}")

    if log_to_wandb and wandb is not None:
      try:
        run_name = wandb_run_name or dataset_name
        is_wandb_active = wandb.run is not None
        if not is_wandb_active:
          wandb.init(
              project=wandb_project,
              name=run_name,
              config={"dataset_name": dataset_name, "threshold": threshold},
          )

        wandb_metrics = {f"eval/{k}": v for k, v in summary.items()}
        wandb.log(wandb_metrics)
        for k, v in wandb_metrics.items():
          wandb.summary[k] = v

        table_cols = [
            "prompt",
            "completion",
            "token_length",
            "distinct_2",
            "repetition_rate",
        ]
        if "rougeL_f1" in dataset.column_names:
          table_cols.append("rougeL_f1")
        if "bertscore_f1" in dataset.column_names:
          table_cols.append("bertscore_f1")
        if "perplexity" in dataset.column_names:
          table_cols.append("perplexity")
        if "scores" in dataset.column_names:
          table_cols.append("scores")
        if "classifications" in dataset.column_names:
          table_cols.append("classifications")

        ref_col = self.find_reference_column(dataset.column_names)
        if ref_col and ref_col not in table_cols:
          table_cols.insert(2, ref_col)

        table_data = []
        sample_indices = range(min(500, len(dataset)))
        for idx in sample_indices:
          row = [
              dataset[col][idx]
              if col in dataset.column_names
              else None
              for col in table_cols
          ]
          table_data.append(row)

        eval_table = wandb.Table(columns=table_cols, data=table_data)
        wandb.log({"eval/generations_table": eval_table})
        print("Logged interactive generation table and metrics to WandB.")
      except Exception as e:
        print(f"Warning: Failed to log metrics to WandB: {e}")

    return summary_path
