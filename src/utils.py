"""Utility functions and argument dataclasses for training and evaluation scripts.

Argument dataclasses for script configuration, helper functions for LoRA
argument parsing, and utilities for analyzing and visualizing model scores,
including ROC analysis and histogram plotting.
"""

import argparse
from dataclasses import dataclass, field
import hashlib
import os
import re
from typing import Any, Dict, Optional, Sequence, Union

try:
  import matplotlib.pyplot as plt
except ImportError:
  plt = None

try:
  import numpy as np
except ImportError:
  np = None

try:
  from sklearn.metrics import accuracy_score, roc_curve
except ImportError:
  accuracy_score = None
  roc_curve = None


@dataclass
class ScriptArguments:
  """Arguments common to all scripts (reward model, SFT, PERL, DPO).

  Attributes:
      task_name (str): Name of the task.
      dataset_repo_id (str): Dataset repository identifier.
      model_repo_id (str): Model repository identifier.
      num_fewshot (Optional[int]): Number of few-shot examples to use.
      reward_model_path (Optional[str]): Path to the reward model checkpoint.
      sft_model_path (Optional[str]): Path to the SFT model adapter checkpoint.
      sft_data_fraction (Optional[float]): Fraction of SFT training data to use
        (e.g. 0.5 for SCOPE Stage 1).
      reward_penalty_alpha (float): Multiplier on negative logit differences to
        asymmetrically penalize hallucinations (defaults to 1.0, symmetric).
      reward_max_length (int): Token budget used whenever the reward model
        scores a (prompt, completion) pair, both at reward-model training time
        and at PE-RL scoring time.
  """

  task_name: str
  dataset_repo_id: str
  model_repo_id: str
  num_fewshot: Optional[int] = None
  reward_model_path: Optional[str] = None
  sft_model_path: Optional[str] = None
  sft_data_fraction: Optional[float] = None
  reward_penalty_alpha: float = 1.0
  # Token budget for reward-model scoring. Used BOTH when training the reward
  # model and when querying it during PE-RL; the two must never diverge.
  # It must hold the prompt *and* the completion: if the completion is
  # truncated away, every rollout of a prompt gets an identical reward, the
  # RLOO advantage is exactly zero, and only the KL term is optimized.
  # RAGTruth: 2048 covers ~p98 of prompt+response, 4096 covers all of it.
  reward_max_length: int = 2048



@dataclass
class ScopeDataGenArguments:
  """Arguments for SCOPE synthetic preference data generation (Algorithm 1 in https://arxiv.org/pdf/2502.13674#page=4.69).

  Attributes:
      task_name (str): Name of the task (npov, bosch, ragtruth).
      dataset_repo_id (str): SFT dataset repository identifier.
      model_repo_id (str): Pre-trained base model repository identifier (e.g.
        google/gemma-4-E4B).
      sft_model_path (str): Fine-tuned SFT checkpoint adapter path or repository
        identifier.
      output_dataset_repo_id (Optional[str]): Repository identifier for saving
        generated preference data.
      alpha (float): Mixing probability / noise level for noisy decoding
        (Algorithm 1). Default: 0.5.
      split_ratio (float): Fraction of SFT data used for SFT training D1 vs
        SCOPE D2 generation. Default: 0.5.
      sampling_mode (str): Sampling strategy: 'bernoulli' (Algorithm 1),
        'prob_mix', 'cad', or 'logit_mix'. Default: 'bernoulli'.
      temperature (float): Sampling temperature (0.0 for greedy decoding,
        matching paper). Default: 0.0.
      top_p (float): Nucleus sampling top_p threshold. Default: 1.0.
      top_k (int): Top-k token filtering. Default: 0.
      repetition_penalty (float): Repetition penalty for generation. Default:
        1.0.
      max_new_tokens (int): Maximum new tokens to generate per completion.
        Default: 250.
      seed (int): Random seed. Default: 12345.
      max_samples (Optional[int]): Maximum samples to process (useful for
        testing/debugging). Default: None.
      batch_size (int): Batch size for parallel noisy decoding. Default: 16.
      n_untouched_logits (int): Number of initial tokens to keep purely
        conditional from the SFT model (matching mixture_n_untouched in SCOPE).
        Default: 2.
      push_to_hub (bool): Whether to push the resulting preference dataset to HF
        Hub. Default: True.
      output_dir (Optional[str]): Optional local directory to save the generated
        dataset. Default: None.
  """

  task_name: str
  dataset_repo_id: str
  model_repo_id: str
  sft_model_path: str
  output_dataset_repo_id: Optional[str] = None
  alpha: float = 0.5
  split_ratio: float = 0.5
  sampling_mode: str = "bernoulli"
  temperature: float = 0.0
  top_p: float = 1.0
  top_k: int = 0
  repetition_penalty: float = 1.0
  max_new_tokens: int = 250
  seed: int = 12345
  max_samples: Optional[int] = None
  batch_size: int = 16
  n_untouched_logits: int = 2
  push_to_hub: bool = True
  output_dir: Optional[str] = None


@dataclass
class SsfoDataGenArguments:
  """Arguments for SSFO synthetic preference data generation.

  Reference: "SSFO: Self-Supervised Faithfulness Optimization for
  Retrieval-Augmented Generation" (https://arxiv.org/abs/2508.17225).

  Attributes:
      task_name (str): Name of the task (npov, bosch, ragtruth).
      dataset_repo_id (str): SFT dataset repository identifier.
      model_repo_id (str): Base model repository identifier.
      sft_model_path (str): Fine-tuned SFT checkpoint adapter path or repository
        identifier.
      output_dataset_repo_id (Optional[str]): Repository identifier for saving
        generated preference data.
      temperature (float): Sampling temperature (0.0 for greedy). Default: 0.0.
      top_p (float): Nucleus sampling top_p threshold. Default: 1.0.
      top_k (int): Top-k token filtering. Default: 0.
      repetition_penalty (float): Repetition penalty for generation. Default:
        1.0.
      max_new_tokens (int): Maximum new tokens to generate per completion.
        Default: 250.
      seed (int): Random seed. Default: 12345.
      max_samples (Optional[int]): Maximum samples to process (useful for
        testing/debugging). Default: None.
      batch_size (int): Batch size for parallel generation. Default: 16.
      use_ground_truth_chosen (bool): If True, use ground truth target as chosen
        instead of SFT model generation with context. Default: False.
      use_base_model_for_rejected (bool): If True, use base instruction-tuned
        model (without SFT adapter) to generate context-free rejected completions.
        Default: True.
      push_to_hub (bool): Whether to push the resulting preference dataset to HF
        Hub. Default: True.
      output_dir (Optional[str]): Optional local directory to save the generated
        dataset. Default: None.
      max_model_len (Optional[int]): Context window vLLM allocates a KV cache
        for. None uses the model's declared `max_position_embeddings`, which
        can be 262144 on recent Qwen checkpoints and will OOM at engine
        startup on smaller GPUs. Default: None.
  """

  task_name: str
  dataset_repo_id: str
  model_repo_id: str
  sft_model_path: str
  output_dataset_repo_id: Optional[str] = None
  temperature: float = 0.0
  top_p: float = 1.0
  top_k: int = 0
  repetition_penalty: float = 1.0
  max_new_tokens: int = 250
  seed: int = 12345
  max_samples: Optional[int] = None
  batch_size: int = 16
  use_ground_truth_chosen: bool = False
  use_base_model_for_rejected: bool = True
  push_to_hub: bool = True
  output_dir: Optional[str] = None
  max_model_len: Optional[int] = None


@dataclass
class LLMSynthScriptArguments:
  """Additional script arguments controlling how many organic and structured samples to keep.

  Attributes:
      num_struct_hallus_to_keep (Optional[int]): Number of structured
        hallucinations to keep.
      num_organic_hallus_to_keep (Optional[int]): Number of organic
        hallucinations to keep.
  """

  num_struct_hallus_to_keep: Optional[int] = 0
  num_organic_hallus_to_keep: Optional[int] = 0


@dataclass
class LoraArguments:
  """Arguments for LoRA and PEFT configuration.

  Attributes:
      task_type (str): Task type for PEFT (e.g., CAUSAL_LM, SEQ_CLS).
      peft_type (str): PEFT method type (e.g., LORA).
      lora_r (int): LoRA attention dimension (rank).
      lora_alpha (int): LoRA alpha parameter for scaling.
      lora_dropout (float): LoRA dropout probability.
      lora_target_modules (Optional[str]): Comma-separated module names to
        adapt, or None to let PEFT infer them from the architecture.
  """

  task_type: str = field(
      default="CAUSAL_LM",
      metadata={
          "help": (
              "Task type for PEFT (e.g., CAUSAL_LM, SEQ_CLS). Default:"
              " CAUSAL_LM"
          )
      },
  )
  peft_type: str = field(
      default="LORA",
      metadata={"help": "PEFT method type. Default: LORA"},
  )
  lora_r: int = field(
      default=8,
      metadata={"help": "LoRA attention dimension (rank). Default: 8"},
  )
  lora_alpha: int = field(
      default=16,
      metadata={"help": "LoRA alpha parameter for scaling. Default: 16"},
  )
  lora_dropout: float = field(
      default=0.0,
      metadata={"help": "LoRA dropout probability. Default: 0.0"},
  )
  lora_target_modules: Optional[str] = field(
      default=None,
      metadata={
          "help": (
              "Comma-separated module names to attach LoRA to, e.g."
              " 'q_proj,k_proj,v_proj,o_proj'. Leave unset to let PEFT infer"
              " them from the architecture, which is what every Gemma run has"
              " done. Only needed when PEFT does not recognise the model"
              " family."
          )
      },
  )


def create_lora_argument_parser() -> argparse.ArgumentParser:
  """Create an argument parser for LoRA and PEFT configuration.

  Returns:
      argparse.ArgumentParser: Configured argument parser for LoRA/PEFT
      arguments.
  """
  parser = argparse.ArgumentParser(
      description="Training script with LoRA configuration"
  )

  parser.add_argument(
      "--task_type",
      type=str,
      default="CAUSAL_LM",
      choices=[
          "CAUSAL_LM",
          "SEQ_CLS",
          "SEQ_2_SEQ_LM",
          "TOKEN_CLS",
          "QUESTION_ANS",
          "FEATURE_EXTRACTION",
      ],
      help="Task type for PEFT. Default: CAUSAL_LM",
  )
  parser.add_argument(
      "--peft_type",
      type=str,
      default="LORA",
      choices=[
          "LORA",
          "PROMPT_TUNING",
          "P_TUNING",
          "PREFIX_TUNING",
          "IA3",
          "ADALORA",
      ],
      help="PEFT method type. Default: LORA",
  )

  parser.add_argument(
      "--lora_r",
      type=int,
      default=8,
      help="LoRA attention dimension (rank). Default: 8",
  )
  parser.add_argument(
      "--lora_alpha",
      type=int,
      default=16,
      help="LoRA alpha parameter for scaling. Default: 16",
  )
  parser.add_argument(
      "--lora_dropout",
      type=float,
      default=0.0,
      help="LoRA dropout probability. Default: 0.0",
  )
  parser.add_argument(
      "--lora_target_modules",
      type=str,
      default=None,
      help=(
          "Comma-separated module names to attach LoRA to, e.g."
          " 'q_proj,k_proj,v_proj,o_proj'. Default: unset, meaning PEFT infers"
          " them from the architecture. Only needed when PEFT does not"
          " recognise the model family."
      ),
  )

  return parser


@dataclass
class EvalArguments:
  """Arguments for evaluation scripts (generation, scoring, autorater eval)."""

  task_name: str = field(metadata={"help": "Name of the task (NPOV, HalOmi)."})

  user: str = field(
      metadata={
          "help": "The user to use for writing and loading to and from HF."
      },
  )

  writer_model_lora: str = field(
      metadata={"help": "The path to the LoRA adapters of the writer model."}
  )

  sft_model_path: Optional[str] = field(
      default=None,
      metadata={
          "help": (
              "Optional path or Hugging Face repo ID of the SFT LoRA adapter "
              "to combine with writer_model_lora when evaluating RL/DPO models."
          )
      },
  )

  allow_missing_sft_adapter: bool = field(
      default=False,
      metadata={
          "help": (
              "Allow evaluating an RL/DPO checkpoint without its SFT adapter. "
              "Off by default: such a checkpoint is a LoRA delta trained on "
              "merged SFT weights, so serving it on the bare base model is a "
              "different model from the one that was trained. Set True only "
              "for a deliberate ablation."
          )
      },
  )

  autorater_num_samples: int = field(
      default=1,
      metadata={
          "help": (
              "Number of independent autorater calls per sample. The judge is "
              "a remote LLM whose scores jitter between calls even at "
              "temperature 0, so k > 1 queries it k times and keeps the "
              "median, additionally reporting the observed spread. Default 1 "
              "(one call per sample, no aggregation); k > 1 multiplies the "
              "API cost by k."
          )
      },
  )

  mode: Optional[str] = field(
      default=None,
      metadata={
          "help": "Evaluation mode: 'generate', 'score', or 'autoratereval'."
      },
  )

  dataset_labels: Optional[str] = field(
      default=None,
      metadata={
          "help": (
              "Dataset with hallucination labels, for evaluation of evaluator"
              " or for getting few-shot examples."
          )
      },
  )

  dataset_labels_split: Optional[str] = field(
      default=None,
      metadata={"help": "What split of the dataset_labels to use."},
  )

  dataset_prompts: Optional[str] = field(
      default=None,
      metadata={
          "help": (
              "Dataset with prompts to be used for generation (not necessarily"
              " has hallucination labels)."
          )
      },
  )

  dataset_prompts_split: Optional[str] = field(
      default=None,
      metadata={"help": "What split of the dataset_prompts to use."},
  )

  writer_model_base: Optional[str] = field(
      default=None,
      metadata={"help": "The base model for the writer (name or path)."},
  )

  evaluator_model: str = field(
      default="gemini-2.0-flash",
      metadata={
          "help": (
              "The model name or path to the model to use as evaluator (e.g."
              " gemini-2.0-flash or google/gemma-4-26B-A4B-it)."
          )
      },
  )

  use_gemini: bool = field(
      default=True,
      metadata={
          "help": "Using the Gemini API (e.g. gemini-2.0-flash) as evaluator"
      },
  )

  run_autorater: bool = field(
      default=True,
      metadata={
          "help": (
              "Whether to run the autorater (Gemini or CausalLM) for"
              " hallucination scoring. Set False to skip."
          )
      },
  )

  gemini_api_key: Optional[str] = field(
      default="", metadata={"help": "API key for calling Gemini"}
  )

  seed: int = field(default=12345)

  max_eval_samples: Optional[int] = field(
      default=1000,
      metadata={
          "help": (
              "Maximum number of samples to evaluate (subsampling). Set to -1"
              " or 0 to evaluate full dataset."
          )
      },
  )

  eval_batch_size: int = field(
      default=32,
      metadata={"help": "Batch size for local model evaluation."},
  )

  max_workers: int = field(
      default=16,
      metadata={
          "help": "Number of concurrent worker threads for Gemini API scoring."
      },
  )

  max_tokens: int = field(default=128)

  max_model_len: Optional[int] = field(
      default=None,
      metadata={
          "help": (
              "Context window vLLM should allocate a KV cache for. Leave unset"
              " to use the model's declared `max_position_embeddings`. Models"
              " advertising very long contexts (e.g."
              " Qwen/Qwen3-4B-Instruct-2507 declares 262144) will OOM at"
              " engine startup on smaller GPUs; cap this to something the"
              " prompts actually need (e.g. 8192)."
          )
      },
  )

  temperature: float = field(default=0.0)

  top_p: float = field(default=1)

  top_k: int = field(
      default=0,
      metadata={
          "help": (
              "The number of highest probability vocabulary tokens to keep for"
              " top-k-filtering. 0 means no top-k filtering."
          )
      },
  )

  evaluator_num_fewshot: Optional[int] = field(
      default=0,
      metadata={
          "help": (
              "The number of fewshot examples to give to the evaluator. Half"
              " will be positive (contain hallucination), half will be"
              " negative."
          )
      },
  )

  evaluate_evaluator: bool = field(
      default=False,
      metadata={"help": "Whether to run evaluation of the evaluator or not."},
  )

  threshold: float = field(
      default=0.5,
      metadata={
          "help": "The value of the threshold to turn scores into classif."
      },
  )

  dataset_with_completions: Optional[str] = field(
      default=None,
      metadata={
          "help": (
              "Name of the dataset with completions on HF Hub. Required when"
              " evaluate_evaluator is False and not generating."
          )
      },
  )

  writer_num_fewshot: int = field(
      default=0,
      metadata={
          "help": (
              "Number of few-shot examples to prepend to each prompt. If zero,"
              " no few-shot examples are prepended."
          )
      },
  )

  compute_generation_metrics: bool = field(
      default=True,
      metadata={
          "help": (
              "Whether to compute full generation quality metrics (ROUGE,"
              " BERTScore, lengths, distinct-n, repetition, perplexity)."
          )
      },
  )

  compute_bertscore: bool = field(
      default=True,
      metadata={"help": "Whether to compute BERTScore F1 metric."},
  )

  bertscore_model: str = field(
      default="sentence-transformers/all-MiniLM-L6-v2",
      metadata={"help": "Model identifier for semantic similarity embeddings."},
  )

  compute_perplexity: bool = field(
      default=False,
      metadata={"help": "Whether to compute conditional perplexity."},
  )

  fluency_model: Optional[str] = field(
      default=None,
      metadata={
          "help": (
              "Pre-trained causal LM model identifier for fluency perplexity."
          )
      },
  )

  reference_column: Optional[str] = field(
      default=None,
      metadata={
          "help": (
              "Name of reference/ground truth column in dataset. If None,"
              " auto-detected."
          )
      },
  )

  log_to_wandb: bool = field(
      default=False,
      metadata={
          "help": "Whether to log evaluation metrics and tables to WandB."
      },
  )

  wandb_project: str = field(
      default="new_perl_eval",
      metadata={"help": "WandB project name for evaluation logging."},
  )

  wandb_run_name: Optional[str] = field(
      default=None,
      metadata={"help": "Optional WandB run name."},
  )

  overwrite_scores: bool = field(
      default=False,
      metadata={
          "help": (
              "Whether to overwrite existing autorater scores and re-score"
              " all entries from scratch."
          )
      },
  )

  scores_checkpoint_path: Optional[str] = field(
      default=None,
      metadata={
          "help": (
              "Custom path to save/load autorater scores checkpoints for"
              " resuming incomplete scoring runs."
          )
      },
  )


def hallucination_rate_from_score_file(
    filepath: str, threshold: float
) -> Optional[float]:
  """Compute the hallucination rate from a score file given a threshold.

  Args:
      filepath (str): Path to the file containing scores (one per line).
      threshold (float): Threshold below which a score is considered a
        hallucination.

  Returns:
      Optional[float]: Fraction of scores below the threshold, or None if file
      is empty or not found.
  """

  below_threshold_count = 0
  total_count = 0

  try:
    with open(filepath, "r") as file:
      for line in file:
        try:
          value = float(line.strip())
          total_count += 1
          if value < threshold:
            below_threshold_count += 1
        except ValueError:
          continue

    if total_count == 0:
      return None

    fraction_below_threshold = below_threshold_count / total_count
    return fraction_below_threshold

  except FileNotFoundError:
    print(f"Error: The file at '{filepath}' was not found.")
    return None


def average_from_score_file(filepath: str) -> Optional[float]:
  """Compute the average score from a score file.

  Args:
      filepath (str): Path to the file containing scores (one per line).

  Returns:
      Optional[float]: Average score, or None if file is empty or not found.
  """

  total_count = 0
  total_value = 0

  try:
    with open(filepath, "r") as file:
      for line in file:
        try:
          value = float(line.strip())
          total_value += value
          total_count += 1
        except ValueError:
          continue

    if total_count == 0:
      return None

    avg = total_value / total_count
    return avg

  except FileNotFoundError:
    print(f"Error: The file at '{filepath}' was not found.")
    return None


def histogram_from_score_file(filepath: str) -> None:
  """Plot and save a histogram of scores from a file.

  Args:
      filepath (str): Path to the file containing scores (one per line).

  Returns:
      None
  """

  name = filepath.rstrip(".txt")
  scores = []

  try:
    with open(filepath, "r") as file:
      for line in file:
        try:
          value = float(line.strip())
          scores.append(value)
        except ValueError:
          continue

    bins = np.linspace(0, 1, 10)
    plt.hist(scores, bins=bins, density=True, alpha=0.5, label=name)
    plt.legend()
    plt.savefig(f"{name}_histogram.png")

    return None

  except FileNotFoundError:
    print(f"Error: The file at '{filepath}' was not found.")
    return None


def compute_best_roc_threshold(
    labels: Union[Sequence[int], Any],
    scores: Union[Sequence[float], Any],
) -> Dict[str, float]:
  """Compute the best ROC threshold and related metrics.

  Given ground truth labels and prediction scores, compute the ROC curve,
  find the best finite threshold (maximizing TPR-FPR), and return threshold,
  TPR, FPR, and accuracy.

  Args:
      labels (array-like): Ground truth binary labels.
      scores (array-like): Prediction scores.

  Returns:
      dict: Dictionary with keys 'best_threshold', 'tpr_at_best_threshold',
        'fpr_at_best_threshold', and 'accuracy_at_best_threshold'.
  """

  fpr, tpr, thresholds = roc_curve(labels, scores)
  finite_mask = np.isfinite(thresholds)
  if np.any(finite_mask):
    valid_indices = np.where(finite_mask)[0]
    best_sub_idx = np.argmax(tpr[valid_indices] - fpr[valid_indices])
    threshold_idx = valid_indices[best_sub_idx]
  else:
    threshold_idx = 0
  threshold = float(thresholds[threshold_idx])
  classif_at_threshold = [0 if score < threshold else 1 for score in scores]
  accuracy = accuracy_score(labels, classif_at_threshold)
  metrics = {
      "best_threshold": threshold,
      "tpr_at_best_threshold": float(tpr[threshold_idx]),
      "fpr_at_best_threshold": float(fpr[threshold_idx]),
      "accuracy_at_best_threshold": float(accuracy),
  }

  return metrics


def recompute_autorater_metrics(
    ground_truth_filepath: str,
    scores_filepath: str,
    output_metrics_filepath: Optional[str] = None,
) -> Dict[str, float]:
  """Recomputes autorater evaluation metrics from saved ground truth and scores files.

  Args:
      ground_truth_filepath (str): Path to ground truth file (one int per line).
      scores_filepath (str): Path to scores file (one float per line).
      output_metrics_filepath (Optional[str]): Optional path to save the
        recomputed metrics.

  Returns:
      Dict[str, float]: Computed metrics dictionary (AUC, Threshold, TPR, FPR,
      Accuracy, Precision).
  """
  from sklearn.metrics import precision_score, roc_auc_score

  with open(ground_truth_filepath, "r") as f:
    ground_truth = [int(line.strip()) for line in f if line.strip()]

  with open(scores_filepath, "r") as f:
    scores = [float(line.strip()) for line in f if line.strip()]

  if len(ground_truth) != len(scores):
    raise ValueError(
        f"Length mismatch: {len(ground_truth)} ground truth vs {len(scores)}"
        " scores"
    )

  scores_arr = np.array(scores)
  labels_arr = np.array(ground_truth)

  try:
    auc = float(roc_auc_score(labels_arr, scores_arr))
  except Exception:
    auc = 0.5

  roc_metrics = compute_best_roc_threshold(labels_arr, scores_arr)
  threshold = roc_metrics["best_threshold"]
  tpr = roc_metrics["tpr_at_best_threshold"]
  fpr = roc_metrics["fpr_at_best_threshold"]
  accuracy = roc_metrics["accuracy_at_best_threshold"]

  classif_at_threshold = [0 if s < threshold else 1 for s in scores_arr]
  try:
    precision = float(
        precision_score(labels_arr, classif_at_threshold, zero_division=0)
    )
  except Exception:
    precision = 0.0

  results = {
      "auc": auc,
      "threshold": threshold,
      "tpr": tpr,
      "fpr": fpr,
      "accuracy": accuracy,
      "precision": precision,
  }

  if output_metrics_filepath:
    with open(output_metrics_filepath, "w") as f:
      f.write(f"AUC: {auc:.5f}\n")
      f.write(f"Threshold: {threshold:.5f}\n")
      f.write(f"TPR (recall): {tpr:.5f}\n")
      f.write(f"FPR: {fpr:.5f}\n")
      f.write(f"Accuracy: {accuracy:.5f}\n")
      f.write(f"Precision: {precision:.5f}\n")

  return results


def get_task_processor(task_name: str) -> Any:
  """Return the TaskProcessor class for a given task name.

  This helper performs local imports to avoid circular import issues
  when pipelines import utils.

  Args:
      task_name (str): One of 'npov', 'bosch', 'ragtruth', 'ragtruth-qa',
        'ragtruth-summarization'.

  Returns:
      class: The TaskProcessor class corresponding to task_name.

  Raises:
      ValueError: If an unknown task_name is provided.
  """
  if task_name == "npov":
    from src.task_processors.npov_task_processor import NPOVTaskProcessor
    return NPOVTaskProcessor
  elif task_name == "bosch":
    from src.task_processors.bosch_task_processor import BoschTaskProcessor
    return BoschTaskProcessor
  elif (
      task_name in ("ragtruth", "ragtruth-qa", "ragtruth-summarization")
      or task_name.startswith("ragtruth")
  ):
    from src.task_processors.ragtruth_task_processor import RagtruthTaskProcessor
    return RagtruthTaskProcessor
  else:
    raise ValueError(f"Unknown task: {task_name}")


def compact_model_name(model_name: str) -> str:
  """Compacts float parameters in a model name/run identifier to prevent excessive string lengths.

  For example:
      'npov_PERL_google_S130104_epo0.2_lr2.3663655877360862e-05_beta0.04259128063013425_2608141244'
  becomes:
      'npov_PERL_google_S130104_epo0.2_lr2.4e-05_beta0.043_2608141244'

  Args:
      model_name (str): The raw model name or run identifier.

  Returns:
      str: The compacted model name.
  """
  if not model_name:
    return model_name

  def _replace_lr(match: re.Match) -> str:
    prefix, val_str = match.group(1), match.group(2)
    try:
      val = float(val_str)
      return f"{prefix}{val:.1e}"
    except ValueError:
      return match.group(0)

  def _replace_beta(match: re.Match) -> str:
    prefix, val_str = match.group(1), match.group(2)
    try:
      val = float(val_str)
      formatted = f"{val:.2g}" if val >= 0.001 else f"{val:.1e}"
      return f"{prefix}{formatted}"
    except ValueError:
      return match.group(0)

  def _replace_generic_float(match: re.Match) -> str:
    prefix, val_str = match.group(1), match.group(2)
    try:
      val = float(val_str)
      return f"{prefix}{val:.2g}"
    except ValueError:
      return match.group(0)

  name = re.sub(
      r"((?:^|_)(?:lr|learning_rate))([0-9]+\.[0-9]+(?:e-?[0-9]+)?)",
      _replace_lr,
      model_name,
      flags=re.IGNORECASE,
  )
  name = re.sub(
      r"((?:^|_)(?:beta|b))([0-9]+\.[0-9]+(?:e-?[0-9]+)?)",
      _replace_beta,
      name,
      flags=re.IGNORECASE,
  )
  name = re.sub(
      r"((?:^|_)(?:epo|epochs?|alpha|temp|temperature))([0-9]+\.[0-9]{3,})",
      _replace_generic_float,
      name,
      flags=re.IGNORECASE,
  )
  return name


def _bound_repo_name(repo_name: str, max_length: int) -> str:
  """Shortens ``repo_name`` to ``max_length`` without losing its tail.

  Repo names are built as ``{task}_{stage}_{model}_{flavor}_{hparams}_
  {timestamp}``, so the *most* disambiguating part - the timestamp - sits at
  the end. Plain right-truncation therefore removes exactly the characters
  that make the name unique: with a model name a dozen characters longer than
  ``gemma-4-E4B-it`` the whole timestamp disappears, and two campaigns run on
  the same day with the same hyperparameters silently push to one repository.

  Shortening from the middle keeps both the human-readable prefix and the
  timestamp, and the interposed hash of the full name restores the uniqueness
  the elided characters carried.

  Args:
      repo_name: The unbounded repository name.
      max_length: Maximum number of characters allowed.

  Returns:
      ``repo_name`` if it already fits, otherwise a middle-elided form of it.
  """
  if max_length <= 0:
    return ""
  if len(repo_name) <= max_length:
    return repo_name

  digest = hashlib.sha1(repo_name.encode("utf-8")).hexdigest()[:6]
  marker = f"_{digest}_"
  # Too tight to keep both ends legible: fall back to the old behaviour but
  # keep the hash, so the name is still unique.
  if max_length <= len(marker) + 8:
    return (digest + repo_name)[:max_length].rstrip(".-_")

  remaining = max_length - len(marker)
  # Bias towards the tail: the timestamp and the hyperparameters that differ
  # between sibling runs both live there.
  tail_len = min(len(repo_name), remaining * 2 // 3)
  head_len = remaining - tail_len
  head = repo_name[:head_len].rstrip(".-_")
  tail = repo_name[len(repo_name) - tail_len:].lstrip(".-_")
  return f"{head}{marker}{tail}"


def sanitize_hf_repo_id(repo_id: str, max_length: int = 96) -> str:
  """Sanitizes and bounds a Hugging Face repository ID to comply with Hugging Face Hub constraints.

  Hugging Face Hub rules:
  - Maximum repo_id length: 96 chars.
  - Allowed characters: alphanumeric, '-', '_'
  - Replaces all '.' with '_' to eliminate period-related validation issues.
  - Cannot start or end with '-' or '.'

  Note that uppercase characters are *allowed* by the Hub (``Qwen/Qwen3-4B``,
  ``mistralai/Mistral-7B-Instruct-v0.3``) and are deliberately preserved, so
  that a published checkpoint still names its base model recognisably.

  Args:
      repo_id (str): The repository identifier (e.g. 'user/dataset_name').
      max_length (int): Maximum length allowed (default 96).

  Returns:
      str: Sanitized and length-bounded repository identifier.
  """
  if not repo_id:
    return repo_id

  if "/" in repo_id:
    namespace, repo_name = repo_id.split("/", 1)
    namespace = namespace.strip()
    repo_name = repo_name.strip()
  else:
    namespace = None
    repo_name = repo_id.strip()

  repo_name = compact_model_name(repo_name)
  # Replace periods with underscores so there are NO periods in repo names
  repo_name = repo_name.replace(".", "_")
  # Replace any other non-alphanumeric characters (except - and _) with _
  repo_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", repo_name)
  # Collapse consecutive underscores and hyphens
  repo_name = re.sub(r"_+", "_", repo_name)
  repo_name = re.sub(r"-+", "-", repo_name)
  repo_name = repo_name.strip(".-_")

  if namespace:
    namespace = namespace.replace(".", "_")
    namespace = re.sub(r"[^a-zA-Z0-9_\-]", "_", namespace).strip(".-_")
    repo_name = _bound_repo_name(repo_name, max_length - len(namespace) - 1)
    return f"{namespace}/{repo_name}"
  else:
    return _bound_repo_name(repo_name, max_length)


#: String spellings of "no model here" that shells and configs keep producing.
#: ``scripts/*.sh`` expand an unset variable to an empty string, and the CLI
#: wizard writes the literal ``"none"``/``"false"``; both must mean the same
#: thing to the adapter loader and to the dataset naming, otherwise a run can
#: silently serve one model and be filed under another model's name.
FALSY_MODEL_REFS = frozenset({"", "none", "false", "null", "no", '""', "''"})

#: Substrings that identify a checkpoint as an RL/preference-tuned adapter,
#: which by construction is a delta on top of the SFT weights.
_RL_CHECKPOINT_MARKERS = (
    "perl",
    "rloo",
    "rlhf",
    "grpo",
    "ppo",
    "dpo",
    "ssfo",
    "scope",
)


def is_falsy_model_ref(model_ref: Optional[str]) -> bool:
  """Returns True when a model reference means "no model".

  Args:
      model_ref (Optional[str]): Adapter path, repo ID, or a placeholder.

  Returns:
      bool: True for None, empty/whitespace, or a "none"-like spelling.
  """
  if model_ref is None:
    return True
  return str(model_ref).strip().lower() in FALSY_MODEL_REFS


def normalize_model_ref(model_ref: Optional[str]) -> str:
  """Normalizes a model reference for comparison and fingerprinting.

  Args:
      model_ref (Optional[str]): Adapter path or Hugging Face repo ID.

  Returns:
      str: The trimmed reference without trailing slashes, or '' if falsy.
  """
  if is_falsy_model_ref(model_ref):
    return ""
  return str(model_ref).strip().rstrip("/")


def is_sft_adapter_stacked(
    sft_model_path: Optional[str],
    writer_model_lora: Optional[str] = None,
) -> bool:
  """Returns True when an SFT adapter is stacked on top of the writer adapter.

  Mirrors the condition used by
  ``EvaluationGenerationPipeline.setup_model`` to decide whether to combine
  two LoRA adapters. Keep the two in sync: the served weights and the
  dataset name are derived from this single predicate.

  Args:
      sft_model_path (Optional[str]): SFT adapter reference, if any.
      writer_model_lora (Optional[str]): The adapter being evaluated. When it
        is the SFT adapter itself, nothing is stacked.

  Returns:
      bool: True when two distinct adapters are combined for serving.
  """
  sft_ref = normalize_model_ref(sft_model_path)
  if not sft_ref:
    return False
  writer_ref = normalize_model_ref(writer_model_lora)
  return not (writer_ref and writer_ref.casefold() == sft_ref.casefold())


def sft_stacking_marker(
    sft_model_path: Optional[str],
    writer_model_lora: Optional[str] = None,
    hash_len: int = 6,
) -> str:
  """Builds the repo-name marker describing the SFT stacking configuration.

  The marker distinguishes the two ways the same RL checkpoint can be served:
  on the bare base model ('_nosft') or on top of an SFT adapter
  ('_sft<fingerprint>'). Without it both runs would push their completions to
  the same dataset repository and silently overwrite each other.

  The fingerprint is taken over the *reference string*, so the same checkpoint
  addressed as a local path and as a Hub repo ID yields two names. That is the
  safe direction to err in: a spurious split is visible, a spurious merge is
  not.

  Args:
      sft_model_path (Optional[str]): SFT adapter reference, if any.
      writer_model_lora (Optional[str]): The adapter being evaluated.
      hash_len (int): Number of hex characters of the fingerprint.

  Returns:
      str: '_nosft', or '_sft' followed by ``hash_len`` hex characters.
  """
  if not is_sft_adapter_stacked(sft_model_path, writer_model_lora):
    return "_nosft"
  digest = hashlib.sha256(
      normalize_model_ref(sft_model_path).casefold().encode("utf-8")
  ).hexdigest()[:hash_len]
  return f"_sft{digest}"


def looks_like_rl_checkpoint(model_ref: Optional[str]) -> bool:
  """Heuristically detects an RL/preference-tuned checkpoint from its name.

  Such a checkpoint is a LoRA delta trained on top of *merged* SFT weights, so
  evaluating it without its SFT adapter serves a model that never existed
  during training. Names produced by the orchestrator carry an explicit stage
  tag ('..._PERL_...', '..._SFT_...'), which makes this reliable in practice;
  an SFT tag always wins to avoid false positives from project prefixes such
  as 'new_perl'.

  Args:
      model_ref (Optional[str]): Adapter path or Hugging Face repo ID.

  Returns:
      bool: True when the reference looks like an RL/DPO checkpoint.
  """
  name = normalize_model_ref(model_ref).split("/")[-1].casefold()
  if not name or "sft" in name:
    return False
  return any(marker in name for marker in _RL_CHECKPOINT_MARKERS)


def _truncate_with_digest(name: str, allowed_len: int) -> str:
  """Shortens ``name`` to ``allowed_len`` chars while keeping it injective.

  Checkpoint names differ in their tail ('..._lr2_4e-05' versus
  '..._lr2_5e-05'), so a plain truncation can map two different runs onto one
  repository. Appending a digest of the full name keeps distinct inputs
  distinct.

  Args:
      name (str): The already-sanitized model name component.
      allowed_len (int): Maximum number of characters available.

  Returns:
      str: ``name`` when it already fits, else a truncation carrying a 4-hex
        digest of the full name.
  """
  if allowed_len <= 0:
    return ""
  if len(name) <= allowed_len:
    return name
  digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:4]
  if allowed_len <= len(digest):
    return digest[:allowed_len]
  keep = allowed_len - len(digest) - 1
  return f"{name[:keep].rstrip('.-_')}_{digest}"


def build_eval_dataset_repo_id(
    user: str,
    writer_model_lora: str,
    temperature: float = 0.0,
    writer_num_fewshot: int = 0,
    task_name: Optional[str] = None,
    sft_model_path: Optional[str] = None,
    seed: int = 12345,
    max_tokens: int = 250,
    max_length: int = 96,
) -> str:
  """Constructs a deterministic and valid Hugging Face dataset repo ID for generation completions.

  The ID must capture every input that changes the generated completions,
  otherwise two configurations push to the same repository and silently
  overwrite each other. Besides the writer adapter and the sampling settings,
  that includes whether an SFT adapter was stacked underneath the writer
  adapter ('..._nosft' versus '..._sft<fingerprint>') and the seed, which
  selects *which* prompts the evaluation subsamples.

  Not encoded, deliberately: ``top_p`` / ``top_k`` are inert at the
  temperature 0 this pipeline runs at, and ``max_eval_samples`` only ever
  shrinks the row set of the same repository. Change either of those together
  with something that is encoded, or regenerate into a fresh repo.

  Args:
      user (str): Hugging Face username / namespace.
      writer_model_lora (str): Writer model LoRA adapter path or repo ID.
      temperature (float): Sampling temperature used for generation.
      writer_num_fewshot (int): Number of fewshot examples prepended to prompts.
      task_name (Optional[str]): Task name (e.g., 'npov', 'bosch', 'ragtruth').
        Prefixes the repository name to prevent collision between tasks.
      sft_model_path (Optional[str]): SFT adapter combined with
        ``writer_model_lora`` at serving time, if any.
      seed (int): Seed driving the test-set subsample and the few-shot draw.
      max_tokens (int): Generation length cap, which truncates completions and
        therefore changes every downstream metric.
      max_length (int): Maximum allowed repo ID length (default 96).

  Returns:
      str: A compliant Hugging Face dataset repository identifier (<= max_length
        chars).
  """
  cleaned_path = (writer_model_lora or "").rstrip("/")
  if f"{user}/" in cleaned_path:
    model_name = cleaned_path.split(f"{user}/", 1)[1]
  elif "/" in cleaned_path:
    model_name = cleaned_path.split("/")[-1]
  else:
    model_name = cleaned_path

  compacted_model = compact_model_name(model_name).replace(".", "_")
  compacted_model = re.sub(r"[^a-zA-Z0-9_\-]", "_", compacted_model)
  compacted_model = re.sub(r"_+", "_", compacted_model).strip(".-_")

  temp_val = float(temperature)
  temp_str = (
      f"{int(temp_val)}"
      if temp_val.is_integer()
      else f"{temp_val:.2g}".replace(".", "_")
  )
  marker = sft_stacking_marker(sft_model_path, writer_model_lora)
  suffix = (
      f"_gens_T{temp_str}_wfs{writer_num_fewshot}"
      f"_s{int(seed)}_mt{int(max_tokens)}{marker}"
  )

  if task_name:
    task_clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", task_name.strip().lower())
    if not (
        compacted_model.lower().startswith(f"{task_clean}_")
        or compacted_model.lower() == task_clean
    ):
      prefix = f"eval_{task_clean}_"
    else:
      prefix = "eval_"
  else:
    prefix = "eval_"

  # The model name absorbs the whole overflow here, before
  # `sanitize_hf_repo_id` ever sees the name. That keeps the SFT marker and
  # the generation settings intact and, crucially, keeps the overflow
  # attributable: shortening the model deliberately is more legible than
  # letting the generic bounding elide an arbitrary middle span.
  allowed_model_len = max_length - len(user) - 1 - len(prefix) - len(suffix)
  if allowed_model_len <= 0:
    compacted_model = ""
  elif len(compacted_model) > allowed_model_len:
    compacted_model = _truncate_with_digest(compacted_model, allowed_model_len)

  full_name = f"{prefix}{compacted_model}{suffix}"
  return sanitize_hf_repo_id(f"{user}/{full_name}", max_length=max_length)


def autorater_eval_run_name(
    evaluator_model: Optional[str],
    dataset_labels: Optional[str],
    evaluator_num_fewshot: Optional[int],
    seed: int,
) -> str:
  """Names the log directory an autorater calibration run writes into.

  Every input that changes the fitted threshold is in the name: the judge,
  the labelled set it is judged against, how many few-shot examples it was
  given and the seed that drew them. Two calibrations that differ in any of
  those are different calibrations and must not overwrite each other.

  Shared by the pipeline that writes the directory and by the orchestrator
  stage that reads it back. They used to be two copies of one f-string,
  which is the same arrangement that made the evaluation summary lookup
  fragile (see ``EvalStage._summary_path``).

  Args:
      evaluator_model (Optional[str]): Judge model, e.g. 'gemini-2.5-flash'.
      dataset_labels (Optional[str]): Repo ID of the human-labelled set.
      evaluator_num_fewshot (Optional[int]): Few-shot examples given to the
        judge.
      seed (int): Seed for the subsample and the few-shot draw.

  Returns:
      str: The directory name, relative to ``logs/``.
  """
  eval_model_name = (
      evaluator_model.split("/")[-1] if evaluator_model else "gemini"
  )
  dataset_name = dataset_labels.split("/")[-1] if dataset_labels else "labels"
  return (
      f"eval_autorater_{eval_model_name}"
      f"_autorater_num_fewshot_{evaluator_num_fewshot}"
      f"_data_{dataset_name}"
      f"_seed_{seed}"
  )


def autorater_eval_metrics_path(
    evaluator_model: Optional[str],
    dataset_labels: Optional[str],
    evaluator_num_fewshot: Optional[int],
    seed: int,
) -> str:
  """Returns the machine-readable metrics file of an autorater calibration.

  The sibling ``eval_autorater_metrics.txt`` is written for a human; this
  JSON is what the orchestrator parses to pick up the fitted threshold.

  Args:
      evaluator_model (Optional[str]): Judge model.
      dataset_labels (Optional[str]): Repo ID of the human-labelled set.
      evaluator_num_fewshot (Optional[int]): Few-shot examples given.
      seed (int): Seed for the subsample and the few-shot draw.

  Returns:
      str: Path to the JSON metrics file.
  """
  name = autorater_eval_run_name(
      evaluator_model, dataset_labels, evaluator_num_fewshot, seed
  )
  return os.path.join("logs", name, "eval_autorater_metrics.json")
