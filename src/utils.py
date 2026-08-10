"""Utility functions and argument dataclasses for training and evaluation scripts.

Argument dataclasses for script configuration, helper functions for LoRA argument parsing, and utilities for analyzing and visualizing model scores, including ROC analysis and histogram plotting.
"""

import argparse
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve

from src.task_processors.base_task_processor import BaseTaskProcessor
from src.task_processors.bosch_task_processor import BoschTaskProcessor
from src.task_processors.npov_task_processor import NPOVTaskProcessor
from src.task_processors.ragtruth_task_processor import RagtruthTaskProcessor


@dataclass
class ScriptArguments:
    """Arguments common to all scripts (reward model, SFT, PERL).

    Attributes:
        task_name (str): Name of the task.
        dataset_repo_id (str): Dataset repository identifier.
        model_repo_id (str): Model repository identifier.
        num_fewshot (Optional[int]): Number of few-shot examples to use.
        reward_model_path (Optional[str]): Path to the reward model checkpoint.
        sft_model_path (Optional[str]): Path to the SFT model adapter checkpoint.
    """

    task_name: str
    dataset_repo_id: str
    model_repo_id: str
    num_fewshot: Optional[int] = None
    reward_model_path: Optional[str] = None
    sft_model_path: Optional[str] = None


@dataclass
class LLMSynthScriptArguments:
    """Additional script arguments controlling how many organic and structured samples to keep.

    Attributes:
        num_struct_hallus_to_keep (Optional[int]): Number of structured hallucinations to keep.
        num_organic_hallus_to_keep (Optional[int]): Number of organic hallucinations to keep.
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
    """

    task_type: str = field(
        default="CAUSAL_LM",
        metadata={
            "help": "Task type for PEFT (e.g., CAUSAL_LM, SEQ_CLS). Default: CAUSAL_LM"
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


def create_lora_argument_parser() -> argparse.ArgumentParser:
    """Create an argument parser for LoRA and PEFT configuration.

    Returns:
        argparse.ArgumentParser: Configured argument parser for LoRA/PEFT arguments.
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

    return parser


@dataclass
class EvalArguments:
    task_name: str = field(
        metadata={"help": "Name of the task (NPOV, HalOmi)."}
    )

    user: str = field(
        metadata={
            "help": "The user to use for writing and loading to and from HF."
        },
    )

    writer_model_lora: str = field(
        metadata={"help": "The path to the LoRA adapters of the writer model."}
    )

    dataset_labels: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset with hallucination labels, for evaluation of evaluator or for getting few-shot examples."
        },
    )

    dataset_labels_split: Optional[str] = field(
        default=None,
        metadata={"help": "What split of the dataset_labels to use."},
    )

    dataset_prompts: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset with prompts to be used for generation (not necessarily has hallucination labels)."
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
        default="gemini-3.5-flash",
        metadata={
            "help": "The model name or path to the model to use as evaluator (e.g. gemini-3.5-flash or google/gemma-4-26B-A4B-it)."
        },
    )

    use_gemini: bool = field(
        default=True,
        metadata={"help": "Using the Gemini API (e.g. gemini-3.5-flash) as evaluator"},
    )

    gemini_api_key: Optional[str] = field(
        default="", metadata={"help": "API key for calling Gemini"}
    )

    seed: int = field(default=12345)

    eval_batch_size: int = field(default=1)

    max_tokens: int = field(default=128)

    temperature: float = field(default=1)

    top_p: float = field(default=1)

    top_k: int = field(
        default=0,
        metadata={
            "help": "The number of highest probability vocabulary tokens to keep for top-k-filtering. 0 means no top-k filtering."
        },
    )

    evaluator_num_fewshot: Optional[int] = field(
        default=0,
        metadata={
            "help": "The number of fewshot examples to give to the evaluator. Half will be positive (contain hallucination), half will be negative."
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
            "help": "Name of the dataset with completions on HF Hub. Required when evaluate_evaluator is False and not generating."
        },
    )

    writer_num_fewshot: int = field(
        default=0,
        metadata={
            "help": "Number of few-shot examples to prepend to each prompt. If zero, no few-shot examples are prepended."
        },
    )


def hallucination_rate_from_score_file(
    filepath: str, threshold: float
) -> Optional[float]:
    """Compute the hallucination rate from a score file given a threshold.

    Args:
        filepath (str): Path to the file containing scores (one per line).
        threshold (float): Threshold below which a score is considered a hallucination.

    Returns:
        Optional[float]: Fraction of scores below the threshold, or None if file is empty or not found.
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
    labels: Union[Sequence[int], np.ndarray],
    scores: Union[Sequence[float], np.ndarray],
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
        ground_truth_filepath (str): Path to ground truth file (one int per
          line).
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
            f"Length mismatch: {len(ground_truth)} ground truth vs {len(scores)} scores"
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


def get_task_processor(task_name: str) -> type[BaseTaskProcessor]:
    """Return the TaskProcessor class for a given task name.

    This helper performs local imports to avoid circular import issues
    when pipelines import utils.

    Args:
        task_name (str): One of 'npov', 'bosch', 'ragtruth'.

    Returns:
        class: The TaskProcessor class corresponding to task_name.

    Raises:
        Exception: If an unknown task_name is provided.
    """

    task_map = {
        "npov": NPOVTaskProcessor,
        "bosch": BoschTaskProcessor,
        "ragtruth": RagtruthTaskProcessor,
    }

    processor_cls = task_map.get(task_name)
    if processor_cls is None:
        raise Exception(f"Unknown task: {task_name}")
    return processor_cls
