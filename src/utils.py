"""Utility functions and argument dataclasses for training and evaluation scripts.

This module provides argument dataclasses for script configuration, helper functions for LoRA argument parsing, and utilities for analyzing and visualizing model scores, including ROC analysis and histogram plotting.
"""

import argparse
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve


@dataclass
class ScriptArguments:
    """Arguments common to all scripts (reward model, SFT, PERL).

    Attributes:
        task_name (str): Name of the task.
        dataset_repo_id (str): Dataset repository identifier.
        model_repo_id (str): Model repository identifier.
        num_fewshot (Optional[int]): Number of few-shot examples to use.
    """

    task_name: str
    dataset_repo_id: str
    model_repo_id: str
    num_fewshot: Optional[int] = None


@dataclass
class LLMSynthScriptArguments:
    """Additional script arguments controlling how many organic and structured samples to keep.

    Attributes:
        num_struct_hallus_to_keep (Optional[int]): Number of structured hallucinations to keep.
        num_organic_hallus_to_keep (Optional[int]): Number of organic hallucinations to keep.
    """

    num_struct_hallus_to_keep: Optional[int] = 0
    num_organic_hallus_to_keep: Optional[int] = 0


def create_lora_argument_parser():
    """Create an argument parser for LoRA and PEFT configuration. This is a workaround the fact that the original LoraConfig is not compatible with HfArgumentParser due to the use of complex type hints.

    Returns:
        argparse.ArgumentParser: Configured argument parser for LoRA/PEFT arguments.
    """
    parser = argparse.ArgumentParser(
        description="Training script with LoRA configuration"
    )

    # PEFT configuration arguments
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

    # LoRA configuration arguments
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


def hallucination_rate_from_score_file(filepath, threshold):
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


def average_from_score_file(filepath):
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


def histogram_from_score_file(filepath):
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


def compute_best_roc_threshold(labels, scores):
    """Compute the best ROC threshold and related metrics.

    Given ground truth labels and prediction scores, compute the ROC curve, find the best threshold (maximizing TPR-FPR), and return threshold, TPR, FPR, and accuracy.

    Args:
        labels (array-like): Ground truth binary labels.
        scores (array-like): Prediction scores.

    Returns:
        dict: Dictionary with keys 'best_threshold', 'tpr_at_best_threshold', 'fpr_at_best_threshold', and 'accuracy_at_best_threshold'.
    """

    fpr, tpr, thresholds = roc_curve(labels, scores)
    threshold_idx = np.argmax(tpr - fpr)
    threshold = thresholds[threshold_idx]
    classif_at_threshold = [0 if score < threshold else 1 for score in scores]
    accuracy = accuracy_score(labels, classif_at_threshold)
    metrics = {
        "best_threshold": threshold,
        "tpr_at_best_threshold": tpr[threshold_idx],
        "fpr_at_best_threshold": fpr[threshold_idx],
        "accuracy_at_best_threshold": accuracy,
    }

    return metrics
