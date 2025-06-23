"""TODO: write docstring."""

from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import wandb
from peft import LoraConfig
from sklearn.metrics import accuracy_score, roc_curve


@dataclass
class ScriptArguments:
    """Arguments common to all scripts (reward model, SFT, PERL)."""

    task: str
    dataset_repo_id: str
    model_repo_id: str
    num_fewshot: Optional[int] = None


@dataclass
class CustomLoraConfig(LoraConfig):
    """Work around HfArgumentParser bug..."""

    init_lora_weights: bool = field(default=True)
    layers_to_transform: int = field(default=None)
    loftq_config: dict = field(default_factory=dict)


def hallucination_rate_from_score_file(filepath, threshold):
    """Given a scores.txt file and a threshold, returns hallucination rate."""

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
    """Given a scores.txt file and a threshold, returns its average."""

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
    """Histogram of scores."""

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


def compute_best_roc_threshold_and_log(labels, scores, log_to_wandb=False):
    """
    Given ground truth labels and prediction scores, compute the ROC curve,
    find the best threshold (maximizing tpr-fpr), and return threshold, tpr, fpr, accuracy.
    Optionally log these metrics to wandb.
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
    if log_to_wandb:
        wandb.log(metrics)
    return metrics
