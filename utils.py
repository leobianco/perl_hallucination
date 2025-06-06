"""TODO: write docstring."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from peft import LoraConfig


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


def count_trainable_parameters(model):
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params
