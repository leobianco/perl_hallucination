"""TODO: write docstring."""

from dataclasses import dataclass, field

from peft import LoraConfig


@dataclass
class ScriptArguments:
    """Arguments common to all scripts (reward model, SFT, PERL)."""

    task: str
    dataset_repo_id: str
    model_repo_id: str


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
