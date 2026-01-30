"""Utility functions and argument dataclasses for training and evaluation scripts.

Argument dataclasses for script configuration, helper functions for LoRA argument parsing, and utilities for analyzing and visualizing model scores, including ROC analysis and histogram plotting.
"""

import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Union

import gspread
import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, roc_curve
from transformers import TrainingArguments

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
    """

    task_name: str
    dataset_repo_id: str
    model_repo_id: str
    num_fewshot: Optional[int] = None
    gsheets_name: Optional[str] = ""


@dataclass
class LLMSynthScriptArguments:
    """Additional script arguments controlling how many organic and structured samples to keep.

    Attributes:
        num_struct_hallus_to_keep (Optional[int]): Number of structured hallucinations to keep.
        num_organic_hallus_to_keep (Optional[int]): Number of organic hallucinations to keep.
    """

    num_struct_hallus_to_keep: Optional[int] = 0
    num_organic_hallus_to_keep: Optional[int] = 0


def create_lora_argument_parser() -> argparse.ArgumentParser:
    """Create an argument parser for LoRA and PEFT configuration. This is a workaround the fact that the original LoraConfig is not compatible with HfArgumentParser due to the use of complex type hints.

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
        default="google/gemma-2-27b-it",
        metadata={
            "help": "The model name or path to the model to use as evaluator."
        },
    )

    use_gemini: bool = field(
        default=True,
        metadata={"help": "Using the latest Gemini model as evaluator"},
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


def setup_gsheets(
    sheet_name: str,
    worksheet_name: str,
):
    try:
        gc = gspread.service_account()  # assumes JSON key at ~/.config/gspread
        sh = gc.open(sheet_name)
    except Exception as e:
        print("Failed to open Google Sheet:", e)
        return None

    try:
        ws = sh.worksheet(worksheet_name)
    except Exception as e:
        print(f"Worksheet {worksheet_name} not found:", e)
        return None

    return ws


def add_row_to_gsheets(
    ws: Any,
    data: list,
    index: int,
):
    """Insert a row into the worksheet of the given Google Sheet."""
    try:
        ws.insert_row(data, index=index)
        print(f"Inserted row into at {index}")
        return None
    except Exception as e:
        print("Failed to insert row into worksheet:", e)
        return None


def find_insertion_index_rm(ws, data):
    """
    Find the correct index to insert a new row while maintaining sort order.

    Sort order (within matching Dataset/Model/Hallu.type):
    1. lora_r (ascending)
    2. lr (ascending) - for ties on lora_r
    3. batch (ascending) - for ties on both lora_r and lr
    4. epoch (ascending) - for ties on lora_r, lr, and batch

    Args:
        ws: The worksheet object
        data: List containing [Dataset, Model, Hallu. type, batch, lr, lora_r, epoch]

    Returns:
        int: The index where the new row should be inserted
    """
    # Get all rows from the worksheet (including header)
    all_rows = ws.get_all_values()

    # Extract the values we're matching on
    target_dataset = data[0]
    target_model = data[1]
    target_hallu_type = data[2]
    target_batch = data[8]
    target_lr = data[9]
    target_lora_r = data[10]
    target_epoch = data[7]

    # Start from row 2 (skip header at row 1)
    insertion_index = len(all_rows) + 1  # Default to end if no match found

    for i, row in enumerate(
        all_rows[1:], start=2
    ):  # Start enumeration at 2 (row index)
        # Check if this row matches our Dataset, Model, and Hallu. type
        if (
            row[0] == target_dataset
            and row[1] == target_model
            and row[2] == target_hallu_type
        ):
            # Convert to appropriate types for comparison
            row_lora_r = int(row[10]) if row[10] else 0
            row_lr = float(row[9].replace(",", ".")) if row[9] else 0.0
            row_batch = int(row[8]) if row[8] else 0
            row_epoch = int(row[7]) if row[7] else 0

            # Compare based on sorting criteria
            # Insert before this row if our new row should come first
            if target_lora_r < row_lora_r:
                insertion_index = i
                break
            elif target_lora_r == row_lora_r:
                if target_lr < row_lr:
                    insertion_index = i
                    break
                elif target_lr == row_lr:
                    if target_batch < row_batch:
                        insertion_index = i
                        break
                    elif target_batch == row_batch:
                        if target_epoch < row_epoch:
                            insertion_index = i
                            break

    # If we found matching rows but never broke, insert after all matching rows
    # If no matching rows found, insertion_index stays at end
    if insertion_index == len(all_rows) + 1:
        # Find the last row with matching Dataset/Model/Hallu.type
        for i in range(len(all_rows), 1, -1):  # Count backwards from end
            row = all_rows[i - 1]
            if (
                row[0] == target_dataset
                and row[1] == target_model
                and row[2] == target_hallu_type
            ):
                insertion_index = i + 1
                break

    return insertion_index


def find_insertion_index_sft(ws, data):
    """
    Find the correct index to insert a new row while maintaining sort order.

    Sort order (within matching Dataset/Model/Hallu.type):
    1. lora_r (ascending)
    2. lr (ascending) - for ties on lora_r
    3. batch (ascending) - for ties on both lora_r and lr
    4. epoch (ascending) - for ties on lora_r, lr, and batch

    Args:
        ws: The worksheet object
        data: List containing [Dataset, Model, Hallu. type, batch, lr, lora_r, epoch]

    Returns:
        int: The index where the new row should be inserted
    """
    # Get all rows from the worksheet (including header)
    all_rows = ws.get_all_values()

    # Extract the values we're matching on
    target_dataset = data[0]
    target_model = data[1]
    target_batch = data[4]
    target_lr = data[5]
    target_lora_r = data[6]
    target_epoch = data[3]

    # Start from row 2 (skip header at row 1)
    insertion_index = len(all_rows) + 1  # Default to end if no match found

    for i, row in enumerate(
        all_rows[1:], start=2
    ):  # Start enumeration at 2 (row index)
        # Check if this row matches our Dataset, Model, and Hallu. type
        if row[0] == target_dataset and row[1] == target_model:
            # Convert to appropriate types for comparison
            row_lora_r = int(row[6]) if row[6] else 0
            row_lr = float(row[5].replace(",", ".")) if row[5] else 0.0
            row_batch = int(row[4]) if row[4] else 0
            row_epoch = int(row[3]) if row[3] else 0

            # Compare based on sorting criteria
            # Insert before this row if our new row should come first
            if target_lora_r < row_lora_r:
                insertion_index = i
                break
            elif target_lora_r == row_lora_r:
                if target_lr < row_lr:
                    insertion_index = i
                    break
                elif target_lr == row_lr:
                    if target_batch < row_batch:
                        insertion_index = i
                        break
                    elif target_batch == row_batch:
                        if target_epoch < row_epoch:
                            insertion_index = i
                            break

    # If we found matching rows but never broke, insert after all matching rows
    # If no matching rows found, insertion_index stays at end
    if insertion_index == len(all_rows) + 1:
        # Find the last row with matching Dataset/Model/Hallu.type
        for i in range(len(all_rows), 1, -1):  # Count backwards from end
            row = all_rows[i - 1]
            if row[0] == target_dataset and row[1] == target_model:
                insertion_index = i + 1
                break

    return insertion_index


def update_gsheets_rm(
    ws: Any,
    args: ScriptArguments,
    lora_args: argparse.Namespace,
    training_args: TrainingArguments,
    trainer: Any,
):
    """Build experiment metadata from pipeline args and insert into GSheets.

    This wraps the lower-level `insert_rm_row_to_gsheets` and extracts
    Dataset, Model, Hallu.type, numeric training params, and WandB URL.
    """
    try:
        sheet_name = getattr(args, "gsheets_name", None)
        if not sheet_name:
            return

        task_map = {"npov": "NPOV", "bosch": "Bosch", "ragtruth": "RAGTruth"}
        dataset_value = task_map.get(args.task_name, args.task_name)
        model_repo = args.model_repo_id.split("/")[0]
        model_map = {"google": "Gemma", "mistralai": "Mistral"}
        model_value = model_map.get(model_repo, model_repo)

        try:
            hallu_suffix = args.dataset_repo_id.split("_")[-1]
        except Exception:
            hallu_suffix = ""
        hallu_map = {"organic": "Organic", "struct": "Structured"}
        hallu_value = hallu_map.get(hallu_suffix, hallu_suffix)

        organic_kept = getattr(args, "num_organic_hallus_to_keep", None)
        struct_kept = getattr(args, "num_struct_hallus_to_keep", None)

        seed = int(training_args.seed)
        epochs = int(training_args.num_train_epochs)
        batch = int(training_args.per_device_train_batch_size)
        lr = float(training_args.learning_rate)
        lora_r = int(getattr(lora_args, "lora_r", 0))

        wandb_url = None
        try:
            if getattr(wandb, "run", None) is not None:
                try:
                    wandb_url = wandb.run.get_url()
                except Exception:
                    wandb_url = None
        except Exception:
            wandb_url = None

        try:
            log_hist = getattr(trainer.state, "log_history", [])
            last_auc = None
            last_eval_loss = None
            for entry in reversed(log_hist):
                if last_auc is None and "eval_roc_auc" in entry:
                    last_auc = entry.get("eval_roc_auc")
                if last_eval_loss is None and "eval_loss" in entry:
                    last_eval_loss = entry.get("eval_loss")
                if last_auc is not None and last_eval_loss is not None:
                    break
            if last_auc is not None:
                AUC_value = float(last_auc)
            if last_eval_loss is not None:
                eval_loss_value = float(last_eval_loss)
        except Exception:
            pass

        row = [
            dataset_value,
            model_value,
            hallu_value,
            organic_kept if organic_kept and int(organic_kept) != 0 else "",
            struct_kept if struct_kept and int(struct_kept) != 0 else "",
            "",
            int(seed),
            int(epochs),
            int(batch),
            float(lr),
            int(lora_r),
            AUC_value or "",
            eval_loss_value or "",
            wandb_url or "",
            "",
            "Done",
        ]

        # Find the index in which to add this new row
        index = find_insertion_index_rm(ws, row)

        add_row_to_gsheets(ws, row, index)

        return None

    except Exception as e:
        print("Failed to prepare or insert GSheets row:", e)


def update_gsheets_sft(
    ws: Any,
    args: ScriptArguments,
    lora_args: argparse.Namespace,
    training_args: TrainingArguments,
    trainer: Any,
):
    """Build experiment metadata from pipeline args and insert into GSheets."""
    try:
        sheet_name = getattr(args, "gsheets_name", None)
        if not sheet_name:
            return

        task_map = {"npov": "NPOV", "bosch": "Bosch", "ragtruth": "RAGTruth"}
        dataset_value = task_map.get(args.task_name, args.task_name)
        model_repo = args.model_repo_id.split("/")[0]
        model_map = {"google": "Gemma", "mistralai": "Mistral"}
        model_value = model_map.get(model_repo, model_repo)

        seed = int(training_args.seed)
        epochs = int(training_args.num_train_epochs)
        batch = int(training_args.per_device_train_batch_size)
        lr = float(training_args.learning_rate)
        lora_r = int(getattr(lora_args, "lora_r", 0))

        wandb_url = None
        try:
            if getattr(wandb, "run", None) is not None:
                try:
                    wandb_url = wandb.run.get_url()
                except Exception:
                    wandb_url = None
        except Exception:
            wandb_url = None

        row = [
            dataset_value,
            model_value,
            int(seed),
            int(epochs),
            int(batch),
            float(lr),
            int(lora_r),
            "",  # Temperature during evaluation
            "",  # Hallucination rate during evaluation
            wandb_url or "",
            "",  # Generations link
            "",  # Comments
            "In Progress",
        ]

        # Find the index in which to add this new row
        index = find_insertion_index_sft(ws, row)

        add_row_to_gsheets(ws, row, index)

        return None

    except Exception as e:
        print("Failed to prepare or insert GSheets row:", e)
