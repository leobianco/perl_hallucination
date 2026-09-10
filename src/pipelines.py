"""Defines an abstract Pipeline class and concrete pipeline implementations for SFT (writer_sft), reward model training, and PERL (rlhf) workflows.

Also defines EvaluationPipeline and concrete classes corresponding to each step
of the evaluation procedure (autorater evaluation, completion generation,
scoring of completions).

Each pipeline implements the sequence of steps described in the project:
1. setup_arguments
2. setup_tokenizer
3. load_data
4. process_data
5. setup_model
6. setup_trainer
7. run_and_save
"""

from __future__ import annotations

import abc
import concurrent.futures
import glob
import math
import os
import random
import re
import sys
import threading
import time
from typing import Any, Optional, Sequence

# Patch transformers heterogeneity configuration to allow global attribute access in vLLM
try:
  import transformers
  from transformers.configuration_utils import PretrainedConfig

  PretrainedConfig.allow_global_per_layer_attribute_access = True
  from transformers.integrations.heterogeneity.configuration_utils import (
      HeterogeneousPretrainedConfig,
  )

  HeterogeneousPretrainedConfig.allow_global_per_layer_attribute_access = True
except Exception:
  pass

from datasets import (
    Dataset,
    DatasetDict,
    Value,
    concatenate_datasets,
    load_dataset,
)
import evaluate
from google import genai
from google.genai import types
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import matplotlib.pyplot as plt
import numpy as np
from peft import LoraConfig, PeftModel, get_peft_model
import scipy.special
from sklearn.metrics import (
    RocCurveDisplay,
    precision_score,
    roc_auc_score,
)
from src.metrics import GenerationMetricsEvaluator
from src.models import (
    Gemma4ForSequenceClassification,
    register_gemma4_for_sequence_classification,
)
from src.utils import (
    EvalArguments,
    LLMSynthScriptArguments,
    LoraArguments,
    ScopeDataGenArguments,
    ScriptArguments,
    SsfoDataGenArguments,
    build_eval_dataset_repo_id,
    compact_model_name,
    compute_best_roc_threshold,
    create_lora_argument_parser,
    get_task_processor,
    sanitize_hf_repo_id,
)
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    LogitsProcessor,
    LogitsProcessorList,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import (
    DPOConfig,
    DPOTrainer,
    RLOOConfig,
    RLOOTrainer,
    SFTConfig,
    SFTTrainer,
)

try:
  from vllm import LLM, SamplingParams
  from vllm.lora.request import LoRARequest
  _VLLM_AVAILABLE = True
except ImportError:
  LLM = None
  SamplingParams = None
  LoRARequest = None
  _VLLM_AVAILABLE = False

try:
  import wandb
except ImportError:
  wandb = None


def _clean_cli_args(cli_args: Sequence[str] | None = None) -> list[str]:
  """Filter out stray '--' tokens from arguments before passing to argparse."""
  raw = list(cli_args) if cli_args else sys.argv[1:]
  return [arg for arg in raw if arg != "--"]


def parse_hf_repo_reference(
    ref: str,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
  """Parses a Hugging Face hub reference into (repo_id, subfolder, revision).

  Supports formats:
  - username/repo_name
  - username/repo_name/checkpoint-50
  - username/repo_name:checkpoint-50
  - username/repo_name@revision
  - username/repo_name/tree/revision/subfolder
  - https://huggingface.co/username/repo_name/...
  - hf://username/repo_name/...
  """
  if not isinstance(ref, str) or not ref.strip():
    return None, None, None

  cleaned = ref.strip()
  for prefix in (
      "https://huggingface.co/",
      "http://huggingface.co/",
      "hf://",
      "hf.co/",
  ):
    if cleaned.startswith(prefix):
      cleaned = cleaned[len(prefix) :]
      break

  revision = None
  subfolder = None

  if "@" in cleaned:
    cleaned, revision = cleaned.split("@", 1)

  if ":" in cleaned:
    cleaned, subfolder = cleaned.split(":", 1)

  parts = [p for p in cleaned.strip("/").split("/") if p]
  if not parts:
    return None, None, None

  if len(parts) == 1:
    return parts[0], subfolder, revision

  if "tree" in parts:
    tree_idx = parts.index("tree")
    repo_id = "/".join(parts[:tree_idx])
    if tree_idx + 1 < len(parts):
      revision = parts[tree_idx + 1]
    if tree_idx + 2 < len(parts):
      subfolder = "/".join(parts[tree_idx + 2 :])
    return repo_id, subfolder, revision

  repo_id = f"{parts[0]}/{parts[1]}"
  if len(parts) > 2:
    extra_path = "/".join(parts[2:])
    subfolder = extra_path if subfolder is None else f"{extra_path}/{subfolder}"

  return repo_id, subfolder, revision


class WandbResumptionCallback(TrainerCallback):
  """TrainerCallback that records the active WandB run ID to disk and HF Hub for seamless cross-machine resumption."""

  def on_train_begin(
      self,
      args: TrainingArguments,
      state: TrainerState,
      control: TrainerControl,
      **kwargs,
  ):
    if (
        getattr(state, "is_world_process_zero", True)
        and wandb is not None
        and getattr(wandb, "run", None) is not None
    ):
      try:
        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step")
        wandb.define_metric("eval/*", step_metric="train/global_step")
        wandb.define_metric("eval_*", step_metric="train/global_step")
      except Exception:
        pass

      run_id = wandb.run.id
      output_dir = getattr(args, "output_dir", None)
      if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        id_file = os.path.join(output_dir, "wandb_run_id.txt")
        try:
          with open(id_file, "w") as f:
            f.write(run_id)
        except Exception as e:
          print(f"[WandB] Notice: Could not write {id_file}: {e}")

        hub_model_id = getattr(args, "hub_model_id", None)
        push_to_hub = getattr(args, "push_to_hub", False)
        if push_to_hub and hub_model_id and os.path.isfile(id_file):
          try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=id_file,
                path_in_repo="wandb_run_id.txt",
                repo_id=hub_model_id,
                commit_message="Add wandb_run_id for cross-machine resumption",
            )
          except Exception:
            pass

  def on_save(
      self,
      args: TrainingArguments,
      state: TrainerState,
      control: TrainerControl,
      **kwargs,
  ):
    if (
        getattr(state, "is_world_process_zero", True)
        and wandb is not None
        and getattr(wandb, "run", None) is not None
    ):
      run_id = wandb.run.id
      output_dir = getattr(args, "output_dir", None)
      if output_dir:
        checkpoint_folder = f"checkpoint-{state.global_step}"
        checkpoint_dir = os.path.join(output_dir, checkpoint_folder)
        if os.path.isdir(checkpoint_dir):
          id_file = os.path.join(checkpoint_dir, "wandb_run_id.txt")
          try:
            with open(id_file, "w") as f:
              f.write(run_id)
          except Exception as e:
            print(f"[WandB] Notice: Could not write {id_file}: {e}")

        root_id_file = os.path.join(output_dir, "wandb_run_id.txt")
        try:
          with open(root_id_file, "w") as f:
            f.write(run_id)
        except Exception as e:
          print(f"[WandB] Notice: Could not write {root_id_file}: {e}")

        hub_model_id = getattr(args, "hub_model_id", None)
        push_to_hub = getattr(args, "push_to_hub", False)
        if push_to_hub and hub_model_id and os.path.isfile(root_id_file):
          try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=root_id_file,
                path_in_repo="wandb_run_id.txt",
                repo_id=hub_model_id,
                commit_message=(
                    f"Update wandb_run_id at step {state.global_step}"
                ),
            )
          except Exception:
            pass


class Pipeline(abc.ABC):
  """Abstract pipeline describing the training workflow.

  Subclasses should implement each step. The public method `run` executes
  the whole pipeline in order.
  """

  def __init__(self):
    self.args = None
    self.training_args = None
    self.tokenizer: Optional[AutoTokenizer] = None
    self.data: Optional[Any] = None
    self.model: Optional[torch.nn.Module] = None
    self.trainer: Optional[Any] = None
    self.enable_lora: bool = False
    self.lora_path: Optional[str] = None
    self.vllm_model: Optional[str] = None
    self.use_vllm: bool = False

  def run(self, *cli_args, **cli_kwargs) -> None:
    self.setup_arguments(*cli_args, **cli_kwargs)
    self.setup_tokenizer()
    self.load_data()
    self.process_data()
    self._configure_warmup_steps()
    self.setup_model()
    self.setup_trainer()
    self.run_and_save()

  def _setup_wandb_resumption(self) -> None:
    """Inspects resume_from_checkpoint (local or HF Hub) and output_dir to restore WandB run ID."""
    if self.training_args is None:
      return

    resume_ckpt = getattr(self.training_args, "resume_from_checkpoint", None)
    if not resume_ckpt or resume_ckpt in (
        False,
        "False",
        "false",
        "None",
        "none",
        "no",
    ):
      return

    is_auto = resume_ckpt in (True, "True", "true", "auto", "AUTO")
    if is_auto and os.environ.get("WANDB_SWEEP_ID"):
      return

    output_dir = getattr(self.training_args, "output_dir", None)
    hub_model_id = getattr(self.training_args, "hub_model_id", None)

    search_dirs = []
    if isinstance(resume_ckpt, str) and os.path.isdir(resume_ckpt):
      search_dirs.append(resume_ckpt)
      parent_dir = os.path.dirname(os.path.normpath(resume_ckpt))
      if parent_dir and os.path.isdir(parent_dir):
        search_dirs.append(parent_dir)
    elif is_auto and output_dir and os.path.isdir(output_dir):
      search_dirs.append(output_dir)

    # 1. Search local directories first
    for d in search_dirs:
      id_file = os.path.join(d, "wandb_run_id.txt")
      if os.path.isfile(id_file):
        try:
          with open(id_file, "r") as f:
            saved_run_id = f.read().strip()
          if saved_run_id:
            os.environ["WANDB_RUN_ID"] = saved_run_id
            os.environ["WANDB_RESUME"] = "allow"
            print(
                f"[WandB] Restored run ID '{saved_run_id}' with resume='allow'"
                f" from local: {id_file}"
            )
            return
        except Exception as e:
          print(f"[WandB] Notice: could not read {id_file}: {e}")

    # 2. Check Hugging Face Hub if resume_ckpt is a HF reference or hub_model_id is set
    hf_candidates = []
    if (
        isinstance(resume_ckpt, str)
        and not os.path.isdir(resume_ckpt)
        and not is_auto
    ):
      repo_id, subfolder, revision = parse_hf_repo_reference(resume_ckpt)
      if repo_id:
        hf_candidates.append((repo_id, subfolder, revision))
    elif is_auto and hub_model_id and isinstance(hub_model_id, str):
      repo_id, subfolder, revision = parse_hf_repo_reference(hub_model_id)
      if repo_id and (repo_id, subfolder, revision) not in hf_candidates:
        hf_candidates.append((repo_id, subfolder, revision))

    for repo_id, subfolder, revision in hf_candidates:
      try:
        filenames_to_try = []
        if subfolder:
          filenames_to_try.append(f"{subfolder}/wandb_run_id.txt")
        filenames_to_try.append("wandb_run_id.txt")

        for fname in filenames_to_try:
          try:
            downloaded_file = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                revision=revision,
            )
            if os.path.isfile(downloaded_file):
              with open(downloaded_file, "r") as f:
                saved_run_id = f.read().strip()
              if saved_run_id:
                os.environ["WANDB_RUN_ID"] = saved_run_id
                os.environ["WANDB_RESUME"] = "allow"
                print(
                    f"[WandB] Restored run ID '{saved_run_id}' with"
                    f" resume='allow' from Hugging Face Hub: {repo_id}/{fname}"
                )
                return
          except Exception:
            continue
      except Exception as e:
        print(
            "[WandB] Notice: Could not fetch wandb_run_id.txt from HF Hub"
            f" ({repo_id}): {e}"
        )

  def _download_hf_checkpoint(
      self,
      repo_id: str,
      subfolder: Optional[str] = None,
      revision: Optional[str] = None,
  ) -> Optional[str]:
    """Downloads a checkpoint snapshot from Hugging Face Hub and returns the local directory path."""
    try:
      print(
          f"Downloading checkpoint from Hugging Face Hub: repo_id='{repo_id}'"
          f"{f', subfolder={subfolder}' if subfolder else ''}"
          f"{f', revision={revision}' if revision else ''}..."
      )
      downloaded_dir = snapshot_download(
          repo_id=repo_id,
          revision=revision,
      )

      target_dir = (
          os.path.join(downloaded_dir, subfolder)
          if subfolder
          else downloaded_dir
      )

      if not os.path.isdir(target_dir):
        print(
            f"Subfolder '{subfolder}' not found in downloaded repo '{repo_id}'."
            f" Using root directory: {downloaded_dir}"
        )
        target_dir = downloaded_dir

      # Check if target_dir has nested checkpoint-XXX folders
      try:
        last_ckpt = get_last_checkpoint(target_dir)
      except Exception:
        last_ckpt = None

      if last_ckpt is not None:
        print(
            f"Found latest checkpoint in Hugging Face Hub repo '{repo_id}':"
            f" {last_ckpt}"
        )
        resolved_path = last_ckpt
      else:
        print(
            "Using checkpoint directory from Hugging Face Hub repo"
            f" '{repo_id}': {target_dir}"
        )
        resolved_path = target_dir

      # Update WandB resumption with any wandb_run_id.txt found in the downloaded files
      self._setup_wandb_resumption()

      return resolved_path

    except Exception as e:
      print(
          f"Error downloading checkpoint from Hugging Face Hub ({repo_id}): {e}"
      )
      return None

  def _resolve_lora_adapter_path(
      self,
      adapter_path_or_repo: Optional[str],
      base_model_id: Optional[str] = None,
  ) -> tuple[bool, Optional[str]]:
    """Resolves whether a model reference requires LoRA and its directory path containing adapter_config.json.

    Supports:
    - None or empty string -> (False, None)
    - Base model repository ID (equal to base_model_id) -> (False, None)
    - Local directory with adapter_config.json directly -> (True, local_dir)
    - Local run directory with checkpoint-XXX subfolders -> (True,
    checkpoint_dir)
    - Hugging Face Hub repository with adapter_config.json at root -> (True,
    downloaded_dir)
    - Hugging Face Hub repository with subfolders or checkpoint-XXX -> (True,
    target_dir)

    Returns:
        (enable_lora, resolved_adapter_path)
    """
    if not adapter_path_or_repo or not str(adapter_path_or_repo).strip():
      return False, None

    cleaned_ref = str(adapter_path_or_repo).strip()
    if base_model_id and cleaned_ref == str(base_model_id).strip():
      return False, None

    # 1. Check if it's an existing local directory
    if os.path.isdir(cleaned_ref):
      if os.path.isfile(os.path.join(cleaned_ref, "adapter_config.json")):
        return True, cleaned_ref
      try:
        last_ckpt = get_last_checkpoint(cleaned_ref)
      except Exception:
        last_ckpt = None
      if last_ckpt and os.path.isfile(
          os.path.join(last_ckpt, "adapter_config.json")
      ):
        return True, last_ckpt
      adapter_files = sorted(
          glob.glob(
              os.path.join(cleaned_ref, "**/adapter_config.json"),
              recursive=True,
          )
      )
      if adapter_files:
        return True, os.path.dirname(adapter_files[-1])
      return False, cleaned_ref

    # 2. Check Hugging Face Hub repo reference
    repo_id, subfolder, revision = parse_hf_repo_reference(cleaned_ref)
    if repo_id:
      try:
        print(
            "Resolving/downloading adapter checkpoint from HF Hub:"
            f" repo_id='{repo_id}'"
            f"{f', subfolder={subfolder}' if subfolder else ''}"
            f"{f', revision={revision}' if revision else ''}..."
        )
        downloaded_dir = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.bin"],
        )
        target_dir = (
            os.path.join(downloaded_dir, subfolder)
            if subfolder
            else downloaded_dir
        )
        if os.path.isdir(target_dir):
          if os.path.isfile(os.path.join(target_dir, "adapter_config.json")):
            return True, target_dir
          try:
            last_ckpt = get_last_checkpoint(target_dir)
          except Exception:
            last_ckpt = None
          if last_ckpt and os.path.isfile(
              os.path.join(last_ckpt, "adapter_config.json")
          ):
            return True, last_ckpt
          adapter_files = sorted(
              glob.glob(
                  os.path.join(target_dir, "**/adapter_config.json"),
                  recursive=True,
              )
          )
          if adapter_files:
            return True, os.path.dirname(adapter_files[-1])
          return False, target_dir
        return True, downloaded_dir
      except Exception as e:
        print(
            f"Warning: Could not resolve/download HF Hub repo '{repo_id}': {e}"
        )
        return False, None

    return False, None

  def _resolve_resume_checkpoint(self) -> Optional[str]:
    """Resolves the checkpoint path to resume from (supporting local paths, auto-discovery, and Hugging Face Hub repos)."""
    if self.training_args is None:
      return None

    resume_arg = getattr(self.training_args, "resume_from_checkpoint", None)
    if (
        resume_arg is None
        or resume_arg is False
        or str(resume_arg).lower() in ("false", "none", "0", "")
    ):
      return None

    output_dir = getattr(self.training_args, "output_dir", None)
    hub_model_id = getattr(self.training_args, "hub_model_id", None)

    # 1. Handle boolean / auto / latest flags
    if isinstance(resume_arg, bool) or str(resume_arg).lower() in (
        "true",
        "auto",
        "latest",
        "1",
    ):
      # First try local output_dir
      if output_dir and os.path.isdir(output_dir):
        try:
          last_ckpt = get_last_checkpoint(output_dir)
        except Exception:
          last_ckpt = None
        if last_ckpt is not None:
          print(f"Resuming training from latest local checkpoint: {last_ckpt}")
          return last_ckpt

      # If not found locally, try hub_model_id on Hugging Face Hub
      if hub_model_id and isinstance(hub_model_id, str):
        repo_id, subfolder, revision = parse_hf_repo_reference(hub_model_id)
        if repo_id:
          print(
              "No local checkpoint in output_dir. Attempting to download"
              f" latest checkpoint from Hugging Face Hub repo '{repo_id}'..."
          )
          hf_ckpt = self._download_hf_checkpoint(repo_id, subfolder, revision)
          if hf_ckpt is not None:
            return hf_ckpt

      print(
          f"No checkpoint found in output_dir '{output_dir}'. Starting"
          " training from scratch."
      )
      return None

    # 2. Handle string path or Hugging Face Hub repo reference
    if isinstance(resume_arg, str):
      # If it is an existing local directory:
      if os.path.isdir(resume_arg):
        try:
          last_ckpt = get_last_checkpoint(resume_arg)
        except Exception:
          last_ckpt = None
        if last_ckpt is not None and last_ckpt != resume_arg:
          print(
              f"Resuming training from checkpoint found in '{resume_arg}':"
              f" {last_ckpt}"
          )
          return last_ckpt
        return resume_arg

      # If it's not a local directory, check if it's a Hugging Face Hub path/repo
      repo_id, subfolder, revision = parse_hf_repo_reference(resume_arg)
      if repo_id:
        print(
            f"Fetching checkpoint from Hugging Face Hub: repo_id='{repo_id}'"
            f"{f', subfolder={subfolder}' if subfolder else ''}"
            f"{f', revision={revision}' if revision else ''}..."
        )
        hf_ckpt = self._download_hf_checkpoint(repo_id, subfolder, revision)
        if hf_ckpt is not None:
          return hf_ckpt

      print(
          f"Specified checkpoint '{resume_arg}' was not found locally or on"
          " Hugging Face Hub. Starting from scratch."
      )
      return None

    return None

  def _configure_warmup_steps(self) -> None:
    """Compute and set warmup_steps from warmup_ratio to avoid HF deprecation warnings."""
    if self.training_args is None:
      return

    warmup_ratio = getattr(self.training_args, "warmup_ratio", 0.0)
    if warmup_ratio is None or warmup_ratio <= 0:
      warmup_ratio = getattr(self.args, "warmup_ratio", 0.0) or 0.0

    if warmup_ratio > 0:
      if getattr(self.training_args, "warmup_steps", 0) <= 0:
        train_dataset = (
            self.data["train"]
            if isinstance(self.data, dict) and "train" in self.data
            else None
        )
        if train_dataset is not None:
          train_size = len(train_dataset)
          batch_size = getattr(
              self.training_args, "per_device_train_batch_size", 1
          )
          grad_accum = getattr(
              self.training_args, "gradient_accumulation_steps", 1
          )
          epochs = getattr(self.training_args, "num_train_epochs", 1)
          world_size = getattr(self.training_args, "world_size", 1)
          if world_size <= 0:
            world_size = (
                torch.cuda.device_count() if torch.cuda.is_available() else 1
            )
          steps_per_epoch = max(
              1, int(train_size / (batch_size * world_size * grad_accum))
          )
          total_steps = max(1, int(steps_per_epoch * epochs))
          self.training_args.warmup_steps = max(
              1, int(total_steps * warmup_ratio)
          )
      if hasattr(self.training_args, "warmup_ratio"):
        self.training_args.warmup_ratio = 0.0

  @abc.abstractmethod
  def setup_arguments(self, *cli_args, **cli_kwargs) -> None:
    raise NotImplementedError()

  def setup_tokenizer(self) -> None:
    """Default tokenizer loading uses self.args.model_repo_id or training args.

    Subclasses may override if they need a different tokenizer setup.
    """
    model_repo = getattr(self.args, "model_repo_id", None) or getattr(
        self.training_args, "sft_model_path", None
    )
    if model_repo is None:
      raise ValueError("No model repo specified for tokenizer setup")

    self.tokenizer = AutoTokenizer.from_pretrained(
        model_repo,
        padding_side="left",
    )

    # Set pad token if not present
    if self.tokenizer.pad_token_id is None:
      if self.tokenizer.eos_token_id is not None:
        self.tokenizer.pad_token = self.tokenizer.eos_token
      elif self.tokenizer.unk_token_id is not None:
        self.tokenizer.pad_token = self.tokenizer.unk_token

  @abc.abstractmethod
  def load_data(self) -> None:
    raise NotImplementedError()

  @abc.abstractmethod
  def process_data(self) -> None:
    raise NotImplementedError()

  @abc.abstractmethod
  def setup_model(self) -> None:
    raise NotImplementedError()

  @abc.abstractmethod
  def setup_trainer(self) -> None:
    raise NotImplementedError()

  @abc.abstractmethod
  def run_and_save(self) -> None:
    raise NotImplementedError()


class SFTPipeline(Pipeline):
  """Pipeline for supervised fine-tuning (writer_sft.py)."""

  def setup_arguments(self, *cli_args, **cli_kwargs) -> None:
    clean_args = _clean_cli_args(cli_args)
    parser_lora = create_lora_argument_parser()
    lora_args, remaining_args = parser_lora.parse_known_args(clean_args)
    clean_remaining = [a for a in remaining_args if a != "--"]
    parser = HfArgumentParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_into_dataclasses(
        clean_remaining
    )

    self.args = script_args
    self.training_args = training_args
    set_seed(training_args.seed)
    # keep lora config object around for setup_model
    self._lora_args = lora_args

    task = script_args.task_name or "sft"
    lr = training_args.learning_rate
    r = getattr(lora_args, "lora_r", None)
    epochs = getattr(training_args, "num_train_epochs", None)
    run_name_parts = [f"{task}_SFT", f"lr{lr:.1e}"]
    if r is not None:
      run_name_parts.append(f"r{r}")
    if epochs is not None:
      run_name_parts.append(f"epo{epochs}")
    run_name = "_".join(run_name_parts)
    if (
        not training_args.run_name
        or "sweep" in str(training_args.run_name).lower()
    ):
      self.training_args.run_name = run_name
    if wandb is not None and getattr(wandb, "run", None) is not None:
      wandb.run.name = run_name
    self._setup_wandb_resumption()

  def load_data(self) -> None:
    train_dataset = load_dataset(self.args.dataset_repo_id, split="train")
    if (
        getattr(self.args, "sft_data_fraction", None) is not None
        and 0.0 < self.args.sft_data_fraction < 1.0
    ):
      train_dataset = train_dataset.shuffle(seed=self.training_args.seed)
      num_samples = int(len(train_dataset) * self.args.sft_data_fraction)
      train_dataset = train_dataset.select(range(num_samples))

    self.data = {
        "train": train_dataset,
        "test": load_dataset(self.args.dataset_repo_id, split="test"),
    }

  def process_data(self) -> None:
    # lora config created from parsed args
    self._lora_config = LoraConfig(
        task_type=self._lora_args.task_type,
        peft_type=self._lora_args.peft_type,
        r=self._lora_args.lora_r,
        lora_alpha=self._lora_args.lora_alpha,
        lora_dropout=self._lora_args.lora_dropout,
    )

    processor_cls = get_task_processor(self.args.task_name)

    fewshot_examples = None
    if (
        self.args.task_name == "npov"
        and self.args.num_fewshot is not None
        and self.args.num_fewshot > 0
    ):
      fewshot_examples = (
          self.data["train"]
          .shuffle(seed=self.training_args.seed)
          .select(range(self.args.num_fewshot))
      )

    formatting_prompts_func, response_template = (
        processor_cls.get_formatting_prompts_and_response_template(
            eos_token=self.tokenizer.eos_token,
            fewshot_examples=fewshot_examples,
            model_repo_id=self.args.model_repo_id,
        )
    )

    self.formatting_prompts_func = formatting_prompts_func
    self.response_template = response_template

  def setup_model(self) -> None:
    model = AutoModelForCausalLM.from_pretrained(
        self.args.model_repo_id,
        torch_dtype=torch.bfloat16,
    )

    # Ensure pad token is set
    if self.tokenizer.pad_token_id is not None:
      model.config.pad_token_id = self.tokenizer.pad_token_id

    self.model = get_peft_model(model, self._lora_config)
    self.model.to(torch.bfloat16)

  def setup_trainer(self) -> None:
    self.trainer = SFTTrainer(
        model=self.model,
        args=self.training_args,
        train_dataset=self.data["train"],
        eval_dataset=self.data["test"],
        processing_class=self.tokenizer,
        callbacks=[WandbResumptionCallback()],
    )

  def run_and_save(self) -> None:
    resume_ckpt = self._resolve_resume_checkpoint()
    self.trainer.train(resume_from_checkpoint=resume_ckpt)
    if getattr(self.training_args, "push_to_hub", False):
      self.trainer.push_to_hub()


class RewardModelPipeline(Pipeline):
  """Pipeline for reward model training (reward_model.py)."""

  def setup_arguments(self, *cli_args, **cli_kwargs) -> None:
    clean_args = _clean_cli_args(cli_args)
    parser_lora = create_lora_argument_parser()
    lora_args, remaining_args = parser_lora.parse_known_args(clean_args)

    parser = HfArgumentParser(
        (ScriptArguments, LLMSynthScriptArguments, TrainingArguments)
    )
    clean_remaining = [a for a in remaining_args if a != "--"]
    script_args, llm_synth_args, training_args = (
        parser.parse_args_into_dataclasses(clean_remaining)
    )

    self.args = script_args
    self._llm_synth_args = llm_synth_args
    self.training_args = training_args
    self._lora_args = lora_args
    set_seed(training_args.seed)

    task = script_args.task_name or "rm"
    lr = training_args.learning_rate
    r = getattr(lora_args, "lora_r", None)
    epochs = getattr(training_args, "num_train_epochs", None)
    run_name_parts = [f"{task}_RM", f"lr{lr:.1e}"]
    if r is not None:
      run_name_parts.append(f"r{r}")
    if epochs is not None:
      run_name_parts.append(f"epo{epochs}")
    run_name = "_".join(run_name_parts)
    if (
        not training_args.run_name
        or "sweep" in str(training_args.run_name).lower()
    ):
      self.training_args.run_name = run_name
    if wandb is not None and getattr(wandb, "run", None) is not None:
      wandb.run.name = run_name
    self._setup_wandb_resumption()

  def load_data(self) -> None:
    self.data = load_dataset(self.args.dataset_repo_id)

  def process_data(self) -> None:
    self._lora_config = LoraConfig(
        r=self._lora_args.lora_r,
        lora_alpha=self._lora_args.lora_alpha,
        lora_dropout=self._lora_args.lora_dropout,
        task_type=self._lora_args.task_type,
        peft_type=self._lora_args.peft_type,
    )

    data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
    self.data_collator = data_collator

    if self.training_args.fp16:
      self.torch_dtype = torch.float16
    elif self.training_args.bf16:
      self.torch_dtype = torch.bfloat16
    else:
      raise Exception("Not training in mixed precision!")

    # Possibly augment dataset for synthetic_llm cases using the task
    # processor extension point.
    processor_cls = get_task_processor(self.args.task_name)

    self.data["train"] = processor_cls.augment_training_split(
        self.data["train"],
        self._llm_synth_args,
        self.training_args,
        self.args.dataset_repo_id,
    )

    # Tokenize and cast labels
    def encode(examples):
      return self.tokenizer(
          examples["prompt"],
          padding=True,
          truncation=True,
          return_tensors="pt",
      )

    for split in self.data.keys():
      self.data[split] = self.data[split].map(encode, batched=True)
      self.data[split].set_format("torch")

      new_features = self.data[split].features.copy()
      new_features["label"] = Value("int32")
      self.data[split] = self.data[split].cast(new_features)

  def setup_model(self) -> None:
    register_gemma4_for_sequence_classification(self.args.model_repo_id)
    id2label = {0: "Yes", 1: "No"}
    label2id = {"Yes": 0, "No": 1}

    try:
      self.model = AutoModelForSequenceClassification.from_pretrained(
          self.args.model_repo_id,
          num_labels=2,
          id2label=id2label,
          label2id=label2id,
          torch_dtype=self.torch_dtype,
      )
    except ValueError as e:
      if (
          "Gemma4Config" in str(e)
          or "gemma-4" in str(self.args.model_repo_id).lower()
          or "gemma4" in str(self.args.model_repo_id).lower()
      ):
        self.model = Gemma4ForSequenceClassification.from_pretrained(
            self.args.model_repo_id,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            torch_dtype=self.torch_dtype,
        )
      else:
        raise

    if self.tokenizer.pad_token_id is not None:
      self.model.config.pad_token_id = self.tokenizer.pad_token_id

    self.model = get_peft_model(self.model, self._lora_config)
    self.model.to(self.torch_dtype)

    # Adjust score parameters and ensure all modules_to_save heads
    # match torch_dtype
    score_head = getattr(self.model, "score", None)
    if score_head is None and hasattr(self.model, "base_model"):
      score_head = getattr(self.model.base_model, "score", None)
    if score_head is not None:
      score_head.to(self.torch_dtype)
      if hasattr(score_head, "modules_to_save"):
        for mod in score_head.modules_to_save.values():
          if hasattr(mod, "weight") and mod.weight is not None:
            mod.to(self.torch_dtype)
            mod.requires_grad_()
            with torch.no_grad():
              mod.weight.mul_(0.1)
      elif hasattr(score_head, "weight") and score_head.weight is not None:
        score_head.to(self.torch_dtype)
        score_head.requires_grad_()
        with torch.no_grad():
          score_head.weight.mul_(0.1)

  def setup_trainer(self) -> None:
    metric = evaluate.load("roc_auc")

    def compute_metrics(eval_preds):
      logits = eval_preds.predictions
      # Numerically stable softmax: probability of label 1 ("No" hallucination) is the reward score
      probs = scipy.special.softmax(logits, axis=-1)
      scores = probs[:, 1]
      label_ids = eval_preds.label_ids
      metrics = metric.compute(references=label_ids, prediction_scores=scores)
      # reuse helper from utils
      threshold_metrics = compute_best_roc_threshold(label_ids, scores)
      metrics.update(threshold_metrics)
      scores = np.array(scores)
      label_ids = np.array(label_ids)
      avg_score_true_positives = (
          scores[label_ids == 1].mean()
          if np.any(label_ids == 1)
          else float("nan")
      )
      avg_score_true_negatives = (
          scores[label_ids == 0].mean()
          if np.any(label_ids == 0)
          else float("nan")
      )
      metrics.update({
          "avg_score_true_positives": avg_score_true_positives,
          "avg_score_true_negatives": avg_score_true_negatives,
      })
      return metrics

    self.trainer = Trainer(
        model=self.model,
        args=self.training_args,
        train_dataset=self.data["train"],
        eval_dataset=self.data["test"],
        processing_class=self.tokenizer,
        data_collator=self.data_collator,
        compute_metrics=compute_metrics,
        callbacks=[WandbResumptionCallback()],
    )

  def run_and_save(self) -> None:
    resume_ckpt = self._resolve_resume_checkpoint()
    self.trainer.train(resume_from_checkpoint=resume_ckpt)
    if getattr(self.training_args, "output_dir", None):
      self.trainer.save_model(self.training_args.output_dir)
    if getattr(self.training_args, "push_to_hub", False) and getattr(
        self.training_args, "hub_model_id", None
    ):
      self.model.push_to_hub(self.training_args.hub_model_id)


class PERLPipeline(Pipeline):
  """Pipeline for PERL training (perl.py).

  This pipeline loads datasets, initializes policy and reward model,
  and uses RLOOTrainer with standard upstream TRL to run RL training.
  """

  def setup_arguments(self, *cli_args, **cli_kwargs) -> None:
    clean_args = _clean_cli_args(cli_args)
    parser = HfArgumentParser((ScriptArguments, RLOOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses(clean_args)
    self.args = script_args
    self.training_args = training_args
    set_seed(training_args.seed)

    task = script_args.task_name or "perl"
    lr = training_args.learning_rate
    beta = getattr(training_args, "beta", None)
    temp = getattr(training_args, "temperature", None)
    epochs = getattr(training_args, "num_train_epochs", None)
    run_name_parts = [f"{task}_PERL", f"lr{lr:.1e}"]
    if beta is not None:
      name_beta = f"{beta:.2g}" if beta >= 0.001 else f"{beta:.1e}"
      run_name_parts.append(f"beta{name_beta}")
    if temp is not None:
      run_name_parts.append(f"T{temp}")
    if epochs is not None:
      run_name_parts.append(f"epo{epochs}")
    run_name = "_".join(run_name_parts)
    if (
        not training_args.run_name
        or "sweep" in str(training_args.run_name).lower()
    ):
      self.training_args.run_name = run_name
    if wandb is not None and getattr(wandb, "run", None) is not None:
      wandb.run.name = run_name
    self._setup_wandb_resumption()

  def load_data(self) -> None:
    self.data = load_dataset(self.args.dataset_repo_id)

  def process_data(self) -> None:
    # In modern TRL RLOOTrainer, datasets only need a 'prompt' column
    # with raw text or standard format, and tokenization is handled by processing_class.
    pass

  def setup_model(self) -> None:
    id2label = {0: "Yes", 1: "No"}
    label2id = {"Yes": 0, "No": 1}

    reward_model_path = self.args.reward_model_path or getattr(
        self.training_args, "reward_model_path", None
    )
    sft_model_path = self.args.sft_model_path or getattr(
        self.training_args, "sft_model_path", None
    )

    register_gemma4_for_sequence_classification(reward_model_path)
    try:
      self.reward_model = AutoModelForSequenceClassification.from_pretrained(
          reward_model_path,
          num_labels=2,
          id2label=id2label,
          label2id=label2id,
          torch_dtype=torch.bfloat16,
      )
    except ValueError as e:
      if "Gemma4Config" in str(e) or (
          reward_model_path
          and (
              "gemma-4" in str(reward_model_path).lower()
              or "gemma4" in str(reward_model_path).lower()
          )
      ):
        self.reward_model = Gemma4ForSequenceClassification.from_pretrained(
            reward_model_path,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            torch_dtype=torch.bfloat16,
        )
      else:
        raise
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.reward_model.to(torch.bfloat16)
    self.reward_model.to(device)
    self.reward_model.eval()

    self.reward_tokenizer = AutoTokenizer.from_pretrained(
        reward_model_path,
        padding_side="right",
    )
    if self.reward_tokenizer.pad_token is None:
      self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
    if self.reward_tokenizer.pad_token_id is not None:
      self.reward_model.config.pad_token_id = self.reward_tokenizer.pad_token_id

    policy_base = AutoModelForCausalLM.from_pretrained(
        self.args.model_repo_id,
        torch_dtype=torch.bfloat16,
    )

    if self.tokenizer.pad_token_id is not None:
      policy_base.config.pad_token_id = self.tokenizer.pad_token_id

    self.policy = PeftModel.from_pretrained(
        policy_base, sft_model_path, is_trainable=True
    )
    self.policy.to(torch.bfloat16)

  def setup_trainer(self) -> None:
    reward_model = self.reward_model
    reward_tokenizer = self.reward_tokenizer

    # Custom reward function that computes scalar reward as the logit difference
    # (log-odds) of label 1 ("No" hallucination) vs label 0 ("Yes" hallucination).
    # This matches the standard Bradley-Terry reward formulation r(x, y) = z_1 - z_0
    # and prevents sigmoid saturation/vanishing gradients in RLOO when models are overconfident.
    def reward_fn(
        prompts: list[str], completions: list[str], **kwargs
    ) -> list[float]:
      texts = [p + c for p, c in zip(prompts, completions)]
      inputs = reward_tokenizer(
          texts,
          padding=True,
          truncation=True,
          max_length=512,
          return_tensors="pt",
      ).to(reward_model.device)
      with torch.no_grad():
        # Under DeepSpeed ZeRO Stage 3, all model parameters instantiated
        # in the process are partitioned into 1-D flat slices across GPUs.
        # Note on TRL's `ds3_gather_for_generation`: TRL's built-in flag
        # only gathers the policy model (`self.model`) during `generate()`.
        # Because `reward_model` is an auxiliary model called inside this
        # custom `reward_fn` callable, TRL does not manage or gather its
        # weights automatically.
        # If parameters are partitioned (detected via `ds_id`), we must
        # explicitly gather them across all ranks with `GatheredParameters`
        # (`modifier_rank=None`) so embedding and linear layers
        # reconstruct their 2-D shapes for inference.
        is_zero3 = any(hasattr(p, "ds_id") for p in reward_model.parameters())
        if is_zero3:
          import deepspeed

          with deepspeed.zero.GatheredParameters(
              list(reward_model.parameters()), modifier_rank=None
          ):
            logits = reward_model(**inputs).logits
        else:
          logits = reward_model(**inputs).logits

        # Logit difference: r(x, y) = logits[:, 1] - logits[:, 0]
        rewards = (logits[:, 1] - logits[:, 0]).cpu().tolist()
      return rewards

    self.trainer = RLOOTrainer(
        model=self.policy,
        reward_funcs=reward_fn,
        args=self.training_args,
        train_dataset=self.data["train"],
        eval_dataset=self.data.get("test", None),
        processing_class=self.tokenizer,
        callbacks=[WandbResumptionCallback()],
    )

  def run_and_save(self) -> None:
    if self.training_args.do_train:
      resume_ckpt = self._resolve_resume_checkpoint()
      self.trainer.train(resume_from_checkpoint=resume_ckpt)
      if getattr(self.training_args, "push_to_hub", False):
        self.trainer.push_to_hub()


class DPOPipeline(Pipeline):
  """Pipeline for Direct Preference Optimization (DPO) preference tuning.

  This pipeline loads a preference dataset (containing 'prompt', 'chosen', and
  'rejected' columns), initializes the policy model from the SFT checkpoint
  (using
  LoRA adapters), and fine-tunes using TRL's DPOTrainer.
  """

  def setup_arguments(self, *cli_args, **cli_kwargs) -> None:
    clean_args = _clean_cli_args(cli_args)
    parser_lora = create_lora_argument_parser()
    lora_args, remaining_args = parser_lora.parse_known_args(clean_args)
    clean_remaining = [a for a in remaining_args if a != "--"]
    parser = HfArgumentParser((ScriptArguments, DPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses(
        clean_remaining
    )

    self.args = script_args
    self.training_args = training_args
    set_seed(training_args.seed)
    self._lora_args = lora_args

    task = script_args.task_name or "dpo"
    lr = training_args.learning_rate
    beta = getattr(training_args, "beta", None)
    r = getattr(lora_args, "lora_r", None)
    epochs = getattr(training_args, "num_train_epochs", None)
    run_name_parts = [f"{task}_DPO", f"lr{lr:.1e}"]
    if beta is not None:
      name_beta = f"{beta:.2g}" if beta >= 0.001 else f"{beta:.1e}"
      run_name_parts.append(f"beta{name_beta}")
    if r is not None:
      run_name_parts.append(f"r{r}")
    if epochs is not None:
      run_name_parts.append(f"epo{epochs}")
    run_name = "_".join(run_name_parts)
    if (
        not training_args.run_name
        or "sweep" in str(training_args.run_name).lower()
    ):
      self.training_args.run_name = run_name
    if wandb is not None and getattr(wandb, "run", None) is not None:
      wandb.run.name = run_name
    self._setup_wandb_resumption()

  def load_data(self) -> None:
    data = load_dataset(self.args.dataset_repo_id)
    if isinstance(data, (DatasetDict, dict)):
      self.data = {
          "train": data["train"],
          "test": data.get("test", data.get("validation", None)),
      }
    else:
      self.data = {
          "train": load_dataset(self.args.dataset_repo_id, split="train"),
          "test": load_dataset(self.args.dataset_repo_id, split="test"),
      }

  def process_data(self) -> None:
    self._lora_config = LoraConfig(
        task_type=self._lora_args.task_type,
        peft_type=self._lora_args.peft_type,
        r=self._lora_args.lora_r,
        lora_alpha=self._lora_args.lora_alpha,
        lora_dropout=self._lora_args.lora_dropout,
    )

  def setup_model(self) -> None:
    sft_model_path = self.args.sft_model_path or getattr(
        self.training_args, "sft_model_path", None
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        self.args.model_repo_id,
        torch_dtype=torch.bfloat16,
    )
    if self.tokenizer.pad_token_id is not None:
      base_model.config.pad_token_id = self.tokenizer.pad_token_id

    enable_lora, resolved_adapter_path = self._resolve_lora_adapter_path(
        sft_model_path, self.args.model_repo_id
    )

    if enable_lora and resolved_adapter_path:
      # In SCOPE/SSFO, the policy is initialized at the SFT model p_{\theta_0}, and
      # the reference model is also p_{\theta_0}.
      # By loading and merging the SFT adapter into the base weights, then applying
      # a fresh LoRA adapter for DPO, DPOTrainer's adapter disabling automatically
      # evaluates the exact SFT reference model without extra memory overhead.
      sft_peft = PeftModel.from_pretrained(base_model, resolved_adapter_path)
      merged_base = sft_peft.merge_and_unload()
      self.model = get_peft_model(merged_base, self._lora_config)
    else:
      self.model = get_peft_model(base_model, self._lora_config)

    self.model.to(torch.bfloat16)
    self.ref_model = None

  def setup_trainer(self) -> None:
    self.trainer = DPOTrainer(
        model=self.model,
        ref_model=self.ref_model,
        args=self.training_args,
        train_dataset=self.data["train"],
        eval_dataset=self.data.get("test", None),
        processing_class=self.tokenizer,
        callbacks=[WandbResumptionCallback()],
    )

  def run_and_save(self) -> None:
    if self.training_args.do_train:
      resume_ckpt = self._resolve_resume_checkpoint()
      self.trainer.train(resume_from_checkpoint=resume_ckpt)
      if getattr(self.training_args, "push_to_hub", False):
        self.trainer.push_to_hub()


class ScopeMixtureLogitsProcessor(LogitsProcessor):
  """LogitsProcessor for SCOPE noisy decoding (Algorithm 1 / flbbb/scope-decoding).

  Injects unfaithful parametric noise by mixing or substituting conditional SFT
  logits
  with unconditional base model logits at decoding step t >= n_untouched_logits.
  """

  def __init__(
      self,
      main_input_length: int,
      noise_input_ids: torch.Tensor,
      noise_attention_mask: torch.Tensor,
      noise_model: torch.nn.Module,
      alpha: float = 0.5,
      sampling_mode: str = "bernoulli",
      n_untouched_logits: int = 2,
  ):
    super().__init__()
    self.main_input_length = main_input_length
    self.noise_input_ids = noise_input_ids
    self.noise_attention_mask = noise_attention_mask
    self.noise_model = noise_model
    self.alpha = alpha
    self.sampling_mode = sampling_mode
    self.n_untouched_logits = n_untouched_logits
    self.noise_past_key_values = None
    self.noise_cur_mask = noise_attention_mask

  def __call__(
      self, input_ids: torch.LongTensor, scores: torch.FloatTensor
  ) -> torch.FloatTensor:
    if self.alpha == 0.0:
      return scores

    with torch.no_grad():
      if self.noise_past_key_values is None:
        noise_out = self.noise_model(
            input_ids=self.noise_input_ids,
            attention_mask=self.noise_attention_mask,
            use_cache=True,
        )
      else:
        last_token = input_ids[:, -1:]
        self.noise_cur_mask = torch.cat(
            [
                self.noise_cur_mask,
                torch.ones(
                    (input_ids.shape[0], 1),
                    dtype=torch.long,
                    device=input_ids.device,
                ),
            ],
            dim=1,
        )
        noise_out = self.noise_model(
            input_ids=last_token,
            attention_mask=self.noise_cur_mask,
            past_key_values=self.noise_past_key_values,
            use_cache=True,
        )
      self.noise_past_key_values = noise_out.past_key_values
      logits_base = noise_out.logits[:, -1, :].clone()

    step = input_ids.shape[1] - self.main_input_length
    if step < self.n_untouched_logits:
      return scores

    # Mask invalid/special tokens
    inf_indices = torch.isinf(scores) | (scores <= -1e9)
    logits_base[inf_indices] = float("-inf")

    if self.sampling_mode in ("bernoulli", "hard"):
      alpha_mask = torch.bernoulli(
          torch.full((scores.shape[0], 1), self.alpha, device=scores.device)
      )
      processed_scores = torch.where(alpha_mask == 1.0, logits_base, scores)
    elif self.sampling_mode == "prob_mix":
      probs_sft = torch.softmax(scores, dim=-1)
      probs_base = torch.softmax(logits_base, dim=-1)
      mixed_probs = (1.0 - self.alpha) * probs_sft + self.alpha * probs_base
      processed_scores = torch.log(mixed_probs.clamp(min=1e-8))
    elif self.sampling_mode == "cad":
      uncond_scores = torch.softmax(logits_base, dim=-1)
      cond_scores = torch.softmax(scores, dim=-1)
      cad_scores = (1.0 + self.alpha) * cond_scores - self.alpha * uncond_scores
      cad_scores = torch.clamp(cad_scores, min=0.0)
      cad_probs = cad_scores / cad_scores.sum(dim=-1, keepdim=True).clamp(
          min=1e-8
      )
      processed_scores = torch.log(cad_probs.clamp(min=1e-8))
    elif self.sampling_mode == "logit_mix":
      processed_scores = (1.0 - self.alpha) * scores + self.alpha * logits_base
    else:
      processed_scores = scores

    return processed_scores


class ScopeDataGenerationPipeline(Pipeline):
  """Pipeline for generating synthetic preference datasets for SCOPE.

  Implements noisy decoding (Section 3 / Algorithm 1 of the SCOPE paper)
  by sampling mixed tokens between the fine-tuned SFT checkpoint and the
  pre-trained base model.
  """

  def setup_arguments(self, *cli_args, **cli_kwargs) -> None:
    clean_args = _clean_cli_args(cli_args)
    parser = HfArgumentParser(ScopeDataGenArguments)
    script_args = parser.parse_args_into_dataclasses(clean_args)[0]
    self.args = script_args
    set_seed(self.args.seed)

  def setup_tokenizer(self) -> None:
    self.tokenizer = AutoTokenizer.from_pretrained(
        self.args.model_repo_id,
        padding_side="left",
    )
    if self.tokenizer.pad_token_id is None:
      if self.tokenizer.eos_token_id is not None:
        self.tokenizer.pad_token = self.tokenizer.eos_token
      elif self.tokenizer.unk_token_id is not None:
        self.tokenizer.pad_token = self.tokenizer.unk_token

  def load_data(self) -> None:
    dataset = load_dataset(self.args.dataset_repo_id)
    self.raw_dataset = dataset

  def _extract_prompts(self, entry: dict) -> Tuple[str, str, str]:
    """Extract (prompt_with_context, prompt_without_context, ground_truth_chosen) from entry."""
    task_name = self.args.task_name
    processor_cls = get_task_processor(task_name)

    gt = entry.get(
        "npov_response",
        entry.get(
            "completion",
            entry.get("response", entry.get("chosen", "")),
        ),
    )

    if task_name == "npov":
      if "perspective_1" in entry and "perspective_2" in entry:
        p1_name = entry.get("perspective_1_name", "Perspective 1")
        p1 = entry.get("perspective_1", "")
        p2_name = entry.get("perspective_2_name", "Perspective 2")
        p2 = entry.get("perspective_2", "")
        user_query = entry.get("user_query", "")
        prompt_with_context = (
            f"User query: {user_query}\n{p1_name} arguments provided:"
            f" {p1}\n{p2_name} arguments provided: {p2}\nNeutral point-of-view"
            " answer to user query, rewriting provided arguments in natural"
            " language:\n"
        )
      elif "prompt" in entry and entry["prompt"]:
        prompt_raw = entry["prompt"]
        if gt and prompt_raw.endswith(gt):
          prompt_with_context = prompt_raw[: -len(gt)].rstrip() + "\n"
        else:
          prompt_with_context = prompt_raw
        user_query = entry.get("user_query", "")
        if not user_query and "User query: " in prompt_raw:
          after_query = prompt_raw.split("User query: ", 1)[1]
          user_query = after_query.split("\n", 1)[0].strip()
      else:
        prompt_with_context = str(entry)
        user_query = entry.get("user_query", "")

      prompt_without_context = (
          f"<start_of_turn>user\n{user_query}<end_of_turn>\n<start_of_turn>model\n"
      )
      return prompt_with_context, prompt_without_context, gt

    elif task_name == "bosch":
      if "prompt" in entry and entry["prompt"]:
        prompt_raw = entry["prompt"]
        if gt and prompt_raw.endswith(gt):
          prompt_with_context = prompt_raw[: -len(gt)].rstrip() + "\n"
        else:
          prompt_with_context = prompt_raw
        question = entry.get("Question", "")
        if not question and "User question:\n" in prompt_raw:
          question = (
              prompt_raw.split("User question:\n", 1)[1]
              .split("\nManual information:\n", 1)[0]
              .strip()
          )
      else:
        question = entry.get("Question", "")
        context = entry.get("Context", "")
        prompt_with_context = (
            "You are a helpful assistant to car related questions. You will be"
            " given an user's question, and the relevant part of the car"
            " manual. Your task is to answer the user's question using the"
            " information given. Do not add to your answer any information"
            " other than those present in the manual excerpt.\nUser"
            f" question:\n{question}\nManual information:\n{context}\nAnswer to"
            " user's question:\n"
        )
      prompt_without_context = (
          f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
      )
      return prompt_with_context, prompt_without_context, gt

    elif task_name == "ragtruth":
      if "prompt" in entry and entry["prompt"]:
        prompt_raw = entry["prompt"]
        if gt and prompt_raw.endswith(gt):
          prompt_with_context = prompt_raw[: -len(gt)].rstrip() + "\n"
        else:
          prompt_with_context = prompt_raw
        user_query = entry.get(
            "user_query", entry.get("query", entry.get("question", ""))
        )
        if not user_query and "Question: " in prompt_raw:
          user_query = (
              prompt_raw.split("Question: ", 1)[1]
              .split("\nAnswer:\n", 1)[0]
              .strip()
          )
      else:
        user_query = entry.get(
            "user_query", entry.get("query", entry.get("question", ""))
        )
        context = entry.get(
            "context", entry.get("passage", entry.get("document", ""))
        )
        if context:
          prompt_with_context = (
              f"Context: {context}\nQuestion: {user_query}\nAnswer:\n"
          )
        else:
          prompt_with_context = f"Question: {user_query}\nAnswer:\n"
      prompt_without_context = (
          f"<start_of_turn>user\n{user_query}<end_of_turn>\n<start_of_turn>model\n"
      )
      return prompt_with_context, prompt_without_context, gt

    else:
      if "prompt" in entry and entry["prompt"]:
        prompt_with_context = entry["prompt"]
      else:
        prompt_with_context = str(entry)
      raw_query = entry.get(
          "user_query",
          entry.get(
              "query",
              entry.get(
                  "question",
                  entry.get("prompt_no_context", prompt_with_context),
              ),
          ),
      )
      prompt_without_context = (
          f"<start_of_turn>user\n{raw_query}<end_of_turn>\n<start_of_turn>model\n"
      )
      return prompt_with_context, prompt_without_context, gt

  def _extract_prompt_and_chosen(self, entry: dict) -> Tuple[str, str]:
    """Extract standard prompt and chosen completion from dataset entry."""
    p_ctx, _, gt = self._extract_prompts(entry)
    return p_ctx, gt

  def process_data(self) -> None:
    train_split = (
        self.raw_dataset["train"]
        if isinstance(self.raw_dataset, dict)
        else self.raw_dataset
    )
    shuffled_train = train_split.shuffle(seed=self.args.seed)
    n_total = len(shuffled_train)
    n_d1 = int(n_total * self.args.split_ratio)
    # SCOPE uses D2 (second half) for preference generation
    d2_split = shuffled_train.select(range(n_d1, n_total))

    if self.args.max_samples is not None and self.args.max_samples > 0:
      d2_split = d2_split.select(
          range(min(len(d2_split), self.args.max_samples))
      )

    self.d2_split = d2_split

  def setup_model(self) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.device = device

    if self.tokenizer.pad_token_id is None:
      if self.tokenizer.eos_token_id is not None:
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
      elif self.tokenizer.unk_token_id is not None:
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
    self.tokenizer.padding_side = "left"

    print(f"Loading base pre-trained model: {self.args.model_repo_id}...")
    self.base_model = AutoModelForCausalLM.from_pretrained(
        self.args.model_repo_id,
        torch_dtype=torch.bfloat16,
    ).to(device)
    if self.tokenizer.pad_token_id is not None:
      self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
    self.base_model.eval()

    enable_lora, resolved_adapter_path = self._resolve_lora_adapter_path(
        self.args.sft_model_path, self.args.model_repo_id
    )
    self.enable_lora = enable_lora
    self.lora_path = resolved_adapter_path

    print(
        f"Loading SFT model from base={self.args.model_repo_id}"
        f" adapter={self.lora_path} (enable_lora={self.enable_lora})..."
    )
    sft_base = AutoModelForCausalLM.from_pretrained(
        self.args.model_repo_id,
        torch_dtype=torch.bfloat16,
    ).to(device)
    if self.tokenizer.pad_token_id is not None:
      sft_base.config.pad_token_id = self.tokenizer.pad_token_id

    if self.enable_lora and self.lora_path:
      print(
          f"Loading LoRA adapter from {self.lora_path} and merging weights..."
      )
      peft_model = PeftModel.from_pretrained(sft_base, self.lora_path)
      if hasattr(peft_model, "merge_and_unload"):
        self.sft_model = peft_model.merge_and_unload().to(device)
      else:
        self.sft_model = peft_model.to(device)
    else:
      self.sft_model = sft_base
    self.sft_model.eval()

  def setup_trainer(self) -> None:
    pass

  def _generate_unfaithful_samples_batch(
      self,
      prompts_with_ctx: list[str],
      prompts_without_ctx: list[str] | None = None,
  ) -> list[str]:
    """Generate dispreferred samples for a batch using SCOPE noisy decoding via sft_model.generate()."""
    if not prompts_with_ctx:
      return []

    if prompts_without_ctx is None or len(prompts_without_ctx) != len(
        prompts_with_ctx
    ):
      prompts_without_ctx = prompts_with_ctx

    tokenizer = self.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
      if tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
      elif tokenizer.unk_token_id is not None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

    device = self.device
    alpha = self.args.alpha
    temperature = self.args.temperature
    top_p = self.args.top_p
    top_k = self.args.top_k
    max_new_tokens = self.args.max_new_tokens
    sampling_mode = self.args.sampling_mode
    n_untouched = getattr(self.args, "n_untouched_logits", 2) or 0
    min_tokens = getattr(self.args, "min_tokens", 10) or 10

    sft_inputs = tokenizer(
        prompts_with_ctx,
        return_tensors="pt",
        padding=True,
    ).to(device)
    sft_input_ids = (
        sft_inputs.input_ids
        if hasattr(sft_inputs, "input_ids")
        else sft_inputs["input_ids"]
    )
    sft_attention_mask = (
        getattr(sft_inputs, "attention_mask", None)
        if hasattr(sft_inputs, "attention_mask")
        else sft_inputs.get("attention_mask", None)
    )

    base_inputs = tokenizer(
        prompts_without_ctx,
        return_tensors="pt",
        padding=True,
    ).to(device)
    base_input_ids = (
        base_inputs.input_ids
        if hasattr(base_inputs, "input_ids")
        else base_inputs["input_ids"]
    )
    base_attention_mask = (
        getattr(base_inputs, "attention_mask", None)
        if hasattr(base_inputs, "attention_mask")
        else base_inputs.get("attention_mask", None)
    )

    logits_processor = ScopeMixtureLogitsProcessor(
        main_input_length=sft_input_ids.shape[1],
        noise_input_ids=base_input_ids,
        noise_attention_mask=base_attention_mask,
        noise_model=self.base_model,
        alpha=alpha,
        sampling_mode=sampling_mode,
        n_untouched_logits=n_untouched,
    )

    gen_kwargs = {
        "input_ids": sft_input_ids,
        "attention_mask": sft_attention_mask,
        "logits_processor": LogitsProcessorList([logits_processor]),
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_tokens,
        "do_sample": temperature > 0.0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0.0:
      gen_kwargs["temperature"] = temperature
      if top_p < 1.0:
        gen_kwargs["top_p"] = top_p
      if top_k > 0:
        gen_kwargs["top_k"] = top_k

    with torch.no_grad():
      outputs = self.sft_model.generate(**gen_kwargs)

    prompt_length = sft_input_ids.shape[1]
    generated_tokens = outputs[:, prompt_length:]

    if hasattr(tokenizer, "batch_decode"):
      decoded = tokenizer.batch_decode(
          generated_tokens, skip_special_tokens=True
      )
    else:
      decoded = [
          tokenizer.decode(seq, skip_special_tokens=True)
          for seq in generated_tokens
      ]

    clean_completions = [
        d.replace("<end_of_turn>", "").replace("<eos>", "").strip()
        for d in decoded
    ]
    return clean_completions

  def _generate_unfaithful_sample(
      self,
      prompt: str,
  ) -> str:
    """Generate a single dispreferred sample (delegates to batch method)."""
    return self._generate_unfaithful_samples_batch([prompt])[0]

  def run_and_save(self) -> None:
    batch_size = getattr(self.args, "batch_size", 16) or 16
    prompts = []
    chosens = []
    rejecteds = []

    print(
        "Generating synthetic dispreferred completions for"
        f" {len(self.d2_split)} samples (batch_size={batch_size})..."
    )
    entries = list(self.d2_split)
    for i in tqdm(
        range(0, len(entries), batch_size), desc="SCOPE noisy decoding"
    ):
      batch_entries = entries[i : i + batch_size]
      batch_prompts_ctx = []
      batch_prompts_no_ctx = []
      batch_chosens = []
      for entry in batch_entries:
        p_ctx, p_no_ctx, gt = self._extract_prompts(entry)
        batch_prompts_ctx.append(p_ctx)
        batch_prompts_no_ctx.append(p_no_ctx)
        batch_chosens.append(gt)

      batch_rejecteds = self._generate_unfaithful_samples_batch(
          prompts_with_ctx=batch_prompts_ctx,
          prompts_without_ctx=batch_prompts_no_ctx,
      )
      prompts.extend(batch_prompts_ctx)
      chosens.extend(batch_chosens)
      rejecteds.extend(batch_rejecteds)

    pref_dict = {
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    }
    train_dataset = Dataset.from_dict(pref_dict)

    # Split generated synthetic preference dataset into 90% train, 10% validation
    if len(train_dataset) > 1:
      splits = train_dataset.train_test_split(
          test_size=0.1, seed=self.args.seed
      )
      preference_dataset = DatasetDict(
          {"train": splits["train"], "test": splits["test"]}
      )
    else:
      preference_dataset = DatasetDict(
          {"train": train_dataset, "test": train_dataset}
      )

    if self.args.output_dataset_repo_id:
      out_repo = sanitize_hf_repo_id(self.args.output_dataset_repo_id)
    else:
      model_part = compact_model_name(self.args.model_repo_id)
      alpha_val = str(getattr(self.args, "alpha", 0.5)).replace(".", "_")
      timestamp = time.strftime("%y%m%d%H%M")
      task = getattr(self.args, "task_name", "npov") or "npov"
      user_prefix = ""
      if "/" in self.args.dataset_repo_id:
        user_prefix = self.args.dataset_repo_id.split("/")[0] + "/"
      samples_tag = (
          f"_n{self.args.max_samples}" if self.args.max_samples else ""
      )
      out_repo = sanitize_hf_repo_id(
          f"{user_prefix}{task}_scope_preference_{model_part}_alpha_{alpha_val}{samples_tag}_{timestamp}"
      )
    if self.args.output_dir:
      os.makedirs(self.args.output_dir, exist_ok=True)
      preference_dataset.save_to_disk(self.args.output_dir)
      print(f"Preference dataset saved locally to {self.args.output_dir}")

    if self.args.push_to_hub:
      print(f"Pushing preference dataset to Hub: {out_repo}...")
      preference_dataset.push_to_hub(out_repo)


class SSFODataGenerationPipeline(Pipeline):
  """Pipeline for generating synthetic preference datasets for SSFO.

  Implements Self-Supervised Faithfulness Optimization (SSFO, arXiv:2508.17225)
  data generation by contrasting SFT model generations with context (chosen)
  and without context (rejected / hallucinated).
  """

  def setup_arguments(self, *cli_args, **cli_kwargs) -> None:
    clean_args = _clean_cli_args(cli_args)
    parser = HfArgumentParser(SsfoDataGenArguments)
    script_args = parser.parse_args_into_dataclasses(clean_args)[0]
    self.args = script_args
    self.enable_lora = False
    self.lora_path = None
    self.vllm_model = self.args.model_repo_id
    set_seed(self.args.seed)

  def setup_tokenizer(self) -> None:
    # Not needed for vLLM generation
    pass

  def load_data(self) -> None:
    dataset = load_dataset(self.args.dataset_repo_id)
    self.raw_dataset = dataset

  def _extract_prompts(self, entry: dict) -> Tuple[str, str, str]:
    """Extract (prompt_with_context, prompt_without_context, ground_truth_chosen) from entry."""
    task_name = self.args.task_name
    processor_cls = get_task_processor(task_name)

    gt = entry.get(
        "npov_response",
        entry.get(
            "completion",
            entry.get("response", entry.get("chosen", "")),
        ),
    )

    if task_name == "npov":
      if "prompt" in entry and entry["prompt"]:
        prompt_raw = entry["prompt"]
        if gt and prompt_raw.endswith(gt):
          prompt_with_context = prompt_raw[: -len(gt)].rstrip() + "\n"
        else:
          prompt_with_context = prompt_raw
        user_query = entry.get("user_query", "")
        if not user_query and "User query: " in prompt_raw:
          after_query = prompt_raw.split("User query: ", 1)[1]
          user_query = after_query.split("\n", 1)[0].strip()
      elif "perspective_1" in entry and "perspective_2" in entry:
        prompt_with_context = processor_cls._writer_prompt(entry, SFT=False)[
            "prompt"
        ]
        user_query = entry.get("user_query", "")
      else:
        prompt_with_context = str(entry)
        user_query = entry.get("user_query", "")

      prompt_without_context = (
          f"<start_of_turn>user\n{user_query}<end_of_turn>\n<start_of_turn>model\n"
      )
      return prompt_with_context, prompt_without_context, gt

    elif task_name == "bosch":
      if "prompt" in entry and entry["prompt"]:
        prompt_raw = entry["prompt"]
        if gt and prompt_raw.endswith(gt):
          prompt_with_context = prompt_raw[: -len(gt)].rstrip() + "\n"
        else:
          prompt_with_context = prompt_raw
        question = entry.get("Question", "")
        if not question and "User question:\n" in prompt_raw:
          question = (
              prompt_raw.split("User question:\n", 1)[1]
              .split("\nManual information:\n", 1)[0]
              .strip()
          )
      else:
        question = entry.get("Question", "")
        context = entry.get("Context", "")
        prompt_with_context = (
            "You are a helpful assistant to car related questions. You will be"
            " given an user's question, and the relevant part of the car"
            " manual. Your task is to answer the user's question using the"
            " information given. Do not add to your answer any information"
            " other than those present in the manual excerpt.\nUser"
            f" question:\n{question}\nManual information:\n{context}\nAnswer to"
            " user's question:\n"
        )
      prompt_without_context = (
          f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
      )
      return prompt_with_context, prompt_without_context, gt

    elif task_name == "ragtruth":
      if "prompt" in entry and entry["prompt"]:
        prompt_raw = entry["prompt"]
        if gt and prompt_raw.endswith(gt):
          prompt_with_context = prompt_raw[: -len(gt)].rstrip() + "\n"
        else:
          prompt_with_context = prompt_raw
        user_query = entry.get(
            "user_query", entry.get("query", entry.get("question", ""))
        )
        if not user_query and "Question: " in prompt_raw:
          user_query = (
              prompt_raw.split("Question: ", 1)[1]
              .split("\nAnswer:\n", 1)[0]
              .strip()
          )
      else:
        user_query = entry.get(
            "user_query", entry.get("query", entry.get("question", ""))
        )
        context = entry.get(
            "context", entry.get("passage", entry.get("document", ""))
        )
        if context:
          prompt_with_context = (
              f"Context: {context}\nQuestion: {user_query}\nAnswer:\n"
          )
        else:
          prompt_with_context = f"Question: {user_query}\nAnswer:\n"
      prompt_without_context = (
          f"<start_of_turn>user\n{user_query}<end_of_turn>\n<start_of_turn>model\n"
      )
      return prompt_with_context, prompt_without_context, gt

    else:
      if "prompt" in entry and entry["prompt"]:
        prompt_with_context = entry["prompt"]
      else:
        prompt_with_context = str(entry)
      raw_query = entry.get(
          "user_query",
          entry.get(
              "query",
              entry.get(
                  "question",
                  entry.get("prompt_no_context", prompt_with_context),
              ),
          ),
      )
      prompt_without_context = (
          f"<start_of_turn>user\n{raw_query}<end_of_turn>\n<start_of_turn>model\n"
      )
      return prompt_with_context, prompt_without_context, gt

  def process_data(self) -> None:
    train_split = (
        self.raw_dataset["train"]
        if isinstance(self.raw_dataset, dict)
        else self.raw_dataset
    )
    shuffled_train = train_split.shuffle(seed=self.args.seed)
    if self.args.max_samples is not None and self.args.max_samples > 0:
      shuffled_train = shuffled_train.select(
          range(min(len(shuffled_train), self.args.max_samples))
      )
    self.train_split = shuffled_train

  def setup_model(self) -> None:
    enable_lora, lora_path = self._resolve_lora_adapter_path(
        self.args.sft_model_path, self.args.model_repo_id
    )
    self.enable_lora = enable_lora
    self.lora_path = lora_path
    self.vllm_model = self.args.model_repo_id

  def setup_trainer(self) -> None:
    pass

  def _log_sample_inspection(
      self,
      prompts_with_ctx: list[str],
      prompts_without_ctx: list[str],
      chosens: list[str],
      rejecteds: list[str],
      gt_chosens: list[str] | None = None,
      max_samples_to_log: int = 5,
  ) -> None:
    """Logs detailed input prompts and model outputs for inspection."""
    num_samples = min(max_samples_to_log, len(prompts_with_ctx))
    if num_samples == 0:
      return

    print("\n" + "=" * 80)
    print(f"[SSFO Detailed Inspection - First {num_samples} Sample Pairs]")
    print("=" * 80)

    for i in range(num_samples):
      print(f"\n{'#' * 30} Sample {i + 1} / {len(prompts_with_ctx)} {'#' * 30}")
      print(
          "[1. Input Prompt WITH Context (for Chosen y+)] (Length:"
          f" {len(prompts_with_ctx[i])} chars):"
      )
      print("-" * 70)
      print(prompts_with_ctx[i])
      print("-" * 70)

      no_ctx_str = (
          prompts_without_ctx[i] if i < len(prompts_without_ctx) else "N/A"
      )
      print(
          "[2. Input Prompt WITHOUT Context (for Rejected y-)] (Length:"
          f" {len(no_ctx_str)} chars):"
      )
      print("-" * 70)
      print(no_ctx_str)
      print("-" * 70)

      chosen_str = chosens[i] if i < len(chosens) else "N/A"
      print(
          f"[3. Generated Chosen Completion (y+)] (Length: {len(chosen_str)}"
          " chars):"
      )
      print("-" * 70)
      print(chosen_str)
      print("-" * 70)

      rejected_str = rejecteds[i] if i < len(rejecteds) else "N/A"
      print(
          "[4. Generated Rejected Completion (y-)] (Length:"
          f" {len(rejected_str)} chars):"
      )
      print("-" * 70)
      print(rejected_str)
      print("-" * 70)

      if gt_chosens and i < len(gt_chosens) and gt_chosens[i]:
        print(f"[5. Ground Truth Reference Completion]:")
        print("-" * 70)
        print(gt_chosens[i])
        print("-" * 70)

    print("=" * 80 + "\n")

    # Also save up to 25 samples to a local text inspection file
    try:
      os.makedirs("logs", exist_ok=True)
      timestamp = time.strftime("%y%m%d%H%M")
      log_file = (
          f"logs/ssfo_inputs_and_outputs_{self.args.task_name}_{timestamp}.txt"
      )
      file_samples = min(25, len(prompts_with_ctx))
      with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"SSFO Data Generation Detailed Inspection Log\n")
        f.write(f"Task: {self.args.task_name}\n")
        f.write(f"Dataset: {self.args.dataset_repo_id}\n")
        f.write(f"Model: {self.args.model_repo_id}\n")
        f.write(f"SFT Adapter: {self.args.sft_model_path}\n")
        f.write(
            "Base Model for Rejected:"
            f" {getattr(self.args, 'use_base_model_for_rejected', True)}\n"
        )
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Samples: {len(prompts_with_ctx)}\n")
        f.write("=" * 80 + "\n\n")
        for i in range(file_samples):
          f.write(
              f"\n{'#' * 35} Sample {i + 1} / {len(prompts_with_ctx)}"
              f" {'#' * 35}\n"
          )
          f.write(f"[1. Input Prompt WITH Context (for Chosen y+)]:\n")
          f.write(prompts_with_ctx[i] + "\n\n")
          f.write(f"[2. Input Prompt WITHOUT Context (for Rejected y-)]:\n")
          f.write(
              (
                  prompts_without_ctx[i]
                  if i < len(prompts_without_ctx)
                  else "N/A"
              )
              + "\n\n"
          )
          f.write(f"[3. Generated Chosen Completion (y+)]:\n")
          f.write((chosens[i] if i < len(chosens) else "N/A") + "\n\n")
          f.write(f"[4. Generated Rejected Completion (y-)]:\n")
          f.write((rejecteds[i] if i < len(rejecteds) else "N/A") + "\n\n")
          if gt_chosens and i < len(gt_chosens) and gt_chosens[i]:
            f.write(f"[5. Ground Truth Reference Completion]:\n")
            f.write(gt_chosens[i] + "\n\n")
      print(
          f"[SSFO Inspection Log] Saved first {file_samples} sample"
          f" prompt-generation pairs to {log_file}"
      )
    except Exception as e:
      print(f"[SSFO Inspection Log] Could not save log file: {e}")

  def run_and_save(self) -> None:
    if not _VLLM_AVAILABLE:
      raise ImportError(
          "vLLM is required for SSFO data generation, but is not installed."
          " Please run in an environment with vLLM installed."
      )
    entries = list(self.train_split)
    all_prompt_ctx = []
    all_prompt_no_ctx = []
    all_gt_chosen = []

    for entry in entries:
      p_ctx, p_no_ctx, gt = self._extract_prompts(entry)
      all_prompt_ctx.append(p_ctx)
      all_prompt_no_ctx.append(p_no_ctx)
      all_gt_chosen.append(gt)

    print(
        f"[SSFO Dataset Input Summary] Total samples: {len(entries)}, "
        f"Columns: {list(entries[0].keys()) if entries else []}"
    )
    if all_prompt_ctx:
      print("\n" + "=" * 80)
      print(f"[SSFO Input Prompt Preview (Sample 1 of {len(all_prompt_ctx)})]")
      print("-" * 80)
      print("--- Prompt WITH Context (Passed to SFT Model for Chosen y+) ---")
      print(all_prompt_ctx[0])
      print("-" * 80)
      print(
          "--- Prompt WITHOUT Context (Passed to SFT Model for Rejected y-) ---"
      )
      print(all_prompt_no_ctx[0])
      if all_gt_chosen and all_gt_chosen[0]:
        print("-" * 80)
        print(f"--- Ground Truth Completion in Dataset (if any) ---")
        print(all_gt_chosen[0])
      print("=" * 80 + "\n")

    if not entries:
      print("Warning: Dataset is empty, creating empty preference dataset.")
      train_dataset = Dataset.from_dict({
          "prompt": [],
          "chosen": [],
          "rejected": [],
      })
    else:
      # Prepare sampling parameters for vLLM (matching EvaluationGenerationPipeline)
      top_k = self.args.top_k if self.args.top_k > 0 else -1
      rep_penalty = getattr(self.args, "repetition_penalty", 1.0) or 1.0
      sampling_params = SamplingParams(
          seed=self.args.seed,
          temperature=self.args.temperature,
          top_p=self.args.top_p,
          top_k=top_k,
          repetition_penalty=rep_penalty,
          min_tokens=getattr(self.args, "min_tokens", 10),
          max_tokens=self.args.max_new_tokens,
      )

      # Instantiate vLLM LLM (matching EvaluationGenerationPipeline)
      vllm_model = getattr(self, "vllm_model", None) or getattr(
          self.args, "model_repo_id", None
      )
      llm_kwargs = {
          "model": vllm_model,
          "enable_lora": getattr(self, "enable_lora", False),
          "max_lora_rank": 64,
          "dtype": "bfloat16",
          "hf_overrides": {"allow_global_per_layer_attribute_access": True},
      }
      if hasattr(self, "llm") and self.llm is not None:
        llm = self.llm
      else:
        try:
          llm = LLM(**llm_kwargs)
        except TypeError:
          llm_kwargs.pop("hf_overrides", None)
          llm = LLM(**llm_kwargs)

      lora_request = (
          LoRARequest("writer_lora_adapter", 1, lora_path=self.lora_path)
          if (self.enable_lora and self.lora_path is not None)
          else None
      )

      # Generate Chosen (y+) with context
      if self.args.use_ground_truth_chosen and all(all_gt_chosen):
        chosens = all_gt_chosen
      else:
        print(
            "Generating chosen completions with context for"
            f" {len(all_prompt_ctx)} samples..."
        )
        if lora_request is not None:
          outputs_ctx = llm.generate(
              all_prompt_ctx,
              sampling_params,
              lora_request=lora_request,
          )
        else:
          outputs_ctx = llm.generate(all_prompt_ctx, sampling_params)
        chosens = [output.outputs[0].text for output in outputs_ctx]

      # Generate Rejected (y-) without context
      use_base_for_rejected = getattr(
          self.args, "use_base_model_for_rejected", True
      )
      if use_base_for_rejected:
        print(
            "Generating rejected completions without context using BASE model"
            f" (no adapter) for {len(all_prompt_no_ctx)} samples..."
        )
        outputs_no_ctx = llm.generate(all_prompt_no_ctx, sampling_params)
      else:
        print(
            "Generating rejected completions without context using SFT adapter"
            f" for {len(all_prompt_no_ctx)} samples..."
        )
        if lora_request is not None:
          outputs_no_ctx = llm.generate(
              all_prompt_no_ctx,
              sampling_params,
              lora_request=lora_request,
          )
        else:
          outputs_no_ctx = llm.generate(all_prompt_no_ctx, sampling_params)
      rejecteds = [output.outputs[0].text for output in outputs_no_ctx]

      # Log inspection samples
      self._log_sample_inspection(
          prompts_with_ctx=all_prompt_ctx,
          prompts_without_ctx=all_prompt_no_ctx,
          chosens=chosens,
          rejecteds=rejecteds,
          gt_chosens=all_gt_chosen,
      )

      pref_dict = {
          "prompt": all_prompt_ctx,
          "chosen": chosens,
          "rejected": rejecteds,
      }
      train_dataset = Dataset.from_dict(pref_dict)

    # Split generated synthetic preference dataset into 90% train, 10% validation
    if len(train_dataset) > 1:
      splits = train_dataset.train_test_split(
          test_size=0.1, seed=self.args.seed
      )
      preference_dataset = DatasetDict(
          {"train": splits["train"], "test": splits["test"]}
      )
    else:
      preference_dataset = DatasetDict(
          {"train": train_dataset, "test": train_dataset}
      )

    if self.args.output_dataset_repo_id:
      out_repo = sanitize_hf_repo_id(self.args.output_dataset_repo_id)
    else:
      model_part = compact_model_name(self.args.model_repo_id)
      timestamp = time.strftime("%y%m%d%H%M")
      task = getattr(self.args, "task_name", "npov") or "npov"
      user_prefix = ""
      if "/" in self.args.dataset_repo_id:
        user_prefix = self.args.dataset_repo_id.split("/")[0] + "/"
      samples_tag = (
          f"_n{self.args.max_samples}" if self.args.max_samples else ""
      )
      out_repo = sanitize_hf_repo_id(
          f"{user_prefix}{task}_ssfo_preference_{model_part}{samples_tag}_{timestamp}"
      )
    if self.args.output_dir:
      os.makedirs(self.args.output_dir, exist_ok=True)
      preference_dataset.save_to_disk(self.args.output_dir)
      print(f"Preference dataset saved locally to {self.args.output_dir}")

    if self.args.push_to_hub:
      print(f"Pushing preference dataset to Hub: {out_repo}...")
      preference_dataset.push_to_hub(out_repo)


class EvaluationPipeline(Pipeline):
  """Base class for evaluator flows.

  Provides helper methods for autorater evaluation, completion generation, and
  scoring.
  """

  def setup_trainer(self) -> None:
    # Evaluation pipelines do not have trainers
    pass

  def safe_load_dataset(
      self, path: str, split: Optional[str] = None
  ) -> Dataset:
    """Loads dataset from Hugging Face Hub with parquet snapshot fallback."""
    target_split = split or "test"
    try:
      return load_dataset(path, split=target_split)
    except Exception as e:
      print(
          f"Notice: Standard load_dataset failed for {path}"
          f" (split={target_split}): {e}. Falling back to direct parquet"
          " download..."
      )
      local_dir = snapshot_download(
          repo_id=path,
          repo_type="dataset",
          allow_patterns=["*.parquet"],
      )
      split_parquet_files = sorted(
          glob.glob(
              os.path.join(local_dir, "**", f"{target_split}-*.parquet"),
              recursive=True,
          )
      )
      if split_parquet_files:
        parquet_files = split_parquet_files
      else:
        parquet_files = sorted(
            glob.glob(
                os.path.join(local_dir, "**", "*.parquet"), recursive=True
            )
        )
      if not parquet_files:
        raise ValueError(f"No parquet files found in dataset {path}") from e
      return Dataset.from_parquet(parquet_files)

  def _subsample_dataset(self, dataset: Dataset) -> Dataset:
    """Subsamples dataset to max_eval_samples using seed if specified."""
    max_samples = getattr(self.args, "max_eval_samples", None)
    if (
        max_samples is not None
        and max_samples > 0
        and len(dataset) > max_samples
    ):
      print(
          f"Subsampling evaluation dataset from {len(dataset)} to"
          f" {max_samples} samples (seed={self.args.seed})..."
      )
      return dataset.shuffle(seed=self.args.seed).select(range(max_samples))
    return dataset

  def get_fewshot_examples(
      self, data: Dataset, n_yes: int, n_no: int, seed: int
  ) -> Optional[Dataset]:
    if n_yes == 0 and n_no == 0:
      return None
    positive_examples = (
        data.filter(lambda entry: entry["class_hall"] == "Yes")
        .shuffle(seed=seed)
        .select(range(n_yes))
    )
    negative_examples = (
        data.filter(lambda entry: entry["class_hall"] == "No")
        .shuffle(seed=seed)
        .select(range(n_no))
    )
    fewshot_examples = (
        concatenate_datasets([positive_examples, negative_examples]).shuffle(
            seed=seed
        )
        if positive_examples is not None
        else None
    )
    return fewshot_examples

  def evaluator_score_batch(
      self,
      evaluator: AutoModelForCausalLM,
      tokenized_prompts: dict[str, torch.Tensor],
      yes_token_id: int,
      no_token_id: int,
  ) -> torch.Tensor:
    with torch.no_grad():
      outputs = evaluator(**tokenized_prompts, use_cache=False)
      score_yes = outputs.logits[:, -1, yes_token_id]
      score_no = outputs.logits[:, -1, no_token_id]
      # Numerically stable score computation: sigmoid(score_no - score_yes) = P(No)
      score_batch = torch.sigmoid(score_no - score_yes)
    return score_batch

  def evaluator_score(
      self,
      data: Dataset,
      script_args: ScriptArguments,
      tokenizer: AutoTokenizer,
      evaluator: AutoModelForCausalLM,
      yes_token_id: int,
      no_token_id: int,
  ) -> torch.Tensor:
    batch_size = getattr(script_args, "eval_batch_size", 32)
    iterator = data.iter(batch_size=batch_size)
    num_batches = (
        int(math.ceil(data.num_rows / batch_size))
        if hasattr(data, "num_rows")
        else None
    )
    all_scores: list[torch.Tensor] = []
    for batch in tqdm(iterator, desc="Evaluator scoring", total=num_batches):
      tokenized_prompts = tokenizer(
          batch["evaluator_prompt"],
          return_tensors="pt",
          padding="longest",
      )
      tokenized_prompts = {
          k: v.to(evaluator.device) for k, v in tokenized_prompts.items()
      }
      score_batch = self.evaluator_score_batch(
          evaluator, tokenized_prompts, yes_token_id, no_token_id
      )
      all_scores.append(score_batch.cpu())
      del tokenized_prompts
    if not all_scores:
      return torch.tensor([])
    return torch.cat(all_scores, dim=0)

  def create_gemini_client(self) -> genai.Client:
    """Initializes Google GenAI Client supporting Vertex AI or API Key."""
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in (
        "true",
        "1",
    ) or bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
    if use_vertex:
      project = os.environ.get("GOOGLE_CLOUD_PROJECT")
      location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
      print(
          "Initializing Google GenAI Client using Vertex AI"
          f" (project={project}, location={location})..."
      )
      return genai.Client(vertexai=True, project=project, location=location)
    elif getattr(self.args, "gemini_api_key", None):
      return genai.Client(api_key=self.args.gemini_api_key)
    else:
      return genai.Client()

  def gemini_score_response(
      self,
      response: types.GenerateContentResponse,
      entry_idx: Optional[int] = None,
  ) -> float:
    """Converts a Gemini API response to a normalized score.

    Extracts log probabilities from constrained enum decoding ('No' / 'Yes')
    across candidate steps, calculating P(No) = exp(logp_no) / (exp(logp_no) + exp(logp_yes)).
    Falls back to deterministic text classification if logprobs are unavailable,
    logging explicit warnings whenever a fallback path is taken.

    Args:
        response (google.genai.types.GenerateContentResponse): Gemini API
          response.
        entry_idx (Optional[int]): Sample index for descriptive logging.

    Returns:
        Optional[float]: Normalized score in [0.0, 1.0] for the response (P(No
        hallucination)), or None if scoring failed/unresolved.
    """
    prefix = (
        f"[Autorater fallback - sample #{entry_idx}]"
        if entry_idx is not None
        else "[Autorater fallback]"
    )

    if not response or not getattr(response, "candidates", None):
      print(
          f"{prefix} Empty response or blocked by safety filters (candidates is empty), "
          "no score assigned."
      )
      return None

    candidate = response.candidates[0]
    text = (response.text or "").strip() if getattr(response, "text", None) else ""
    clean_text = text.strip("\"'` \n\r\t").lower()

    # 1. Search for 'no' and 'yes' logprobs across all candidate steps
    if (
        hasattr(candidate, "logprobs_result")
        and candidate.logprobs_result is not None
    ):
      logprobs_res = candidate.logprobs_result
      top_candidates_steps = (
          getattr(logprobs_res, "top_candidates", None)
          or getattr(logprobs_res, "chosen_candidates", None)
          or []
      )

      for step in top_candidates_steps:
        candidates = (
            getattr(step, "candidates", None)
            or getattr(step, "top_candidates", None)
            or ([step] if hasattr(step, "token") else [])
        )
        logp_no = None
        logp_yes = None
        for cand in candidates:
          cand_token = (
              getattr(cand, "token", "").strip().strip("\"'`").lower()
          )
          logprob = getattr(cand, "log_prob", getattr(cand, "logprob", None))
          if cand_token == "no":
            logp_no = logprob
          elif cand_token == "yes":
            logp_yes = logprob

        if logp_no is not None and logp_yes is not None:
          p_no = float(math.exp(logp_no))
          p_yes = float(math.exp(logp_yes))
          denom = p_no + p_yes
          if denom > 0:
            return float(p_no / denom)
        elif logp_no is not None:
          print(
              f"{prefix} Only 'No' token found in candidate step (logprob={logp_no:.4f}, 'Yes' probability negligible), "
              "assigning 1.0."
          )
          return 1.0
        elif logp_yes is not None:
          print(
              f"{prefix} Only 'Yes' token found in candidate step (logprob={logp_yes:.4f}, 'No' probability negligible), "
              "assigning 0.0."
          )
          return 0.0

      # Check chosen_candidates as fallback if top_candidates was not structured
      chosen = getattr(logprobs_res, "chosen_candidates", None)
      if chosen:
        for cand in chosen:
          cand_token = (
              getattr(cand, "token", "").strip().strip("\"'`").lower()
          )
          if cand_token == "no":
            print(
                f"{prefix} Logprob alternatives missing, falling back to chosen token 'No' -> 1.0."
            )
            return 1.0
          elif cand_token == "yes":
            print(
                f"{prefix} Logprob alternatives missing, falling back to chosen token 'Yes' -> 0.0."
            )
            return 0.0

    # 2. Check avg_logprobs if available
    avg_logprob = getattr(candidate, "avg_logprobs", None)
    if isinstance(avg_logprob, (int, float)):
      if clean_text.startswith("no"):
        score = float(math.exp(avg_logprob))
        print(
            f"{prefix} Step logprobs missing, falling back to avg_logprobs for text '{text}' -> {score:.4f}."
        )
        return score
      elif clean_text.startswith("yes"):
        score = float(1.0 - math.exp(avg_logprob))
        print(
            f"{prefix} Step logprobs missing, falling back to avg_logprobs for text '{text}' -> {score:.4f}."
        )
        return score

    # 3. Fallback based on text classification
    if clean_text.startswith("no") or clean_text == "no":
      print(
          f"{prefix} Logprobs unavailable, falling back to text classification for response '{text}' -> 1.0."
      )
      return 1.0
    elif clean_text.startswith("yes") or clean_text == "yes":
      print(
          f"{prefix} Logprobs unavailable, falling back to text classification for response '{text}' -> 0.0."
      )
      return 0.0

    # Word-boundary search for "no" or "yes" in JSON or surrounding text
    match = re.search(r"\b(yes|no)\b", clean_text, re.IGNORECASE)
    if match:
      found = match.group(1).lower()
      score = 1.0 if found == "no" else 0.0
      print(
          f"{prefix} Logprobs unavailable, falling back to regex word match '{found}' for response '{text}' -> {score}."
      )
      return score

    print(
        f"{prefix} Unresolved response (response text: {repr(text[:80])}), "
        "no score assigned."
    )
    return None

  def gemini_score_dataset(
      self,
      client: genai.Client,
      dataset: Dataset,
      script_args: ScriptArguments,
  ) -> list[Optional[float]]:
    """Scores a dataset using the Gemini API, with checkpointing and retries.

    Args:
        client (google.genai.Client): Initialized Gemini API client.
        dataset (datasets.Dataset): Dataset with 'evaluator_prompt' column.
        script_args (ScriptArguments): Parsed script arguments.

    Returns:
        list[Optional[float]]: Scores for each entry in the dataset. Unscored
          entries are None.
    """
    model = getattr(script_args, "evaluator_model", None) or "gemini-2.5-flash"
    try:
      schema = types.Schema(
          type=types.Type.STRING,
          enum=["No", "Yes"],
      )
    except Exception:
      schema = {"type": "STRING", "enum": ["No", "Yes"]}
    print(f"Calling the Gemini API with model {model} (constrained decoding)...")

    # Checkpoint path is purely optional: only active if explicitly configured
    checkpoint_path = getattr(script_args, "scores_checkpoint_path", None)
    if checkpoint_path:
      checkpoint_dir = os.path.dirname(checkpoint_path)
      if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize scores: load from checkpoint if exists, else from dataset or None
    overwrite = getattr(script_args, "overwrite_scores", False)
    if not overwrite and checkpoint_path and os.path.exists(checkpoint_path):
      print(f"Loading scores from local checkpoint: {checkpoint_path}")
      loaded_scores = torch.load(checkpoint_path)
      if hasattr(loaded_scores, "tolist"):
        loaded_scores = loaded_scores.tolist()
      scores = list(loaded_scores)
      # If checkpoint is shorter than dataset (e.g. dataset updated), pad with None
      if len(scores) < len(dataset):
        scores = list(scores) + [None] * (len(dataset) - len(scores))
      elif len(scores) > len(dataset):
        scores = list(scores)[: len(dataset)]
    elif not overwrite and "scores" in dataset.column_names:
      print(
          "Found existing 'scores' column in dataset. Resuming from existing"
          " scores on Hugging Face Hub..."
      )
      raw_scores = list(dataset["scores"])
      if len(raw_scores) < len(dataset):
        raw_scores = list(raw_scores) + [None] * (len(dataset) - len(raw_scores))
      elif len(raw_scores) > len(dataset):
        raw_scores = list(raw_scores)[: len(dataset)]
      scores = raw_scores
    else:
      scores = [None] * len(dataset)

    # Normalize existing scores: ensure missing, nan, or non-numeric entries are strictly None
    for i in range(len(scores)):
      val = scores[i]
      if val is None:
        continue
      if isinstance(val, str) and val.strip().lower() in (
          "",
          "none",
          "nan",
          "null",
      ):
        scores[i] = None
      else:
        try:
          f_val = float(val)
          if math.isnan(f_val):
            scores[i] = None
          else:
            scores[i] = f_val
        except (ValueError, TypeError):
          scores[i] = None

    max_workers = getattr(script_args, "max_workers", 16) or 16
    save_frequency = 50  # Save progress every 50 entries
    server_retry_wait = 5  # Base seconds to wait between server error retries
    server_max_retries = 4  # Number of times to retry on server error
    entries_to_score = [i for i, score in enumerate(scores) if score is None]
    already_scored = len(scores) - len(entries_to_score)
    if already_scored > 0:
      print(
          f"Reusing {already_scored} already scored entries. "
          f"{len(entries_to_score)} missing entries remaining to score."
      )
    else:
      print(f"Found {len(entries_to_score)} entries to score.")

    if entries_to_score:
      print(
          f"Scoring {len(entries_to_score)} entries concurrently with"
          f" {max_workers} worker threads..."
      )

      lock = threading.Lock()
      use_logprobs_flag = [True]
      completed_count = 0

      def score_single_entry(idx: int) -> tuple[int, Optional[float]]:
        prompt_content = dataset[idx]["evaluator_prompt"]
        retry_count = 0
        while retry_count < server_max_retries:
          current_use_logprobs = True
          try:
            with lock:
              current_use_logprobs = use_logprobs_flag[0]

            config_kwargs = {
                "response_mime_type": "application/json",
                "response_schema": schema,
                "temperature": 0,
                "max_output_tokens": 10,
                "seed": script_args.seed,
            }

            if current_use_logprobs:
              config_kwargs["response_logprobs"] = True
              config_kwargs["logprobs"] = 5

            # Explicitly set zero thinking budget to ensure immediate
            # classification token
            try:
              if hasattr(types, "ThinkingConfig"):
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=0
                )
            except Exception:
              pass

            response = client.models.generate_content(
                model=model,
                contents=prompt_content,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            score = self.gemini_score_response(response, entry_idx=idx)
            return idx, score

          except Exception as e:
            error_str = str(e).lower()
            if "thinking" in error_str and "thinking_config" in config_kwargs:
              print(
                  f"[Autorater fallback - sample #{idx}] ThinkingConfig not"
                  f" supported by model/API ({error_str[:60]}),"
                  " retrying without thinking_config."
              )
              config_kwargs.pop("thinking_config", None)
              continue
            if "schema" in error_str and "response_schema" in config_kwargs:
              print(
                  f"[Autorater fallback - sample #{idx}] Response schema not"
                  f" supported by model/API ({error_str[:80]}), retrying with"
                  " unconstrained decoding."
              )
              config_kwargs.pop("response_schema", None)
              config_kwargs.pop("response_mime_type", None)
              continue
            if "logprob" in error_str and current_use_logprobs:
              print(
                  f"[Autorater fallback - sample #{idx}] Logprobs error"
                  f" ({error_str[:80]}), retrying in text-only mode."
              )
              with lock:
                use_logprobs_flag[0] = False
              continue
            if (
                "unavailable" in error_str
                or "overloaded" in error_str
                or "overcharged" in error_str
                or "server" in error_str
                or "resource_exhausted" in error_str
                or "429" in error_str
                or "503" in error_str
            ) and retry_count < server_max_retries - 1:
              backoff = server_retry_wait * (1.5**retry_count) + random.uniform(
                  0.5, 2.0
              )
              time.sleep(backoff)
              retry_count += 1
              continue
            else:
              print(
                  f"[Autorater failure - sample #{idx}] Gemini API failed with"
                  f" error: {error_str[:120]}. No score assigned."
              )
              return idx, None

        print(
            f"[Autorater failure - sample #{idx}] Failed to score entry after"
            f" {server_max_retries} attempts. No score assigned."
        )
        return idx, None

      try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
          future_to_idx = {
              executor.submit(score_single_entry, idx): idx
              for idx in entries_to_score
          }
          for future in tqdm(
              concurrent.futures.as_completed(future_to_idx),
              total=len(entries_to_score),
              desc="Scoring with Gemini API (parallel)",
          ):
            try:
              idx, score = future.result()
            except Exception as thread_err:
              idx = future_to_idx[future]
              score = None
              print(
                  f"[Autorater failure - sample #{idx}] Thread execution error:"
                  f" {thread_err}. No score assigned."
              )
            with lock:
              scores[idx] = score
              completed_count += 1
              if checkpoint_path and completed_count % save_frequency == 0:
                try:
                  torch.save(scores, checkpoint_path)
                except Exception as save_err:
                  print(
                      "Warning: Could not save intermediate progress locally:"
                      f" {save_err}"
                  )

      except Exception as e:
        print(f"\nError encountered during parallel scoring: {str(e)}")
        if checkpoint_path:
          print("Saving current progress locally...")
          try:
            torch.save(scores, checkpoint_path)
          except Exception as save_error:
            print(f"Error saving progress locally: {save_error}")
        raise e

    # Summary of score distribution
    n_total = len(scores)
    unscored_count = sum(1 for s in scores if s is None)
    scored_count = n_total - unscored_count

    scored_vals = [s for s in scores if s is not None]
    n_ones = sum(1 for s in scored_vals if s == 1.0)
    n_zeros = sum(1 for s in scored_vals if s == 0.0)
    n_probabilistic = scored_count - n_ones - n_zeros

    if unscored_count > 0:
      border = "=" * 70
      hf_repo = getattr(
          script_args, "dataset_with_completions", None
      ) or getattr(script_args, "dataset_labels", None)
      print(
          f"\n{border}\n"
          f"[WARNING] {unscored_count} out of {n_total} entries could not be"
          " scored by the autorater.\n"
          "These entries have NO score set (kept empty/None)."
      )
      if hf_repo:
        print(
            f"All progress is preserved directly on Hugging Face Hub:"
            f" {hf_repo}\n"
            "You can relaunch the evaluator on any VM to retry scoring only"
            " the missing entries\n"
            "without re-scoring already scored samples."
        )
      if checkpoint_path:
        print(f"Local checkpoint saved to: {checkpoint_path}")
        try:
          torch.save(scores, checkpoint_path)
        except Exception as save_err:
          print(f"Warning: Could not save scores checkpoint locally: {save_err}")
      print(f"{border}\n")
    else:
      print("\n[Autorater Scoring] All entries successfully scored!")
      # Clean up local checkpoint after scoring is complete
      if checkpoint_path and os.path.exists(checkpoint_path):
        try:
          os.remove(checkpoint_path)
        except Exception:
          pass

    print(
        f"[Autorater Scoring Complete] {scored_count}/{n_total} samples"
        f" scored: {n_probabilistic} continuous probabilities, {n_ones} No"
        f" (1.0), {n_zeros} Yes (0.0). ({unscored_count} entries"
        " unscored/empty)."
    )
    return scores


class EvaluationAutoraterPipeline(EvaluationPipeline):
  """Pipeline for the autorater evaluation (script_args.evaluate_evaluator == True)."""

  def setup_arguments(self, *cli_args, **cli_kwargs) -> None:
    parser = HfArgumentParser(EvalArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    self.args = script_args
    set_seed(self.args.seed)

  def setup_tokenizer(self) -> None:
    if not self.args.use_gemini:
      self.tokenizer = AutoTokenizer.from_pretrained(
          self.args.evaluator_model, padding_side="left"
      )

  def load_data(self) -> None:
    # Load dataset with hallucination labels
    if self.args.dataset_labels is None:
      raise ValueError("dataset_labels is required for autorater evaluation")

    self.data = self.safe_load_dataset(
        self.args.dataset_labels, split=self.args.dataset_labels_split
    )
    self.data = self._subsample_dataset(self.data)

  def process_data(self) -> None:
    # Map the evaluator prompt onto the dataset
    processor_cls = get_task_processor(self.args.task_name)

    prompt_fn = processor_cls.get_evaluator_prompt()
    # Map using fewshot examples if requested
    fewshot_examples = None
    if self.args.evaluator_num_fewshot and self.args.evaluator_num_fewshot > 0:
      # derive fewshot examples from the loaded dataset
      n_yes = self.args.evaluator_num_fewshot // 2
      n_no = self.args.evaluator_num_fewshot - n_yes
      fewshot_examples = self.get_fewshot_examples(
          self.data, n_yes, n_no, self.args.seed
      )

    self.data = self.data.map(
        prompt_fn,
        fn_kwargs={
            "fewshot_examples": fewshot_examples,
            "use_true_label": True,
        },
    )

  def setup_model(self) -> None:
    # Load evaluator as causal LM and set eval mode
    if not self.args.use_gemini:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.evaluator = AutoModelForCausalLM.from_pretrained(
          self.args.evaluator_model,
          torch_dtype=torch.bfloat16,
      ).to(device)
      self.evaluator.eval()

  def run_and_save(self) -> None:
    # Score using evaluator (either gemini or local model)
    if self.args.use_gemini:
      client = self.create_gemini_client()
      scores = self.gemini_score_dataset(client, self.data, self.args)
    else:
      # Tokenize in batches and compute scores using tokenizer token ids
      # for Yes/No
      yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
      no_token_id = self.tokenizer.convert_tokens_to_ids("No")
      scores = self.evaluator_score(
          self.data,
          self.args,
          self.tokenizer,
          self.evaluator,
          yes_token_id,
          no_token_id,
      )

    # Compute metrics
    ground_truth = self.data["label"]

    # Save evaluator prompts
    eval_model_name = (
        self.args.evaluator_model.split("/")[-1]
        if self.args.evaluator_model
        else "gemini"
    )
    dataset_name = (
        self.args.dataset_labels.split("/")[-1]
        if self.args.dataset_labels
        else "labels"
    )
    name_for_saving = (
        f"eval_autorater_{eval_model_name}"
        + f"_autorater_num_fewshot_{self.args.evaluator_num_fewshot}"
        + f"_data_{dataset_name}"
        + f"_seed_{self.args.seed}"
    )

    filepath = f"logs/{name_for_saving}/eval_autorater_evaluator_prompts.txt"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
      for idx, prompt in enumerate(self.data["evaluator_prompt"]):
        f.write(f"\n{idx}. ----------\n" + prompt)
    print(f"Evaluator prompts saved to {filepath}")

    # Save ground_truth labels
    filepath = f"logs/{name_for_saving}/eval_autorater_ground_truth.txt"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
      for label in ground_truth:
        f.write(f"{label}\n")
    print(f"Ground truth labels saved to {filepath}")

    # Save scores
    scores_list = (
        scores.tolist() if isinstance(scores, torch.Tensor) else list(scores)
    )
    scores_list = [
        float(s)
        if s is not None and not (isinstance(s, float) and math.isnan(s))
        else None
        for s in scores_list
    ]

    filepath = f"logs/{name_for_saving}/eval_autorater_scores.txt"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
      for score in scores_list:
        if score is None:
          f.write("None\n")
        else:
          f.write(f"{score:.5f}\n")
    print(f"Scores saved to {filepath}")

    valid_indices = [i for i, s in enumerate(scores_list) if s is not None]
    unscored_count = len(scores_list) - len(valid_indices)
    if unscored_count > 0:
      print(
          f"\n[WARNING] {unscored_count} out of {len(scores_list)} samples"
          " could not be scored. Excluding unscored samples from ROC-AUC and"
          " threshold metrics."
      )

    if not valid_indices:
      print(
          "\n[WARNING] No samples were successfully scored. Skipping ROC-AUC"
          " metric calculation."
      )
      return

    clean_gt = [ground_truth[i] for i in valid_indices]
    clean_scores = np.array(
        [scores_list[i] for i in valid_indices], dtype=np.float64
    )

    auc = roc_auc_score(clean_gt, clean_scores)
    metrics = compute_best_roc_threshold(clean_gt, clean_scores)
    threshold = metrics["best_threshold"]
    tpr = metrics["tpr_at_best_threshold"]
    fpr = metrics["fpr_at_best_threshold"]
    accuracy = metrics["accuracy_at_best_threshold"]
    classif_at_threshold = [
        0 if score < threshold else 1 for score in clean_scores
    ]

    metrics_filepath = f"logs/{name_for_saving}/eval_autorater_metrics.txt"
    os.makedirs(os.path.dirname(metrics_filepath), exist_ok=True)
    with open(metrics_filepath, "w") as f:
      f.write(f"Scored Samples: {len(valid_indices)}/{len(scores_list)}\n")
      f.write(f"Unscored Samples: {unscored_count}\n")
      f.write("AUC: {:.5f}\n".format(auc))
      f.write("Threshold: {:.5f}\n".format(threshold))
      f.write("TPR (recall): {:.5f}\n".format(tpr))
      f.write("FPR: {:.5f}\n".format(fpr))
      f.write("Accuracy: {:.5f}\n".format(accuracy))
      f.write(
          "Precision: {:.5f}\n".format(
              precision_score(clean_gt, classif_at_threshold)
          )
      )
    print(f"Metrics saved to {metrics_filepath}")

    print("AUC: {:.5f}".format(auc))
    print("Threshold: {:.5f}".format(threshold))
    print("TPR (recall): {:.5f}".format(tpr))
    print("FPR: {:.5f}".format(fpr))
    print("Accuracy: {:.5f}".format(accuracy))
    print(
        "Precision: {:.5f}".format(
            precision_score(clean_gt, classif_at_threshold)
        )
    )

    # ROC-AUC plot
    RocCurveDisplay.from_predictions(clean_gt, clean_scores)
    plt.title(f"ROC Curve (Threshold: {threshold:.5f})")
    plt.scatter([fpr], [tpr], c="r")
    os.makedirs(f"logs/{name_for_saving}/", exist_ok=True)
    plt.savefig(
        f"logs/{name_for_saving}/"
        + f"eval_autorater_auc_curve_{self.args.evaluator_num_fewshot}_shot"
    )
    plt.clf()

    # Histogram of scores
    bins = np.arange(0, 1, 0.05)
    scores_no = [
        clean_scores[idx]
        for idx, gt_val in enumerate(clean_gt)
        if gt_val == 1
    ]
    scores_yes = [
        clean_scores[idx]
        for idx, gt_val in enumerate(clean_gt)
        if gt_val == 0
    ]
    plt.vlines(x=threshold, ymin=0, ymax=175, colors="r")
    if scores_no:
      plt.hist(scores_no, bins=bins, alpha=0.5, label="No")
    if scores_yes:
      plt.hist(scores_yes, bins=bins, alpha=0.5, label="Yes")
    plt.legend()
    plt.savefig(
        f"logs/{name_for_saving}/"
        + f"eval_autorater_{self.args.evaluator_num_fewshot}_shot"
    )
    plt.clf()

class EvaluationGenerationPipeline(EvaluationPipeline):
  """Pipeline for generation: create completions with writer model."""

  def setup_arguments(self, *cli_args, **cli_kwargs) -> None:
    parser = HfArgumentParser(EvalArguments)
    self.args = parser.parse_args_into_dataclasses()[0]
    set_seed(self.args.seed)

  def setup_tokenizer(self) -> None:
    # This part of the evaluation pipeline does not need a tokenizer
    pass

  def load_data(self) -> None:
    self.dataset_prompts = self.safe_load_dataset(
        self.args.dataset_prompts, split=self.args.dataset_prompts_split
    )
    self.dataset_prompts = self._subsample_dataset(self.dataset_prompts)

  def process_data(self) -> None:
    # Prepare prompts and fewshot if requested
    prompts = [
        self.dataset_prompts[i]["prompt"]
        for i in range(self.dataset_prompts.num_rows)
    ]

    # Prepend few-shot examples if requested
    if (
        self.args.writer_num_fewshot > 0
        and self.args.dataset_labels is not None
    ):
      fewshot_data = self.safe_load_dataset(
          self.args.dataset_labels,
          split=self.args.dataset_labels_split,
      )
      fewshot_examples = self.get_fewshot_examples(
          fewshot_data,
          n_yes=0,
          n_no=self.args.writer_num_fewshot,
          seed=self.args.seed,
      )
      fewshot_prompts = "\n".join([ex["prompt"] for ex in fewshot_examples])
      prompts = [fewshot_prompts + "\n" + p for p in prompts]

    self.prompts = prompts

  def setup_model(self) -> None:
    enable_lora, lora_path = self._resolve_lora_adapter_path(
        self.args.writer_model_lora, self.args.writer_model_base
    )
    self.enable_lora = enable_lora
    self.lora_path = lora_path
    self.vllm_model = self.args.writer_model_base

  def run_and_save(self) -> None:
    if not _VLLM_AVAILABLE:
      raise ImportError(
          "vLLM is required for local completion generation, but is not"
          " installed. Please run generation on a GPU environment with"
          " vLLM installed."
      )
    # Prepare sampling parameters for vLLM
    top_k = self.args.top_k if self.args.top_k > 0 else -1
    sampling_params = SamplingParams(
        seed=self.args.seed,
        temperature=self.args.temperature,
        top_p=self.args.top_p,
        top_k=top_k,
        min_tokens=getattr(self.args, "min_tokens", 10),
        max_tokens=self.args.max_tokens,
    )

    # Instantiate vLLM LLM (use bfloat16 dtype and allow global attribute access)
    llm_kwargs = {
        "model": self.vllm_model,
        "enable_lora": self.enable_lora,
        "max_lora_rank": 64,
        "dtype": "bfloat16",
        "hf_overrides": {"allow_global_per_layer_attribute_access": True},
    }
    try:
      llm = LLM(**llm_kwargs)
    except TypeError:
      llm_kwargs.pop("hf_overrides", None)
      llm = LLM(**llm_kwargs)

    # Run generation
    if self.enable_lora and self.lora_path is not None:
      outputs = llm.generate(
          self.prompts,
          sampling_params,
          lora_request=LoRARequest(
              "writer_lora_adapter", 1, lora_path=self.lora_path
          ),
      )
    else:
      outputs = llm.generate(self.prompts, sampling_params)

    generations = [output.outputs[0].text for output in outputs]
    self.dataset_prompts = self.dataset_prompts.add_column(
        "completion", generations
    )
    if getattr(self.args, "dataset_with_completions", None):
      repo_id = sanitize_hf_repo_id(self.args.dataset_with_completions)
    else:
      repo_id = build_eval_dataset_repo_id(
          user=self.args.user,
          writer_model_lora=self.args.writer_model_lora,
          temperature=self.args.temperature,
          writer_num_fewshot=self.args.writer_num_fewshot,
      )

    print(f"Pushing dataset with generations to {repo_id}...")
    clean_dataset = Dataset.from_dict(self.dataset_prompts.to_dict())
    split_name = self.args.dataset_prompts_split or "test"
    dataset_dict = DatasetDict({split_name: clean_dataset})
    dataset_dict.push_to_hub(repo_id)


class EvaluationScoringPipeline(EvaluationPipeline):
  """Pipeline for scoring existing dataset with completions."""

  def setup_arguments(self, *cli_args, **cli_kwargs):
    parser = HfArgumentParser(EvalArguments)
    self.args = parser.parse_args_into_dataclasses()[0]
    set_seed(self.args.seed)

  def setup_tokenizer(self):
    if getattr(self.args, "run_autorater", True) and not self.args.use_gemini:
      self.tokenizer = AutoTokenizer.from_pretrained(
          self.args.evaluator_model, padding_side="left"
      )
    if getattr(self.args, "compute_perplexity", False):
      fluency_model_name = getattr(
          self.args, "fluency_model", None
      ) or getattr(self.args, "writer_model_base", None)
      if fluency_model_name:
        print(f"Loading fluency tokenizer: {fluency_model_name}...")
        self.fluency_tokenizer = AutoTokenizer.from_pretrained(
            fluency_model_name, padding_side="left"
        )
      else:
        self.fluency_tokenizer = None

  def load_data(self):
    if not self.args.dataset_with_completions and getattr(
        self.args, "writer_model_lora", None
    ):
      self.args.dataset_with_completions = build_eval_dataset_repo_id(
          user=self.args.user,
          writer_model_lora=self.args.writer_model_lora,
          temperature=self.args.temperature,
          writer_num_fewshot=self.args.writer_num_fewshot,
      )
    elif self.args.dataset_with_completions:
      self.args.dataset_with_completions = sanitize_hf_repo_id(
          self.args.dataset_with_completions
      )

    if self.args.dataset_with_completions is None:
      raise ValueError("dataset_with_completions is required for scoring mode")
    print(
        "Loading dataset with completions from"
        f" {self.args.dataset_with_completions}..."
    )
    self.full_dataset = self.safe_load_dataset(
        self.args.dataset_with_completions, split="test"
    )
    max_samples = getattr(self.args, "max_eval_samples", None)
    if (
        max_samples is not None
        and max_samples > 0
        and len(self.full_dataset) > max_samples
    ):
      print(
          f"Subsampling evaluation to {max_samples} out of"
          f" {len(self.full_dataset)} samples (seed={self.args.seed}). Full"
          " dataset completions will be preserved on hub."
      )
      rng = random.Random(self.args.seed)
      all_indices = list(range(len(self.full_dataset)))
      rng.shuffle(all_indices)
      self.eval_indices = all_indices[:max_samples]
      self.val_data = self.full_dataset.select(self.eval_indices)
      self.is_subsampled = True
    else:
      self.eval_indices = list(range(len(self.full_dataset)))
      self.val_data = self.full_dataset
      self.is_subsampled = False

  def process_data(self):
    if not getattr(self.args, "run_autorater", True):
      return

    processor_cls = get_task_processor(self.args.task_name)
    prompt_fn = processor_cls.get_evaluator_prompt()
    fewshot_examples = None
    if (
        self.args.evaluator_num_fewshot
        and self.args.evaluator_num_fewshot > 0
        and self.args.dataset_labels
    ):
      label_data = self.safe_load_dataset(
          self.args.dataset_labels, split=self.args.dataset_labels_split
      )
      n_yes = self.args.evaluator_num_fewshot // 2
      n_no = self.args.evaluator_num_fewshot - n_yes
      fewshot_examples = self.get_fewshot_examples(
          label_data, n_yes, n_no, self.args.seed
      )
    self.val_data = self.val_data.map(
        prompt_fn, fn_kwargs={"fewshot_examples": fewshot_examples}
    )

  def setup_model(self):
    if getattr(self.args, "run_autorater", True) and not self.args.use_gemini:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      # Use causal LM evaluator and set eval mode
      self.evaluator = AutoModelForCausalLM.from_pretrained(
          self.args.evaluator_model,
          torch_dtype=torch.bfloat16,
      ).to(device)
      self.evaluator.eval()

    if getattr(self.args, "compute_perplexity", False):
      fluency_model_name = getattr(
          self.args, "fluency_model", None
      ) or getattr(self.args, "writer_model_base", None)
      if fluency_model_name:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            "Loading fluency base model for perplexity evaluation:"
            f" {fluency_model_name}..."
        )
        self.fluency_model = AutoModelForCausalLM.from_pretrained(
            fluency_model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.fluency_model.eval()
      else:
        self.fluency_model = None

  def run_and_save(self):
    autorater_score_list = None
    if getattr(self.args, "run_autorater", True):
      # Score dataset using tokenizer and evaluator (support gemini)
      if self.args.use_gemini:
        client = self.create_gemini_client()
        scores = self.gemini_score_dataset(client, self.val_data, self.args)
      else:
        yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        no_token_id = self.tokenizer.convert_tokens_to_ids("No")
        scores = self.evaluator_score(
            self.val_data,
            self.args,
            self.tokenizer,
            self.evaluator,
            yes_token_id,
            no_token_id,
        )
      # Extract list of scores, properly handling both list and torch.Tensor
      scores_list = (
          scores.tolist() if isinstance(scores, torch.Tensor) else list(scores)
      )
      # Ensure unscored / missing entries are strictly None (no numeric fallback)
      scores_list = [
          float(s)
          if s is not None and not (isinstance(s, float) and math.isnan(s))
          else None
          for s in scores_list
      ]
      # Compute classifications: 1 for non-hallucination (>= threshold),
      # 0 for hallucination (< threshold), and None for entries that could
      # not be scored by the autorater.
      classifs_list = [
          (1 if s >= self.args.threshold else 0)
          if s is not None
          else None
          for s in scores_list
      ]
      if "scores" in self.val_data.column_names:
        self.val_data = self.val_data.remove_columns("scores")
      self.val_data = self.val_data.add_column("scores", scores_list)
      if "classifications" in self.val_data.column_names:
        self.val_data = self.val_data.remove_columns("classifications")
      self.val_data = self.val_data.add_column(
          "classifications", classifs_list
      )
      autorater_score_list = scores_list
    elif "scores" in self.val_data.column_names:
      autorater_score_list = [
          float(s)
          if s is not None and not (isinstance(s, float) and math.isnan(s))
          else None
          for s in self.val_data["scores"]
      ]

    # Compute generation metrics (ROUGE, BERTScore, lengths, PPL)
    if getattr(self.args, "compute_generation_metrics", True):
      evaluator = GenerationMetricsEvaluator(
          bertscore_model=getattr(
              self.args,
              "bertscore_model",
              "sentence-transformers/all-MiniLM-L6-v2",
          ),
          compute_bertscore_metric=getattr(
              self.args, "compute_bertscore", True
          ),
          compute_perplexity_metric=getattr(
              self.args, "compute_perplexity", False
          ),
          fluency_model=getattr(self, "fluency_model", None),
          fluency_tokenizer=getattr(self, "fluency_tokenizer", None),
      )
      self.val_data, summary = evaluator.evaluate_dataset(
          self.val_data,
          prompt_column="prompt",
          completion_column="completion",
          reference_column=getattr(self.args, "reference_column", None),
          tokenizer=getattr(
              self, "fluency_tokenizer", getattr(self, "tokenizer", None)
          ),
      )
      dataset_name = self.args.dataset_with_completions.split("/")[-1]
      evaluator.save_and_log_results(
          self.val_data,
          summary,
          output_dir=os.path.join("logs", "eval"),
          dataset_name=dataset_name,
          autorater_scores=autorater_score_list,
          threshold=self.args.threshold,
          log_to_wandb=getattr(self.args, "log_to_wandb", False),
          wandb_project=getattr(self.args, "wandb_project", "new_perl_eval"),
          wandb_run_name=getattr(self.args, "wandb_run_name", None),
      )

    if self.is_subsampled:
      # Merge scored subset columns back into full dataset, preserving all rows
      final_dict = (
          dict(self.full_dataset.to_dict())
          if hasattr(self.full_dataset, "to_dict")
          else {
              col: list(self.full_dataset[col])
              for col in self.full_dataset.column_names
          }
      )
      n_total = len(self.full_dataset)
      evaluated_metric_cols = [
          "scores",
          "classifications",
          "rouge1_f1",
          "rouge1_precision",
          "rouge1_recall",
          "rouge2_f1",
          "rouge2_precision",
          "rouge2_recall",
          "rougeL_f1",
          "rougeL_precision",
          "rougeL_recall",
          "bertscore_f1",
          "char_length",
          "token_length",
          "distinct_1",
          "distinct_2",
          "repetition_rate",
          "perplexity",
      ]
      for col_name in self.val_data.column_names:
        if col_name in evaluated_metric_cols or col_name not in final_dict:
          if col_name in final_dict:
            full_col_vals = list(final_dict[col_name])
          else:
            full_col_vals = [None] * n_total
          for idx_in_val, orig_idx in enumerate(self.eval_indices):
            full_col_vals[orig_idx] = self.val_data[col_name][idx_in_val]
          final_dict[col_name] = full_col_vals

      # Remove any stale reference metric columns from prior runs if not
      # evaluated now
      for metric_col in [
          "rouge1_f1",
          "rouge1_precision",
          "rouge1_recall",
          "rouge2_f1",
          "rouge2_precision",
          "rouge2_recall",
          "rougeL_f1",
          "rougeL_precision",
          "rougeL_recall",
          "bertscore_f1",
          "perplexity",
      ]:
        if (
            metric_col not in self.val_data.column_names
            and metric_col in final_dict
        ):
          final_dict.pop(metric_col, None)

      self.full_dataset = Dataset.from_dict(final_dict)
      print(
          f"Pushing full dataset ({len(self.full_dataset)} total samples,"
          f" {len(self.val_data)} scored) to"
          f" {self.args.dataset_with_completions}"
      )
      dataset_dict = DatasetDict({"test": self.full_dataset})
      dataset_dict.push_to_hub(self.args.dataset_with_completions)
    else:
      clean_dataset = (
          Dataset.from_dict(self.val_data.to_dict())
          if hasattr(self.val_data, "to_dict")
          else self.val_data
      )
      dataset_dict = DatasetDict({"test": clean_dataset})
      dataset_dict.push_to_hub(self.args.dataset_with_completions)
