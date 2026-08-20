"""Experimental pipelines for studying terminal token mismatch in RM training and PE-RL.

This module implements:
1. TokenMismatchRewardModelPipeline: Configurable terminal token for RM training (<eos>, <end_of_turn>, none, custom).
2. TokenMismatchPERLPipeline: Configurable terminal token for PE-RL rollout reward scoring.
"""

from __future__ import annotations

import argparse
from typing import Any, Optional, Sequence

try:
  import datasets
  from datasets import Value, load_dataset
except ImportError:
  datasets = None
  Value = None
  load_dataset = None

try:
  import numpy as np
except ImportError:
  np = None

try:
  import scipy.special
except ImportError:
  scipy = None

try:
  import evaluate
except ImportError:
  evaluate = None

try:
  import wandb
except ImportError:
  wandb = None

try:
  import torch
except ImportError:
  torch = None

try:
  from transformers import (
      AutoModelForCausalLM,
      AutoModelForSequenceClassification,
      AutoTokenizer,
      DataCollatorWithPadding,
      HfArgumentParser,
      Trainer,
      TrainingArguments,
      set_seed,
  )
except ImportError:
  AutoModelForCausalLM = None
  AutoModelForSequenceClassification = None
  AutoTokenizer = None
  DataCollatorWithPadding = None
  HfArgumentParser = None
  Trainer = None
  TrainingArguments = None
  set_seed = lambda s: None

try:
  from trl import RLOOConfig, RLOOTrainer
except ImportError:
  RLOOConfig = None
  RLOOTrainer = None

try:
  from peft import PeftModel, get_peft_model, LoraConfig
except ImportError:
  PeftModel = None
  get_peft_model = None
  LoraConfig = None

from src.models import (
    Gemma4ForSequenceClassification,
    register_gemma4_for_sequence_classification,
)
from src.pipelines import (
    PERLPipeline,
    RewardModelPipeline,
    WandbResumptionCallback,
    _clean_cli_args,
)
from src.utils import (
    LLMSynthScriptArguments,
    LoraArguments,
    ScriptArguments,
    compute_best_roc_threshold,
    create_lora_argument_parser,
    get_task_processor,
)


def resolve_terminal_token_string(
    token_spec: Optional[str],
    tokenizer: Any = None,
) -> Optional[str]:
  """Resolves token specification ('eos', 'end_of_turn', 'none', or custom string) into a concrete token string."""
  if token_spec is None:
    return None
  normalized = str(token_spec).strip()
  if normalized.lower() in ("none", "raw", "as_is", "as_generated", ""):
    return None
  if normalized.lower() in ("eos", "<eos>"):
    eos_tok = getattr(tokenizer, "eos_token", None)
    return eos_tok if eos_tok else "<eos>"
  if normalized.lower() in ("end_of_turn", "<end_of_turn>", "eot"):
    return "<end_of_turn>"
  return normalized


def apply_terminal_token_formatting(
    text: str,
    target_token: Optional[str],
    strip_existing_terminal_tokens: bool = True,
) -> str:
  """Applies terminal token formatting to a text prompt/completion.

  Strips trailing whitespace and any known terminal delimiters (<eos>, <end_of_turn>, </s>, etc.)
  if requested, then appends target_token if specified.
  """
  if text is None:
    return ""
  cleaned = str(text).rstrip()
  if strip_existing_terminal_tokens:
    for tok in ("<end_of_turn>", "<eos>", "</s>", "<|endoftext|>", "<|im_end|>"):
      if cleaned.endswith(tok):
        cleaned = cleaned[: -len(tok)].rstrip()
        break
  if target_token:
    cleaned = f"{cleaned}{target_token}"
  return cleaned


class TokenMismatchRewardModelPipeline(RewardModelPipeline):
  """Reward Model training pipeline with configurable terminal token formatting.

  Allows training reward models to score explicitly on:
  - <eos> (default / previous implementation)
  - <end_of_turn>
  - none (raw untagged completion)
  - any custom token
  """

  def __init__(self):
    super().__init__()
    self.rm_terminal_token: str = "eos"

  def setup_arguments(self, *cli_args, **cli_kwargs) -> None:
    clean_args = _clean_cli_args(cli_args)
    parser_lora = create_lora_argument_parser()
    parser_lora.add_argument(
        "--rm_terminal_token",
        type=str,
        default="eos",
        help="Terminal token for RM training ('eos', 'end_of_turn', 'none', or custom string). Default: 'eos'.",
    )
    lora_args, remaining_args = parser_lora.parse_known_args(clean_args)
    self.rm_terminal_token = getattr(lora_args, "rm_terminal_token", "eos")

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
    if set_seed is not None:
      set_seed(training_args.seed)

    task = script_args.task_name or "rm"
    lr = training_args.learning_rate
    r = getattr(lora_args, "lora_r", None)
    epochs = getattr(training_args, "num_train_epochs", None)
    tok_tag = f"tok_{self.rm_terminal_token.replace('<', '').replace('>', '')}"
    run_name_parts = [f"{task}_RM", tok_tag, f"lr{lr:.1e}"]
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

  def process_data(self) -> None:
    if LoraConfig is not None:
      self._lora_config = LoraConfig(
          r=self._lora_args.lora_r,
          lora_alpha=self._lora_args.lora_alpha,
          lora_dropout=self._lora_args.lora_dropout,
          task_type=self._lora_args.task_type,
          peft_type=self._lora_args.peft_type,
      )

    if DataCollatorWithPadding is not None:
      data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
      self.data_collator = data_collator

    if self.training_args.fp16:
      self.torch_dtype = torch.float16 if torch else None
    elif self.training_args.bf16:
      self.torch_dtype = torch.bfloat16 if torch else None
    else:
      raise Exception("Not training in mixed precision!")

    processor_cls = get_task_processor(self.args.task_name)
    self.data["train"] = processor_cls.augment_training_split(
        self.data["train"],
        self._llm_synth_args,
        self.training_args,
        self.args.dataset_repo_id,
    )

    resolved_tok = resolve_terminal_token_string(
        self.rm_terminal_token, self.tokenizer
    )

    def format_terminal_token(example):
      formatted_prompt = apply_terminal_token_formatting(
          example["prompt"],
          target_token=resolved_tok,
          strip_existing_terminal_tokens=True,
      )
      return {"prompt": formatted_prompt}

    # Tokenize and cast labels
    def encode(examples):
      return self.tokenizer(
          examples["prompt"],
          padding=True,
          truncation=True,
          return_tensors="pt",
      )

    for split in self.data.keys():
      self.data[split] = self.data[split].map(format_terminal_token)
      self.data[split] = self.data[split].map(encode, batched=True)
      self.data[split].set_format("torch")

      if Value is not None:
        new_features = self.data[split].features.copy()
        new_features["label"] = Value("int32")
        self.data[split] = self.data[split].cast(new_features)


class TokenMismatchPERLPipeline(PERLPipeline):
  """PERL Pipeline with configurable terminal token handling for rollout reward scoring.

  Allows scoring policy completions under:
  - 'as_generated' (exact text produced by policy, typically ending in <end_of_turn> for Gemma)
  - 'end_of_turn' (ensures completion ends in <end_of_turn>)
  - 'eos' (ensures completion ends in <eos>, simulating matched token scoring if RM was trained on <eos>)
  - 'none' (strips trailing special tokens)
  """

  def __init__(self):
    super().__init__()
    self.scoring_terminal_token: str = "end_of_turn"
    self.force_terminal_token_swap: bool = False

  def setup_arguments(self, *cli_args, **cli_kwargs) -> None:
    clean_args = _clean_cli_args(cli_args)
    parser_custom = argparse.ArgumentParser(add_help=False)
    parser_custom.add_argument(
        "--scoring_terminal_token",
        type=str,
        default="end_of_turn",
        help="Terminal token for rollout scoring ('end_of_turn', 'eos', 'as_generated', 'none', or custom string). Default: 'end_of_turn'.",
    )
    parser_custom.add_argument(
        "--force_terminal_token_swap",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Whether to explicitly replace existing terminal token with scoring_terminal_token.",
    )
    custom_args, remaining_args = parser_custom.parse_known_args(clean_args)
    self.scoring_terminal_token = custom_args.scoring_terminal_token
    self.force_terminal_token_swap = custom_args.force_terminal_token_swap

    parser = HfArgumentParser((ScriptArguments, RLOOConfig))
    clean_remaining = [a for a in remaining_args if a != "--"]
    script_args, training_args = parser.parse_args_into_dataclasses(clean_remaining)
    self.args = script_args
    self.training_args = training_args
    if set_seed is not None:
      set_seed(training_args.seed)

    task = script_args.task_name or "perl"
    lr = training_args.learning_rate
    beta = getattr(training_args, "beta", None)
    temp = getattr(training_args, "temperature", None)
    epochs = getattr(training_args, "num_train_epochs", None)
    tok_tag = f"score_tok_{self.scoring_terminal_token.replace('<', '').replace('>', '')}"
    run_name_parts = [f"{task}_PERL", tok_tag, f"lr{lr:.1e}"]
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

  def setup_trainer(self) -> None:
    reward_model = self.reward_model
    reward_tokenizer = self.reward_tokenizer
    scoring_token = self.scoring_terminal_token
    force_swap = self.force_terminal_token_swap

    resolved_tok = resolve_terminal_token_string(
        scoring_token, reward_tokenizer
    )

    def reward_fn(
        prompts: list[str], completions: list[str], **kwargs
    ) -> list[float]:
      formatted_texts = []
      for p, c in zip(prompts, completions):
        if scoring_token.lower() in ("as_generated", "raw", "as_is"):
          formatted_c = c
        else:
          formatted_c = apply_terminal_token_formatting(
              c,
              target_token=resolved_tok,
              strip_existing_terminal_tokens=force_swap or (scoring_token.lower() != "as_generated"),
          )
        formatted_texts.append(p + formatted_c)

      inputs = reward_tokenizer(
          formatted_texts,
          padding=True,
          truncation=True,
          max_length=512,
          return_tensors="pt",
      ).to(reward_model.device)

      with torch.no_grad():
        is_zero3 = any(hasattr(param, "ds_id") for param in reward_model.parameters())
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

    if RLOOTrainer is not None:
      self.trainer = RLOOTrainer(
          model=self.policy,
          reward_funcs=reward_fn,
          args=self.training_args,
          train_dataset=self.data["train"],
          eval_dataset=self.data.get("test", None),
          processing_class=self.tokenizer,
          callbacks=[WandbResumptionCallback()],
      )
