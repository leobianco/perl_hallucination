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
import os
import time
from typing import Any, Optional

from datasets import (
    Dataset,
    Value,
    concatenate_datasets,
    load_dataset,
)
import evaluate
from google import genai
from google.genai import types
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
import numpy as np
from peft import LoraConfig, PeftModel, get_peft_model
import scipy.special
from sklearn.metrics import (
    RocCurveDisplay,
    precision_score,
    roc_auc_score,
)
from src.utils import (
    DpoScriptArguments,
    EvalArguments,
    ScopeDataGenArguments,
    SsfoDataGenArguments,
    LLMSynthScriptArguments,
    LoraArguments,
    ScriptArguments,
    compute_best_roc_threshold,
    create_lora_argument_parser,
    get_task_processor,
)
import torch
from tqdm import tqdm
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
from trl import (
    DPOConfig,
    DPOTrainer,
    RLOOConfig,
    RLOOTrainer,
    SFTConfig,
    SFTTrainer,
)
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


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

  def run(self, *cli_args, **cli_kwargs) -> None:
    self.setup_arguments(*cli_args, **cli_kwargs)
    self.setup_tokenizer()
    self.load_data()
    self.process_data()
    self.setup_model()
    self.setup_trainer()
    self.run_and_save()

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
    parser_lora = create_lora_argument_parser()
    lora_args, remaining_args = parser_lora.parse_known_args()
    parser = HfArgumentParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_into_dataclasses(
        remaining_args
    )

    self.args = script_args
    self.training_args = training_args
    set_seed(training_args.seed)
    # keep lora config object around for setup_model
    self._lora_args = lora_args

  def load_data(self) -> None:
    self.data = {
        "train": load_dataset(self.args.dataset_repo_id, split="train"),
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
    )

  def run_and_save(self) -> None:
    self.trainer.train()
    self.trainer.push_to_hub()


class RewardModelPipeline(Pipeline):
  """Pipeline for reward model training (reward_model.py)."""

  def setup_arguments(self, *cli_args, **cli_kwargs) -> None:
    parser_lora = create_lora_argument_parser()
    lora_args, remaining_args = parser_lora.parse_known_args()

    parser = HfArgumentParser(
        (ScriptArguments, LLMSynthScriptArguments, TrainingArguments)
    )
    script_args, llm_synth_args, training_args = (
        parser.parse_args_into_dataclasses(remaining_args)
    )

    self.args = script_args
    self._llm_synth_args = llm_synth_args
    self.training_args = training_args
    self._lora_args = lora_args
    set_seed(training_args.seed)

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
    id2label = {0: "Yes", 1: "No"}
    label2id = {"Yes": 0, "No": 1}

    self.model = AutoModelForSequenceClassification.from_pretrained(
        self.args.model_repo_id,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        torch_dtype=self.torch_dtype,
    )

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
    )

  def run_and_save(self) -> None:
    self.trainer.train()
    self.model.push_to_hub(self.training_args.hub_model_id)


class PERLPipeline(Pipeline):
  """Pipeline for PERL training (perl.py).

  This pipeline loads datasets, initializes policy and reward model,
  and uses RLOOTrainer with standard upstream TRL to run RL training.
  """

  def setup_arguments(self, *cli_args, **cli_kwargs) -> None:
    parser = HfArgumentParser((ScriptArguments, RLOOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    self.args = script_args
    self.training_args = training_args
    set_seed(training_args.seed)

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

    self.reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.bfloat16,
    )
    self.reward_model.to(torch.bfloat16)
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

    # Custom reward function that computes scalar reward as the probability of label 1 ("No" hallucination).
    def reward_fn(
        prompts: list[str], completions: list[str], **kwargs
    ) -> list[float]:
      texts = [p + c for p, c in zip(prompts, completions)]
      inputs = reward_tokenizer(
          texts,
          padding=True,
          truncation=True,
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

        # Probability of label 1 ("No" hallucination = non-hallucinated score)
        probs = torch.softmax(logits, dim=-1)[:, 1]
      return probs.cpu().tolist()

    self.trainer = RLOOTrainer(
        model=self.policy,
        reward_funcs=reward_fn,
        args=self.training_args,
        train_dataset=self.data["train"],
        eval_dataset=self.data.get("test", None),
        processing_class=self.tokenizer,
    )

  def run_and_save(self) -> None:
    if self.training_args.do_train:
      self.trainer.train()
      self.trainer.push_to_hub()


class DPOPipeline(Pipeline):
  """Pipeline for Direct Preference Optimization (DPO) preference tuning.

  This pipeline loads a preference dataset (containing 'prompt', 'chosen', and
  'rejected' columns), initializes the policy model from the SFT checkpoint
  (using
  LoRA adapters), and fine-tunes using TRL's DPOTrainer.
  """

  def setup_arguments(self, *cli_args, **cli_kwargs) -> None:
    parser_lora = create_lora_argument_parser()
    lora_args, remaining_args = parser_lora.parse_known_args()
    parser = HfArgumentParser((ScriptArguments, DPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses(
        remaining_args
    )

    self.args = script_args
    self.training_args = training_args
    set_seed(training_args.seed)
    self._lora_args = lora_args

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

    if sft_model_path:
      # In SCOPE, the policy is initialized at the SFT model p_{\theta_0}, and
      # the reference model is also p_{\theta_0}.
      # By loading and merging the SFT adapter into the base weights, then applying
      # a fresh LoRA adapter for DPO, DPOTrainer's adapter disabling automatically
      # evaluates the exact SFT reference model without extra memory overhead.
      sft_peft = PeftModel.from_pretrained(base_model, sft_model_path)
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
    )

  def run_and_save(self) -> None:
    if self.training_args.do_train:
      self.trainer.train()
      self.trainer.push_to_hub()


class ScopeDataGenerationPipeline(Pipeline):
  """Pipeline for generating synthetic preference datasets for SCOPE.

  Implements noisy decoding (Section 3 / Algorithm 1 of the SCOPE paper)
  by sampling mixed tokens between the fine-tuned SFT checkpoint and the
  pre-trained base model.
  """

  def setup_arguments(self, *cli_args, **cli_kwargs) -> None:
    parser = HfArgumentParser(ScopeDataGenArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
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

  def _extract_prompt_and_chosen(self, entry: dict) -> Tuple[str, str]:
    """Extract standard prompt and chosen completion from dataset entry."""
    task_name = self.args.task_name
    if "prompt" in entry and "chosen" in entry:
      return entry["prompt"], entry["chosen"]
    if "prompt" in entry and "completion" in entry:
      return entry["prompt"], entry["completion"]
    if "prompt" in entry and "npov_response" in entry:
      return entry["prompt"], entry["npov_response"]

    # Fallback to task processor prompt formatting
    processor_cls = get_task_processor(task_name)
    if task_name == "npov":
      prompt = processor_cls._writer_prompt(entry, SFT=False)["prompt"]
      chosen = entry.get("npov_response", entry.get("completion", ""))
      return prompt, chosen
    elif task_name == "bosch":
      prompt = (
          "You are a helpful assistant to car related questions. You will be"
          " given an user's question, and the relevant part of the car"
          " manual. Your task is to answer the user's question using the"
          " information giver. Do not add to your answer any information other"
          " than those present in the manual excerpt.\nUser"
          f" question:\n{entry.get('Question', '')}\nManual"
          f" information:\n{entry.get('Context', '')}\nAnswer to user's"
          " question:\n"
      )
      chosen = entry.get("response", entry.get("completion", ""))
      return prompt, chosen
    elif task_name == "ragtruth":
      prompt = entry.get("prompt", entry.get("user_query", ""))
      chosen = entry.get("completion", entry.get("response", ""))
      return prompt, chosen
    else:
      prompt = entry.get("prompt", str(entry))
      chosen = entry.get("chosen", entry.get("completion", ""))
      return prompt, chosen

  def process_data(self) -> None:
    train_split = (
        self.raw_dataset["train"]
        if isinstance(self.raw_dataset, (DatasetDict, dict))
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

    print(f"Loading base pre-trained model: {self.args.model_repo_id}...")
    self.base_model = AutoModelForCausalLM.from_pretrained(
        self.args.model_repo_id,
        torch_dtype=torch.bfloat16,
    ).to(device)
    self.base_model.eval()

    print(f"Loading SFT model from: {self.args.sft_model_path}...")
    sft_base = AutoModelForCausalLM.from_pretrained(
        self.args.model_repo_id,
        torch_dtype=torch.bfloat16,
    ).to(device)
    self.sft_model = PeftModel.from_pretrained(
        sft_base, self.args.sft_model_path
    ).to(device)
    self.sft_model.eval()

  def setup_trainer(self) -> None:
    pass

  def _generate_unfaithful_sample(
      self,
      prompt: str,
  ) -> str:
    """Generate a dispreferred sample using SCOPE noisy decoding (Algorithm 1)."""
    tokenizer = self.tokenizer
    device = self.device
    alpha = self.args.alpha
    temperature = self.args.temperature
    top_p = self.args.top_p
    top_k = self.args.top_k
    max_new_tokens = self.args.max_new_tokens
    sampling_mode = self.args.sampling_mode

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    if tokenizer.bos_token_id is not None:
      base_input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    else:
      base_input_ids = prompt_ids[:, :1]

    generated_tokens = []
    past_key_values_sft = None
    past_key_values_base = None
    cur_sft_input = prompt_ids
    cur_base_input = base_input_ids

    eos_token_ids = {tokenizer.eos_token_id}
    if getattr(tokenizer, "pad_token_id", None) is not None:
      eos_token_ids.add(tokenizer.pad_token_id)
    if hasattr(tokenizer, "additional_special_tokens_ids"):
      eos_token_ids.update(tokenizer.additional_special_tokens_ids)

    with torch.no_grad():
      for _ in range(max_new_tokens):
        sft_out = self.sft_model(
            input_ids=cur_sft_input,
            past_key_values=past_key_values_sft,
            use_cache=True,
        )
        base_out = self.base_model(
            input_ids=cur_base_input,
            past_key_values=past_key_values_base,
            use_cache=True,
        )

        past_key_values_sft = sft_out.past_key_values
        past_key_values_base = base_out.past_key_values

        logits_sft = sft_out.logits[:, -1, :].clone()
        logits_base = base_out.logits[:, -1, :].clone()

        if temperature > 0 and temperature != 1.0:
          logits_sft = logits_sft / temperature
          logits_base = logits_base / temperature

        if sampling_mode == "bernoulli":
          # Algorithm 1: alpha_t ~ Bernoulli(alpha)
          alpha_t = torch.bernoulli(torch.tensor([alpha], device=device)).item()
          selected_logits = logits_base if (alpha_t == 1.0) else logits_sft
          probs = torch.softmax(selected_logits, dim=-1)
        elif sampling_mode == "prob_mix":
          probs_sft = torch.softmax(logits_sft, dim=-1)
          probs_base = torch.softmax(logits_base, dim=-1)
          probs = (1.0 - alpha) * probs_sft + alpha * probs_base
        elif sampling_mode == "logit_mix":
          mixed_logits = (1.0 - alpha) * logits_sft + alpha * logits_base
          probs = torch.softmax(mixed_logits, dim=-1)
        else:
          probs = torch.softmax(logits_sft, dim=-1)

        if top_k > 0 and top_k < probs.size(-1):
          top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
          probs = torch.zeros_like(probs).scatter_(
              -1, top_k_indices, top_k_probs
          )
          probs = probs / probs.sum(dim=-1, keepdim=True)

        if top_p < 1.0:
          sorted_probs, sorted_indices = torch.sort(
              probs, descending=True, dim=-1
          )
          cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
          sorted_indices_to_remove = cumulative_probs > top_p
          sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
              ..., :-1
          ].clone()
          sorted_indices_to_remove[..., 0] = 0
          sorted_probs[sorted_indices_to_remove] = 0.0
          probs = torch.zeros_like(probs).scatter_(
              -1, sorted_indices, sorted_probs
          )
          probs = probs / probs.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()

        if token_id in eos_token_ids:
          break
        token_str = tokenizer.decode([token_id])
        if "<end_of_turn>" in token_str or "<eos>" in token_str:
          break

        generated_tokens.append(token_id)
        cur_sft_input = next_token
        cur_base_input = next_token

    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

  def run_and_save(self) -> None:
    prompts = []
    chosens = []
    rejecteds = []

    print(
        "Generating synthetic dispreferred completions for"
        f" {len(self.d2_split)} samples..."
    )
    for entry in tqdm(self.d2_split, desc="SCOPE noisy decoding"):
      prompt, chosen = self._extract_prompt_and_chosen(entry)
      rejected = self._generate_unfaithful_sample(prompt)
      prompts.append(prompt)
      chosens.append(chosen)
      rejecteds.append(rejected)

    pref_dict = {
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    }
    train_dataset = Dataset.from_dict(pref_dict)

    # Process test split if available
    test_split = None
    if (
        isinstance(self.raw_dataset, (DatasetDict, dict))
        and "test" in self.raw_dataset
    ):
      test_prompts, test_chosens, test_rejecteds = [], [], []
      for entry in self.raw_dataset["test"]:
        prompt, chosen = self._extract_prompt_and_chosen(entry)
        test_prompts.append(prompt)
        test_chosens.append(chosen)
        test_rejecteds.append(chosen)
      test_split = Dataset.from_dict({
          "prompt": test_prompts,
          "chosen": test_chosens,
          "rejected": test_rejecteds,
      })

    if test_split is not None:
      preference_dataset = DatasetDict(
          {"train": train_dataset, "test": test_split}
      )
    else:
      preference_dataset = DatasetDict({"train": train_dataset})

    out_repo = (
        self.args.output_dataset_repo_id
        or f"{self.args.dataset_repo_id}_scope_preference"
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
    iterator = data.iter(batch_size=script_args.eval_batch_size)
    num_batches = int(data.num_rows / script_args.eval_batch_size)
    scores = torch.tensor([])
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
      score_batch = score_batch.cpu()
      scores = torch.cat((scores, score_batch))
      del tokenized_prompts
    return scores

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
          f"Initializing Google GenAI Client using Vertex AI (project={project}, location={location})..."
      )
      return genai.Client(vertexai=True, project=project, location=location)
    elif getattr(self.args, "gemini_api_key", None):
      return genai.Client(api_key=self.args.gemini_api_key)
    else:
      return genai.Client()

  def gemini_score_response(
      self, response: types.GenerateContentResponse
  ) -> float:
    """Converts a Gemini API response to a normalized score.

    Args:
        response (google.genai.types.GenerateContentResponse): Gemini API
          response.

    Returns:
        float: Normalized score in [0.0, 1.0] for the response (P(No
        hallucination)).
    """
    if not response or not response.candidates:
      return 0.5

    candidate = response.candidates[0]
    text = response.text.strip() if response.text else ""
    clean_text = text.strip("\"'` \n\r\t")

    # 1. Check if token logprobs are available in candidate.logprobs_result
    if (
        hasattr(candidate, "logprobs_result")
        and candidate.logprobs_result is not None
    ):
      chosen = getattr(candidate.logprobs_result, "chosen_candidates", None)
      if chosen and len(chosen) > 0:
        top_cands = getattr(chosen[0], "top_candidates", None)
        if top_cands:
          logp_no = None
          logp_yes = None
          for cand in top_cands:
            cand_token = (
                getattr(cand, "token", "").strip().strip("\"'`").lower()
            )
            if cand_token == "no":
              logp_no = getattr(cand, "logprob", None)
            elif cand_token == "yes":
              logp_yes = getattr(cand, "logprob", None)
          if logp_no is not None and logp_yes is not None:
            p_no = float(np.exp(logp_no))
            p_yes = float(np.exp(logp_yes))
            denom = p_no + p_yes
            if denom > 0:
              return float(p_no / denom)

    # 2. Check avg_logprobs if available
    if getattr(candidate, "avg_logprobs", None) is not None:
      avg_logprob = candidate.avg_logprobs
      if clean_text.lower().startswith("no"):
        return float(np.exp(avg_logprob))
      elif clean_text.lower().startswith("yes"):
        return float(1.0 - np.exp(avg_logprob))

    # 3. Fallback based on text classification
    if clean_text.lower().startswith("no"):
      return 1.0
    elif clean_text.lower().startswith("yes"):
      return 0.0

    return 0.5

  def gemini_score_dataset(
      self,
      client: genai.Client,
      dataset: Dataset,
      script_args: ScriptArguments,
  ) -> torch.Tensor:
    """Scores a dataset using the Gemini API, with checkpointing and retries.

    Args:
        client (google.genai.Client): Initialized Gemini API client.
        dataset (datasets.Dataset): Dataset with 'evaluator_prompt' column.
        script_args (ScriptArguments): Parsed script arguments.

    Returns:
        torch.Tensor: Scores for each entry in the dataset.
    """
    model = getattr(script_args, "evaluator_model", None) or "gemini-2.0-flash"
    schema = {"type": "STRING", "enum": ["No", "Yes"]}
    print(f"Calling the Gemini API with model {model}...")

    # Prepare local checkpoint directory
    checkpoint_dir = os.path.join("checkpoints", "eval")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if (
        hasattr(script_args, "dataset_with_completions")
        and script_args.dataset_with_completions
    ):
      dataset_name = script_args.dataset_with_completions.split("/")[-1]
    elif hasattr(script_args, "dataset_labels") and script_args.dataset_labels:
      dataset_name = script_args.dataset_labels.split("/")[-1]
    else:
      dataset_name = "eval"

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{dataset_name}_scores_checkpoint.pt",
    )

    # Initialize scores: load from checkpoint if exists, else from dataset or None
    if os.path.exists(checkpoint_path):
      print(f"Loading scores from checkpoint: {checkpoint_path}")
      scores = torch.load(checkpoint_path)
      # If checkpoint is shorter than dataset (e.g. dataset updated), pad with None
      if len(scores) < len(dataset):
        scores = list(scores) + [None] * (len(dataset) - len(scores))
      elif len(scores) > len(dataset):
        scores = list(scores)[: len(dataset)]
    elif "scores" in dataset.column_names:
      scores = list(dataset["scores"])
    else:
      scores = [None] * len(dataset)

    queries_per_minute = 2000
    time_window = 60
    query_count = 0
    start_time = time.time()
    save_frequency = 50  # Save progress every 50 entries
    server_retry_wait = 20  # seconds to wait between server error retries
    server_max_retries = 3  # number of times to retry on server error
    entries_to_score = [i for i, score in enumerate(scores) if score is None]
    print(f"Found {len(entries_to_score)} entries that need scoring...")

    use_logprobs = True
    try:
      for n, idx in enumerate(
          tqdm(entries_to_score, desc="Scoring with Gemini API")
      ):
        elapsed_time = time.time() - start_time
        if query_count == (queries_per_minute - 1):
          if elapsed_time < time_window:
            wait_time = time_window - elapsed_time
            print(f"Wait {wait_time:.2f} seconds to avoid API rate limit...")
            time.sleep(wait_time + 1)
          query_count = 0
          start_time = time.time()

        retry_count = 0
        while retry_count < server_max_retries:
          try:
            if use_logprobs:
              # When requesting logprobs, do not pass response_mime_type="text/x.enum"
              # as structured enum decoding can conflict with logprobs on Vertex AI
              config_kwargs = {
                  "temperature": 0,
                  "max_output_tokens": 10,
                  "response_logprobs": True,
                  "logprobs": 5,
                  "seed": script_args.seed,
              }
            else:
              config_kwargs = {
                  "response_mime_type": "text/x.enum",
                  "response_schema": schema,
                  "temperature": 0,
                  "max_output_tokens": 10,
                  "seed": script_args.seed,
              }

            response = client.models.generate_content(
                model=model,
                contents=dataset[idx]["evaluator_prompt"],
                config=types.GenerateContentConfig(**config_kwargs),
            )
            scores[idx] = self.gemini_score_response(response)
            query_count += 1
            break  # Success, break out of retry loop
          except Exception as e:
            error_str = str(e).lower()
            if "logprob" in error_str and use_logprobs:
              print(
                  f"\nNotice: Logprobs request failed with error: {e}. "
                  "Falling back to text classification."
              )
              use_logprobs = False
              continue
            if (
                "unavailable" in error_str
                or "overloaded" in error_str
                or "overcharged" in error_str
                or "server" in error_str
                or "resource_exhausted" in error_str
            ) and retry_count < server_max_retries - 1:
              print(
                  f"Server unavailable/overloaded at entry {idx}, attempt"
                  f" {retry_count + 1}/{server_max_retries}. Waiting"
                  f" {server_retry_wait} seconds before retrying..."
              )
              time.sleep(server_retry_wait)
              retry_count += 1
              continue
            else:
              raise  # Not a server error or max retries reached

        # Save progress periodically to local file
        if (n + 1) % save_frequency == 0:
          print(f"\nSaving progress locally after {n + 1} new entries...")
          try:
            torch.save(scores, checkpoint_path)
          except Exception as e:
            print(f"Warning: Could not save intermediate progress locally: {e}")

    except Exception as e:
      print(f"\nError encountered at entry {idx}: {str(e)}")
      print("Saving current progress locally...")
      try:
        torch.save(scores, checkpoint_path)
      except Exception as save_error:
        print(f"Error saving progress locally: {save_error}")
      raise e

    # Final save after all scoring is done
    dataset = (
        dataset.remove_columns("scores")
        if "scores" in dataset.column_names
        else dataset
    )
    dataset = dataset.add_column("scores", scores)
    if (
        hasattr(script_args, "dataset_with_completions")
        and script_args.dataset_with_completions
    ):
      try:
        dataset.push_to_hub(script_args.dataset_with_completions)
      except Exception as e:
        print(f"Warning: Could not save final progress to hub: {e}")
    # Remove local checkpoint after successful push
    if os.path.exists(checkpoint_path):
      os.remove(checkpoint_path)

    # Convert scores to tensor, replacing any remaining None with 0
    scores_tensor = torch.tensor([s if s is not None else 0 for s in scores])
    return scores_tensor


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

    self.data = load_dataset(
        self.args.dataset_labels, split=self.args.dataset_labels_split
    )

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
      self.evaluator = AutoModelForCausalLM.from_pretrained(
          self.args.evaluator_model,
          torch_dtype=torch.bfloat16,
      )
      self.evaluator.eval()

  def run_and_save(self) -> None:
    # Score using evaluator (either gemini or local model)
    if self.args.use_gemini:
      client = self.create_gemini_client()
      scores = self.gemini_score_dataset(client, self.data, self.args)
    else:
      # Tokenize in batches and compute scores using tokenizer token ids for Yes/No
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
    filepath = f"logs/{name_for_saving}/eval_autorater_scores.txt"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
      for score in scores:
        f.write(f"{score.item():.5f}\n")
    print(f"Scores saved to {filepath}")

    auc = roc_auc_score(ground_truth, scores.numpy())
    metrics = compute_best_roc_threshold(ground_truth, scores.numpy())
    threshold = metrics["best_threshold"]
    tpr = metrics["tpr_at_best_threshold"]
    fpr = metrics["fpr_at_best_threshold"]
    accuracy = metrics["accuracy_at_best_threshold"]
    classif_at_threshold = [0 if score < threshold else 1 for score in scores]

    metrics_filepath = f"logs/{name_for_saving}/eval_autorater_metrics.txt"
    os.makedirs(os.path.dirname(metrics_filepath), exist_ok=True)
    with open(metrics_filepath, "w") as f:
      f.write("AUC: {:.5f}\n".format(auc))
      f.write("Threshold: {:.5f}\n".format(threshold))
      f.write("TPR (recall): {:.5f}\n".format(tpr))
      f.write("FPR: {:.5f}\n".format(fpr))
      f.write("Accuracy: {:.5f}\n".format(accuracy))
      f.write(
          "Precision: {:.5f}\n".format(
              precision_score(ground_truth, classif_at_threshold)
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
            precision_score(ground_truth, classif_at_threshold)
        )
    )

    # ROC-AUC plot
    RocCurveDisplay.from_predictions(ground_truth, scores.numpy())
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
        score.item()
        for idx, score in enumerate(scores)
        if ground_truth[idx] == 1
    ]
    scores_yes = [
        score.item()
        for idx, score in enumerate(scores)
        if ground_truth[idx] == 0
    ]
    plt.vlines(x=threshold, ymin=0, ymax=175, colors="r")
    plt.hist(scores_no, bins=bins, alpha=0.5, label="No")
    plt.hist(scores_yes, bins=bins, alpha=0.5, label="Yes")
    plt.legend()
    plt.savefig(
        f"logs/{name_for_saving}/"
        + f"eval_autorater_{self.args.evaluator_num_fewshot}_shot"
    )


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
    self.dataset_prompts = load_dataset(
        self.args.dataset_prompts, split=self.args.dataset_prompts_split
    )

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
      fewshot_data = load_dataset(
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
    enable_lora = (
        False
        if self.args.writer_model_base == self.args.writer_model_lora
        else True
    )
    self.enable_lora = enable_lora
    # vLLM model identifier (instantiated in run_and_save)
    self.vllm_model = self.args.writer_model_base

  def run_and_save(self) -> None:
    # Prepare sampling parameters for vLLM
    sampling_params = SamplingParams(
        seed=self.args.seed,
        temperature=self.args.temperature,
        top_p=self.args.top_p,
        top_k=self.args.top_k,
        min_tokens=10,
        max_tokens=self.args.max_tokens,
    )

    # If LoRA is enabled, ensure we have the adapter path available (download if needed)
    lora_path = None
    if self.enable_lora:
      if os.path.exists(self.args.writer_model_lora):
        lora_path = self.args.writer_model_lora
      else:
        lora_path = snapshot_download(
            repo_id=self.args.writer_model_lora,
            allow_patterns=["*.json", "*.safetensors"],
        )

    # Instantiate vLLM LLM (use bfloat16 dtype)
    llm = LLM(
        model=self.vllm_model,
        enable_lora=self.enable_lora,
        max_lora_rank=64,
        dtype="bfloat16",
    )

    # Run generation
    if self.enable_lora and lora_path is not None:
      outputs = llm.generate(
          self.prompts,
          sampling_params,
          lora_request=LoRARequest(
              "writer_lora_adapter", 1, lora_path=lora_path
          ),
      )
    else:
      outputs = llm.generate(self.prompts, sampling_params)

    generations = [output.outputs[0].text for output in outputs]
    self.dataset_prompts = self.dataset_prompts.add_column(
        "completion", generations
    )
    try:
      name_for_saving = self.args.writer_model_lora.split(f"{self.args.user}/")[
          1
      ]
    except Exception:
      name_for_saving = self.args.writer_model_lora.split("/")[1]

    name_for_saving = "eval_" + name_for_saving + "_gens"
    name_for_saving += f"_T{str(float(self.args.temperature))}"
    name_for_saving += f"_wfs{self.args.writer_num_fewshot}"

    print(
        "Pushing dataset with generations to"
        f" {self.args.user}/{name_for_saving}"
    )
    self.dataset_prompts.push_to_hub(f"{self.args.user}/{name_for_saving}")


class EvaluationScoringPipeline(EvaluationPipeline):
  """Pipeline for scoring existing dataset with completions."""

  def setup_arguments(self, *cli_args, **cli_kwargs):
    parser = HfArgumentParser(EvalArguments)
    self.args = parser.parse_args_into_dataclasses()[0]
    set_seed(self.args.seed)

  def setup_tokenizer(self):
    if not self.args.use_gemini:
      self.tokenizer = AutoTokenizer.from_pretrained(
          self.args.evaluator_model, padding_side="left"
      )

  def load_data(self):
    if self.args.dataset_with_completions is None:
      raise ValueError("dataset_with_completions is required for scoring mode")
    self.val_data = load_dataset(
        self.args.dataset_with_completions, split="test"
    )

  def process_data(self):
    processor_cls = get_task_processor(self.args.task_name)
    prompt_fn = processor_cls.get_evaluator_prompt()
    fewshot_examples = None
    if (
        self.args.evaluator_num_fewshot
        and self.args.evaluator_num_fewshot > 0
        and self.args.dataset_labels
    ):
      label_data = load_dataset(
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
    if not self.args.use_gemini:
      # Use causal LM evaluator and set eval mode
      self.evaluator = AutoModelForCausalLM.from_pretrained(
          self.args.evaluator_model,
          torch_dtype=torch.bfloat16,
      )
      self.evaluator.eval()

  def run_and_save(self):
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
    # Compute simple rate and push dataset updated
    t = torch.nn.Threshold(self.args.threshold, 0, inplace=False)
    classifs = torch.ceil(t(scores)).clamp(0, 1)
    self.val_data = self.val_data.add_column("scores", scores.tolist())
    self.val_data = self.val_data.add_column(
        "classifications", classifs.tolist()
    )
    self.val_data.push_to_hub(self.args.dataset_with_completions)
