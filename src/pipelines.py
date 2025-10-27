"""
pipelines.py

Defines an abstract Pipeline class and concrete pipeline implementations for
SFT (writer_sft), reward model training, and PERL (rlhf) workflows.

Each pipeline implements the sequence of steps described in the project:
1. setup_arguments
2. setup_tokenizer
3. load_data
4. process_data
5. setup_model
6. setup_trainer
7. run_and_save

The goal is to centralize common code and make the three scripts thin wrappers
that create and run the appropriate pipeline.
"""

from __future__ import annotations

import abc
import os
from typing import Any, Optional

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Value, concatenate_datasets, load_dataset
from google import genai
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import (
    RocCurveDisplay,
    precision_score,
    roc_auc_score,
)
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
    DataCollatorForCompletionOnlyLM,
    RLOOConfig,
    RLOOTrainer,
    SFTConfig,
    SFTTrainer,
    TrlParser,
)
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Task processor classes are imported lazily inside utils.get_task_processor
from .utils import (
    EvalArguments,
    LLMSynthScriptArguments,
    ScriptArguments,
    compute_best_roc_threshold,
    create_lora_argument_parser,
    get_task_processor,
)


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

    def run(self, *cli_args, **cli_kwargs):
        self.setup_arguments(*cli_args, **cli_kwargs)
        self.setup_tokenizer()
        self.load_data()
        self.process_data()
        self.setup_model()
        self.setup_trainer()
        self.run_and_save()

    # Task processor lookup moved to `utils.get_task_processor` to keep
    # pipelines focused on orchestration. Call `get_task_processor(task_name)`
    # from utils where needed.

    @abc.abstractmethod
    def setup_arguments(self, *cli_args, **cli_kwargs):
        raise NotImplementedError()

    def setup_tokenizer(self):
        """
        Default tokenizer loading uses self.args.model_repo_id or training args.
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

        # Some models don't have pad token
        if "pad_token" not in self.tokenizer.special_tokens_map.keys():
            self.tokenizer.pad_token = self.tokenizer.unk_token

    @abc.abstractmethod
    def load_data(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def process_data(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def setup_model(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def setup_trainer(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def run_and_save(self):
        raise NotImplementedError()


class SFTPipeline(Pipeline):
    """Pipeline for supervised fine-tuning (writer_sft.py).

    This class re-implements the logic that was previously in writer_sft.py but
    keeps the external script as a thin wrapper.
    """

    def setup_arguments(self, *cli_args, **cli_kwargs):
        parser_lora = create_lora_argument_parser()
        lora_args, remaining_args = parser_lora.parse_known_args()
        parser = TrlParser((ScriptArguments, SFTConfig))
        script_args, training_args = parser.parse_args_into_dataclasses(
            remaining_args
        )

        self.args = script_args
        self.training_args = training_args
        set_seed(training_args.seed)
        # keep lora config object around for setup_model
        self._lora_args = lora_args

    def load_data(self):
        self.data = {
            "train": load_dataset(self.args.dataset_repo_id, split="train"),
            "test": load_dataset(self.args.dataset_repo_id, split="test"),
        }

    def process_data(self):
        # task-specific processor
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

        response_template_ids = self.tokenizer.encode(
            response_template, add_special_tokens=False
        )
        self.data_collator = DataCollatorForCompletionOnlyLM(
            response_template_ids, tokenizer=self.tokenizer
        )

        # expose formatting func for trainer
        self.formatting_prompts_func = formatting_prompts_func

    def setup_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_repo_id,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        )

        # If pad token was added, need to resize embeddings.
        if "pad_token" not in self.tokenizer.special_tokens_map.keys():
            model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model = get_peft_model(model, self._lora_config)

    def setup_trainer(self):
        self.trainer = SFTTrainer(
            self.model,
            args=self.training_args,
            data_collator=self.data_collator,
            train_dataset=self.data["train"],
            eval_dataset=self.data["test"],
            processing_class=self.tokenizer,
            formatting_func=self.formatting_prompts_func,
        )

    def run_and_save(self):
        self.trainer.train()
        self.trainer.push_to_hub()


class RewardModelPipeline(Pipeline):
    """Pipeline for reward model training (reward_model.py)."""

    def setup_arguments(self, *cli_args, **cli_kwargs):
        parser_lora = create_lora_argument_parser()
        lora_args, remaining_args = parser_lora.parse_known_args()

        # HfArgumentParser equivalence: we'll import TrainingArguments through HfArgumentParser in the original script
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

    def load_data(self):
        self.data = load_dataset(self.args.dataset_repo_id)

    def process_data(self):
        # Build lora config
        self._lora_config = LoraConfig(
            r=self._lora_args.lora_r,
            lora_alpha=self._lora_args.lora_alpha,
            lora_dropout=self._lora_args.lora_dropout,
            task_type=self._lora_args.task_type,
            peft_type=self._lora_args.peft_type,
        )

        # Tokenizer pad token already set in base
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.data_collator = data_collator

        # Determine torch dtype from training args
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

    def setup_model(self):
        id2label = {0: "Yes", 1: "No"}
        label2id = {"Yes": 0, "No": 1}

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_repo_id,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            torch_dtype=self.torch_dtype,
            attn_implementation="eager",
        )

        if "pad_token" not in self.tokenizer.special_tokens_map.keys():
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model = get_peft_model(self.model, self._lora_config)

        # adjust score parameters as in original
        self.model.score.requires_grad_()
        with torch.no_grad():
            self.model.score.weight.mul_(0.1)

    def setup_trainer(self):
        metric = evaluate.load("roc_auc")

        def compute_metrics(eval_preds):
            logits = eval_preds.predictions
            yes_scores = np.exp(logits)[:, 0]
            no_scores = np.exp(logits)[:, 1]
            scores = no_scores / (yes_scores + no_scores)
            label_ids = eval_preds.label_ids
            metrics = metric.compute(
                references=label_ids, prediction_scores=scores
            )
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
            metrics.update(
                {
                    "avg_score_true_positives": avg_score_true_positives,
                    "avg_score_true_negatives": avg_score_true_negatives,
                }
            )
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

    def run_and_save(self):
        self.trainer.train()
        self.model.push_to_hub(self.training_args.hub_model_id)


class PERLPipeline(Pipeline):
    """Pipeline for PERL training (perl.py).

    This pipeline loads tokenized perl datasets, sets up policy, reference policy,
    reward model, and uses RLOOTrainer to run RL training.
    """

    def setup_arguments(self, *cli_args, **cli_kwargs):
        parser = HfArgumentParser((ScriptArguments, RLOOConfig))
        script_args, training_args = parser.parse_args_into_dataclasses()
        self.args = script_args
        self.training_args = training_args
        set_seed(training_args.seed)

    def load_data(self):
        self.data = load_dataset(self.args.dataset_repo_id)

    def process_data(self):
        # Tokenize data similarly to original perl.py
        def encode(examples):
            return self.tokenizer(
                examples["prompt"],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        for split in self.data.keys():
            self.data[split] = self.data[split].map(
                encode,
                remove_columns=self.data[split].column_names,
                batched=True,
            )
            self.data[split].set_format("torch")

    def setup_model(self):
        id2label = {0: "Yes", 1: "No"}
        label2id = {"Yes": 0, "No": 1}

        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.training_args.reward_model_path,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        )

        self.ref_policy = AutoModelForCausalLM.from_pretrained(
            self.training_args.sft_model_path,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        )

        policy_base = AutoModelForCausalLM.from_pretrained(
            self.args.model_repo_id,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        )

        self.policy = PeftModel.from_pretrained(
            policy_base, self.training_args.sft_model_path, is_trainable=True
        )

    def setup_trainer(self):
        self.trainer = RLOOTrainer(
            config=self.training_args,
            processing_class=self.tokenizer,
            ref_policy=self.ref_policy,
            policy=self.policy,
            reward_model=self.reward_model,
            train_dataset=self.data["train"],
            eval_dataset=self.data["test"],
        )

    def run_and_save(self):
        if self.training_args.do_train:
            self.trainer.train()
            self.trainer.push_to_hub()

        name_for_saving = self.training_args.run_name.split("/")[1]
        filepath = os.path.join("logs", name_for_saving, "logs.txt")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            for d in self.trainer.state.log_history:
                f.write(str(d) + "\n----------\n")
        print(f"Logs saved to {filepath}")


class EvaluationPipeline(Pipeline):
    """Base class for evaluator flows. Provides helper methods previously
    implemented in `evaluator_utils.py` so evaluation pipelines can call
    them as instance methods and share configuration/state.
    """

    def get_fewshot_examples(self, data, n_yes: int, n_no: int, seed: int):
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
            concatenate_datasets(
                [positive_examples, negative_examples]
            ).shuffle(seed=seed)
            if positive_examples is not None
            else None
        )
        return fewshot_examples

    def evaluator_score_batch(
        self, evaluator, tokenized_prompts, yes_token_id, no_token_id
    ):
        with torch.no_grad():
            outputs = evaluator(**tokenized_prompts, use_cache=False)
            score_yes = torch.exp(outputs.logits[:, -1, yes_token_id])
            score_no = torch.exp(outputs.logits[:, -1, no_token_id])
            score_batch = score_no / (score_yes + score_no)
        return score_batch

    def evaluator_score(
        self, data, script_args, tokenizer, evaluator, yes_token_id, no_token_id
    ):
        iterator = data.iter(batch_size=script_args.eval_batch_size)
        num_batches = int(data.num_rows / script_args.eval_batch_size)
        scores = torch.tensor([])
        for batch in tqdm(
            iterator, desc="Evaluator scoring", total=num_batches
        ):
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

    def gemini_score_dataset(self, client, dataset, script_args):
        model = "gemini-2.0-flash-001"
        scores = []
        for entry in dataset:
            resp = client.models.generate_content(
                model=model, contents=entry["evaluator_prompt"]
            )
            # reproduce gemini score response logic
            if resp.candidates[0].avg_logprobs is None:
                scores.append(0)
            elif resp.text == "No":
                scores.append(np.exp(resp.candidates[0].avg_logprobs))
            elif resp.text == "Yes":
                scores.append(1 - np.exp(resp.candidates[0].avg_logprobs))
            else:
                raise Exception("Invalid response")
        return torch.tensor(scores)


class EvaluationAutoraterPipeline(EvaluationPipeline):
    """Pipeline for the autorater evaluation (script_args.evaluate_evaluator == True)."""

    def setup_arguments(self, *cli_args, **cli_kwargs):
        parser = HfArgumentParser(EvalArguments)
        script_args = parser.parse_args_into_dataclasses()[0]
        self.args = script_args
        set_seed(self.args.seed)

    def setup_tokenizer(self):
        # Tokenizer for the evaluator model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.evaluator_model, padding_side="left"
        )

    def load_data(self):
        # Load dataset with hallucination labels
        if self.args.dataset_labels is None:
            raise ValueError(
                "dataset_labels is required for autorater evaluation"
            )

        self.data = load_dataset(
            self.args.dataset_labels, split=self.args.dataset_labels_split
        )

    def process_data(self):
        # Map the evaluator prompt onto the dataset
        processor_cls = get_task_processor(self.args.task_name)

        prompt_fn = processor_cls.get_evaluator_prompt()
        # Map using fewshot examples if requested
        fewshot_examples = None
        if (
            self.args.evaluator_num_fewshot
            and self.args.evaluator_num_fewshot > 0
        ):
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

    def setup_model(self):
        # Load evaluator as causal LM (same as original evaluator.py) and set eval mode
        self.evaluator = AutoModelForCausalLM.from_pretrained(
            self.args.evaluator_model,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        )
        self.evaluator.eval()

    # use gemini helpers from scripts.evaluator_utils

    def run_and_save(self):
        # Score using evaluator (either gemini or local model)
        if self.args.use_gemini:
            client = genai.Client(api_key=self.args.gemini_api_key)
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

        # Compute metrics (restore original evaluator metrics & plotting)
        ground_truth = self.data["label"]

        # Save evaluator prompts
        name_for_saving = (
            f"eval_autorater_{self.args.evaluator_model.split('/')[-1]}"
            + f"_autorater_num_fewshot_{self.args.evaluator_num_fewshot}"
            + f"_data_{self.args.dataset_labels.split('/')[-1]}"
            + f"_seed_{self.args.seed}"
        )

        filepath = (
            f"logs/{name_for_saving}/eval_autorater_evaluator_prompts.txt"
        )
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
        classif_at_threshold = [
            0 if score < threshold else 1 for score in scores
        ]

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

    def setup_arguments(self, *cli_args, **cli_kwargs):
        parser = HfArgumentParser(EvalArguments)
        self.args = parser.parse_args_into_dataclasses()[0]
        set_seed(self.args.seed)

    def load_data(self):
        self.dataset_prompts = load_dataset(
            self.args.dataset_prompts, split=self.args.dataset_prompts_split
        )

    def setup_model(self):
        enable_lora = (
            False
            if self.args.writer_model_base == self.args.writer_model_lora
            else True
        )
        self.enable_lora = enable_lora
        # vLLM model identifier (we instantiate the LLM in run_and_save)
        self.vllm_model = self.args.writer_model_base

    def process_data(self):
        # Prepare prompts and fewshot if requested
        prompts = [
            self.dataset_prompts[i]["prompt"]
            for i in range(self.dataset_prompts.num_rows)
        ]

        # Prepend few-shot examples if requested (same behavior as original evaluator.py)
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
            fewshot_prompts = "\n".join(
                [ex["prompt"] for ex in fewshot_examples]
            )
            prompts = [fewshot_prompts + "\n" + p for p in prompts]

        self.prompts = prompts

    def run_and_save(self):
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

        # Instantiate vLLM LLM (use bfloat16 dtype to match original)
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
            name_for_saving = self.args.writer_model_lora.split(
                f"{self.args.user}/"
            )[1]
        except Exception:
            name_for_saving = self.args.writer_model_lora.split("/")[1]

        name_for_saving = "eval_" + name_for_saving + "_gens"
        name_for_saving += f"_T{str(float(self.args.temperature))}"
        name_for_saving += f"_wfs{self.args.writer_num_fewshot}"

        print(
            f"Pushing dataset with generations to {self.args.user}/{name_for_saving}"
        )
        self.dataset_prompts.push_to_hub(f"{self.args.user}/{name_for_saving}")


class EvaluationScoringPipeline(EvaluationPipeline):
    """Pipeline for scoring existing dataset with completions."""

    def setup_arguments(self, *cli_args, **cli_kwargs):
        parser = HfArgumentParser(EvalArguments)
        self.args = parser.parse_args_into_dataclasses()[0]
        set_seed(self.args.seed)

    def load_data(self):
        if self.args.dataset_with_completions is None:
            raise ValueError(
                "dataset_with_completions is required for scoring mode"
            )
        self.val_data = load_dataset(
            self.args.dataset_with_completions, split="test"
        )

    def setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.evaluator_model, padding_side="left"
        )

    def setup_model(self):
        # Use causal LM evaluator (same as original evaluator) and set eval mode
        self.evaluator = AutoModelForCausalLM.from_pretrained(
            self.args.evaluator_model,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        )
        self.evaluator.eval()

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

    def run_and_save(self):
        # Score dataset using tokenizer and evaluator (support gemini)
        if self.args.use_gemini:
            client = genai.Client(api_key=self.args.gemini_api_key)
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
