"""
Defines an abstract Pipeline class and concrete pipeline implementations for SFT (writer_sft), reward model training, and PERL (rlhf) workflows. Also defines EvaluationPipeline and concrete classes corresponding to each step of the evaluation procedure (autorater evaluation, completion generation, scoring of completions).

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

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import (
    Dataset,
    Value,
    concatenate_datasets,
    load_dataset,
)
from google import genai
from google.genai import types
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

from src.utils import (
    EvalArguments,
    LLMSynthScriptArguments,
    ScriptArguments,
    add_row_to_gsheets,
    compute_best_roc_threshold,
    create_lora_argument_parser,
    find_insertion_index_rm,
    get_task_processor,
    setup_gsheets,
    update_gsheets_evaluation_generation,
    update_gsheets_perl,
    update_gsheets_rm,
    update_gsheets_sft,
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
        parser = TrlParser((ScriptArguments, SFTConfig))
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

    def setup_model(self) -> None:
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_repo_id,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        )

        # If pad token was added, need to resize embeddings.
        if "pad_token" not in self.tokenizer.special_tokens_map.keys():
            model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model = get_peft_model(model, self._lora_config)

    def setup_trainer(self) -> None:
        self.trainer = SFTTrainer(
            self.model,
            args=self.training_args,
            # data_collator=self.data_collator,
            train_dataset=self.data["train"],
            eval_dataset=self.data["test"],
            processing_class=self.tokenizer,
            # formatting_func=self.formatting_prompts_func,
        )

    def run_and_save(self) -> None:
        self.trainer.train()
        self.trainer.push_to_hub()

        if self.trainer.is_world_process_zero():
            if self.args.gsheets_name:
                ws_sft = setup_gsheets(self.args.gsheets_name, "SFT")
                update_gsheets_sft(
                    ws_sft,
                    self.args,
                    self._lora_args,
                    self.training_args,
                    self.trainer,
                )


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
            attn_implementation="eager",
        )

        if "pad_token" not in self.tokenizer.special_tokens_map.keys():
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model = get_peft_model(self.model, self._lora_config)

        # adjust score parameters
        self.model.score.requires_grad_()
        with torch.no_grad():
            self.model.score.weight.mul_(0.1)

    def setup_trainer(self) -> None:
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

    def run_and_save(self) -> None:
        self.trainer.train()
        self.model.push_to_hub(self.training_args.hub_model_id)

        if self.trainer.is_world_process_zero():
            if self.args.gsheets_name:
                ws_rm = setup_gsheets(self.args.gsheets_name, "RM")
                update_gsheets_rm(
                    ws_rm,
                    self.args,
                    self._lora_args,
                    self.training_args,
                    self.trainer,
                )


class PERLPipeline(Pipeline):
    """Pipeline for PERL training (perl.py).

    This pipeline loads tokenized perl datasets, sets up policy, reference policy,
    reward model, and uses RLOOTrainer to run RL training.
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

    def setup_model(self) -> None:
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

    def setup_trainer(self) -> None:
        self.trainer = RLOOTrainer(
            config=self.training_args,
            processing_class=self.tokenizer,
            ref_policy=self.ref_policy,
            policy=self.policy,
            reward_model=self.reward_model,
            train_dataset=self.data["train"],
            eval_dataset=self.data["test"],
        )

    def run_and_save(self) -> None:
        self.trainer.train()
        self.trainer.push_to_hub()

        if self.trainer.is_world_process_zero():
            if self.args.gsheets_name:
                ws_perl = setup_gsheets(self.args.gsheets_name, "PERL")
                update_gsheets_perl(
                    ws_perl,
                    self.args,
                    self._lora_args,
                    self.training_args,
                    self.trainer,
                )

        # Was getting errors with this (TODO: fix)
        # name_for_saving = self.training_args.run_name.split("/")[1]
        # filepath = os.path.join("logs", name_for_saving, "logs.txt")
        # os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # with open(filepath, "w") as f:
        #     for d in self.trainer.state.log_history:
        #         f.write(str(d) + "\n----------\n")
        # print(f"Logs saved to {filepath}")


class EvaluationPipeline(Pipeline):
    """Base class for evaluator flows. Provides helper methods previously
    implemented in `evaluator_utils.py` so evaluation pipelines can call
    them as instance methods and share configuration/state.
    """

    def setup_trainer(self) -> None:
        # Evaluation pipelines do not have trainers
        pass

    def get_fewshot_examples(
        self, data: Dataset, n_yes: int, n_no: int, seed: int
    ) -> Dataset:
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
        self,
        evaluator: AutoModelForCausalLM,
        tokenized_prompts: dict[str, torch.Tensor],
        yes_token_id: int,
        no_token_id: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = evaluator(**tokenized_prompts, use_cache=False)
            score_yes = torch.exp(outputs.logits[:, -1, yes_token_id])
            score_no = torch.exp(outputs.logits[:, -1, no_token_id])
            score_batch = score_no / (score_yes + score_no)
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

    def gemini_score_response(
        self, response: types.GenerateContentResponse
    ) -> float:
        """Converts a Gemini API response to a normalized score.

        Args:
            response (google.genai.types.GenerateContentResponse): Gemini API response.

        Returns:
            float: Normalized score for the response.
        """

        if response.candidates[0].avg_logprobs is None:
            return 0
        if response.text == "No":
            return np.exp(response.candidates[0].avg_logprobs)
        elif response.text == "Yes":
            return 1 - np.exp(response.candidates[0].avg_logprobs)
        else:
            raise Exception("Invalid response")

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

        model = "gemini-2.0-flash-001"
        schema = {"type": "STRING", "enum": ["No", "Yes"]}
        print("Calling the Gemini API...")

        # Prepare local checkpoint directory
        checkpoint_dir = os.path.join("checkpoints", "eval")
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Use only the dataset name after the slash for checkpoint filename
        if (
            script_args.dataset_with_completions
            and "/" in script_args.dataset_with_completions
        ):
            dataset_name = script_args.dataset_with_completions.split("/", 1)[1]
        else:
            dataset_name = str(script_args.dataset_with_completions)
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{dataset_name}_scores_checkpoint.pt",
        )

        # Initialize scores: load from checkpoint if exists, else from dataset or None
        if os.path.exists(checkpoint_path):
            print(f"Loading scores from checkpoint: {checkpoint_path}")
            scores = torch.load(checkpoint_path, weights_only=False)
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
        entries_to_score = [
            i for i, score in enumerate(scores) if score is None
        ]
        print(f"Found {len(entries_to_score)} entries that need scoring...")

        try:
            for n, idx in enumerate(
                tqdm(entries_to_score, desc="Scoring with Gemini API")
            ):
                elapsed_time = time.time() - start_time
                if query_count == (queries_per_minute - 1):
                    if elapsed_time < time_window:
                        wait_time = time_window - elapsed_time
                        print(
                            f"Wait {wait_time:.2f} seconds to avoid API rate limit..."
                        )
                        time.sleep(wait_time + 1)
                    query_count = 0
                    start_time = time.time()

                retry_count = 0
                while retry_count < server_max_retries:
                    try:
                        response = client.models.generate_content(
                            model=model,
                            contents=dataset[idx]["evaluator_prompt"],
                            config=types.GenerateContentConfig(
                                response_mime_type="text/x.enum",
                                response_schema=schema,
                                temperature=0,
                                max_output_tokens=1,
                                seed=script_args.seed,
                            ),
                        )
                        scores[idx] = self.gemini_score_response(response)
                        query_count += 1
                        break  # Success, break out of retry loop
                    except Exception as e:
                        error_str = str(e).lower()
                        if (
                            "unavailable" in error_str
                            or "overloaded" in error_str
                            or "overcharged" in error_str
                            or "server" in error_str
                        ) and retry_count < server_max_retries - 1:
                            print(
                                f"Server unavailable/overloaded at entry {idx}, attempt {retry_count + 1}/{server_max_retries}. Waiting {server_retry_wait} seconds before retrying..."
                            )
                            time.sleep(server_retry_wait)
                            retry_count += 1
                            continue
                        else:
                            raise  # Not a server error or max retries reached

                # Save progress periodically to local file
                if (n + 1) % save_frequency == 0:
                    print(
                        f"\nSaving progress locally after {n + 1} new entries..."
                    )
                    try:
                        torch.save(scores, checkpoint_path)
                    except Exception as e:
                        print(
                            f"Warning: Could not save intermediate progress locally: {e}"
                        )

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
        scores_tensor = torch.tensor(
            [s if s is not None else 0 for s in scores]
        )
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
            raise ValueError(
                "dataset_labels is required for autorater evaluation"
            )

        self.data = load_dataset(
            self.args.dataset_labels, split=self.args.dataset_labels_split
        )

    def process_data(self) -> None:
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

    def setup_model(self) -> None:
        # Load evaluator as causal LM and set eval mode
        if not self.args.use_gemini:
            self.evaluator = AutoModelForCausalLM.from_pretrained(
                self.args.evaluator_model,
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
            )
            self.evaluator.eval()

    def run_and_save(self) -> None:
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

        # Compute metrics
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
            fewshot_prompts = "\n".join(
                [ex["prompt"] for ex in fewshot_examples]
            )
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
        if "completion" in self.dataset_prompts.column_names:
            self.dataset_prompts = self.dataset_prompts.remove_columns(
                "completion"
            )
        self.dataset_prompts = self.dataset_prompts.add_column(
            "completion", generations
        )

        # OLD CONCISENESS CALCULATION
        # if "conciseness" in self.dataset_prompts.column_names:
        #     self.dataset_prompts = self.dataset_prompts.remove_columns(
        #         "conciseness"
        #     )
        # conciseness: list[float] = []
        # completions = self.dataset_prompts["completion"]
        # contexts = self.dataset_prompts["context"]
        # for completion, context in zip(completions, contexts):
        #     context_length = len(context) if context else 0
        #     completion_length = len(completion) if completion else 0
        #     ratio = (
        #         float(completion_length) / float(context_length)
        #         if context_length
        #         else 0.0
        #     )
        #     conciseness.append(ratio)
        # self.dataset_prompts = self.dataset_prompts.add_column(
        #     "conciseness", conciseness
        # )

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

        if self.args.gsheets_name:
            if "SFT" in self.args.writer_model_lora:
                ws_eval_gen = setup_gsheets(self.args.gsheets_name, "SFT")
            elif "PERL" in self.args.writer_model_lora:
                ws_eval_gen = setup_gsheets(self.args.gsheets_name, "PERL")
            else:
                print("Failed to open the sheet!")
                return
            
            update_gsheets_evaluation_generation(
                ws_eval_gen,
                self.args,
                name_for_saving,
            )


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
            raise ValueError(
                "dataset_with_completions is required for scoring mode"
            )
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
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
            )
            self.evaluator.eval()

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
