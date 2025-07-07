"""
reward_model.py

This module implements the training and evaluation pipeline for a reward model using sequence classification with LoRA (Low-Rank Adaptation) and Hugging Face Transformers. It supports augmenting datasets with organic and structured hallucination samples, computes ROC-AUC metrics, and pushes the trained model to the Hugging Face Hub.

Functions:
    main(): Entry point for training and evaluating the reward model.

Usage: call the associated shell script along with the corresponding task. E.g.:
    ./reward_model.sh npov
"""

import evaluate
import numpy as np
import torch
from datasets import Value, concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from utils import (
    LLMSynthScriptArguments,
    ScriptArguments,
    compute_best_roc_threshold,
    create_lora_argument_parser,
)


def main():
    """
    Trains and evaluates a reward model for sequence classification using LoRA and Hugging Face Transformers.

    This function parses command-line arguments for model, dataset, and training configuration, loads and optionally augments the dataset, tokenizes the data, prepares the model with LoRA, and trains the model. It computes evaluation metrics including ROC-AUC and pushes the trained model to the Hugging Face Hub.
    """

    # HfArgumentParser does not work with LoraConfig due to type hints
    parser_lora = create_lora_argument_parser()
    lora_args, remaining_args = parser_lora.parse_known_args()
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        task_type=lora_args.task_type,
        peft_type=lora_args.peft_type,
    )

    parser = HfArgumentParser(
        (
            ScriptArguments,
            LLMSynthScriptArguments,
            TrainingArguments,
        )
    )

    (
        script_args,
        llm_synth_args,
        training_args,
    ) = parser.parse_args_into_dataclasses(remaining_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_repo_id,
        padding_side="left",
    )

    # Some models (e.g. Mistral) don't have a pad token.
    # So we set it to some existing token.
    # Advice: avoid changing vocabulary size. It is possible, but it's a hassle.
    pad_token_modified = False
    if "pad_token" not in tokenizer.special_tokens_map.keys():
        tokenizer.pad_token = tokenizer.unk_token
        pad_token_modified = True

    def encode(examples):
        """Tokenizes the 'prompt' field in the dataset examples using the loaded tokenizer.

        Args:
            examples (dict): A batch of dataset examples with a 'prompt' field.

        Returns:
            dict: Tokenized examples as PyTorch tensors.
        """

        return tokenizer(
            examples["prompt"],
            padding=True,
            truncation=True,
            return_tensors="pt",  # using map() => set_format("torch") later
        )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if training_args.fp16:
        torch_dtype = torch.float16
    elif training_args.bf16:
        torch_dtype = torch.bfloat16
    else:
        raise Exception("Not training in mixed precision!")

    id2label = {
        0: "Yes",
        1: "No",
    }

    label2id = {
        "Yes": 0,
        "No": 1,
    }

    rm_data = load_dataset(script_args.dataset_repo_id)

    # In the case of LLM-generated synthetic hallucinations, we
    # might want to add some organic or structured hallucination
    # samples to the dataset. We do this here.
    if script_args.dataset_repo_id.endswith("synthetic_llm"):
        new_train_split = rm_data["train"]

        if llm_synth_args.num_organic_hallus_to_keep > 0:
            organic_dataset_name = (
                script_args.dataset_repo_id.removesuffix("synthetic_llm")
                + "organic"
            )
            organic_dataset = load_dataset(organic_dataset_name)
            organic_hallus_to_keep = (
                organic_dataset["train"]
                .filter(lambda x: x["class_hall"] == "Yes")
                .shuffle(seed=training_args.seed)
                .select(range(llm_synth_args.num_organic_hallus_to_keep))
            )
            new_train_split = concatenate_datasets(
                [new_train_split, organic_hallus_to_keep]
            )

        if llm_synth_args.num_struct_hallus_to_keep > 0:
            struct_dataset_name = (
                script_args.dataset_repo_id.removesuffix("synthetic_llm")
                + "synthetic_struct"
            )
            struct_dataset = load_dataset(struct_dataset_name)
            struct_hallus_to_keep = (
                struct_dataset["train"]
                .filter(lambda x: x["class_hall"] == "Yes")
                .shuffle(seed=training_args.seed)
                .select(range(llm_synth_args.num_struct_hallus_to_keep))
            )
            new_train_split = concatenate_datasets(
                [new_train_split, struct_hallus_to_keep]
            )

        # Mix them in training split
        rm_data["train"] = new_train_split.shuffle(seed=training_args.seed)

    for split in rm_data.keys():
        rm_data[split] = rm_data[split].map(
            encode,
            batched=True,
        )
        rm_data[split].set_format("torch")  # due to using map()
        # But not for the labels, those we need to be ints
        new_features = rm_data[split].features.copy()
        new_features["label"] = Value("int32")
        rm_data[split] = rm_data[split].cast(new_features)

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_repo_id,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )

    if pad_token_modified:
        reward_model.config.pad_token_id = tokenizer.pad_token_id

    reward_model = get_peft_model(reward_model, lora_config)

    # Loading LoRA freezes the projection layer at initialization value,
    # which we don't want! Let us make them trainable again.
    reward_model.score.requires_grad_()

    # Let us also multiply them by a small constant, to decrease the
    # deviation of the (observed) initial loss from its expected value
    with torch.no_grad():
        reward_model.score.weight.mul_(0.1)

    metric = evaluate.load("roc_auc")

    def compute_metrics(eval_preds):
        """Recall that whereas logits where torch tensors before, now they are
        numpy arrays.

        logits here are the processed_logits from the process_logits_for_evaluation
        function.

        metric is a 'global' function inside the scope of main.
        """

        # Compute scores
        logits = eval_preds.predictions
        yes_scores = np.exp(logits)[:, 0]
        no_scores = np.exp(logits)[:, 1]
        scores = no_scores / (yes_scores + no_scores)

        # Compute ground truth
        label_ids = eval_preds.label_ids

        # Compute AUC and log statistics at best threshold
        metrics = metric.compute(references=label_ids, prediction_scores=scores)
        threshold_metrics = compute_best_roc_threshold(label_ids, scores)
        metrics.update(threshold_metrics)

        # Compute and log average score for true positives and true negatives
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
        avg_score_metrics = {
            "avg_score_true_positives": avg_score_true_positives,
            "avg_score_true_negatives": avg_score_true_negatives,
        }
        metrics.update(avg_score_metrics)

        return metrics

    trainer = Trainer(
        model=reward_model,
        args=training_args,
        train_dataset=rm_data["train"],
        eval_dataset=rm_data["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    reward_model.push_to_hub(training_args.hub_model_id)


if __name__ == "__main__":
    main()
