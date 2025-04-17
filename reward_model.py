""" "
TODO: write proper docstring.
TODO: clean imports.
"""

import os

import torch
import evaluate
import numpy as np
from datasets import load_dataset
from peft import PeftModel, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from utils import CustomLoraConfig, ScriptArguments


def main():
    # SETUP
    parser = HfArgumentParser(
        (ScriptArguments, TrainingArguments, CustomLoraConfig)
    )

    (
        script_args,
        training_args,
        peft_args,
    ) = parser.parse_args_into_dataclasses()

    name_for_saving = training_args.run_name.split("/")[1]

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_repo_id,
        padding_side="left",
    )
    # Make special tokens (pad, end of sequence) common to all models
    tokenizer.add_special_tokens(
        {
            "pad_token": "<pad>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
        }
    )

    if training_args.fp16:
        torch_dtype = torch.float16
    elif training_args.bf16:
        torch_dtype = torch.bfloat16
    else:
        raise Exception("Not training in mixed precision!")

    # DATA
    rm_data = load_dataset(script_args.dataset_repo_id)

    id2label = {
        0: "Yes",
        1: "No",
    }

    label2id = {
        "Yes": 0,
        "No": 1,
    }

    # MODEL
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_repo_id,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )

    reward_model = get_peft_model(reward_model, peft_args)

    # EVALUATION SETUP
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

        metrics = metric.compute(references=label_ids, prediction_scores=scores)

        return metrics

    # TRAINING
    trainer = Trainer(
        model=reward_model,
        args=training_args,
        train_dataset=rm_data["train"],
        eval_dataset=rm_data["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        reward_model.save_pretrained(f"checkpoints/{name_for_saving}/")

        if training_args.push_to_hub:
            reward_model.push_to_hub(training_args.hub_model_id)

    # EVALUATION
    if training_args.do_eval:
        trainer.evaluate()

    filepath = f"logs/{name_for_saving}/logs.txt"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for d in trainer.state.log_history:
            f.write(str(d) + "\n----------\n")
    print(f"Logs saved to {filepath}")


if __name__ == "__main__":
    main()
