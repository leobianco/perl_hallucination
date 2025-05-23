""" "
TODO: write proper docstring.
TODO: clean imports.
"""

import os

import evaluate
import numpy as np
import torch
from datasets import Value, load_dataset
from peft import get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from utils import CustomLoraConfig, ScriptArguments


def main():
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

    # Some models (e.g. Mistral) don't have a pad token.
    # So we set it to some existing token.
    # Advice: avoid changing vocabulary size. It is possible, but it's a hassle.
    pad_token_modified = False
    if "pad_token" not in tokenizer.special_tokens_map.keys():
        tokenizer.pad_token = tokenizer.unk_token
        pad_token_modified = True

    def encode(examples):
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

    reward_model = get_peft_model(reward_model, peft_args)

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

    trainer = Trainer(
        model=reward_model,
        args=training_args,
        train_dataset=rm_data["train"],
        eval_dataset=rm_data["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        reward_model.save_pretrained(f"checkpoints/{name_for_saving}/")

        if training_args.push_to_hub:
            reward_model.push_to_hub(training_args.hub_model_id)

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
