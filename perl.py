"""
TODO: write proper docstring.
"""

import os

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from trl import RLOOConfig, RLOOTrainer

from utils import ScriptArguments


def main():
    parser = HfArgumentParser(
        (
            ScriptArguments,
            RLOOConfig,
        )
    )

    (
        script_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    name_for_saving = training_args.run_name.split("/")[1]
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.sft_model_path,
        padding_side="left",
    )

    # Tokenize data
    def encode(examples):
        return tokenizer(
            examples["prompt"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    perl_data = load_dataset(script_args.dataset_repo_id)
    for split in perl_data.keys():
        perl_data[split] = perl_data[split].map(
            encode,
            remove_columns=perl_data[split].column_names,
            batched=True,
        )
        perl_data[split].set_format("torch")

    id2label = {
        0: "Yes",
        1: "No",
    }

    label2id = {
        "Yes": 0,
        "No": 1,
    }

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )

    ref_policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )

    policy_base = AutoModelForCausalLM.from_pretrained(
        script_args.model_repo_id,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )

    policy = PeftModel.from_pretrained(
        policy_base,
        training_args.sft_model_path,
        is_trainable=True,  # loading adapters this way prevents overwriting
    )

    trainer = RLOOTrainer(
        config=training_args,
        processing_class=tokenizer,
        ref_policy=ref_policy,
        policy=policy,
        reward_model=reward_model,
        train_dataset=perl_data["train"],
        eval_dataset=perl_data["test"],
    )

    if training_args.do_train:
        trainer.train()

    filepath = f"logs/{name_for_saving}/logs.txt"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for d in trainer.state.log_history:
            f.write(str(d) + "\n----------\n")
    print(f"Logs saved to {filepath}")


if __name__ == "__main__":
    main()
