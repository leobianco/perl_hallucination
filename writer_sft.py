"""
This module provides the main script for supervised fine-tuning (SFT) of language models using LoRA (Low-Rank Adaptation) and the TRL library.

It loads datasets, configures LoRA and SFT training arguments, prepares tokenizers and data collators, and launches training for various prompt-based tasks (e.g., npov, bosch, ragtruth).

Functions:
    main: Entry point for parsing arguments, preparing data, configuring the model, and running SFT training.

Usage: call the associated shell script along with the corresponding task. E.g.:
    ./writer_sft.sh npov
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import (
    DataCollatorForCompletionOnlyLM,
    SFTConfig,
    SFTTrainer,
    TrlParser,
)

from data.npov_task_processor import NPOVTaskProcessor
from data.bosch_task_processor import BoschTaskProcessor
from data.ragtruth_task_processor import RagtruthTaskProcessor
from utils import ScriptArguments, create_lora_argument_parser


def main():
    """
    Main entry point for supervised fine-tuning (SFT) with LoRA and TRL.

    This function parses LoRA and SFT arguments, loads datasets, prepares the tokenizer and data collator,
    configures the model with LoRA, and launches SFT training for the specified task.

    Raises:
        Exception: If the response template is not specified for the selected model.
    """

    # Parse LoRA arguments first
    parser_lora = create_lora_argument_parser()
    lora_args, remaining_args = parser_lora.parse_known_args()
    lora_config = LoraConfig(
        task_type=lora_args.task_type,
        peft_type=lora_args.peft_type,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
    )

    # Parse the rest of the arguments
    parser = TrlParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_into_dataclasses(
        remaining_args
    )
    set_seed(training_args.seed)

    sft_data = load_dataset(script_args.dataset_repo_id, split="train")
    eval_data = load_dataset(script_args.dataset_repo_id, split="test")

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_repo_id,
        padding_side="left",
    )

    # Some models (e.g. Mistral) don't have a pad token.
    pad_token_modified = False
    if "pad_token" not in tokenizer.special_tokens_map.keys():
        tokenizer.pad_token = tokenizer.unk_token
        pad_token_modified = True

    task_map = {
        "npov": NPOVTaskProcessor,
        "bosch": BoschTaskProcessor,
        "ragtruth": RagtruthTaskProcessor,
    }

    processor_cls = task_map.get(script_args.task_name)
    if processor_cls is None:
        raise Exception(f"Unknown task: {script_args.task_name}")

    fewshot_examples = None
    if (
        script_args.task_name == "npov"
        and script_args.num_fewshot is not None
        and script_args.num_fewshot > 0
    ):
        fewshot_examples = sft_data.shuffle(seed=training_args.seed).select(
            range(script_args.num_fewshot)
        )

    formatting_prompts_func, response_template = (
        processor_cls.get_formatting_prompts_and_response_template(
            eos_token=tokenizer.eos_token,
            fewshot_examples=fewshot_examples,
            model_repo_id=script_args.model_repo_id,
        )
    )

    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )
    collator_completions = DataCollatorForCompletionOnlyLM(
        response_template_ids,
        tokenizer=tokenizer,
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_repo_id,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )

    # If pad token was added, need to resize embeddings.
    if pad_token_modified:
        model.config.pad_token_id = tokenizer.pad_token_id

    model = get_peft_model(model, lora_config)

    trainer = SFTTrainer(
        model,
        args=training_args,
        data_collator=collator_completions,
        train_dataset=sft_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
    )

    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
