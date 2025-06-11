"""TODO: write docstring."""

import torch
from datasets import load_dataset
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import (
    DataCollatorForCompletionOnlyLM,
    SFTConfig,
    SFTTrainer,
    TrlParser,
)

from data import (
    bosch_formatting_prompts_func,
    npov_formatting_prompts_func,
    npov_formatting_prompts_func_from_fewshot_examples,
    ragtruth_formatting_prompts_func,
)
from utils import CustomLoraConfig, ScriptArguments


def main():
    parser = TrlParser((ScriptArguments, SFTConfig, CustomLoraConfig))
    (script_args, training_args, peft_args) = (
        parser.parse_args_into_dataclasses()
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

    # For training on completions only.
    if script_args.task == "npov":
        # Response template is model dependent... 
        # For Mistral, "point-of-view" forward.
        # For Gemma, "\nNeutral point-of-view" forward... 
        # This is because the tokenizer tokenizes differently depending 
        # on context.
        model_company = script_args.model_repo_id.split("/")[0]
        if model_company=="google":
            response_template = "\nNeutral point-of-view answer to user query, rewriting provided arguments in natural language:\n"
        elif model_company=="mistralai":
            response_template = "point-of-view answer to user query, rewriting provided arguments in natural language:\n"
        else:
            raise Exception("Response template not specified for model!")

        if script_args.num_fewshot == 0 or script_args.num_fewshot is None:
            formatting_prompts_func = npov_formatting_prompts_func(tokenizer.eos_token)
        else:
            # Get fewshot examples and add to formatting_prompts_func
            fewshot_examples = sft_data.shuffle(seed=training_args.seed).select(
                range(script_args.num_fewshot)
            )
            formatting_prompts_func = (
                npov_formatting_prompts_func_from_fewshot_examples(
                    fewshot_examples=fewshot_examples,
                    eos_token=tokenizer.eos_token,
                )
            )
    elif script_args.task == "bosch":
        response_template = (
            "\nAnswer to user's question:\n"
        )
        formatting_prompts_func = bosch_formatting_prompts_func(tokenizer.eos_token)
    elif script_args.task == "ragtruth":
        response_template = "\n\noutput:\n"
        formatting_prompts_func = ragtruth_formatting_prompts_func(tokenizer.eos_token)

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

    model = get_peft_model(model, peft_args)

    trainer = SFTTrainer(
        model,
        args=training_args,
        data_collator=collator_completions,
        train_dataset=sft_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
    )

    if training_args.do_train:
        trainer.train()


if __name__ == "__main__":
    main()
