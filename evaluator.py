"""
For an interactive Python shell after running evaluation, run:
python -m IPython -i evaluator.py \
        -- \
        --seed 130104 \
        --writer_model_base "google/gemma-2-2b-it" \
        --writer_model_lora "leobianco/HALOMI_SFT_seed_130104_epochs_1_lr_5e-5_lora_32" \
        --max_tokens 256 \
        --eval_batch_size 4 \
        --evaluator_model "google/gemma-2-27b-it" \
        --evaluator_num_fewshot 4 \
        --evaluate_evaluator False \
        --threshold 0.144

Evaluation script.

Given a checkpoint of the writer and validation prompts, we use the writer to 
generate completions, and send the prompts + completions to the evaluator. We 
then get the score for the token "No", normalized to consider only "Yes" as 
an alternative.

Checkpoints that we will consider: writer SFT (baseline), writer checkpoints 
over PERL. All of these are fine-tuned Gemma 2 2b models.

There is a separate script to evaluate the quality of the evaluator 
itself, and to choose the classification threshold.

The writer checkpoint will be loaded as a vLLM LLM.
"""

import gc
import os
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import snapshot_download
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.lora.request import LoRARequest

from data import npov_writer_prompt


@dataclass
class ScriptArguments:
    task: str = field(metadata={"help": "Name of the task (NPOV, HalOmi)."})

    user: str = field(
        metadata={
            "help": "The user to use for writing and loading to and from HF."
        },
    )

    dataset_labels: str = field(
        metadata={
            "help": "Dataset with hallucination labels, for evaluation of evaluator or for getting few-shot examples. Test split will be used!"
        }
    )

    dataset_prompts: str = field(
        metadata={
            "help": "Dataset with prompts to be used for generation (not necessarily has hallucination labels). Test split will be used!"
        }
    )

    writer_model_base: str = field(
        metadata={"help": "The base model for the writer (name or path)."}
    )

    writer_model_lora: str = field(
        metadata={"help": "The path to the LoRA adapters of the writer model."}
    )

    evaluator_model: str = field(
        default="google/gemma-2-27b-it",
        metadata={
            "help": "The model name or path to the model to use as evaluator."
        },
    )

    seed: int = field(default=12345)

    max_tokens: int = field(default=128)

    eval_batch_size: int = field(default=1)

    temperature: float = field(default=1)

    top_p: float = field(default=1)

    evaluator_num_fewshot: Optional[int] = field(
        default=0,
        metadata={
            "help": "The number of fewshot examples to give to the evaluator. Half will be positive (contain hallucination), half will be negative."
        },
    )

    writer_num_fewshot: Optional[int] = field(
        default=0,
        metadata={
            "help": "The number of fewshot examples to give to the writer. Must all be without hallucinations."
        },
    )

    evaluate_evaluator: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation of the evaluator or not."},
    )

    threshold: float = field(
        default=0.5,
        metadata={
            "help": "The value of the threshold to turn scores into classif."
        },
    )


def get_fewshot_examples(data, n_yes: int, n_no: int, seed: int):
    """Returns n balanced fewshot examples from data."""

    if n_yes == 0 and n_no == 0:
        fewshot_examples = None

        return fewshot_examples

    else:
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

        fewshot_examples = concatenate_datasets(
            [positive_examples, negative_examples]
        ).shuffle(seed=seed)

        return fewshot_examples


def halomi_evaluator_prompt(entry, fewshot_examples=None, use_true_label=False):
    """Function for transforming entries in the HalOmi dataset into prompts for
    the evaluator model.

    TO DO (LEO): perhaps split this function into two functions instead of
    using the use_true_label parameter. Either way you must do something, because
    it looks really bad.
    """

    preamble = (
        "<start_of_turn>user\nThe following are examples of an expert translator "
        "and linguist noting when the Translation of an Original text contains "
        "additional information that is not part of the original text.<end_of_turn>\n"
    )

    prompt = preamble

    template = (
        "<start_of_turn>user\n"
        "Original text in {src_lang}:{src_text}"
        "\nTranslated text in {tgt_lang}:{mt_text}"
        "\nExpert translator and linguist review: The Translated text contains "
        "additional information with respect to the Original text (Yes/No):"
        "<end_of_turn>\n<start_of_turn>model\n{ans}"
    )

    # Depending if evaluation of evaluator or of writer checkpoint.
    translation = entry["mt_text"] if use_true_label else entry["completion"]

    formatted_prompt = template.format(
        src_lang=entry["src_lang"],
        tgt_lang=entry["tgt_lang"],
        src_text=entry["src_text"],
        mt_text=translation,
        ans="",
    )

    if fewshot_examples is not None:
        for fewshot_example in fewshot_examples:
            fewshot_prompt = template.format(
                src_lang=fewshot_example["src_lang"],
                tgt_lang=fewshot_example["tgt_lang"],
                src_text=fewshot_example["src_text"],
                mt_text=fewshot_example["mt_text"],
                ans=fewshot_example["class_hall"],
            )
            prompt += fewshot_prompt + "<end_of_turn>\n"

        prompt += formatted_prompt

    else:
        prompt += formatted_prompt

    entry["evaluator_prompt"] = prompt

    return entry


def npov_evaluator_prompt(entry, fewshot_examples=None, use_true_label=False):
    """Function for transforming entries in the NPOV dataset into prompts for
    the evaluator model.

    TO DO (LEO): perhaps split this function into two functions instead of
    using the use_true_label parameter. Either way you must do something, because
    it looks really bad.
    """

    preamble = "<start_of_turn>user\nBelow are examples where an expert linguist identifies when the neutral natural language rewritings of arguments used to answer a user query contains additional arguments not present in the original list.<end_of_turn>\n"

    prompt = preamble

    template = "<start_of_turn>user\nUser query: {user_query}\n{perspective_1_name} arguments provided: {perspective_1}\n{perspective_2_name} arguments provided: {perspective_2}\nNeutral point-of-view answer to user query, rewriting provided arguments in natural language:{npov_response}\nExpert linguist review: the rewriting of the provided arguments contains additional arguments not present in the original list (Yes/No):<end_of_turn>\n<start_of_turn>model\n{ans}"

    # Depending if evaluation of evaluator or of writer checkpoint.
    response = entry["npov_response"] if use_true_label else entry["completion"]

    formatted_prompt = template.format(
        user_query=entry["user_query"],
        perspective_1_name=entry["perspective_1_name"],
        perspective_1=entry["perspective_1"],
        perspective_2_name=entry["perspective_2_name"],
        perspective_2=entry["perspective_2"],
        npov_response=response,
        ans="",
    )

    if fewshot_examples is not None:
        for fewshot_example in fewshot_examples:
            fewshot_prompt = template.format(
                user_query=fewshot_example["user_query"],
                perspective_1_name=fewshot_example["perspective_1_name"],
                perspective_1=fewshot_example["perspective_1"],
                perspective_2_name=fewshot_example["perspective_2_name"],
                perspective_2=fewshot_example["perspective_2"],
                npov_response=fewshot_example["npov_response"],
                ans=fewshot_example["class_hall"],
            )
            prompt += fewshot_prompt + "<end_of_turn>\n"
        prompt += formatted_prompt
    else:
        prompt += formatted_prompt

    entry["evaluator_prompt"] = prompt

    return entry


def evaluator_score_batch(
    evaluator, tokenized_prompts, yes_token_id, no_token_id
):
    """Evaluator scoring of a batch."""

    with torch.no_grad():
        # Cache in Gemma is different and was giving me errors, so I disable it.
        # See https://huggingface.co/docs/transformers/en/kv_cache#model-specific-cache-classes
        # See https://github.com/huggingface/transformers/issues/33147
        outputs = evaluator(**tokenized_prompts, use_cache=False)
        score_yes = torch.exp(outputs.logits[:, -1, yes_token_id])
        score_no = torch.exp(outputs.logits[:, -1, no_token_id])
        score_batch = score_no / (score_yes + score_no)

    return score_batch


def evaluator_score(
    data,
    script_args,
    tokenizer,
    evaluator,
    yes_token_id,
    no_token_id,
):
    """Scores the whole Dataset `data`, based on the column 'evaluator_prompt'."""

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

        score_batch = evaluator_score_batch(
            evaluator, tokenized_prompts, yes_token_id, no_token_id
        )

        score_batch = score_batch.cpu()
        scores = torch.cat((scores, score_batch))
        del tokenized_prompts

    return scores


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    set_seed(script_args.seed)

    try:
        name_for_saving = script_args.writer_model_lora.split(
            f"{script_args.user}/"
        )[1]
    except:
        name_for_saving = script_args.writer_model_lora.split("/")[1]

    # Load dataset with hallucination labels (for evaluating the
    # evaluator, or for getting fewshot examples).
    data = load_dataset(
        script_args.dataset_labels,
        split="test",
    )

    evaluator_prompt = (
        halomi_evaluator_prompt
        if script_args.task == "halomi"
        else npov_evaluator_prompt
    )

    # Get fewshot examples to aid the evaluator. These come from the dataset
    # with labels.
    if script_args.evaluator_num_fewshot == 0:
        evaluator_fewshot_examples = None
    else:
        evaluator_fewshot_examples = get_fewshot_examples(
            data,
            script_args.evaluator_num_fewshot // 2,
            script_args.evaluator_num_fewshot // 2,
            seed=script_args.seed,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.evaluator_model,
    )
    yes_token_id = tokenizer.convert_tokens_to_ids("Yes")
    no_token_id = tokenizer.convert_tokens_to_ids("No")

    if script_args.evaluate_evaluator:
        evaluator = AutoModelForCausalLM.from_pretrained(
            script_args.evaluator_model,
            device_map="auto",
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        )
        evaluator.eval()

        data = data.map(
            evaluator_prompt,
            fn_kwargs=dict(
                fewshot_examples=evaluator_fewshot_examples, use_true_label=True
            ),
        )

        scores = evaluator_score(
            data,
            script_args,
            tokenizer,
            evaluator,
            yes_token_id,
            no_token_id,
        )

        # Calculate Metrics
        ground_truth = data["class_hall_num"]
        auc = roc_auc_score(ground_truth, scores)
        fpr, tpr, thresholds = roc_curve(ground_truth, scores.numpy())
        threshold_idx = np.argmax(tpr - fpr)
        threshold = thresholds[threshold_idx]
        classif_at_threshold = [
            0 if score < threshold else 1 for score in scores
        ]

        # Display Metrics and Save Plots
        print("AUC:", auc)
        print("Threshold:", threshold)
        print("TPR (recall):", tpr[threshold_idx])
        print("FPR:", fpr[threshold_idx])
        print("Accuracy:", accuracy_score(ground_truth, classif_at_threshold))
        print("Precision:", precision_score(ground_truth, classif_at_threshold))

        # ROC-AUC plot
        RocCurveDisplay.from_predictions(ground_truth, scores)
        plt.scatter([fpr[threshold_idx]], [tpr[threshold_idx]], c="r")
        plt.savefig(
            f"logs/{name_for_saving}/"
            + f"eval_evaluator_auc_curve_{script_args.evaluator_num_fewshot}_shot"
        )
        plt.clf()

        # Histogram
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
            + f"eval_evaluator_histogram_{script_args.evaluator_num_fewshot}_shot"
        )

    else:
        # Load dataset with prompts to generations (not necessarily labeled).
        val_data = load_dataset(
            script_args.dataset_prompts,
            split="test",
        )

        # Generate Completions

        # To evaluate the base model (no LoRA), just re-use the base model path
        # on the LoRA adapters path.
        enable_lora = (
            False
            if script_args.writer_model_base == script_args.writer_model_lora
            else True
        )

        if enable_lora:
            # Check if the path given is local, and if not, download from HF
            if os.path.exists(script_args.writer_model_lora):
                lora_path = script_args.writer_model_lora
                print("Local LoRA path found:", lora_path)
            else:
                # Dowload the LoRA adapters and save locally.
                lora_path = snapshot_download(
                    repo_id=script_args.writer_model_lora,
                    allow_patterns=["*.json", "*.safetensors"],
                )
                print("LoRA path (downloaded from remote):", lora_path)

        # Instantiate evaluated checkpoint as a vLLM LLM.
        llm = LLM(
            model=script_args.writer_model_base,
            enable_lora=enable_lora,
            max_lora_rank=64,  # currently maximum available in vLLM.
            dtype="bfloat16",  # important for Gemma.
        )

        # Generate completions with writer.
        sampling_params = SamplingParams(
            seed=script_args.seed,
            temperature=script_args.temperature,
            top_p=script_args.top_p,
            min_tokens=10,  # avoid empty generations
            max_tokens=script_args.max_tokens,
        )

        prompts = [val_data[i]["prompt"] for i in range(val_data.num_rows)]

        # TODO: once new evaluation set is available, fix this.
        # CUSTOM RUNS: this is a temporary, _ugly_ solution to run evaluation
        # on the hc_rm5x_test split. It should be changed once the new eval
        # set is available. The problem is that the "prompt" column of the RM
        # dataset contains the answer, and here we want to generate it.
        # The solution is to call npov_writer_prompt from data.py.
        if script_args.dataset_prompts == "leobianco/npov_rm_processed":
            del prompts
            train_data = load_dataset(
                script_args.dataset_prompts,
                split="train",
            )
            writer_fewshot_examples = get_fewshot_examples(
                train_data,
                n_yes=0,
                n_no=script_args.writer_num_fewshot,
                seed=script_args.seed,
            )
            prompts = val_data.map(
                npov_writer_prompt,
                fn_kwargs=dict(fewshot_examples=writer_fewshot_examples),
            )["prompt"]

        if enable_lora:
            outputs = llm.generate(
                prompts,
                sampling_params,
                lora_request=LoRARequest(
                    "writer_lora_adapter", 1, lora_path=lora_path
                ),
            )
        else:
            outputs = llm.generate(prompts, sampling_params)

        generations = [output.outputs[0].text for output in outputs]

        filepath = f"logs/{name_for_saving}/generations.txt"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            for idx, generation in enumerate(generations):
                f.write(f"\n{idx}. ----------\n" + generation)
        print(f"Generations saved to {filepath}")

        # Add generations to the val_data under a column named "completion".
        val_data = val_data.add_column("completion", generations)

        # Free vLLM memory to free up space for evaluator.
        destroy_model_parallel()
        del llm.llm_engine.model_executor.driver_worker
        del llm
        gc.collect()
        torch.cuda.empty_cache()

        # Score Completions

        # Instantiate evaluator.
        evaluator = AutoModelForCausalLM.from_pretrained(
            script_args.evaluator_model,
            device_map="auto",
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        )
        evaluator.eval()

        # Build the evaluator prompts using fewshot examples + generations.
        val_data = val_data.map(
            evaluator_prompt,
            fn_kwargs=dict(fewshot_examples=evaluator_fewshot_examples),
        )

        filepath = f"logs/{name_for_saving}/evaluator_prompts.txt"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            for idx, prompt in enumerate(val_data["evaluator_prompt"]):
                f.write(f"\n{idx}. ----------\n" + prompt)
        print(f"Evaluator prompts saved to {filepath}")

        scores = evaluator_score(
            val_data,
            script_args,
            tokenizer,
            evaluator,
            yes_token_id,
            no_token_id,
        )

        # Calculate Metrics
        t = nn.Threshold(script_args.threshold, 0, inplace=False)
        classifs = torch.ceil(t(scores))
        rate_hallucination = 1 - torch.mean(classifs)
        print("Rate of hallucination:", rate_hallucination.item())

        filepath = f"logs/{name_for_saving}/scores.txt"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            for idx, score in enumerate(scores):
                f.write("{:.3f}\n".format(score.item()))
        print(f"Scores saved to {filepath}")
