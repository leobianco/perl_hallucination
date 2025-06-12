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
import time
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from google import genai
from google.genai import types
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


@dataclass
class ScriptArguments:
    task: str = field(metadata={"help": "Name of the task (NPOV, HalOmi)."})

    user: str = field(
        metadata={
            "help": "The user to use for writing and loading to and from HF."
        },
    )

    writer_model_lora: str = field(
        metadata={"help": "The path to the LoRA adapters of the writer model."}
    )

    dataset_labels: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset with hallucination labels, for evaluation of evaluator or for getting few-shot examples."
        },
    )

    dataset_labels_split: Optional[str] = field(
        default=None,
        metadata={"help": "What split of the dataset_labels to use."},
    )

    dataset_prompts: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset with prompts to be used for generation (not necessarily has hallucination labels)."
        },
    )

    dataset_prompts_split: Optional[str] = field(
        default=None,
        metadata={"help": "What split of the dataset_prompts to use."},
    )

    writer_model_base: Optional[str] = field(
        default=None,
        metadata={"help": "The base model for the writer (name or path)."},
    )

    evaluator_model: str = field(
        default="google/gemma-2-27b-it",
        metadata={
            "help": "The model name or path to the model to use as evaluator."
        },
    )

    use_gemini: bool = field(
        default=True,
        metadata={"help": "Using the latest Gemini model as evaluator"},
    )

    gemini_api_key: Optional[str] = field(
        default="", metadata={"help": "API key for calling Gemini"}
    )

    seed: int = field(default=12345)

    eval_batch_size: int = field(default=1)

    max_tokens: int = field(default=128)

    temperature: float = field(default=1)

    top_p: float = field(default=1)

    evaluator_num_fewshot: Optional[int] = field(
        default=0,
        metadata={
            "help": "The number of fewshot examples to give to the evaluator. Half will be positive (contain hallucination), half will be negative."
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

    dataset_with_completions: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the dataset with completions on HF Hub. Required when evaluate_evaluator is False and not generating."
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


def bosch_evaluator_prompt(entry, fewshot_examples=None, use_true_label=False):
    """Function for transforming entries in the Bosch dataset into prompts
    for the evaluator model.

    TO DO (LEO): perhaps split this function into two functions instead of
    using the use_true_label parameter. Either way you must do something, because
    it looks really bad.
    """

    preamble = """<start_of_turn>user
You are an expert linguist and fact-checker responsible for determining when an answer contains information not supported by a source text.

In this task:
1. A user asks a question about their car
2. You receive an excerpt from the car's manual
3. You examine a proposed answer to the user's question

Your ONLY job is to determine: Does the proposed answer contain ANY statements, claims, instructions, or implications that are NOT explicitly supported by the manual excerpt?

Important evaluation criteria:
- If the answer makes ANY claim not directly supported by the manual, respond with "Yes"
- If the answer contains any logical implication not explicitly appearing in the manual excerpt, valid or not, respond with "Yes"
- If the answer only rephrases information from the manual without adding new claims, respond with "No"
- If the answer is FULLY supported by the manual, but truncated at the last sentence (due to word limit), answer "No"
- Introduction sentences such as "Follow these steps:" and conclusions such as "I hope these helps!" are allowed. If all other sentences are supported by the context, answer "No"
- Ignore differences in tone, style, or level of detail if the underlying information is supported
- Focus solely on faithfulness to the source, not helpfulness or completeness

Provide ONLY "Yes" or "No" as your final answer.
<end_of_turn>
"""

    prompt = preamble

    template = """<start_of_turn>user
Question: {question}

Manual excerpt: {context}

Proposed answer: {response}

Analysis:
1. Compare each statement in the proposed answer to the manual excerpt
2. Identify any claims in the answer not explicitly supported by the manual
3. Consider whether the answer introduces new information not present in the manual

Does the proposed answer state anything not supported by the information in the manual? (Yes/No):
<end_of_turn>
<start_of_turn>model
{ans}
"""

    # Depending if evaluation of evaluator or of writer checkpoint.
    response = entry["response"] if use_true_label else entry["completion"]

    formatted_prompt = template.format(
        question=entry["Question"],
        context=entry["Context"],
        response=response,
        ans="",
    )

    if fewshot_examples is not None:
        for fewshot_example in fewshot_examples:
            fewshot_prompt = template.format(
                question=fewshot_example["Question"],
                context=fewshot_example["Context"],
                response=fewshot_example["response"],
                ans=fewshot_example["class_hall"],
            )
            prompt += fewshot_prompt + "<end_of_turn>\n"
        prompt += formatted_prompt
    else:
        prompt += formatted_prompt

    entry["evaluator_prompt"] = prompt

    return entry


def ragtruth_evaluator_prompt(
    entry, fewshot_examples=None, use_true_label=False
):
    """Function for transforming entries in the RAGTruth dataset into prompts
    for the evaluator model.

    TO DO (LEO): perhaps split this function into two functions instead of
    using the use_true_label parameter. Either way you must do something, because
    it looks really bad.
    """

    preamble = """You are an expert evaluator specializing in detecting hallucinations in text summarization. Your task is to determine whether a summary contains any information not present in the original text.

In this evaluation:
1. You will be given an original text that was meant to be summarized
2. You will examine a summary generated from that text
3. You must determine if the summary contains ANY information, statements, claims, or implications that are NOT supported by the original text

Evaluation guidelines:
- A hallucination occurs when the summary includes facts, details, or conclusions absent from the original text
- Reasonable paraphrasing and reorganization of information is acceptable
- Focus only on factual accuracy, not on style, tone, or format
- Be vigilant about subtle additions that might seem plausible but aren't in the source

Your response must be ONLY "Yes" (hallucination detected) or "No" (no hallucination detected).
"""

    prompt = preamble

    template = """

Original text to be summarized: {user_query}
{response}

Evaluation process:
1. Read the original text carefully
2. Examine each claim or statement in the summary (output)
3. Verify that every piece of information in the summary (output) is supported by the original text
4. Check for subtle additions, expansions, or assumptions not justified by the original

Does the summary (output) contain ANY information not present in or directly inferable from the original text? (Yes/No):{ans}
"""

    # Depending if evaluation of evaluator or of writer checkpoint.
    response = entry["response"] if use_true_label else entry["completion"]

    formatted_prompt = template.format(
        user_query=entry["user_query"],
        response=response,
        ans="",
    )

    if fewshot_examples is not None:
        for fewshot_example in fewshot_examples:
            fewshot_prompt = template.format(
                user_query=fewshot_example["user_query"],
                response=fewshot_example["response"],
                ans=fewshot_example["class_hall"],
            )
            prompt += fewshot_prompt
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


def gemini_score_response(response):
    if response.candidates[0].avg_logprobs is None:
        return 0
    if response.text == "No":
        return np.exp(response.candidates[0].avg_logprobs)
    elif response.text == "Yes":
        return 1 - np.exp(response.candidates[0].avg_logprobs)
    else:
        raise Exception("Invalid response")


def gemini_score_dataset(client, dataset, script_args):
    """Score a whole dataset with Gemini API.

    Args:
        client: Initialized Gemini API client
        dataset: Dataset containing evaluator_prompt column
        script_args: Script arguments containing seed

    Returns:
        torch.tensor: Tensor of scores
    """
    model = "gemini-2.0-flash-001"
    schema = {"type": "STRING", "enum": ["No", "Yes"]}
    scores = []
    print("Calling the Gemini API...")

    queries_per_minute = 2000
    time_window = 60
    query_count = 0
    start_time = time.time()

    for query in tqdm(
        dataset["evaluator_prompt"], desc="Scoring with Gemini API"
    ):
        # Check if we are approaching the rate limit
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

        response = client.models.generate_content(
            model=model,
            contents=query,
            config=types.GenerateContentConfig(
                response_mime_type="text/x.enum",
                response_schema=schema,
                temperature=0,
                max_output_tokens=1,
                seed=script_args.seed,
            ),
        )
        scores.append(gemini_score_response(response))
        query_count += 1

    return torch.tensor(scores)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    set_seed(script_args.seed)

    # Load dataset with hallucination labels (for evaluating the
    # evaluator, or for getting fewshot examples).
    if script_args.dataset_labels is not None:
        data = load_dataset(
            script_args.dataset_labels,
            split=script_args.dataset_labels_split,
        )

        if script_args.task == "ragtruth":
            evaluator_prompt = ragtruth_evaluator_prompt
        elif script_args.task == "npov":
            evaluator_prompt = npov_evaluator_prompt
        elif script_args.task == "bosch":
            evaluator_prompt = bosch_evaluator_prompt

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

    if script_args.evaluate_evaluator:
        if script_args.dataset_labels is None:
            raise ValueError(
                "dataset_labels is required when evaluate_evaluator is True"
            )

        data = data.map(
            evaluator_prompt,
            fn_kwargs=dict(
                fewshot_examples=evaluator_fewshot_examples, use_true_label=True
            ),
        )

        if script_args.use_gemini:
            client = genai.Client(api_key=script_args.gemini_api_key)
            scores = gemini_score_dataset(client, data, script_args)

        else:
            tokenizer = AutoTokenizer.from_pretrained(
                script_args.evaluator_model,
            )
            yes_token_id = tokenizer.convert_tokens_to_ids("Yes")
            no_token_id = tokenizer.convert_tokens_to_ids("No")

            evaluator = AutoModelForCausalLM.from_pretrained(
                script_args.evaluator_model,
                device_map="auto",
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
            )
            evaluator.eval()

            scores = evaluator_score(
                data,
                script_args,
                tokenizer,
                evaluator,
                yes_token_id,
                no_token_id,
            )

        # Save results
        name_for_saving = (
            f"eval_autorater_{script_args.evaluator_model.split('/')[-1]}"
            + f"_autorater_num_fewshot_{script_args.evaluator_num_fewshot}"
            + f"_data_{script_args.dataset_labels.split('/')[-1]}"
            + f"_seed_{script_args.seed}"
        )

        # Calculate Metrics
        ground_truth = data["label"]

        # Save evaluator prompts
        filepath = (
            f"logs/{name_for_saving}/eval_autorater_evaluator_prompts.txt"
        )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            for idx, prompt in enumerate(data["evaluator_prompt"]):
                f.write(f"\n{idx}. ----------\n" + prompt)
        print(f"Evaluator prompts saved to {filepath}")

        # Save ground_truth labels to a file
        filepath = f"logs/{name_for_saving}/eval_autorater_ground_truth.txt"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            for label in ground_truth:
                f.write(f"{label}\n")
        print(f"Ground truth labels saved to {filepath}")

        # Save scores to a file
        filepath = f"logs/{name_for_saving}/eval_autorater_scores.txt"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            for score in scores:
                f.write(f"{score.item():.5f}\n")
        print(f"Scores saved to {filepath}")

        auc = roc_auc_score(ground_truth, scores)
        fpr, tpr, thresholds = roc_curve(ground_truth, scores.numpy())
        threshold_idx = np.argmax(tpr - fpr)
        threshold = thresholds[threshold_idx]
        classif_at_threshold = [
            0 if score < threshold else 1 for score in scores
        ]

        # Display Metrics and Save Plots
        metrics_filepath = f"logs/{name_for_saving}/eval_autorater_metrics.txt"
        os.makedirs(os.path.dirname(metrics_filepath), exist_ok=True)
        with open(metrics_filepath, "w") as f:
            f.write("AUC: {:.5f}\n".format(auc))
            f.write("Threshold: {:.5f}\n".format(threshold))
            f.write("TPR (recall): {:.5f}\n".format(tpr[threshold_idx]))
            f.write("FPR: {:.5f}\n".format(fpr[threshold_idx]))
            f.write(
                "Accuracy: {:.5f}\n".format(
                    accuracy_score(ground_truth, classif_at_threshold)
                )
            )
            f.write(
                "Precision: {:.5f}\n".format(
                    precision_score(ground_truth, classif_at_threshold)
                )
            )
        print(f"Metrics saved to {metrics_filepath}")

        print("AUC: {:.5f}".format(auc))
        print("Threshold: {:.5f}".format(threshold))
        print("TPR (recall): {:.5f}".format(tpr[threshold_idx]))
        print("FPR: {:.5f}".format(fpr[threshold_idx]))
        print(
            "Accuracy: {:.5f}".format(
                accuracy_score(ground_truth, classif_at_threshold)
            )
        )
        print(
            "Precision: {:.5f}".format(
                precision_score(ground_truth, classif_at_threshold)
            )
        )

        # ROC-AUC plot
        RocCurveDisplay.from_predictions(ground_truth, scores)
        plt.title(f"ROC Curve (Threshold: {threshold:.5f})")
        plt.scatter([fpr[threshold_idx]], [tpr[threshold_idx]], c="r")
        os.makedirs(f"logs/{name_for_saving}/", exist_ok=True)
        plt.savefig(
            f"logs/{name_for_saving}/"
            + f"eval_autorater_auc_curve_{script_args.evaluator_num_fewshot}_shot"
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
            + f"eval_autorater_{script_args.evaluator_num_fewshot}_shot"
        )

    elif script_args.dataset_with_completions is None:
        # Generation mode

        # Load dataset with prompts to generations (not necessarily labeled).
        dataset_prompts = load_dataset(
            script_args.dataset_prompts,
            split=script_args.dataset_prompts_split,
        )

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
            dtype="bfloat16",
        )

        # Generate completions with writer.
        sampling_params = SamplingParams(
            seed=script_args.seed,
            temperature=script_args.temperature,
            top_p=script_args.top_p,
            min_tokens=10,  # avoid empty generations
            max_tokens=script_args.max_tokens,
        )

        prompts = [
            dataset_prompts[i]["prompt"]
            for i in range(dataset_prompts.num_rows)
        ]

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

        # Free vLLM memory
        destroy_model_parallel()
        del llm.llm_engine.model_executor.driver_worker
        del llm
        gc.collect()
        torch.cuda.empty_cache()

        # Save generations locally
        try:
            name_for_saving = (
                "eval_"
                + script_args.writer_model_lora.split(f"{script_args.user}/")[1]
            )
        except Exception:
            name_for_saving = (
                "eval_" + script_args.writer_model_lora.split("/")[1]
            )

        filepath = f"logs/{name_for_saving}/generations.txt"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            for idx, generation in enumerate(generations):
                f.write(f"\n{idx}. ----------\n" + generation)
        print(f"Generations saved to {filepath}")

        # Add generations to the dataset_prompts under a column named "completion".
        dataset_prompts = dataset_prompts.add_column("completion", generations)

        # Save the dataset with completions to HF Hub
        print(
            f"Pushing dataset with completions to {script_args.user}/{name_for_saving}_completions"
        )
        dataset_name = f"{script_args.user}/{name_for_saving}_completions"
        dataset_prompts.push_to_hub(dataset_name)

    else:
        # Scoring mode

        # Load dataset with completions from HF Hub
        val_data = load_dataset(
            script_args.dataset_with_completions, split="test"
        )

        # Build the evaluator prompts using fewshot examples + generations.
        val_data = val_data.map(
            evaluator_prompt,
            fn_kwargs=dict(fewshot_examples=evaluator_fewshot_examples),
        )

        # Save evaluator prompts
        try:
            name_for_saving = (
                "eval_"
                + script_args.writer_model_lora.split(f"{script_args.user}/")[1]
            )
        except Exception:
            name_for_saving = (
                "eval_" + script_args.writer_model_lora.split("/")[1]
            )

        filepath = f"logs/{name_for_saving}/evaluator_prompts.txt"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            for idx, prompt in enumerate(val_data["evaluator_prompt"]):
                f.write(f"\n{idx}. ----------\n" + prompt)
        print(f"Evaluator prompts saved to {filepath}")

        # Score Completions

        # Instantiate evaluator.
        if script_args.use_gemini:
            client = genai.Client(api_key=script_args.gemini_api_key)
            scores = gemini_score_dataset(client, val_data, script_args)

        else:
            tokenizer = AutoTokenizer.from_pretrained(
                script_args.evaluator_model,
            )
            yes_token_id = tokenizer.convert_tokens_to_ids("Yes")
            no_token_id = tokenizer.convert_tokens_to_ids("No")

            evaluator = AutoModelForCausalLM.from_pretrained(
                script_args.evaluator_model,
                device_map="auto",
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
            )
            evaluator.eval()

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
        classifs = torch.ceil(t(scores)).clamp(0, 1)
        rate_hallucination = 1 - torch.mean(classifs)
        print("Rate of hallucination:", rate_hallucination.item())

        filepath = f"logs/{name_for_saving}/scores.txt"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            for idx, score in enumerate(scores):
                f.write("{:.5f}\n".format(score.item()))
        print(f"Scores saved to {filepath}")

        filepath = f"logs/{name_for_saving}/classifs.txt"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            for idx, classif in enumerate(classifs):
                f.write("{:.5f}\n".format(classif.item()))
        print(f"Classifs saved to {filepath}")
        print(f"Classif mean: {torch.mean(classifs)}")
