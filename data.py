"""Data processing functions for the HalOmi and NPOV datasets.

This script contains functions for loading, processing, and saving the HalOmi
and NPOV to Hugging Face Hub. Call this script via the shell script data.sh
with the dataset name as an argument ("halomi" or "npov").
"""

import os
import random
from argparse import ArgumentParser
from copy import deepcopy
from itertools import combinations

import nltk
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from google import genai
from google.genai import types

if not os.path.exists(os.path.expanduser("~/nltk_data/tokenizers/punkt_tab")):
    nltk.download("punkt_tab")

##################
# NPOV FUNCTIONS #
##################


# Preprocessing


def npov_hallucination_labels_to_numerical(entry):
    """Data processing utility function which replaces the hallucination labels by
    a numerical version."""

    entry["class_hall_num"] = 0 if entry["class_hall"] == "Yes" else 1

    return entry


def npov_omission_labels_to_numerical(entry):
    """Data processing utility function which replaces the omission labels by
    a numerical version."""

    entry["class_omit_num"] = 0 if entry["class_omit"] == "Yes" else 1

    return entry


def npov_change_hallucination_labels(entry):
    """Data processing utility function which replaces the hallucination labels by
    a simplified version.
    """

    class_hallucination_to_label = {
        "NO": "No",
        "No": "No",
        "YES": "Yes",
        "Yes": "Yes",
    }

    entry["class_hall"] = class_hallucination_to_label[entry["class_hall"]]

    entry["has synthetic hallucination"] = class_hallucination_to_label[
        entry["has synthetic hallucination"]
    ]

    return entry


def npov_change_omission_labels(entry):
    """Data processing utility function which replaces the omission labels by
    a simplified version.
    """

    class_omission_to_label = {
        "NO": "No",
        "No": "No",
        "YES": "Yes",
        "Yes": "Yes",
    }

    entry["class_omit"] = class_omission_to_label[entry["class_omit"]]

    entry["has synthetic coverage issue"] = class_omission_to_label[
        entry["has synthetic coverage issue"]
    ]

    return entry


# Reward Model


def npov_rm_prompt(entry):
    """Function for transforming entries in the NPOV dataset into training
    prompts for the reward model.
    """

    template = (
        "User query: {user_query}\n"
        "{perspective_1_name} arguments provided: {perspective_1}\n"
        "{perspective_2_name} arguments provided: {perspective_2}\n"
        "Neutral point-of-view answer to user query, rewriting provided"
        " arguments in natural language:\n"
        "{npov_response}"
    )

    formatted_prompt = template.format(
        user_query=entry["user_query"],
        perspective_1_name=entry["perspective_1_name"],
        perspective_1=entry["perspective_1"],
        perspective_2_name=entry["perspective_2_name"],
        perspective_2=entry["perspective_2"],
        npov_response=entry["npov_response"],
    )

    entry["prompt"] = formatted_prompt

    return entry


def npov_process_data_for_rm(
    npov_data,
):
    # Copy original NPOV data
    npov_rm_data = deepcopy(npov_data)

    if "has hallucination" in npov_rm_data.column_names:
        npov_rm_data = npov_rm_data.rename_column(
            "has hallucination", "class_hall"
        )
    if "has coverage issue" in npov_rm_data.column_names:
        npov_rm_data = npov_rm_data.rename_column(
            "has coverage issue", "class_omit"
        )

    npov_rm_data = npov_rm_data.select_columns(
        [
            "topic",
            "user_query",
            "npov_response",
            "perspective_1",
            "perspective_1_name",
            "perspective_2",
            "perspective_2_name",
            "class_hall",
            "has synthetic hallucination",
            "class_omit",
            "has synthetic coverage issue",
        ]
    )
    npov_rm_data = npov_rm_data.map(npov_change_hallucination_labels)
    npov_rm_data = npov_rm_data.map(npov_change_omission_labels)
    npov_rm_data = npov_rm_data.map(npov_hallucination_labels_to_numerical)
    npov_rm_data = npov_rm_data.map(npov_omission_labels_to_numerical)
    npov_rm_data = npov_rm_data.map(npov_rm_prompt)
    npov_rm_data = npov_rm_data.rename_column("class_hall_num", "label")

    return npov_rm_data


# Writer SFT


def npov_process_data_for_sft(npov_data):
    """Preprocesses NPOV SFT dataset."""

    # Select relevant subset of columns
    npov_data = npov_data.select_columns(
        [
            "topic",
            "user_query",
            "npov_response_combined",
            "perspective_1",
            "perspective_1_name",
            "perspective_2",
            "perspective_2_name",
        ]
    )

    npov_data = npov_data.rename_column(
        "npov_response_combined", "npov_response"
    )

    return npov_data


def npov_writer_prompt(entry, SFT=False, fewshot_examples=None):
    """Function for transforming entries in the NPOV dataset into prompts for
    the writer to rewrite.

    Previously used for SFT, but now used as prompt for generation during PERL.
    The SFT now uses the "npov_formatting_prompts_func" function, which has
    the same prompt as this one, but is used by the SFTTrainer differently.
    """

    template = (
        "User query: {user_query}\n"
        "{perspective_1_name} arguments provided: {perspective_1}\n"
        "{perspective_2_name} arguments provided: {perspective_2}\n"
        "Neutral point-of-view answer to user query, rewriting provided"
        " arguments in natural language:\n"
        "{npov_response}"
    )

    npov_response = (
        entry["npov_response"] if SFT else ""
    )

    formatted_prompt = template.format(
        user_query=entry["user_query"],
        perspective_1_name=entry["perspective_1_name"],
        perspective_1=entry["perspective_1"],
        perspective_2_name=entry["perspective_2_name"],
        perspective_2=entry["perspective_2"],
        npov_response=npov_response,
    )

    prompt = ""

    if fewshot_examples is not None:
        preamble = (
            "Your task is to answer an user's query"
            " by rewriting the provided arguments in natural language. Do not "
            "generate arguments other than those provided. "
            f"We provide {fewshot_examples.num_rows} example(s) of what is "
            "expected, then it is your turn.\n"
        )

        prompt += preamble

        for fewshot_example in fewshot_examples:
            fewshot_prompt = template.format(
                user_query=fewshot_example["user_query"],
                perspective_1_name=fewshot_example["perspective_1_name"],
                perspective_1=fewshot_example["perspective_1"],
                perspective_2_name=fewshot_example["perspective_2_name"],
                perspective_2=fewshot_example["perspective_2"],
                npov_response=fewshot_example["npov_response"],
            )
            prompt += fewshot_prompt + "\n"

        prompt += formatted_prompt

    else:
        prompt += formatted_prompt

    entry["prompt"] = prompt

    return entry


def npov_formatting_prompts_func(entry):
    """Formatting function for SFTTrainer. Imported in writer_sft.py."""

    template = (
        "User query: {user_query}\n"
        "{perspective_1_name} arguments provided: {perspective_1}\n"
        "{perspective_2_name} arguments provided: {perspective_2}"
    )

    output_texts = []

    for i in range(len(entry["user_query"])):
        formatted_prompt = template.format(
            user_query=entry["user_query"][i],
            perspective_1_name=entry["perspective_1_name"][i],
            perspective_1=entry["perspective_1"][i],
            perspective_2_name=entry["perspective_2_name"][i],
            perspective_2=entry["perspective_2"][i],
        )

        text = (
            f"{formatted_prompt}\nNeutral point-of-view answer to user query, "
            "rewriting provided arguments in natural language:\n"
            f"{entry['npov_response'][i]}"
        )

        output_texts.append(text)

    return output_texts


def npov_formatting_prompts_func_from_fewshot_examples(fewshot_examples):
    """Given a fewshot example, returns a formatting prompts function for writer SFT that includes the given fewshot example in its prompt."""

    preamble = (
        "Your task is to answer an user's query"
        " by rewriting the provided arguments in natural language. Do not "
        "generate arguments other than those provided. "
        f"We provide {fewshot_examples.num_rows} example(s) of what is "
        "expected, then it is your turn.\n"
    )

    prompt = preamble

    template_fewshot = (
        "User query: {user_query}\n"
        "{perspective_1_name} arguments provided: {perspective_1}\n"
        "{perspective_2_name} arguments provided: {perspective_2}\n"
        "Example neutral point-of-view answer to user query, rewriting provided"
        " arguments in natural language:\n"
        "{npov_response}"
    )

    for fewshot_example in fewshot_examples:
        fewshot_prompt = template_fewshot.format(
            user_query=fewshot_example["user_query"],
            perspective_1_name=fewshot_example["perspective_1_name"],
            perspective_1=fewshot_example["perspective_1"],
            perspective_2_name=fewshot_example["perspective_2_name"],
            perspective_2=fewshot_example["perspective_2"],
            npov_response=fewshot_example["npov_response"],
        )
        prompt += fewshot_prompt + "\n"

    def formatting_prompts_func(entry):
        template = (
            "User query: {user_query}\n"
            "{perspective_1_name} arguments provided: {perspective_1}\n"
            "{perspective_2_name} arguments provided: {perspective_2}"
            "\nNeutral point-of-view answer to user query, rewriting provided"
            " arguments in natural language:\n"
            "{npov_response}"
        )

        output_texts = []

        for i in range(len(entry["user_query"])):
            formatted_prompt = template.format(
                user_query=entry["user_query"][i],
                perspective_1_name=entry["perspective_1_name"][i],
                perspective_1=entry["perspective_1"][i],
                perspective_2_name=entry["perspective_2_name"][i],
                perspective_2=entry["perspective_2"][i],
                npov_response=entry["npov_response"][i],
            )

            output_texts.append(prompt + formatted_prompt)

        return output_texts

    return formatting_prompts_func


# PERL


def npov_process_data_for_perl(
    npov_rm_data,
    npov_sft_data,
    seed=12345,
):
    """
    Processes NPOV data for PERL by creating train and test splits
    from the RM and SFT datasets.
    """

    train_data = npov_rm_data["validation"]
    test_data = npov_sft_data["test"]

    train_data = train_data.shuffle(seed=seed)
    test_data = test_data.shuffle(seed=seed)

    train_data = train_data.map(npov_writer_prompt)
    test_data = test_data.map(npov_writer_prompt)

    # The columns across splits must match.
    train_data = train_data.select_columns(
        [
            "topic",
            "user_query",
            "npov_response",
            "perspective_1",
            "perspective_1_name",
            "perspective_2",
            "perspective_2_name",
            "prompt",
        ]
    )

    npov_perl_data = DatasetDict(
        {
            "train": train_data,
            "test": test_data,
        }
    )

    return npov_perl_data


# Data augmentation


def npov_data_augmentation(data):
    """
    Augment NPOV dataset by
    (1) Adding more unique perspectives to some topics.
    (2) Generating combinations of arguments for different perspectives
    on each topic.

    Notice that for the moment I have hard coded pairs of arguments.
    Initially, the different topics do not have the same amount of
    arguments. The combinatorics of the thing makes some topics appear
    much more than others.
    """

    topics = set(data["topic"])
    p1_name = data[0]["perspective_1_name"]
    p2_name = data[0]["perspective_2_name"]

    all_topics = []
    all_user_queries = []
    all_p1_arguments = []
    all_p2_arguments = []
    all_p1_names = []
    all_p2_names = []
    result = {}

    def extract_perspective_arguments(
        topic: str, perspective_col: str, perspective_name: str
    ):
        """
        Extract unique arguments for a given topic and perspective (pro or con).
        """

        topic_data = data.filter(lambda x: x["topic"] == topic)[perspective_col]

        arguments = {
            f"{perspective_name}: {arg.strip()}"
            for text in topic_data
            for arg in text.split(f"{perspective_name}:")
            if arg.strip()
        }

        return arguments

    # Load new perspectives from CSV
    # hard coded the file name due to hurry -> ugly, fix later
    new_perspectives = pd.read_csv("npov_new_perspectives.csv")

    for topic in topics:
        user_query = data.filter(lambda x: x["topic"] == topic)["user_query"][0]

        p1_args = extract_perspective_arguments(topic, "perspective_1", p1_name)
        p2_args = extract_perspective_arguments(topic, "perspective_2", p2_name)

        # Filter new perspectives for the current topic
        new_perspectives_for_topic = new_perspectives[
            new_perspectives["topic"] == topic
        ]

        # Extract new pro and con arguments
        new_p1_args = set(
            new_perspectives_for_topic[
                new_perspectives_for_topic["perspective"] == "pro"
            ]["argument"]
        )
        new_p2_args = set(
            new_perspectives_for_topic[
                new_perspectives_for_topic["perspective"] == "con"
            ]["argument"]
        )

        # Add new arguments to the existing ones
        p1_args.update(new_p1_args)
        p2_args.update(new_p2_args)

        # Create combinations of perspectives
        p1_pairs = list(combinations(p1_args, 2))
        p2_pairs = list(combinations(p2_args, 2))

        for p1_pair in p1_pairs:
            for p2_pair in p2_pairs:
                p1_combined = " ".join(p1_pair)
                p2_combined = " ".join(p2_pair)
                all_p1_arguments.append(p1_combined)
                all_p2_arguments.append(p2_combined)
                all_p1_names.append(p1_name)
                all_p2_names.append(p2_name)
                all_topics.append(topic)
                all_user_queries.append(user_query)

    result = {
        "topic": all_topics,
        "user_query": all_user_queries,
        "perspective_1": all_p1_arguments,
        "perspective_1_name": all_p1_names,
        "perspective_2": all_p2_arguments,
        "perspective_2_name": all_p2_names,
    }

    return result


# Synthetic Hallucinations
def npov_rm_synthetic_hall_llm(entry):
    prompt = f"""
Task: Generate a modified answer that includes controlled hallucinations

Question: {entry["user_query"]}

Context:
{entry["perspective_1_name"]} arguments:
{entry["perspective_1"]}

{entry["perspective_2_name"]} arguments:
{entry["perspective_2"]}

Neutral Point of View Answer:
{entry["npov_response"]}

Instructions:
1. Create a modified version of the neutral answer that does ONE of the following:
   a. Adds one entirely new argument (either pro or con) that is NOT present in the context
   b. Distorts one or two existing arguments to change their meaning or implications

2. If adding a new argument:
   - Make it plausible and thematically consistent with the topic
   - Ensure it appears to be a natural extension of the discussion
   - Integrate it seamlessly without calling attention to it

3. If distorting existing arguments:
   - Subtly change key details, statistics, or conclusions
   - Maintain enough of the original argument to make it recognizable
   - Ensure the distortion meaningfully alters the argument's strength or implications

4. The hallucinated content should not be marked or highlighted in any way
5. Maintain the overall balanced structure, tone, and appearance of neutrality
6. The rest of the answer should remain faithful to the original arguments

Modified Answer with Synthetic Hallucinations:
"""

    return prompt


def npov_function_to_map_synthetic_halls_llm(
    entry, client, gemini_model, generation_config
):
    prompt_to_send = npov_rm_synthetic_hall_llm(entry)

    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt_to_send,
        config=generation_config,
    )

    return {
        "npov_response": response.text,
        "class_hall": "Yes",
        "label": 0,
        "has synthetic hallucination": "Yes",
    }


def bosch_load_and_process_data(data_path, flip_path):
    # Load
    data = pd.read_csv(data_path)
    data_to_flip = pd.read_csv(flip_path)

    # Filter unanswerable samples.
    data = data.loc[data["Answerable"] == True].drop(
        labels=["Answerable"], axis=1
    )

    # Flip labels of mis-annotated samples.
    data.loc[
        data["sample_id"].isin(data_to_flip["sample_id"]),
        "Label",
    ] = "Not Hallucinated"

    # Rename and create necessary columns
    data = data.rename({"Label": "class_hall", "Answer": "response"}, axis=1)
    data["class_hall"] = data["class_hall"].apply(
        lambda x: "Yes" if x == "Hallucinated" else "No"
    )
    data["label"] = data["class_hall"].apply(lambda x: 1 if x == "No" else 0)
    data["prompt"] = (
        "You are a helpful assistant to car related questions. You will be given an user's question, and the relevant part of the car manual. Your task is to answer the user's question using the information giver. Do not add to your answer any information other than those present in the manual excerpt.\n"
        + "User question:\n"
        + data["Question"]
        + "\nManual information:\n"
        + data["Context"]
        + "\nAnswer to user's question:\n"
    )

    return data


def bosch_rm_prompt(entry):
    """The dataset already contains a prompt column, which is an instruction for the writer. We just append the generation."""

    entry["prompt"] += entry["response"]

    return entry


def bosch_formatting_prompts_func(entry):
    """Formatting function for SFTTrainer. Imported in writer_sft.py."""

    template = (
        "You are a helpful assistant to car related questions. You will be given an user's question, and the relevant part of the car manual. Your task is to answer the user's question using the information given.\n"
        "User question:\n{question}"
        "\nManual information:\n{context}"
        "\nAnswer to user's question:\n"
    )

    output_texts = []

    for i in range(len(entry["Question"])):
        formatted_prompt = template.format(
            question=entry["Question"][i],
            context=entry["Context"][i],
        )

        output_texts.append(formatted_prompt)

    return output_texts


def bosch_rm_synthetic_hall_structured(entry, data):
    """Given an entry and the rest of the data, select a random sentence of
    the response in the entry, a random different entry in the data and a
    random sentence in it, and swap the first by the second.
    TO DO: I am not particularly worried with seeds here.
    """

    # Break response into sentences and filter out small ones
    tok = nltk.sent_tokenize(entry["response"])
    tok_filt = [i for i in tok if len(i) > 5]

    # Choose a random different entry and do the same
    entry2 = data.shuffle()[0]
    tok2 = nltk.sent_tokenize(entry2["response"])
    tok_filt2 = [i for i in tok2 if len(i) > 5]

    # Randomly select sentences in both entries
    rand = random.choice(tok_filt)
    rand_idx = tok.index(rand)
    rand2 = random.choice(tok_filt2)
    rand_idx2 = tok2.index(rand2)

    # Switch sentence and join
    tok[rand_idx] = tok2[rand_idx2]
    new_response = " ".join(tok)

    # Update response and labels
    entry["response"] = new_response
    entry["class_hall"] = "Yes"
    entry["label"] = 0

    # Important: you need to retokenize these!

    return entry


def bosch_rm_synthetic_hall_llm(entry, fewshot_examples=None):
    preamble = """
    Objective:
    Generate a modified version of a given answer that includes subtle, natural-sounding synthetic hallucinations (information plausible but not directly supported by the provided context), mimicking the kinds of errors LLMs sometimes make organically.

    Goal Context:
    We are creating training data for a hallucination detection model. Previous attempts generated hallucinations that were too obvious, causing the model to overfit. The goal now is to generate *subtle* hallucinations that resemble real-world LLM errors, making the synthetic data more realistic and improving the detector's generalization.

    What Makes a Good Subtle Hallucination (for this task):
    - It often feels like a minor factual inaccuracy, a slight misinterpretation, or a plausible detail/reason/consequence that *could* be true but isn't mentioned.
    - It blends seamlessly with the surrounding text derived from the context.
    - It's not easily identifiable as completely fabricated or drastically contradicting the context.
    - It avoids being the main point of the answer; it's usually a supporting detail.
    """

    prompt = preamble

    if fewshot_examples is not None:
        prompt += """
        
        --- START FEW-SHOT EXAMPLES OF DESIRED OUTPUT ---

        """

        for fewshot in fewshot_examples:
            prompt += f"""Example:
            
            [QUESTION] {fewshot["Question"]}

            [CONTEXT] {fewshot["Context"]}

            [ANSWER WITH HALLUCINATION] {fewshot["response"]}

            """

        prompt += """
        --- END FEW-SHOT EXAMPLES ---

        """

    prompt += f"""Task:
    You are given a Question, a Context used to answer it, and an Original Grounded Answer derived solely from that Context. Your task is to rewrite the Original Grounded Answer to create a Modified Answer containing subtle hallucinations.

    Requirements for the Modified Answer:
    1.  Address the Question: It must still fundamentally answer the original Question.
    2.  Incorporate Subtle Hallucinations: Include 1 or 2 specific pieces of information that are plausible and relevant BUT NOT DIRECTLY SUPPORTED by the provided Context. Aim for the kind of subtlety shown in the examples.
    3.  Plausibility & Consistency: The added information should sound highly plausible within the domain, be consistent with the overall tone and information in the context, and not be easily flagged as fake. Avoid contradictions.
    4.  Natural Integration: Weave the hallucinated information smoothly into the text. Do not make it stand out awkwardly. It should blend seamlessly with the grounded information.
    5.  Minimal Other Changes: Preserve the core grounded information and structure of the Original Grounded Answer as much as possible, augmenting it *only* with the subtle hallucinations.
    6.  Output: Provide only the text of the Modified Answer.

    Input Information:

    --- START QUESTION ---
    {entry["Question"]}
    --- END QUESTION ---

    --- START CONTEXT ---
    {entry["Context"]}
    --- END CONTEXT ---

    --- START ORIGINAL GROUNDED ANSWER ---
    {entry["response"]}
    --- END ORIGINAL GROUNDED ANSWER ---

    Instruction:
    Please generate the Modified Answer based on the requirements and examples above.

    Modified Answer:
"""

    return prompt


def bosch_function_to_map_synthetic_halls_llm(
    entry, fewshot_examples, client, gemini_model, generation_config
):
    prompt_to_send = bosch_rm_synthetic_hall_llm(entry, fewshot_examples)

    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt_to_send,
        config=generation_config,
    )

    return {
        "response": response.text,
        "class_hall": "Yes",
        "label": 0,
    }


def ragtruth_rm_prompt(entry):
    """The dataset already contains a prompt column, which is an instruction for the writer. We just append the generation."""

    entry["prompt"] = entry["user_query"] + "\n" + entry["response"]

    return entry


def ragtruth_rm_synthetic_hall_llm(entry):
    prompt = f"""Task: Generate a modified summary that includes controlled hallucinations

Source Text:
{entry["user_query"]}

Grounded Summary:
{entry["response"]}

Instructions:
1. Create a modified version of the grounded summary that includes 1-2 subtle pieces of information that are NOT present in the source text.
2. These synthetic hallucinations should:
   - Be plausible and contextually relevant to the topic
   - Blend naturally with the factual information
   - Be specific rather than vague (e.g., details, statistics, claims, or implications)
   - Change or extend the meaning in a meaningful but subtle way

3. The hallucinated content should:
   - Not be marked or highlighted in any way
   - Appear to be a natural part of the summary
   - Be difficult to detect without comparing to the source text

4. Maintain the same overall structure, tone, and approximate length as the original summary.
5. The majority of the summary should remain faithful to the source text.
6. The modified summary should still read as a coherent, well-formed summary.

Modified Summary with Subtle Hallucinations:
"""

    return prompt


def ragtruth_function_to_map_synthetic_halls_llm(
    entry, client, gemini_model, generation_config
):
    prompt_to_send = ragtruth_rm_synthetic_hall_llm(entry)

    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt_to_send,
        config=generation_config,
    )

    return {
        "response": response.text,
        "class_hall": "Yes",
        "label": 0,
    }


def ragtruth_formatting_prompts_func(entry):
    """Formatting function for SFTTrainer. Imported in writer_sft.py."""

    template = "{user_query}" + "\n" + "{response}"

    output_texts = []

    for i in range(len(entry["user_query"])):
        formatted_prompt = template.format(
            user_query=entry["user_query"][i],
            response=entry["response"][i],
        )

        output_texts.append(formatted_prompt)

    return output_texts


def main():
    parser = ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--synthetic_hallus_llm",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument(
        "--synthetic_hallus_struct",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument("--num_synth_hallus", type=int, default=0)
    parser.add_argument("--gemini_api_key", type=str)
    parser.add_argument("--synth_llm_temperature", type=float, default=0.7)
    parser.add_argument("--synth_llm_num_fewshot", type=int, default=2)
    args = parser.parse_args()

    if args.task == "npov":
        # GENERAL DATA PROCESSING
        npov_rm_data = load_dataset(
            "leobianco/npov",
            data_files={
                "train": "hc_rm5x_train.json",
                "validation": "hc_rm5x_validation.json",
                "test": "hc_rm5x_test.json",
            },
        )

        for split in npov_rm_data.keys():
            npov_rm_data[split] = npov_process_data_for_rm(
                npov_rm_data[split],
            )
            npov_rm_data[split].push_to_hub(repo_id=args.task + "_processed")

        # AUTORATER
        npov_autorater_data = concatenate_datasets(
            [
                npov_rm_data["train"],
                npov_rm_data["validation"],
                npov_rm_data["test"],
            ]
        )
        # Keep organic hallucinations only
        npov_autorater_data = npov_autorater_data.filter(
            lambda x: x["has synthetic hallucination"] == "No"
        )
        # Save
        npov_autorater_data.push_to_hub(
            repo_id=args.task + "_autorater",
            split="test",
        )

        # ORGANIC HALLUCINATIONS
        npov_rm_data_organic = deepcopy(npov_rm_data)

        # Filter out synthetic hallucinations from train set
        npov_rm_data_organic["train"] = npov_rm_data_organic["train"].filter(
            lambda x: not (
                x["class_hall"] == "Yes"
                and x["has synthetic hallucination"] == "Yes"
            )
        )

        # Filter out synthetic hallucinations from validation set
        npov_rm_data_organic["validation"] = npov_rm_data_organic[
            "validation"
        ].filter(
            lambda x: not (
                x["class_hall"] == "Yes"
                and x["has synthetic hallucination"] == "Yes"
            )
        )

        # Put in splits and save
        npov_rm_data_organic_dataset = DatasetDict(
            {
                "train": npov_rm_data_organic["train"],
                "test": npov_rm_data_organic["validation"],
            }
        )

        npov_rm_data_organic_dataset.push_to_hub(
            repo_id=args.task + "_rm_organic"
        )

        # Release memory
        del npov_rm_data_organic_dataset
        del npov_rm_data_organic

        # SYNTHETIC HALLUCINATIONS

        # LLM-GENERATED
        if args.synthetic_hallus_llm:
            # Copy data
            npov_rm_data_synth_llm = deepcopy(npov_rm_data)

            # Filter out synthetic hallucinations from validation set
            npov_rm_data_synth_llm["validation"] = npov_rm_data_synth_llm[
                "validation"
            ].filter(
                lambda x: not (
                    x["class_hall"] == "Yes"
                    and x["has synthetic hallucination"] == "Yes"
                )
            )

            # Get non-hallucinations
            npov_rm_train_non = (
                npov_rm_data_synth_llm["train"]
                .filter(lambda x: x["class_hall"] == "No")
                .shuffle(seed=args.seed)
            )

            # We use some non-hallus. to generate synthetic hallus.
            npov_rm_train_synth_llm = npov_rm_train_non.shuffle(
                seed=args.seed
            ).select(range(args.num_synth_hallus))
            npov_rm_train_non = npov_rm_train_non.filter(
                lambda x: x not in npov_rm_train_synth_llm
            )

            # API config
            client = genai.Client(api_key=args.gemini_api_key)
            gemini_model = "gemini-2.0-flash-001"
            generation_config = types.GenerateContentConfig(
                temperature=args.synth_llm_temperature,
                seed=args.seed,
            )

            print("Calling Gemini's API...")

            synthetic_hallucinations_llm = npov_rm_train_synth_llm.map(
                npov_function_to_map_synthetic_halls_llm,
                fn_kwargs=dict(
                    client=client,
                    gemini_model=gemini_model,
                    generation_config=generation_config,
                ),
            )

            # Now we create the RM training data
            synthetic_hallucinations_llm_train_data = concatenate_datasets(
                [synthetic_hallucinations_llm, npov_rm_train_non]
            ).shuffle(seed=args.seed)

            # Merge the two in the appropriate splits
            synthetic_hallucinations_llm_data = DatasetDict(
                {
                    "train": synthetic_hallucinations_llm_train_data,
                    "test": npov_rm_data_synth_llm["validation"],
                }
            )

            # Since the response changed, you need to rewrite the
            # prompt
            for split in synthetic_hallucinations_llm_data.keys():
                synthetic_hallucinations_llm_data[split] = (
                    npov_process_data_for_rm(
                        synthetic_hallucinations_llm_data[split],
                    )
                )

                synthetic_hallucinations_llm_data[split].push_to_hub(
                    repo_id=args.task + "_rm_synthetic_llm",
                    split=split,
                )

        # STRUCTURED
        if args.synthetic_hallus_struct:
            # Copy data
            npov_rm_data_synth_struct = deepcopy(npov_rm_data)

            # To train on synthetic hallucinations only and evaluate on organic,
            # drop the organic ones from the training set, and the synthetic
            # ones from the validation set. Do not mix splits, as this would
            # mix topics. Do not pass the test set into the validation one,
            # because it will be the test set for PERL later on and we agreed
            # that we cannot test both the RM and PERL on the same examples.
            npov_rm_data_synth_struct["train"] = npov_rm_data_synth_struct[
                "train"
            ].filter(
                lambda x: not (
                    x["class_hall"] == "Yes"
                    and x["has synthetic hallucination"] == "No"
                )
            )

            # Filter out synthetic hallucinations from validation set
            npov_rm_data_synth_struct["validation"] = npov_rm_data_synth_struct[
                "validation"
            ].filter(
                lambda x: not (
                    x["class_hall"] == "Yes"
                    and x["has synthetic hallucination"] == "Yes"
                )
            )

            # Put in splits and save
            npov_rm_data_synth_struct_dataset = DatasetDict(
                {
                    "train": npov_rm_data_synth_struct["train"],
                    "test": npov_rm_data_synth_struct["validation"],
                }
            )

            npov_rm_data_synth_struct_dataset.push_to_hub(
                repo_id=args.task + "_rm_synthetic_struct"
            )

            # Release memory
            del npov_rm_data_synth_struct_dataset
            del npov_rm_data_synth_struct

        # SFT
        npov_sft_data = load_dataset(
            "leobianco/npov",
            data_files={
                "train": "writer_train.json",
                "validation": "writer_validation.json",
                "test": "writer_test.json",
            },
        )

        for split in npov_sft_data.keys():
            npov_sft_data[split] = npov_process_data_for_sft(
                npov_sft_data[split]
            )

            npov_sft_data[split].push_to_hub(
                repo_id=args.task + "_sft",
                split=split,
            )

        # PERL
        npov_perl_data = npov_process_data_for_perl(
            npov_rm_data,
            npov_sft_data,
            seed=args.seed,
        )

        for split in npov_perl_data.keys():
            npov_perl_data[split].push_to_hub(
                repo_id=args.task + "_perl",
                split=split,
            )

        # FINAL TEST SET
        npov_augmented_data = npov_data_augmentation(npov_rm_data["test"])
        npov_augmented_data_dict = DatasetDict(
            {"test": Dataset.from_dict(npov_augmented_data)}
        )
        npov_augmented_data_dict["test"] = npov_augmented_data_dict["test"].map(
            npov_writer_prompt
        )
        npov_augmented_data_dict.push_to_hub(
            repo_id=args.task + "_final_test_set",
        )

    elif args.task == "bosch":
        # GENERAL DATA PROCESSING
        # TODO: put these files on a HF repo.
        data_train = bosch_load_and_process_data(
            "/home/leo/Downloads/DelucionQA_data/cleaned/train.csv",
            "/home/leo/Downloads/DelucionQA_data/cleaned_leo/leo_flip_train.csv",
        )
        data_val = bosch_load_and_process_data(
            "/home/leo/Downloads/DelucionQA_data/cleaned/dev.csv",
            "/home/leo/Downloads/DelucionQA_data/cleaned_leo/leo_flip_dev.csv",
        )
        data_test = bosch_load_and_process_data(
            "/home/leo/Downloads/DelucionQA_data/cleaned/test.csv",
            "/home/leo/Downloads/DelucionQA_data/cleaned_leo/leo_flip_test.csv",
        )

        # Convert to Hugging Face Dataset and push to hub
        dataset_train = Dataset.from_pandas(data_train, split="train")
        dataset_val = Dataset.from_pandas(data_val, split="validation")
        dataset_test = Dataset.from_pandas(data_test, split="test")

        # IMPORTANT: notice how we switch the original train and test splits.
        # This is because we need enough test samples to measure hallucination
        # rate decrease in the end.
        dataset = DatasetDict(
            {
                "train": dataset_train,
                "validation": dataset_val,
                "test": dataset_test,
            }
        )
        dataset.push_to_hub(args.task + "_processed")

        # AUTORATER
        # For the autorater, we want to measure its ability on all samples, so
        # we just merge all of them and save as a single test split.
        autorater_dataset = concatenate_datasets(
            [dataset_train, dataset_val, dataset_test]
        )
        autorater_dataset.push_to_hub(
            repo_id=args.task + "_autorater", split="test"
        )

        # SFT
        sft_data = dataset_val.filter(lambda entry: entry["class_hall"] == "No")
        sft_data.push_to_hub(repo_id=args.task + "_sft")

        # REWARD MODEL
        # ORGANIC HALLUCINATIONS
        # train -> validation, test -> test
        bosch_rm_organic = DatasetDict(
            {"train": dataset_val, "test": dataset_test}
        )
        # RM prompt
        for split in bosch_rm_organic.keys():
            bosch_rm_organic[split] = bosch_rm_organic[split].map(
                bosch_rm_prompt
            )
        # Save
        bosch_rm_organic.push_to_hub(repo_id=args.task + "_rm_organic")

        # SYNTHETIC HALLUCINATIONS

        # GENERAL ORGANIZATION
        # train -> validation | non-hallucinated (some -> hall.), test -> test
        non_hallucinated_data = dataset_val.filter(
            lambda entry: entry["class_hall"] == "No"
        )
        hallucinated_data = dataset_val.filter(
            lambda entry: entry["class_hall"] == "Yes"
        )
        # Get some random non-hallucinations to become hallucinations
        to_become_hallus = non_hallucinated_data.shuffle(seed=args.seed).select(
            range(args.num_synth_hallus)
        )
        # Remove them from the rest
        non_hallucinated_data = non_hallucinated_data.filter(
            lambda x: x not in to_become_hallus
        )
        # Build RM prompts for the non_hallucinated_data
        non_hallucinated_data = non_hallucinated_data.map(bosch_rm_prompt)

        # LLM-GENERATED
        if args.synthetic_hallus_llm:
            # Fewshot examples of organic hallucinations to help LLM
            n_fewshot_examples_synth_llm = args.synth_llm_num_fewshot
            fewshot_examples_synth_llm = hallucinated_data.shuffle(
                seed=args.seed
            ).select(range(n_fewshot_examples_synth_llm))

            # API config
            client = genai.Client(api_key=args.gemini_api_key)
            gemini_model = "gemini-2.0-flash-001"
            generation_config = types.GenerateContentConfig(
                temperature=args.synth_llm_temperature,
                seed=args.seed,
            )

            print("Calling Gemini's API...")

            synthetic_hallucinations_llm = to_become_hallus.map(
                bosch_function_to_map_synthetic_halls_llm,
                fn_kwargs=dict(
                    fewshot_examples=fewshot_examples_synth_llm,
                    client=client,
                    gemini_model=gemini_model,
                    generation_config=generation_config,
                ),
            )

            # Write the RM prompts.
            synthetic_hallucinations_llm = synthetic_hallucinations_llm.map(
                bosch_rm_prompt
            )

            # Now we create the RM training data by joining and shuffling
            synthetic_hallucinations_llm_train_data = concatenate_datasets(
                [synthetic_hallucinations_llm, non_hallucinated_data]
            ).shuffle(seed=args.seed)

            # Create dataset with same test split as before
            synthetic_hallucinations_llm_data = DatasetDict(
                {
                    "train": synthetic_hallucinations_llm_train_data,
                    "test": bosch_rm_organic["test"],
                }
            )

            synthetic_hallucinations_llm_data.push_to_hub(
                repo_id=args.task + "_rm_synthetic_llm"
            )

        # STRUCTURED
        if args.synthetic_hallus_struct:
            # Apply the map that switches sentences
            synthetic_hallucinations_struct = to_become_hallus.map(
                bosch_rm_synthetic_hall_structured,
                fn_kwargs=dict(data=dataset_val),
            )

            # Write the RM prompts with the new response.
            synthetic_hallucinations_struct = (
                synthetic_hallucinations_struct.map(bosch_rm_prompt)
            )

            # Add some non-hallucinated examples and shuffle!
            synthetic_hallucinations_struct_train_data = concatenate_datasets(
                [synthetic_hallucinations_struct, non_hallucinated_data]
            ).shuffle(seed=args.seed)

            # Create dataset with same test split as before
            synthetic_hallucinations_struct_data = DatasetDict(
                {
                    "train": synthetic_hallucinations_struct_train_data,
                    "test": bosch_rm_organic["test"],
                }
            )

            synthetic_hallucinations_struct_data.push_to_hub(
                repo_id=args.task + "_rm_synthetic_struct"
            )

        # PERL
        # train -> test, test -> a few random samples (not the final test set).
        perl_data = DatasetDict(
            {"train": dataset_test, "test": dataset_val.select(range(10))}
        )

        perl_data.push_to_hub(repo_id=args.task + "_perl")

        # FINAL TEST SET
        dataset_train.push_to_hub(
            repo_id=args.task + "_final_test_set", split="test"
        )

    elif args.task == "ragtruth":
        # GENERAL DATA PROCESSING
        # TODO: put these files on huggingface.
        data_sources = pd.read_json(
            "/home/leo/Downloads/RAGTruth/dataset/source_info.jsonl", lines=True
        )
        data_responses = pd.read_json(
            "/home/leo/Downloads/RAGTruth/dataset/response.jsonl", lines=True
        )
        # Filter only summarization entries and unify the two data files.
        sources = data_sources[data_sources["task_type"] == "Summary"]
        responses = data_responses[
            data_responses["source_id"].isin(sources["source_id"])
        ]
        unified = pd.merge(sources, responses, on="source_id", how="left")
        unified = unified.drop(
            ["source_info", "task_type", "source", "id"], axis=1
        )
        # Define necessary columns.
        unified["label"] = unified.apply(
            lambda entry: 1 if len(entry["labels"]) == 0 else 0, axis=1
        )
        unified["class_hall"] = unified.apply(
            lambda entry: "No" if len(entry["labels"]) == 0 else "Yes", axis=1
        )
        # Filter only good-quality entries, and rename some columns.
        unified = unified[unified["quality"] == "good"]
        unified = unified.rename(
            columns={"labels": "explanation", "prompt": "user_query"}
        )
        # Separate data splits.
        train_data = unified[unified["split"] == "train"]
        test_data = unified[unified["split"] == "test"]
        # Create a validation split by taking the samples corresponding to
        # the first 150 unique queries in the train set
        val_source_ids = train_data["source_id"].drop_duplicates().iloc[:150]
        val_data = train_data[
            train_data["source_id"].isin(val_source_ids)
        ].copy()
        train_data = train_data[
            ~train_data["source_id"].isin(val_source_ids)
        ].copy()
        # Convert to dataset format
        train_dataset = Dataset.from_pandas(train_data)
        val_dataset = Dataset.from_pandas(val_data)
        test_dataset = Dataset.from_pandas(test_data)

        # AUTORATER
        # This unified data will be used to evaluate the autorater.
        unified_dataset = concatenate_datasets(
            [train_dataset, val_dataset, test_dataset]
        )
        unified_dataset.push_to_hub(
            repo_id=args.task + "_autorater", split="test"
        )

        # SFT data
        # We use the non-hallucinated samples in the validation split for SFT.
        sft_data = test_dataset.filter(
            lambda entry: entry["class_hall"] == "No"
        )
        sft_data.push_to_hub(repo_id=args.task + "_sft")

        # REWARD MODEL
        # ORGANIC HALLUCINATIONS
        # train -> validation, test -> test
        ragtruth_rm_organic = DatasetDict(
            {"train": test_dataset, "test": val_dataset}
        )
        # RM prompt
        for split in ragtruth_rm_organic.keys():
            ragtruth_rm_organic[split] = ragtruth_rm_organic[split].map(
                ragtruth_rm_prompt
            )
        # Save
        ragtruth_rm_organic.push_to_hub(repo_id=args.task + "_rm_organic")

        # SYNTHETIC HALLUCINATIONS

        # GENERAL ORGANIZATION
        non_hallucinated_data = test_dataset.filter(
            lambda entry: entry["class_hall"] == "No"
        )
        hallucinated_data = test_dataset.filter(
            lambda entry: entry["class_hall"] == "Yes"
        )
        # Get some random non-hallucinations to become hallucinations
        to_become_hallus = non_hallucinated_data.shuffle(seed=args.seed).select(
            range(args.num_synth_hallus)
        )
        # Remove them from the rest
        non_hallucinated_data = non_hallucinated_data.filter(
            lambda x: x not in to_become_hallus
        )
        # Build RM prompts for the non_hallucinated_data
        non_hallucinated_data = non_hallucinated_data.map(ragtruth_rm_prompt)

        # LLM GENERATED
        if args.synthetic_hallus_llm:
            # API config
            client = genai.Client(api_key=args.gemini_api_key)
            gemini_model = "gemini-2.0-flash-001"
            generation_config = types.GenerateContentConfig(
                temperature=args.synth_llm_temperature,
                seed=args.seed,
            )

            print("Calling Gemini's API...")

            synthetic_hallucinations_llm = to_become_hallus.map(
                ragtruth_function_to_map_synthetic_halls_llm,
                fn_kwargs=dict(
                    client=client,
                    gemini_model=gemini_model,
                    generation_config=generation_config,
                ),
            )

            # Write the RM prompts
            synthetic_hallucinations_llm = synthetic_hallucinations_llm.map(
                ragtruth_rm_prompt
            )

            # Now we create the RM training data
            synthetic_hallucinations_llm_train_data = concatenate_datasets(
                [synthetic_hallucinations_llm, non_hallucinated_data]
            ).shuffle(seed=args.seed)

            # Merge the two in the appropriate splits
            synthetic_hallucinations_llm_data = DatasetDict(
                {
                    "train": synthetic_hallucinations_llm_train_data,
                    "test": ragtruth_rm_organic["test"],
                }
            )

            synthetic_hallucinations_llm_data.push_to_hub(
                repo_id=args.task + "_rm_synthetic_llm"
            )

        # STRUCTURED
        elif args.synthetic_hallus_struct:
            # Tbh, you can use the same bosch_rm_synthetic_hall_structured
            # function that was used for Bosch, and then retokenize.
            synthetic_hallucinations_struct = to_become_hallus.map(
                bosch_rm_synthetic_hall_structured,
                fn_kwargs=dict(data=test_dataset),
            )

            # Write the RM prompts with the new response.
            synthetic_hallucinations_struct = (
                synthetic_hallucinations_struct.map(ragtruth_rm_prompt)
            )

            # Add some non-hallucinated examples and shuffle!
            synthetic_hallucinations_struct_train_data = concatenate_datasets(
                [synthetic_hallucinations_struct, non_hallucinated_data]
            ).shuffle(seed=args.seed)

            # Create dataset with same test split as before
            synthetic_hallucinations_struct_data = DatasetDict(
                {
                    "train": synthetic_hallucinations_struct_train_data,
                    "test": ragtruth_rm_organic["test"],
                }
            )

            synthetic_hallucinations_struct_data.push_to_hub(
                repo_id=args.task + "_rm_synthetic_struct"
            )

        # PERL
        # In the case of PERL, prompt = user_query. It needs a few samples in
        # the test split, but it is not the real test split!
        perl_data = DatasetDict(
            {"train": val_dataset, "test": test_dataset.select(range(10))}
        )
        for split in perl_data.keys():
            perl_data[split] = perl_data[split].map(
                lambda x: {**x, "prompt": x["user_query"]}
            )
        perl_data.push_to_hub(args.task + "_perl")

        # FINAL TEST SET
        # It is simply the train split. Rename "user_query" column to "prompt".
        train_dataset = train_dataset.rename_column("user_query", "prompt")
        train_dataset.push_to_hub(args.task + "_final_test_set", split="test")


if __name__ == "__main__":
    main()
