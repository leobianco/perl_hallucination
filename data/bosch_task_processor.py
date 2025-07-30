import random

import nltk
from datasets import DatasetDict, concatenate_datasets, load_dataset
from google import genai
from google.genai import types

from data.base_task_processor import BaseTaskProcessor


class BoschTaskProcessor(BaseTaskProcessor):
    def _load_data(self):
        data = load_dataset(
            self.args.hf_repo,
            data_files={
                "train": "train.csv",
                "validation": "val.csv",
                "test": "test.csv",
            },
        )

        return data

    def _preprocess_data(self, data):
        # Filter unanswerable samples.
        data = data.filter(lambda entry: entry.get("Answerable"))
        data = data.remove_columns("Answerable")

        # Rename, create, and process the necessary columns.
        data = data.rename_columns(
            {"Label": "class_hall", "Answer": "response"}
        )
        data = data.map(
            lambda entry: {
                "class_hall": "Yes"
                if entry["class_hall"] == "Hallucinated"
                else "No"
            }
        )
        data = data.map(
            lambda entry: {"label": 1 if entry["class_hall"] == "No" else 0}
        )
        data = data.map(
            lambda entry: {
                "prompt": (
                    "You are a helpful assistant to car related questions. You will be given an user's question, and the relevant part of the car manual. Your task is to answer the user's question using the information giver. Do not add to your answer any information other than those present in the manual excerpt.\n"
                    + "User question:\n"
                    + entry["Question"]
                    + "\nManual information:\n"
                    + entry["Context"]
                    + "\nAnswer to user's question:\n"
                )
            }
        )

        # Flip labels of mis-annotated samples.
        data_to_flip = load_dataset(
            self.args.hf_repo,
            data_files={
                "train": "flip_train.csv",
                "validation": "flip_val.csv",
                "test": "flip_test.csv",
            },
        )

        data = self._flip_entries(data, data_to_flip)

        return data

    def _make_sft_data(self, data):
        """For the Bosch task, we SFT on the non-hallucinated samples of the validation split."""

        sft_data = data["test"].filter(
            lambda entry: entry["class_hall"] == "No"
        )

        return sft_data

    def _make_organic_hallucinations_data(self, data):
        organic_hallucinations_data = DatasetDict(
            {"train": data["test"], "test": data["validation"]}
        )

        organic_hallucinations_data = organic_hallucinations_data.map(
            self._rm_prompt
        )

        return organic_hallucinations_data

    def _make_structured_hallucinations_data(self, data):
        """Create structured synthetic hallucinations data for reward model training."""

        args = self.args
        # Use validation split for both non-hallucinated and hallucinated (TO BE CORRECTED).
        to_become_hallus, non_hallucinated_data_rest = (
            self._split_nonhallucinated_for_synthetic(
                data["validation"], args.num_synth_hallus, args.seed
            )
        )
        # Build RM prompts for the non-hallucinated data.
        non_hallucinated_data_rest = non_hallucinated_data_rest.map(
            self._rm_prompt
        )
        # Apply the map that creates structured hallucinations.
        synthetic_hallucinations_struct = to_become_hallus.map(
            self._synthetic_hall_structured
        )
        # Write the RM prompts with the modified entries.
        synthetic_hallucinations_struct = synthetic_hallucinations_struct.map(
            self._rm_prompt
        )
        # Add the non-hallucinated examples and shuffle.
        synthetic_hallucinations_struct_train_data = concatenate_datasets(
            [synthetic_hallucinations_struct, non_hallucinated_data_rest]
        ).shuffle(seed=args.seed)
        # Create dataset with same test split as organic.
        # TODO: take the change of splits into account!
        organic = self._make_organic_hallucinations_data(data)
        synthetic_hallucinations_struct_data = DatasetDict(
            {
                "train": synthetic_hallucinations_struct_train_data,
                "test": organic["test"],
            }
        )

        return synthetic_hallucinations_struct_data

    def _make_llm_hallucinations_data(self, data, organic_hallucinations_data):
        """Generate synthetic hallucinations using LLM and return DatasetDict for reward model training."""

        args = self.args
        # Use validation split for both non-hallucinated and hallucinated
        to_become_hallus, non_hallucinated_data_rest = (
            self._split_nonhallucinated_for_synthetic(
                data["validation"], args.num_synth_hallus, args.seed
            )
        )
        hallucinated_data = data["validation"].filter(
            lambda entry: entry["class_hall"] == "Yes"
        )
        # Build RM prompts for the non-hallucinated data
        non_hallucinated_data_rest = non_hallucinated_data_rest.map(
            self._rm_prompt
        )
        # Fewshot examples of organic hallucinations to help LLM
        n_fewshot_examples_synth_llm = args.synth_llm_num_fewshot
        fewshot_examples_synth_llm = hallucinated_data.shuffle(
            seed=args.seed
        ).select(range(n_fewshot_examples_synth_llm))

        client = genai.Client(api_key=args.gemini_api_key)
        gemini_model = "gemini-2.0-flash-001"
        generation_config = types.GenerateContentConfig(
            temperature=args.synth_llm_temperature,
            seed=args.seed,
        )
        # Call Gemini's API for synthetic hallucinations
        synthetic_hallucinations_llm = to_become_hallus.map(
            self._function_to_map_synthetic_halls_llm,
            fn_kwargs=dict(
                fewshot_examples=fewshot_examples_synth_llm,
                client=client,
                gemini_model=gemini_model,
                generation_config=generation_config,
            ),
        )
        # Write the RM prompts
        synthetic_hallucinations_llm = synthetic_hallucinations_llm.map(
            self._rm_prompt
        )
        # Now we create the RM training data by joining and shuffling
        synthetic_hallucinations_llm_train_data = concatenate_datasets(
            [synthetic_hallucinations_llm, non_hallucinated_data_rest]
        ).shuffle(seed=args.seed)
        # Create dataset with same test split as organic
        synthetic_hallucinations_llm_data = DatasetDict(
            {
                "train": synthetic_hallucinations_llm_train_data,
                "test": organic_hallucinations_data["test"],
            }
        )

        return synthetic_hallucinations_llm_data

    def _function_to_map_synthetic_halls_llm(
        self, entry, fewshot_examples, client, gemini_model, generation_config
    ):
        """Call Gemini LLM to generate a synthetic hallucinated Bosch response.

        Args:
            entry (dict): Entry to modify.
            fewshot_examples (datasets.Dataset): Few-shot examples for the LLM.
            client: Gemini API client.
            gemini_model (str): Model name.
            generation_config: Generation configuration for the LLM.

        Returns:
            dict: Entry with hallucinated response and updated labels.
        """

        prompt_to_send = self._rm_synthetic_hall_llm(entry, fewshot_examples)

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

    def _make_perl_data(self, data, sft_data, seed=12345):
        perl_data = DatasetDict(
            {
                "train": data["validation"],
                "test": data["test"].select(range(10)),
            }
        )

        return perl_data

    def _make_autorater_data(self, data):
        """For the autorater, we want to measure its ability on all samples, so we just merge all of them and save as a single test split."""

        autorater_dataset = concatenate_datasets(
            [data["train"], data["validation"], data["test"]]
        )

        return autorater_dataset

    def _make_evaluation_data(self, data):
        evaluation_data = DatasetDict({"test": data["train"]})

        return evaluation_data

    @staticmethod
    def _flip_entries(data, data_to_flip):
        invert_class_hall = {"Yes": "No", "No": "Yes"}

        for split in data.keys():
            ids_to_flip = set(data_to_flip[split]["sample_id"])

            data[split] = data[split].map(
                lambda entry: {
                    "label": 1 - entry["label"]
                    if entry.get("sample_id") in ids_to_flip
                    else entry["label"],
                    "class_hall": invert_class_hall[entry["class_hall"]]
                    if entry.get("sample_id") in ids_to_flip
                    else entry["class_hall"],
                }
            )

        return data

    @staticmethod
    def _rm_prompt(entry):
        """Format a reward model prompt for a Bosch entry.

        Args:
            entry (dict): Entry with 'prompt' and 'response' fields.

        Returns:
            dict: Entry with updated 'prompt' field.
        """

        entry["prompt"] += entry["response"]

        return entry

    @staticmethod
    def _split_nonhallucinated_for_synthetic(
        data_split, num_synth_hallus, seed
    ):
        """
        Helper to split a data split into:
        - to_become_hallus: random non-hallucinated samples to become hallucinations
        - non_hallucinated_data_rest: the rest of non-hallucinated samples
        """
        non_hallucinated_data = data_split.filter(
            lambda entry: entry["class_hall"] == "No"
        )
        to_become_hallus = non_hallucinated_data.shuffle(seed=seed).select(
            range(num_synth_hallus)
        )
        non_hallucinated_data_rest = non_hallucinated_data.filter(
            lambda x: x not in to_become_hallus
        )

        return to_become_hallus, non_hallucinated_data_rest

    @staticmethod
    def _rm_synthetic_hall_llm(entry, fewshot_examples=None):
        """Generate a prompt for LLM-based synthetic hallucination creation for Bosch entries.

        Args:
            entry (dict): Entry with question, context, and response.
            fewshot_examples (datasets.Dataset, optional): Few-shot examples to include. Defaults to None.

        Returns:
            str: Prompt for LLM to generate a hallucinated answer.
        """

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

    @staticmethod
    def _synthetic_hall_structured(entry):
        """Create a structured synthetic hallucination in Bosch data by removing sentences from the context.

        Args:
            entry (dict): Entry to modify.
            data (datasets.Dataset): The full validation dataset for context.

        Returns:
            dict: Entry with modified context and updated labels.
        """

        # Break response into sentences and filter out small ones
        tok = nltk.sent_tokenize(entry["Context"])

        # Randomly select sentences in the context.
        # One if there are less than 5 sentences, 2 otherwise.
        num_to_sample = 1 if len(tok) < 5 else 2
        rand = random.sample(tok, num_to_sample)
        rand_idx = [tok.index(r) for r in rand]

        # Erase that sentence
        for r_idx in rand_idx:
            tok[r_idx] = ""

        # Join everything back together
        new_context = " ".join(tok)

        # Update response and labels
        entry["Context"] = new_context
        entry["class_hall"] = "Yes"
        entry["label"] = 0

        # Important: you need to retokenize these later!
        return entry
