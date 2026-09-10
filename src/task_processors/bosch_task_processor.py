import collections
import random
import re
from typing import Any, Callable, Optional, Tuple

import evaluate
import nltk
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from google import genai
from google.genai import types

from src.task_processors.base_task_processor import BaseTaskProcessor

# Ensure punkt and punkt_tab tokenizers are available
nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)


class _PurePythonRougeScorer:
    """Pure-Python in-memory ROUGE-1 F1 scorer with zero network or filesystem overhead."""

    def __init__(self, use_stemmer: bool = True):
        self.stemmer = None
        if use_stemmer:
            try:
                import nltk
                from nltk.stem import PorterStemmer

                self.stemmer = PorterStemmer()
            except Exception:
                self.stemmer = None

    def _tokenize_and_stem(self, text: str) -> list[str]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        stemmer = self.stemmer
        if stemmer is not None:
            try:
                return [stemmer.stem(t) for t in tokens]
            except Exception:
                return tokens
        return tokens

    def score(self, target: str, prediction: str) -> dict[str, Any]:
        """Compute ROUGE-1 F1 between target (reference) and prediction."""
        ref_tokens = self._tokenize_and_stem(target)
        pred_tokens = self._tokenize_and_stem(prediction)

        ScoreType = collections.namedtuple(
            "Score", ["precision", "recall", "fmeasure"]
        )
        if not ref_tokens or not pred_tokens:
            return {
                "rouge1": ScoreType(precision=0.0, recall=0.0, fmeasure=0.0)
            }

        ref_counts = collections.Counter(ref_tokens)
        pred_counts = collections.Counter(pred_tokens)

        overlap = sum(
            min(count, pred_counts[token])
            for token, count in ref_counts.items()
        )
        precision = overlap / len(pred_tokens) if pred_tokens else 0.0
        recall = overlap / len(ref_tokens) if ref_tokens else 0.0

        if precision + recall > 0.0:
            fmeasure = 2.0 * (precision * recall) / (precision + recall)
        else:
            fmeasure = 0.0

        return {
            "rouge1": ScoreType(
                precision=precision, recall=recall, fmeasure=fmeasure
            )
        }


class BoschTaskProcessor(BaseTaskProcessor):
    @staticmethod
    def _get_rouge_scorer() -> Any:
        """Return an in-memory ROUGE-1 scorer.

        Prefers google-research's official `rouge_score.rouge_scorer.RougeScorer`
        for fast, in-memory computation (~0.05ms per pair).
        Falls back to `_PurePythonRougeScorer` if `rouge_score` is not installed.
        """
        try:
            from rouge_score import rouge_scorer

            return rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        except ImportError:
            return _PurePythonRougeScorer(use_stemmer=True)
    def _load_data(self) -> DatasetDict:
        data = load_dataset(
            self.args.hf_repo,
            data_files={
                "train": "train.csv",
                "validation": "val.csv",
                "test": "test.csv",
            },
        )

        return data

    def _preprocess_data(self, data: DatasetDict) -> DatasetDict:
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
                    "You are a helpful assistant to car related questions. You will be given an user's question, and the relevant part of the car manual. Your task is to answer the user's question using the information given. Do not add to your answer any information other than those present in the manual excerpt.\n"
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

    def _make_sft_data(self, data: DatasetDict) -> DatasetDict:
        """For the Bosch task, we SFT on the non-hallucinated samples of the validation split."""

        sft_data = data["test"].filter(
            lambda entry: entry["class_hall"] == "No"
        )

        # prompt-completion format, for training on completion only
        sft_data = sft_data.rename_column("response", "completion")

        sft_data = sft_data.train_test_split(test_size=0.2)

        return sft_data

    def _make_organic_hallucinations_data(
        self, data: DatasetDict
    ) -> DatasetDict:
        organic_hallucinations_data = DatasetDict(
            {"train": data["test"], "test": data["validation"]}
        )

        organic_hallucinations_data = organic_hallucinations_data.map(
            self._rm_prompt
        )

        return organic_hallucinations_data

    def _make_structured_hallucinations_data(
        self, data: DatasetDict
    ) -> DatasetDict:
        """Create structured synthetic hallucinations data for reward model training.

        Implements three synthetic data generation schemas:
        - Idea 1: Non-hallucinated samples by removing context sentences that do
          not match any sentence in the generation (low ROUGE overlap).
        - Idea 2: Hallucinated samples by removing top-k matching context
          sentences (rank 1, 2, 3...) above a relevance threshold.
        - Idea 3: Non-hallucinated samples by removing sentences from the
          generation (introducing coverage reduction without hallucination).

        Also maintains rough balancing between hallucinated (label 0) and
        non-hallucinated (label 1) classes.
        """
        args = self.args
        seed = getattr(args, "seed", 12345)
        num_synth_hallus = getattr(args, "num_synth_hallus", 0)
        top_k = getattr(args, "synth_struct_top_k", 3)
        hallu_threshold = getattr(args, "synth_struct_hallu_threshold", 0.40)
        irrelevant_threshold = getattr(
            args, "synth_struct_irrelevant_threshold", 0.15
        )
        max_nonhall_per_entry = getattr(
            args, "synth_struct_max_nonhall_per_entry", 2
        )
        balance_ratio = getattr(args, "synth_struct_balance_ratio", 1.25)

        # Source pool: non-hallucinated validation samples
        non_hallucinated_data = data["validation"].filter(
            lambda entry: entry["class_hall"] == "No"
        )
        shuffled_source = non_hallucinated_data.shuffle(seed=seed)

        # If num_synth_hallus is -1 or exceeds source length, use all available samples
        use_all = (
            num_synth_hallus is None
            or num_synth_hallus == -1
            or num_synth_hallus >= len(shuffled_source)
        )
        if not use_all and num_synth_hallus > 0:
            source_pool = shuffled_source.select(range(num_synth_hallus))
            rest_pool = shuffled_source.select(
                range(num_synth_hallus, len(shuffled_source))
            )
        else:
            source_pool = shuffled_source
            rest_pool = None

        rouge_metric = self._get_rouge_scorer()

        hallucinations: list[dict[str, Any]] = []
        non_hallucinations: list[dict[str, Any]] = []

        total_source = len(source_pool)
        print(
            f"[Structured Synth] Generating synthetic hallucinations from {total_source} validation samples..."
        )

        for entry_idx, entry in enumerate(source_pool):
            if (entry_idx + 1) % 10 == 0 or (entry_idx + 1) == total_source:
                print(
                    f"[Structured Synth] Processing sample {entry_idx + 1}/{total_source}..."
                )
            question = entry.get("Question", "")
            context = entry.get("Context", "")
            response = entry.get("response", "")

            sentences_context = (
                nltk.sent_tokenize(context) if context else []
            )
            sentences_response = (
                nltk.sent_tokenize(response) if response else []
            )

            if not sentences_context or not sentences_response:
                continue

            # Pairwise sentence ROUGE scoring
            scores = self._compute_sentence_rouge_scores(
                sentences_context=sentences_context,
                sentences_response=sentences_response,
                rouge_metric=rouge_metric,
            )

            entry_hallus: list[dict[str, Any]] = []
            entry_nonhallus: list[dict[str, Any]] = []

            # Idea 2: Generate hallucinated samples (top-k context removal)
            if len(sentences_context) >= 2:
                entry_hallus = self._generate_hallucinations_top_k(
                    question=question,
                    sentences_context=sentences_context,
                    response=response,
                    scores=scores,
                    top_k=top_k,
                    hallu_threshold=hallu_threshold,
                )

            # Idea 1: Generate non-hallucinated samples (irrelevant context removal)
            if len(sentences_context) >= 2:
                irrelevant_samples = (
                    self._generate_nonhallucinations_irrelevant_context(
                        question=question,
                        sentences_context=sentences_context,
                        response=response,
                        scores=scores,
                        irrelevant_threshold=irrelevant_threshold,
                        max_candidates=max_nonhall_per_entry,
                    )
                )
                entry_nonhallus.extend(irrelevant_samples)

            # Idea 3: Generate non-hallucinated samples (response sentence removal)
            if len(sentences_response) >= 2:
                resp_drops = self._generate_nonhallucinations_response_removal(
                    question=question,
                    context=context,
                    sentences_response=sentences_response,
                    max_drops=max_nonhall_per_entry,
                )
                entry_nonhallus.extend(resp_drops)

            # Original grounded non-hallucinated entry
            base_prompt = self._format_prompt(question, context)
            orig_grounded = {
                "Question": question,
                "Context": context,
                "response": response,
                "prompt": base_prompt + response,
                "class_hall": "No",
                "label": 1,
                "synthetic_strategy": "original_non_hallucinated",
                "erased_context": "",
                "erased_response": "",
                "rouge1_score": 0.0,
            }
            entry_nonhallus.append(orig_grounded)

            # Propagate common optional metadata keys
            for optional_key in ("sample_id", "uid", "Answerable"):
                if optional_key in entry:
                    orig_grounded[optional_key] = entry[optional_key]
                    for s in entry_hallus:
                        s[optional_key] = entry[optional_key]
                    for s in entry_nonhallus:
                        s[optional_key] = entry[optional_key]

            hallucinations.extend(entry_hallus)
            non_hallucinations.extend(entry_nonhallus)

        print(
            f"[Structured Synth] Generated {len(hallucinations)} hallucinated and "
            f"{len(non_hallucinations)} non-hallucinated candidates before balancing."
        )

        # Include remaining unedited non-hallucinated samples if split
        if rest_pool is not None:
            for entry in rest_pool:
                q = entry.get("Question", "")
                c = entry.get("Context", "")
                r = entry.get("response", "")
                base_p = self._format_prompt(q, c)
                item = {
                    "Question": q,
                    "Context": c,
                    "response": r,
                    "prompt": base_p + r,
                    "class_hall": "No",
                    "label": 1,
                    "synthetic_strategy": "original_non_hallucinated",
                    "erased_context": "",
                    "erased_response": "",
                    "rouge1_score": 0.0,
                }
                for optional_key in ("sample_id", "uid", "Answerable"):
                    if optional_key in entry:
                        item[optional_key] = entry[optional_key]
                non_hallucinations.append(item)

        # Rough balancing between hallucinated (label 0) and non-hallucinated (label 1)
        rng = random.Random(seed)
        n_hall = len(hallucinations)
        n_non_hall = len(non_hallucinations)

        if n_hall > 0 and n_non_hall > 0:
            if n_hall > n_non_hall * balance_ratio:
                target_hall = max(1, int(n_non_hall * balance_ratio))
                hallucinations = rng.sample(hallucinations, target_hall)
            elif n_non_hall > n_hall * balance_ratio:
                target_non_hall = max(1, int(n_hall * balance_ratio))
                non_hallucinations = rng.sample(
                    non_hallucinations, target_non_hall
                )

        combined_train = hallucinations + non_hallucinations
        rng.shuffle(combined_train)

        print(
            f"[Structured Synth] Balanced dataset: {len(hallucinations)} hallucinated and "
            f"{len(non_hallucinations)} non-hallucinated samples (total train: {len(combined_train)})."
        )

        train_dataset = Dataset.from_list(combined_train)

        # Create test split matching organic test split with aligned columns
        organic = self._make_organic_hallucinations_data(data)
        organic_test_split = organic["test"]

        test_len = len(organic_test_split)
        if "synthetic_strategy" not in organic_test_split.column_names:
            organic_test_split = organic_test_split.add_column(
                "synthetic_strategy", ["organic"] * test_len
            )
        if "erased_context" not in organic_test_split.column_names:
            organic_test_split = organic_test_split.add_column(
                "erased_context", [""] * test_len
            )
        if "erased_response" not in organic_test_split.column_names:
            organic_test_split = organic_test_split.add_column(
                "erased_response", [""] * test_len
            )
        if "rouge1_score" not in organic_test_split.column_names:
            organic_test_split = organic_test_split.add_column(
                "rouge1_score", [0.0] * test_len
            )

        synthetic_hallucinations_struct_data = DatasetDict(
            {
                "train": train_dataset,
                "test": organic_test_split,
            }
        )

        return synthetic_hallucinations_struct_data

    def _make_llm_hallucinations_data(
        self, data: DatasetDict, organic_hallucinations_data: DatasetDict
    ) -> DatasetDict:
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
        gemini_model = "gemini-3.5-flash"
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
        self,
        entry: dict,
        fewshot_examples: Dataset,
        client: genai.Client,
        gemini_model: str,
        generation_config: types.GenerateContentConfig,
    ) -> dict:
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

    def _make_perl_data(
        self, data: DatasetDict, sft_data: DatasetDict, seed: int = 12345
    ) -> DatasetDict:
        perl_data = DatasetDict(
            {
                "train": data["validation"],
                "test": data["test"].select(range(10)),
            }
        )

        return perl_data

    def _make_autorater_data(self, data: DatasetDict) -> Dataset:
        """For the autorater, we want to measure its ability on all samples, so we just merge all of them and save as a single test split."""

        autorater_dataset = concatenate_datasets(
            [data["train"], data["validation"], data["test"]]
        )

        return autorater_dataset

    def _make_evaluation_data(self, data: DatasetDict) -> DatasetDict:
        evaluation_data = DatasetDict({"test": data["train"]})

        return evaluation_data

    @staticmethod
    def _flip_entries(
        data: DatasetDict, data_to_flip: DatasetDict
    ) -> DatasetDict:
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
    def _rm_prompt(entry: dict) -> dict:
        """Format a reward model prompt for a Bosch entry.

        Args:
            entry (dict): Entry with 'prompt' and 'response' fields.

        Returns:
            dict: Entry with updated 'prompt' field.
        """

        entry["prompt"] += entry["response"]

        return entry

    @classmethod
    def get_formatting_prompts_and_response_template(
        cls,
        eos_token: str,
        fewshot_examples: Optional[Dataset] = None,
        model_repo_id: Optional[str] = None,
    ) -> Tuple[Callable[[Any], list[str]], str]:
        """Return a formatting function and response template for Bosch SFT.

        For Bosch the response template is fixed and the formatting function
        appends the response to the prompt (the RM prompt already contains the
        rest). This mirrors the old `bosch_formatting_prompts_func` behavior.
        """

        response_template = "\nAnswer to user's question:\n"

        def formatting_prompts_func(entry: dict) -> list[str]:
            template = (
                "You are a helpful assistant to car related questions. You will be given an user's question, and the relevant part of the car manual. Your task is to answer the user's question using the information given.\n"
                "User question:\n{question}"
                "\nManual information:\n{context}"
                "\nAnswer to user's question:\n{answer}{eos_token}"
            )

            output_texts = []

            for i in range(len(entry["Question"])):
                formatted_prompt = template.format(
                    question=entry["Question"][i],
                    context=entry["Context"][i],
                    answer=entry["Answer"][i],
                    eos_token=eos_token,
                )

                output_texts.append(formatted_prompt)

            return output_texts

        return formatting_prompts_func, response_template

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
        shuffled = non_hallucinated_data.shuffle(seed=seed)
        use_all = (
            num_synth_hallus is None
            or num_synth_hallus == -1
            or num_synth_hallus >= len(shuffled)
        )
        if not use_all and num_synth_hallus > 0:
            to_become_hallus = shuffled.select(range(num_synth_hallus))
            non_hallucinated_data_rest = shuffled.select(
                range(num_synth_hallus, len(shuffled))
            )
        else:
            to_become_hallus = shuffled
            non_hallucinated_data_rest = shuffled.select([])

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

    @classmethod
    def get_evaluator_prompt(
        cls,
    ) -> Callable[[dict, Optional[Dataset], bool], dict]:
        """Return a function that builds evaluator prompts for Bosch entries."""

        def bosch_evaluator_prompt(
            entry: dict,
            fewshot_examples: Optional[Dataset] = None,
            use_true_label: bool = False,
        ) -> dict:
            preamble = '<start_of_turn>user\nYou are an expert linguist and fact-checker responsible for determining when an answer contains information not supported by a source text.\n\nIn this task:\n1. A user asks a question about their car\n2. You receive an excerpt from the car\'s manual\n3. You examine a proposed answer to the user\'s question\n\nYour ONLY job is to determine: Does the proposed answer contain ANY statements, claims, instructions, or implications that are NOT explicitly supported by the manual excerpt?\n\nImportant evaluation criteria:\n- If the answer makes ANY claim not directly supported by the manual, respond with "Yes"\n- If the answer contains any logical implication not explicitly appearing in the manual excerpt, valid or not, respond with "Yes"\n- If the answer only rephrases information from the manual without adding new claims, respond with "No"\n- If the answer is FULLY supported by the manual, but truncated at the last sentence (due to word limit), answer "No"\n- Introduction sentences such as "Follow these steps:" and conclusions such as "I hope these helps!" are allowed. If all other sentences are supported by the context, answer "No"\n- Ignore differences in tone, style, or level of detail if the underlying information is supported\n- Focus solely on faithfulness to the source, not helpfulness or completeness\n\nProvide ONLY "Yes" or "No" as your final answer.\n<end_of_turn>\n'

            template = "<start_of_turn>user\nQuestion: {question}\n\nManual excerpt: {context}\n\nProposed answer: {response}\n\nAnalysis:\n1. Compare each statement in the proposed answer to the manual excerpt\n2. Identify any claims in the answer not explicitly supported by the manual\n3. Consider whether the answer introduces new information not present in the manual\n\nDoes the proposed answer state anything not supported by the information in the manual? (Yes/No):\n<end_of_turn>\n<start_of_turn>model\n{ans}\n"

            response = (
                entry["response"] if use_true_label else entry["completion"]
            )

            formatted_prompt = template.format(
                question=entry["Question"],
                context=entry["Context"],
                response=response,
                ans="",
            )

            prompt = preamble

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

        return bosch_evaluator_prompt

    @staticmethod
    def _synthetic_hall_structured(entry: dict, rouge_metric: Any) -> dict:
        """Create a structured synthetic hallucination in Bosch data by removing sentences from the context.

        Args:
            entry (dict): Entry to modify.
            data (datasets.Dataset): The full validation dataset for context.

        Returns:
            dict: Entry with modified context and updated labels.
        """

        # Break response into sentences and filter out small ones
        sentences_context = nltk.sent_tokenize(entry["Context"])
        sentences_response = nltk.sent_tokenize(entry["response"])

        # Pick a random sentence in generation
        random_sentence_response = random.sample(sentences_response, 1)

        # Compute the ROUGE-1 score between this and sentences in context
        # and pick sentence in context with maximal ROUGE-1 score to erase
        best_idx = 0
        max_rouge = 0.0
        has_score_method = hasattr(rouge_metric, "score")
        for idx_context, sentence_context in enumerate(sentences_context):
            if has_score_method:
                res = rouge_metric.score(
                    target=sentence_context,
                    prediction=random_sentence_response[0]
                    if random_sentence_response
                    else "",
                )
                val = res.get("rouge1", 0.0)
                r1 = float(val.fmeasure if hasattr(val, "fmeasure") else val)
            else:
                rouge_results = rouge_metric.compute(
                    predictions=random_sentence_response,
                    references=[sentence_context],
                )
                r1 = float(rouge_results.get("rouge1", 0.0))
            if r1 > max_rouge:
                best_idx = idx_context
                max_rouge = r1

        # Store information about erased sentence
        entry["erased_context"] = sentences_context[best_idx]
        entry["rouge1_score"] = max_rouge

        # Erase the chosen sentence and join everything back together
        sentences_context[best_idx] = ""
        new_context = " ".join(sentences_context)
        entry["Context"] = new_context
        entry["class_hall"] = "Yes"
        entry["label"] = 0
        entry["prompt"] = (
            "You are a helpful assistant to car related questions. You will be given an user's question, and the relevant part of the car manual. Your task is to answer the user's question using the information given. Do not add to your answer any information other than those present in the manual excerpt.\n"
            + "User question:\n"
            + entry["Question"]
            + "\nManual information:\n"
            + entry["Context"]
            + "\nAnswer to user's question:\n"
        )

        # Important: you need to retokenize these later!
        return entry

    @staticmethod
    def _format_prompt(question: str, context: str) -> str:
        """Format the standard prompt for Bosch car manual QA."""
        return (
            "You are a helpful assistant to car related questions. You will be"
            " given an user's question, and the relevant part of the car"
            " manual. Your task is to answer the user's question using the"
            " information given. Do not add to your answer any information"
            " other than those present in the manual excerpt.\n"
            f"User question:\n{question}\nManual information:\n{context}\nAnswer"
            " to user's question:\n"
        )

    @staticmethod
    def _compute_sentence_rouge_scores(
        sentences_context: list[str],
        sentences_response: list[str],
        rouge_metric: Any,
    ) -> list[dict[str, Any]]:
        """Compute peak ROUGE-1 score for each context sentence against all response sentences."""
        scores = []
        has_score_method = hasattr(rouge_metric, "score")
        for ctx_idx, c_sent in enumerate(sentences_context):
            max_rouge = 0.0
            best_resp_idx = 0
            for resp_idx, r_sent in enumerate(sentences_response):
                if has_score_method:
                    res = rouge_metric.score(target=c_sent, prediction=r_sent)
                    val = res.get("rouge1", 0.0)
                    r1 = float(
                        val.fmeasure if hasattr(val, "fmeasure") else val
                    )
                else:
                    res = rouge_metric.compute(
                        predictions=[r_sent],
                        references=[c_sent],
                    )
                    r1 = float(res.get("rouge1", 0.0))
                if r1 > max_rouge:
                    max_rouge = r1
                    best_resp_idx = resp_idx
            scores.append({
                "context_idx": ctx_idx,
                "context_sentence": c_sent,
                "max_rouge": max_rouge,
                "best_response_idx": best_resp_idx,
            })
        return scores

    @classmethod
    def _generate_hallucinations_top_k(
        cls,
        question: str,
        sentences_context: list[str],
        response: str,
        scores: list[dict[str, Any]],
        top_k: int = 3,
        hallu_threshold: float = 0.40,
    ) -> list[dict[str, Any]]:
        """Idea 2: Remove top-k matching context sentences individually to induce hallucinations."""
        if len(sentences_context) < 2:
            return []

        sorted_scores = sorted(
            scores, key=lambda x: x["max_rouge"], reverse=True
        )
        hallucinations = []

        for rank_idx in range(min(top_k, len(sorted_scores))):
            item = sorted_scores[rank_idx]
            if item["max_rouge"] < hallu_threshold:
                continue

            erase_idx = item["context_idx"]
            remaining_ctx = [
                s for idx, s in enumerate(sentences_context) if idx != erase_idx
            ]
            new_context = " ".join(remaining_ctx)
            base_prompt = cls._format_prompt(question, new_context)

            hallucinations.append({
                "Question": question,
                "Context": new_context,
                "response": response,
                "prompt": base_prompt + response,
                "class_hall": "Yes",
                "label": 0,
                "synthetic_strategy": f"context_erasure_top_{rank_idx + 1}",
                "erased_context": item["context_sentence"],
                "erased_response": "",
                "rouge1_score": item["max_rouge"],
            })

        return hallucinations

    @classmethod
    def _generate_nonhallucinations_irrelevant_context(
        cls,
        question: str,
        sentences_context: list[str],
        response: str,
        scores: list[dict[str, Any]],
        irrelevant_threshold: float = 0.15,
        max_candidates: int = 1,
    ) -> list[dict[str, Any]]:
        """Idea 1: Remove irrelevant context sentences (low ROUGE overlap) preserving grounding."""
        if len(sentences_context) < 2:
            return []

        irrelevant_candidates = [
            item for item in scores if item["max_rouge"] <= irrelevant_threshold
        ]
        if not irrelevant_candidates:
            return []

        # Sort by lowest ROUGE score first
        irrelevant_candidates.sort(key=lambda x: x["max_rouge"])
        non_hallucinations = []

        for item in irrelevant_candidates[:max_candidates]:
            erase_idx = item["context_idx"]
            remaining_ctx = [
                s for idx, s in enumerate(sentences_context) if idx != erase_idx
            ]
            new_context = " ".join(remaining_ctx)
            base_prompt = cls._format_prompt(question, new_context)

            non_hallucinations.append({
                "Question": question,
                "Context": new_context,
                "response": response,
                "prompt": base_prompt + response,
                "class_hall": "No",
                "label": 1,
                "synthetic_strategy": "context_erasure_irrelevant",
                "erased_context": item["context_sentence"],
                "erased_response": "",
                "rouge1_score": item["max_rouge"],
            })

        return non_hallucinations

    @classmethod
    def _generate_nonhallucinations_response_removal(
        cls,
        question: str,
        context: str,
        sentences_response: list[str],
        max_drops: int = 1,
    ) -> list[dict[str, Any]]:
        """Idea 3: Remove sentences from generation (coverage reduction without hallucination)."""
        if len(sentences_response) < 2:
            return []

        non_hallucinations = []
        base_prompt = cls._format_prompt(question, context)

        for drop_idx in range(min(max_drops, len(sentences_response))):
            remaining_resp = [
                r
                for idx, r in enumerate(sentences_response)
                if idx != drop_idx
            ]
            new_response = " ".join(remaining_resp).strip()
            if not new_response:
                continue

            non_hallucinations.append({
                "Question": question,
                "Context": context,
                "response": new_response,
                "prompt": base_prompt + new_response,
                "class_hall": "No",
                "label": 1,
                "synthetic_strategy": "response_sentence_removal",
                "erased_context": "",
                "erased_response": sentences_response[drop_idx],
                "rouge1_score": 0.0,
            })

        return non_hallucinations
