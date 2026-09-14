"""Task processor for the RAGTruth dataset.

RAGTruth: A Hallucination Benchmark for Retrieval-Augmented Generation (ACL 2024).
Provides data loading, normalization, SFT splitting, organic/synthetic reward model
dataset generation, PERL prompt extraction, and autorater evaluation formatting.
"""

import collections
import json
import random
import re
from typing import Any, Callable, Optional, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from google import genai
from google.genai import types

from src.task_processors.base_task_processor import BaseTaskProcessor

try:
    import nltk

    try:
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)
    except Exception:
        pass
except ImportError:
    nltk = None


class _PurePythonRougeScorer:
    """Pure-Python in-memory ROUGE-1 F1 scorer with zero network or filesystem overhead."""

    def __init__(self, use_stemmer: bool = True):
        self.stemmer = None
        if use_stemmer:
            try:
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

        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)

        if precision + recall > 0.0:
            fmeasure = 2.0 * (precision * recall) / (precision + recall)
        else:
            fmeasure = 0.0

        return {
            "rouge1": ScoreType(
                precision=precision, recall=recall, fmeasure=fmeasure
            )
        }


class RagtruthTaskProcessor(BaseTaskProcessor):
    """Task processor for the RAGTruth benchmark dataset."""

    @staticmethod
    def _get_rouge_scorer() -> Any:
        """Return an in-memory ROUGE-1 scorer.

        Prefers google-research's official `rouge_score.rouge_scorer.RougeScorer`
        for fast computation. Falls back to `_PurePythonRougeScorer` if not installed.
        """
        try:
            from rouge_score import rouge_scorer

            return rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        except ImportError:
            return _PurePythonRougeScorer(use_stemmer=True)

    def _load_data(self) -> DatasetDict:
        """Load RAGTruth dataset from Hugging Face Hub or local files.

        Supports:
        - Standard pre-split Hugging Face repositories (e.g. wandb/RAGTruth-processed or user repo)
        - Relational JSONL datasets with source_info.jsonl and response.jsonl
        - Explicit file mapping for train and test splits
        """
        repo_or_path = (
            getattr(self.args, "hf_repo", "ParticleMedia/RAGTruth")
            or "ParticleMedia/RAGTruth"
        )
        repos_to_try = [repo_or_path]
        if repo_or_path != "ParticleMedia/RAGTruth":
            repos_to_try.append("ParticleMedia/RAGTruth")

        last_error = None
        for current_repo in repos_to_try:
            # 1. Attempt loading relational JSONL files (response.jsonl + source_info.jsonl, e.g. ParticleMedia/RAGTruth)
            try:
                responses = load_dataset(current_repo, data_files="response.jsonl")["train"]
                source_info = load_dataset(current_repo, data_files="source_info.jsonl")["train"]
                if "split" in responses.column_names:
                    train_resp = responses.filter(lambda x: x["split"] == "train")
                    test_resp = responses.filter(lambda x: x["split"] == "test")
                    raw_dict = DatasetDict({"train": train_resp, "test": test_resp})
                else:
                    splits = responses.train_test_split(
                        test_size=0.15, seed=getattr(self.args, "seed", 12345)
                    )
                    raw_dict = DatasetDict({"train": splits["train"], "test": splits["test"]})
                return self._merge_source_info(raw_dict, source_info)
            except Exception as e:
                last_error = e

            # 2. Attempt loading standard train/test splits or direct dataset
            try:
                data = load_dataset(current_repo)
                if isinstance(data, DatasetDict) and "train" in data and "test" in data:
                    # Check if responses need relational joining with source_info
                    if (
                        "source_id" in data["train"].column_names
                        and "base_content" not in data["train"].column_names
                    ):
                        try:
                            source_info = load_dataset(
                                current_repo, data_files="source_info.jsonl"
                            )["train"]
                            data = self._merge_source_info(data, source_info)
                        except Exception:
                            pass
                    return data
            except Exception as e:
                last_error = e

            # 3. Attempt loading explicit JSONL files (source_info + response or train/test)
            try:
                data = load_dataset(
                    current_repo,
                    data_files={
                        "train": "train.jsonl",
                        "test": "test.jsonl",
                    },
                )
                return data
            except Exception as e:
                last_error = e

            # 4. Fall back to loading single files or default split
            try:
                loaded = load_dataset(current_repo)
                if "train" in loaded:
                    split_data = loaded["train"].train_test_split(
                        test_size=0.15, seed=getattr(self.args, "seed", 12345)
                    )
                    return DatasetDict({
                        "train": split_data["train"],
                        "test": split_data["test"],
                    })
                if isinstance(loaded, DatasetDict):
                    return loaded
            except Exception as e:
                last_error = e

        raise RuntimeError(
            f"Failed to load RAGTruth dataset from '{repo_or_path}': {last_error}"
        )

    @staticmethod
    def _merge_source_info(data: DatasetDict, source_info_ds: Dataset) -> DatasetDict:
        """Join response records with source_info records on source_id."""
        source_map = {}
        for item in source_info_ds:
            s_id = item.get("source_id")
            if s_id is not None:
                source_map[s_id] = item

        merged_splits = {}
        for split_name, split_ds in data.items():
            merged_rows = []
            for row in split_ds:
                s_id = row.get("source_id")
                src_meta = source_map.get(s_id, {})
                combined = {**src_meta, **row}
                merged_rows.append(combined)
            merged_splits[split_name] = Dataset.from_list(merged_rows)

        return DatasetDict(merged_splits)

    @staticmethod
    def _normalize_context(base_content: Any) -> str:
        """Convert heterogeneous base_content (string or dict) to a clean text passage."""
        if not base_content:
            return ""
        if isinstance(base_content, dict):
            lines = []
            for k, v in base_content.items():
                if isinstance(v, (list, tuple)):
                    v_str = ", ".join(str(item) for item in v)
                elif isinstance(v, dict):
                    v_str = json.dumps(v)
                else:
                    v_str = str(v).strip()
                lines.append(f"{k}: {v_str}")
            return "\n".join(lines)
        return str(base_content).strip()

    def _get_target_subtask(self) -> str:
        """Resolve target subtask from args.task_name.

        Returns 'qa' for 'ragtruth' or 'ragtruth-qa', and 'summarization'
        for 'ragtruth-summarization'. Data-to-text is dropped and raises an error.
        """
        task_name = str(
            getattr(self.args, "task_name", "ragtruth-qa") or "ragtruth-qa"
        ).lower()
        if "data" in task_name:
            raise ValueError(
                "Data-to-text subtask has been dropped from RAGTruth in this codebase. "
                "Supported task names are 'ragtruth-qa' and 'ragtruth-summarization'."
            )
        if "sum" in task_name:
            return "summarization"
        return "qa"

    @classmethod
    def _extract_query(cls, entry: dict, target_subtask: str = "qa") -> str:
        """Extract a clean question or task instruction from the entry."""
        for key in ("user_query", "question", "query"):
            val = entry.get(key)
            if val and str(val).strip():
                return str(val).strip()

        task_type = str(entry.get("task_type", "")).lower()
        if "sum" in task_type or target_subtask == "summarization":
            return "Summarize the above document."

        prompt_raw = entry.get("prompt", "")
        if prompt_raw:
            cleaned = re.sub(r"\[/?INST\]|<s>|</s>", "", prompt_raw).strip()
            if cleaned:
                return cleaned

        return "Answer the question based on the provided context."

    @staticmethod
    def _format_prompt(context: str, query: str) -> str:
        """Format the standardized RAG prompt ending with Answer delimiter."""
        return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:\n"

    def _preprocess_data(self, data: DatasetDict) -> DatasetDict:
        """Standardize raw RAGTruth entries into unified schema across all splits."""
        target_subtask = self._get_target_subtask()
        processed = {}

        for split in data.keys():
            split_data = data[split]

            def is_target_subtask(entry: dict) -> bool:
                task_type = str(entry.get("task_type", "")).lower()
                # Explicitly drop data-to-text
                if "data" in task_type or "table" in task_type:
                    return False
                if target_subtask == "summarization":
                    return "sum" in task_type
                else:  # qa
                    if "sum" in task_type:
                        return False
                    return True

            filtered_split = split_data.filter(is_target_subtask)

            def process_entry(entry: dict) -> dict:
                context = self._normalize_context(
                    entry.get(
                        "base_content",
                        entry.get("context", entry.get("passage", "")),
                    )
                )
                query = self._extract_query(entry, target_subtask=target_subtask)
                prompt = self._format_prompt(context, query)

                response = str(
                    entry.get(
                        "response",
                        entry.get("answer", entry.get("completion", "")),
                    )
                ).strip()

                labels = entry.get("labels", [])
                if entry.get("class_hall") is not None:
                    class_hall = entry["class_hall"]
                elif entry.get("hallucination_label") is not None:
                    h_val = entry["hallucination_label"]
                    class_hall = (
                        "Yes"
                        if h_val in (1, "1", "Yes", True, "hallucinated")
                        else "No"
                    )
                else:
                    has_hall = bool(labels and len(labels) > 0)
                    class_hall = "Yes" if has_hall else "No"

                label = 0 if class_hall == "Yes" else 1

                out = dict(entry)
                out["context"] = context
                out["user_query"] = query
                out["prompt"] = prompt
                out["response"] = response
                out["class_hall"] = class_hall
                out["label"] = label
                return out

            processed[split] = filtered_split.map(process_entry)

        return DatasetDict(processed)

    def _make_sft_data(self, data: DatasetDict) -> DatasetDict:
        """Extract verified faithful samples from training split for SFT."""
        faithful = data["train"].filter(lambda entry: entry["class_hall"] == "No")

        if "response" in faithful.column_names:
            faithful = faithful.rename_column("response", "completion")

        seed = getattr(self.args, "seed", 12345)
        # Deterministic 85/15 train/validation split
        try:
            sft_splits = faithful.train_test_split(test_size=0.15, seed=seed)
        except AttributeError:
            shuffled = faithful.shuffle(seed=seed)
            n_total = len(shuffled)
            n_val = max(1, int(n_total * 0.15)) if n_total > 1 else 0
            n_train = n_total - n_val
            sft_splits = DatasetDict({
                "train": shuffled.select(range(n_train)),
                "test": shuffled.select(range(n_train, n_total)),
            })

        return sft_splits

    @staticmethod
    def _rm_prompt(entry: dict) -> dict:
        """Format entry for reward sequence classification by concatenating response to prompt."""
        resp = str(entry.get("response", entry.get("completion", "")) or "").strip()
        prompt = str(entry.get("prompt", "") or "")
        if resp and not prompt.endswith(resp):
            entry["prompt"] = prompt + resp
        return entry

    def _make_organic_hallucinations_data(self, data: DatasetDict) -> DatasetDict:
        """Create organic reward model dataset from human-annotated LLM outputs."""
        organic_train = data["train"].map(self._rm_prompt)
        organic_test = data["test"].map(self._rm_prompt)
        return DatasetDict({"train": organic_train, "test": organic_test})

    def _compute_sentence_rouge_scores(
        self,
        sentences_context: list[str],
        sentences_response: list[str],
        rouge_metric: Any,
    ) -> list[dict[str, Any]]:
        """Compute pairwise sentence ROUGE-1 F1 scores between context and response sentences."""
        scores = []
        for c_idx, c_sent in enumerate(sentences_context):
            max_f1 = 0.0
            best_r_idx = 0
            for r_idx, r_sent in enumerate(sentences_response):
                res = rouge_metric.score(c_sent, r_sent)
                f1 = res["rouge1"].fmeasure
                if f1 > max_f1:
                    max_f1 = f1
                    best_r_idx = r_idx
            scores.append({
                "context_idx": c_idx,
                "context_sentence": c_sent,
                "max_f1": max_f1,
                "matched_response_idx": best_r_idx,
            })
        return scores

    @staticmethod
    def _split_into_sentences(text: str) -> Tuple[list[str], str]:
        """Split text into sentences or lines, returning components and joining delimiter."""
        if not text or not text.strip():
            return [], " "
        stripped = text.strip()
        if "\n" in stripped:
            lines = [line.strip() for line in stripped.split("\n") if line.strip()]
            if len(lines) >= 2:
                return lines, "\n"
        if nltk is not None:
            try:
                sents = nltk.sent_tokenize(stripped)
                return sents, " "
            except Exception:
                pass
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", stripped) if s.strip()]
        return sents, " "

    def _generate_hallucinations_top_k(
        self,
        query: str,
        sentences_context: list[str],
        response: str,
        scores: list[dict[str, Any]],
        top_k: int,
        hallu_threshold: float,
        delimiter: str = " ",
    ) -> list[dict[str, Any]]:
        """Schema 1: Generate hallucinated sample by removing relevant context sentences."""
        sorted_scores = sorted(scores, key=lambda x: x["max_f1"], reverse=True)
        candidates = [s for s in sorted_scores if s["max_f1"] >= hallu_threshold]

        if not candidates:
            return []

        hallus = []
        for rank in range(min(top_k, len(candidates))):
            target = candidates[rank]
            remove_idx = target["context_idx"]
            new_ctx_sentences = [
                s for i, s in enumerate(sentences_context) if i != remove_idx
            ]
            if not new_ctx_sentences:
                continue
            new_context = delimiter.join(new_ctx_sentences)
            prompt = self._format_prompt(new_context, query)
            hallus.append({
                "context": new_context,
                "user_query": query,
                "response": response,
                "prompt": prompt + response,
                "class_hall": "Yes",
                "label": 0,
                "synthetic_strategy": f"erase_top_{rank + 1}_context",
                "erased_context": target["context_sentence"],
                "erased_response": "",
                "rouge1_score": float(target["max_f1"]),
            })
        return hallus

    def _generate_nonhallucinations_irrelevant_context(
        self,
        query: str,
        sentences_context: list[str],
        response: str,
        scores: list[dict[str, Any]],
        irrelevant_threshold: float,
        max_candidates: int,
        delimiter: str = " ",
    ) -> list[dict[str, Any]]:
        """Schema 2: Generate non-hallucinated sample by removing irrelevant context sentences."""
        irrelevant = [s for s in scores if s["max_f1"] <= irrelevant_threshold]
        if not irrelevant:
            return []

        nonhallus = []
        for s in irrelevant[:max_candidates]:
            remove_idx = s["context_idx"]
            new_ctx_sentences = [
                sent for i, sent in enumerate(sentences_context) if i != remove_idx
            ]
            if not new_ctx_sentences:
                continue
            new_context = delimiter.join(new_ctx_sentences)
            prompt = self._format_prompt(new_context, query)
            nonhallus.append({
                "context": new_context,
                "user_query": query,
                "response": response,
                "prompt": prompt + response,
                "class_hall": "No",
                "label": 1,
                "synthetic_strategy": "erase_irrelevant_context",
                "erased_context": s["context_sentence"],
                "erased_response": "",
                "rouge1_score": float(s["max_f1"]),
            })
        return nonhallus

    def _generate_nonhallucinations_response_removal(
        self,
        query: str,
        context: str,
        sentences_response: list[str],
        max_drops: int,
        delimiter: str = " ",
    ) -> list[dict[str, Any]]:
        """Schema 3: Generate non-hallucinated sample by dropping response sentences."""
        if len(sentences_response) < 2:
            return []

        nonhallus = []
        drops_to_try = min(max_drops, len(sentences_response))
        for drop_idx in range(drops_to_try):
            new_resp_sentences = [
                s for i, s in enumerate(sentences_response) if i != drop_idx
            ]
            if not new_resp_sentences:
                continue
            new_resp = delimiter.join(new_resp_sentences)
            prompt = self._format_prompt(context, query)
            nonhallus.append({
                "context": context,
                "user_query": query,
                "response": new_resp,
                "prompt": prompt + new_resp,
                "class_hall": "No",
                "label": 1,
                "synthetic_strategy": "erase_response_sentence",
                "erased_context": "",
                "erased_response": sentences_response[drop_idx],
                "rouge1_score": 0.0,
            })
        return nonhallus

    def _make_structured_hallucinations_data(self, data: DatasetDict) -> DatasetDict:
        """Create structured synthetic hallucinations via ROUGE-based context tampering."""
        args = self.args
        seed = getattr(args, "seed", 12345)
        num_synth_hallus = getattr(args, "num_synth_hallus", 0)
        top_k = getattr(args, "synth_struct_top_k", 3)
        hallu_threshold = getattr(args, "synth_struct_hallu_threshold", 0.40)
        irrelevant_threshold = getattr(args, "synth_struct_irrelevant_threshold", 0.15)
        max_nonhall_per_entry = getattr(args, "synth_struct_max_nonhall_per_entry", 2)
        balance_ratio = getattr(args, "synth_struct_balance_ratio", 1.25)

        faithful_train = data["train"].filter(lambda entry: entry["class_hall"] == "No")
        shuffled = faithful_train.shuffle(seed=seed)

        use_all = (
            num_synth_hallus is None
            or num_synth_hallus == -1
            or num_synth_hallus >= len(shuffled)
        )
        if not use_all and num_synth_hallus > 0:
            source_pool = shuffled.select(range(num_synth_hallus))
            rest_pool = shuffled.select(range(num_synth_hallus, len(shuffled)))
        else:
            source_pool = shuffled
            rest_pool = None

        rouge_metric = self._get_rouge_scorer()
        hallucinations: list[dict[str, Any]] = []
        non_hallucinations: list[dict[str, Any]] = []

        for entry in source_pool:
            query = entry.get("user_query", "")
            context = entry.get("context", "")
            response = entry.get("response", "")

            sentences_context, ctx_delimiter = self._split_into_sentences(context)
            sentences_response, resp_delimiter = self._split_into_sentences(response)

            if not sentences_context or not sentences_response:
                continue

            scores = self._compute_sentence_rouge_scores(
                sentences_context=sentences_context,
                sentences_response=sentences_response,
                rouge_metric=rouge_metric,
            )

            entry_hallus = []
            entry_nonhallus = []

            # Schema 1: Top-k context removal (hallucination)
            if len(sentences_context) >= 2:
                entry_hallus = self._generate_hallucinations_top_k(
                    query=query,
                    sentences_context=sentences_context,
                    response=response,
                    scores=scores,
                    top_k=top_k,
                    hallu_threshold=hallu_threshold,
                    delimiter=ctx_delimiter,
                )

            # Schema 2: Irrelevant context removal (faithful)
            if len(sentences_context) >= 2:
                irrelevant_samples = self._generate_nonhallucinations_irrelevant_context(
                    query=query,
                    sentences_context=sentences_context,
                    response=response,
                    scores=scores,
                    irrelevant_threshold=irrelevant_threshold,
                    max_candidates=max_nonhall_per_entry,
                    delimiter=ctx_delimiter,
                )
                entry_nonhallus.extend(irrelevant_samples)

            # Schema 3: Response sentence removal (faithful)
            if len(sentences_response) >= 2:
                resp_drops = self._generate_nonhallucinations_response_removal(
                    query=query,
                    context=context,
                    sentences_response=sentences_response,
                    max_drops=max_nonhall_per_entry,
                    delimiter=resp_delimiter,
                )
                entry_nonhallus.extend(resp_drops)

            # Original grounded non-hallucinated entry
            base_p = self._format_prompt(context, query)
            orig_grounded = {
                "context": context,
                "user_query": query,
                "response": response,
                "prompt": base_p + response,
                "class_hall": "No",
                "label": 1,
                "synthetic_strategy": "original_non_hallucinated",
                "erased_context": "",
                "erased_response": "",
                "rouge1_score": 0.0,
            }
            entry_nonhallus.append(orig_grounded)

            hallucinations.extend(entry_hallus)
            non_hallucinations.extend(entry_nonhallus)

        # Include remaining unedited faithful samples if split
        if rest_pool is not None:
            for entry in rest_pool:
                q = entry.get("user_query", "")
                c = entry.get("context", "")
                r = entry.get("response", "")
                base_p = self._format_prompt(c, q)
                non_hallucinations.append({
                    "context": c,
                    "user_query": q,
                    "response": r,
                    "prompt": base_p + r,
                    "class_hall": "No",
                    "label": 1,
                    "synthetic_strategy": "original_non_hallucinated",
                    "erased_context": "",
                    "erased_response": "",
                    "rouge1_score": 0.0,
                })

        # Class balancing
        rng = random.Random(seed)
        n_hall = len(hallucinations)
        n_non_hall = len(non_hallucinations)

        if n_hall > 0 and n_non_hall > 0:
            if n_hall > n_non_hall * balance_ratio:
                target_hall = max(1, int(n_non_hall * balance_ratio))
                hallucinations = rng.sample(hallucinations, target_hall)
            elif n_non_hall > n_hall * balance_ratio:
                target_non_hall = max(1, int(n_hall * balance_ratio))
                non_hallucinations = rng.sample(non_hallucinations, target_non_hall)

        combined_train = hallucinations + non_hallucinations
        rng.shuffle(combined_train)

        train_dataset = Dataset.from_list(combined_train)

        # Organic test split with matching columns
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

        test_rows = []
        train_cols = train_dataset.column_names
        for entry in organic_test_split:
            row = {}
            for col in train_cols:
                val = entry.get(col)
                if val is None:
                    if col == "rouge1_score":
                        val = 0.0
                    elif col == "label":
                        val = 1
                    else:
                        val = ""
                row[col] = val
            test_rows.append(row)

        test_dataset = Dataset.from_list(test_rows)

        return DatasetDict({
            "train": train_dataset,
            "test": test_dataset,
        })

    def _make_llm_hallucinations_data(
        self, data: DatasetDict, organic_hallucinations_data: DatasetDict
    ) -> DatasetDict:
        """Generate subtle synthetic hallucinations using Gemini LLM."""
        args = self.args
        gemini_model = "gemini-2.5-flash"
        generation_config = types.GenerateContentConfig(
            temperature=getattr(args, "synth_llm_temperature", 0.7),
            seed=getattr(args, "seed", 12345),
            response_mime_type="text/plain",
        )
        api_key = getattr(args, "gemini_api_key", None)
        client = genai.Client(api_key=api_key) if api_key else genai.Client()

        # Extract faithful training samples
        faithful = data["train"].filter(lambda entry: entry["class_hall"] == "No")
        shuffled = faithful.shuffle(seed=args.seed)

        num_synth = getattr(args, "num_synth_hallus", 0)
        if not isinstance(num_synth, int):
            try:
                num_synth = int(num_synth)
            except Exception:
                num_synth = 0

        use_all = num_synth is None or num_synth == -1 or num_synth >= len(shuffled)
        if not use_all and num_synth > 0:
            to_perturb = shuffled.select(range(num_synth))
            rest = shuffled.select(range(num_synth, len(shuffled)))
        else:
            to_perturb = shuffled
            rest = shuffled.select([])

        # Select few-shot examples from organic hallucinations
        organic_hallus = organic_hallucinations_data["train"].filter(
            lambda x: x["class_hall"] == "Yes"
        )
        n_fewshot = getattr(args, "synth_llm_num_fewshot", 2)
        if not isinstance(n_fewshot, int):
            try:
                n_fewshot = int(n_fewshot)
            except Exception:
                n_fewshot = 2
        fewshot_examples = (
            organic_hallus.select(range(min(n_fewshot, len(organic_hallus))))
            if len(organic_hallus) > 0
            else None
        )

        def map_llm_entry(entry: dict) -> dict:
            prompt_content = self._rm_synthetic_hall_llm(entry, fewshot_examples)
            try:
                resp = client.models.generate_content(
                    model=gemini_model,
                    contents=prompt_content,
                    config=generation_config,
                )
                if (
                    resp is not None
                    and hasattr(resp, "text")
                    and isinstance(resp.text, str)
                    and resp.text.strip()
                ):
                    hallu_text = resp.text.strip()
                else:
                    base_r = str(entry.get("response", ""))
                    hallu_text = f"{base_r} (unsupported detail)"
            except Exception:
                base_r = str(entry.get("response", ""))
                hallu_text = f"{base_r} (unsupported detail)"

            return {
                "response": hallu_text,
                "class_hall": "Yes",
                "label": 0,
            }

        perturbed = to_perturb.map(map_llm_entry)
        perturbed = perturbed.map(self._rm_prompt)
        rest = rest.map(self._rm_prompt)

        combined_train = concatenate_datasets([perturbed, rest]).shuffle(seed=args.seed)

        return DatasetDict({
            "train": combined_train,
            "test": organic_hallucinations_data["test"],
        })

    @staticmethod
    def _rm_synthetic_hall_llm(entry: dict, fewshot_examples: Optional[Dataset] = None) -> str:
        """Construct prompt for Gemini API to generate realistic subtle hallucinations."""
        preamble = """Objective:
Generate a modified version of the given answer that introduces subtle, natural-sounding synthetic hallucinations (factual inaccuracies, unsupported details, or mild misinterpretations not backed by the provided context), mimicking real-world LLM errors.

Guidelines:
- Maintain fluent language and natural style.
- The hallucination should feel plausible but must NOT be directly supported by the context.
- Avoid blatant contradictions; prefer subtle ungrounded details or minor factual drifts.
"""
        prompt = preamble
        if fewshot_examples is not None:
            prompt += "\n--- FEW-SHOT EXAMPLES ---\n"
            for ex in fewshot_examples:
                ctx = ex.get("context", "")
                q = ex.get("user_query", "")
                r = ex.get("response", "")
                prompt += f"\nContext:\n{ctx}\nQuestion:\n{q}\nHallucinated Answer:\n{r}\n"
            prompt += "\n--- NOW PROCESS THE FOLLOWING ---\n"

        ctx = entry.get("context", "")
        q = entry.get("user_query", "")
        r = entry.get("response", "")
        prompt += f"\nContext:\n{ctx}\nQuestion:\n{q}\nOriginal Answer:\n{r}\nModified Hallucinated Answer:\n"
        return prompt

    def _make_perl_data(
        self, data: DatasetDict, sft_data: DatasetDict, seed: int = 12345
    ) -> DatasetDict:
        """Prepare prompts for online policy rollouts in PE-RL (RLOO)."""
        # Deduplicate prompts from sft_data["train"] while retaining full metadata
        sft_train = sft_data["train"].shuffle(seed=seed)
        seen_train_prompts = set()
        unique_train_rows = []
        for entry in sft_train:
            p = entry.get("prompt", "")
            if p not in seen_train_prompts:
                seen_train_prompts.add(p)
                unique_train_rows.append(entry)
        train_prompts = Dataset.from_list(unique_train_rows)

        # Deduplicate prompts from data["test"] and select up to 50
        seen_test_prompts = set()
        unique_test_rows = []
        for entry in data["test"]:
            p = entry.get("prompt", "")
            if p not in seen_test_prompts:
                seen_test_prompts.add(p)
                unique_test_rows.append(entry)
                if len(unique_test_rows) >= 50:
                    break
        test_prompts = Dataset.from_list(unique_test_rows)

        return DatasetDict({"train": train_prompts, "test": test_prompts})

    def _make_autorater_data(self, data: DatasetDict) -> Dataset:
        """Prepare autorater evaluation split from test data with rm_prompt."""
        target_split = (
            data["test"] if "test" in data else next(iter(data.values()))
        )
        autorater_ds = target_split.map(self._rm_prompt)
        return autorater_ds

    def _make_evaluation_data(self, data: DatasetDict) -> DatasetDict:
        """Prepare test split for model completion generation."""
        return DatasetDict({"test": data["test"]})

    @classmethod
    def get_formatting_prompts_and_response_template(
        cls,
        eos_token: str,
        fewshot_examples: Optional[Dataset] = None,
        model_repo_id: Optional[str] = None,
    ) -> Tuple[Callable[[Any], list[str]], str]:
        """Return formatting callable and response delimiter for TRL SFTTrainer."""
        del fewshot_examples, model_repo_id
        response_template = "\nAnswer:\n"

        def formatting_prompts_func(entry: dict) -> list[str]:
            prompts = entry.get("prompt", [])
            completions = entry.get("completion", entry.get("response", []))
            if isinstance(prompts, str):
                prompts = [prompts]
            if isinstance(completions, str):
                completions = [completions]
            output_texts = []
            for p, c in zip(prompts, completions):
                text = f"{p}{c}{eos_token}"
                output_texts.append(text)
            return output_texts

        return formatting_prompts_func, response_template

    @classmethod
    def get_evaluator_prompt(
        cls,
    ) -> Callable[[dict, Optional[Dataset], bool], dict]:
        """Return a function that constructs evaluation prompts for the Gemini autorater."""

        def ragtruth_evaluator_prompt(
            entry: dict,
            fewshot_examples: Optional[Dataset] = None,
            use_true_label: bool = False,
        ) -> dict:
            preamble = (
                "You are an expert evaluator specializing in detecting hallucinations in Retrieval-Augmented Generation (RAG) outputs.\n"
                "Your task is to determine whether the provided answer contains ANY claim, statement, or detail that is NOT supported by the source context.\n\n"
                "Evaluation Guidelines:\n"
                "- A hallucination occurs when the answer includes facts, details, or conclusions absent from the source context.\n"
                "- Reasonable paraphrasing and direct logical deductions are acceptable.\n"
                "- Focus strictly on factual grounding in the context, not on fluency or style.\n"
                "- Output ONLY 'Yes' (hallucination detected) or 'No' (no hallucination detected).\n\n"
            )

            template = (
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer: {answer}\n\n"
                "Does the answer contain ANY information not supported by the context? (Yes/No):{ans}\n"
            )

            context = entry.get("context", "")
            question = entry.get("user_query", entry.get("question", ""))
            response = (
                entry.get("response", entry.get("completion", ""))
                if use_true_label
                else entry.get("completion", entry.get("response", ""))
            )

            prompt = preamble
            if fewshot_examples is not None:
                prompt += "--- EXAMPLES ---\n"
                for ex in fewshot_examples:
                    ex_ctx = ex.get("context", "")
                    ex_q = ex.get("user_query", ex.get("question", ""))
                    ex_r = ex.get("response", ex.get("completion", ""))
                    ex_ans = ex.get("class_hall", "No")
                    prompt += template.format(
                        context=ex_ctx,
                        question=ex_q,
                        answer=ex_r,
                        ans=f" {ex_ans}\n\n",
                    )
                prompt += "--- EVALUATION ---\n"

            prompt += template.format(
                context=context,
                question=question,
                answer=response,
                ans="",
            )

            entry["evaluator_prompt"] = prompt
            return entry

        return ragtruth_evaluator_prompt
