"""Task processor for the RAGTruth dataset.

RAGTruth: A Hallucination Benchmark for Retrieval-Augmented Generation (ACL 2024).
Provides data loading, normalization, SFT splitting, organic/synthetic reward model
dataset generation, PERL prompt extraction, and autorater evaluation formatting.
"""

import collections
import hashlib
import json
import random
import re
from typing import Any, Callable, Optional, Tuple
import urllib.request

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


# Fraction of the official test split reserved for model selection ('dev');
# the remainder is the 'final' pool used only for the reported score.
#
# RAGTruth ships 150 test SOURCE passages per subtask (900 test responses =
# 150 sources x 6 generating LLMs). Because every consumer is deduplicated to
# one prompt per source, 150 is the hard ceiling on unique evaluation prompts
# for a subtask -- there is no way to reach 1,000 from the official test split.
#
# 1/3 therefore gives 50 sources for tuning (reward-model early stopping, the
# autorater threshold, PE-RL eval prompts) and 100 for the reported number,
# which is the largest reported set that still leaves a usable selection pool.
_DEFAULT_TEST_DEV_FRACTION = 1.0 / 3.0


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

    _GITHUB_RAGTRUTH_BASE_URL = (
        "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
    )

    def _load_github_ragtruth(self) -> DatasetDict:
        """Load relational JSONL files directly from official ParticleMedia/RAGTruth GitHub."""
        url_si = f"{self._GITHUB_RAGTRUTH_BASE_URL}/source_info.jsonl"
        url_r = f"{self._GITHUB_RAGTRUTH_BASE_URL}/response.jsonl"

        source_rows = []
        with urllib.request.urlopen(url_si) as resp:
            for line in resp.read().decode("utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                si = item.get("source_info")
                if isinstance(si, dict):
                    if "passages" in si and not item.get("base_content"):
                        item["base_content"] = str(si["passages"]).strip()
                    elif not item.get("base_content"):
                        item["base_content"] = self._normalize_context(si)
                    if "question" in si and not item.get("question"):
                        item["question"] = str(si["question"]).strip()
                    item["source_info"] = json.dumps(si)
                elif isinstance(si, str) and not item.get("base_content"):
                    item["base_content"] = si.strip()
                source_rows.append(item)

        source_info_ds = Dataset.from_list(source_rows)

        train_resp, test_resp, unsplit_resp = [], [], []
        with urllib.request.urlopen(url_r) as resp:
            for line in resp.read().decode("utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if (
                    row.get("class_hall") is None
                    and row.get("hallucination_label") is None
                ):
                    row["class_hall"] = (
                        "Yes"
                        if self._has_hallucination_spans(row.get("labels"))
                        else "No"
                    )
                if "labels" in row and not isinstance(row["labels"], str):
                    row["labels"] = json.dumps(row["labels"])
                sp = row.get("split")
                if sp == "train":
                    train_resp.append(row)
                elif sp == "test":
                    test_resp.append(row)
                else:
                    unsplit_resp.append(row)

        if train_resp or test_resp:
            raw_dict = DatasetDict({
                "train": Dataset.from_list(train_resp),
                "test": Dataset.from_list(test_resp),
            })
        else:
            all_ds = Dataset.from_list(unsplit_resp)
            splits = all_ds.train_test_split(
                test_size=0.15, seed=getattr(self.args, "seed", 12345)
            )
            raw_dict = DatasetDict({
                "train": splits["train"],
                "test": splits["test"],
            })

        return self._merge_source_info(raw_dict, source_info_ds)

    def _load_data(self) -> DatasetDict:
        """Load RAGTruth dataset from Hugging Face Hub, GitHub, or local files.

        Supports:
        - Standard pre-split Hugging Face repositories (e.g. wandb/RAGTruth-processed or user repo)
        - Relational JSONL datasets with source_info.jsonl and response.jsonl (HF Hub or GitHub)
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
                if current_repo == "ParticleMedia/RAGTruth":
                    try:
                        return self._load_github_ragtruth()
                    except Exception as gh_e:
                        last_error = gh_e

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
        for raw_item in source_info_ds:
            item = dict(raw_item)
            s_id = item.get("source_id")
            if s_id is not None:
                si = item.get("source_info")
                if isinstance(si, dict):
                    if "passages" in si and not item.get("base_content"):
                        item["base_content"] = str(si["passages"]).strip()
                    elif not item.get("base_content"):
                        item["base_content"] = RagtruthTaskProcessor._normalize_context(si)
                    if "question" in si and not item.get("question"):
                        item["question"] = str(si["question"]).strip()
                    item["source_info"] = json.dumps(si)
                elif isinstance(si, str) and not item.get("base_content"):
                    item["base_content"] = si.strip()
                source_map[s_id] = item
                source_map[str(s_id)] = item

        merged_splits = {}
        for split_name, split_ds in data.items():
            merged_rows = []
            for row in split_ds:
                s_id = row.get("source_id")
                src_meta = source_map.get(s_id, source_map.get(str(s_id), {}))
                combined = {**src_meta, **row}
                if (
                    combined.get("class_hall") is None
                    and combined.get("hallucination_label") is None
                    and "labels" in combined
                ):
                    combined["class_hall"] = (
                        "Yes"
                        if RagtruthTaskProcessor._has_hallucination_spans(
                            combined.get("labels")
                        )
                        else "No"
                    )
                if "labels" in combined and not isinstance(
                    combined["labels"], str
                ):
                    combined["labels"] = json.dumps(combined["labels"])
                merged_rows.append(combined)
            merged_splits[split_name] = Dataset.from_list(merged_rows)

        return DatasetDict(merged_splits)

    @classmethod
    def _has_hallucination_spans(cls, labels: Any) -> bool:
        """Determine whether a RAGTruth ``labels`` field contains hallucination spans.

        Handles raw Python lists of span dicts (``[{"start": 0, ...}]``),
        JSON-encoded span lists (``"[]"``), and Hugging Face ``datasets``
        ``Sequence(struct)`` column representations
        (``{"start": [], "end": [], "text": [], ...}``), where ``len(labels)``
        equals the number of struct keys even when the span list is empty.
        """
        if not labels:
            return False
        if isinstance(labels, dict):
            for v in labels.values():
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    return True
            return False
        if isinstance(labels, str):
            stripped = labels.strip()
            if not stripped or stripped in ("[]", "{}", "null", "None", "none"):
                return False
            if stripped.startswith(("[", "{")):
                try:
                    parsed = json.loads(stripped)
                    return cls._has_hallucination_spans(parsed)
                except Exception:
                    return True
            return True
        if isinstance(labels, (list, tuple)):
            return len(labels) > 0
        return bool(labels)

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

        si = entry.get("source_info")
        if isinstance(si, dict):
            for key in ("question", "query", "user_query"):
                val = si.get(key)
                if val and str(val).strip():
                    return str(val).strip()
        elif isinstance(si, str) and si.strip().startswith("{"):
            try:
                si_dict = json.loads(si)
                if isinstance(si_dict, dict):
                    for key in ("question", "query", "user_query"):
                        val = si_dict.get(key)
                        if val and str(val).strip():
                            return str(val).strip()
            except Exception:
                pass

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

    @staticmethod
    def _group_key(entry: dict) -> str:
        """Return a stable grouping key identifying the underlying passage.

        RAGTruth ships up to six responses per ``source_id``, all sharing one
        context and prompt. Every split in this processor is performed over
        these keys rather than over rows, so that no passage can leak between
        train and validation, or between the SFT, PE-RL and reward-model
        stages.

        Falls back to a hash of context+query when ``source_id`` is absent,
        so derived datasets that dropped the column still group correctly.

        Args:
            entry: A dataset row.

        Returns:
            A stable string key shared by all rows of the same passage.
        """
        s_id = entry.get("source_id")
        if s_id is not None and str(s_id).strip():
            return f"sid:{s_id}"
        basis = (
            str(entry.get("context", ""))
            + "||"
            + str(entry.get("user_query", ""))
        ).strip("| ")
        if not basis:
            basis = str(entry.get("prompt", ""))
        return "h:" + hashlib.sha1(basis.encode("utf-8")).hexdigest()

    @staticmethod
    def _dataset_from_list(rows: list, template: Dataset) -> Dataset:
        """Build a Dataset from rows, preserving schema when rows is empty.

        ``Dataset.from_list([])`` raises, so an empty result is materialised
        as an empty selection of the template instead.

        Args:
            rows: The rows to wrap.
            template: Dataset whose schema is reused when ``rows`` is empty.

        Returns:
            A Dataset holding ``rows``, or an empty Dataset with the
            template's schema.
        """
        if rows:
            return Dataset.from_list(rows)
        return template.select([])

    def _arg_value(self, name: str, default):
        """Read an argument, falling back only when it is missing or None.

        The `getattr(self.args, name, default) or default` idiom silently
        rewrites a deliberate 0 or 0.0 into the default, which hides
        misconfiguration instead of surfacing it.

        Args:
            name: Attribute name on the argument namespace.
            default: Value to use when the attribute is absent or None.

        Returns:
            The configured value, or the default.
        """
        value = getattr(self.args, name, None)
        return default if value is None else value

    def _block_fractions(self) -> Tuple[float, float]:
        """Return the (SFT, PE-RL) source fractions; the rest goes to the RM.

        Returns:
            The SFT and PE-RL fractions of the training sources.

        Raises:
            ValueError: If the fractions leave no sources for the reward model.
        """
        frac_sft = float(self._arg_value("split_frac_sft", 0.40))
        frac_perl = float(self._arg_value("split_frac_perl", 0.25))
        if frac_sft <= 0 or frac_perl <= 0 or frac_sft + frac_perl >= 1.0:
            raise ValueError(
                "split_frac_sft and split_frac_perl must be positive and sum "
                f"to less than 1.0 (got {frac_sft} and {frac_perl}); the "
                "remainder is the reward-model block."
            )
        return frac_sft, frac_perl

    def _assign_source_blocks(self, train_ds: Dataset) -> dict:
        """Deterministically partition training sources into disjoint blocks.

        Each unique source key is assigned to exactly one of 'sft', 'perl' or
        'rm'. The assignment depends only on the seed and the set of source
        keys -- not on row order, and not on when it is called -- so every
        builder sees the identical partition. This matters because the reward
        model splits are built before the SFT split exists.

        The result is memoised per key set, so repeated calls during one run
        do not re-shuffle.

        Args:
            train_ds: The training split whose sources are to be partitioned.

        Returns:
            A mapping from source key to one of 'sft', 'perl' or 'rm'.
        """
        if "source_id" in train_ds.column_names:
            keys = [
                self._group_key({"source_id": s})
                for s in train_ds["source_id"]
            ]
        else:
            keys = [self._group_key(e) for e in train_ds]

        unique_keys = sorted(set(keys))
        # Hash the whole key set: sampling only the endpoints would let two
        # different sets of equal length reuse a stale partition.
        signature = hashlib.sha1(
            "\x00".join(unique_keys).encode("utf-8")
        ).hexdigest()
        if (
            getattr(self, "_block_signature", None) == signature
            and getattr(self, "_block_map", None) is not None
        ):
            return self._block_map

        seed = int(self._arg_value("seed", 12345))
        rng = random.Random(seed)
        shuffled = list(unique_keys)
        rng.shuffle(shuffled)

        frac_sft, frac_perl = self._block_fractions()
        n_total = len(shuffled)
        n_sft = min(int(round(n_total * frac_sft)), n_total)
        n_perl = min(int(round(n_total * frac_perl)), n_total - n_sft)

        mapping = {}
        for idx, key in enumerate(shuffled):
            if idx < n_sft:
                mapping[key] = "sft"
            elif idx < n_sft + n_perl:
                mapping[key] = "perl"
            else:
                mapping[key] = "rm"

        self._block_map = mapping
        self._block_signature = signature
        return mapping

    def _select_block(self, train_ds: Dataset, block: str) -> Dataset:
        """Return only the rows whose source belongs to the given block.

        Args:
            train_ds: The training split to filter.
            block: One of 'sft', 'perl' or 'rm'.

        Returns:
            The subset of rows whose source was assigned to that block.
        """
        mapping = self._assign_source_blocks(train_ds)
        return train_ds.filter(
            lambda e: mapping.get(self._group_key(e)) == block
        )

    def _test_dev_fraction(self) -> float:
        """Return the fraction of test sources reserved for model selection.

        Returns:
            The 'dev' fraction of the official test split.

        Raises:
            ValueError: If the fraction would leave one of the pools empty.
        """
        frac = float(self._arg_value("test_dev_fraction", _DEFAULT_TEST_DEV_FRACTION))
        if frac <= 0 or frac >= 1.0:
            raise ValueError(
                "test_dev_fraction must be strictly between 0 and 1 (got "
                f"{frac}). Setting it to 0 or 1 would make the reported score "
                "and the model-selection score read the same rows, which is "
                "the leak this split exists to prevent."
            )
        return frac

    def _assign_test_pools(self, test_ds: Dataset) -> dict:
        """Partition the official test sources into 'dev' and 'final' pools.

        The official RAGTruth test split is held out from training, but that
        alone does not make a number computed on it a held-out number. The
        reward model's eval arm, the autorater, and the PE-RL eval prompts all
        read this split, and sweep trials are ranked on their best step, so
        reporting the final score on the same rows reports the maximum of many
        noisy estimates rather than generalisation.

        Splitting by source, not by row, matters for the same reason it does
        on the train side: one source carries up to six responses sharing a
        single prompt.

        The partition depends only on the seed and the key set, and is
        memoised, so every consumer observes the identical pools.

        Args:
            test_ds: The official test split.

        Returns:
            A mapping from source key to either 'dev' or 'final'.
        """
        if "source_id" in test_ds.column_names:
            keys = [
                self._group_key({"source_id": s})
                for s in test_ds["source_id"]
            ]
        else:
            keys = [self._group_key(e) for e in test_ds]

        unique_keys = sorted(set(keys))
        signature = hashlib.sha1(
            "\x00".join(unique_keys).encode("utf-8")
        ).hexdigest()
        if (
            getattr(self, "_test_pool_signature", None) == signature
            and getattr(self, "_test_pool_map", None) is not None
        ):
            return self._test_pool_map

        # Offset the seed so this partition is not correlated with the
        # train-side block assignment (seed) or the synthesis arms (seed + 1).
        seed = int(self._arg_value("seed", 12345)) + 2
        rng = random.Random(seed)
        shuffled = list(unique_keys)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_dev = int(round(n_total * self._test_dev_fraction()))
        if n_total == 1:
            # One source cannot be split two ways. Give it to 'dev' so the
            # selection machinery still has data; 'final' is then empty, which
            # is the correct and visible signal that a single source cannot
            # support a reported score. Note round(1 * 0.5) == 0 in Python
            # (banker's rounding), so without this the 'dev' pool would have
            # silently come out empty instead.
            n_dev = 1
        elif n_total > 1:
            # Never let rounding starve a pool when both can be non-empty.
            n_dev = max(1, min(n_dev, n_total - 1))

        mapping = {
            key: ("dev" if idx < n_dev else "final")
            for idx, key in enumerate(shuffled)
        }

        self._test_pool_map = mapping
        self._test_pool_signature = signature
        return mapping

    def _select_test_pool(self, test_ds: Dataset, pool: str) -> Dataset:
        """Return only the test rows belonging to the given pool.

        Args:
            test_ds: The official test split.
            pool: Either 'dev' (model selection) or 'final' (reported score).

        Returns:
            The subset of rows whose source was assigned to that pool.
        """
        mapping = self._assign_test_pools(test_ds)
        return test_ds.filter(
            lambda e: mapping.get(self._group_key(e)) == pool
        )

    def _cap_responses_per_source(self, ds: Dataset, seed: int) -> Dataset:
        """Limit how many responses per source enter synthetic generation.

        All responses of a source share one context, so tampering each of them
        yields near-duplicate rows.

        Args:
            ds: Rows eligible for synthesis.
            seed: Seed for the shuffle that decides which responses survive.

        Returns:
            A subset holding at most ``synth_struct_max_responses_per_source``
            rows per source. Returns ``ds`` unchanged when the cap is <= 0.
        """
        max_per = int(
            self._arg_value("synth_struct_max_responses_per_source", 2)
        )
        if max_per <= 0:
            return ds
        shuffled = ds.shuffle(seed=seed)
        counts: dict = {}
        keep_indices = []
        for idx, entry in enumerate(shuffled):
            key = self._group_key(entry)
            seen = counts.get(key, 0)
            if seen < max_per:
                counts[key] = seen + 1
                keep_indices.append(idx)
        return shuffled.select(keep_indices)

    def _split_synthesis_arms(self, ds: Dataset, seed: int) -> Tuple[set, set]:
        """Partition sources into (perturbed, clean) synthesis arms.

        The arms are disjoint at source level so that the reward model never
        sees the same context with both a hallucinated and a faithful label.

        Args:
            ds: Rows eligible for synthesis.
            seed: Base seed; offset internally to decorrelate from the block
                assignment, which is drawn from the same seed.

        Returns:
            A (perturbed_keys, clean_keys) pair of disjoint source-key sets.
        """
        keys = sorted({self._group_key(e) for e in ds})
        # Offset the seed so the arm split is not correlated with the
        # SFT/PE-RL/RM block assignment drawn from the same seed.
        rng = random.Random(int(seed) + 1)
        shuffled = list(keys)
        rng.shuffle(shuffled)
        frac = float(self._arg_value("synth_perturb_fraction", 0.5))
        n_perturb = int(round(len(shuffled) * frac))
        return set(shuffled[:n_perturb]), set(shuffled[n_perturb:])

    def _preprocess_data(self, data: DatasetDict) -> DatasetDict:
        """Standardize raw RAGTruth entries into unified schema across all splits."""
        target_subtask = self._get_target_subtask()
        drop_bad_quality = getattr(self.args, "drop_bad_quality", True)
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

            # Drop annotator-flagged rows ('incorrect_refusal', 'truncated').
            # These are labelled non-hallucinated, so without this filter the
            # refusals flow into the SFT targets and teach the policy to
            # refuse. In QA they are 142 train / 25 test rows.
            if drop_bad_quality and "quality" in filtered_split.column_names:
                filtered_split = filtered_split.filter(
                    lambda e: str(e.get("quality", "good")).strip().lower()
                    == "good"
                )

            def process_entry(entry: dict) -> dict:
                raw_ctx = entry.get(
                    "base_content",
                    entry.get("context", entry.get("passage")),
                )
                if not raw_ctx:
                    si = entry.get("source_info")
                    if isinstance(si, dict):
                        raw_ctx = si.get("passages", si)
                    elif isinstance(si, str) and si.strip().startswith("{"):
                        try:
                            si_dict = json.loads(si)
                            if isinstance(si_dict, dict):
                                raw_ctx = si_dict.get("passages", si_dict)
                            else:
                                raw_ctx = si
                        except Exception:
                            raw_ctx = si
                    else:
                        raw_ctx = si
                context = self._normalize_context(raw_ctx)
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
                    has_hall = self._has_hallucination_spans(labels)
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

            mapped_split = filtered_split.map(process_entry)
            if "labels" in mapped_split.column_names:
                mapped_split = mapped_split.remove_columns("labels")
            processed[split] = mapped_split

        return DatasetDict(processed)

    def _make_sft_data(self, data: DatasetDict) -> DatasetDict:
        """Extract verified faithful samples from the SFT source block."""
        sft_pool = self._select_block(data["train"], "sft")
        faithful = sft_pool.filter(lambda entry: entry["class_hall"] == "No")

        if "response" in faithful.column_names:
            faithful = faithful.rename_column("response", "completion")
        if "labels" in faithful.column_names:
            faithful = faithful.remove_columns("labels")

        seed = int(self._arg_value("seed", 12345))
        val_fraction = float(self._arg_value("sft_val_fraction", 0.15))

        # Split by source, never by row. A row-level split puts the same
        # prompt in both train and validation, because one source contributes
        # up to six responses.
        keys = sorted({self._group_key(e) for e in faithful})
        rng = random.Random(seed)
        shuffled_keys = list(keys)
        rng.shuffle(shuffled_keys)

        n_val = int(len(shuffled_keys) * val_fraction)
        # Guarantee a non-empty validation split when one was requested, but
        # honour an explicit 0.0 meaning "no validation split".
        if val_fraction > 0 and len(shuffled_keys) > 1:
            n_val = max(1, n_val)
        val_keys = set(shuffled_keys[:n_val])

        train_split = faithful.filter(
            lambda e: self._group_key(e) not in val_keys
        )
        val_split = faithful.filter(lambda e: self._group_key(e) in val_keys)

        if len(sft_pool) > 0 and len(faithful) == 0:
            raise RuntimeError(
                "SFT split is empty (0 faithful samples found in non-empty SFT block). "
                "Check hallucination span label parsing in _preprocess_data."
            )

        return DatasetDict({"train": train_split, "test": val_split})

    @staticmethod
    def _rm_prompt(entry: dict) -> dict:
        """Format entry for reward sequence classification by concatenating response to prompt."""
        resp = str(entry.get("response", entry.get("completion", "")) or "").strip()
        prompt = str(entry.get("prompt", "") or "")
        if resp and not prompt.endswith(resp):
            entry["prompt"] = prompt + resp
        return entry

    def _make_organic_hallucinations_data(self, data: DatasetDict) -> DatasetDict:
        """Create organic reward model dataset from the RM source block.

        Training rows are restricted to the reward-model block so the RM is
        never trained on prompts the policy was fine-tuned on or will be
        rolled out on.

        The eval arm reads the test 'dev' pool, not the whole test split.
        Holding the official test split out of *training* is not by itself
        enough: the reward model is early-stopped and selected on this arm, so
        scoring the final result on the same rows would report the best of
        many noisy estimates rather than generalisation. The 'final' pool is
        reserved for `_make_evaluation_data` alone.

        Args:
            data: Preprocessed dataset with 'train' and 'test' splits.

        Returns:
            A DatasetDict of reward-model rows whose 'prompt' already has the
            response appended.
        """
        rm_train = self._select_block(data["train"], "rm")
        organic_train = rm_train.map(self._rm_prompt)
        organic_test = self._select_test_pool(data["test"], "dev").map(
            self._rm_prompt
        )
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

        # Restrict synthesis to the reward-model source block (disjoint from
        # the SFT and PE-RL blocks) so the reward model never scores contexts
        # the policy was fine-tuned on or will be rolled out on.
        rm_faithful = self._select_block(data["train"], "rm").filter(
            lambda entry: entry["class_hall"] == "No"
        )
        # RAGTruth ships up to 6 responses per source_id, all sharing a single
        # context. Tampering every one of them produces near-duplicate rows
        # that teach the reward model to memorise contexts, so cap how many
        # responses per source enter synthesis.
        rm_faithful = self._cap_responses_per_source(rm_faithful, seed)

        # Split into two source-disjoint arms: contexts in the perturbed arm
        # only ever yield hallucinated rows, contexts in the clean arm only
        # ever yield faithful rows. Without this, the same context appears on
        # both sides of the label boundary differing only by which sentences
        # were deleted, and "how much context was removed" becomes the easiest
        # discriminative signal.
        perturb_keys, clean_keys = self._split_synthesis_arms(rm_faithful, seed)
        source_pool = rm_faithful.filter(
            lambda e: self._group_key(e) in perturb_keys
        )
        rest_pool = rm_faithful.filter(
            lambda e: self._group_key(e) in clean_keys
        )

        # num_synth_hallus, when positive, further caps the perturbed arm.
        if 0 < num_synth_hallus < len(source_pool):
            source_pool = source_pool.shuffle(seed=seed).select(
                range(num_synth_hallus)
            )

        max_hallu_per_entry = int(
            self._arg_value("synth_struct_max_hallu_per_entry", 1)
        )
        effective_top_k = max(1, min(int(top_k), max_hallu_per_entry))

        rouge_metric = self._get_rouge_scorer()
        hallucinations: list[dict[str, Any]] = []
        non_hallucinations: list[dict[str, Any]] = []

        # Perturbed arm: hallucinated rows only (schema 1).
        for entry in source_pool:
            query = entry.get("user_query", "")
            context = entry.get("context", "")
            response = entry.get("response", "")

            sentences_context, ctx_delimiter = self._split_into_sentences(context)
            sentences_response, _ = self._split_into_sentences(response)

            if not sentences_context or not sentences_response:
                continue
            if len(sentences_context) < 2:
                continue

            scores = self._compute_sentence_rouge_scores(
                sentences_context=sentences_context,
                sentences_response=sentences_response,
                rouge_metric=rouge_metric,
            )

            generated = self._generate_hallucinations_top_k(
                query=query,
                sentences_context=sentences_context,
                response=response,
                scores=scores,
                top_k=effective_top_k,
                hallu_threshold=hallu_threshold,
                delimiter=ctx_delimiter,
            )
            # Keep provenance: the generators build fresh dicts, and without
            # the originating source_id the synthetic rows cannot be traced
            # back to their passage or grouped for any later split.
            for row in generated:
                row["source_id"] = entry.get("source_id")
            hallucinations.extend(generated)

        # Clean arm: faithful rows only (schemas 2 and 3, plus the original).
        for entry in rest_pool:
            query = entry.get("user_query", "")
            context = entry.get("context", "")
            response = entry.get("response", "")

            sentences_context, ctx_delimiter = self._split_into_sentences(context)
            sentences_response, resp_delimiter = self._split_into_sentences(response)

            if not sentences_context or not sentences_response:
                continue

            entry_nonhallus = []

            # Schema 2: Irrelevant context removal (still faithful). This also
            # shortens the context, matching the perturbed arm, so context
            # length alone cannot separate the classes.
            if len(sentences_context) >= 2:
                scores = self._compute_sentence_rouge_scores(
                    sentences_context=sentences_context,
                    sentences_response=sentences_response,
                    rouge_metric=rouge_metric,
                )
                entry_nonhallus.extend(
                    self._generate_nonhallucinations_irrelevant_context(
                        query=query,
                        sentences_context=sentences_context,
                        response=response,
                        scores=scores,
                        irrelevant_threshold=irrelevant_threshold,
                        max_candidates=max_nonhall_per_entry,
                        delimiter=ctx_delimiter,
                    )
                )

            # Schema 3: Response sentence removal (still faithful).
            if len(sentences_response) >= 2:
                entry_nonhallus.extend(
                    self._generate_nonhallucinations_response_removal(
                        query=query,
                        context=context,
                        sentences_response=sentences_response,
                        max_drops=max_nonhall_per_entry,
                        delimiter=resp_delimiter,
                    )
                )

            # Original grounded non-hallucinated entry
            base_p = self._format_prompt(context, query)
            entry_nonhallus.append({
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
            })

            for row in entry_nonhallus:
                row["source_id"] = entry.get("source_id")
            non_hallucinations.extend(entry_nonhallus)

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

        # Restrict synthesis to the reward-model source block (disjoint from
        # the SFT and PE-RL blocks), then split that block into two
        # source-disjoint arms so that no context is ever seen by the reward
        # model with both a faithful and a hallucinated label.
        rm_faithful = self._select_block(data["train"], "rm").filter(
            lambda entry: entry["class_hall"] == "No"
        )
        rm_faithful = self._cap_responses_per_source(rm_faithful, args.seed)
        perturb_keys, clean_keys = self._split_synthesis_arms(
            rm_faithful, args.seed
        )

        to_perturb = rm_faithful.filter(
            lambda e: self._group_key(e) in perturb_keys
        )
        rest = rm_faithful.filter(lambda e: self._group_key(e) in clean_keys)

        # num_synth_hallus, when positive, further caps the perturbed arm.
        num_synth = getattr(args, "num_synth_hallus", -1)
        if not isinstance(num_synth, int):
            try:
                num_synth = int(num_synth)
            except (TypeError, ValueError):
                num_synth = -1
        if 0 < num_synth < len(to_perturb):
            to_perturb = to_perturb.shuffle(seed=args.seed).select(
                range(num_synth)
            )


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
        """Prepare prompts for online policy rollouts in PE-RL (RLOO).

        Prompts are drawn from the dedicated PE-RL source block, which is
        disjoint (at ``source_id`` level) from both the SFT and the reward
        model blocks.

        ``sft_data`` is accepted for signature compatibility with
        ``BaseTaskProcessor.run`` but is deliberately unused: rolling out on
        the very prompts the policy was fine-tuned on biases PE-RL towards
        memorised completions, and scoring those rollouts with a reward model
        trained on the same prompts hides reward hacking.

        Args:
            data: Preprocessed dataset with 'train' and 'test' splits.
            sft_data: Unused; see above.
            seed: Seed for the shuffle applied before prompt deduplication.

        Returns:
            A DatasetDict with 'train' rollout prompts drawn from the PE-RL
            block and 'test' prompts drawn from the held-out test split.
        """
        del sft_data  # Intentionally unused; see docstring.

        perl_train = self._select_block(data["train"], "perl")

        # Deduplicate prompts while retaining full metadata.
        shuffled = perl_train.shuffle(seed=seed)
        seen_train_prompts = set()
        unique_train_rows = []
        for entry in shuffled:
            p = entry.get("prompt", "")
            if p not in seen_train_prompts:
                seen_train_prompts.add(p)
                unique_train_rows.append(entry)
        train_prompts = self._dataset_from_list(unique_train_rows, perl_train)

        # Deduplicate prompts from the test 'dev' pool. These prompts drive
        # eval during PE-RL and therefore checkpoint selection, so they must
        # not come from the 'final' pool the reported score is computed on.
        max_test_prompts = int(self._arg_value("perl_num_test_prompts", 50))
        dev_pool = self._select_test_pool(data["test"], "dev")
        seen_test_prompts = set()
        unique_test_rows = []
        for entry in dev_pool:
            p = entry.get("prompt", "")
            if p not in seen_test_prompts:
                seen_test_prompts.add(p)
                unique_test_rows.append(entry)
                if len(unique_test_rows) >= max_test_prompts:
                    break
        test_prompts = self._dataset_from_list(unique_test_rows, dev_pool)

        return DatasetDict({"train": train_prompts, "test": test_prompts})

    def _make_autorater_data(self, data: DatasetDict) -> Dataset:
        """Prepare autorater calibration data from the test 'dev' pool.

        The autorater's decision threshold is fitted on these rows, which
        makes this a model-selection step. It therefore reads the 'dev' pool
        and never the 'final' pool that the reported score is computed on.

        Args:
            data: Preprocessed dataset, normally with a 'test' split.

        Returns:
            A Dataset whose 'prompt' already has the response appended.
        """
        if "test" in data:
            target_split = self._select_test_pool(data["test"], "dev")
        else:
            target_split = next(iter(data.values()))
        autorater_ds = target_split.map(self._rm_prompt)
        return autorater_ds

    def _make_evaluation_data(self, data: DatasetDict) -> DatasetDict:
        """Prepare the reported evaluation set: the test 'final' pool.

        This is the only consumer of the 'final' pool. Nothing else -- not the
        reward model's eval arm, not the autorater's threshold, not the PE-RL
        eval prompts, and therefore not checkpoint or trial selection -- is
        allowed to observe these sources, so the score computed here measures
        generalisation rather than the maximum of many noisy estimates.

        Args:
            data: Preprocessed dataset with a 'test' split.

        Returns:
            A DatasetDict with a single 'test' split holding the final pool.
        """
        return DatasetDict({
            "test": self._select_test_pool(data["test"], "final")
        })

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
