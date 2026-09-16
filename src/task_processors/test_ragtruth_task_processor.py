"""Hermetic unit tests for RagtruthTaskProcessor.

Tests:
1. Pure Python ROUGE-1 F1 scorer precision, recall, and harmonic mean.
2. Context normalization (strings, dicts, lists, nested dicts, empty).
3. Query extraction (explicit keys, task types, instruction cleaning, defaults).
4. Label conversion (labels span list, hallucination_label, existing class_hall).
5. Relational join (source_info and response on source_id).
6. SFT data creation (filtering faithful only, renaming completion, 85/15 split).
7. Organic RM data creation (_rm_prompt formatting).
8. Synthetic hallucination generation (Schema 1: top-k removal, Schema 2: irrelevant removal,
   Schema 3: response removal, class balancing, column alignment).
9. PERL data extraction (prompts-only splits).
10. Autorater and evaluation splits.
11. Downstream callables (formatting prompts, response template, evaluator prompt).
"""

import json
import re
import sys
import types
import unittest
from unittest.mock import MagicMock, Mock

# Hermetic mocks for external heavy libraries if not in the current environment

if "nltk" not in sys.modules:
  nltk_mock = types.ModuleType("nltk")
  nltk_mock.download = lambda *args, **kwargs: None

  def _mock_sent_tokenize(text):
    if not text or not text.strip():
      return []
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

  nltk_mock.sent_tokenize = _mock_sent_tokenize
  sys.modules["nltk"] = nltk_mock


class MockRougeMetric:
  """Lightweight deterministic ROUGE-1 F1 computer for testing without network."""

  def compute(self, predictions, references):
    pred_words = set(predictions[0].lower().split()) if predictions else set()
    ref_words = set(references[0].lower().split()) if references else set()
    if not pred_words or not ref_words:
      return {"rouge1": 0.0}
    overlap = len(pred_words.intersection(ref_words))
    prec = overlap / len(pred_words)
    rec = overlap / len(ref_words)
    if prec + rec == 0:
      return {"rouge1": 0.0}
    f1 = 2 * (prec * rec) / (prec + rec)
    return {"rouge1": f1}


if "evaluate" not in sys.modules:
  evaluate_mock = types.ModuleType("evaluate")
  evaluate_mock.load = lambda metric_name: MockRougeMetric()
  sys.modules["evaluate"] = evaluate_mock

if "google" not in sys.modules:
  google_mock = types.ModuleType("google")
  genai_mock = types.ModuleType("genai")
  genai_mock.Client = MagicMock
  types_mock = types.ModuleType("types")
  types_mock.GenerateContentConfig = MagicMock
  genai_mock.types = types_mock
  google_mock.genai = genai_mock
  sys.modules["google"] = google_mock
  sys.modules["google.genai"] = genai_mock
  sys.modules["google.genai.types"] = types_mock
elif "google.genai" not in sys.modules:
  genai_mock = types.ModuleType("genai")
  genai_mock.Client = MagicMock
  types_mock = types.ModuleType("types")
  types_mock.GenerateContentConfig = MagicMock
  genai_mock.types = types_mock
  sys.modules["google.genai"] = genai_mock
  sys.modules["google.genai.types"] = types_mock

if "datasets" not in sys.modules:

  class _MockDataset:

    def __init__(self, data):
      if isinstance(data, list):
        keys = []
        for d in data:
          for k in d.keys():
            if k not in keys:
              keys.append(k)
        self._data = {k: [d.get(k) for d in data] for k in keys}
      elif isinstance(data, dict):
        self._data = dict(data)
      else:
        self._data = {}
      self.column_names = list(self._data.keys())

    def __getitem__(self, key):
      if isinstance(key, int):
        return {k: self._data[k][key] for k in self._data}
      return self._data[key]

    def __len__(self):
      first_col = next(iter(self._data.values())) if self._data else []
      return len(first_col)

    def to_dict(self):
      return dict(self._data)

    def add_column(self, column_name, values):
      new_data = dict(self._data)
      new_data[column_name] = list(values)
      return _MockDataset(new_data)

    def remove_columns(self, column_name):
      new_data = {k: v for k, v in self._data.items() if k != column_name}
      return _MockDataset(new_data)

    def rename_column(self, original_column_name, new_column_name):
      new_data = {
          (new_column_name if k == original_column_name else k): v
          for k, v in self._data.items()
      }
      return _MockDataset(new_data)

    def select_columns(self, column_names):
      new_data = {k: v for k, v in self._data.items() if k in column_names}
      return _MockDataset(new_data)

    def select(self, indices):
      new_data = {k: [self._data[k][i] for i in indices] for k in self._data}
      return _MockDataset(new_data)

    def shuffle(self, seed=42):
      del seed
      return self

    def filter(self, fn):
      num_rows = len(self)
      new_rows = []
      for i in range(num_rows):
        entry = {k: self._data[k][i] for k in self._data}
        if fn(entry):
          new_rows.append(entry)
      if not new_rows:
        return _MockDataset({k: [] for k in self._data})
      new_data = {k: [row[k] for row in new_rows] for k in self._data}
      return _MockDataset(new_data)

    def map(self, fn, fn_kwargs=None):
      kwargs = fn_kwargs or {}
      num_rows = len(self)
      new_rows = []
      for i in range(num_rows):
        entry = {k: self._data[k][i] for k in self._data}
        res = fn(entry, **kwargs)
        if isinstance(res, dict):
          entry.update(res)
        new_rows.append(entry)
      if not new_rows:
        return _MockDataset({k: [] for k in self._data})
      all_keys = list(new_rows[0].keys())
      new_data = {k: [row[k] for row in new_rows] for k in all_keys}
      return _MockDataset(new_data)

    def __iter__(self):
      num_rows = len(self)
      for i in range(num_rows):
        yield {k: self._data[k][i] for k in self._data}

  class Dataset:

    @classmethod
    def from_dict(cls, d):
      return _MockDataset(d)

    @classmethod
    def from_list(cls, lst, **kwargs):
      del kwargs
      return _MockDataset(lst)

  class DatasetDict(dict):

    def map(self, fn, **kwargs):
      return DatasetDict({k: v.map(fn, **kwargs) for k, v in self.items()})

  datasets_mock = types.ModuleType("datasets")
  datasets_mock.Dataset = Dataset
  datasets_mock.DatasetDict = DatasetDict
  datasets_mock.concatenate_datasets = lambda d_list: _MockDataset(
      {
          k: [item for d in d_list for item in d._data.get(k, [])]
          for k in set().union(*(d._data.keys() for d in d_list))
      }
      if d_list
      else {}
  )
  datasets_mock.load_dataset = MagicMock()
  sys.modules["datasets"] = datasets_mock
else:
  from datasets import Dataset, DatasetDict, concatenate_datasets

from src.task_processors.ragtruth_task_processor import (
    _PurePythonRougeScorer,
    RagtruthTaskProcessor,
)


def make_mock_args(**overrides):
  """Builds a Mock args namespace with realistic numeric defaults.

  A bare `Mock()` returns a `Mock` for every attribute access, so any knob the
  processor reads with `getattr(self.args, name, default)` resolves to a Mock
  rather than the default. That blows up as soon as the value is used in
  arithmetic. This factory pins the values that actually matter.

  Args:
    **overrides: Attribute values to override on top of the defaults.

  Returns:
    A Mock configured with the processor's expected argument namespace.
  """
  args = Mock()
  args.seed = 12345
  args.task_name = "ragtruth"
  args.subtask = None
  args.drop_bad_quality = True
  args.split_frac_sft = 0.40
  args.split_frac_perl = 0.25
  args.sft_val_fraction = 0.15
  args.perl_num_test_prompts = 50
  args.num_synth_hallus = -1
  args.synth_struct_top_k = 3
  args.synth_struct_hallu_threshold = 0.40
  args.synth_struct_irrelevant_threshold = 0.15
  args.synth_struct_max_nonhall_per_entry = 2
  args.synth_struct_balance_ratio = 1.25
  args.synth_struct_max_responses_per_source = 2
  args.synth_struct_max_hallu_per_entry = 1
  args.synth_perturb_fraction = 0.5
  for key, value in overrides.items():
    setattr(args, key, value)
  return args


class TestPurePythonRougeScorer(unittest.TestCase):
  """Tests for the zero-dependency pure-Python in-memory ROUGE-1 scorer."""

  def setUp(self):
    self.scorer = _PurePythonRougeScorer(use_stemmer=False)

  def test_exact_match(self):
    res = self.scorer.score("the cat sat on the mat", "the cat sat on the mat")
    score = res["rouge1"]
    self.assertAlmostEqual(score.precision, 1.0)
    self.assertAlmostEqual(score.recall, 1.0)
    self.assertAlmostEqual(score.fmeasure, 1.0)

  def test_partial_overlap(self):
    res = self.scorer.score("the quick brown fox", "the brown fox jumps")
    score = res["rouge1"]
    # Target tokens: [the, quick, brown, fox] (4 tokens)
    # Pred tokens: [the, brown, fox, jumps] (4 tokens)
    # Overlap: the, brown, fox (3 tokens)
    # Precision = 3/4 = 0.75, Recall = 3/4 = 0.75, F1 = 0.75
    self.assertAlmostEqual(score.precision, 0.75)
    self.assertAlmostEqual(score.recall, 0.75)
    self.assertAlmostEqual(score.fmeasure, 0.75)

  def test_zero_overlap(self):
    res = self.scorer.score("apple orange banana", "cat dog elephant")
    score = res["rouge1"]
    self.assertEqual(score.precision, 0.0)
    self.assertEqual(score.recall, 0.0)
    self.assertEqual(score.fmeasure, 0.0)

  def test_empty_strings(self):
    res1 = self.scorer.score("", "hello world")
    self.assertEqual(res1["rouge1"].fmeasure, 0.0)
    res2 = self.scorer.score("hello world", "")
    self.assertEqual(res2["rouge1"].fmeasure, 0.0)
    res3 = self.scorer.score("", "")
    self.assertEqual(res3["rouge1"].fmeasure, 0.0)

  def test_case_and_punctuation(self):
    res = self.scorer.score("Hello, World!", "hello world")
    score = res["rouge1"]
    self.assertAlmostEqual(score.fmeasure, 1.0)


class TestRagtruthTaskProcessorPreprocessing(unittest.TestCase):
  """Tests for context normalization, query extraction, label conversion, and preprocessing."""

  def setUp(self):
    self.mock_args = make_mock_args(seed=12345, task_name="ragtruth")
    self.processor = RagtruthTaskProcessor(self.mock_args)

  def test_normalize_context_string(self):
    raw_str = "   This is a passage about quantum physics.\nIt continues here.   "
    normalized = RagtruthTaskProcessor._normalize_context(raw_str)
    self.assertEqual(
        normalized,
        "This is a passage about quantum physics.\nIt continues here.",
    )

  def test_normalize_context_dict(self):
    raw_dict = {
        "title": "Quantum Computing",
        "authors": ["Alice", "Bob"],
        "year": 2024,
        "metadata": {"journal": "Nature"},
    }
    normalized = RagtruthTaskProcessor._normalize_context(raw_dict)
    self.assertIn("title: Quantum Computing", normalized)
    self.assertIn("authors: Alice, Bob", normalized)
    self.assertIn("year: 2024", normalized)
    self.assertIn('metadata: {"journal": "Nature"}', normalized)

  def test_normalize_context_empty(self):
    self.assertEqual(RagtruthTaskProcessor._normalize_context(None), "")
    self.assertEqual(RagtruthTaskProcessor._normalize_context(""), "")
    self.assertEqual(RagtruthTaskProcessor._normalize_context({}), "")

  def test_extract_query_explicit_keys(self):
    entry1 = {"user_query": "What is photon entanglement?"}
    self.assertEqual(
        RagtruthTaskProcessor._extract_query(entry1),
        "What is photon entanglement?",
    )

    entry2 = {"question": "How fast does light travel?"}
    self.assertEqual(
        RagtruthTaskProcessor._extract_query(entry2),
        "How fast does light travel?",
    )

    entry3 = {"query": "Explain qubit coherence."}
    self.assertEqual(
        RagtruthTaskProcessor._extract_query(entry3),
        "Explain qubit coherence.",
    )

  def test_extract_query_task_type_fallback(self):
    summary_entry = {"task_type": "text_summarization"}
    self.assertEqual(
        RagtruthTaskProcessor._extract_query(
            summary_entry, target_subtask="summarization"
        ),
        "Summarize the above document.",
    )

    qa_entry = {"task_type": "qa"}
    self.assertEqual(
        RagtruthTaskProcessor._extract_query(qa_entry, target_subtask="qa"),
        "Answer the question based on the provided context.",
    )

  def test_extract_query_inst_prompt_cleaning(self):
    prompt_entry = {"prompt": "[INST] What is general relativity? [/INST]"}
    self.assertEqual(
        RagtruthTaskProcessor._extract_query(prompt_entry),
        "What is general relativity?",
    )

  def test_extract_query_default(self):
    empty_entry = {}
    self.assertEqual(
        RagtruthTaskProcessor._extract_query(empty_entry),
        "Answer the question based on the provided context.",
    )

  def test_format_prompt(self):
    prompt = RagtruthTaskProcessor._format_prompt(
        "Context line 1", "Question line 1"
    )
    expected = (
        "Context:\nContext line 1\n\nQuestion: Question line 1\n\nAnswer:\n"
    )
    self.assertEqual(prompt, expected)

  def test_merge_source_info(self):
    source_ds = Dataset.from_list([
        {
            "source_id": 101,
            "base_content": "Source text 101",
            "task_type": "qa",
            "question": "Q101",
        },
        {
            "source_id": 102,
            "base_content": "Source text 102",
            "task_type": "summary",
        },
    ])
    response_ds = Dataset.from_list([
        {"source_id": 101, "response": "Response 101", "labels": []},
        {
            "source_id": 102,
            "response": "Response 102",
            "labels": [{"text": "wrong", "start": 0, "end": 5}],
        },
    ])
    raw_dict = DatasetDict({"train": response_ds})
    merged = RagtruthTaskProcessor._merge_source_info(raw_dict, source_ds)

    train_merged = merged["train"]
    self.assertEqual(len(train_merged), 2)
    first = train_merged[0]
    self.assertEqual(first["source_id"], 101)
    self.assertEqual(first["base_content"], "Source text 101")
    self.assertEqual(first["response"], "Response 101")
    self.assertEqual(first["question"], "Q101")

  def test_preprocess_data_labels_mapping(self):
    raw_data = DatasetDict({
        "train": Dataset.from_list([
            # Faithful example (empty labels)
            {
                "base_content": "Passage 1 content.",
                "question": "What is passage 1?",
                "response": "Passage 1 is factual.",
                "labels": [],
            },
            # Hallucinated example (non-empty labels span list)
            {
                "base_content": "Passage 2 content.",
                "question": "What is passage 2?",
                "response": "Passage 2 contains ungrounded claims.",
                "labels": [
                    {"text": "ungrounded claims", "start": 19, "end": 36}
                ],
            },
            # Hallucination flag: hallucination_label = 0 (faithful)
            {
                "base_content": "Passage 3 content.",
                "user_query": "What is passage 3?",
                "response": "Passage 3 answer.",
                "hallucination_label": 0,
            },
            # Hallucination flag: hallucination_label = 1 (hallucinated)
            {
                "base_content": "Passage 4 content.",
                "user_query": "What is passage 4?",
                "response": "Passage 4 answer.",
                "hallucination_label": 1,
            },
        ]),
        "test": Dataset.from_list([
            {
                "base_content": "Test passage.",
                "question": "Test query?",
                "response": "Test response.",
                "class_hall": "No",
            }
        ]),
    })

    preprocessed = self.processor._preprocess_data(raw_data)
    train_split = preprocessed["train"]

    # Entry 0: faithful -> class_hall == "No", label == 1
    self.assertEqual(train_split[0]["class_hall"], "No")
    self.assertEqual(train_split[0]["label"], 1)
    self.assertTrue(train_split[0]["prompt"].endswith("Answer:\n"))

    # Entry 1: hallucinated -> class_hall == "Yes", label == 0
    self.assertEqual(train_split[1]["class_hall"], "Yes")
    self.assertEqual(train_split[1]["label"], 0)

    # Entry 2: hallucination_label = 0 -> faithful
    self.assertEqual(train_split[2]["class_hall"], "No")
    self.assertEqual(train_split[2]["label"], 1)

    # Entry 3: hallucination_label = 1 -> hallucinated
    self.assertEqual(train_split[3]["class_hall"], "Yes")
    self.assertEqual(train_split[3]["label"], 0)

    # Test split: class_hall == "No" preserved
    test_split = preprocessed["test"]
    self.assertEqual(test_split[0]["class_hall"], "No")
    self.assertEqual(test_split[0]["label"], 1)

  def test_get_target_subtask(self):
    proc_qa = RagtruthTaskProcessor(Mock(task_name="ragtruth-qa"))
    self.assertEqual(proc_qa._get_target_subtask(), "qa")

    proc_default = RagtruthTaskProcessor(Mock(task_name="ragtruth"))
    self.assertEqual(proc_default._get_target_subtask(), "qa")

    proc_sum = RagtruthTaskProcessor(
        Mock(task_name="ragtruth-summarization")
    )
    self.assertEqual(proc_sum._get_target_subtask(), "summarization")

    proc_data = RagtruthTaskProcessor(
        Mock(task_name="ragtruth-data2text")
    )
    with self.assertRaises(ValueError):
      proc_data._get_target_subtask()

  def test_preprocess_data_subtask_qa_filtering(self):
    args = Mock(task_name="ragtruth-qa", seed=12345)
    processor = RagtruthTaskProcessor(args)

    raw_data = DatasetDict({
        "train": Dataset.from_list([
            {
                "task_type": "qa",
                "base_content": "Context 1",
                "question": "Q1?",
                "response": "A1",
                "labels": [],
            },
            {
                "task_type": "text_summarization",
                "base_content": "News article",
                "response": "Article summary",
                "labels": [],
            },
            {
                "task_type": "data2txt",
                "base_content": "name: Eagle",
                "response": "Eagle is a place",
                "labels": [],
            },
        ]),
        "test": Dataset.from_list([
            {
                "task_type": "qa",
                "base_content": "Context 2",
                "question": "Q2?",
                "response": "A2",
                "labels": [],
            }
        ]),
    })

    processed = processor._preprocess_data(raw_data)
    # QA split should keep only the qa entry (1 entry in train, 1 in test)
    self.assertEqual(len(processed["train"]), 1)
    self.assertEqual(processed["train"][0]["user_query"], "Q1?")
    self.assertEqual(len(processed["test"]), 1)
    self.assertEqual(processed["test"][0]["user_query"], "Q2?")

  def test_preprocess_data_subtask_summarization_filtering(self):
    args = Mock(task_name="ragtruth-summarization", seed=12345)
    processor = RagtruthTaskProcessor(args)

    raw_data = DatasetDict({
        "train": Dataset.from_list([
            {
                "task_type": "qa",
                "base_content": "Context 1",
                "question": "Q1?",
                "response": "A1",
                "labels": [],
            },
            {
                "task_type": "text_summarization",
                "base_content": "News article",
                "response": "Article summary",
                "labels": [],
            },
            {
                "task_type": "data2txt",
                "base_content": "name: Eagle",
                "response": "Eagle is a place",
                "labels": [],
            },
        ]),
        "test": Dataset.from_list([
            {
                "task_type": "summary",
                "base_content": "Context news",
                "response": "Summary test",
                "labels": [],
            }
        ]),
    })

    processed = processor._preprocess_data(raw_data)
    # Summarization split should keep only the summarization entry
    self.assertEqual(len(processed["train"]), 1)
    self.assertEqual(
        processed["train"][0]["user_query"], "Summarize the above document."
    )
    self.assertEqual(len(processed["test"]), 1)
    self.assertEqual(
        processed["test"][0]["user_query"], "Summarize the above document."
    )


class TestRagtruthTaskProcessorSFTAndRM(unittest.TestCase):
  """Tests for SFT dataset filtering, column renaming, and organic RM formatting."""

  def setUp(self):
    self.mock_args = make_mock_args(seed=42, task_name="ragtruth")
    self.processor = RagtruthTaskProcessor(self.mock_args)

  def test_make_sft_data(self):
    # 12 sources x 3 responses. Sources 0-8 are faithful, 9-11 hallucinated.
    train_rows = [
        {
            "prompt": f"Context:\nC{i}\n\nQuestion: Q{i}\n\nAnswer:\n",
            "response": f"Faithful answer {i}-{j}",
            "class_hall": "No",
            "label": 1,
            "source_id": f"s{i}",
        }
        for i in range(9)
        for j in range(3)
    ] + [
        {
            "prompt": f"Context:\nC{i}\n\nQuestion: Q{i}\n\nAnswer:\n",
            "response": f"Hallucinated answer {i}-{j}",
            "class_hall": "Yes",
            "label": 0,
            "source_id": f"s{i}",
        }
        for i in range(9, 12)
        for j in range(3)
    ]
    data = DatasetDict({"train": Dataset.from_list(train_rows)})

    sft_data = self.processor._make_sft_data(data)
    self.assertIn("train", sft_data)
    self.assertIn("test", sft_data)

    # Column 'response' should be renamed to 'completion'.
    self.assertIn("completion", sft_data["train"].column_names)
    self.assertNotIn("response", sft_data["train"].column_names)

    train_sources = set(sft_data["train"]["source_id"])
    val_sources = set(sft_data["test"]["source_id"])

    # Only faithful sources may appear.
    self.assertTrue(train_sources.issubset({f"s{i}" for i in range(9)}))
    self.assertTrue(val_sources.issubset({f"s{i}" for i in range(9)}))

    # SFT must consume only its own block, never the whole training pool.
    self.assertLess(len(train_sources | val_sources), 9)

    # The SFT train/val boundary must be drawn over sources, not rows: a
    # single source fans out into several responses sharing one prompt, so a
    # row-level split would put the same prompt on both sides.
    self.assertEqual(train_sources & val_sources, set())
    self.assertEqual(
        set(sft_data["train"]["prompt"]) & set(sft_data["test"]["prompt"]),
        set(),
    )

  def test_make_sft_data_degenerate_single_source(self):
    """A one-source corpus cannot fill three disjoint blocks; must not crash."""
    single_data = DatasetDict({
        "train": Dataset.from_list([{
            "prompt": "Context:\nC\n\nQuestion: Q\n\nAnswer:\n",
            "response": "Single answer",
            "class_hall": "No",
            "label": 1,
            "source_id": "s0",
        }])
    })
    sft_data = self.processor._make_sft_data(single_data)
    self.assertIn("train", sft_data)
    self.assertIn("test", sft_data)
    # The lone source lands in exactly one block, so SFT gets 1 row or 0.
    self.assertLessEqual(len(sft_data["train"]) + len(sft_data["test"]), 1)

  def test_rm_prompt_formatting(self):
    entry = {
        "prompt": "Context:\nC\n\nQuestion: Q\n\nAnswer:\n",
        "response": "Paris is the capital of France.",
    }
    updated = RagtruthTaskProcessor._rm_prompt(entry)
    self.assertEqual(
        updated["prompt"],
        "Context:\nC\n\nQuestion: Q\n\nAnswer:\nParis is the capital of"
        " France.",
    )

    # Idempotent if already formatted
    updated_again = RagtruthTaskProcessor._rm_prompt(dict(updated))
    self.assertEqual(updated_again["prompt"], updated["prompt"])

  def test_make_organic_hallucinations_data(self):
    data = DatasetDict({
        "train": Dataset.from_list([{
            "prompt": "Context:\nC1\n\nQuestion: Q1\n\nAnswer:\n",
            "response": "R1",
            "class_hall": "No",
            "label": 1,
        }]),
        "test": Dataset.from_list([{
            "prompt": "Context:\nC2\n\nQuestion: Q2\n\nAnswer:\n",
            "response": "R2",
            "class_hall": "Yes",
            "label": 0,
        }]),
    })
    organic = self.processor._make_organic_hallucinations_data(data)
    self.assertEqual(
        organic["train"][0]["prompt"],
        "Context:\nC1\n\nQuestion: Q1\n\nAnswer:\nR1",
    )
    self.assertEqual(
        organic["test"][0]["prompt"],
        "Context:\nC2\n\nQuestion: Q2\n\nAnswer:\nR2",
    )


class TestRagtruthTaskProcessorSynthetic(unittest.TestCase):
  """Tests for ROUGE scoring and Schemas 1, 2, 3 of structured synthetic perturbations."""

  def setUp(self):
    self.mock_args = make_mock_args(
        seed=12345,
        num_synth_hallus=10,
        synth_struct_top_k=3,
        synth_struct_hallu_threshold=0.40,
        synth_struct_irrelevant_threshold=0.15,
        synth_struct_max_nonhall_per_entry=2,
        synth_struct_balance_ratio=1.30,
    )
    self.processor = RagtruthTaskProcessor(self.mock_args)
    self.rouge_metric = MockRougeMetric()

  def test_compute_sentence_rouge_scores(self):
    context_sentences = [
        "Jupiter is the largest planet in our solar system.",
        "Saturn is famous for its prominent ring system.",
        "Mercury is the closest planet to the Sun.",
    ]
    response_sentences = [
        "The largest planet in the solar system is Jupiter.",
        "It has a Great Red Spot.",
    ]

    scores = self.processor._compute_sentence_rouge_scores(
        sentences_context=context_sentences,
        sentences_response=response_sentences,
        rouge_metric=self.processor._get_rouge_scorer(),
    )

    self.assertEqual(len(scores), 3)
    # The first context sentence should have high overlap with response sentence 0
    self.assertGreater(scores[0]["max_f1"], 0.5)
    self.assertEqual(scores[0]["matched_response_idx"], 0)
    # The third sentence (Mercury) should have much lower overlap
    self.assertLess(scores[2]["max_f1"], scores[0]["max_f1"])

  def test_generate_hallucinations_top_k(self):
    query = "What is the largest planet?"
    context_sentences = [
        "Jupiter is the largest planet in the solar system.",
        "It is a gas giant with many moons.",
        "Saturn is the second largest planet.",
    ]
    response = "Jupiter is the largest planet in the solar system."
    scores = [
        {
            "context_idx": 0,
            "context_sentence": (
                "Jupiter is the largest planet in the solar system."
            ),
            "max_f1": 1.0,
            "matched_response_idx": 0,
        },
        {
            "context_idx": 1,
            "context_sentence": "It is a gas giant with many moons.",
            "max_f1": 0.2,
            "matched_response_idx": 0,
        },
        {
            "context_idx": 2,
            "context_sentence": "Saturn is the second largest planet.",
            "max_f1": 0.3,
            "matched_response_idx": 0,
        },
    ]

    hallus = self.processor._generate_hallucinations_top_k(
        query=query,
        sentences_context=context_sentences,
        response=response,
        scores=scores,
        top_k=2,
        hallu_threshold=0.40,
    )

    self.assertEqual(len(hallus), 1)  # Only sentence 0 >= 0.40
    cand = hallus[0]
    self.assertEqual(cand["class_hall"], "Yes")
    self.assertEqual(cand["label"], 0)
    self.assertEqual(cand["synthetic_strategy"], "erase_top_1_context")
    self.assertNotIn(
        "Jupiter is the largest planet in the solar system.", cand["context"]
    )
    self.assertIn("It is a gas giant with many moons.", cand["context"])

  def test_generate_nonhallucinations_irrelevant_context(self):
    query = "What is the largest planet?"
    context_sentences = [
        "Jupiter is the largest planet in the solar system.",
        "Pluto is considered a dwarf planet.",
        "Neptune is very cold and windy.",
    ]
    response = "Jupiter is the largest planet in the solar system."
    scores = [
        {
            "context_idx": 0,
            "context_sentence": (
                "Jupiter is the largest planet in the solar system."
            ),
            "max_f1": 1.0,
            "matched_response_idx": 0,
        },
        {
            "context_idx": 1,
            "context_sentence": "Pluto is considered a dwarf planet.",
            "max_f1": 0.05,
            "matched_response_idx": 0,
        },
        {
            "context_idx": 2,
            "context_sentence": "Neptune is very cold and windy.",
            "max_f1": 0.08,
            "matched_response_idx": 0,
        },
    ]

    nonhallus = (
        self.processor._generate_nonhallucinations_irrelevant_context(
            query=query,
            sentences_context=context_sentences,
            response=response,
            scores=scores,
            irrelevant_threshold=0.15,
            max_candidates=2,
        )
    )

    self.assertEqual(len(nonhallus), 2)
    for nh in nonhallus:
      self.assertEqual(nh["class_hall"], "No")
      self.assertEqual(nh["label"], 1)
      self.assertEqual(nh["synthetic_strategy"], "erase_irrelevant_context")
      # Crucial information (sentence 0) must remain intact
      self.assertIn(
          "Jupiter is the largest planet in the solar system.", nh["context"]
      )

  def test_generate_nonhallucinations_response_removal(self):
    query = "Describe Jupiter."
    context = (
        "Jupiter is the largest planet in our solar system. It is a gas giant."
    )
    sentences_response = [
        "Jupiter is the largest planet.",
        "It is a gas giant.",
        "It has a Great Red Spot.",
    ]

    resp_drops = self.processor._generate_nonhallucinations_response_removal(
        query=query,
        context=context,
        sentences_response=sentences_response,
        max_drops=2,
    )

    self.assertEqual(len(resp_drops), 2)
    for rd in resp_drops:
      self.assertEqual(rd["class_hall"], "No")
      self.assertEqual(rd["label"], 1)
      self.assertEqual(rd["synthetic_strategy"], "erase_response_sentence")

    # The first drop removes sentence 0
    self.assertNotIn("Jupiter is the largest planet.", resp_drops[0]["response"])
    self.assertIn("It is a gas giant.", resp_drops[0]["response"])

  def test_synthetic_edge_cases_single_sentences(self):
    query = "Question"
    context = "Single sentence context."
    response = "Single sentence response."
    scores = [{
        "context_idx": 0,
        "context_sentence": context,
        "max_f1": 0.8,
        "matched_response_idx": 0,
    }]

    # Top-k with single context sentence should not produce empty context
    hallus = self.processor._generate_hallucinations_top_k(
        query=query,
        sentences_context=[context],
        response=response,
        scores=scores,
        top_k=1,
        hallu_threshold=0.40,
    )
    self.assertEqual(len(hallus), 0)

    # Response removal with single response sentence should not produce empty response
    resp_drops = self.processor._generate_nonhallucinations_response_removal(
        query=query,
        context=context,
        sentences_response=[response],
        max_drops=2,
    )
    self.assertEqual(len(resp_drops), 0)

  def test_make_structured_hallucinations_data_balancing_and_columns(self):
    # The reward-model block is a fraction of the corpus and is then split
    # into two disjoint synthesis arms, so the fixture needs enough distinct
    # sources for both arms to be non-empty.
    templates = [
        {
            "user_query": "What is photon entanglement?",
            "context": (
                "Entanglement occurs when quantum particles interact."
                " The quantum state of each particle cannot be described"
                " independently. Albert Einstein called this spooky action at a"
                " distance. It has applications in quantum cryptography."
            ),
            "response": (
                "Quantum entanglement occurs when particles interact."
                " Their states cannot be described independently."
            ),
        },
        {
            "user_query": "Explain photosynthesis.",
            "context": (
                "Photosynthesis is used by plants to convert light into"
                " chemical energy. Chlorophyll is the green pigment responsible"
                " for light absorption. Oxygen is released as a byproduct."
                " Desert plants use CAM photosynthesis to conserve water."
            ),
            "response": (
                "Photosynthesis converts light into chemical energy using"
                " chlorophyll. Oxygen is released as a byproduct."
            ),
        },
    ]
    train_entries = []
    for i in range(12):
      template = templates[i % len(templates)]
      train_entries.append({
          "user_query": template["user_query"],
          "context": template["context"],
          "response": template["response"],
          "prompt": (
              f"Context:\n...\n\nQuestion: {template['user_query']}"
              f" (variant {i})\n\nAnswer:\n"
          ),
          "class_hall": "No",
          "label": 1,
          "source_id": f"s{i}",
      })
    test_entries = [{
        "user_query": "Test query",
        "context": "Test context",
        "response": "Test response",
        "prompt": "Context:\n...\n\nQuestion: Test\n\nAnswer:\n",
        "class_hall": "Yes",
        "label": 0,
        "source_id": "t0",
    }]

    raw_data = DatasetDict({
        "train": Dataset.from_list(train_entries),
        "test": Dataset.from_list(test_entries),
    })

    struct_data = self.processor._make_structured_hallucinations_data(raw_data)
    train_split = struct_data["train"]
    test_split = struct_data["test"]

    # Verify column alignment across test split
    for col in (
        "synthetic_strategy",
        "erased_context",
        "erased_response",
        "rouge1_score",
    ):
      self.assertIn(col, test_split.column_names)
      self.assertIn(col, train_split.column_names)

    # Verify class balance
    hallus = [row for row in train_split if row["class_hall"] == "Yes"]
    non_hallus = [row for row in train_split if row["class_hall"] == "No"]
    self.assertGreater(len(hallus), 0)
    self.assertGreater(len(non_hallus), 0)
    ratio = max(len(hallus), len(non_hallus)) / min(
        len(hallus), len(non_hallus)
    )
    # Balance ratio threshold (with rounding buffer)
    self.assertLessEqual(ratio, self.mock_args.synth_struct_balance_ratio + 0.5)

    # Provenance must survive synthesis, otherwise leakage is unauditable.
    self.assertIn("source_id", train_split.column_names)
    self.assertNotIn(None, list(train_split["source_id"]))

    # The two synthesis arms must be disjoint by source. If one context were
    # emitted both perturbed and clean, the reward model could learn to key on
    # the context itself rather than on the faithfulness of the response.
    hallu_sources = {row["source_id"] for row in hallus}
    faithful_sources = {row["source_id"] for row in non_hallus}
    self.assertEqual(hallu_sources & faithful_sources, set())

  def test_split_into_sentences_newline_structured(self):
    """Test that newline-separated key-value tables (Data2text) are split by lines with newline delimiter."""
    data2text_ctx = "name: The Eagle\ntype: pub\nfood: English\nprice: moderate"
    lines, delimiter = RagtruthTaskProcessor._split_into_sentences(data2text_ctx)
    self.assertEqual(len(lines), 4)
    self.assertEqual(delimiter, "\n")
    self.assertEqual(lines[0], "name: The Eagle")
    self.assertEqual(lines[1], "type: pub")

    # Standard prose should split with space delimiter
    prose = "Albert Einstein was a physicist. He developed general relativity."
    sents, delimiter = RagtruthTaskProcessor._split_into_sentences(prose)
    self.assertEqual(len(sents), 2)
    self.assertEqual(delimiter, " ")

  def test_data2text_structured_hallucination_preserves_newlines(self):
    """Verify Schema 1 on Data2text context preserves newline structure instead of collapsing to single line."""
    query = "Describe the establishment."
    context = "name: The Eagle\ntype: pub\nfood: English\nprice: moderate"
    response = "The Eagle is a pub serving English food."
    scores = [
        {"context_idx": 0, "context_sentence": "name: The Eagle", "max_f1": 0.9, "matched_response_idx": 0},
        {"context_idx": 1, "context_sentence": "type: pub", "max_f1": 0.8, "matched_response_idx": 0},
        {"context_idx": 2, "context_sentence": "food: English", "max_f1": 0.8, "matched_response_idx": 0},
        {"context_idx": 3, "context_sentence": "price: moderate", "max_f1": 0.05, "matched_response_idx": 0},
    ]
    hallus = self.processor._generate_hallucinations_top_k(
        query=query,
        sentences_context=["name: The Eagle", "type: pub", "food: English", "price: moderate"],
        response=response,
        scores=scores,
        top_k=1,
        hallu_threshold=0.40,
        delimiter="\n",
    )
    self.assertEqual(len(hallus), 1)
    # The top candidate (name: The Eagle) was removed, and remaining lines are joined by newline
    self.assertNotIn("name: The Eagle", hallus[0]["context"])
    self.assertIn("type: pub\nfood: English\nprice: moderate", hallus[0]["context"])

  def test_make_llm_hallucinations_data_rest_mapped(self):
    """Verify that unperturbed faithful samples in 'rest' have rm_prompt properly applied."""
    raw_data = DatasetDict({
        "train": Dataset.from_list([
            {
                "prompt": f"Context:\nC{i}\n\nQuestion: Q{i}\n\nAnswer:\n",
                "response": f"Faithful Answer {i}",
                "class_hall": "No",
                "label": 1,
                "source_id": f"s{i}",
            }
            for i in range(12)
        ]),
        "test": Dataset.from_list([]),
    })
    organic_rm = DatasetDict({
        "train": Dataset.from_list([
            {
                "prompt": "Context:\nC_org\n\nQuestion: Q\n\nAnswer:\nOrg Hallu",
                "response": "Org Hallu",
                "class_hall": "Yes",
                "label": 0,
                "source_id": "org0",
            }
        ]),
        "test": Dataset.from_list([]),
    })
    self.mock_args.num_synth_hallus = 1
    llm_data = self.processor._make_llm_hallucinations_data(raw_data, organic_rm)
    train_split = llm_data["train"]
    self.assertGreater(len(train_split), 0)

    # Only the reward-model block may be consumed, never the whole pool.
    self.assertLess(len(train_split), 12)

    # Both perturbed and rest samples MUST have their response appended to prompt
    for row in train_split:
        self.assertTrue(
            row["prompt"].endswith(row["response"]),
            f"Prompt '{row['prompt']}' should end with response '{row['response']}'",
        )

    # Perturbed and clean arms must not share a source.
    hallu_sources = {
        row["source_id"] for row in train_split if row["class_hall"] == "Yes"
    }
    faithful_sources = {
        row["source_id"] for row in train_split if row["class_hall"] == "No"
    }
    self.assertEqual(hallu_sources & faithful_sources, set())


class TestRagtruthTaskProcessorDownstream(unittest.TestCase):
  """Tests for PERL, autorater, evaluation splits, and prompt callables."""

  def setUp(self):
    self.mock_args = make_mock_args(seed=12345, task_name="ragtruth")
    self.processor = RagtruthTaskProcessor(self.mock_args)

  def test_make_perl_data(self):
    # 12 sources x 2 faithful responses each, so prompts repeat within a
    # source and the PE-RL block must dedupe them down to one per source.
    train_rows = [
        {
            "prompt": f"Prompt {i}",
            "response": f"Resp {i}-{j}",
            "user_query": f"Q{i}",
            "context": f"C{i}",
            "class_hall": "No",
            "label": 1,
            "source_id": f"s{i}",
        }
        for i in range(12)
        for j in range(2)
    ]
    data = DatasetDict({
        "train": Dataset.from_list(train_rows),
        "test": Dataset.from_list([
            {
                "prompt": f"Test Prompt {i // 2}",
                "response": f"R {i}",
                "user_query": f"Q {i // 2}",
                "context": f"C {i // 2}",
                "source_id": f"t{i // 2}",
            }
            for i in range(10)
        ]),
    })

    sft_data = self.processor._make_sft_data(data)
    perl_data = self.processor._make_perl_data(data, sft_data, seed=42)

    # PE-RL must draw rollout prompts from its own block. Reusing SFT prompts
    # would mean optimizing the policy on exactly what it memorized.
    perl_prompts = set(perl_data["train"]["prompt"])
    self.assertTrue(perl_prompts)
    self.assertEqual(perl_prompts & set(sft_data["train"]["prompt"]), set())
    self.assertEqual(perl_prompts & set(sft_data["test"]["prompt"]), set())

    # One prompt per source: duplicates collapse.
    self.assertEqual(len(perl_data["train"]), len(perl_prompts))

    # Test prompts dedupe from 10 rows to 5 unique prompts.
    self.assertEqual(len(perl_data["test"]), 5)

    # Columns must retain prompt, user_query, context, and completion/response
    for split in ("train", "test"):
        self.assertIn("prompt", perl_data[split].column_names)
        self.assertIn("user_query", perl_data[split].column_names)
        self.assertIn("context", perl_data[split].column_names)

  def test_make_autorater_and_evaluation_data(self):
    test_ds = Dataset.from_list([{
        "prompt": "Context:\nC\n\nQuestion: Q\n\nAnswer:\n",
        "response": "Answer text",
        "class_hall": "No",
        "label": 1,
    }])
    data = DatasetDict({"test": test_ds})

    autorater = self.processor._make_autorater_data(data)
    self.assertEqual(
        autorater[0]["prompt"],
        "Context:\nC\n\nQuestion: Q\n\nAnswer:\nAnswer text",
    )

    eval_data = self.processor._make_evaluation_data(data)
    self.assertEqual(len(eval_data["test"]), 1)

  def test_get_formatting_prompts_and_response_template(self):
    formatting_fn, resp_template = (
        RagtruthTaskProcessor.get_formatting_prompts_and_response_template(
            eos_token="<eos>"
        )
    )
    self.assertEqual(resp_template, "\nAnswer:\n")

    batch = {
        "prompt": [
            "Context:\nC1\n\nQuestion: Q1\n\nAnswer:\n",
            "Context:\nC2\n\nQuestion: Q2\n\nAnswer:\n",
        ],
        "completion": ["Answer 1.", "Answer 2."],
    }
    formatted_texts = formatting_fn(batch)
    self.assertEqual(len(formatted_texts), 2)
    self.assertEqual(
        formatted_texts[0],
        "Context:\nC1\n\nQuestion: Q1\n\nAnswer:\nAnswer 1.<eos>",
    )
    self.assertEqual(
        formatted_texts[1],
        "Context:\nC2\n\nQuestion: Q2\n\nAnswer:\nAnswer 2.<eos>",
    )

    # Test single-item string format
    single_entry = {
        "prompt": "Context:\nC3\n\nQuestion: Q3\n\nAnswer:\n",
        "completion": "Answer 3.",
    }
    formatted_single = formatting_fn(single_entry)
    self.assertEqual(len(formatted_single), 1)
    self.assertEqual(
        formatted_single[0],
        "Context:\nC3\n\nQuestion: Q3\n\nAnswer:\nAnswer 3.<eos>",
    )

  def test_get_evaluator_prompt(self):
    eval_fn = RagtruthTaskProcessor.get_evaluator_prompt()

    entry = {
        "context": "Albert Einstein published the theory of relativity.",
        "user_query": "Who published the theory of relativity?",
        "response": "Albert Einstein published the theory of relativity.",
        "completion": "Isaac Newton published relativity.",
        "class_hall": "No",
    }

    # autorater evaluation mode (use_true_label=True)
    out_true = eval_fn(dict(entry), use_true_label=True)
    self.assertIn("evaluator_prompt", out_true)
    prompt_true = out_true["evaluator_prompt"]
    self.assertIn("Albert Einstein", prompt_true)
    self.assertIn("Does the answer contain ANY information", prompt_true)

    # scoring generated completions mode (use_true_label=False)
    out_gen = eval_fn(dict(entry), use_true_label=False)
    prompt_gen = out_gen["evaluator_prompt"]
    self.assertIn("Isaac Newton", prompt_gen)

    # Fewshot examples
    fewshots = Dataset.from_list([{
        "context": "Context example",
        "user_query": "Query example",
        "response": "Response example",
        "class_hall": "Yes",
    }])
    out_fewshot = eval_fn(dict(entry), fewshot_examples=fewshots)
    prompt_fs = out_fewshot["evaluator_prompt"]
    self.assertIn("--- EXAMPLES ---", prompt_fs)
    self.assertIn("Context example", prompt_fs)
    self.assertIn("Yes", prompt_fs)


class TestRagtruthTaskProcessorLeakageControls(unittest.TestCase):
  """Tests for quality filtering and the disjoint source-block partition."""

  def setUp(self):
    self.mock_args = make_mock_args(seed=12345, task_name="ragtruth-qa")
    self.processor = RagtruthTaskProcessor(self.mock_args)

  def _qa_rows(self, n):
    return [
        {
            "task_type": "qa",
            "base_content": f"Context {i}",
            "question": f"Q{i}?",
            "response": f"A{i}",
            "labels": [],
            "source_id": f"s{i}",
        }
        for i in range(n)
    ]

  def test_preprocess_drops_bad_quality_rows(self):
    """'incorrect_refusal' and 'truncated' rows must not reach any split.

    RAGTruth labels these rows non-hallucinated, so leaving them in would push
    refusals and cut-off text into the SFT targets as if they were good
    answers.
    """
    rows = self._qa_rows(3)
    rows[0]["quality"] = "good"
    rows[1]["quality"] = "incorrect_refusal"
    rows[2]["quality"] = "truncated"
    raw_data = DatasetDict({
        "train": Dataset.from_list(rows),
        "test": Dataset.from_list([]),
    })

    processed = self.processor._preprocess_data(raw_data)
    self.assertEqual(len(processed["train"]), 1)
    self.assertEqual(processed["train"][0]["user_query"], "Q0?")

  def test_preprocess_keeps_bad_quality_when_disabled(self):
    rows = self._qa_rows(3)
    rows[0]["quality"] = "good"
    rows[1]["quality"] = "incorrect_refusal"
    rows[2]["quality"] = "truncated"
    args = make_mock_args(
        seed=12345, task_name="ragtruth-qa", drop_bad_quality=False
    )
    processor = RagtruthTaskProcessor(args)
    raw_data = DatasetDict({
        "train": Dataset.from_list(rows),
        "test": Dataset.from_list([]),
    })

    processed = processor._preprocess_data(raw_data)
    self.assertEqual(len(processed["train"]), 3)

  def test_preprocess_tolerates_missing_quality_column(self):
    raw_data = DatasetDict({
        "train": Dataset.from_list(self._qa_rows(3)),
        "test": Dataset.from_list([]),
    })
    processed = self.processor._preprocess_data(raw_data)
    self.assertEqual(len(processed["train"]), 3)

  def test_source_blocks_partition_all_sources(self):
    train_ds = Dataset.from_list([
        {"source_id": f"s{i}", "prompt": f"P{i}", "response": f"R{i}"}
        for i in range(100)
    ])
    mapping = self.processor._assign_source_blocks(train_ds)

    self.assertEqual(len(mapping), 100)
    self.assertEqual(set(mapping.values()), {"sft", "perl", "rm"})

    blocks = {"sft": set(), "perl": set(), "rm": set()}
    for key, block in mapping.items():
      blocks[block].add(key)

    # Disjoint by construction, and together they cover every source.
    self.assertEqual(blocks["sft"] & blocks["perl"], set())
    self.assertEqual(blocks["sft"] & blocks["rm"], set())
    self.assertEqual(blocks["perl"] & blocks["rm"], set())
    self.assertEqual(
        len(blocks["sft"]) + len(blocks["perl"]) + len(blocks["rm"]), 100
    )

    # Sizes follow the requested fractions (40 / 25 / 35).
    self.assertEqual(len(blocks["sft"]), 40)
    self.assertEqual(len(blocks["perl"]), 25)
    self.assertEqual(len(blocks["rm"]), 35)

  def test_source_blocks_are_deterministic_across_processors(self):
    """Builders run at different times must see the same partition.

    The reward-model splits are built before the SFT split exists, so the two
    can only stay disjoint if the partition is a pure function of the seed and
    the set of source keys.
    """
    rows = [
        {"source_id": f"s{i}", "prompt": f"P{i}", "response": f"R{i}"}
        for i in range(50)
    ]
    first = RagtruthTaskProcessor(
        make_mock_args(seed=777)
    )._assign_source_blocks(Dataset.from_list(rows))
    # Shuffled row order must not change the outcome.
    second = RagtruthTaskProcessor(
        make_mock_args(seed=777)
    )._assign_source_blocks(Dataset.from_list(list(reversed(rows))))
    self.assertEqual(first, second)

    different_seed = RagtruthTaskProcessor(
        make_mock_args(seed=778)
    )._assign_source_blocks(Dataset.from_list(rows))
    self.assertNotEqual(first, different_seed)

  def test_all_responses_of_a_source_land_in_one_block(self):
    """The whole point of the source-level split.

    Six responses share one prompt; if they scattered across blocks the same
    prompt would appear in SFT, PE-RL and the reward model at once.
    """
    train_ds = Dataset.from_list([
        {"source_id": f"s{i}", "prompt": f"P{i}", "response": f"R{i}-{j}"}
        for i in range(30)
        for j in range(6)
    ])
    mapping = self.processor._assign_source_blocks(train_ds)

    seen = {}
    for block in ("sft", "perl", "rm"):
      for row in self.processor._select_block(train_ds, block):
        sid = row["source_id"]
        self.assertEqual(seen.setdefault(sid, block), block)
    self.assertEqual(len(seen), 30)
    self.assertEqual(len(mapping), 30)

  def test_block_fractions_reject_impossible_split(self):
    """A split leaving nothing for the reward model must fail loudly."""
    for bad in ((0.8, 0.3), (0.0, 0.25), (0.4, -0.1), (0.75, 0.25)):
      processor = RagtruthTaskProcessor(
          make_mock_args(split_frac_sft=bad[0], split_frac_perl=bad[1])
      )
      with self.assertRaises(ValueError):
        processor._block_fractions()

  def test_split_synthesis_arms_are_disjoint(self):
    ds = Dataset.from_list([
        {"source_id": f"s{i}", "prompt": f"P{i}", "response": f"R{i}"}
        for i in range(40)
    ])
    perturbed, clean = self.processor._split_synthesis_arms(ds, seed=12345)
    self.assertEqual(perturbed & clean, set())
    self.assertEqual(len(perturbed) + len(clean), 40)
    self.assertEqual(len(perturbed), 20)

  def test_cap_responses_per_source(self):
    ds = Dataset.from_list([
        {"source_id": f"s{i}", "prompt": f"P{i}", "response": f"R{i}-{j}"}
        for i in range(10)
        for j in range(6)
    ])
    capped = self.processor._cap_responses_per_source(ds, seed=1)
    counts = {}
    for row in capped:
      counts[row["source_id"]] = counts.get(row["source_id"], 0) + 1
    self.assertEqual(len(counts), 10)
    self.assertTrue(all(c <= 2 for c in counts.values()))


if __name__ == "__main__":
  unittest.main()
