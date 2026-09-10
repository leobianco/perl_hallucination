"""Unit tests for BoschTaskProcessor synthetic hallucination schemas.

Tests:
1. Idea 1: Irrelevant context sentence removal (label 1 / non-hallucinated).
2. Idea 2: Top-k context sentence removal (label 0 / hallucinated).
3. Idea 3: Response sentence removal (label 1 / non-hallucinated).
4. Edge cases: Single-sentence context, single-sentence response, thresholds.
5. Rough balancing between hallucinated and non-hallucinated classes.
6. Schema uniformity between train and test splits.
"""

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
        keys = list(data[0].keys()) if data else []
        self._data = {k: [d.get(k) for d in data] for k in keys}
      elif isinstance(data, dict):
        self._data = dict(data)
      else:
        self._data = {}
      self.column_names = list(self._data.keys())

    def __getitem__(self, key):
      return self._data[key]

    def __len__(self):
      first_col = next(iter(self._data.values())) if self._data else []
      return len(first_col)

    def to_dict(self):
      return dict(self._data)

    def add_column(self, column_name, values):
      new_data = dict(self._data)
      new_data[column_name] = values
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
      return _MockDataset(lst)

  class DatasetDict(dict):

    def map(self, fn, **kwargs):
      return DatasetDict({k: v.map(fn, **kwargs) for k, v in self.items()})

  datasets_mock = types.ModuleType("datasets")
  datasets_mock.Dataset = Dataset
  datasets_mock.DatasetDict = DatasetDict
  datasets_mock.concatenate_datasets = lambda d_list: _MockDataset(
      {
          k: [item for d in d_list for item in d._data[k]]
          for k in d_list[0]._data
      }
  )
  datasets_mock.load_dataset = MagicMock()
  sys.modules["datasets"] = datasets_mock


from src.task_processors.bosch_task_processor import (
    _PurePythonRougeScorer,
    BoschTaskProcessor,
)


class TestBoschTaskProcessorSynthetic(unittest.TestCase):

  def setUp(self):
    self.mock_args = Mock()
    self.mock_args.seed = 12345
    self.mock_args.num_synth_hallus = 10
    self.mock_args.synth_struct_top_k = 3
    self.mock_args.synth_struct_hallu_threshold = 0.40
    self.mock_args.synth_struct_irrelevant_threshold = 0.15
    self.mock_args.synth_struct_max_nonhall_per_entry = 2
    self.mock_args.synth_struct_balance_ratio = 1.30
    self.processor = BoschTaskProcessor(self.mock_args)
    self.rouge_metric = MockRougeMetric()

  def test_format_prompt(self):
    prompt = BoschTaskProcessor._format_prompt(
        "How to check oil?", "Oil dipstick is yellow."
    )
    self.assertIn("User question:\nHow to check oil?", prompt)
    self.assertIn("Manual information:\nOil dipstick is yellow.", prompt)
    self.assertTrue(prompt.endswith("Answer to user's question:\n"))

  def test_compute_sentence_rouge_scores(self):
    context_sentences = [
        "Check the engine oil level using the yellow dipstick.",
        "The tire pressure should be checked every month.",
        "Bananas grow in tropical climates.",
    ]
    response_sentences = [
        "To check the engine oil, pull the yellow dipstick and wipe it.",
        "Inspect the oil level between the markings.",
    ]
    scores = BoschTaskProcessor._compute_sentence_rouge_scores(
        context_sentences, response_sentences, self.rouge_metric
    )
    self.assertEqual(len(scores), 3)
    # The oil sentence should have high ROUGE
    self.assertGreater(scores[0]["max_rouge"], 0.3)
    # Bananas should have zero ROUGE
    self.assertEqual(scores[2]["max_rouge"], 0.0)

  def test_idea_1_irrelevant_context_removal(self):
    question = "How to inspect oil?"
    context_sentences = [
        "Check the engine oil level using the yellow dipstick.",
        "Keep the vehicle parked on level ground.",
        "Jupiter is the largest planet in our solar system.",
    ]
    response = "Check the engine oil with the dipstick on level ground."
    scores = BoschTaskProcessor._compute_sentence_rouge_scores(
        context_sentences, [response], self.rouge_metric
    )
    samples = BoschTaskProcessor._generate_nonhallucinations_irrelevant_context(
        question=question,
        sentences_context=context_sentences,
        response=response,
        scores=scores,
        irrelevant_threshold=0.15,
        max_candidates=1,
    )
    self.assertEqual(len(samples), 1)
    sample = samples[0]
    self.assertEqual(sample["label"], 1)
    self.assertEqual(sample["class_hall"], "No")
    self.assertEqual(sample["synthetic_strategy"], "context_erasure_irrelevant")
    self.assertIn("Jupiter", sample["erased_context"])
    self.assertNotIn("Jupiter", sample["Context"])
    self.assertIn("yellow dipstick", sample["Context"])
    self.assertIn("level ground", sample["Context"])

  def test_idea_2_top_k_hallucinations(self):
    question = "How do I check oil and tires?"
    context_sentences = [
        "Check the engine oil using the yellow dipstick.",
        "Tire pressure should be maintained at 32 psi.",
        "Rotate wheels every six months for even tread.",
    ]
    response = "Check engine oil with yellow dipstick and keep tire pressure at 32 psi."
    scores = BoschTaskProcessor._compute_sentence_rouge_scores(
        context_sentences, response.split(" and "), self.rouge_metric
    )
    samples = BoschTaskProcessor._generate_hallucinations_top_k(
        question=question,
        sentences_context=context_sentences,
        response=response,
        scores=scores,
        top_k=2,
        hallu_threshold=0.20,
    )
    self.assertGreaterEqual(len(samples), 1)
    for sample in samples:
      self.assertEqual(sample["label"], 0)
      self.assertEqual(sample["class_hall"], "Yes")
      self.assertTrue(sample["synthetic_strategy"].startswith("context_erasure_top_"))
      self.assertNotEqual(sample["erased_context"], "")
      self.assertNotIn(sample["erased_context"], sample["Context"])

    if len(samples) >= 2:
      self.assertNotEqual(samples[0]["erased_context"], samples[1]["erased_context"])
      self.assertIn(samples[0]["erased_context"], samples[1]["Context"])

  def test_idea_3_response_sentence_removal(self):
    question = "What are the maintenance steps?"
    context = "Check oil level regularly. Replace engine air filter every year."
    response_sentences = [
        "First check the oil level regularly.",
        "Second replace the engine air filter every year.",
    ]
    samples = BoschTaskProcessor._generate_nonhallucinations_response_removal(
        question=question,
        context=context,
        sentences_response=response_sentences,
        max_drops=1,
    )
    self.assertEqual(len(samples), 1)
    sample = samples[0]
    self.assertEqual(sample["label"], 1)
    self.assertEqual(sample["class_hall"], "No")
    self.assertEqual(sample["synthetic_strategy"], "response_sentence_removal")
    self.assertEqual(sample["erased_response"], response_sentences[0])
    self.assertEqual(sample["response"], response_sentences[1])
    self.assertEqual(sample["Context"], context)

  def test_single_sentence_context_and_response_edge_cases(self):
    question = "How to open fuel flap?"
    context_sentences = ["Press the fuel flap button inside the cabin."]
    response = "Press the fuel flap button."
    scores = BoschTaskProcessor._compute_sentence_rouge_scores(
        context_sentences, [response], self.rouge_metric
    )
    hallu_samples = BoschTaskProcessor._generate_hallucinations_top_k(
        question=question,
        sentences_context=context_sentences,
        response=response,
        scores=scores,
        top_k=2,
    )
    self.assertEqual(len(hallu_samples), 0)

    nonhall_ctx = BoschTaskProcessor._generate_nonhallucinations_irrelevant_context(
        question=question,
        sentences_context=context_sentences,
        response=response,
        scores=scores,
    )
    self.assertEqual(len(nonhall_ctx), 0)

    resp_drops = BoschTaskProcessor._generate_nonhallucinations_response_removal(
        question=question,
        context=context_sentences[0],
        sentences_response=[response],
    )
    self.assertEqual(len(resp_drops), 0)

  def test_rough_balancing_and_datasetdict_assembly(self):
    validation_rows = [
        {
            "Question": "How do I check oil?",
            "Context": (
                "Check engine oil using the yellow dipstick. Keep car on level ground."
                " Bananas are yellow fruits."
            ),
            "response": "Check engine oil using the yellow dipstick on level ground.",
            "class_hall": "No",
            "label": 1,
            "prompt": "prompt1",
        },
        {
            "Question": "How to adjust headlights?",
            "Context": (
                "Turn the dial to position 0 for unladen vehicle."
                " Position 1 is for passenger load. Do not look into laser."
            ),
            "response": "Turn the dial to position 0 or position 1 depending on load.",
            "class_hall": "No",
            "label": 1,
            "prompt": "prompt2",
        },
    ]
    mock_data = DatasetDict({
        "validation": Dataset.from_list(validation_rows),
        "test": Dataset.from_list(validation_rows),
        "train": Dataset.from_list(validation_rows),
    })

    result = self.processor._make_structured_hallucinations_data(mock_data)
    self.assertIn("train", result)
    self.assertIn("test", result)

    train_data = result["train"]
    test_data = result["test"]

    for col in (
        "Question",
        "Context",
        "response",
        "prompt",
        "class_hall",
        "label",
        "synthetic_strategy",
        "erased_context",
        "erased_response",
        "rouge1_score",
    ):
      self.assertIn(col, train_data.column_names)
      self.assertIn(col, test_data.column_names)

    labels = train_data["label"]
    n_0 = sum(1 for l in labels if l == 0)
    n_1 = sum(1 for l in labels if l == 1)
    self.assertGreater(n_0, 0)
    self.assertGreater(n_1, 0)
    ratio = max(n_0, n_1) / min(n_0, n_1)
    self.assertLessEqual(ratio, 1.5)

  def test_idea_1_all_relevant_sentences_returns_empty(self):
    # When every context sentence is relevant, no candidate should be removed
    question = "How to replace battery?"
    context_sentences = [
        "Disconnect the negative terminal first.",
        "Remove the battery bracket completely.",
    ]
    response = "Disconnect negative terminal and remove battery bracket."
    scores = BoschTaskProcessor._compute_sentence_rouge_scores(
        context_sentences, [response], self.rouge_metric
    )
    samples = BoschTaskProcessor._generate_nonhallucinations_irrelevant_context(
        question=question,
        sentences_context=context_sentences,
        response=response,
        scores=scores,
        irrelevant_threshold=0.05,  # Very low threshold
    )
    self.assertEqual(len(samples), 0)

  def test_idea_2_all_below_threshold_returns_empty(self):
    # When no context sentence matches the hallucination threshold, no sample is produced
    question = "What is the gear ratio?"
    context_sentences = [
        "Apples and oranges are grown in different orchards.",
        "The weather in Lisbon is sunny in June.",
    ]
    response = "The gear ratio is four to one."
    scores = BoschTaskProcessor._compute_sentence_rouge_scores(
        context_sentences, [response], self.rouge_metric
    )
    samples = BoschTaskProcessor._generate_hallucinations_top_k(
        question=question,
        sentences_context=context_sentences,
        response=response,
        scores=scores,
        top_k=2,
        hallu_threshold=0.50,
    )
    self.assertEqual(len(samples), 0)

  def test_top_k_parameter_bounds(self):
    question = "Maintenance question?"
    context_sentences = [
        "Check engine oil regularly.",
        "Check tire pressure regularly.",
        "Check brake fluid regularly.",
        "Check coolant level regularly.",
    ]
    response = (
        "Check engine oil regularly and tire pressure regularly and brake fluid"
        " regularly."
    )
    scores = BoschTaskProcessor._compute_sentence_rouge_scores(
        context_sentences, response.split(" and "), self.rouge_metric
    )
    # top_k = 1 should produce at most 1
    samples_k1 = BoschTaskProcessor._generate_hallucinations_top_k(
        question=question,
        sentences_context=context_sentences,
        response=response,
        scores=scores,
        top_k=1,
        hallu_threshold=0.20,
    )
    self.assertEqual(len(samples_k1), 1)

    # top_k = 3 should produce 3 distinct hallucinations
    samples_k3 = BoschTaskProcessor._generate_hallucinations_top_k(
        question=question,
        sentences_context=context_sentences,
        response=response,
        scores=scores,
        top_k=3,
        hallu_threshold=0.20,
    )
    self.assertEqual(len(samples_k3), 3)

  def test_multiple_response_drops(self):
    question = "Steps to follow?"
    context = "Do step A. Do step B. Do step C."
    response_sentences = ["First do step A.", "Next do step B.", "Finally do step C."]
    samples = BoschTaskProcessor._generate_nonhallucinations_response_removal(
        question=question,
        context=context,
        sentences_response=response_sentences,
        max_drops=2,
    )
    self.assertEqual(len(samples), 2)
    self.assertEqual(samples[0]["erased_response"], response_sentences[0])
    self.assertEqual(samples[1]["erased_response"], response_sentences[1])

  def test_seed_determinism(self):
    validation_rows = [
        {
            "Question": "Q1?",
            "Context": "Sentence one is relevant. Sentence two is irrelevant info about astronomy.",
            "response": "Sentence one is relevant.",
            "class_hall": "No",
            "label": 1,
            "prompt": "p1",
        },
        {
            "Question": "Q2?",
            "Context": "Check oil level here. Keep car level. Check battery voltage.",
            "response": "Check oil level here and keep car level.",
            "class_hall": "No",
            "label": 1,
            "prompt": "p2",
        },
    ]
    mock_data = DatasetDict({
        "validation": Dataset.from_list(validation_rows),
        "test": Dataset.from_list(validation_rows),
        "train": Dataset.from_list(validation_rows),
    })

    res1 = self.processor._make_structured_hallucinations_data(mock_data)
    res2 = self.processor._make_structured_hallucinations_data(mock_data)

    self.assertEqual(len(res1["train"]), len(res2["train"]))
    self.assertEqual(res1["train"]["prompt"], res2["train"]["prompt"])
    self.assertEqual(res1["train"]["label"], res2["train"]["label"])

  def test_num_synth_hallus_all_with_negative_one(self):
    validation_rows = [
        {
            "Question": f"Q{i}?",
            "Context": f"Sentence {i} is relevant. Sentence {i}b is irrelevant astronomy text.",
            "response": f"Sentence {i} is relevant.",
            "class_hall": "No",
            "label": 1,
            "prompt": f"p{i}",
        }
        for i in range(5)
    ]
    mock_data = DatasetDict({
        "validation": Dataset.from_list(validation_rows),
        "test": Dataset.from_list(validation_rows[:2]),
        "train": Dataset.from_list(validation_rows[:2]),
    })

    self.mock_args.num_synth_hallus = -1
    res = self.processor._make_structured_hallucinations_data(mock_data)
    # Verify all 5 entries were processed through synthetic generation
    train_prompts = res["train"]["prompt"]
    self.assertGreater(len(train_prompts), 5)

  def test_split_nonhallucinated_negative_one(self):
    rows = [
        {"class_hall": "No", "text": f"item_{i}"}
        for i in range(4)
    ]
    split_dataset = Dataset.from_list(rows)
    to_become, rest = BoschTaskProcessor._split_nonhallucinated_for_synthetic(
        split_dataset, num_synth_hallus=-1, seed=12345
    )
    self.assertEqual(len(to_become), 4)
    self.assertEqual(len(rest), 0)

  def test_pure_python_rouge_scorer_exact_match(self):
    scorer = _PurePythonRougeScorer()
    res = scorer.score(
        "Check engine oil level regularly.", "Check engine oil level regularly."
    )
    score = res["rouge1"]
    self.assertAlmostEqual(score.precision, 1.0)
    self.assertAlmostEqual(score.recall, 1.0)
    self.assertAlmostEqual(score.fmeasure, 1.0)

  def test_pure_python_rouge_scorer_disjoint(self):
    scorer = _PurePythonRougeScorer()
    res = scorer.score("Apples and oranges.", "Jupiter orbit spacecraft.")
    score = res["rouge1"]
    self.assertEqual(score.fmeasure, 0.0)

  def test_pure_python_rouge_scorer_partial_overlap(self):
    scorer = _PurePythonRougeScorer()
    res = scorer.score(
        "Check engine oil level.", "Check engine oil dipstick markings."
    )
    score = res["rouge1"]
    self.assertGreater(score.fmeasure, 0.4)
    self.assertLess(score.fmeasure, 1.0)

  def test_compute_sentence_rouge_scores_with_in_memory_scorer(self):
    scorer = _PurePythonRougeScorer()
    context_sentences = [
        "Check the engine oil level using the yellow dipstick.",
        "The tire pressure should be checked every month.",
    ]
    response_sentences = [
        "To check the engine oil, pull the yellow dipstick and wipe it.",
    ]
    scores = BoschTaskProcessor._compute_sentence_rouge_scores(
        context_sentences, response_sentences, scorer
    )
    self.assertEqual(len(scores), 2)
    self.assertGreater(scores[0]["max_rouge"], 0.3)
    self.assertLess(scores[1]["max_rouge"], scores[0]["max_rouge"])

  def test_features_alignment_between_train_and_test(self):
    validation_rows = [
        {
            "Question": "Q?",
            "Context": "Context sentence 1. Context sentence 2.",
            "response": "Response sentence 1.",
            "class_hall": "No",
            "label": 1,
            "prompt": "p",
            "sample_id": "sid_1",
            "Retreival Setting": "setting_a",
            "Answer_sent_tokenized": "tok_a",
            "Sentence_labels": "labels_a",
            "Does_not_answer": False,
        }
    ]
    mock_data = DatasetDict({
        "validation": Dataset.from_list(validation_rows),
        "test": Dataset.from_list(validation_rows),
        "train": Dataset.from_list(validation_rows),
    })

    result = self.processor._make_structured_hallucinations_data(mock_data)
    train_cols = result["train"].column_names
    test_cols = result["test"].column_names
    self.assertEqual(train_cols, test_cols)


if __name__ == "__main__":
  unittest.main()
