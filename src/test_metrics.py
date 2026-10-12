"""Unit tests for src.metrics module."""

import json
import math
import os
import pickle
import shutil
import sys
import tempfile
import types
from typing import Any, Optional
import unittest
from unittest.mock import MagicMock, patch

from src.metrics import (
    GenerationMetricsEvaluator,
    compute_bertscore,
    compute_conditional_perplexity,
    compute_distinct_n,
    compute_length_metrics,
    compute_repetition_rate,
    compute_rouge,
    compute_summary_statistics,
    tokenize_words,
)


class MockDataset:
  """Mock Dataset for testing when Hugging Face datasets is mocked or unavailable."""

  def __init__(self, data: Any):
    if isinstance(data, list):
      keys = list(data[0].keys()) if data else []
      self._data = {k: [d.get(k) for d in data] for k in keys}
    elif isinstance(data, dict):
      self._data = dict(data)
    else:
      self._data = {}
    self.column_names = list(self._data.keys())

  @classmethod
  def from_dict(cls, d: dict[str, list[Any]]):
    return cls(d)

  @classmethod
  def from_list(cls, lst: list[dict[str, Any]], **kwargs):
    return cls(lst)

  def __getitem__(self, key: Any):
    if isinstance(key, str):
      return self._data[key]
    elif isinstance(key, int):
      return {col: self._data[col][key] for col in self.column_names}
    elif isinstance(key, slice):
      return {col: self._data[col][key] for col in self.column_names}
    raise TypeError(f"Invalid key type: {type(key)}")

  def __len__(self):
    if not self._data:
      return 0
    first_col = next(iter(self._data.values()))
    return len(first_col)

  @property
  def num_rows(self):
    return len(self)

  def to_dict(self):
    return dict(self._data)

  def remove_columns(self, column_name: Any):
    cols = [column_name] if isinstance(column_name, str) else column_name
    new_data = {k: v for k, v in self._data.items() if k not in cols}
    return MockDataset(new_data)

  def add_column(self, column_name: str, values: list[Any]):
    new_data = dict(self._data)
    new_data[column_name] = values
    return MockDataset(new_data)

  def rename_column(self, original_column_name: str, new_column_name: str):
    new_data = {
        (new_column_name if k == original_column_name else k): v
        for k, v in self._data.items()
    }
    return MockDataset(new_data)

  def select_columns(self, column_names: list[str]):
    new_data = {k: v for k, v in self._data.items() if k in column_names}
    return MockDataset(new_data)

  def select(self, indices: Any):
    new_data = {k: [self._data[k][i] for i in indices] for k in self.column_names}
    return MockDataset(new_data)

  def shuffle(self, seed: int = 42):
    return self

  def filter(self, fn: Any):
    num_rows = len(self)
    new_rows = []
    for i in range(num_rows):
      entry = {k: self._data[k][i] for k in self.column_names}
      if fn(entry):
        new_rows.append(entry)
    if not new_rows:
      return MockDataset({k: [] for k in self.column_names})
    new_data = {k: [row[k] for row in new_rows] for k in self.column_names}
    return MockDataset(new_data)

  def map(self, fn: Any, fn_kwargs: Optional[dict] = None):
    kwargs = fn_kwargs or {}
    num_rows = len(self)
    new_rows = []
    for i in range(num_rows):
      entry = {k: self._data[k][i] for k in self.column_names}
      res = fn(entry, **kwargs)
      if isinstance(res, dict):
        entry.update(res)
      new_rows.append(entry)
    if not new_rows:
      return MockDataset({k: [] for k in self.column_names})
    all_keys = list(new_rows[0].keys())
    new_data = {k: [row[k] for row in new_rows] for k in all_keys}
    return MockDataset(new_data)

  def __iter__(self):
    num_rows = len(self)
    for i in range(num_rows):
      yield {k: self._data[k][i] for k in self.column_names}


class DatasetDict(dict):

  def push_to_hub(self, repo_id: str, **kwargs):
    pass

  def save_to_disk(self, path: str, **kwargs):
    pass

  def map(self, fn, **kwargs):
    return DatasetDict({k: v.map(fn, **kwargs) for k, v in self.items()})


def _mock_concatenate_datasets(d_list):
  if not d_list:
    return MockDataset({})
  all_keys = d_list[0].column_names
  return MockDataset(
      {k: [item for d in d_list for item in d._data[k]] for k in all_keys}
  )


datasets_mod = types.ModuleType("datasets")
datasets_mod.Dataset = MockDataset
datasets_mod.DatasetDict = DatasetDict
datasets_mod.Value = MagicMock()
datasets_mod.load_dataset = MagicMock()
datasets_mod.concatenate_datasets = _mock_concatenate_datasets
sys.modules["datasets"] = datasets_mod

tqdm_mod = types.ModuleType("tqdm")
tqdm_mod.tqdm = lambda it, *args, **kwargs: it
sys.modules["tqdm"] = tqdm_mod

def _safe_pickle_save(obj, path):
  with open(path, "wb") as f:
    pickle.dump(obj, f)


def _safe_pickle_load(path):
  with open(path, "rb") as f:
    return pickle.load(f)


torch_mod = types.ModuleType("torch")
torch_mod.save = _safe_pickle_save
torch_mod.load = _safe_pickle_load
sys.modules["torch"] = torch_mod
sys.modules["torch.nn"] = types.ModuleType("torch.nn")

for _mod_name in [
    "transformers",
    "transformers.trainer_utils",
    "transformers.configuration_utils",
    "transformers.integrations",
    "transformers.integrations.heterogeneity",
    "transformers.integrations.heterogeneity.configuration_utils",
    "trl",
    "peft",
    "vllm",
    "vllm.lora",
    "vllm.lora.request",
    "evaluate",
    "scipy",
    "scipy.special",
    "sklearn",
    "sklearn.metrics",
    "google",
    "google.genai",
    "google.genai.types",
    "huggingface_hub",
    "matplotlib",
    "matplotlib.pyplot",
    "numpy",
]:
  if _mod_name not in sys.modules:
    sys.modules[_mod_name] = MagicMock()

try:
  from src.pipelines import (
      EvaluationAutoraterPipeline,
      EvaluationGenerationPipeline,
      EvaluationScoringPipeline,
      SSFODataGenerationPipeline,
      _VLLM_AVAILABLE,
      build_reward_hacking_fewshot_examples,
      degenerate_reward_hacked_response,
  )
except ImportError:
  EvaluationAutoraterPipeline = None
  EvaluationGenerationPipeline = None
  EvaluationScoringPipeline = None
  SSFODataGenerationPipeline = None
  _VLLM_AVAILABLE = False
  build_reward_hacking_fewshot_examples = None
  degenerate_reward_hacked_response = None

from src.utils import (
    REWARD_HACKING_SCALE_MAX,
    REWARD_HACKING_SCALE_MIN,
    build_eval_dataset_repo_id,
    compact_model_name,
    looks_like_rl_checkpoint,
    reward_hacking_dimension_keys,
    sanitize_hf_repo_id,
    sft_stacking_marker,
)


class MockTokenizer:
  """Mock tokenizer for testing length and perplexity."""

  def __init__(self):
    self.pad_token_id = 0
    self.eos_token_id = 1

  def encode(
      self,
      text: str,
      add_special_tokens: bool = False,
      truncation: bool = False,
      max_length: int = 2048,
  ) -> list[int]:
    words = tokenize_words(text)
    return [hash(w) % 1000 + 2 for w in words][:max_length]


class TestMetrics(unittest.TestCase):
  """Test suite for generation evaluation metrics."""

  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)

  def test_tokenize_words(self):
    self.assertEqual(tokenize_words("Hello, world!"), ["hello", "world"])
    self.assertEqual(tokenize_words(""), [])
    self.assertEqual(tokenize_words(None), [])
    self.assertEqual(tokenize_words("   "), [])

  def test_compute_rouge_exact_match(self):
    preds = ["The quick brown fox jumps over the lazy dog."]
    refs = ["The quick brown fox jumps over the lazy dog."]
    res = compute_rouge(preds, refs)
    self.assertAlmostEqual(res["rouge1_f1"][0], 1.0, places=4)
    self.assertAlmostEqual(res["rouge2_f1"][0], 1.0, places=4)
    self.assertAlmostEqual(res["rougeL_f1"][0], 1.0, places=4)

  def test_compute_rouge_disjoint_and_empty(self):
    preds = ["Apple banana cherry.", "", "Hello"]
    refs = ["Dog cat elephant.", "Hello world", ""]
    res = compute_rouge(preds, refs)
    self.assertAlmostEqual(res["rouge1_f1"][0], 0.0, places=4)
    self.assertEqual(res["rouge1_f1"][1], 0.0)
    self.assertEqual(res["rouge1_f1"][2], 0.0)

  def test_compute_rouge_mismatched_lengths(self):
    with self.assertRaises(ValueError):
      compute_rouge(["a"], ["a", "b"])

  def test_compute_length_metrics(self):
    completions = ["Hello world", "", "One two three four five"]
    res = compute_length_metrics(completions)
    self.assertEqual(res["char_length"], [11, 0, 23])
    self.assertEqual(res["token_length"], [2, 0, 5])

    tokenizer = MockTokenizer()
    res_tok = compute_length_metrics(completions, tokenizer=tokenizer)
    self.assertEqual(res_tok["token_length"], [2, 0, 5])

  def test_compute_distinct_n(self):
    all_unique = ["apple orange banana grape"]
    self.assertAlmostEqual(compute_distinct_n(all_unique, n=1)[0], 1.0)
    self.assertAlmostEqual(compute_distinct_n(all_unique, n=2)[0], 1.0)

    all_repeated = ["apple apple apple apple"]
    self.assertAlmostEqual(compute_distinct_n(all_repeated, n=1)[0], 0.25)
    self.assertAlmostEqual(
        compute_distinct_n(all_repeated, n=2)[0], 1.0 / 3.0, places=3
    )

    self.assertEqual(compute_distinct_n([""], n=1)[0], 0.0)
    self.assertEqual(compute_distinct_n([""], n=2)[0], 0.0)

  def test_compute_repetition_rate(self):
    all_repeated = ["apple apple apple apple"]
    distinct_1 = compute_distinct_n(all_repeated, n=1)[0]
    self.assertAlmostEqual(1.0 - distinct_1, 0.75)

    rep_rate = compute_repetition_rate(all_repeated, n=1)[0]
    self.assertAlmostEqual(rep_rate, 0.75)

  def test_compute_summary_statistics(self):
    values = [10.0, 20.0, 30.0]
    stats = compute_summary_statistics(values)
    self.assertAlmostEqual(stats["mean"], 20.0)
    self.assertAlmostEqual(stats["median"], 20.0)
    self.assertAlmostEqual(stats["min"], 10.0)
    self.assertAlmostEqual(stats["max"], 30.0)

    empty_stats = compute_summary_statistics([])
    self.assertEqual(empty_stats["mean"], 0.0)

  def test_reference_column_auto_detection(self):
    evaluator = GenerationMetricsEvaluator()
    self.assertEqual(
        evaluator.find_reference_column(
            ["prompt", "completion", "npov_response"]
        ),
        "npov_response",
    )
    self.assertEqual(
        evaluator.find_reference_column(["prompt", "completion", "target"]),
        "target",
    )
    self.assertEqual(
        evaluator.find_reference_column(
            ["prompt", "completion", "ground_truth"]
        ),
        "ground_truth",
    )
    self.assertIsNone(evaluator.find_reference_column(["prompt", "completion"]))

  def test_generation_metrics_evaluator_pipeline(self):
    data = {
        "prompt": ["Summarize this:", "Explain gravity:"],
        "completion": [
            "This is a test summary of the topic.",
            "Gravity is a fundamental force attracting masses.",
        ],
        "npov_response": [
            "This is a test summary of the topic.",
            "Gravity attracts two bodies with mass.",
        ],
    }
    dataset = MockDataset(data)

    evaluator = GenerationMetricsEvaluator(
        compute_bertscore_metric=False,
        compute_perplexity_metric=False,
    )
    updated_dataset, summary = evaluator.evaluate_dataset(
        dataset,
        prompt_column="prompt",
        completion_column="completion",
    )

    self.assertIn("char_length", updated_dataset.column_names)
    self.assertIn("token_length", updated_dataset.column_names)
    self.assertIn("distinct_1", updated_dataset.column_names)
    self.assertIn("distinct_2", updated_dataset.column_names)
    self.assertIn("repetition_rate", updated_dataset.column_names)
    self.assertIn("rouge1_f1", updated_dataset.column_names)
    self.assertIn("rouge2_f1", updated_dataset.column_names)
    self.assertIn("rougeL_f1", updated_dataset.column_names)

    self.assertAlmostEqual(updated_dataset["rougeL_f1"][0], 1.0, places=4)
    self.assertIn("rougeL_f1_mean", summary)
    self.assertIn("char_length_mean", summary)
    self.assertIn("token_length_mean", summary)

    saved_path = evaluator.save_and_log_results(
        updated_dataset,
        summary,
        output_dir=self.temp_dir,
        dataset_name="test_run",
        autorater_scores=[0.95, 0.80],
        threshold=0.5,
        log_to_wandb=False,
    )
    self.assertTrue(os.path.exists(saved_path))
    with open(saved_path, "r") as f:
      saved_summary = json.load(f)
    self.assertIn("hallucination_rate", saved_summary)
    self.assertAlmostEqual(saved_summary["hallucination_rate"], 0.0)
    self.assertIn("rougeL_f1_mean", saved_summary)

  @patch("src.metrics.wandb")
  def test_wandb_logging(self, mock_wandb):
    mock_wandb.run = None
    mock_wandb.Table = MagicMock(return_value="mock_table")

    data = {
        "prompt": ["Test prompt"],
        "completion": ["Test completion"],
        "reference": ["Test reference"],
    }
    dataset = MockDataset(data)

    evaluator = GenerationMetricsEvaluator(
        compute_bertscore_metric=False,
        compute_perplexity_metric=False,
    )
    updated_dataset, summary = evaluator.evaluate_dataset(dataset)

    evaluator.save_and_log_results(
        updated_dataset,
        summary,
        output_dir=self.temp_dir,
        dataset_name="wandb_test_run",
        autorater_scores=[0.95],
        threshold=0.5,
        log_to_wandb=True,
        wandb_project="test_proj",
    )

    mock_wandb.init.assert_called_once()
    mock_wandb.log.assert_called()

  def test_subsampling(self):
    """Test subsampling logic with mock dataset."""
    data = {
        "prompt": [f"Prompt {i}" for i in range(50)],
        "completion": [f"Completion {i}" for i in range(50)],
    }
    dataset = MockDataset(data)

    def subsample(data_obj, max_samples, seed=42):
      if (
          max_samples is not None
          and max_samples > 0
          and len(data_obj) > max_samples
      ):
        return data_obj.shuffle(seed=seed).select(range(max_samples))
      return data_obj

    subsampled = subsample(dataset, 10, 42)
    self.assertEqual(len(subsampled), 10)

    # When max_eval_samples is -1 or None, should not subsample
    full = subsample(dataset, -1, 42)
    self.assertEqual(len(full), 50)

  def test_subsampled_scoring_preserves_full_dataset(self):
    """Verifies that scoring a subsample preserves all un-evaluated rows."""
    full_data = {
        "prompt": [f"Prompt {i}" for i in range(100)],
        "completion": [f"Completion {i}" for i in range(100)],
        "reference": [f"Reference {i}" for i in range(100)],
    }
    full_dataset = MockDataset(full_data)

    # Subsample 10 rows
    import random

    rng = random.Random(12345)
    all_indices = list(range(len(full_dataset)))
    rng.shuffle(all_indices)
    eval_indices = all_indices[:10]
    sub_val_data = full_dataset.select(eval_indices)

    # Evaluate the 10 rows
    evaluator = GenerationMetricsEvaluator(
        compute_bertscore_metric=False,
        compute_perplexity_metric=False,
    )
    sub_val_data, summary = evaluator.evaluate_dataset(sub_val_data)
    sub_scores = [0.99] * 10
    sub_val_data = sub_val_data.add_column("scores", sub_scores)

    # Merge back into full dataset
    final_dataset = full_dataset
    evaluated_cols = ["scores", "rouge1_f1", "rougeL_f1", "token_length"]
    for col in evaluated_cols:
      full_col_vals = [None] * len(final_dataset)
      for idx_in_val, orig_idx in enumerate(eval_indices):
        full_col_vals[orig_idx] = sub_val_data[col][idx_in_val]
      final_dataset = final_dataset.add_column(col, full_col_vals)

    # Total rows must remain 100!
    self.assertEqual(len(final_dataset), 100)
    # The 10 evaluated rows have valid metric values
    for orig_idx in eval_indices:
      self.assertIsNotNone(final_dataset["scores"][orig_idx])
      self.assertEqual(final_dataset["scores"][orig_idx], 0.99)
      self.assertIsNotNone(final_dataset["rougeL_f1"][orig_idx])
    # The remaining 90 rows have None for scores, but keep their original prompt/completion
    unscored_indices = [i for i in range(100) if i not in eval_indices]
    self.assertEqual(len(unscored_indices), 90)
    for orig_idx in unscored_indices:
      self.assertIsNone(final_dataset["scores"][orig_idx])
      self.assertEqual(final_dataset["prompt"][orig_idx], f"Prompt {orig_idx}")
      self.assertEqual(
          final_dataset["completion"][orig_idx], f"Completion {orig_idx}"
      )

  def test_save_and_log_results_with_none_autorater_scores(self):
    """Verifies that save_and_log_results handles None in autorater_scores without crashing."""
    evaluator = GenerationMetricsEvaluator(
        compute_bertscore_metric=False,
        compute_perplexity_metric=False,
    )
    dataset = MockDataset({
        "prompt": ["P1", "P2", "P3"],
        "completion": ["C1", "C2", "C3"],
        "token_length": [10, 20, 30],
        "distinct_2": [0.8, 0.9, 1.0],
        "repetition_rate": [0.0, 0.1, 0.0],
    })
    summary = {"avg_score": 0.9}
    scores_with_none = [0.95, None, 0.88]
    summary_path = evaluator.save_and_log_results(
        dataset,
        summary,
        output_dir=self.temp_dir,
        dataset_name="test_none_scores",
        autorater_scores=scores_with_none,
        threshold=0.9,
    )
    self.assertTrue(os.path.exists(summary_path))
    with open(summary_path) as f:
      saved_data = json.load(f)
    self.assertIn("hallucination_rate", saved_data)
    self.assertIn("faithfulness_rate", saved_data)
    self.assertAlmostEqual(saved_data["faithfulness_rate"], 0.5, places=2)
    self.assertEqual(saved_data["autorater_scored_count"], 2)
    self.assertEqual(saved_data["autorater_unscored_count"], 1)

  def test_evaluate_dataset_idempotent_recalculation(self):
    """Verifies that calling evaluate_dataset multiple times cleanly updates existing metric columns."""
    evaluator = GenerationMetricsEvaluator(
        compute_bertscore_metric=False,
        compute_perplexity_metric=False,
    )
    dataset = MockDataset({
        "prompt": ["P1", "P2"],
        "completion": ["Hello world", "Another sentence test"],
        "reference": ["Hello world", "Sentence reference"],
        "token_length": [999, 999],  # Stale values from prior run
    })
    updated_ds, summary = evaluator.evaluate_dataset(dataset)
    # Stale token_length should be overwritten with accurate values
    self.assertEqual(updated_ds["token_length"], [2, 3])

  def test_evaluate_dataset_multi_perspective_context(self):
    """Verifies that ROUGE & context grounding works on NPOV datasets with perspective_1/2."""
    evaluator = GenerationMetricsEvaluator(
        compute_bertscore_metric=False,
        compute_perplexity_metric=False,
    )
    dataset = MockDataset({
        "prompt": ["Write NPOV response"],
        "completion": [
            "Nuclear energy produces zero emissions but waste storage is an"
            " issue."
        ],
        "perspective_1": [
            "Nuclear energy produces zero greenhouse gas emissions."
        ],
        "perspective_1_name": ["pro"],
        "perspective_2": ["Waste storage is a major hazard."],
        "perspective_2_name": ["con"],
    })
    updated_ds, summary = evaluator.evaluate_dataset(dataset)
    self.assertIn("rouge1_precision", updated_ds.column_names)
    self.assertIn("rouge1_recall", updated_ds.column_names)
    self.assertIn("rougeL_f1", updated_ds.column_names)
    self.assertGreater(updated_ds["rouge1_precision"][0], 0.0)
    self.assertGreater(updated_ds["rouge1_recall"][0], 0.0)
    self.assertGreater(updated_ds["rougeL_f1"][0], 0.0)

  def test_compute_conditional_perplexity_no_torch_or_model(self):
    """Test that missing torch or model returns 0.0 without error."""
    res = compute_conditional_perplexity(
        ["Prompt"], ["Completion"], model=None, tokenizer=None
    )
    self.assertEqual(res, [0.0])

  def test_compute_conditional_perplexity_empty(self):
    """Test that empty or missing completions return 0.0 perplexity."""
    tokenizer = MockTokenizer()
    model = MagicMock()
    ppl = compute_conditional_perplexity(
        ["Prompt"], [""], model=model, tokenizer=tokenizer
    )
    self.assertEqual(ppl, [0.0])

  def test_compute_conditional_perplexity_with_mock_model(self):
    """Test conditional perplexity calculation with causal LM when torch is present."""
    try:
      import torch
    except ImportError:
      torch = None

    if torch is None or not hasattr(torch, "__file__"):
      res = compute_conditional_perplexity(
          ["Prompt"],
          ["Completion"],
          model=MagicMock(),
          tokenizer=MockTokenizer(),
      )
      self.assertEqual(res, [0.0])
      return

    class MockCausalLM:

      def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self._param = torch.nn.Parameter(torch.zeros(1))

      def parameters(self):
        yield self._param

      def eval(self):
        pass

      def __call__(self, input_ids=None):
        batch_size, seq_len = input_ids.shape
        logits = torch.ones(
            (batch_size, seq_len, self.vocab_size), dtype=torch.float32
        )
        out = MagicMock()
        out.logits = logits
        return out

    model = MockCausalLM(vocab_size=1000)
    tokenizer = MockTokenizer()
    prompts = ["Question: What is gravity?"]
    completions = ["Gravity is a natural phenomenon attracting masses."]

    ppl = compute_conditional_perplexity(
        prompts, completions, model=model, tokenizer=tokenizer
    )
    self.assertEqual(len(ppl), 1)
    # With uniform logits over 1000 vocab, cross-entropy is ln(1000) ~ 6.908, PPL is ~ 1000.0
    self.assertAlmostEqual(ppl[0], 1000.0, places=1)

  def test_generation_metrics_evaluator_with_perplexity(self):
    """Test GenerationMetricsEvaluator with fluency model and tokenizer enabled."""
    model = MagicMock()
    tokenizer = MockTokenizer()

    data = {
        "prompt": ["What is AI?", "What is water?"],
        "completion": ["Artificial intelligence.", "A chemical compound H2O."],
        "reference": ["AI overview", "Water chemistry"],
    }
    dataset = MockDataset(data)
    evaluator = GenerationMetricsEvaluator(
        compute_bertscore_metric=False,
        compute_perplexity_metric=True,
        fluency_model=model,
        fluency_tokenizer=tokenizer,
    )
    updated_ds, summary = evaluator.evaluate_dataset(dataset)

    self.assertIn("perplexity", updated_ds.column_names)
    self.assertIn("perplexity_mean", summary)
    self.assertIn("perplexity_median", summary)
    self.assertEqual(len(updated_ds["perplexity"]), 2)


class TestEvalRepoNaming(unittest.TestCase):
  """Unit tests for Hugging Face repo ID sanitization, compaction, and building."""

  def test_compact_model_name_floats(self):
    raw_name = "npov_PERL_google_S130104_epo0.2_lr2.3663655877360862e-05_beta0.04259128063013425_2608141244"
    compacted = compact_model_name(raw_name)
    self.assertEqual(
        compacted,
        "npov_PERL_google_S130104_epo0.2_lr2.4e-05_beta0.043_2608141244",
    )

  def test_compact_model_name_standard(self):
    raw_name = "google/gemma-4-E2B-it"
    compacted = compact_model_name(raw_name)
    self.assertEqual(compacted, "google/gemma-4-E2B-it")

  def test_build_eval_dataset_repo_id_user_reported_case(self):
    raw_lora = "leobianco/npov_PERL_google_S130104_epo0.2_lr2.3663655877360862e-05_beta0.04259128063013425_2608141244"
    repo_id = build_eval_dataset_repo_id(
        user="leobianco",
        writer_model_lora=raw_lora,
        temperature=0.0,
        writer_num_fewshot=0,
    )
    self.assertLessEqual(len(repo_id), 96)
    self.assertNotIn(".", repo_id)
    self.assertTrue(repo_id.endswith("_gens_T0_wfs0_s12345_mt250_nosft"))
    self.assertTrue(
        repo_id.startswith(
            "leobianco/eval_npov_PERL_google_S130104_epo0_2_lr2_4e-05_be"
        )
    )
    # The overflowing tail is replaced by a digest of the full name rather
    # than simply cut, so two checkpoints differing only there stay distinct.
    self.assertRegex(repo_id, r"_[0-9a-f]{4}_gens_T0_wfs0_s12345_mt250_nosft$")

  def test_sanitize_hf_repo_id_keeps_the_timestamp_of_a_long_model(self):
    """Overlong names must lose the middle, never the tail.

    The tail is the timestamp. Right-truncation removed it whole for any
    base model a dozen characters longer than `gemma-4-E4B-it`, so two
    campaigns run on the same day with the same hyperparameters pushed to
    one repository. `Mistral-7B-Instruct-v0.3` is exactly that case.
    """
    stamps = ("2609200900", "2609201500")
    ids = set()
    for stamp in stamps:
      repo_id = sanitize_hf_repo_id(
          "leobianco/ragtruth-summarization_PERL_synthetic_struct_"
          f"Mistral-7B-Instruct-v0.3_S130104_epo3_lr1e-04_beta0.05_r16_{stamp}"
      )
      self.assertLessEqual(len(repo_id), 96)
      self.assertTrue(repo_id.endswith(stamp), repo_id)
      self.assertTrue(repo_id.startswith("leobianco/ragtruth"), repo_id)
      ids.add(repo_id)
    self.assertEqual(len(ids), 2, ids)

  def test_sanitize_hf_repo_id_preserves_uppercase(self):
    # The Hub allows uppercase; lowercasing would stop a published
    # checkpoint from naming its base model recognisably.
    self.assertEqual(
        sanitize_hf_repo_id("leobianco/npov_SFT_Qwen3-4B-Instruct-2507"),
        "leobianco/npov_SFT_Qwen3-4B-Instruct-2507",
    )

  def test_sanitize_hf_repo_id_matches_build(self):
    uncompacted_eval = "leobianco/eval_npov_PERL_google_S130104_epo0.2_lr2.3663655877360862e-05_beta0.04259128063013425_2608141244_gens_T0.0_wfs0"
    sanitized = sanitize_hf_repo_id(uncompacted_eval)
    self.assertLessEqual(len(sanitized), 96)
    self.assertNotIn(".", sanitized)
    self.assertEqual(
        sanitized,
        "leobianco/eval_npov_PERL_google_S130104_epo0_2_lr2_4e-05_beta0_043_2608141244_gens_T0_0_wfs0",
    )

  def test_build_eval_dataset_repo_id_extreme_length(self):
    huge_model_name = "leobianco/" + ("very_long_model_name_identifier_" * 5)
    repo_id = build_eval_dataset_repo_id(
        user="leobianco",
        writer_model_lora=huge_model_name,
        temperature=0.7,
        writer_num_fewshot=2,
    )
    self.assertLessEqual(len(repo_id), 96)
    self.assertNotIn(".", repo_id)
    self.assertTrue(repo_id.startswith("leobianco/eval_"))
    self.assertTrue(repo_id.endswith("_gens_T0_7_wfs2_s12345_mt250_nosft"))

  def test_build_eval_dataset_repo_id_marker_survives_truncation(self):
    """An overlong name must lose model characters, never the SFT marker.

    The marker is the last thing in the name, so it must be protected by
    shrinking the model name up front. Appending it and hoping the generic
    bounding spares it would silently merge the stacked and un-stacked runs
    back into one repository.
    """
    huge_model_name = "leobianco/" + ("very_long_model_name_identifier_" * 10)
    stacked = build_eval_dataset_repo_id(
        user="leobianco",
        writer_model_lora=huge_model_name,
        temperature=0.0,
        writer_num_fewshot=0,
        task_name="npov",
        sft_model_path="leobianco/npov_SFT_ckpt",
    )
    unstacked = build_eval_dataset_repo_id(
        user="leobianco",
        writer_model_lora=huge_model_name,
        temperature=0.0,
        writer_num_fewshot=0,
        task_name="npov",
    )
    self.assertLessEqual(len(stacked), 96)
    self.assertLessEqual(len(unstacked), 96)
    self.assertNotEqual(stacked, unstacked)
    self.assertTrue(unstacked.endswith("_gens_T0_wfs0_s12345_mt250_nosft"))
    self.assertRegex(stacked, r"_gens_T0_wfs0_s12345_mt250_sft[0-9a-f]{6}$")

  def test_build_eval_dataset_repo_id_distinguishes_sft_stacking(self):
    """The same PE-RL checkpoint served two ways gets two repositories."""
    perl_ckpt = "leobianco/npov_PERL_gemma-4-E4B-it_S130104"
    with_sft = build_eval_dataset_repo_id(
        user="leobianco",
        writer_model_lora=perl_ckpt,
        temperature=0.0,
        writer_num_fewshot=0,
        task_name="npov",
        sft_model_path="leobianco/npov_SFT_gemma-4-E4B-it_S130104",
    )
    without_sft = build_eval_dataset_repo_id(
        user="leobianco",
        writer_model_lora=perl_ckpt,
        temperature=0.0,
        writer_num_fewshot=0,
        task_name="npov",
        sft_model_path="",
    )
    other_sft = build_eval_dataset_repo_id(
        user="leobianco",
        writer_model_lora=perl_ckpt,
        temperature=0.0,
        writer_num_fewshot=0,
        task_name="npov",
        sft_model_path="leobianco/npov_SFT_a_different_run",
    )
    self.assertNotEqual(with_sft, without_sft)
    self.assertNotEqual(with_sft, other_sft)
    self.assertTrue(without_sft.endswith("_nosft"))
    self.assertRegex(with_sft, r"_sft[0-9a-f]{6}$")

  def test_build_eval_dataset_repo_id_falsy_sft_refs_are_nosft(self):
    """Shell and config spellings of "nothing" must not fork the name."""
    names = {
        build_eval_dataset_repo_id(
            user="leobianco",
            writer_model_lora="leobianco/npov_PERL_ckpt",
            task_name="npov",
            sft_model_path=value,
        )
        for value in (None, "", "   ", "none", "None", "false", "null")
    }
    self.assertEqual(len(names), 1)
    self.assertTrue(names.pop().endswith("_nosft"))

  def test_build_eval_dataset_repo_id_self_stacking_is_nosft(self):
    """Evaluating the SFT checkpoint itself stacks nothing."""
    sft_ckpt = "leobianco/npov_SFT_gemma-4-E4B-it_S130104"
    self.assertEqual(
        build_eval_dataset_repo_id(
            user="leobianco",
            writer_model_lora=sft_ckpt,
            task_name="npov",
            sft_model_path=sft_ckpt,
        ),
        build_eval_dataset_repo_id(
            user="leobianco",
            writer_model_lora=sft_ckpt,
            task_name="npov",
        ),
    )

  def test_build_eval_dataset_repo_id_marker_ignores_trailing_slash(self):
    """A trailing slash is not a different checkpoint."""
    self.assertEqual(
        sft_stacking_marker("leobianco/npov_SFT_ckpt/"),
        sft_stacking_marker("leobianco/npov_SFT_ckpt"),
    )

  def test_looks_like_rl_checkpoint(self):
    self.assertTrue(
        looks_like_rl_checkpoint("leobianco/npov_PERL_gemma-4-E4B-it_S1")
    )
    self.assertTrue(looks_like_rl_checkpoint("leobianco/bosch_DPO_ssfo_run"))
    # An SFT tag wins, so the 'new_perl' project prefix cannot false-positive.
    self.assertFalse(
        looks_like_rl_checkpoint("leobianco/new_perl_npov_SFT_run")
    )
    self.assertFalse(looks_like_rl_checkpoint("google/gemma-4-E4B-it"))
    self.assertFalse(looks_like_rl_checkpoint(""))
    self.assertFalse(looks_like_rl_checkpoint(None))

  def test_build_eval_dataset_repo_id_distinct_for_different_tasks(self):
    base_model = "google/gemma-4-E4B-it"
    bosch_id = build_eval_dataset_repo_id(
        user="leobianco",
        writer_model_lora=base_model,
        temperature=0.0,
        writer_num_fewshot=0,
        task_name="bosch",
    )
    npov_id = build_eval_dataset_repo_id(
        user="leobianco",
        writer_model_lora=base_model,
        temperature=0.0,
        writer_num_fewshot=0,
        task_name="npov",
    )
    ragtruth_id = build_eval_dataset_repo_id(
        user="leobianco",
        writer_model_lora=base_model,
        temperature=0.0,
        writer_num_fewshot=0,
        task_name="ragtruth",
    )
    self.assertNotEqual(bosch_id, npov_id)
    self.assertNotEqual(bosch_id, ragtruth_id)
    self.assertEqual(
        bosch_id, "leobianco/eval_bosch_gemma-4-E4B-it_gens_T0_wfs0_s12345_mt250_nosft"
    )
    self.assertEqual(
        npov_id, "leobianco/eval_npov_gemma-4-E4B-it_gens_T0_wfs0_s12345_mt250_nosft"
    )
    self.assertEqual(
        ragtruth_id,
        "leobianco/eval_ragtruth_gemma-4-E4B-it_gens_T0_wfs0_s12345_mt250_nosft",
    )

  def test_build_eval_dataset_repo_id_no_duplicate_task_prefix(self):
    adapter = "leobianco/bosch_PERL_gemma-4-E2B-it_S130104"
    repo_id = build_eval_dataset_repo_id(
        user="leobianco",
        writer_model_lora=adapter,
        temperature=0.0,
        writer_num_fewshot=0,
        task_name="bosch",
    )
    self.assertFalse(repo_id.startswith("leobianco/eval_bosch_bosch_"))
    self.assertTrue(repo_id.startswith("leobianco/eval_bosch_PERL_gemma-4-E2B-it_S130104"))


class _LogprobCandidate:
  """Mirrors `google.genai.types.LogprobsResultCandidate`.

  A plain `MagicMock` invents any attribute that is read, so a test written
  against the wrong field name (`log_prob`) passed while production code read
  `None` from every real response. This stub exposes exactly the fields the
  SDK exposes, so a field rename breaks the test instead of the pipeline.
  """

  __slots__ = ("token", "token_id", "log_probability")

  def __init__(self, token: str, log_probability: Optional[float] = None):
    self.token = token
    self.token_id = 0
    self.log_probability = log_probability


class _LogprobStep:
  """Mirrors `google.genai.types.LogprobsResultTopCandidates`."""

  __slots__ = ("candidates",)

  def __init__(self, candidates):
    self.candidates = candidates


class TestGeminiScoreResponse(unittest.TestCase):
  """Unit tests for gemini_score_response in EvaluationPipeline."""

  def setUp(self):
    if EvaluationScoringPipeline is None:
      self.skipTest("pipelines module not available in lightweight test env")
    self.pipeline = EvaluationScoringPipeline()

  def _response(self, text, top_candidates=None, chosen=None, avg=None):
    candidate = MagicMock()
    if top_candidates is None and chosen is None:
      candidate.logprobs_result = None
    else:
      candidate.logprobs_result.top_candidates = top_candidates or []
      candidate.logprobs_result.chosen_candidates = chosen or []
    candidate.avg_logprobs = avg
    return MagicMock(candidates=[candidate], text=text)

  def test_sdk_field_name_is_log_probability(self):
    """Guards the exact field the production parser reads.

    Reading a field the SDK does not have (`log_prob`) returns None silently
    and collapses every score onto the binary text fallback, which is exactly
    how the autorater produced 0 continuous probabilities.
    """
    try:
      from google.genai import types as genai_types  # pylint: disable=g-import-not-at-top
    except ImportError:
      self.skipTest("google-genai not installed in this environment")
    fields = getattr(
        getattr(genai_types, "LogprobsResultCandidate", None),
        "model_fields",
        None,
    )
    if not isinstance(fields, dict):
      self.skipTest("google.genai is stubbed out in this test environment")
    self.assertIn("log_probability", fields)

  def test_both_tokens_in_logprobs(self):
    response = self._response(
        '"No"',
        top_candidates=[
            _LogprobStep([_LogprobCandidate('"', -0.01)]),
            _LogprobStep([
                _LogprobCandidate("No", -0.2),
                _LogprobCandidate("Yes", -1.8),
            ]),
        ],
    )
    score = self.pipeline.gemini_score_response(response)
    expected = math.exp(-0.2) / (math.exp(-0.2) + math.exp(-1.8))
    self.assertAlmostEqual(score, expected, places=4)

  def test_only_no_token_in_logprobs_is_continuous(self):
    response = self._response(
        '"No"',
        top_candidates=[_LogprobStep([_LogprobCandidate("No", -0.7)])],
    )
    score = self.pipeline.gemini_score_response(response)
    self.assertAlmostEqual(score, math.exp(-0.7), places=6)

  def test_only_yes_token_in_logprobs_is_continuous(self):
    response = self._response(
        '"Yes"',
        top_candidates=[_LogprobStep([_LogprobCandidate("Yes", -0.7)])],
    )
    score = self.pipeline.gemini_score_response(response)
    self.assertAlmostEqual(score, 1.0 - math.exp(-0.7), places=6)

  def test_missing_logprob_values_do_not_crash(self):
    """A step whose candidates carry no logprob must not poison the score."""
    response = self._response(
        '"No"',
        top_candidates=[
            _LogprobStep([
                _LogprobCandidate("No", None),
                _LogprobCandidate("Yes", None),
            ])
        ],
    )
    score = self.pipeline.gemini_score_response(response)
    self.assertEqual(score, 1.0)  # Falls through to the text classifier.

  def test_text_fallback_no_quotes(self):
    response = self._response("No")
    score = self.pipeline.gemini_score_response(response)
    self.assertEqual(score, 1.0)

  def test_text_fallback_json_string_yes(self):
    response = self._response('  "Yes"  ')
    score = self.pipeline.gemini_score_response(response)
    self.assertEqual(score, 0.0)

  def test_zero_avg_logprobs_is_not_treated_as_confidence(self):
    """avg_logprobs == 0.0 means 'not computed', not 'P == 1'."""
    response = self._response("No", avg=0.0)
    score = self.pipeline.gemini_score_response(response)
    self.assertEqual(score, 1.0)  # From the text classifier, not exp(0.0).

  def test_empty_response_fallback(self):
    response = MagicMock(candidates=[])
    score = self.pipeline.gemini_score_response(response)
    self.assertIsNone(score)

  def test_safety_blocked_candidate(self):
    candidate = MagicMock()
    candidate.finish_reason = "SAFETY"
    candidate.content.parts = []
    candidate.logprobs_result = None
    candidate.avg_logprobs = None
    response = MagicMock(candidates=[candidate], text="")
    score = self.pipeline.gemini_score_response(response)
    self.assertIsNone(score)

  def test_unresolved_text_fallback(self):
    response = self._response(
        "I cannot determine faithfulness for this sample."
    )
    score = self.pipeline.gemini_score_response(response)
    self.assertIsNone(score)

  def test_chosen_candidates_fallback_uses_logprob(self):
    response = self._response(
        '"No"',
        top_candidates=[],
        chosen=[_LogprobCandidate("No", -0.3)],
    )
    score = self.pipeline.gemini_score_response(response)
    self.assertAlmostEqual(score, math.exp(-0.3), places=6)

  def test_chosen_candidates_fallback_without_logprob(self):
    response = self._response(
        '"No"', top_candidates=[], chosen=[_LogprobCandidate("No", None)]
    )
    score = self.pipeline.gemini_score_response(response)
    self.assertEqual(score, 1.0)

  def test_structured_json_with_preamble(self):
    response = self._response(
        'Here is the JSON requested:\n```json\n{"answer": "No"}\n```'
    )
    score = self.pipeline.gemini_score_response(response)
    self.assertEqual(score, 1.0)

  def test_truncated_preamble_returns_none(self):
    response = self._response("Here is the JSON requested:\n```json")
    score = self.pipeline.gemini_score_response(response)
    self.assertIsNone(score)


class TestGeminiScoreDataset(unittest.TestCase):
  """Unit tests for gemini_score_dataset error handling and resumption."""

  def setUp(self):
    if EvaluationScoringPipeline is None:
      self.skipTest("pipelines module not available in lightweight test env")
    self.pipeline = EvaluationScoringPipeline()
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)

  def _make_dummy_dataset(self, prompts: list[str]):
    return MockDataset({
        "evaluator_prompt": prompts,
        "prompt": [f"P{i}" for i in range(len(prompts))],
        "completion": [f"C{i}" for i in range(len(prompts))],
    })

  def test_dataset_scoring_with_api_failures_returns_none_and_checkpoints(self):
    ckpt_file = os.path.join(self.temp_dir, "test_checkpoint.pt")
    script_args = MagicMock(
        seed=42,
        evaluator_model="gemini-2.5-flash",
        evaluator_num_fewshot=0,
        overwrite_scores=False,
        max_workers=1,
        dataset_with_completions=None,
        dataset_labels="test_eval",
        scores_checkpoint_path=ckpt_file,
    )
    dataset = self._make_dummy_dataset(["Prompt 0", "Prompt 1", "Prompt 2"])

    def mock_generate(model, contents, config):
      p = str(contents)
      if "Prompt 1" in p:
        raise RuntimeError("Simulated unrecoverable API error")
      cand = MagicMock()
      cand.logprobs_result.top_candidates = [
          MagicMock(candidates=[MagicMock(token="No", log_prob=-0.01)])
      ]
      return MagicMock(candidates=[cand], text='"No"')

    client = MagicMock()
    client.models.generate_content.side_effect = mock_generate

    scores = self.pipeline.gemini_score_dataset(client, dataset, script_args)
    self.assertEqual(scores, [1.0, None, 1.0])
    # Checkpoint must be saved because unscored_count > 0
    self.assertTrue(os.path.exists(ckpt_file))

  def test_dataset_scoring_resumes_missing_entries_from_checkpoint(self):
    ckpt_file = os.path.join(self.temp_dir, "resume_checkpoint.pt")
    # Pre-save checkpoint with sample 0 and 2 scored, sample 1 missing
    sys.modules["torch"].save([1.0, None, 1.0], ckpt_file)

    script_args = MagicMock(
        seed=42,
        evaluator_model="gemini-2.5-flash",
        evaluator_num_fewshot=0,
        overwrite_scores=False,
        max_workers=1,
        dataset_with_completions=None,
        dataset_labels="test_eval",
        scores_checkpoint_path=ckpt_file,
    )
    dataset = self._make_dummy_dataset(["Prompt 0", "Prompt 1", "Prompt 2"])

    called_prompts = []

    def mock_generate(model, contents, config):
      p = str(contents)
      called_prompts.append(p)
      cand = MagicMock()
      cand.logprobs_result.top_candidates = [
          MagicMock(candidates=[MagicMock(token="Yes", log_prob=-0.01)])
      ]
      return MagicMock(candidates=[cand], text='"Yes"')

    client = MagicMock()
    client.models.generate_content.side_effect = mock_generate

    scores = self.pipeline.gemini_score_dataset(client, dataset, script_args)
    # Only Prompt 1 should have been called
    self.assertEqual(len(called_prompts), 1)
    self.assertIn("Prompt 1", called_prompts[0])
    self.assertEqual(scores, [1.0, 0.0, 1.0])
    # Checkpoint deleted upon 100% completion
    self.assertFalse(os.path.exists(ckpt_file))

  def test_dataset_scoring_resumes_missing_entries_from_dataset_column(self):
    ckpt_file = os.path.join(self.temp_dir, "nonexistent_ckpt.pt")
    script_args = MagicMock(
        seed=42,
        evaluator_model="gemini-2.5-flash",
        evaluator_num_fewshot=0,
        overwrite_scores=False,
        max_workers=1,
        dataset_with_completions=None,
        dataset_labels="test_eval",
        scores_checkpoint_path=ckpt_file,
    )
    dataset = MockDataset({
        "evaluator_prompt": ["Prompt 0", "Prompt 1", "Prompt 2"],
        "scores": [0.85, None, 0.15],
    })

    called_prompts = []

    def mock_generate(model, contents, config):
      p = str(contents)
      called_prompts.append(p)
      cand = MagicMock()
      cand.logprobs_result.top_candidates = [
          MagicMock(candidates=[MagicMock(token="No", log_prob=-0.01)])
      ]
      return MagicMock(candidates=[cand], text='"No"')

    client = MagicMock()
    client.models.generate_content.side_effect = mock_generate

    scores = self.pipeline.gemini_score_dataset(client, dataset, script_args)
    self.assertEqual(len(called_prompts), 1)
    self.assertIn("Prompt 1", called_prompts[0])
    self.assertEqual(scores, [0.85, 1.0, 0.15])

  def test_dataset_scoring_overwrite_scores_forces_full_rescore(self):
    ckpt_file = os.path.join(self.temp_dir, "overwrite_ckpt.pt")
    sys.modules["torch"].save([0.2, 0.3, 0.4], ckpt_file)

    script_args = MagicMock(
        seed=42,
        evaluator_model="gemini-2.5-flash",
        evaluator_num_fewshot=0,
        overwrite_scores=True,
        max_workers=1,
        dataset_with_completions=None,
        dataset_labels="test_eval",
        scores_checkpoint_path=ckpt_file,
    )
    dataset = self._make_dummy_dataset(["Prompt 0", "Prompt 1", "Prompt 2"])

    called_prompts = []

    def mock_generate(model, contents, config):
      p = str(contents)
      called_prompts.append(p)
      cand = MagicMock()
      cand.logprobs_result.top_candidates = [
          MagicMock(candidates=[MagicMock(token="No", log_prob=-0.01)])
      ]
      return MagicMock(candidates=[cand], text='"No"')

    client = MagicMock()
    client.models.generate_content.side_effect = mock_generate

    scores = self.pipeline.gemini_score_dataset(client, dataset, script_args)
    # With overwrite_scores=True, all 3 must be scored
    self.assertEqual(len(called_prompts), 3)
    self.assertEqual(scores, [1.0, 1.0, 1.0])

  def test_huggingface_only_resumption_no_local_checkpoint(self):
    """Verify Hugging Face-only resumption with zero local checkpoint files."""
    script_args = MagicMock(
        seed=42,
        evaluator_model="gemini-2.5-flash",
        evaluator_num_fewshot=0,
        overwrite_scores=False,
        max_workers=1,
        dataset_with_completions="leobianco/test_completions",
        dataset_labels=None,
        scores_checkpoint_path=None,
    )
    # Simulate dataset loaded from HF Hub with 2 existing scores and 1 missing
    dataset = MockDataset({
        "evaluator_prompt": ["Prompt 0", "Prompt 1", "Prompt 2"],
        "scores": [0.95, None, 0.05],
    })

    called_prompts = []

    def mock_generate(model, contents, config):
      p = str(contents)
      called_prompts.append(p)
      cand = MagicMock()
      cand.logprobs_result.top_candidates = [
          MagicMock(candidates=[MagicMock(token="No", log_prob=-0.01)])
      ]
      return MagicMock(candidates=[cand], text='"No"')

    client = MagicMock()
    client.models.generate_content.side_effect = mock_generate

    scores = self.pipeline.gemini_score_dataset(client, dataset, script_args)
    # Only Prompt 1 should have been called
    self.assertEqual(len(called_prompts), 1)
    self.assertIn("Prompt 1", called_prompts[0])
    self.assertEqual(scores, [0.95, 1.0, 0.05])

  def test_dataset_scoring_token_budget_and_system_instruction(self):
    """Test that Gemini API config includes 64 tokens and system instruction."""
    dataset = MockDataset({"evaluator_prompt": ["Evaluate faithfulness."]})
    script_args = MagicMock(
        evaluator_model="gemini-2.5-flash",
        scores_checkpoint_path=None,
        overwrite_scores=False,
        max_workers=1,
        seed=42,
    )
    captured_calls = []

    def mock_generate(model, contents, config):
      captured_calls.append(config)
      cand = MagicMock()
      cand.logprobs_result.top_candidates = [
          _LogprobStep([_LogprobCandidate("No", -0.01)])
      ]
      cand.logprobs_result.chosen_candidates = []
      cand.avg_logprobs = None
      return MagicMock(candidates=[cand], text='"No"')

    client = MagicMock()
    client.models.generate_content.side_effect = mock_generate

    scores = self.pipeline.gemini_score_dataset(client, dataset, script_args)
    self.assertEqual(len(captured_calls), 1)
    # Only 'No' was in the top-k, so the score is exp(-0.01), not a hard 1.0.
    self.assertAlmostEqual(scores[0], math.exp(-0.01), places=6)

    call_config = captured_calls[0]
    from google.genai import types as genai_types

    if (
        hasattr(genai_types.GenerateContentConfig, "call_args")
        and genai_types.GenerateContentConfig.call_args is not None
    ):
      kwargs = genai_types.GenerateContentConfig.call_args.kwargs
      max_tokens = kwargs.get("max_output_tokens", 0)
      sys_inst = kwargs.get("system_instruction", "")
    else:
      max_tokens = getattr(call_config, "max_output_tokens", 0)
      sys_inst = getattr(call_config, "system_instruction", "")
    self.assertGreaterEqual(max_tokens, 64)
    self.assertIn("Yes", str(sys_inst))
    self.assertIn("No", str(sys_inst))

  def _continuous_response(self, p_no: float = -0.01):
    """Builds a response carrying both logprobs, i.e. a continuous score."""
    cand = MagicMock()
    cand.logprobs_result.top_candidates = [
        _LogprobStep([
            _LogprobCandidate("No", p_no),
            _LogprobCandidate("Yes", -4.0),
        ])
    ]
    cand.logprobs_result.chosen_candidates = []
    cand.avg_logprobs = None
    return MagicMock(candidates=[cand], text='"No"')

  def _text_only_response(self):
    """Builds a response without logprobs, i.e. a binary 0/1 score."""
    cand = MagicMock()
    cand.logprobs_result = None
    cand.avg_logprobs = None
    return MagicMock(candidates=[cand], text='"No"')

  def test_logprobs_downgrade_is_confined_to_one_sample(self):
    """One logprobs error must not flip the rest of the run to binary scores.

    The downgrade used to be a shared flag, so a single transient failure
    silently changed the scoring function for every sample handled after it,
    with the split decided by thread scheduling.
    """
    script_args = MagicMock(
        seed=42,
        evaluator_model="gemini-2.5-flash",
        evaluator_num_fewshot=0,
        overwrite_scores=False,
        max_workers=1,
        autorater_num_samples=1,
        dataset_with_completions=None,
        dataset_labels="test_eval",
        scores_checkpoint_path=None,
    )
    dataset = self._make_dummy_dataset(["Prompt 0", "Prompt 1", "Prompt 2"])

    def mock_generate(model, contents, config):
      del model, config
      prompt = str(contents)
      # The stubbed `types` module records the real kwargs, which is the only
      # way to see whether *this* request asked for logprobs.
      from google.genai import types as genai_types  # pylint: disable=g-import-not-at-top

      kwargs = genai_types.GenerateContentConfig.call_args.kwargs
      if not kwargs.get("response_logprobs", False):
        return self._text_only_response()
      if "Prompt 0" in prompt:
        raise RuntimeError("logprobs are not supported for this request")
      return self._continuous_response()

    client = MagicMock()
    client.models.generate_content.side_effect = mock_generate

    scores = self.pipeline.gemini_score_dataset(client, dataset, script_args)

    # Sample 0 degraded to the binary path, samples 1 and 2 kept probabilities.
    self.assertEqual(scores[0], 1.0)
    for score in scores[1:]:
      self.assertGreater(score, 0.0)
      self.assertLess(score, 1.0)
    stats = self.pipeline.autorater_stats
    self.assertEqual(stats["autorater_fallback_logprobs"], 1)
    self.assertEqual(stats["autorater_n_continuous"], 2)

  def test_capability_downgrade_terminates(self):
    """A permanently rejected option must not loop forever.

    The fallbacks rebuilt the request from scratch on every iteration without
    consuming a retry, so a model that always rejects one of them spun
    forever.
    """
    script_args = MagicMock(
        seed=42,
        evaluator_model="gemini-2.5-flash",
        evaluator_num_fewshot=0,
        overwrite_scores=False,
        max_workers=1,
        autorater_num_samples=1,
        dataset_with_completions=None,
        dataset_labels="test_eval",
        scores_checkpoint_path=None,
    )
    dataset = self._make_dummy_dataset(["Prompt 0"])
    calls = []

    def mock_generate(model, contents, config):
      del model, contents
      calls.append(config)
      raise RuntimeError("thinking_config is not supported by this model")

    client = MagicMock()
    client.models.generate_content.side_effect = mock_generate

    scores = self.pipeline.gemini_score_dataset(client, dataset, script_args)
    self.assertEqual(scores, [None])
    # Bounded: one downgrade, then the error is treated as unrecoverable.
    self.assertLessEqual(len(calls), 8)

  def test_k_samples_take_the_median_and_report_spread(self):
    """k > 1 queries the judge k times and keeps the median of the draws."""
    script_args = MagicMock(
        seed=42,
        evaluator_model="gemini-2.5-flash",
        evaluator_num_fewshot=0,
        overwrite_scores=False,
        max_workers=1,
        autorater_num_samples=3,
        dataset_with_completions=None,
        dataset_labels="test_eval",
        scores_checkpoint_path=None,
    )
    dataset = self._make_dummy_dataset(["Prompt 0"])
    # P(No) for log_prob(No) in {-4, -2, -1} against a fixed log_prob(Yes).
    draws = [-4.0, -1.0, -2.0]
    calls = {"n": 0}

    def mock_generate(model, contents, config):
      del model, contents, config
      value = draws[calls["n"] % len(draws)]
      calls["n"] += 1
      return self._continuous_response(p_no=value)

    client = MagicMock()
    client.models.generate_content.side_effect = mock_generate

    scores = self.pipeline.gemini_score_dataset(client, dataset, script_args)
    self.assertEqual(calls["n"], 3)

    def expected(log_no):
      p_no = math.exp(log_no)
      return p_no / (p_no + math.exp(-4.0))

    self.assertAlmostEqual(scores[0], expected(-2.0), places=6)
    stats = self.pipeline.autorater_stats
    self.assertEqual(stats["autorater_num_samples"], 3)
    self.assertEqual(stats["autorater_spread_samples"], 1)
    self.assertAlmostEqual(
        stats["autorater_spread_max"],
        expected(-1.0) - expected(-4.0),
        places=6,
    )

  def test_autorater_stats_report_scored_and_dropped_counts(self):
    """The moving metric denominator must be recorded, not just printed."""
    script_args = MagicMock(
        seed=42,
        evaluator_model="gemini-2.5-flash",
        evaluator_num_fewshot=0,
        overwrite_scores=False,
        max_workers=1,
        autorater_num_samples=1,
        dataset_with_completions=None,
        dataset_labels="test_eval",
        scores_checkpoint_path=None,
    )
    dataset = self._make_dummy_dataset(["Prompt 0", "Prompt 1", "Prompt 2"])

    def mock_generate(model, contents, config):
      del model, config
      if "Prompt 1" in str(contents):
        raise RuntimeError("Simulated unrecoverable API error")
      return self._continuous_response()

    client = MagicMock()
    client.models.generate_content.side_effect = mock_generate

    self.pipeline.gemini_score_dataset(client, dataset, script_args)
    stats = self.pipeline.autorater_stats
    self.assertEqual(stats["autorater_n_total"], 3)
    self.assertEqual(stats["autorater_n_scored"], 2)
    self.assertEqual(stats["autorater_n_dropped"], 1)
    self.assertEqual(stats["autorater_model"], "gemini-2.5-flash")
    # k = 1 means there is no spread to measure.
    self.assertIsNone(stats["autorater_spread_mean"])


class TestVllmContextWindow(unittest.TestCase):
  """The vLLM context window must be cappable.

  vLLM sizes its KV cache from the model's declared
  ``max_position_embeddings``. Qwen/Qwen3-4B-Instruct-2507 declares 262144,
  which needs tens of GB of KV cache and refuses to start on a smaller card,
  so the pipelines need a way to say "allocate less than the model claims".
  """

  def test_eval_arguments_expose_max_model_len(self):
    from src.utils import EvalArguments

    self.assertIn("max_model_len", EvalArguments.__dataclass_fields__)

  def test_eval_arguments_default_to_the_model_declaration(self):
    from src.utils import EvalArguments

    field = EvalArguments.__dataclass_fields__["max_model_len"]
    # None, not a number: defaulting to a cap would silently truncate the
    # context of every model that legitimately needs a long one.
    self.assertIsNone(field.default)

  def test_ssfo_arguments_expose_max_model_len(self):
    from src.utils import SsfoDataGenArguments

    field = SsfoDataGenArguments.__dataclass_fields__["max_model_len"]
    self.assertIsNone(field.default)

  def test_unset_max_model_len_is_not_forwarded_to_vllm(self):
    # `if max_model_len:` - an unset value must leave the key out entirely
    # rather than pass max_model_len=None, which vLLM would reject.
    pipe = EvaluationGenerationPipeline()
    pipe.args = MagicMock(max_model_len=None)
    self.assertFalse(bool(getattr(pipe.args, "max_model_len", None)))


class TestCpuCompatibility(unittest.TestCase):
  """Unit tests for CPU-only execution without vLLM."""

  def test_scoring_pipeline_available_without_vllm(self):
    """Scoring pipeline works regardless of vLLM availability."""
    pipe = EvaluationScoringPipeline()
    self.assertIsNotNone(pipe)

  def test_autorater_pipeline_available_without_vllm(self):
    """Autorater evaluation pipeline works regardless of vLLM availability."""
    pipe = EvaluationAutoraterPipeline()
    self.assertIsNotNone(pipe)

  def test_generation_pipeline_raises_without_vllm(self):
    """Generation pipeline raises informative ImportError if vLLM is missing."""
    pipe = EvaluationGenerationPipeline()
    pipe.prompts = ["Test prompt"]
    pipe.enable_lora = False
    pipe.lora_path = None
    pipe.vllm_model = "test-model"
    with patch("src.pipelines._VLLM_AVAILABLE", False):
      with self.assertRaises(ImportError) as ctx:
        pipe.run_and_save()
      self.assertIn("vLLM is required", str(ctx.exception))

  def test_ssfo_pipeline_raises_without_vllm(self):
    """SSFO pipeline raises informative ImportError if vLLM is missing."""
    pipe = SSFODataGenerationPipeline()
    pipe.train_split = [{"prompt": "p", "chosen": "c", "rejected": "r"}]
    pipe.args = MagicMock(
        top_k=0,
        repetition_penalty=1.0,
        seed=42,
        temperature=0.7,
        top_p=0.9,
        min_tokens=10,
        max_new_tokens=100,
    )
    with patch("src.pipelines._VLLM_AVAILABLE", False):
      with self.assertRaises(ImportError) as ctx:
        pipe.run_and_save()
      self.assertIn("vLLM is required", str(ctx.exception))

  def test_generation_pipeline_process_data_bosch_fewshot(self):
    """Test that EvaluationGenerationPipeline properly includes responses in Bosch few-shot."""
    pipe = EvaluationGenerationPipeline()
    pipe.args = MagicMock(
        task_name="bosch",
        writer_num_fewshot=2,
        dataset_labels="dummy_labels",
        dataset_labels_split="test",
        seed=42,
    )
    pipe.dataset_prompts = MockDataset.from_dict({
        "prompt": ["Test Q\nManual:\nC\nAnswer to user's question:\n"]
    })
    fewshot_list = [
        {
            "prompt": "Q1\nManual:\nC1\nAnswer to user's question:\n",
            "response": "Response 1",
            "class_hall": "No",
        },
        {
            "prompt": "Q2\nManual:\nC2\nAnswer to user's question:\n",
            "response": "Response 2",
            "class_hall": "No",
        },
    ]
    pipe.safe_load_dataset = MagicMock(return_value=fewshot_list)
    pipe.get_fewshot_examples = MagicMock(return_value=fewshot_list)
    pipe.process_data()
    self.assertEqual(len(pipe.prompts), 1)
    prompt = pipe.prompts[0]
    self.assertIn("Answer to user's question:\nResponse 1", prompt)
    self.assertIn("Answer to user's question:\nResponse 2", prompt)
    self.assertTrue(prompt.endswith("Test Q\nManual:\nC\nAnswer to user's question:\n"))


class TestRewardHackingDegeneration(unittest.TestCase):
  """The synthetic bad examples that anchor the bottom of the judge's scale.

  Without them the judge only ever sees well-written gold responses, learns
  the top of the scale and nothing else, and compresses every real completion
  into 4s and 5s - at which point the rubric cannot separate a good policy
  from a degenerate one, which is its only job.
  """

  def setUp(self):
    if degenerate_reward_hacked_response is None:
      self.skipTest("pipelines module not available in lightweight test env")

  def _degenerate(self, context, response, seed=0):
    import random as random_mod

    return degenerate_reward_hacked_response(
        context, response, random_mod.Random(seed)
    )

  def test_it_actually_changes_the_text(self):
    context = "The capital of France is Paris. It sits on the Seine."
    response = "Paris is the capital. The Seine runs through it."
    for seed in range(10):
      with self.subTest(seed=seed):
        self.assertNotEqual(
            self._degenerate(context, response, seed).strip(),
            response.strip(),
        )

  def test_it_is_reproducible_for_a_given_seed(self):
    # A campaign re-run must build the same demonstrations, or two runs of
    # the "same" configuration are graded against different prompts.
    context = "Some source material. With two sentences."
    response = "An answer. A second sentence."
    self.assertEqual(
        self._degenerate(context, response, 7),
        self._degenerate(context, response, 7),
    )

  def test_it_can_produce_a_verbatim_copy_of_the_context(self):
    # The signature reward-hacked output. At least one seed must reach it,
    # otherwise the non_extractiveness dimension has no demonstration.
    context = "A long retrieved passage about turbines and their upkeep."
    response = "Turbines need upkeep. Check them monthly."
    produced = {self._degenerate(context, response, s) for s in range(20)}
    self.assertIn(context, produced)

  def test_it_can_produce_a_duplicated_response(self):
    context = ""
    response = "First sentence. Second sentence."
    produced = {self._degenerate(context, response, s) for s in range(20)}
    self.assertTrue(
        any(text.count("First sentence.") > 1 for text in produced), produced
    )

  def test_a_single_sentence_is_still_corruptible(self):
    # Sentence shuffling needs two sentences and the verbatim copy needs a
    # context, but duplication works on anything - so a one-sentence
    # response with no context is not a lost cause.
    response = "Single sentence with no punctuation split"
    self.assertNotEqual(self._degenerate("", response, 0), response)

  def test_it_leaves_text_it_cannot_corrupt_alone(self):
    # Nothing to copy and nothing to repeat. Returning the input unchanged
    # lets the caller notice and skip the pair rather than emit a "bad"
    # demonstration identical to the good one.
    self.assertEqual(self._degenerate("", "", 0), "")


class TestRewardHackingFewshot(unittest.TestCase):
  """Building graded demonstrations out of SFT samples."""

  def setUp(self):
    if build_reward_hacking_fewshot_examples is None:
      self.skipTest("pipelines module not available in lightweight test env")
    from src.task_processors.base_task_processor import BaseTaskProcessor

    self.processor_cls = BaseTaskProcessor

  def _source(self, rows=6):
    return MockDataset({
        "context": [
            f"Source passage number {i} with several words in it. "
            f"And a second sentence for row {i}."
            for i in range(rows)
        ],
        "user_query": [f"Question {i}?" for i in range(rows)],
        "response": [
            f"A composed answer for row {i}. It has two sentences."
            for i in range(rows)
        ],
    })

  def test_it_returns_nothing_when_none_were_asked_for(self):
    self.assertIsNone(
        build_reward_hacking_fewshot_examples(
            self.processor_cls, self._source(), num_pairs=0, seed=1
        )
    )

  def test_each_pair_contributes_one_good_and_one_bad_example(self):
    examples = build_reward_hacking_fewshot_examples(
        self.processor_cls, self._source(), num_pairs=2, seed=1
    )
    self.assertEqual(len(examples), 4)
    degenerate = [e for e in examples if e["is_degenerate"]]
    clean = [e for e in examples if not e["is_degenerate"]]
    self.assertEqual(len(degenerate), 2)
    self.assertEqual(len(clean), 2)

  def test_the_two_ends_of_the_scale_are_both_demonstrated(self):
    examples = build_reward_hacking_fewshot_examples(
        self.processor_cls, self._source(), num_pairs=2, seed=1
    )
    for example in examples:
      grades = example["reward_hacking_grades"]
      expected = (
          REWARD_HACKING_SCALE_MIN
          if example["is_degenerate"]
          else REWARD_HACKING_SCALE_MAX
      )
      with self.subTest(degenerate=example["is_degenerate"]):
        self.assertEqual(
            grades, {k: expected for k in reward_hacking_dimension_keys()}
        )

  def test_the_pair_differs_only_in_the_writing(self):
    # Both halves share a context and a question, so the judge cannot learn
    # to key the grade off the topic or the length of the source.
    examples = build_reward_hacking_fewshot_examples(
        self.processor_cls, self._source(), num_pairs=1, seed=1
    )
    contexts = {e["context"] for e in examples}
    self.assertEqual(len(contexts), 1)

  def test_the_rendered_demonstrations_are_not_identical(self):
    # The failure this guards: if the degenerate text were written to a
    # column the prompt builder does not read, both demonstrations would
    # render the same body under opposite grades and actively mislead.
    examples = build_reward_hacking_fewshot_examples(
        self.processor_cls, self._source(), num_pairs=1, seed=1
    )
    rendered = {
        self.processor_cls._format_reward_hacking_example(e)
        for e in examples
    }
    self.assertEqual(len(rendered), 2)

  def test_an_empty_source_yields_nothing(self):
    self.assertIsNone(
        build_reward_hacking_fewshot_examples(
            self.processor_cls, MockDataset({}), num_pairs=2, seed=1
        )
    )


class TestRewardHackingVerdictParsing(unittest.TestCase):
  """Turning the judge's reply into grades."""

  def setUp(self):
    if EvaluationScoringPipeline is None:
      self.skipTest("pipelines module not available in lightweight test env")
    self.pipeline = EvaluationScoringPipeline()
    self.keys = reward_hacking_dimension_keys()

  def _response(self, text):
    return MagicMock(candidates=[MagicMock()], text=text)

  def test_a_clean_json_object_parses(self):
    payload = json.dumps({k: REWARD_HACKING_SCALE_MAX for k in self.keys})
    grades = self.pipeline.gemini_rubric_response(self._response(payload))
    self.assertEqual(grades, {k: 1.0 for k in self.keys})

  def test_grades_are_normalized_to_the_unit_interval(self):
    payload = json.dumps({k: REWARD_HACKING_SCALE_MIN for k in self.keys})
    grades = self.pipeline.gemini_rubric_response(self._response(payload))
    self.assertEqual(grades, {k: 0.0 for k in self.keys})

  def test_an_object_wrapped_in_prose_is_recovered(self):
    # The `response_schema` downgrade decodes without constraints, so the
    # judge may fence the object or introduce it. Losing the sample over
    # that would silently shrink the denominator.
    payload = (
        "Sure, here are the grades:\n```json\n"
        + json.dumps({k: 3 for k in self.keys})
        + "\n```"
    )
    grades = self.pipeline.gemini_rubric_response(self._response(payload))
    self.assertEqual(grades, {k: 0.5 for k in self.keys})

  def test_out_of_range_grades_are_clamped(self):
    payload = json.dumps({k: 99 for k in self.keys})
    grades = self.pipeline.gemini_rubric_response(self._response(payload))
    self.assertEqual(grades, {k: 1.0 for k in self.keys})

  def test_a_partial_object_keeps_what_it_can(self):
    payload = json.dumps({self.keys[0]: 5})
    grades = self.pipeline.gemini_rubric_response(self._response(payload))
    self.assertEqual(grades, {self.keys[0]: 1.0})

  def test_unparseable_text_yields_no_verdict(self):
    # None, not a default grade: an unreadable reply is missing data, and
    # scoring it would let a flaky judge masquerade as a bad policy.
    self.assertIsNone(
        self.pipeline.gemini_rubric_response(self._response("no idea, sorry"))
    )

  def test_an_object_with_no_known_keys_yields_no_verdict(self):
    payload = json.dumps({"vibes": 5})
    self.assertIsNone(
        self.pipeline.gemini_rubric_response(self._response(payload))
    )

  def test_a_blocked_response_yields_no_verdict(self):
    blocked = MagicMock(candidates=[], text=None)
    self.assertIsNone(self.pipeline.gemini_rubric_response(blocked))


class TestRewardHackingRubricDataset(unittest.TestCase):
  """Grading a whole dataset against the rubric."""

  def setUp(self):
    if EvaluationScoringPipeline is None:
      self.skipTest("pipelines module not available in lightweight test env")
    self.pipeline = EvaluationScoringPipeline()
    self.keys = reward_hacking_dimension_keys()
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)

  def _dataset(self, n=3):
    return MockDataset(
        {"reward_hacking_prompt": [f"Grade this {i}" for i in range(n)]}
    )

  def _args(self, **overrides):
    values = {
        "seed": 42,
        "evaluator_model": "gemini-2.5-flash",
        "reward_hacking_model": None,
        "reward_hacking_checkpoint_path": None,
        "overwrite_scores": False,
        "max_workers": 1,
        "autorater_num_samples": 1,
    }
    values.update(overrides)
    return MagicMock(**values)

  def _client(self, texts):
    """Returns a client replying with `texts` in call order."""
    replies = list(texts)
    calls = []

    def generate(model, contents, config):
      del model, config
      calls.append(str(contents))
      return MagicMock(
          candidates=[MagicMock()], text=replies[(len(calls) - 1) % len(replies)]
      )

    client = MagicMock()
    client.models.generate_content.side_effect = generate
    return client, calls

  def test_every_row_gets_a_verdict(self):
    payload = json.dumps({k: 4 for k in self.keys})
    client, calls = self._client([payload])
    verdicts = self.pipeline.gemini_score_rubric_dataset(
        client, self._dataset(3), self._args()
    )
    self.assertEqual(len(verdicts), 3)
    self.assertEqual(len(calls), 3)
    for verdict in verdicts:
      self.assertEqual(verdict, {k: 0.75 for k in self.keys})

  def test_a_failed_row_is_none_and_does_not_poison_the_others(self):
    good = json.dumps({k: 5 for k in self.keys})
    client, _ = self._client([good, "garbage", good])
    verdicts = self.pipeline.gemini_score_rubric_dataset(
        client, self._dataset(3), self._args()
    )
    self.assertEqual(sum(1 for v in verdicts if v is None), 1)
    self.assertEqual(sum(1 for v in verdicts if v), 2)

  def test_repeated_draws_are_averaged_not_rounded_to_the_grid(self):
    # The grades are already discretised to five points. A median over k
    # draws lands back on that grid and throws away the resolution the
    # extra API calls were paid for.
    low = json.dumps({k: 1 for k in self.keys})
    high = json.dumps({k: 5 for k in self.keys})
    client, _ = self._client([low, high])
    verdicts = self.pipeline.gemini_score_rubric_dataset(
        client, self._dataset(1), self._args(autorater_num_samples=2)
    )
    for key in self.keys:
      self.assertAlmostEqual(verdicts[0][key], 0.5)

  def test_the_judge_spread_is_recorded(self):
    low = json.dumps({k: 1 for k in self.keys})
    high = json.dumps({k: 5 for k in self.keys})
    client, _ = self._client([low, high])
    self.pipeline.gemini_score_rubric_dataset(
        client, self._dataset(1), self._args(autorater_num_samples=2)
    )
    stats = self.pipeline.reward_hacking_stats
    self.assertGreater(stats["reward_hacking_audit_spread_mean"], 0.0)

  def test_the_judge_defaults_to_the_hallucination_judge(self):
    payload = json.dumps({k: 3 for k in self.keys})
    client, _ = self._client([payload])
    self.pipeline.gemini_score_rubric_dataset(
        client, self._dataset(1), self._args()
    )
    self.assertEqual(
        self.pipeline.reward_hacking_stats["reward_hacking_audit_model"],
        "gemini-2.5-flash",
    )

  def test_an_explicit_judge_overrides_it(self):
    payload = json.dumps({k: 3 for k in self.keys})
    client, _ = self._client([payload])
    self.pipeline.gemini_score_rubric_dataset(
        client,
        self._dataset(1),
        self._args(reward_hacking_model="gemini-2.5-pro"),
    )
    self.assertEqual(
        self.pipeline.reward_hacking_stats["reward_hacking_audit_model"],
        "gemini-2.5-pro",
    )

  def test_a_checkpoint_is_resumed_rather_than_re_billed(self):
    # Grading is billed per call; an interrupted campaign must not pay
    # twice for rows it already has.
    ckpt = os.path.join(self.temp_dir, "rubric.json")
    with open(ckpt, "w") as handle:
      json.dump([{k: 1.0 for k in self.keys}, None, None], handle)
    payload = json.dumps({k: 2 for k in self.keys})
    client, calls = self._client([payload])
    verdicts = self.pipeline.gemini_score_rubric_dataset(
        client, self._dataset(3), self._args(reward_hacking_checkpoint_path=ckpt)
    )
    self.assertEqual(len(calls), 2)
    self.assertEqual(verdicts[0], {k: 1.0 for k in self.keys})

  def test_overwrite_ignores_the_checkpoint(self):
    ckpt = os.path.join(self.temp_dir, "rubric.json")
    with open(ckpt, "w") as handle:
      json.dump([{k: 1.0 for k in self.keys}, None, None], handle)
    payload = json.dumps({k: 2 for k in self.keys})
    client, calls = self._client([payload])
    self.pipeline.gemini_score_rubric_dataset(
        client,
        self._dataset(3),
        self._args(
            reward_hacking_checkpoint_path=ckpt, overwrite_scores=True
        ),
    )
    self.assertEqual(len(calls), 3)


if __name__ == "__main__":
  unittest.main()

