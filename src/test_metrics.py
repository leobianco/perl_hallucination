"""Unit tests for src.metrics module."""

import json
import math
import os
import shutil
import tempfile
from typing import Any
import unittest
from unittest.mock import MagicMock, patch

from src.metrics import (
    compute_bertscore,
    compute_conditional_perplexity,
    compute_distinct_n,
    compute_length_metrics,
    compute_repetition_rate,
    compute_rouge,
    compute_summary_statistics,
    GenerationMetricsEvaluator,
    tokenize_words,
)


class MockDataset:
  """Mock Dataset for testing when Hugging Face datasets is mocked or unavailable."""

  def __init__(self, data: dict[str, list[Any]]):
    self._data = dict(data)
    self.column_names = list(data.keys())

  def __getitem__(self, key: str):
    return self._data[key]

  def __len__(self):
    first_col = next(iter(self._data.values()))
    return len(first_col)

  def remove_columns(self, column_name: str):
    new_data = {k: v for k, v in self._data.items() if k != column_name}
    return MockDataset(new_data)

  def add_column(self, column_name: str, values: list[Any]):
    new_data = dict(self._data)
    new_data[column_name] = values
    return MockDataset(new_data)

  def select(self, indices: Any):
    new_data = {k: [v[i] for i in indices] for k, v in self._data.items()}
    return MockDataset(new_data)

  def shuffle(self, seed: int = 42):
    import random

    indices = list(range(len(self)))
    r = random.Random(seed)
    r.shuffle(indices)
    return self.select(indices)


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
        evaluator.find_reference_column(["prompt", "completion", "npov_response"]),
        "npov_response",
    )
    self.assertEqual(
        evaluator.find_reference_column(["prompt", "completion", "target"]),
        "target",
    )
    self.assertEqual(
        evaluator.find_reference_column(["prompt", "completion", "ground_truth"]),
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
      if max_samples is not None and max_samples > 0 and len(data_obj) > max_samples:
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
      self.assertEqual(final_dataset["completion"][orig_idx], f"Completion {orig_idx}")

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


if __name__ == "__main__":
  unittest.main()
