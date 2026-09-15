"""Tests for the multi-policy evaluation stage and the robustness helpers."""

import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from src.orchestrator import eval_metrics
from src.orchestrator.config import CampaignConfig
from src.orchestrator.model_manager import ModelManager
from src.orchestrator.process import ProcessOutcome
from src.orchestrator.reporter import CampaignReporter
from src.orchestrator.retry import RetryError
from src.orchestrator.retry import is_retryable
from src.orchestrator.retry import run_with_retries
from src.orchestrator.stages.base import CampaignContext
from src.orchestrator.stages.eval_stage import EvalStage
from src.orchestrator.stages.eval_stage import compute_deltas
from src.orchestrator.state import CampaignState
from src.orchestrator.state import StageResult
from src.orchestrator.state import StageStatus
from src.orchestrator.sweep_controller import SweepController


def _context(config, state, state_path=None):
  return CampaignContext(
      config=config,
      state=state,
      sweep_controller=SweepController(dry_run=True),
      model_manager=ModelManager(dry_run=True),
      state_path=state_path,
  )


class TestEvalTargets(unittest.TestCase):
  """Evaluation covers the trained policies and never the base model."""

  def _stage(self, sft=None, perl=None, state_path=None):
    config = CampaignConfig.create_default(task_name="ragtruth", dry_run=True)
    state = CampaignState(campaign_id="c", task_name="ragtruth")
    if sft:
      state.stages["sft"] = StageResult(
          status=StageStatus.COMPLETED, model_repo_id=sft
      )
    if perl:
      state.stages["perl"] = StageResult(
          status=StageStatus.COMPLETED, model_repo_id=perl
      )
    return EvalStage(_context(config, state, state_path)), state

  def test_both_policies_are_evaluated_in_order(self):
    stage, _ = self._stage(sft="u/sft_ckpt", perl="u/perl_ckpt")
    self.assertEqual(
        stage.resolve_targets(),
        [("sft", "u/sft_ckpt"), ("perl", "u/perl_ckpt")],
    )

  def test_base_model_is_never_a_target(self):
    stage, _ = self._stage(sft="u/sft_ckpt")
    targets = stage.resolve_targets()
    self.assertEqual(targets, [("sft", "u/sft_ckpt")])
    self.assertNotIn(
        stage.config.base_model, [repo for _, repo in targets]
    )

  def test_no_policy_raises_instead_of_scoring_the_base_model(self):
    stage, _ = self._stage()
    with self.assertRaises(ValueError) as ctx:
      stage.resolve_targets()
    self.assertIn("no trained policy", str(ctx.exception))

  def test_duplicate_repo_is_evaluated_once(self):
    stage, _ = self._stage(sft="u/same", perl="u/same")
    self.assertEqual(stage.resolve_targets(), [("sft", "u/same")])

  def test_dry_run_produces_namespaced_metrics_and_deltas(self):
    stage, _ = self._stage(sft="u/sft_ckpt", perl="u/perl_ckpt")
    result = stage.execute()
    self.assertEqual(result.status, StageStatus.COMPLETED)
    self.assertEqual(result.model_repo_id, "u/perl_ckpt")
    self.assertIn("sft/hallucination_rate", result.metrics)
    self.assertIn("perl/hallucination_rate", result.metrics)
    # Lower hallucination is better, so the improvement must be positive.
    self.assertGreater(result.metrics["delta/hallucination_rate"], 0)

  def test_already_scored_target_is_not_re_evaluated(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      state_path = os.path.join(temp_dir, "state.json")
      stage, state = self._stage(
          sft="u/sft_ckpt", perl="u/perl_ckpt", state_path=state_path
      )
      state.stages["eval"] = StageResult(
          status=StageStatus.RUNNING,
          metrics={"sft/hallucination_rate": 0.11},
      )
      previous = stage.previous_metrics()
      self.assertTrue(stage.is_target_scored("sft", previous))
      self.assertFalse(stage.is_target_scored("perl", previous))

  def test_partial_metrics_are_persisted(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      state_path = os.path.join(temp_dir, "state.json")
      stage, _ = self._stage(sft="u/sft_ckpt", state_path=state_path)
      stage.record_partial_metrics({"sft/hallucination_rate": 0.1})
      with open(state_path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)
      self.assertEqual(
          saved["stages"]["eval"]["metrics"]["sft/hallucination_rate"], 0.1
      )


class TestDeltaSigns(unittest.TestCase):
  """A positive delta must always mean 'PE-RL is better'."""

  def test_lower_is_better_metric_is_inverted(self):
    deltas = compute_deltas(
        {"sft/hallucination_rate": 0.10, "perl/hallucination_rate": 0.04}
    )
    self.assertAlmostEqual(deltas["delta/hallucination_rate"], 0.06)

  def test_higher_is_better_metric_keeps_its_sign(self):
    deltas = compute_deltas(
        {"sft/faithfulness_rate": 0.90, "perl/faithfulness_rate": 0.95}
    )
    self.assertAlmostEqual(deltas["delta/faithfulness_rate"], 0.05)

  def test_missing_counterpart_produces_no_delta(self):
    self.assertEqual(compute_deltas({"perl/bertscore_f1": 0.9}), {})

  def test_non_numeric_values_are_skipped(self):
    self.assertEqual(
        compute_deltas({"sft/model": "a", "perl/model": "b"}), {}
    )


class TestEvalMetricsView(unittest.TestCase):
  """Report/dashboard helpers must read the namespaced metrics."""

  def test_headline_prefers_perl(self):
    value = eval_metrics.headline_metric(
        {"sft/hallucination_rate": 0.11, "perl/hallucination_rate": 0.04}
    )
    self.assertAlmostEqual(value, 0.04)

  def test_headline_falls_back_to_sft_then_legacy_keys(self):
    self.assertAlmostEqual(
        eval_metrics.headline_metric({"sft/hallucination_rate": 0.11}), 0.11
    )
    self.assertAlmostEqual(
        eval_metrics.headline_metric({"hallucination_rate": 0.2}), 0.2
    )
    self.assertIsNone(eval_metrics.headline_metric({}))

  def test_comparison_rows_pivot_metrics(self):
    rows = eval_metrics.comparison_rows({
        "sft/hallucination_rate": 0.11,
        "perl/hallucination_rate": 0.04,
        "delta/hallucination_rate": 0.07,
        "sft/bertscore_f1": 0.87,
        "perl/bertscore_f1": 0.89,
    })
    names = [row[0] for row in rows]
    self.assertEqual(names, ["hallucination_rate", "bertscore_f1"])
    self.assertEqual(rows[0][1], [0.11, 0.04])
    self.assertAlmostEqual(rows[0][2], 0.07)


class TestReportComparisonTable(unittest.TestCase):
  """The markdown report shows both policies side by side."""

  def test_report_contains_both_columns_and_delta(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      config = CampaignConfig.create_default(task_name="ragtruth", dry_run=True)
      config.reporting.reports_dir = temp_dir
      state = CampaignState(campaign_id="camp", task_name="ragtruth")
      state.stages["eval"] = StageResult(
          status=StageStatus.COMPLETED,
          model_repo_id="u/perl_ckpt",
          metrics={
              "sft/hallucination_rate": 0.11,
              "perl/hallucination_rate": 0.04,
              "delta/hallucination_rate": 0.07,
          },
      )
      path = CampaignReporter(config, state).generate_markdown_report()
      with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
      self.assertIn("PE-RL", content)
      self.assertIn("SFT", content)
      self.assertIn("+0.0700", content)


class TestRetryPolicy(unittest.TestCase):
  """Transient failures are retried; deterministic ones are not."""

  def test_succeeds_after_transient_failures(self):
    calls = []

    def _flaky():
      calls.append(1)
      if len(calls) < 3:
        raise RuntimeError("503 Service Unavailable")
      return "ok"

    result = run_with_retries(
        _flaky, "flaky call", attempts=4, base_delay_s=0.0
    )
    self.assertEqual(result, "ok")
    self.assertEqual(len(calls), 3)

  def test_gives_up_after_the_last_attempt(self):
    calls = []

    def _always_fails():
      calls.append(1)
      raise RuntimeError("500 boom")

    with self.assertRaises(RuntimeError):
      run_with_retries(
          _always_fails, "doomed", attempts=3, base_delay_s=0.0
      )
    self.assertEqual(len(calls), 3)

  def test_deterministic_failures_are_not_retried(self):
    calls = []

    def _bad_flags():
      calls.append(1)
      raise RuntimeError("unrecognized arguments: --vocab_size")

    with self.assertRaises(RuntimeError):
      run_with_retries(
          _bad_flags, "materialization", attempts=4, base_delay_s=0.0
      )
    self.assertEqual(len(calls), 1)

  def test_interruption_is_not_retried(self):
    calls = []

    def _interrupted():
      calls.append(1)
      raise RuntimeError("Evaluation was interrupted before it completed.")

    with self.assertRaises(RuntimeError):
      run_with_retries(
          _interrupted, "evaluation", attempts=4, base_delay_s=0.0
      )
    self.assertEqual(len(calls), 1)

  def test_stop_request_short_circuits_the_backoff(self):
    calls = []

    def _flaky():
      calls.append(1)
      raise RuntimeError("timeout")

    with self.assertRaises(RuntimeError):
      run_with_retries(
          _flaky,
          "stopped call",
          attempts=5,
          base_delay_s=10.0,
          stop_requested=lambda: True,
      )
    self.assertEqual(len(calls), 1)

  def test_missing_metric_error_is_terminal(self):
    self.assertFalse(
        is_retryable(ValueError("No run in sweep x logged the metric 'y'"))
    )
    self.assertTrue(is_retryable(RuntimeError("502 Bad Gateway")))


class TestFailurePreservesProgress(unittest.TestCase):
  """A failed stage must not erase the sweep it already paid for."""

  def test_sweep_id_and_metrics_survive_a_failure(self):
    state = CampaignState(campaign_id="c", task_name="ragtruth")
    state.stages["sft"] = StageResult(
        status=StageStatus.RUNNING,
        sweep_id="sweep_42",
        metrics={"partial": 1.0},
    )
    state.mark_stage_failed("sft", "HF Hub 500")
    self.assertEqual(state.stages["sft"].status, StageStatus.FAILED)
    self.assertEqual(state.stages["sft"].sweep_id, "sweep_42")
    self.assertEqual(state.stages["sft"].metrics, {"partial": 1.0})
    self.assertIn("500", state.stages["sft"].error_message)

  def test_failure_without_a_prior_result_is_safe(self):
    state = CampaignState(campaign_id="c", task_name="ragtruth")
    state.mark_stage_failed("rm", "boom")
    self.assertEqual(state.stages["rm"].status, StageStatus.FAILED)


class TestHubPushRetries(unittest.TestCase):
  """Push operations to Hugging Face Hub must retry on transient errors."""

  def test_upload_local_checkpoint_retries_transient_error(self):
    mock_api = MagicMock()
    upload_attempts = []

    def _fake_upload(**kwargs):
      upload_attempts.append(kwargs)
      if len(upload_attempts) < 3:
        raise RuntimeError("503 Service Unavailable")
      return None

    mock_api.upload_folder.side_effect = _fake_upload

    hf_mod = types.ModuleType("huggingface_hub")
    hf_mod.HfApi = MagicMock(return_value=mock_api)

    with patch.dict(sys.modules, {"huggingface_hub": hf_mod}):
      mgr = ModelManager(user="leobianco", dry_run=False)
      mgr.robustness.retry_base_delay_s = 0.0
      mgr.robustness.api_attempts = 4

      repo = mgr.upload_local_checkpoint(
          folder_path="/tmp/fake_ckpt",
          repo_id="leobianco/fake_model",
      )
      self.assertEqual(repo, "leobianco/fake_model")
      self.assertEqual(len(upload_attempts), 3)

  @patch("src.orchestrator.model_manager.stream_subprocess")
  def test_materialize_salvages_local_checkpoint_on_push_failure(
      self, mock_stream
  ):
    """If training completes and writes weights, salvage it on push error."""
    with tempfile.TemporaryDirectory() as temp_dir:
      mgr = ModelManager(user="leobianco", dry_run=False)
      mgr.robustness.retry_base_delay_s = 0.0
      mgr.robustness.materialize_attempts = 2

      mgr.upload_local_checkpoint = MagicMock(return_value="leobianco/salvaged")

      def _mock_training(cmd, **_kwargs):
        out_idx = cmd.index("--output_dir") + 1
        real_out = cmd[out_idx]
        os.makedirs(real_out, exist_ok=True)
        with open(os.path.join(real_out, "model.safetensors"), "wb") as f:
          f.write(b"checkpoint weights")
        return ProcessOutcome(returncode=1, interrupted=False)

      mock_stream.side_effect = _mock_training

      repo = mgr.materialize_and_push(
          stage_name="sft",
          task_name="ragtruth",
          base_model="google/gemma-4-E2B-it",
          best_params={"learning_rate": 0.001},
          seed=130104,
      )
      self.assertIn("ragtruth_SFT", repo)
      mgr.upload_local_checkpoint.assert_called_once()
      call_args = mgr.upload_local_checkpoint.call_args[0]
      self.assertTrue(os.path.exists(call_args[0]))
      self.assertEqual(call_args[1], repo)


if __name__ == "__main__":
  unittest.main()
