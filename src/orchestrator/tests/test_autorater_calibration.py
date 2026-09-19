"""Tests for autorater calibration and the threshold it hands downstream.

The properties worth pinning:

1. Calibration is measured with *exactly* the judge configuration the final
   scoring will use. A threshold fitted at 4-shot does not transfer to a
   judge given 2, so a drift between the two stages would silently score
   the campaign at an operating point nobody fitted.
2. The fitted threshold actually reaches the scoring command, survives into
   the state file, and is still used by a later ``--stages eval`` rerun.
3. A weak judge is loud but not fatal - warnings in the log and the report,
   campaign carries on.
"""

import json
import os
import tempfile
import unittest

from src.orchestrator import eval_metrics
from src.orchestrator import flavors
from src.orchestrator.cli import theme as theme_mod
from src.orchestrator.cli import wizard
from src.orchestrator.config import CampaignConfig, VALID_STAGES
from src.orchestrator.model_manager import ModelManager
from src.orchestrator.reporter import CampaignReporter
from src.orchestrator.stages.autorater_stage import AutoraterStage
from src.orchestrator.stages.base import CampaignContext
from src.orchestrator.stages.eval_stage import EvalStage
from src.orchestrator.state import CampaignState, StageResult, StageStatus
from src.orchestrator.sweep_controller import SweepController
from src.utils import autorater_eval_metrics_path


def _config(**kwargs):
  config = CampaignConfig.create_default(task_name="ragtruth", **kwargs)
  config.user = "leobianco"
  return config


def _context(config, state):
  return CampaignContext(
      config=config,
      state=state,
      sweep_controller=SweepController(dry_run=True),
      model_manager=ModelManager(dry_run=True),
  )


def _calibration(**overrides):
  values = {
      "roc_auc": 0.91,
      "best_threshold": 0.42,
      "tpr_at_best_threshold": 0.88,
      "fpr_at_best_threshold": 0.12,
      "accuracy_at_best_threshold": 0.88,
      "balanced_accuracy": 0.88,
      "precision_at_best_threshold": 0.87,
      "scored_samples": 450,
      "unscored_samples": 0,
      "degenerate_roc": False,
  }
  values.update(overrides)
  return values


def _calibrated_state(**overrides):
  """Returns a state whose autorater stage completed."""
  state = CampaignState(campaign_id="c", task_name="ragtruth")
  state.stages["autorater"] = StageResult(
      status=StageStatus.COMPLETED,
      best_metric_val=_calibration(**overrides)["roc_auc"],
      metrics={
          f"autorater/{k}": v for k, v in _calibration(**overrides).items()
      },
  )
  return state


class PipelinePositionTest(unittest.TestCase):
  """Where calibration sits in the campaign."""

  def test_calibration_runs_before_everything_else(self):
    plan = flavors.build_plan(_config())
    self.assertEqual(plan[0].stage_id, "autorater")

  def test_it_is_in_the_default_pipeline(self):
    self.assertIn("autorater", _config().stages)

  def test_the_three_stage_lists_agree(self):
    # config validates against one, the UI orders by another and the wizard
    # offers a third. They drifting apart is how a stage becomes
    # unselectable or gets sorted after 'eval'.
    self.assertEqual(theme_mod.PIPELINE_ORDER, list(VALID_STAGES))
    self.assertEqual(
        [key for key, _, _ in wizard.STAGE_CATALOGUE], list(VALID_STAGES)
    )

  def test_it_is_not_a_branched_kind(self):
    # One judge, one labelled set: calibrating per reward-model flavor
    # would pay Gemini twice for the same number.
    self.assertNotIn("autorater", flavors.BRANCHED_KINDS)
    config = _config(rm_dataset_flavors=["organic", "synthetic_struct"])
    ids = [p.stage_id for p in flavors.build_plan(config)]
    self.assertEqual(ids.count("autorater"), 1)

  def test_its_log_tag_is_distinct(self):
    tags = {flavors.log_tag(s) for s in VALID_STAGES}
    self.assertEqual(len(tags), len(VALID_STAGES), tags)


class CommandTest(unittest.TestCase):
  """The calibration invocation."""

  def setUp(self):
    self.config = _config()
    self.stage = AutoraterStage(_context(self.config, CampaignState(campaign_id="c", task_name="ragtruth")))
    self.cmd = self.stage.build_command()

  def _value(self, flag):
    return self.cmd[self.cmd.index(flag) + 1]

  def test_it_runs_the_autoratereval_mode(self):
    self.assertEqual(self._value("--mode"), "autoratereval")
    self.assertEqual(self._value("--evaluate_evaluator"), "True")

  def test_it_scores_against_the_human_labelled_set(self):
    self.assertEqual(
        self._value("--dataset_labels"), "leobianco/ragtruth_autorater"
    )
    self.assertEqual(self._value("--dataset_labels_split"), "test")

  def test_the_judge_matches_the_one_scoring_will_use(self):
    # The invariant that makes the fitted threshold transferable.
    eval_cmd = EvalStage(
        _context(self.config, CampaignState(campaign_id="c", task_name="ragtruth"))
    )._scoring_command("u/policy")

    def eval_value(flag):
      return eval_cmd[eval_cmd.index(flag) + 1]

    for flag in (
        "--evaluator_model",
        "--evaluator_num_fewshot",
        "--autorater_num_samples",
        "--use_gemini",
        "--seed",
    ):
      with self.subTest(flag=flag):
        self.assertEqual(self._value(flag), eval_value(flag))

  def test_the_default_fewshot_count_is_two(self):
    # Matches scripts/evaluator.sh, which is where the value was settled.
    self.assertEqual(self._value("--evaluator_num_fewshot"), "2")

  def test_it_names_no_policy_of_its_own(self):
    # Calibration runs before any policy exists; the flag is only present
    # because the argument parser requires it.
    self.assertEqual(
        self._value("--writer_model_lora"), self.config.base_model
    )


class ExecutionTest(unittest.TestCase):
  """Running the stage."""

  def test_it_is_skipped_when_there_is_no_evaluation_to_serve(self):
    config = _config()
    config.eval.enabled = False
    result = AutoraterStage(_context(config, CampaignState(campaign_id="c", task_name="ragtruth"))).execute()
    self.assertEqual(result.status, StageStatus.SKIPPED)

  def test_a_dry_run_produces_a_usable_calibration(self):
    config = _config(dry_run=True)
    result = AutoraterStage(_context(config, CampaignState(campaign_id="c", task_name="ragtruth"))).execute()
    self.assertEqual(result.status, StageStatus.COMPLETED)
    self.assertIn("autorater/best_threshold", result.metrics)
    self.assertIsNotNone(result.best_metric_val)


class CalibrationLoadingTest(unittest.TestCase):
  """Reading the pipeline's output back."""

  def setUp(self):
    self.config = _config()
    self.stage = AutoraterStage(_context(self.config, CampaignState(campaign_id="c", task_name="ragtruth")))

  def test_the_path_matches_what_the_pipeline_writes(self):
    self.assertEqual(
        self.stage.calibration_path(),
        autorater_eval_metrics_path(
            evaluator_model=self.config.eval.evaluator_model,
            dataset_labels="leobianco/ragtruth_autorater",
            evaluator_num_fewshot=self.config.eval.evaluator_num_fewshot,
            seed=self.config.eval.seed,
        ),
    )

  def test_a_missing_file_is_an_error_not_a_default(self):
    with self.assertRaisesRegex(RuntimeError, "wrote no metrics"):
      self.stage.load_calibration()

  def test_a_calibration_without_a_threshold_is_rejected(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      cwd = os.getcwd()
      try:
        os.chdir(temp_dir)
        path = self.stage.calibration_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
          json.dump({"roc_auc": 0.9, "best_threshold": None}, handle)
        with self.assertRaisesRegex(RuntimeError, "best_threshold"):
          self.stage.load_calibration()
      finally:
        os.chdir(cwd)


class QualityWarningTest(unittest.TestCase):
  """Flagging a judge that should not be trusted."""

  def setUp(self):
    self.config = _config()
    self.stage = AutoraterStage(_context(self.config, CampaignState(campaign_id="c", task_name="ragtruth")))

  def test_a_healthy_judge_raises_nothing(self):
    self.assertEqual(self.stage.quality_warnings(_calibration()), [])

  def test_an_auc_below_the_floor_is_flagged(self):
    warnings = self.stage.quality_warnings(_calibration(roc_auc=0.71))
    self.assertEqual(len(warnings), 1)
    self.assertIn("0.7100", warnings[0])
    self.assertIn("0.85", warnings[0])

  def test_the_floor_is_configurable(self):
    self.config.eval.min_autorater_auc = 0.5
    self.assertEqual(self.stage.quality_warnings(_calibration(roc_auc=0.71)), [])

  def test_a_degenerate_roc_curve_is_flagged(self):
    warnings = self.stage.quality_warnings(_calibration(degenerate_roc=True))
    self.assertEqual(len(warnings), 1)
    self.assertIn("single operating point", warnings[0])

  def test_a_weak_judge_does_not_stop_the_campaign(self):
    # Advisory by design: the rates are still computable, they just carry
    # more error than the report's precision suggests.
    config = _config(dry_run=True)
    config.eval.min_autorater_auc = 0.99
    result = AutoraterStage(_context(config, CampaignState(campaign_id="c", task_name="ragtruth"))).execute()
    self.assertEqual(result.status, StageStatus.COMPLETED)


class ThresholdHandoffTest(unittest.TestCase):
  """Getting the fitted threshold into the scoring command."""

  def test_evaluation_scores_at_the_fitted_threshold(self):
    config = _config()
    stage = EvalStage(_context(config, _calibrated_state()))
    self.assertEqual(stage.scoring_threshold(), 0.42)

  def test_the_scoring_command_carries_it(self):
    config = _config()
    cmd = EvalStage(
        _context(config, _calibrated_state())
    )._scoring_command("u/policy")
    self.assertEqual(cmd[cmd.index("--threshold") + 1], "0.42")

  def test_without_calibration_the_configured_constant_is_used(self):
    config = _config()
    stage = EvalStage(_context(config, CampaignState(campaign_id="c", task_name="ragtruth")))
    self.assertEqual(stage.scoring_threshold(), config.eval.threshold)

  def test_a_fitted_threshold_of_zero_is_not_mistaken_for_absent(self):
    # 0.0 is falsy; a truthiness check here would silently revert to the
    # configured constant and score the campaign at the wrong point.
    config = _config()
    stage = EvalStage(_context(config, _calibrated_state(best_threshold=0.0)))
    self.assertEqual(stage.scoring_threshold(), 0.0)

  def test_a_non_numeric_threshold_falls_back(self):
    config = _config()
    state = CampaignState(campaign_id="c", task_name="ragtruth")
    state.stages["autorater"] = StageResult(
        status=StageStatus.COMPLETED,
        metrics={"autorater/best_threshold": "not a number"},
    )
    stage = EvalStage(_context(config, state))
    self.assertEqual(stage.scoring_threshold(), config.eval.threshold)

  def test_the_provenance_is_stated_either_way(self):
    config = _config()
    calibrated = EvalStage(_context(config, _calibrated_state()))
    self.assertIn("calibration", calibrated.threshold_provenance())
    plain = EvalStage(_context(config, CampaignState(campaign_id="c", task_name="ragtruth")))
    self.assertIn("configuration", plain.threshold_provenance())


class ReportTest(unittest.TestCase):
  """What the reader of the final report sees."""

  def _report(self, **overrides):
    config = _config()
    state = _calibrated_state(**overrides)
    with tempfile.TemporaryDirectory() as temp_dir:
      config.reporting.reports_dir = temp_dir
      path = CampaignReporter(config, state).generate_markdown_report()
      with open(path, "r", encoding="utf-8") as handle:
        return handle.read()

  def test_the_calibration_has_its_own_section(self):
    content = self._report()
    self.assertIn("Autorater Calibration (Judge Quality)", content)
    self.assertIn("ROC-AUC", content)
    self.assertIn("Fitted threshold", content)

  def test_the_scorecard_shows_the_judge_quality(self):
    self.assertIn("judge roc_auc", self._report())

  def test_a_weak_judge_gets_a_caution_callout(self):
    config = _config()
    state = _calibrated_state(roc_auc=0.62)
    state.stages["autorater"].warnings = [
        AutoraterStage(_context(config, state)).quality_warnings(
            _calibration(roc_auc=0.62)
        )[0]
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
      config.reporting.reports_dir = temp_dir
      path = CampaignReporter(config, state).generate_markdown_report()
      with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
    self.assertIn("[!CAUTION]", content)
    self.assertIn("itself of doubtful quality", content)
    # Not filed under "reduced search", which is a different complaint.
    self.assertNotIn("### 1.1. Degradations", content)


class MetricsVocabularyTest(unittest.TestCase):
  """The shared rendering helpers."""

  def test_the_namespace_is_stripped(self):
    self.assertEqual(
        eval_metrics.autorater_calibration({"autorater/roc_auc": 0.9, "x": 1}),
        {"roc_auc": 0.9},
    )

  def test_absent_values_are_skipped_rather_than_dashed(self):
    rows = eval_metrics.format_autorater_rows({"roc_auc": 0.9})
    self.assertEqual(rows, [("ROC-AUC", "0.9000")])

  def test_the_threshold_keeps_its_full_precision(self):
    # A saturated judge's threshold is 0.9999887757936129; rounded to five
    # decimals it prints as 1.00000, which is unusable as a config value.
    rows = dict(
        eval_metrics.format_autorater_rows(
            {"best_threshold": 0.9999887757936129}
        )
    )
    self.assertEqual(rows["Fitted threshold"], "0.9999887757936129")


if __name__ == "__main__":
  unittest.main()
