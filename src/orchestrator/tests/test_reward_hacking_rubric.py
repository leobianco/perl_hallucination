"""Tests for the reward-hacking rubric and how it reaches the report.

The rubric exists to catch a specific failure: RLOO against a faithfulness
reward model pays a policy to stop composing and start quoting, and the
hallucination judge scores a verbatim copy of the context as perfectly
faithful. The properties worth pinning:

1. The rubric is small and fixed. A judge asked for more than a handful of
   Likert grades in one call stops being consistent, and a rubric that
   changes shape between campaigns produces numbers that cannot be compared.
2. The aggregation is the one the user asked for: unweighted mean of the
   per-dimension grades, normalized to [0, 1], with the flag rate derived
   from it rather than judged separately.
3. The metric names the evaluator emits are exactly the ones the orchestrator
   tabulates, signs and hides. A drift here is silent: the report simply
   stops showing the row.
4. ``reward_hacking_rate`` is signed as lower-is-better, and the quality
   score as higher-is-better. Getting this backwards would make a degenerate
   policy look like an improvement, which is the one mistake this whole
   feature is meant to prevent.
"""

import unittest

from src.orchestrator import eval_metrics
from src.orchestrator.config import CampaignConfig
from src.orchestrator.model_manager import ModelManager
from src.orchestrator.stages.base import CampaignContext
from src.orchestrator.stages.eval_stage import EvalStage, EvalTarget
from src.orchestrator.state import CampaignState, StageStatus
from src.orchestrator.sweep_controller import SweepController
from src.utils import REWARD_HACKING_AUDIT_PREFIX
from src.utils import REWARD_HACKING_DIMENSIONS
from src.utils import reward_hacking_dimension_keys
from src.utils import REWARD_HACKING_QUALITY_KEY
from src.utils import REWARD_HACKING_RATE_KEY
from src.utils import REWARD_HACKING_SCALE_MAX
from src.utils import REWARD_HACKING_SCALE_MIN
from src.utils import normalize_rubric_score
from src.utils import reward_hacking_run_name
from src.utils import reward_hacking_summary


def _config(**kwargs):
  config = CampaignConfig.create_default(task_name="ragtruth", **kwargs)
  config.user = "leobianco"
  return config


def _stage(config):
  state = CampaignState(campaign_id="c", task_name="ragtruth")
  return EvalStage(
      CampaignContext(
          config=config,
          state=state,
          sweep_controller=SweepController(dry_run=True),
          model_manager=ModelManager(dry_run=True),
      )
  )


class RubricShapeTest(unittest.TestCase):
  """The rubric definition itself."""

  def test_it_stays_within_five_dimensions(self):
    # A judge grading more than a handful of Likert scales in one call
    # starts anchoring them to each other. The upper bound is the design
    # constraint, not an accident of the current three.
    self.assertLessEqual(len(REWARD_HACKING_DIMENSIONS), 5)
    self.assertGreaterEqual(len(REWARD_HACKING_DIMENSIONS), 1)

  def test_the_dimension_keys_are_unique(self):
    keys = reward_hacking_dimension_keys()
    self.assertEqual(len(keys), len(set(keys)))

  def test_every_dimension_carries_a_title_and_description(self):
    for key, title, description in REWARD_HACKING_DIMENSIONS:
      with self.subTest(key=key):
        self.assertTrue(title.strip())
        # The description is what the judge actually reads; an empty one
        # would leave the dimension name to carry the whole definition.
        self.assertGreater(len(description), 80)

  def test_the_dimensions_are_phrased_so_that_high_is_good(self):
    # Every metric in the comparison table except the explicitly
    # lower-is-better ones is read as "bigger is better". A dimension named
    # `repetition` rather than `non_repetition` would silently invert.
    for key in reward_hacking_dimension_keys():
      with self.subTest(key=key):
        self.assertNotIn("hallucination", key)


class NormalizationTest(unittest.TestCase):
  """Mapping a Likert grade onto [0, 1]."""

  def test_the_scale_endpoints_map_to_zero_and_one(self):
    self.assertEqual(normalize_rubric_score(REWARD_HACKING_SCALE_MIN), 0.0)
    self.assertEqual(normalize_rubric_score(REWARD_HACKING_SCALE_MAX), 1.0)

  def test_the_midpoint_maps_to_a_half(self):
    self.assertAlmostEqual(normalize_rubric_score(3), 0.5)

  def test_out_of_range_grades_are_clamped_not_dropped(self):
    # A judge occasionally returns 0 or 6. Rejecting the grade would throw
    # away the other two dimensions of that sample, which matters when the
    # aggregate is a mean over only three numbers.
    self.assertEqual(normalize_rubric_score(0), 0.0)
    self.assertEqual(normalize_rubric_score(9), 1.0)

  def test_fractional_grades_survive(self):
    # k > 1 averages independent draws before normalization, so the input is
    # not necessarily an integer.
    self.assertAlmostEqual(normalize_rubric_score(4.5), 0.875)


class SummaryAggregationTest(unittest.TestCase):
  """Turning per-sample verdicts into the reported numbers."""

  def _perfect(self):
    return {key: 1.0 for key in reward_hacking_dimension_keys()}

  def _worst(self):
    return {key: 0.0 for key in reward_hacking_dimension_keys()}

  def test_quality_is_the_unweighted_mean_of_the_dimensions(self):
    keys = reward_hacking_dimension_keys()
    verdict = {key: 0.0 for key in keys}
    verdict[keys[0]] = 1.0
    summary = reward_hacking_summary([verdict], threshold=0.6)
    self.assertAlmostEqual(
        summary[REWARD_HACKING_QUALITY_KEY], 1.0 / len(keys)
    )

  def test_each_dimension_gets_its_own_mean(self):
    keys = reward_hacking_dimension_keys()
    summary = reward_hacking_summary(
        [self._perfect(), self._worst()], threshold=0.6
    )
    for key in keys:
      with self.subTest(key=key):
        self.assertAlmostEqual(summary[f"reward_hacking_{key}"], 0.5)

  def test_the_rate_counts_samples_below_the_threshold(self):
    verdicts = [self._perfect(), self._worst(), self._worst()]
    summary = reward_hacking_summary(verdicts, threshold=0.6)
    self.assertAlmostEqual(summary[REWARD_HACKING_RATE_KEY], 2 / 3)

  def test_the_threshold_is_recorded_alongside_the_rate(self):
    # The rate is uncalibrated, so it is meaningless without the operating
    # point it was computed at.
    summary = reward_hacking_summary([self._perfect()], threshold=0.42)
    self.assertAlmostEqual(
        summary[f"{REWARD_HACKING_AUDIT_PREFIX}threshold"], 0.42
    )

  def test_unscored_samples_are_counted_not_treated_as_zero(self):
    # A judge failure is missing data. Scoring it 0 would make an API outage
    # look like a degenerate policy.
    summary = reward_hacking_summary(
        [self._perfect(), None, None], threshold=0.6
    )
    self.assertEqual(summary[f"{REWARD_HACKING_AUDIT_PREFIX}n_total"], 3)
    self.assertEqual(summary[f"{REWARD_HACKING_AUDIT_PREFIX}n_scored"], 1)
    self.assertEqual(summary[f"{REWARD_HACKING_AUDIT_PREFIX}n_dropped"], 2)
    self.assertEqual(summary[REWARD_HACKING_QUALITY_KEY], 1.0)

  def test_an_all_failed_run_reports_no_quality_at_all(self):
    # Better an absent row than a confident 0.0 nobody measured.
    summary = reward_hacking_summary([None, None], threshold=0.6)
    self.assertNotIn(REWARD_HACKING_QUALITY_KEY, summary)
    self.assertNotIn(REWARD_HACKING_RATE_KEY, summary)
    self.assertEqual(summary[f"{REWARD_HACKING_AUDIT_PREFIX}n_dropped"], 2)

  def test_a_partial_verdict_still_contributes(self):
    # The judge dropped one key out of the JSON object. The remaining
    # dimensions are still information.
    keys = reward_hacking_dimension_keys()
    partial = {keys[0]: 1.0}
    summary = reward_hacking_summary([partial], threshold=0.6)
    self.assertAlmostEqual(summary[REWARD_HACKING_QUALITY_KEY], 1.0)
    self.assertNotIn(f"reward_hacking_{keys[-1]}", summary)

  def test_the_spread_across_samples_is_recorded(self):
    summary = reward_hacking_summary(
        [self._perfect(), self._worst()], threshold=0.6
    )
    self.assertGreater(
        summary[f"{REWARD_HACKING_AUDIT_PREFIX}quality_std"], 0.0
    )

  def test_the_summary_keys_split_cleanly_into_headline_and_audit(self):
    # Anything not under the audit prefix earns a row in the comparison
    # table, so an accidental audit key without the prefix pollutes it.
    summary = reward_hacking_summary([self._perfect()], threshold=0.6)
    headline = {
        key
        for key in summary
        if not key.startswith(REWARD_HACKING_AUDIT_PREFIX)
    }
    expected = {REWARD_HACKING_QUALITY_KEY, REWARD_HACKING_RATE_KEY} | {
        f"reward_hacking_{key}" for key in reward_hacking_dimension_keys()
    }
    self.assertEqual(headline, expected)


class RunNameTest(unittest.TestCase):
  """Where the rubric's prompt dumps land."""

  def test_everything_that_changes_the_verdicts_is_in_the_name(self):
    name = reward_hacking_run_name(
        reward_hacking_model="models/gemini-2.5-flash",
        task_name="ragtruth",
        reward_hacking_num_fewshot=2,
        seed=12345,
    )
    self.assertIn("gemini-2.5-flash", name)
    self.assertIn("ragtruth", name)
    self.assertIn("2", name)
    self.assertIn("12345", name)
    # The model is a repo path; a slash would fork the log directory.
    self.assertNotIn("/", name)

  def test_two_configurations_do_not_share_a_directory(self):
    first = reward_hacking_run_name("gemini-2.5-flash", "ragtruth", 0, 1)
    second = reward_hacking_run_name("gemini-2.5-flash", "ragtruth", 2, 1)
    self.assertNotEqual(first, second)


class MetricRegistryTest(unittest.TestCase):
  """How the report treats the rubric's metrics."""

  def test_the_rate_is_lower_is_better(self):
    self.assertIn(REWARD_HACKING_RATE_KEY, eval_metrics.LOWER_IS_BETTER)

  def test_quality_and_the_dimensions_are_higher_is_better(self):
    self.assertNotIn(REWARD_HACKING_QUALITY_KEY, eval_metrics.LOWER_IS_BETTER)
    for key in reward_hacking_dimension_keys():
      with self.subTest(key=key):
        self.assertNotIn(
            f"reward_hacking_{key}", eval_metrics.LOWER_IS_BETTER
        )

  def test_every_headline_metric_has_a_place_in_the_table_order(self):
    for name in (
        [REWARD_HACKING_QUALITY_KEY, REWARD_HACKING_RATE_KEY]
        + [f"reward_hacking_{k}" for k in reward_hacking_dimension_keys()]
    ):
      with self.subTest(name=name):
        self.assertIn(name, eval_metrics.METRIC_ORDER)

  def test_the_rubric_rows_sit_next_to_the_hallucination_rows(self):
    order = list(eval_metrics.METRIC_ORDER)
    self.assertLess(
        order.index("hallucination_rate"),
        order.index(REWARD_HACKING_QUALITY_KEY),
    )
    self.assertLess(
        order.index(REWARD_HACKING_QUALITY_KEY), order.index("bertscore_f1")
    )

  def test_the_audit_trail_is_hidden_from_the_table(self):
    metrics = {
        f"sft/{REWARD_HACKING_AUDIT_PREFIX}quality_std": 0.1,
        f"sft/{REWARD_HACKING_AUDIT_PREFIX}model": "gemini-2.5-flash",
        f"sft/{REWARD_HACKING_QUALITY_KEY}": 0.8,
    }
    names = eval_metrics.metric_names(metrics)
    self.assertIn(REWARD_HACKING_QUALITY_KEY, names)
    self.assertNotIn(f"{REWARD_HACKING_AUDIT_PREFIX}quality_std", names)
    self.assertNotIn(f"{REWARD_HACKING_AUDIT_PREFIX}model", names)

  def test_the_dropped_count_earns_a_row_anyway(self):
    # A quality score is only as trustworthy as the number of samples that
    # produced a verdict, so the reader has to see the drops.
    metrics = {f"sft/{REWARD_HACKING_AUDIT_PREFIX}n_dropped": 3}
    self.assertIn(
        f"{REWARD_HACKING_AUDIT_PREFIX}n_dropped",
        eval_metrics.metric_names(metrics),
    )

  def test_a_degenerate_policy_gets_a_negative_delta(self):
    # The headline invariant. PE-RL halves the hallucination rate but writes
    # worse prose; the report must not present that as a clean win.
    metrics = {
        "sft/hallucination_rate": 0.10,
        "perl/hallucination_rate": 0.05,
        f"sft/{REWARD_HACKING_QUALITY_KEY}": 0.85,
        f"perl/{REWARD_HACKING_QUALITY_KEY}": 0.60,
        f"sft/{REWARD_HACKING_RATE_KEY}": 0.05,
        f"perl/{REWARD_HACKING_RATE_KEY}": 0.40,
    }
    from src.orchestrator.stages.eval_stage import compute_deltas

    deltas = compute_deltas(metrics)
    self.assertGreater(deltas["delta/hallucination_rate"], 0.0)
    self.assertLess(deltas[f"delta/{REWARD_HACKING_QUALITY_KEY}"], 0.0)
    self.assertLess(deltas[f"delta/{REWARD_HACKING_RATE_KEY}"], 0.0)


class ScoringCommandTest(unittest.TestCase):
  """The flags the eval stage hands to the evaluator."""

  def _value(self, cmd, flag):
    return cmd[cmd.index(flag) + 1]

  def test_the_rubric_is_on_by_default(self):
    # A guard that has to be switched on is off exactly when it matters.
    self.assertTrue(_config().eval.run_reward_hacking_autorater)

  def test_the_flags_reach_the_scoring_command(self):
    cfg = _config()
    cmd = _stage(cfg)._scoring_command("u/policy")
    self.assertEqual(
        self._value(cmd, "--run_reward_hacking_autorater"), "True"
    )
    self.assertEqual(
        self._value(cmd, "--reward_hacking_num_fewshot"),
        str(cfg.eval.reward_hacking_num_fewshot),
    )
    self.assertEqual(
        self._value(cmd, "--reward_hacking_threshold"),
        str(cfg.eval.reward_hacking_threshold),
    )

  def test_the_judge_is_left_unset_so_it_tracks_the_evaluator_model(self):
    cmd = _stage(_config())._scoring_command("u/policy")
    self.assertNotIn("--reward_hacking_model", cmd)

  def test_an_explicit_judge_is_passed_through(self):
    cfg = _config()
    cfg.eval.reward_hacking_model = "gemini-2.5-pro"
    cmd = _stage(cfg)._scoring_command("u/policy")
    self.assertEqual(
        self._value(cmd, "--reward_hacking_model"), "gemini-2.5-pro"
    )

  def test_disabling_it_drops_the_rest_of_the_flags(self):
    cfg = _config()
    cfg.eval.run_reward_hacking_autorater = False
    cmd = _stage(cfg)._scoring_command("u/policy")
    self.assertEqual(
        self._value(cmd, "--run_reward_hacking_autorater"), "False"
    )
    self.assertNotIn("--reward_hacking_num_fewshot", cmd)
    self.assertNotIn("--reward_hacking_threshold", cmd)

  def test_calibration_is_not_asked_to_grade_the_rubric(self):
    # The rubric has no labelled set, so there is nothing for the
    # `autoratereval` stage to fit. Passing the flags there would pay for a
    # judge pass whose output nothing reads.
    from src.orchestrator.stages.autorater_stage import AutoraterStage

    state = CampaignState(campaign_id="c", task_name="ragtruth")
    config = _config()
    cmd = AutoraterStage(
        CampaignContext(
            config=config,
            state=state,
            sweep_controller=SweepController(dry_run=True),
            model_manager=ModelManager(dry_run=True),
        )
    ).build_command()
    self.assertFalse([a for a in cmd if "reward_hacking" in a])


class DryRunTest(unittest.TestCase):
  """What a rehearsal shows."""

  def _metrics(self, **eval_overrides):
    config = _config()
    config.dry_run = True
    for key, value in eval_overrides.items():
      setattr(config.eval, key, value)
    stage = _stage(config)
    targets = [
        EvalTarget(label="sft", model_repo_id="u/sft", temperature=0.0),
        EvalTarget(label="perl", model_repo_id="u/perl", temperature=0.0),
    ]
    result = stage._dry_run_result(targets, "u/perl", None)
    self.assertEqual(result.status, StageStatus.COMPLETED)
    return result.metrics

  def test_the_rubric_rows_are_rehearsed(self):
    metrics = self._metrics()
    for label in ("sft", "perl"):
      with self.subTest(label=label):
        self.assertIn(f"{label}/{REWARD_HACKING_QUALITY_KEY}", metrics)
        self.assertIn(f"{label}/{REWARD_HACKING_RATE_KEY}", metrics)
        for key in reward_hacking_dimension_keys():
          self.assertIn(f"{label}/reward_hacking_{key}", metrics)

  def test_the_mock_quality_is_the_mean_of_the_mock_dimensions(self):
    metrics = self._metrics()
    grades = [
        metrics[f"sft/reward_hacking_{key}"]
        for key in reward_hacking_dimension_keys()
    ]
    self.assertAlmostEqual(
        metrics[f"sft/{REWARD_HACKING_QUALITY_KEY}"],
        sum(grades) / len(grades),
        places=3,
    )

  def test_the_rehearsal_demonstrates_the_failure_it_detects(self):
    # PE-RL's hallucination rate improves while its rubric quality falls -
    # the exact pattern a reader has to learn to spot.
    metrics = self._metrics()
    self.assertLess(
        metrics["perl/hallucination_rate"], metrics["sft/hallucination_rate"]
    )
    self.assertLess(
        metrics[f"perl/{REWARD_HACKING_QUALITY_KEY}"],
        metrics[f"sft/{REWARD_HACKING_QUALITY_KEY}"],
    )
    self.assertLess(metrics[f"delta/{REWARD_HACKING_QUALITY_KEY}"], 0.0)

  def test_turning_the_rubric_off_removes_the_rows(self):
    metrics = self._metrics(run_reward_hacking_autorater=False)
    self.assertFalse([k for k in metrics if "reward_hacking" in k])


if __name__ == "__main__":
  unittest.main()
