"""Tests for evaluating every policy over a shared decoding-temperature grid.

RL concentrates the policy distribution, so scoring a PE-RL checkpoint only at
the temperature its sweep selected reports it in its most favourable regime.
Every policy - SFT and each PE-RL branch - is therefore scored at every
temperature of ``eval.temperature_grid``. Three properties matter here.

1. Every policy is scored at every grid temperature, plus any off-grid
   rollout temperature when ``match_perl_rollout_temperature`` is on.
2. No delta ever spans two decoding regimes: each policy cell is compared with
   the SFT cell decoded identically.
3. A single-temperature plan looks exactly like a campaign run before any of
   this existed: bare labels, old metric keys.
"""

import math
import tempfile
import unittest

from src.orchestrator import eval_metrics
from src.orchestrator import flavors
from src.orchestrator.config import CampaignConfig
from src.orchestrator.model_manager import ModelManager
from src.orchestrator.reporter import CampaignReporter
from src.orchestrator.stages.base import CampaignContext
from src.orchestrator.stages.eval_stage import compute_deltas
from src.orchestrator.stages.eval_stage import EvalStage
from src.orchestrator.stages.eval_stage import EvalTarget
from src.orchestrator.state import CampaignState
from src.orchestrator.state import StageResult
from src.orchestrator.state import StageStatus
from src.orchestrator.sweep_controller import SweepController

GRID_TOKENS = ("0.0", "0.7", "1.0", "1.25")


def _config(flavor_list=None, **kwargs):
  config = CampaignConfig.create_default(
      task_name="ragtruth",
      dry_run=True,
      rm_dataset_flavors=flavor_list,
      **kwargs,
  )
  config.user = "leobianco"
  return config


def _context(config, state):
  return CampaignContext(
      config=config,
      state=state,
      sweep_controller=SweepController(dry_run=True),
      model_manager=ModelManager(dry_run=True),
  )


def _stage(
    flavor_list=None, policies=None, with_sft=True, grid=None, **config_kwargs
):
  """Builds an EvalStage over a finished campaign.

  Args:
    flavor_list: RM dataset flavors the campaign branched over.
    policies: Map of perl stage id -> (repo id, rollout temperature). A
      temperature of None stands for a sweep that pinned it outside its
      parameters block, so nothing was recorded per trial.
    with_sft: Whether the campaign produced an SFT checkpoint.
    grid: Overrides ``eval.temperature_grid``; None keeps the default.
    **config_kwargs: Forwarded to the campaign config.

  Returns:
    The stage, ready to resolve its targets.
  """
  config = _config(flavor_list, **config_kwargs)
  if grid is not None:
    config.eval.temperature_grid = list(grid)
  state = CampaignState(campaign_id="c", task_name="ragtruth")
  if with_sft:
    state.stages["sft"] = StageResult(
        status=StageStatus.COMPLETED, model_repo_id="u/sft"
    )
  for stage_id, (repo_id, temperature) in (policies or {}).items():
    state.stages[stage_id] = StageResult(
        status=StageStatus.COMPLETED,
        model_repo_id=repo_id,
        best_params={} if temperature is None else {
            "temperature": temperature,
            "learning_rate": 1e-5,
        },
    )
  return EvalStage(_context(config, state))


def _labels(stage):
  return [t.label for t in stage.resolve_targets()]


class LabelGrammarTest(unittest.TestCase):
  """The ``sft@t0.6`` label vocabulary."""

  def test_temperatures_round_trip_through_a_label(self):
    for value in (0.0, 0.3, 0.6, 1.0, 0.25, 1.25):
      label = eval_metrics.make_target_label("sft", value)
      stage_id, parsed = eval_metrics.split_target_label(label)
      self.assertEqual(stage_id, "sft")
      self.assertAlmostEqual(parsed, value)

  def test_the_token_always_keeps_one_decimal(self):
    # "T=1" reads like a count; the report is about sampling temperature.
    self.assertEqual(eval_metrics.format_temperature(1.0), "1.0")
    self.assertEqual(eval_metrics.format_temperature(0.0), "0.0")
    self.assertEqual(eval_metrics.format_temperature(0.6), "0.6")
    self.assertEqual(eval_metrics.format_temperature(0.25), "0.25")
    self.assertEqual(eval_metrics.format_temperature(1.25), "1.25")

  def test_an_untagged_label_is_left_alone(self):
    self.assertEqual(eval_metrics.make_target_label("sft", None), "sft")
    self.assertEqual(
        eval_metrics.split_target_label("perl:organic"),
        ("perl:organic", None),
    )

  def test_a_malformed_tag_costs_its_temperature_not_the_report(self):
    self.assertEqual(
        eval_metrics.split_target_label("sft@tbogus"), ("sft@tbogus", None)
    )

  def test_every_sft_row_counts_as_a_baseline(self):
    self.assertTrue(eval_metrics.is_baseline("sft"))
    self.assertTrue(eval_metrics.is_baseline("sft@t0.6"))
    self.assertFalse(eval_metrics.is_baseline("perl"))
    self.assertFalse(eval_metrics.is_baseline("perl:organic@t0.7"))

  def test_titles_name_the_decoding_regime(self):
    self.assertEqual(eval_metrics.target_title("sft"), "SFT")
    self.assertEqual(eval_metrics.target_title("sft@t0.6"), "SFT (T=0.6)")
    self.assertEqual(
        eval_metrics.target_title("perl:organic"), "PE-RL (Organic)"
    )

  def test_a_tagged_delta_keeps_flavor_and_temperature(self):
    # One delta per (branch, temperature): dropping the tag would make the
    # four temperatures of a branch overwrite each other's deltas.
    self.assertEqual(
        eval_metrics.delta_label("perl:organic@t0.7"), "delta:organic@t0.7"
    )
    self.assertEqual(eval_metrics.delta_label("perl@t1.25"), "delta@t1.25")

  def test_an_untagged_delta_keeps_the_old_name(self):
    self.assertEqual(eval_metrics.delta_label("perl"), "delta")
    self.assertEqual(
        eval_metrics.delta_label("perl:organic"), "delta:organic"
    )

  def test_baselines_lead_the_table_coldest_first(self):
    metrics = {
        "perl:organic/x": 1,
        "sft@t1.0/x": 1,
        "sft/x": 1,
        "sft@t0.3/x": 1,
    }
    self.assertEqual(
        eval_metrics.target_labels(metrics),
        ["sft", "sft@t0.3", "sft@t1.0", "perl:organic"],
    )

  def test_a_matched_baseline_is_never_the_headline(self):
    # It is the same SFT checkpoint; treating it as a policy would let the
    # baseline win the campaign.
    metrics = {
        "sft/hallucination_rate": 0.10,
        "sft@t0.6/hallucination_rate": 0.02,
        "perl/hallucination_rate": 0.07,
    }
    self.assertAlmostEqual(eval_metrics.headline_metric(metrics), 0.07)


def _grid_metrics():
  """A two-policy, three-temperature metric map with recorded temperatures."""
  metrics = {}
  for index, token in enumerate(("0.0", "0.7", "1.25")):
    temperature = float(token)
    for stage_id, rate in (("sft", 0.10), ("perl:organic", 0.04)):
      label = f"{stage_id}@t{token}"
      metrics[f"{label}/hallucination_rate"] = rate + 0.05 * index
      metrics[f"{label}/decoding_temperature"] = temperature
  return metrics


class TemperatureGroupingTest(unittest.TestCase):
  """Grouping targets by the temperature they were decoded at."""

  def test_groups_are_coldest_first_and_hold_every_policy(self):
    groups = eval_metrics.temperature_groups(_grid_metrics())
    self.assertEqual([t for t, _ in groups], [0.0, 0.7, 1.25])
    self.assertEqual(groups[0][1], ["sft@t0.0", "perl:organic@t0.0"])

  def test_a_recorded_temperature_beats_the_label(self):
    self.assertAlmostEqual(
        eval_metrics.target_temperature(
            {"perl/decoding_temperature": 0.7}, "perl"
        ),
        0.7,
    )
    self.assertAlmostEqual(
        eval_metrics.target_temperature({}, "perl@t1.25"), 1.25
    )
    self.assertIsNone(eval_metrics.target_temperature({}, "perl"))

  def test_the_headline_is_quoted_at_the_reference_temperature(self):
    # Not the best cell of the grid: picking the minimum over temperatures
    # would be a selection effect of its own.
    metrics = _grid_metrics()
    self.assertAlmostEqual(eval_metrics.headline_metric(metrics), 0.04)
    self.assertAlmostEqual(
        eval_metrics.headline_metric(metrics, reference_temperature=1.25),
        0.14,
    )

  def test_comparison_rows_can_be_restricted_to_one_group(self):
    metrics = _grid_metrics()
    metrics.update(compute_deltas(metrics))
    _, labels = eval_metrics.temperature_groups(metrics)[1]
    rows = dict(
        (name, values)
        for name, values, _ in eval_metrics.comparison_rows(metrics, labels)
    )
    self.assertEqual(len(rows["hallucination_rate"]), 2)


class ConfidenceIntervalTest(unittest.TestCase):
  """95% intervals derived from what the summary JSON already records."""

  def test_wilson_matches_a_reference_value(self):
    low, high = eval_metrics.wilson_interval(0.1, 1000)
    self.assertAlmostEqual(low, 0.0829, places=4)
    self.assertAlmostEqual(high, 0.1202, places=4)

  def test_wilson_stays_inside_the_unit_interval_at_the_edges(self):
    for proportion in (0.0, 1.0):
      low, high = eval_metrics.wilson_interval(proportion, 20)
      self.assertGreaterEqual(low, 0.0)
      self.assertLessEqual(high, 1.0)
      self.assertLess(low, high)

  def test_rates_use_the_autorater_count(self):
    interval = eval_metrics.confidence_interval(
        {"hallucination_rate": 0.1, "autorater_scored_count": 1000},
        "hallucination_rate",
    )
    self.assertAlmostEqual(interval[0], 0.0829, places=4)

  def test_means_use_the_recorded_std(self):
    interval = eval_metrics.confidence_interval(
        {"perplexity_mean": 3.0, "perplexity_std": 1.0, "num_samples": 400},
        "perplexity_mean",
    )
    half = eval_metrics.Z_95 * 1.0 / math.sqrt(400)
    self.assertAlmostEqual(interval[0], 3.0 - half)
    self.assertAlmostEqual(interval[1], 3.0 + half)

  def test_rubric_quality_uses_the_audit_std_and_count(self):
    interval = eval_metrics.confidence_interval(
        {
            "reward_hacking_quality": 0.7,
            "reward_hacking_audit_quality_std": 0.2,
            "reward_hacking_audit_n_scored": 100,
        },
        "reward_hacking_quality",
    )
    half = eval_metrics.Z_95 * 0.2 / 10
    self.assertAlmostEqual(interval[1] - interval[0], 2 * half)

  def test_no_sample_size_means_no_interval(self):
    # An interval invented from a guessed n would look authoritative.
    self.assertIsNone(
        eval_metrics.confidence_interval(
            {"hallucination_rate": 0.1}, "hallucination_rate"
        )
    )

  def test_the_formatted_cell_carries_the_bounds(self):
    cell = eval_metrics.format_with_interval(
        {"hallucination_rate": 0.1, "autorater_scored_count": 1000},
        "hallucination_rate",
    )
    self.assertIn("[", cell)
    self.assertIn("0.0829", cell)


class TemperatureResolutionTest(unittest.TestCase):
  """Where the temperatures come from."""

  def test_the_winning_trial_temperature_is_recorded(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.6)})
    self.assertAlmostEqual(stage.rollout_temperature(flavors.ORGANIC), 0.6)

  def test_a_pinned_zero_is_a_real_answer_not_a_missing_one(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.0)})
    self.assertAlmostEqual(
        stage.context.perl_rollout_temperature_for(flavors.ORGANIC), 0.0
    )

  def test_an_unrecorded_temperature_is_none(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", None)})
    self.assertIsNone(stage.rollout_temperature(flavors.ORGANIC))

  def test_a_branch_that_never_ran_has_no_temperature(self):
    stage = _stage([flavors.ORGANIC], {})
    self.assertIsNone(
        stage.context.perl_rollout_temperature_for(flavors.ORGANIC)
    )

  def test_the_default_grid(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.7)})
    self.assertEqual(stage.temperature_grid(), [0.0, 0.7, 1.0, 1.25])

  def test_the_grid_is_sorted_and_deduplicated_by_token(self):
    stage = _stage(
        [flavors.ORGANIC], {"perl": ("u/policy", 0.7)}, grid=[1.0, 0.0, 1, 0.7]
    )
    self.assertEqual(stage.temperature_grid(), [0.0, 0.7, 1.0])

  def test_an_empty_grid_falls_back_to_the_reference_temperature(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.7)}, grid=[])
    self.assertEqual(stage.temperature_grid(), [0.0])


class TargetPlanTest(unittest.TestCase):
  """Which passes get scheduled, and in what order."""

  def test_every_policy_is_scored_at_every_grid_temperature(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.7)})
    self.assertEqual(
        _labels(stage),
        [f"sft@t{t}" for t in GRID_TOKENS] + [f"perl@t{t}" for t in GRID_TOKENS],
    )

  def test_policy_passes_carry_their_rollout_temperature(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.7)})
    targets = stage.resolve_targets()
    self.assertEqual(
        targets[4],
        EvalTarget("perl@t0.0", "u/policy", 0.0, rollout_temperature=0.7),
    )
    self.assertIsNone(targets[0].rollout_temperature)

  def test_the_policy_stays_last_so_it_remains_the_primary_model(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.7)})
    self.assertEqual(stage.resolve_targets()[-1].model_repo_id, "u/policy")

  def test_branches_each_get_the_full_grid(self):
    stage = _stage(
        [flavors.ORGANIC, flavors.SYNTHETIC_STRUCT],
        {
            "perl:organic": ("u/org", 0.7),
            "perl:synthetic_struct": ("u/syn", 1.0),
        },
    )
    labels = _labels(stage)
    self.assertEqual(len(labels), 12)
    self.assertEqual(labels[4:8], [f"perl:organic@t{t}" for t in GRID_TOKENS])
    self.assertEqual(
        labels[8:], [f"perl:synthetic_struct@t{t}" for t in GRID_TOKENS]
    )

  def test_an_off_grid_rollout_temperature_is_added_for_it_and_sft(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.6)})
    labels = _labels(stage)
    self.assertIn("sft@t0.6", labels)
    self.assertIn("perl@t0.6", labels)
    self.assertEqual(len(labels), 10)

  def test_an_off_grid_point_is_not_added_to_other_branches(self):
    stage = _stage(
        [flavors.ORGANIC, flavors.SYNTHETIC_STRUCT],
        {
            "perl:organic": ("u/org", 0.6),
            "perl:synthetic_struct": ("u/syn", 1.0),
        },
    )
    labels = _labels(stage)
    self.assertIn("perl:organic@t0.6", labels)
    self.assertNotIn("perl:synthetic_struct@t0.6", labels)

  def test_switching_matching_off_keeps_only_the_grid(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.6)})
    stage.config.eval.match_perl_rollout_temperature = False
    self.assertNotIn("sft@t0.6", _labels(stage))
    self.assertEqual(len(_labels(stage)), 8)

  def test_a_single_temperature_plan_keeps_bare_labels(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.6)}, grid=[0.0])
    stage.config.eval.match_perl_rollout_temperature = False
    self.assertEqual(
        stage.resolve_targets(),
        [
            EvalTarget("sft", "u/sft", 0.0),
            EvalTarget("perl", "u/policy", 0.0, rollout_temperature=0.6),
        ],
    )

  def test_a_one_point_grid_plus_an_off_grid_rollout_is_tagged(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.6)}, grid=[0.0])
    self.assertEqual(
        _labels(stage), ["sft@t0.0", "sft@t0.6", "perl@t0.0", "perl@t0.6"]
    )

  def test_no_sft_means_only_policy_passes(self):
    stage = _stage(
        [flavors.ORGANIC], {"perl": ("u/policy", 0.7)}, with_sft=False
    )
    self.assertEqual(_labels(stage), [f"perl@t{t}" for t in GRID_TOKENS])

  def test_a_campaign_with_nothing_trained_still_refuses_to_run(self):
    stage = _stage([flavors.ORGANIC], {}, with_sft=False)
    with self.assertRaises(ValueError):
      stage.resolve_targets()

  def test_the_grid_is_explained_in_the_log(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.7)})
    notes = " ".join(stage.temperature_notes(stage.resolve_targets()))
    self.assertIn("0.0, 0.7, 1.0, 1.25", notes)
    self.assertIn("8 pass(es)", notes)

  def test_an_off_grid_rollout_is_explained_in_the_log(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.6)})
    notes = " ".join(stage.temperature_notes(stage.resolve_targets()))
    self.assertIn("T=0.6, off the grid", notes)

  def test_an_unrecorded_temperature_is_called_out(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", None)})
    notes = " ".join(stage.temperature_notes(stage.resolve_targets()))
    self.assertIn("No rollout temperature was recorded", notes)

  def test_switching_matching_off_says_so(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.6)})
    stage.config.eval.match_perl_rollout_temperature = False
    notes = " ".join(stage.temperature_notes(stage.resolve_targets()))
    self.assertIn("match_perl_rollout_temperature is off", notes)


class ConfigValidationTest(unittest.TestCase):
  """Nonsense grid entries fail at config time, not mid-campaign."""

  def test_the_default_config_validates(self):
    _config([flavors.ORGANIC]).validate()

  def test_bad_grid_entries_are_rejected(self):
    for bad in (-0.1, float("nan"), float("inf"), True):
      config = _config([flavors.ORGANIC])
      config.eval.temperature_grid = [0.0, bad]
      with self.assertRaises(ValueError, msg=repr(bad)):
        config.validate()

class CommandTest(unittest.TestCase):
  """The temperature has to reach the evaluator, consistently."""

  def setUp(self):
    self.stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.6)})

  def _flag(self, cmd, name):
    return cmd[cmd.index(name) + 1]

  def test_generation_uses_the_target_temperature(self):
    cmd = self.stage._generation_command("u/policy", 0.6)
    self.assertEqual(self._flag(cmd, "--temperature"), "0.6")

  def test_scoring_uses_the_same_temperature_as_generation(self):
    gen = self.stage._generation_command("u/policy", 0.6)
    score = self.stage._scoring_command("u/policy", 0.6)
    self.assertEqual(
        self._flag(gen, "--temperature"), self._flag(score, "--temperature")
    )

  def test_omitting_the_temperature_falls_back_to_the_config(self):
    cmd = self.stage._generation_command("u/policy")
    self.assertEqual(self._flag(cmd, "--temperature"), "0.0")

  def test_the_two_baseline_passes_write_to_different_datasets(self):
    # Same adapter, different decode. If these collided, the second pass
    # would silently overwrite the first one's completions and both rows of
    # the report would describe the same generations.
    greedy, greedy_summary = self.stage._summary_path("u/sft", 0.0)
    matched, matched_summary = self.stage._summary_path("u/sft", 0.6)
    self.assertNotEqual(greedy, matched)
    self.assertNotEqual(greedy_summary, matched_summary)


class DeltaPairingTest(unittest.TestCase):
  """Every delta is taken against a baseline decoded the same way."""

  METRICS = {
      "sft/hallucination_rate": 0.10,
      "sft/decoding_temperature": 0.0,
      "sft@t0.6/hallucination_rate": 0.16,
      "sft@t0.6/decoding_temperature": 0.6,
      "perl/hallucination_rate": 0.13,
      "perl/decoding_temperature": 0.6,
  }

  def test_the_matched_baseline_is_used_not_the_greedy_one(self):
    # Against the greedy baseline PE-RL looks 0.03 *worse*; against the
    # baseline sampled the same way it is 0.03 better. This single number is
    # the entire point of the feature.
    deltas = compute_deltas(self.METRICS)
    self.assertAlmostEqual(deltas["delta/hallucination_rate"], 0.03)

  def test_the_temperature_itself_never_gets_a_delta(self):
    deltas = compute_deltas(self.METRICS)
    self.assertNotIn("delta/decoding_temperature", deltas)

  def test_a_matched_baseline_gets_no_delta_of_its_own(self):
    deltas = compute_deltas(self.METRICS)
    self.assertFalse([k for k in deltas if "sft" in k])

  def test_branches_pair_with_their_own_temperature(self):
    deltas = compute_deltas({
        "sft/hallucination_rate": 0.10,
        "sft/decoding_temperature": 0.0,
        "sft@t0.3/hallucination_rate": 0.12,
        "sft@t0.3/decoding_temperature": 0.3,
        "sft@t1.0/hallucination_rate": 0.20,
        "sft@t1.0/decoding_temperature": 1.0,
        "perl:organic/hallucination_rate": 0.09,
        "perl:organic/decoding_temperature": 0.3,
        "perl:synthetic_struct/hallucination_rate": 0.15,
        "perl:synthetic_struct/decoding_temperature": 1.0,
    })
    self.assertAlmostEqual(deltas["delta:organic/hallucination_rate"], 0.03)
    self.assertAlmostEqual(
        deltas["delta:synthetic_struct/hallucination_rate"], 0.05
    )

  def test_without_a_matching_baseline_the_plain_one_is_used(self):
    deltas = compute_deltas({
        "sft/hallucination_rate": 0.10,
        "sft/decoding_temperature": 0.0,
        "perl/hallucination_rate": 0.04,
        "perl/decoding_temperature": 0.6,
    })
    self.assertAlmostEqual(deltas["delta/hallucination_rate"], 0.06)

  def test_metrics_from_before_this_feature_still_produce_deltas(self):
    # No decoding_temperature keys anywhere: an eval recorded by an earlier
    # version of the orchestrator must keep rendering its comparison.
    deltas = compute_deltas({
        "sft/hallucination_rate": 0.10,
        "perl/hallucination_rate": 0.04,
    })
    self.assertAlmostEqual(deltas["delta/hallucination_rate"], 0.06)

  def test_grid_cells_pair_with_the_sft_cell_at_their_own_temperature(self):
    deltas = compute_deltas(_grid_metrics())
    for index, token in enumerate(("0.0", "0.7", "1.25")):
      del index
      self.assertAlmostEqual(
          deltas[f"delta:organic@t{token}/hallucination_rate"], 0.06
      )
    self.assertNotIn("delta:organic/hallucination_rate", deltas)

class RelativeChangeTest(unittest.TestCase):
  """Re-expressing a delta as a share of what it started from.

  The dashboard prints this under every delta column. It cannot look the
  baseline up - which SFT row a policy was compared against depends on the
  branch's decoding temperature - so it inverts the delta instead. These
  tests pin that inversion against the function that produced the delta in
  the first place.
  """

  def test_a_halved_rate_reads_as_fifty_percent(self):
    # The motivating example: 0.20 -> 0.10 is not "-0.10", it is half the
    # hallucinations gone.
    self.assertAlmostEqual(
        eval_metrics.relative_change("hallucination_rate", 0.10, 0.10), 0.5
    )

  def test_a_lower_is_better_metric_keeps_improvement_positive(self):
    # Stored delta is positive-is-better, and so is the ratio, even though
    # the metric itself went down.
    ratio = eval_metrics.relative_change("hallucination_rate", 0.043, 0.062)
    self.assertGreater(ratio, 0.0)
    self.assertAlmostEqual(ratio, 0.043 / 0.105)

  def test_a_higher_is_better_metric_uses_the_other_direction(self):
    # bertscore_f1 rose from 0.874 to 0.892; the delta is +0.018 and the
    # baseline must come back as 0.874, not 0.910.
    self.assertAlmostEqual(
        eval_metrics.baseline_from_delta("bertscore_f1", 0.018, 0.892), 0.874
    )

  def test_a_regression_reads_as_a_negative_share(self):
    # Policy 0.12 against a 0.10 baseline: a fifth worse.
    self.assertAlmostEqual(
        eval_metrics.relative_change("hallucination_rate", -0.02, 0.12), -0.2
    )

  def test_the_inversion_agrees_with_the_delta_it_came_from(self):
    # The contract the dashboard leans on, checked end to end and per
    # branch: each delta must invert back to the baseline row the eval
    # stage actually paired the policy with, not to the greedy `sft` row.
    metrics = {
        "sft/hallucination_rate": 0.10,
        "sft/decoding_temperature": 0.0,
        "sft@t0.3/hallucination_rate": 0.12,
        "sft@t0.3/decoding_temperature": 0.3,
        "sft@t1.0/hallucination_rate": 0.20,
        "sft@t1.0/decoding_temperature": 1.0,
        "perl:organic/hallucination_rate": 0.09,
        "perl:organic/decoding_temperature": 0.3,
        "perl:synthetic_struct/hallucination_rate": 0.15,
        "perl:synthetic_struct/decoding_temperature": 1.0,
    }
    deltas = compute_deltas(metrics)
    for policy, baseline in (
        ("organic", "sft@t0.3"),
        ("synthetic_struct", "sft@t1.0"),
    ):
      recovered = eval_metrics.baseline_from_delta(
          "hallucination_rate",
          deltas[f"delta:{policy}/hallucination_rate"],
          metrics[f"perl:{policy}/hallucination_rate"],
      )
      self.assertAlmostEqual(
          recovered, metrics[f"{baseline}/hallucination_rate"]
      )

  def test_a_zero_baseline_has_no_ratio_rather_than_an_infinite_one(self):
    # Nothing to improve on; 0/0 is not a 100% win.
    self.assertIsNone(
        eval_metrics.relative_change("hallucination_rate", 0.0, 0.0)
    )

  def test_a_negative_baseline_is_refused(self):
    # A share of a negative reference flips sign for no reason a reader
    # could follow, so the cell stays blank.
    self.assertIsNone(
        eval_metrics.relative_change("bertscore_f1", 0.2, -0.1)
    )

  def test_a_missing_or_non_numeric_value_is_refused(self):
    for improvement, policy in ((None, 0.1), (0.1, None), (True, 0.1)):
      self.assertIsNone(
          eval_metrics.relative_change(
              "hallucination_rate", improvement, policy
          )
      )

  def test_the_sign_matches_how_the_delta_was_built(self):
    self.assertEqual(eval_metrics.delta_sign("hallucination_rate"), -1)
    self.assertEqual(eval_metrics.delta_sign("bertscore_f1"), 1)



class ExecutionTest(unittest.TestCase):
  """What a full (dry) pass records."""

  def test_every_target_records_both_temperatures(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.7)})
    metrics = stage.execute().metrics
    for token in GRID_TOKENS:
      self.assertAlmostEqual(
          metrics[f"sft@t{token}/decoding_temperature"], float(token)
      )
      self.assertAlmostEqual(
          metrics[f"perl@t{token}/decoding_temperature"], float(token)
      )
      self.assertAlmostEqual(metrics[f"perl@t{token}/rollout_temperature"], 0.7)

  def test_each_delta_is_computed_at_its_own_temperature(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.7)})
    metrics = stage.execute().metrics
    for token in GRID_TOKENS:
      expected = (
          metrics[f"sft@t{token}/hallucination_rate"]
          - metrics[f"perl@t{token}/hallucination_rate"]
      )
      self.assertAlmostEqual(
          metrics[f"delta@t{token}/hallucination_rate"], expected
      )

  def test_the_rehearsal_gets_worse_as_temperature_rises(self):
    # The dry-run mock is what the dashboard is developed against; a flat
    # mock would hide a broken chart.
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.7)})
    metrics = stage.execute().metrics
    rates = [metrics[f"sft@t{t}/hallucination_rate"] for t in GRID_TOKENS]
    self.assertEqual(rates, sorted(rates))

  def test_a_single_temperature_campaign_writes_the_old_keys(self):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.6)}, grid=[0.0])
    stage.config.eval.match_perl_rollout_temperature = False
    metrics = stage.execute().metrics
    self.assertFalse([k for k in metrics if "@t" in str(k)])
    self.assertIn("delta/hallucination_rate", metrics)


class RealExecutionPathTest(unittest.TestCase):
  """The non-dry path, with the subprocesses stubbed out.

  ``execute()`` short-circuits into ``_dry_run_result`` long before
  ``_evaluate_target`` runs, so a rehearsal exercises none of the command
  construction. Everything below is reachable only with ``dry_run`` off.
  """

  def setUp(self):
    self.stage = _stage(
        [flavors.ORGANIC], {"perl": ("u/policy", 0.6)}, grid=[0.0]
    )
    self.stage.config.dry_run = False
    self.commands = []
    self.stage._run_subprocess = self._capture
    self.stage._load_summary = lambda repo, temperature: {
        "hallucination_rate": 0.05
    }

  def _capture(self, cmd, *_args, **_kwargs):
    self.commands.append(cmd)
    return {}

  def _temperature_of(self, cmd):
    return cmd[cmd.index("--temperature") + 1]

  def _pairs(self):
    """Returns (mode, temperature) for each captured evaluator call."""
    return [
        (cmd[cmd.index("--mode") + 1], self._temperature_of(cmd))
        for cmd in self.commands
    ]

  def test_generation_and_scoring_always_agree_on_the_temperature(self):
    # They independently derive the completions repo name from it. If they
    # disagree, scoring silently reads another run's dataset and reports its
    # numbers under this target's name.
    self.stage.execute()
    modes = self._pairs()
    generates = [t for mode, t in modes if mode == "generate"]
    scores = [t for mode, t in modes if mode == "score"]
    self.assertEqual(generates, scores)

  def test_each_target_is_generated_at_its_own_temperature(self):
    self.stage.execute()
    self.assertEqual(
        [t for mode, t in self._pairs() if mode == "generate"],
        ["0.0", "0.6", "0.0", "0.6"],
    )

  def test_the_default_grid_runs_four_passes_per_policy(self):
    self.stage.config.eval.temperature_grid = [0.0, 0.7, 1.0, 1.25]
    self.stage.execute()
    generated = [t for mode, t in self._pairs() if mode == "generate"]
    self.assertEqual(len(generated), 10)  # 4 grid + 1 off-grid, x2 policies

  def test_the_temperatures_are_recorded_on_the_real_path_too(self):
    metrics = self.stage.execute().metrics
    self.assertAlmostEqual(metrics["sft@t0.0/decoding_temperature"], 0.0)
    self.assertAlmostEqual(metrics["sft@t0.6/decoding_temperature"], 0.6)
    self.assertAlmostEqual(metrics["perl@t0.6/decoding_temperature"], 0.6)
    self.assertAlmostEqual(metrics["perl@t0.0/rollout_temperature"], 0.6)

  def test_the_real_path_pairs_each_delta_with_its_own_baseline(self):
    metrics = self.stage.execute().metrics
    self.assertAlmostEqual(metrics["delta@t0.0/hallucination_rate"], 0.0)
    self.assertAlmostEqual(metrics["delta@t0.6/hallucination_rate"], 0.0)

  def test_a_resumed_run_does_not_rescore_a_finished_pass(self):
    # Autorating is billed per call; every pass must be resumable.
    self.stage.state.stages["eval"] = StageResult(
        status=StageStatus.RUNNING,
        metrics={
            "sft@t0.6/hallucination_rate": 0.11,
            "sft@t0.6/decoding_temperature": 0.6,
        },
    )
    self.stage.execute()
    generated = [
        cmd[cmd.index("--writer_model_lora") + 1]
        for cmd in self.commands
        if cmd[cmd.index("--mode") + 1] == "generate"
    ]
    # u/sft still appears once for the greedy pass, not twice.
    self.assertEqual(generated.count("u/sft"), 1)


class ReportTest(unittest.TestCase):
  """The report has to make every decoding regime visible."""

  def _report(self, grid=None):
    stage = _stage([flavors.ORGANIC], {"perl": ("u/policy", 0.7)}, grid=grid)
    result = stage.execute()
    state = stage.state
    state.stages["eval"] = result
    with tempfile.TemporaryDirectory() as temp_dir:
      stage.config.reporting.reports_dir = temp_dir
      path = CampaignReporter(stage.config, state).generate_markdown_report()
      with open(path, "r", encoding="utf-8") as handle:
        return handle.read()

  def test_the_sweep_summary_has_intervals_and_marks_the_rollout(self):
    report = self._report()
    self.assertIn("**Temperature sweep**", report)
    self.assertIn("**Hallucination rate** vs. decoding temperature", report)
    self.assertIn("\u2605", report)
    self.assertRegex(report, r"\[0\.\d+, 0\.\d+\]")

  def test_one_table_per_temperature(self):
    report = self._report()
    for token in GRID_TOKENS:
      self.assertIn(f"Per-policy results at T = {token}", report)

  def test_each_temperature_table_pairs_like_with_like(self):
    self.assertIn("sampled at the same temperature", self._report())

  def test_the_temperature_is_tabulated(self):
    self.assertIn("decoding_temperature", self._report())

  def test_the_temperature_reads_as_a_setting_not_a_measurement(self):
    # "0.7000" in a column of four-decimal rates invites the reader to treat
    # the knob as something that was measured.
    report = self._report()
    self.assertIn("| **0.7** |", report)
    self.assertNotIn("0.7000", report)

  def test_a_single_temperature_report_has_no_sweep_section(self):
    report = self._report(grid=[0.7])
    self.assertNotIn("**Temperature sweep**", report)
    self.assertIn("Per-policy results", report)


if __name__ == "__main__":
  unittest.main()
