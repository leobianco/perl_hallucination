"""Tests for window-average trial ranking and honest candidate accounting.

These cover the three defects that let a PE-RL campaign publish a winner
nobody could account for:

  1. ``reward_penalty_alpha`` was swept while also rescaling the very metric
     the sweep maximized, so trials were ranked on incomparable scores.
  2. ``selection_strategy="final"`` ranked each trial on one logged step of a
     reward averaged over 8 sampled generations, i.e. on noise.
  3. Runs in state ``crashed`` / ``killed`` were dropped from the comparison
     without a word, so a winner chosen from 3 candidates was
     indistinguishable in every artifact from one chosen from 5.
"""

import sys
import tempfile
import types
import unittest

import yaml

from src.orchestrator import reporter as reporter_mod
from src.orchestrator.config import CampaignConfig
from src.orchestrator.model_manager import ModelManager, REWARD_PENALTY_ALPHA
from src.orchestrator.stages.base import SweepExecution
from src.orchestrator.state import CampaignState, StageResult, StageStatus
from src.orchestrator.sweep_controller import RunScore, SweepController
from src.orchestrator.sweep_controller import _tail_mean_history

SWEEP_PERL_PATH = "scripts/sweep_perl.yaml"

REWARD = "train/rewards/reward_fn/mean"


class _FakeRun:
  """Mirrors the slice of the W&B run object the controller reads."""

  def __init__(
      self, run_id, summary, state="finished", config=None, history=None
  ):
    self.id = run_id
    self.summary = summary
    self.state = state
    self.config = config or {}
    #: None means the run exposes no readable history at all.
    self._history = history

  def scan_history(self, keys=None):
    if self._history is None:
      raise RuntimeError("history unavailable")
    if not keys:
      return list(self._history)
    return [row for row in self._history if all(k in row for k in keys)]


class _FakeSweep:

  def __init__(self, runs):
    self.runs = runs


class _FakeApi:

  def __init__(self, sweep):
    self._sweep = sweep

  def sweep(self, _path):
    return self._sweep


class _WandbTestCase(unittest.TestCase):
  """Installs a fake ``wandb`` module for the duration of a test."""

  def install_wandb(self, runs):
    module = types.ModuleType("wandb")
    module.Api = lambda: _FakeApi(_FakeSweep(runs))
    sys.modules["wandb"] = module
    self.addCleanup(sys.modules.pop, "wandb", None)

  def controller(self):
    return SweepController(entity="e", project="p", dry_run=False)


def _reward_run(run_id, values, state="finished", config=None):
  """Builds a trial that logged ``values`` for the PE-RL reward, in order."""
  history = [{REWARD: v, "_step": i} for i, v in enumerate(values)]
  return _FakeRun(
      run_id, {REWARD: values[-1]}, state=state, config=config, history=history
  )


class TailMeanTest(unittest.TestCase):
  """The window helper itself."""

  def test_averages_only_the_trailing_points(self):
    run = _reward_run("r", [0.0, 0.0, 0.0, 1.0, 3.0])
    mean, points = _tail_mean_history(run, REWARD, 2)
    self.assertAlmostEqual(mean, 2.0)
    self.assertEqual(points, 2)

  def test_a_short_trial_averages_what_it_has(self):
    # The window is a ceiling, not a requirement: a trial that logged three
    # points is still rankable, it is just averaged over three.
    run = _reward_run("r", [1.0, 2.0, 3.0])
    mean, points = _tail_mean_history(run, REWARD, 10)
    self.assertAlmostEqual(mean, 2.0)
    self.assertEqual(points, 3)

  def test_a_window_below_one_still_averages_one_point(self):
    # Config validation rejects this, but the helper must not divide by zero
    # if it is ever reached through another path.
    run = _reward_run("r", [1.0, 2.0, 7.0])
    mean, points = _tail_mean_history(run, REWARD, 0)
    self.assertAlmostEqual(mean, 7.0)
    self.assertEqual(points, 1)

  def test_unreadable_history_is_a_miss_not_a_zero(self):
    run = _FakeRun("r", {REWARD: 0.5}, history=None)
    self.assertIsNone(_tail_mean_history(run, REWARD, 5))

  def test_unusable_values_are_skipped(self):
    run = _FakeRun(
        "r",
        {REWARD: 1.0},
        history=[
            {REWARD: 3.0, "_step": 0},
            {REWARD: None, "_step": 1},
            {REWARD: float("inf"), "_step": 2},
            {REWARD: 1.0, "_step": 3},
        ],
    )
    mean, points = _tail_mean_history(run, REWARD, 10)
    self.assertAlmostEqual(mean, 2.0)
    self.assertEqual(points, 2)


class WindowSelectionTest(_WandbTestCase):
  """Ranking on the converged level rather than on one lucky batch."""

  def test_the_window_outranks_a_lucky_last_batch(self):
    # This is the reported bug in miniature. "spiker" is worse throughout but
    # happened to stop on a good batch; "steady" is better on the level.
    # Ranked on the final point, "spiker" wins and gets published.
    self.install_wandb([
        _reward_run("spiker", [0.10, 0.10, 0.10, 0.90]),
        _reward_run("steady", [0.50, 0.55, 0.52, 0.50]),
    ])
    final = self.controller().fetch_best_run_details(
        "s", REWARD, goal="maximize", selection="final"
    )
    self.assertEqual(final.run_id, "spiker")

    windowed = self.controller().fetch_best_run_details(
        "s", REWARD, goal="maximize", selection="final_window", window=4
    )
    self.assertEqual(windowed.run_id, "steady")
    self.assertAlmostEqual(windowed.value, 0.5175)
    self.assertEqual(windowed.window_points, 4)
    self.assertTrue(windowed.from_history)
    # The last point is kept alongside, so the report can quote the drift.
    self.assertAlmostEqual(windowed.final_value, 0.50)
    # A window score belongs to no single step.
    self.assertIsNone(windowed.step)

  def test_minimize_goal_uses_the_window_too(self):
    self.install_wandb([
        _reward_run("noisy", [0.9, 0.9, 0.1]),
        _reward_run("low", [0.3, 0.3, 0.3]),
    ])
    winner = self.controller().fetch_best_run_details(
        "s", REWARD, goal="minimize", selection="final_window", window=3
    )
    self.assertEqual(winner.run_id, "low")

  def test_unreadable_history_degrades_to_the_final_value_loudly(self):
    self.install_wandb([
        _FakeRun("a", {REWARD: 0.10}, history=None),
        _FakeRun("b", {REWARD: 0.80}, history=None),
    ])
    notices = []
    winner = self.controller().fetch_best_run_details(
        "s",
        REWARD,
        goal="maximize",
        selection="final_window",
        window=5,
        live_line_callback=notices.append,
    )
    # Still ranked - discarding the sweep would be worse - but the downgrade
    # is recorded on the score and announced.
    self.assertEqual(winner.run_id, "b")
    self.assertFalse(winner.from_history)
    self.assertIsNone(winner.window_points)
    self.assertIn("final logged value", winner.describe_selection())
    self.assertTrue(any("WARNING" in line for line in notices))

  def test_describe_selection_names_the_window(self):
    score = RunScore(
        run_id="r",
        value=0.5,
        final_value=0.4,
        selection="final_window",
        from_history=True,
        window_points=10,
    )
    described = score.describe_selection()
    self.assertIn("last 10 logged points", described)
    self.assertIn("0.40000", described)

  def test_dry_run_exercises_the_window_path(self):
    controller = SweepController(entity="e", project="p", dry_run=True)
    score = controller.fetch_best_run_details(
        "s", REWARD, goal="maximize", selection="final_window", window=7
    )
    self.assertEqual(score.selection, "final_window")
    self.assertEqual(score.window_points, 7)
    self.assertTrue(score.from_history)


class CandidatePoolTest(_WandbTestCase):
  """A winner must say how large the field it beat actually was."""

  METRIC = "eval/loss"

  def test_a_crashed_run_is_dropped_and_counted(self):
    self.install_wandb([
        _FakeRun("ok1", {self.METRIC: 0.5}),
        _FakeRun("ok2", {self.METRIC: 0.4}),
        _FakeRun("dead", {self.METRIC: 0.1}, state="crashed"),
    ])
    winner = self.controller().fetch_best_run_details(
        "s", self.METRIC, goal="minimize", selection="final"
    )
    # The crashed run had the best number and must still not win: 0.1 is an
    # arbitrary point in a run that never finished.
    self.assertEqual(winner.run_id, "ok2")
    self.assertEqual(winner.pool_total, 3)
    self.assertEqual(winner.pool_scored, 2)
    self.assertEqual(winner.pool_dropped, {"crashed": 1})
    described = winner.describe_pool()
    self.assertIn("2 of 3", described)
    self.assertIn("1 crashed", described)

  def test_killed_and_preempted_are_named_explicitly(self):
    self.install_wandb([
        _FakeRun("ok", {self.METRIC: 0.5}),
        _FakeRun("k", {self.METRIC: 0.1}, state="killed"),
        _FakeRun("p", {self.METRIC: 0.1}, state="preempted"),
    ])
    winner = self.controller().fetch_best_run_details(
        "s", self.METRIC, goal="minimize", selection="final"
    )
    self.assertEqual(winner.run_id, "ok")
    self.assertEqual(winner.pool_dropped, {"killed": 1, "preempted": 1})

  def test_an_unknown_state_is_excluded_rather_than_trusted(self):
    self.install_wandb([
        _FakeRun("ok", {self.METRIC: 0.5}),
        _FakeRun("weird", {self.METRIC: 0.01}, state="teleported"),
    ])
    winner = self.controller().fetch_best_run_details(
        "s", self.METRIC, goal="minimize", selection="final"
    )
    self.assertEqual(winner.run_id, "ok")
    self.assertIn("in state 'teleported'", winner.pool_dropped)

  def test_a_run_missing_the_metric_is_counted_as_dropped(self):
    self.install_wandb([
        _FakeRun("ok", {self.METRIC: 0.5}),
        _FakeRun("silent", {"train/loss": 0.2}),
    ])
    winner = self.controller().fetch_best_run_details(
        "s", self.METRIC, goal="minimize", selection="final"
    )
    self.assertEqual(winner.pool_scored, 1)
    self.assertIn(f"logged no '{self.METRIC}'", winner.pool_dropped)

  def test_unfinished_runs_are_recorded_when_finished_ones_exist(self):
    self.install_wandb([
        _FakeRun("done", {self.METRIC: 0.5}),
        _FakeRun("still_going", {self.METRIC: 0.1}, state="running"),
    ])
    winner = self.controller().fetch_best_run_details(
        "s", self.METRIC, goal="minimize", selection="final"
    )
    self.assertEqual(winner.run_id, "done")
    self.assertEqual(winner.pool_dropped, {"unfinished": 1})

  def test_a_clean_sweep_produces_no_warning(self):
    self.install_wandb([
        _FakeRun("a", {self.METRIC: 0.5}),
        _FakeRun("b", {self.METRIC: 0.4}),
    ])
    winner = self.controller().fetch_best_run_details(
        "s", self.METRIC, goal="minimize", selection="final"
    )
    self.assertEqual(winner.pool_scored, 2)
    self.assertEqual(winner.pool_dropped, {})
    self.assertIsNone(winner.describe_pool())

  def test_all_runs_crashed_still_raises_rather_than_fabricating(self):
    self.install_wandb([
        _FakeRun("a", {self.METRIC: 0.5}, state="crashed"),
        _FakeRun("b", {self.METRIC: 0.4}, state="crashed"),
    ])
    with self.assertRaises(ValueError):
      self.controller().fetch_best_run_details(
          "s", self.METRIC, goal="minimize", selection="final"
      )


class WarningPlumbingTest(unittest.TestCase):
  """Selection warnings must reach the state file alongside trial accounting."""

  def test_both_warning_sources_survive(self):
    execution = SweepExecution(
        trials_done=3,
        trials_total=5,
        outcome="partial",
        warnings=["PERL sweep is INCOMPLETE: 3 of 5 trials finished."],
    )
    fields = execution.stage_result_fields(
        extra_warnings=["PERL: The winner was chosen from 3 of 4 runs."]
    )
    self.assertEqual(len(fields["warnings"]), 2)
    self.assertIn("INCOMPLETE", fields["warnings"][0])
    self.assertIn("chosen from 3 of 4", fields["warnings"][1])

  def test_empty_extras_change_nothing(self):
    execution = SweepExecution(trials_done=5, trials_total=5)
    self.assertEqual(execution.stage_result_fields()["warnings"], [])
    self.assertEqual(
        execution.stage_result_fields(extra_warnings=[None, ""])["warnings"],
        [],
    )


class ConfigTest(unittest.TestCase):
  """The PE-RL stage's ranking configuration."""

  def test_perl_ranks_on_a_window(self):
    config = CampaignConfig.create_default(task_name="ragtruth")
    self.assertEqual(config.perl.selection_strategy, "final_window")
    self.assertGreaterEqual(config.perl.selection_window, 2)

  def test_sft_and_rm_are_untouched(self):
    # They rank on a stable held-out evaluation, where the peak is the
    # checkpoint actually shipped; a window there would be wrong.
    config = CampaignConfig.create_default(task_name="ragtruth")
    self.assertEqual(config.sft.selection_strategy, "best")
    self.assertEqual(config.rm.selection_strategy, "best")

  def test_a_zero_window_is_rejected(self):
    config = CampaignConfig.create_default(task_name="ragtruth")
    config.perl.selection_window = 0
    with self.assertRaises(ValueError) as ctx:
      config.validate()
    self.assertIn("selection_window", str(ctx.exception))

  def test_a_zero_window_is_fine_when_unused(self):
    config = CampaignConfig.create_default(task_name="ragtruth")
    config.perl.selection_strategy = "final"
    config.perl.selection_window = 0
    config.validate()


class AlphaIsNotSweptTest(unittest.TestCase):
  """``reward_penalty_alpha`` rescales the objective, so it cannot be searched."""

  def setUp(self):
    with open(SWEEP_PERL_PATH, "r", encoding="utf-8") as handle:
      self.sweep = yaml.safe_load(handle)

  def test_it_is_absent_from_the_search_space(self):
    self.assertNotIn(
        "reward_penalty_alpha",
        self.sweep.get("parameters", {}),
        "Sweeping reward_penalty_alpha ranks trials on different scales, and "
        "with method: bayes teaches the optimizer to weaken the hallucination "
        "penalty in order to raise the reward it is scored on.",
    )

  def test_every_trial_is_pinned_to_the_same_value(self):
    pinned = [
        arg
        for arg in self.sweep.get("command", [])
        if isinstance(arg, str) and arg.startswith("--reward_penalty_alpha=")
    ]
    self.assertEqual(len(pinned), 1)
    self.assertEqual(float(pinned[0].split("=", 1)[1]), REWARD_PENALTY_ALPHA)

  def test_the_objective_is_still_the_reward(self):
    # Guards the premise of the test above: if the sweep ever ranks on
    # something alpha does not touch, this restriction can be revisited.
    self.assertEqual(self.sweep["metric"]["name"], REWARD)


class RetrainPinsAlphaTest(unittest.TestCase):
  """The published winner must be trained with the reward its trials used."""

  def _plan(self, best_params, tunable_keys):
    manager = ModelManager(user="leobianco", dry_run=True)
    return manager.build_materialization_command(
        stage_name="perl",
        task_name="ragtruth",
        base_model="google/gemma-4-E4B-it",
        best_params=best_params,
        seed=130104,
        sft_model_path="leobianco/sft",
        reward_model_path="leobianco/rm",
        tunable_keys=tunable_keys,
        checkpoint_policy="final",
        eval_steps=50,
        timestamp="2601010000",
    )

  def test_a_stale_winner_alpha_cannot_leak_into_the_retrain(self):
    # A resumed campaign can hand us a W&B config recorded before alpha was
    # removed from the search space. It must be ignored, not replayed.
    plan = self._plan(
        {"learning_rate": 5e-5, "beta": 0.05, "reward_penalty_alpha": 3.0},
        tunable_keys={"learning_rate", "beta", "temperature"},
    )
    command = plan.command
    index = command.index("--reward_penalty_alpha")
    self.assertEqual(float(command[index + 1]), REWARD_PENALTY_ALPHA)
    self.assertEqual(command.count("--reward_penalty_alpha"), 1)

  def test_the_repo_name_does_not_advertise_a_stale_alpha(self):
    plan = self._plan(
        {"learning_rate": 5e-5, "beta": 0.05, "reward_penalty_alpha": 3.0},
        tunable_keys={"learning_rate", "beta"},
    )
    self.assertNotIn("_a3.0", plan.repo_id)

  def test_num_generations_survived_the_pinning(self):
    # Regression guard: the flag list is positional, so inserting a pair in
    # the middle of it is easy to get wrong.
    plan = self._plan({"learning_rate": 5e-5}, tunable_keys={"learning_rate"})
    index = plan.command.index("--num_generations")
    self.assertEqual(plan.command[index + 1], "8")


class ReportTest(unittest.TestCase):
  """The report must say what the headline number actually is."""

  def test_window_selection_is_explained(self):
    result = StageResult(
        status=StageStatus.COMPLETED,
        best_metric_val=0.71,
        final_metric_val=0.62,
        selection_strategy="final_window",
        selection_window=10,
    )
    lines = reporter_mod._selection_lines(result, "reward")
    self.assertTrue(any("last 10 logged points" in line for line in lines))
    self.assertTrue(any("0.62000" in line for line in lines))

  def test_plain_final_selection_stays_silent(self):
    result = StageResult(
        status=StageStatus.COMPLETED,
        best_metric_val=0.71,
        selection_strategy="final",
    )
    self.assertEqual(reporter_mod._selection_lines(result, "reward"), [])

  def test_a_shrunken_pool_reaches_the_markdown_report(self):
    config = CampaignConfig.create_default(task_name="ragtruth", dry_run=True)
    with tempfile.TemporaryDirectory() as temp_dir:
      config.reporting.reports_dir = temp_dir
      state = CampaignState(campaign_id="c", task_name="ragtruth")
      state.stages["perl"] = StageResult(
          status=StageStatus.COMPLETED,
          best_metric_val=0.71,
          warnings=["PERL: The winner was chosen from 3 of 5 runs; 2 crashed."],
      )
      path = reporter_mod.CampaignReporter(
          config, state
      ).generate_markdown_report()
      with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    self.assertIn("chosen from 3 of 5 runs", text)


if __name__ == "__main__":
  unittest.main()
