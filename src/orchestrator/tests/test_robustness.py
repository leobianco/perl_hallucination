"""Regression tests for the correctness fixes applied to the orchestrator.

Each test here maps to a failure mode that would only surface hours into an
unattended campaign, i.e. exactly the cases that cannot be caught by eyeballing
a dry run.
"""

import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from src.orchestrator import model_manager as model_manager_mod
from src.orchestrator.cli import events as events_mod
from src.orchestrator.config import CampaignConfig
from src.orchestrator.model_manager import ModelManager
from src.orchestrator.process import ProcessOutcome
from src.orchestrator.stages.base import BaseStage
from src.orchestrator.stages.base import CampaignContext
from src.orchestrator.stages.sft_stage import SftStage
from src.orchestrator.state import CampaignState
from src.orchestrator.state import StageResult
from src.orchestrator.state import StageStatus
from src.orchestrator.sweep_controller import SweepController


class _RecordingStream:
  """Stand-in for ``stream_subprocess`` capturing the argv it receives."""

  def __init__(self, returncode=0, interrupted=False):
    self.calls = []
    self._outcome = ProcessOutcome(
        returncode=returncode, interrupted=interrupted
    )

  def __call__(self, cmd, on_line=None, stop_requested=None, **kwargs):
    self.calls.append(list(cmd))
    return self._outcome


def _flag_value(cmd, flag):
  """Returns the value following ``flag`` in ``cmd``, or None."""
  if flag not in cmd:
    return None
  return cmd[cmd.index(flag) + 1]


class TestMaterializationAllowlist(unittest.TestCase):
  """The W&B run config is not a valid command line.

  For Hugging Face Trainer runs, ``run.config`` is a merge of the sweep
  parameters, the full ``TrainingArguments`` dump and the model's
  ``config.json``. Replaying all of it would make the training script's
  argument parser abort right after a multi-hour sweep.
  """

  def setUp(self):
    self.original = model_manager_mod.stream_subprocess
    self.stream = _RecordingStream()
    model_manager_mod.stream_subprocess = self.stream

  def tearDown(self):
    model_manager_mod.stream_subprocess = self.original

  def _materialize(self, best_params, tunable_keys=None):
    mgr = ModelManager(user="leobianco", dry_run=False)
    mgr.materialize_and_push(
        stage_name="sft",
        task_name="ragtruth",
        base_model="google/gemma-4-E2B-it",
        best_params=best_params,
        seed=130104,
        tunable_keys=tunable_keys,
    )
    self.assertEqual(len(self.stream.calls), 1)
    return self.stream.calls[0]

  def test_model_config_keys_are_not_forwarded(self):
    cmd = self._materialize(
        {
            "learning_rate": 0.003,
            "num_train_epochs": 2,
            # Injected by WandbCallback from the model's config.json.
            "vocab_size": 262144,
            "rope_theta": 1000000.0,
            "model/num_parameters": 4300000000,
        },
        tunable_keys={"learning_rate", "num_train_epochs"},
    )
    self.assertEqual(_flag_value(cmd, "--learning_rate"), "0.003")
    self.assertEqual(_flag_value(cmd, "--num_train_epochs"), "2")
    for rejected in ("--vocab_size", "--rope_theta", "--model/num_parameters"):
      self.assertNotIn(rejected, cmd)

  def test_containers_and_booleans_are_skipped(self):
    cmd = self._materialize(
        {
            "learning_rate": 0.003,
            "bf16": True,
            "lora_target_modules": ["q_proj", "v_proj"],
            "generation_config": {"top_p": 0.9},
        },
        tunable_keys={"learning_rate", "bf16", "lora_target_modules",
                      "generation_config"},
    )
    self.assertEqual(_flag_value(cmd, "--learning_rate"), "0.003")
    # --bf16 is already set explicitly by the stage; it must not be duplicated.
    self.assertEqual(cmd.count("--bf16"), 1)
    self.assertNotIn("--lora_target_modules", cmd)
    self.assertNotIn("--generation_config", cmd)

  def test_explicit_flags_are_not_overridden(self):
    # --seed is set explicitly by the stage; the run config copy must lose.
    cmd = self._materialize(
        {"learning_rate": 0.003, "seed": 999},
        tunable_keys={"learning_rate", "seed"},
    )
    self.assertEqual(_flag_value(cmd, "--seed"), "130104")
    self.assertEqual(cmd.count("--seed"), 1)

  def test_non_zero_exit_raises(self):
    model_manager_mod.stream_subprocess = _RecordingStream(returncode=1)
    mgr = ModelManager(user="leobianco", dry_run=False)
    with self.assertRaises(RuntimeError):
      mgr.materialize_and_push(
          stage_name="sft",
          task_name="ragtruth",
          base_model="google/gemma-4-E2B-it",
          best_params={"learning_rate": 0.003},
          tunable_keys={"learning_rate"},
      )

  def test_interruption_raises_instead_of_publishing(self):
    model_manager_mod.stream_subprocess = _RecordingStream(
        returncode=-15, interrupted=True
    )
    mgr = ModelManager(user="leobianco", dry_run=False)
    with self.assertRaises(RuntimeError):
      mgr.materialize_and_push(
          stage_name="sft",
          task_name="ragtruth",
          base_model="google/gemma-4-E2B-it",
          best_params={"learning_rate": 0.003},
          tunable_keys={"learning_rate"},
      )


class _FakeSweepController(SweepController):
  """Counts sweep registrations so we can assert on reuse."""

  def __init__(self):
    super().__init__(entity="e", project="p", dry_run=True)
    self.created = 0

  def create_sweep(self, sweep_config, project=None):
    self.created += 1
    return f"new_sweep_{self.created}"


class _ProbeStage(BaseStage):

  @property
  def name(self):
    return "sft"

  def execute(self, live_line_callback=None, stop_requested_callback=None):
    raise NotImplementedError


class TestSweepResumption(unittest.TestCase):
  """A crash mid-sweep must not throw away the trials already paid for."""

  def _stage(self, temp_dir, state):
    config = CampaignConfig.create_default(task_name="ragtruth", dry_run=True)
    controller = _FakeSweepController()
    context = CampaignContext(
        config=config,
        state=state,
        sweep_controller=controller,
        model_manager=ModelManager(user="leobianco", dry_run=True),
        state_path=os.path.join(temp_dir, "state.json"),
    )
    return _ProbeStage(context), controller, context

  def test_existing_incomplete_sweep_is_reused(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      state = CampaignState(campaign_id="c", task_name="ragtruth")
      state.stages["sft"] = StageResult(
          status=StageStatus.RUNNING, sweep_id="old_sweep_42"
      )
      stage, controller, _ = self._stage(temp_dir, state)

      self.assertEqual(stage.resolve_sweep_id({"parameters": {}}),
                       "old_sweep_42")
      self.assertEqual(controller.created, 0)

  def test_completed_stage_starts_a_fresh_sweep(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      state = CampaignState(campaign_id="c", task_name="ragtruth")
      state.stages["sft"] = StageResult(
          status=StageStatus.COMPLETED, sweep_id="old_sweep_42"
      )
      stage, controller, _ = self._stage(temp_dir, state)

      self.assertEqual(stage.resolve_sweep_id({"parameters": {}}),
                       "new_sweep_1")
      self.assertEqual(controller.created, 1)

  def test_new_sweep_id_is_persisted_immediately(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      state = CampaignState(campaign_id="c", task_name="ragtruth")
      stage, _, context = self._stage(temp_dir, state)

      sweep_id = stage.resolve_sweep_id({"parameters": {}})

      with open(context.state_path, "r", encoding="utf-8") as f:
        persisted = json.load(f)
      self.assertEqual(persisted["stages"]["sft"]["sweep_id"], sweep_id)

  def test_tunable_keys_come_from_the_sweep_parameters(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      state = CampaignState(campaign_id="c", task_name="ragtruth")
      stage, _, _ = self._stage(temp_dir, state)

      keys = stage.tunable_keys(
          {"parameters": {"learning_rate": {}, "lora_r": {}}}
      )
      self.assertEqual(keys, {"learning_rate", "lora_r"})
      self.assertEqual(stage.tunable_keys({}), set())


class _VanishingSweepController(_FakeSweepController):
  """Controller whose recorded sweep is reported missing or unverifiable."""

  def __init__(self, exists):
    super().__init__()
    self._exists = exists
    self.existence_checks = []

  def sweep_exists(self, sweep_id):
    self.existence_checks.append(sweep_id)
    return self._exists


class TestDeletedSweepRecovery(unittest.TestCase):
  """A sweep deleted from the W&B UI must not wedge the next run.

  The state file keeps pointing at the old sweep id, so resuming used to
  hand a dead id to ``wandb agent`` and fail the campaign. There is nothing
  to salvage in that situation: registering a new sweep is the only useful
  behaviour.
  """

  def _stage(self, temp_dir, state, exists):
    config = CampaignConfig.create_default(task_name="bosch", dry_run=True)
    controller = _VanishingSweepController(exists)
    context = CampaignContext(
        config=config,
        state=state,
        sweep_controller=controller,
        model_manager=ModelManager(user="leobianco", dry_run=True),
        state_path=os.path.join(temp_dir, "state.json"),
    )
    return _ProbeStage(context), controller, context

  def _running_state(self):
    state = CampaignState(campaign_id="c", task_name="bosch")
    state.stages["sft"] = StageResult(
        status=StageStatus.RUNNING,
        sweep_id="deleted_sweep_42",
        sweep_name="BOSCH gemma-4-E2B-it SFT Sweep #1",
    )
    return state

  def test_deleted_sweep_triggers_a_fresh_registration(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      state = self._running_state()
      stage, controller, _ = self._stage(temp_dir, state, exists=False)

      sweep_id = stage.resolve_sweep_id({"parameters": {}})

      self.assertEqual(controller.existence_checks, ["deleted_sweep_42"])
      self.assertEqual(sweep_id, "new_sweep_1")
      self.assertEqual(controller.created, 1)

  def test_deleted_sweep_warns_the_operator(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      state = self._running_state()
      stage, _, _ = self._stage(temp_dir, state, exists=False)
      lines = []

      stage.resolve_sweep_id({"parameters": {}}, live_line_callback=lines.append)

      self.assertTrue(
          any("no longer exists" in line for line in lines),
          f"expected a warning about the missing sweep, got {lines}",
      )

  def test_stale_pointer_is_cleared_from_disk(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      state = self._running_state()
      stage, _, context = self._stage(temp_dir, state, exists=False)

      sweep_id = stage.resolve_sweep_id({"parameters": {}})

      with open(context.state_path, "r", encoding="utf-8") as f:
        persisted = json.load(f)
      self.assertEqual(persisted["stages"]["sft"]["sweep_id"], sweep_id)
      self.assertNotEqual(
          persisted["stages"]["sft"]["sweep_id"], "deleted_sweep_42"
      )

  def test_unverifiable_sweep_is_kept(self):
    # W&B unreachable: discarding the sweep would throw away paid-for
    # trials, so the recorded id must survive.
    with tempfile.TemporaryDirectory() as temp_dir:
      state = self._running_state()
      stage, controller, _ = self._stage(temp_dir, state, exists=None)

      sweep_id = stage.resolve_sweep_id({"parameters": {}})

      self.assertEqual(sweep_id, "deleted_sweep_42")
      self.assertEqual(controller.created, 0)

  def test_live_sweep_is_still_reused(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      state = self._running_state()
      stage, controller, _ = self._stage(temp_dir, state, exists=True)

      sweep_id = stage.resolve_sweep_id({"parameters": {}})

      self.assertEqual(sweep_id, "deleted_sweep_42")
      self.assertEqual(controller.created, 0)


class TestSweepExistenceProbe(unittest.TestCase):
  """``sweep_exists`` must tell 'deleted' apart from 'cannot reach W&B'."""

  def _controller_raising(self, exc):
    controller = SweepController(entity="e", project="p", dry_run=False)
    fake_wandb = types.ModuleType("wandb")

    class _Api:

      def sweep(self, _sweep_id):
        raise exc

    fake_wandb.Api = _Api
    self.enterContext(mock.patch.dict(sys.modules, {"wandb": fake_wandb}))
    return controller

  def test_missing_sweep_reports_false(self):
    controller = self._controller_raising(
        ValueError("Could not find sweep e/p/gone123")
    )
    self.assertIs(controller.sweep_exists("gone123"), False)

  def test_http_404_reports_false(self):
    controller = self._controller_raising(RuntimeError("404 Client Error"))
    self.assertIs(controller.sweep_exists("gone123"), False)

  def test_transient_error_reports_unknown(self):
    controller = self._controller_raising(
        ConnectionError("503 Service Unavailable")
    )
    self.assertIsNone(controller.sweep_exists("maybe123"))

  def test_dry_run_short_circuits(self):
    controller = SweepController(entity="e", project="p", dry_run=True)
    self.assertIs(controller.sweep_exists("anything"), True)

  def test_empty_id_is_missing(self):
    controller = SweepController(entity="e", project="p", dry_run=True)
    self.assertIs(controller.sweep_exists(""), False)

  def test_qualify_sweep_id_expands_short_forms(self):
    controller = SweepController(entity="e", project="p", dry_run=True)
    self.assertEqual(controller.qualify_sweep_id("abc"), "e/p/abc")
    self.assertEqual(controller.qualify_sweep_id("proj/abc"), "e/proj/abc")
    self.assertEqual(
        controller.qualify_sweep_id("ent/proj/abc"), "ent/proj/abc"
    )


class _FakeRun:

  def __init__(
      self, run_id, summary, state="finished", config=None, history=None
  ):
    self.id = run_id
    self.summary = summary
    self.state = state
    self.config = config or {}
    #: Rows as ``scan_history`` would return them. None means the run
    #: exposes no readable history at all.
    self._history = history

  def scan_history(self, keys=None):
    if self._history is None:
      raise RuntimeError("history unavailable")
    if not keys:
      return list(self._history)
    # Mirrors the real API: only rows carrying every requested key.
    matching = []
    for row in self._history:
      if all(key in row for key in keys):
        matching.append(row)
    return matching


class _FakeSweep:

  def __init__(self, runs):
    self.runs = runs


class _FakeApi:

  def __init__(self, sweep):
    self._sweep = sweep

  def sweep(self, _path):
    return self._sweep


class TestFetchBestRunIsStrict(unittest.TestCase):
  """Never promote an arbitrary run with a fabricated metric of 0.0."""

  def _install_wandb(self, runs):
    module = types.ModuleType("wandb")
    module.Api = lambda: _FakeApi(_FakeSweep(runs))
    sys.modules["wandb"] = module
    self.addCleanup(sys.modules.pop, "wandb", None)

  def test_missing_metric_raises_with_observed_keys(self):
    self._install_wandb([
        _FakeRun("r1", {"train/loss": 1.2, "_step": 10}),
        _FakeRun("r2", {"train/loss": 1.1, "_step": 10}),
    ])
    controller = SweepController(entity="e", project="p", dry_run=False)
    with self.assertRaises(ValueError) as ctx:
      controller.fetch_best_run("sweep123", "eval/roc_auc", goal="maximize")
    message = str(ctx.exception)
    self.assertIn("eval/roc_auc", message)
    self.assertIn("train/loss", message)

  def test_finished_runs_win_over_running_ones(self):
    self._install_wandb([
        _FakeRun("running", {"eval/loss": 0.10}, state="running"),
        _FakeRun("done", {"eval/loss": 0.50}, state="finished",
                 config={"learning_rate": 0.001}),
    ])
    controller = SweepController(entity="e", project="p", dry_run=False)
    run_id, value, config = controller.fetch_best_run(
        "sweep123", "eval/loss", goal="minimize"
    )
    self.assertEqual(run_id, "done")
    self.assertAlmostEqual(value, 0.50)
    self.assertEqual(config, {"learning_rate": 0.001})

  def test_empty_sweep_raises(self):
    self._install_wandb([])
    controller = SweepController(entity="e", project="p", dry_run=False)
    with self.assertRaises(ValueError):
      controller.fetch_best_run("sweep123", "eval/loss")


class TestSelectionStrategy(unittest.TestCase):
  """Trials must be ranked by the score of the checkpoint we actually ship.

  The materialization publishes the best-scoring checkpoint, so a trial that
  bottoms out early and then overfits must be judged on its peak, not on the
  overfitting tail that ``--load_best_model_at_end`` throws away.
  """

  def _install_wandb(self, runs):
    module = types.ModuleType("wandb")
    module.Api = lambda: _FakeApi(_FakeSweep(runs))
    sys.modules["wandb"] = module
    self.addCleanup(sys.modules.pop, "wandb", None)

  def _controller(self):
    return SweepController(entity="e", project="p", dry_run=False)

  def test_final_selection_ignores_the_peak(self):
    # "overfitter" touches 0.10 at step 20 and ends at 0.90; "steady" ends
    # at 0.40. Ranked on the last value, "steady" wins.
    self._install_wandb([
        _FakeRun(
            "overfitter",
            {"eval/loss": 0.90},
            history=[
                {"eval/loss": 0.80, "_step": 10},
                {"eval/loss": 0.10, "_step": 20},
                {"eval/loss": 0.90, "_step": 30},
            ],
        ),
        _FakeRun(
            "steady",
            {"eval/loss": 0.40},
            history=[
                {"eval/loss": 0.60, "_step": 10},
                {"eval/loss": 0.40, "_step": 20},
            ],
        ),
    ])
    winner = self._controller().fetch_best_run_details(
        "s", "eval/loss", goal="minimize", selection="final"
    )
    self.assertEqual(winner.run_id, "steady")
    self.assertAlmostEqual(winner.value, 0.40)
    self.assertIsNone(winner.step)

  def test_best_selection_recovers_the_peak(self):
    self._install_wandb([
        _FakeRun(
            "overfitter",
            {"eval/loss": 0.90},
            history=[
                {"eval/loss": 0.80, "_step": 10},
                {"eval/loss": 0.10, "_step": 20},
                {"eval/loss": 0.90, "_step": 30},
            ],
        ),
        _FakeRun(
            "steady",
            {"eval/loss": 0.40},
            history=[
                {"eval/loss": 0.60, "_step": 10},
                {"eval/loss": 0.40, "_step": 20},
            ],
        ),
    ])
    winner = self._controller().fetch_best_run_details(
        "s", "eval/loss", goal="minimize", selection="best"
    )
    self.assertEqual(winner.run_id, "overfitter")
    self.assertAlmostEqual(winner.value, 0.10)
    self.assertEqual(winner.step, 20)
    self.assertAlmostEqual(winner.final_value, 0.90)
    self.assertTrue(winner.from_history)

  def test_best_selection_maximizes_for_roc_auc(self):
    self._install_wandb([
        _FakeRun(
            "rm",
            {"eval/roc_auc": 0.71},
            history=[
                {"eval/roc_auc": 0.65, "_step": 50},
                {"eval/roc_auc": 0.93, "_step": 100},
                {"eval/roc_auc": 0.71, "_step": 150},
            ],
        ),
    ])
    winner = self._controller().fetch_best_run_details(
        "s", "eval/roc_auc", goal="maximize", selection="best"
    )
    self.assertAlmostEqual(winner.value, 0.93)
    self.assertEqual(winner.step, 100)

  def test_a_run_that_ends_at_its_peak_reports_no_step(self):
    # Reporting a step here would suggest early stopping bought something.
    self._install_wandb([
        _FakeRun(
            "monotonic",
            {"eval/loss": 0.20},
            history=[
                {"eval/loss": 0.60, "_step": 10},
                {"eval/loss": 0.20, "_step": 20},
            ],
        ),
    ])
    winner = self._controller().fetch_best_run_details(
        "s", "eval/loss", goal="minimize", selection="best"
    )
    self.assertAlmostEqual(winner.value, 0.20)
    self.assertIsNone(winner.step)
    self.assertIn("also the final one", winner.describe_selection())

  def test_unreadable_history_falls_back_to_the_summary(self):
    # Losing a whole sweep because an API for a refinement is unavailable
    # would be far worse than scoring the trial on its final value.
    self._install_wandb([
        _FakeRun("no_history", {"eval/loss": 0.33}, history=None),
    ])
    winner = self._controller().fetch_best_run_details(
        "s", "eval/loss", goal="minimize", selection="best"
    )
    self.assertEqual(winner.run_id, "no_history")
    self.assertAlmostEqual(winner.value, 0.33)
    self.assertFalse(winner.from_history)
    self.assertIn("history unavailable", winner.describe_selection())

  def test_fallback_warns_on_the_live_line(self):
    self._install_wandb([
        _FakeRun("no_history", {"eval/loss": 0.33}, history=None),
    ])
    seen = []
    self._controller().fetch_best_run_details(
        "s",
        "eval/loss",
        goal="minimize",
        selection="best",
        live_line_callback=seen.append,
    )
    self.assertTrue(
        any("best evaluation step" in line for line in seen),
        f"no warning about the fallback in {seen}",
    )

  def test_nan_and_inf_history_rows_are_ignored(self):
    # A diverged trial logging inf would otherwise win a maximize sweep.
    self._install_wandb([
        _FakeRun(
            "diverged",
            {"eval/roc_auc": 0.55},
            history=[
                {"eval/roc_auc": float("inf"), "_step": 10},
                {"eval/roc_auc": float("nan"), "_step": 20},
                {"eval/roc_auc": 0.60, "_step": 30},
                {"eval/roc_auc": 0.55, "_step": 40},
            ],
        ),
    ])
    winner = self._controller().fetch_best_run_details(
        "s", "eval/roc_auc", goal="maximize", selection="best"
    )
    self.assertAlmostEqual(winner.value, 0.60)
    self.assertEqual(winner.step, 30)

  def test_history_key_alias_matches_the_summary_key(self):
    # HF Trainer logs eval_loss; the sweep YAML says eval/loss.
    self._install_wandb([
        _FakeRun(
            "aliased",
            {"eval_loss": 0.50},
            history=[
                {"eval_loss": 0.15, "_step": 10},
                {"eval_loss": 0.50, "_step": 20},
            ],
        ),
    ])
    winner = self._controller().fetch_best_run_details(
        "s", "eval/loss", goal="minimize", selection="best"
    )
    self.assertAlmostEqual(winner.value, 0.15)
    self.assertEqual(winner.step, 10)

  def test_missing_step_column_still_yields_the_peak(self):
    self._install_wandb([
        _FakeRun(
            "no_steps",
            {"eval/loss": 0.50},
            history=[{"eval/loss": 0.15}, {"eval/loss": 0.50}],
        ),
    ])
    winner = self._controller().fetch_best_run_details(
        "s", "eval/loss", goal="minimize", selection="best"
    )
    self.assertAlmostEqual(winner.value, 0.15)
    self.assertIsNone(winner.step)


class TestStageSelectionDefaults(unittest.TestCase):
  """SFT and RM early stop; PE-RL deliberately does not."""

  def setUp(self):
    self.config = CampaignConfig.create_default(task_name="npov")

  def test_sft_and_rm_rank_trials_on_their_best_step(self):
    self.assertEqual(self.config.sft.selection_strategy, "best")
    self.assertEqual(self.config.rm.selection_strategy, "best")

  def test_perl_ranks_trials_on_the_end_of_training_not_on_a_peak(self):
    # The PE-RL reward is a per-step training signal over 8 sampled
    # generations: its maximum is a lucky batch, not a better policy. But
    # the single last step is a lucky batch too, so the trials are ranked on
    # the mean of the tail - the level, which is what the smoothed W&B curve
    # shows.
    self.assertEqual(self.config.perl.selection_strategy, "final_window")
    self.assertGreaterEqual(self.config.perl.selection_window, 2)

  def test_the_default_repo_checkpoint_matches_the_ranking(self):
    # Both checkpoints are published either way, but the one the repo serves
    # by default must be the one the trials were ranked on, otherwise the
    # metric quoted in the report describes a model nobody loads.
    #
    # Compared by which end of training each refers to rather than by string
    # equality: "final_window" ranks on the end of the run, so it agrees
    # with a "final" checkpoint even though the words differ.
    served_by = {"final": "final", "final_window": "final", "best": "best"}
    for stage in ("sft", "rm", "perl"):
      with self.subTest(stage=stage):
        stage_cfg = getattr(self.config, stage)
        self.assertEqual(
            stage_cfg.checkpoint_policy,
            served_by[stage_cfg.selection_strategy],
        )


  def test_perl_serves_the_final_checkpoint_by_default(self):
    self.assertEqual(self.config.perl.checkpoint_policy, "final")

  def test_sft_and_rm_serve_the_best_checkpoint_by_default(self):
    self.assertEqual(self.config.sft.checkpoint_policy, "best")
    self.assertEqual(self.config.rm.checkpoint_policy, "best")

  def test_sft_materialization_evaluates_as_often_as_its_sweep(self):
    # scripts/sweep_sft.yaml uses --eval_steps=10. Retraining once per
    # epoch would make the selected peak unreachable.
    self.assertEqual(self.config.sft.materialization_eval_steps, 10)
    self.assertEqual(self.config.rm.materialization_eval_steps, 50)

  def test_perl_materialization_can_locate_its_best_checkpoint(self):
    # PE-RL serves the final checkpoint, but the best one is still published
    # as a companion, so the cadence still has to mirror the sweep.
    self.assertEqual(self.config.perl.materialization_eval_steps, 50)

  def test_validate_rejects_an_unknown_strategy(self):
    self.config.rm.selection_strategy = "peak"
    with self.assertRaises(ValueError) as ctx:
      self.config.validate()
    self.assertIn("selection_strategy", str(ctx.exception))

  def test_validate_rejects_an_unknown_policy(self):
    self.config.rm.checkpoint_policy = "penultimate"
    with self.assertRaises(ValueError) as ctx:
      self.config.validate()
    self.assertIn("checkpoint_policy", str(ctx.exception))

  def test_ranking_on_peak_while_serving_the_last_is_allowed(self):
    # It used to be rejected because the best checkpoint was then never
    # pushed. It is published under "best/" now, so the combination only
    # changes which one is the default and is a legitimate choice.
    self.config.rm.checkpoint_policy = "final"  # selection is "best"
    self.config.validate()

  def test_the_default_campaign_validates(self):
    self.config.validate()


class TestMaterializationCheckpointFlags(unittest.TestCase):
  """The retraining must be able to reproduce the checkpoint we selected."""

  def setUp(self):
    self.manager = ModelManager(user="u", dry_run=True)

  def _flags(self, stage, policy="best", eval_steps=None):
    return self.manager._checkpointing_flags(stage, policy, eval_steps)  # pylint: disable=protected-access

  def _value_of(self, flags, flag):
    return flags[flags.index(flag) + 1]

  def test_best_policy_loads_the_best_checkpoint(self):
    flags = self._flags("rm")
    self.assertEqual(self._value_of(flags, "--load_best_model_at_end"), "True")
    self.assertEqual(self._value_of(flags, "--metric_for_best_model"), "roc_auc")
    self.assertEqual(self._value_of(flags, "--greater_is_better"), "True")

  def test_final_policy_disables_early_stopping(self):
    flags = self._flags("perl", policy="final")
    self.assertEqual(self._value_of(flags, "--load_best_model_at_end"), "False")

  def test_final_policy_still_declares_the_best_metric(self):
    # Without metric_for_best_model the trainer never populates
    # TrainerState.best_model_checkpoint, so the best checkpoint would be
    # neither protected from rotation nor publishable under "best/".
    for stage, metric, greater in (
        ("sft", "loss", "False"),
        ("rm", "roc_auc", "True"),
        ("perl", "rewards/reward_fn/mean", "True"),
    ):
      with self.subTest(stage=stage):
        flags = self._flags(stage, policy="final", eval_steps=50)
        self.assertEqual(
            self._value_of(flags, "--metric_for_best_model"), metric
        )
        self.assertEqual(self._value_of(flags, "--greater_is_better"), greater)

  def test_eval_and_save_cadence_are_kept_identical(self):
    # load_best_model_at_end requires the strategies to match and save_steps
    # to be a multiple of eval_steps.
    flags = self._flags("sft", eval_steps=10)
    self.assertEqual(self._value_of(flags, "--eval_strategy"), "steps")
    self.assertEqual(self._value_of(flags, "--save_strategy"), "steps")
    self.assertEqual(self._value_of(flags, "--eval_steps"), "10")
    self.assertEqual(self._value_of(flags, "--save_steps"), "10")

  def test_no_cadence_falls_back_to_epochs(self):
    flags = self._flags("sft")
    self.assertEqual(self._value_of(flags, "--eval_strategy"), "epoch")
    self.assertEqual(self._value_of(flags, "--save_strategy"), "epoch")
    self.assertNotIn("--eval_steps", flags)

  def test_unknown_policy_is_rejected(self):
    with self.assertRaises(ValueError):
      self._flags("sft", policy="whatever")




class TestEvalStageSeed(unittest.TestCase):
  """Evaluation must subsample the same 1000 examples as the shell script."""

  def test_eval_seed_is_independent_from_the_training_seed(self):
    config = CampaignConfig.create_default(task_name="ragtruth")
    # scripts/evaluator.sh uses SEED=12345; the training seed is 130104.
    self.assertEqual(config.eval.seed, 12345)
    self.assertNotEqual(config.eval.seed, config.seed)


class TestSweepTimeoutBudgets(unittest.TestCase):
  """A fixed timeout would cut large sweeps short and report success."""

  def test_timeout_covers_the_standard_preset(self):
    config = CampaignConfig.create_default(
        task_name="ragtruth", sft_runs=30, rm_runs=30, perl_runs=10
    )
    # 30 SFT trials at ~12 min each need far more than the old 240 min cap.
    self.assertGreater(config.sft.timeout_minutes, 30 * 12)
    self.assertGreater(config.rm.timeout_minutes, 30 * 8)
    self.assertGreater(config.perl.timeout_minutes, 10 * 35)

  def test_small_budgets_keep_the_floor(self):
    config = CampaignConfig.create_default(
        task_name="ragtruth", sft_runs=1, rm_runs=1, perl_runs=1
    )
    self.assertEqual(config.sft.timeout_minutes, 240)
    self.assertEqual(config.perl.timeout_minutes, 240)


class _InterruptibleStream:
  """``stream_subprocess`` stand-in that honors the stop predicate.

  Mirrors the real loop closely enough to reproduce the campaign-killer: a
  predicate that is already True when the subprocess starts yields an
  ``interrupted`` outcome, which ``materialize_and_push`` turns into a
  non-retryable ``RuntimeError``.
  """

  def __init__(self):
    self.calls = []

  def __call__(self, cmd, on_line=None, stop_requested=None, **kwargs):
    self.calls.append(list(cmd))
    if stop_requested is not None and stop_requested():
      return ProcessOutcome(returncode=-15, interrupted=True)
    return ProcessOutcome(returncode=0)


class TestAdvanceDoesNotKillMaterialization(unittest.TestCase):
  """``[a]`` seals the sweep; it must not kill the winner's retraining.

  Regression test for a campaign that failed with "Materialization of the
  best sft model was interrupted before the checkpoint could be pushed":
  the advance flag stayed set while the stage retrained the best trial, so
  the training subprocess was terminated on its first poll.
  """

  def setUp(self):
    super().setUp()
    self.original = model_manager_mod.stream_subprocess
    self.stream = _InterruptibleStream()
    model_manager_mod.stream_subprocess = self.stream
    self.addCleanup(
        setattr, model_manager_mod, "stream_subprocess", self.original
    )

    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    # materialize_and_push creates ./checkpoints/...; keep it out of the tree.
    cwd = os.getcwd()
    os.chdir(self.temp.name)
    self.addCleanup(os.chdir, cwd)

    self.sweep_yaml = os.path.join(self.temp.name, "sweep_sft.yaml")
    with open(self.sweep_yaml, "w", encoding="utf-8") as f:
      f.write(
          "program: src/writer_sft.py\n"
          "method: bayes\n"
          "metric:\n"
          "  name: eval/loss\n"
          "  goal: minimize\n"
          "parameters:\n"
          "  learning_rate:\n"
          "    values: [0.001]\n"
      )

  def _stage(self, controls):
    config = CampaignConfig.create_default(task_name="ragtruth", sft_runs=15)
    config.sft.sweep_config_path = self.sweep_yaml
    context = CampaignContext(
        config=config,
        state=CampaignState(campaign_id="c", task_name="ragtruth"),
        sweep_controller=_FakeSweepController(),
        # Not a dry run: this is the code path that pushes the checkpoint.
        model_manager=ModelManager(user="leobianco", dry_run=False),
        state_path=os.path.join(self.temp.name, "state.json"),
        abort_requested_callback=lambda: controls.abort_requested,
    )
    return SftStage(context)

  def test_advance_lets_the_winner_be_retrained_and_pushed(self):
    controls = events_mod.ControlSignals()
    controls.request_advance()
    stage = self._stage(controls)

    result = stage.execute(
        stop_requested_callback=controls.should_interrupt_stage
    )

    self.assertEqual(result.status, StageStatus.COMPLETED)
    self.assertTrue(result.model_repo_id)
    self.assertEqual(len(self.stream.calls), 1)

  def test_abort_still_stops_the_materialization(self):
    controls = events_mod.ControlSignals()
    controls.request_abort()
    stage = self._stage(controls)

    with self.assertRaises(RuntimeError) as ctx:
      stage.execute(stop_requested_callback=controls.should_interrupt_stage)
    self.assertIn("interrupted", str(ctx.exception))

  def test_materialization_predicate_ignores_advance_and_stop(self):
    controls = events_mod.ControlSignals()
    stage = self._stage(controls)
    predicate = stage.materialization_stop_callback()

    controls.request_advance()
    self.assertFalse(predicate())
    controls.request_stop()
    self.assertFalse(predicate())
    controls.request_abort()
    self.assertTrue(predicate())


if __name__ == "__main__":
  unittest.main()
