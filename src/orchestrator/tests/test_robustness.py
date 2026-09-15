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

from src.orchestrator import model_manager as model_manager_mod
from src.orchestrator.config import CampaignConfig
from src.orchestrator.model_manager import ModelManager
from src.orchestrator.process import ProcessOutcome
from src.orchestrator.stages.base import BaseStage
from src.orchestrator.stages.base import CampaignContext
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


class _FakeRun:

  def __init__(self, run_id, summary, state="finished", config=None):
    self.id = run_id
    self.summary = summary
    self.state = state
    self.config = config or {}


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


if __name__ == "__main__":
  unittest.main()
