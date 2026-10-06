"""Every PE-RL run an orchestrator campaign launches gets continual evaluation.

Sweep trials and the winner's retraining both run with the ``--continual_eval*``
flags of ``PerlStage.continual_eval_flags``: the autorater scores land on each
run's W&B page, and nothing the TUI shows changes. A retraining that fails
between two pauses must never publish one of its pause checkpoints, which hold
earlier policies.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from src.orchestrator.config import CampaignConfig
from src.orchestrator.config import RobustnessConfig
from src.orchestrator.model_manager import CONTINUAL_EVAL_STATUS_FILENAME
from src.orchestrator.model_manager import ModelManager
from src.orchestrator.model_manager import continual_eval_training_completed
from src.orchestrator.process import ProcessOutcome
from src.orchestrator.retry import is_retryable
from src.orchestrator.stages.base import CampaignContext
from src.orchestrator.stages.perl_stage import PerlStage
from src.orchestrator.state import CampaignState
from src.orchestrator.state import StageResult
from src.orchestrator.state import StageStatus
from src.orchestrator.sweep_controller import RunScore

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SWEEP_PERL_YAML = os.path.join(_REPO_ROOT, "scripts", "sweep_perl.yaml")


def _flag_values(command, name):
  """Returns every value of ``--name=value`` in a sweep command."""
  prefix = f"--{name}="
  return [
      arg[len(prefix):]
      for arg in command
      if isinstance(arg, str) and arg.startswith(prefix)
  ]


class PerlStageContinualEvalTest(unittest.TestCase):
  """Sweep trials and the winner's retraining get the same flags."""

  def _execute(self, calibrated_threshold=None, configure=None):
    """Runs the stage against mocks; returns it, the trial command, the push."""
    config = CampaignConfig.create_default(task_name="npov", dry_run=True)
    config.perl.sweep_config_path = _SWEEP_PERL_YAML
    if configure is not None:
      configure(config)
    state = CampaignState(campaign_id="c", task_name="npov")
    for done, repo in (("sft", "u/npov_SFT"), ("rm", "u/npov_RM")):
      state.record_stage_result(
          done, StageResult(status=StageStatus.COMPLETED, model_repo_id=repo)
      )
    if calibrated_threshold is not None:
      state.stages["autorater"] = StageResult(
          status=StageStatus.COMPLETED,
          metrics={"autorater/best_threshold": calibrated_threshold},
      )
    controller = mock.MagicMock()
    controller.create_sweep.return_value = "sweep-x"
    controller.sweep_exists.return_value = False
    controller.count_finished_runs.return_value = 0
    controller.run_sweep_agent.return_value = 0
    controller.fetch_best_run_details.return_value = RunScore(
        run_id="win1",
        value=0.3,
        params={"learning_rate": 1e-5},
        final_value=0.3,
        selection=config.perl.selection_strategy,
    )
    model_manager = mock.MagicMock()
    model_manager.materialize_and_push.return_value = "u/out"
    stage = PerlStage(
        CampaignContext(
            config=config,
            state=state,
            sweep_controller=controller,
            model_manager=model_manager,
        )
    )
    stage.execute()
    (sweep_dict,), _ = controller.create_sweep.call_args
    return (
        stage,
        sweep_dict["command"],
        model_manager.materialize_and_push.call_args.kwargs,
    )

  def test_trials_run_with_continual_evaluation(self):
    stage, command, _ = self._execute()
    # W&B expands the sampled hyperparameters there; it must stay last.
    self.assertEqual(command[-1], "${args}")
    # The coordinator only takes over runs that train.
    self.assertEqual(_flag_values(command, "do_train"), ["True"])
    expected = {
        **stage.continual_eval_flags(),
        **PerlStage.SWEEP_TRIAL_CHECKPOINT_FLAGS,
    }
    for name, value in expected.items():
      with self.subTest(flag=name):
        self.assertEqual(_flag_values(command, name), [value])
    self.assertEqual(_flag_values(command, "continual_eval"), ["True"])
    self.assertEqual(_flag_values(command, "save_total_limit"), ["2"])
    # Segments are relaunched under the profile the trial was launched with.
    self.assertEqual(
        _flag_values(command, "continual_eval_launch_config"),
        _flag_values(command, "config_file"),
    )

  def test_retraining_gets_the_trials_flags(self):
    stage, command, push = self._execute()
    self.assertEqual(push["continual_eval_flags"], stage.continual_eval_flags())
    for name, value in push["continual_eval_flags"].items():
      with self.subTest(flag=name):
        self.assertEqual(_flag_values(command, name), [value])
    # The retraining pins its own checkpoint budget.
    self.assertNotIn("save_total_limit", push["continual_eval_flags"])

  def test_runs_score_at_the_calibrated_threshold(self):
    _, command, push = self._execute(calibrated_threshold=0.37)
    self.assertEqual(
        _flag_values(command, "continual_eval_threshold"), ["0.37"]
    )
    self.assertEqual(
        push["continual_eval_flags"]["continual_eval_threshold"], "0.37"
    )
    stage, command, _ = self._execute()
    self.assertEqual(
        _flag_values(command, "continual_eval_threshold"),
        [str(stage.config.eval.threshold)],
    )

  def _agent_timeout(self, stage):
    agent = stage.sweep_controller.run_sweep_agent_detailed
    return agent.call_args.kwargs["timeout_minutes"]

  def test_agent_timeout_covers_continual_evaluation(self):
    # A config saved before continual evaluation existed, e.g. a resumed
    # campaign's, sized the budget for training alone: 10 * 35 * 1.5.
    def saved_before_continual_eval(config):
      config.perl.timeout_minutes = 525

    stage, _, _ = self._execute(configure=saved_before_continual_eval)
    # 10 * (35 training + 75 evaluation) * 1.5.
    self.assertEqual(self._agent_timeout(stage), 1650)
    # Only the agent's budget changes, not the campaign's config.
    self.assertEqual(stage.config.perl.timeout_minutes, 525)

  def test_agent_timeout_follows_the_eval_sample_count(self):
    # The wizard and `run --preset` set it after create_default has sized
    # the timeout for the default 1000.
    def thorough(config):
      config.eval.max_eval_samples = 2000

    stage, _, _ = self._execute(configure=thorough)
    # 10 * (35 + 115) * 1.5.
    self.assertEqual(self._agent_timeout(stage), 2250)

  def test_a_larger_configured_timeout_is_kept(self):
    def generous(config):
      config.perl.timeout_minutes = 99999

    stage, _, _ = self._execute(configure=generous)
    self.assertEqual(self._agent_timeout(stage), 99999)

  def test_flags_the_sweep_file_pins_are_left_alone(self):
    command = ["launch", "--continual_eval_max_samples=10", "${args}"]
    PerlStage.add_flags_to_sweep_command(
        command,
        {"continual_eval_max_samples": "1000", "continual_eval": "True"},
    )
    self.assertEqual(
        command,
        [
            "launch",
            "--continual_eval_max_samples=10",
            "--continual_eval=True",
            "${args}",
        ],
    )
    command = ["launch"]
    PerlStage.add_flags_to_sweep_command(command, {"continual_eval": "True"})
    self.assertEqual(command, ["launch", "--continual_eval=True"])


class MaterializationCommandTest(unittest.TestCase):
  """The winner's retraining carries the continual-evaluation flags."""

  def test_flags_are_appended_once(self):
    mgr = ModelManager(user="leobianco", dry_run=True)
    plan = mgr.build_materialization_command(
        stage_name="perl",
        task_name="npov",
        base_model="google/gemma-4-E4B-it",
        best_params={"learning_rate": 1e-5},
        sft_model_path="u/npov_SFT",
        reward_model_path="u/npov_RM",
        continual_eval_flags={
            "continual_eval": "True",
            "continual_eval_threshold": "0.37",
            # Pinned by the retraining already: its own value wins.
            "save_total_limit": "9",
        },
    )
    cmd = plan.command
    self.assertEqual(cmd[cmd.index("--continual_eval") + 1], "True")
    self.assertEqual(cmd[cmd.index("--continual_eval_threshold") + 1], "0.37")
    self.assertEqual(cmd.count("--save_total_limit"), 1)
    self.assertEqual(cmd[cmd.index("--save_total_limit") + 1], "2")


class ContinualEvalStatusReaderTest(unittest.TestCase):
  """How the orchestrator reads a retraining's continual-evaluation state."""

  def test_states(self):
    with tempfile.TemporaryDirectory() as output_dir:
      self.assertIsNone(continual_eval_training_completed(output_dir))
      for text, expected in (
          (json.dumps({"training_completed": True}), True),
          (json.dumps({"training_completed": False}), False),
          (json.dumps({"current_step": 50}), False),
          # Only an explicit JSON true counts.
          (json.dumps({"training_completed": "true"}), False),
          (json.dumps([True]), False),
          ("{truncated", False),
      ):
        with self.subTest(status=text):
          path = os.path.join(output_dir, CONTINUAL_EVAL_STATUS_FILENAME)
          with open(path, "w", encoding="utf-8") as f:
            f.write(text)
          self.assertIs(
              continual_eval_training_completed(output_dir), expected
          )


class ContinualEvalSalvageGuardTest(unittest.TestCase):
  """A failed retraining never publishes a continual-evaluation pause."""

  _PAUSED = {"training_completed": False, "paused_for_eval": True}
  _DONE = {"training_completed": True, "paused_for_eval": True}

  def setUp(self):
    super().setUp()
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    cwd = os.getcwd()
    # materialize_and_push trains into ./checkpoints/...
    os.chdir(tmp.name)
    self.addCleanup(os.chdir, cwd)
    self.mgr = ModelManager(
        user="leobianco",
        dry_run=False,
        robustness=RobustnessConfig(
            materialize_attempts=1, retry_base_delay_s=0.0
        ),
    )
    self.mgr.upload_local_checkpoint = mock.MagicMock(return_value="u/x")
    self.output_dirs = []

  def _materialize(self, *outcomes):
    """Retrains the winner; attempt ``i`` leaves ``outcomes[i]`` behind.

    Args:
      *outcomes: Per attempt, ``(returncode, status, weight_dirs)``: the exit
        code, the status file payload (None: no file; a str: written
        verbatim) and the folders of output_dir ("" for its root) the attempt
        writes adapter weights to.

    Returns:
      What ``materialize_and_push`` returns.
    """
    attempts = iter(outcomes)

    def training(cmd, **unused_kwargs):
      returncode, status, weight_dirs = next(attempts)
      output_dir = cmd[cmd.index("--output_dir") + 1]
      self.output_dirs.append(output_dir)
      for sub in weight_dirs:
        folder = os.path.join(output_dir, sub)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "adapter_model.safetensors"), "wb") as f:
          f.write(b"weights")
      if status is not None:
        path = os.path.join(output_dir, CONTINUAL_EVAL_STATUS_FILENAME)
        with open(path, "w", encoding="utf-8") as f:
          f.write(status if isinstance(status, str) else json.dumps(status))
      return ProcessOutcome(returncode=returncode, interrupted=False)

    with mock.patch(
        "src.orchestrator.model_manager.stream_subprocess",
        side_effect=training,
    ):
      return self.mgr.materialize_and_push(
          stage_name="perl",
          task_name="npov",
          base_model="google/gemma-4-E4B-it",
          best_params={"learning_rate": 1e-5},
          sft_model_path="u/npov_SFT",
          reward_model_path="u/npov_RM",
          continual_eval_flags={"continual_eval": "True"},
      )

  def test_pauses_are_not_published_before_training_completes(self):
    with self.assertRaisesRegex(
        RuntimeError, "before training completed"
    ) as ctx:
      self._materialize((1, self._PAUSED, ("", "checkpoint-50")))
    self.mgr.upload_local_checkpoint.assert_not_called()
    # Retried, so that the retraining resumes from its last pause.
    self.assertTrue(is_retryable(ctx.exception))

  def test_unreadable_state_is_not_salvaged(self):
    with self.assertRaisesRegex(RuntimeError, "before training completed"):
      self._materialize((1, "{truncated", ("", "checkpoint-50")))
    self.mgr.upload_local_checkpoint.assert_not_called()

  def test_a_retry_resumes_the_same_run(self):
    self.mgr.robustness.materialize_attempts = 2
    self.assertTrue(
        self._materialize(
            (1, self._PAUSED, ("checkpoint-50",)),
            (0, self._DONE, ("",)),
        )
    )
    self.mgr.upload_local_checkpoint.assert_not_called()
    first, second = self.output_dirs
    self.assertEqual(first, second)

  def test_completed_run_salvages_its_final_model_only(self):
    self._materialize((1, self._DONE, ("", "checkpoint-50", "checkpoint-100")))
    self.mgr.upload_local_checkpoint.assert_called_once()
    folder, _ = self.mgr.upload_local_checkpoint.call_args[0]
    self.assertEqual(
        os.path.normpath(folder), os.path.normpath(self.output_dirs[0])
    )

  def test_completed_run_without_a_final_model_is_not_salvaged(self):
    with self.assertRaisesRegex(RuntimeError, "with exit code 1$"):
      self._materialize((1, self._DONE, ("checkpoint-50", "checkpoint-100")))
    self.mgr.upload_local_checkpoint.assert_not_called()

  def test_runs_without_continual_evaluation_still_salvage_checkpoints(self):
    self._materialize((1, None, ("checkpoint-10",)))
    folder, _ = self.mgr.upload_local_checkpoint.call_args[0]
    self.assertEqual(
        os.path.basename(os.path.normpath(folder)), "checkpoint-10"
    )


if __name__ == "__main__":
  unittest.main()
