"""Tests that the campaign's base model reaches every sweep it should.

The failure this guards against is quiet rather than loud: a sweep YAML that
still pins ``google/gemma-4-E4B-it`` trains the *Gemma* reward model while the
campaign believes it is running Qwen, the reward model is then used to score
Qwen rollouts, and nothing anywhere raises.
"""

import os
import tempfile
import unittest

import yaml

from src.orchestrator.config import CampaignConfig
from src.orchestrator import model_manager as model_manager_mod
from src.orchestrator.model_manager import ModelManager
from src.orchestrator.stages.base import CampaignContext
from src.orchestrator.stages.perl_stage import PerlStage
from src.orchestrator.stages.rm_stage import RmStage
from src.orchestrator.stages.sft_stage import SftStage
from src.orchestrator.state import CampaignState
from src.orchestrator.state import StageResult
from src.orchestrator.state import StageStatus
from src.orchestrator.sweep_controller import SweepController


class _Captured(Exception):
  """Stops ``execute`` once the sweep configuration has been assembled."""


_PINNED_YAML = {
    "program": "src/reward_model.py",
    "method": "grid",
    "metric": {"name": "eval/accuracy", "goal": "maximize"},
    "parameters": {"learning_rate": {"values": [1e-4]}},
    "command": [
        "${env}",
        "${program}",
        "--task_name=npov",
        "--seed=1",
        "--dataset_repo_id=someone/else",
        "--model_repo_id=google/gemma-4-E4B-it",
    ],
}

_UNPINNED_YAML = {
    **_PINNED_YAML,
    "command": [c for c in _PINNED_YAML["command"]
                if not c.startswith("--model_repo_id=")],
}


class BaseModelInjectionTest(unittest.TestCase):

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmp.cleanup)

  def _yaml_path(self, contents, name="sweep.yaml"):
    path = os.path.join(self._tmp.name, name)
    with open(path, "w", encoding="utf-8") as f:
      yaml.safe_dump(contents, f)
    return path

  def _config(self, **kwargs):
    config = CampaignConfig.create_default(task_name="npov", dry_run=True)
    config.user = "leobianco"
    config.base_model = "Qwen/Qwen3-4B-Instruct-2507"
    for key, value in kwargs.items():
      setattr(config, key, value)
    return config

  def _context(self, config):
    return CampaignContext(
        config=config,
        state=CampaignState(campaign_id="test", task_name=config.task_name),
        sweep_controller=SweepController(dry_run=True),
        model_manager=ModelManager(dry_run=True),
    )

  def _captured_command(self, stage):
    captured = {}

    def fake_resolve_sweep_id(sweep_dict, live_line_callback=None):
      del live_line_callback
      captured["command"] = sweep_dict["command"]
      raise _Captured()

    stage.resolve_sweep_id = fake_resolve_sweep_id
    with self.assertRaises(_Captured):
      stage.execute()
    return captured["command"]

  def _model_args(self, command):
    return [c for c in command if c.startswith("--model_repo_id=")]

  # --- RM -------------------------------------------------------------

  def test_rm_overrides_a_pinned_yaml_model(self):
    config = self._config()
    config.rm.sweep_config_path = self._yaml_path(_PINNED_YAML)
    command = self._captured_command(RmStage(self._context(config)))
    self.assertEqual(
        self._model_args(command),
        ["--model_repo_id=Qwen/Qwen3-4B-Instruct-2507"],
    )

  def test_rm_injects_a_model_into_an_unpinned_yaml(self):
    config = self._config()
    config.rm.sweep_config_path = self._yaml_path(_UNPINNED_YAML)
    command = self._captured_command(RmStage(self._context(config)))
    self.assertEqual(
        self._model_args(command),
        ["--model_repo_id=Qwen/Qwen3-4B-Instruct-2507"],
    )

  def test_rm_honours_a_separate_reward_base_model(self):
    # A small policy scored by a larger reward model is a normal setup; the
    # reward model must not silently inherit the policy's base.
    config = self._config(reward_base_model="mistralai/Mistral-7B-Instruct-v0.3")
    config.rm.sweep_config_path = self._yaml_path(_PINNED_YAML)
    command = self._captured_command(RmStage(self._context(config)))
    self.assertEqual(
        self._model_args(command),
        ["--model_repo_id=mistralai/Mistral-7B-Instruct-v0.3"],
    )

  def test_rm_defaults_the_reward_base_to_the_policy_base(self):
    config = self._config()
    self.assertEqual(
        config.resolved_reward_base_model(), "Qwen/Qwen3-4B-Instruct-2507"
    )

  # --- SFT and PE-RL ---------------------------------------------------

  def test_sft_overrides_a_pinned_yaml_model(self):
    config = self._config()
    config.sft.sweep_config_path = self._yaml_path(_PINNED_YAML)
    command = self._captured_command(SftStage(self._context(config)))
    self.assertEqual(
        self._model_args(command),
        ["--model_repo_id=Qwen/Qwen3-4B-Instruct-2507"],
    )

  def test_perl_overrides_a_pinned_yaml_model(self):
    config = self._config()
    config.perl.sweep_config_path = self._yaml_path(_PINNED_YAML)
    context = self._context(config)
    for stage, repo in (("sft", "leobianco/npov_sft"),
                        ("rm", "leobianco/npov_rm")):
      context.state.stages[stage] = StageResult(
          status=StageStatus.COMPLETED, model_repo_id=repo
      )
    command = self._captured_command(PerlStage(context))
    self.assertEqual(
        self._model_args(command),
        ["--model_repo_id=Qwen/Qwen3-4B-Instruct-2507"],
    )

  # --- Sweep naming ----------------------------------------------------

  def test_the_sweep_name_reports_the_model_actually_used(self):
    config = self._config(reward_base_model="mistralai/Mistral-7B-Instruct-v0.3")
    stage = RmStage(self._context(config))
    model_short, _ = stage.get_sweep_descriptor({})
    self.assertEqual(model_short, "Mistral-7B-Instruct-v0.3")



class PerlBatchGeometryTest(unittest.TestCase):
  """The PE-RL retraining geometry must stay pinned but be overridable.

  The defaults were tuned for a ~4B policy with
  ``--auto_find_batch_size False``, i.e. with the runtime escape hatch
  deliberately disabled. A 7B policy therefore OOMs with no way out unless the
  geometry itself can be changed.
  """

  def test_defaults_are_unchanged(self):
    geometry = model_manager_mod.perl_batch_geometry(env={})
    self.assertEqual(geometry["num_generations"], "8")
    self.assertEqual(geometry["num_iterations"], "1")
    self.assertEqual(geometry["steps_per_generation"], "16")
    self.assertEqual(geometry["per_device_train_batch_size"], "4")
    self.assertEqual(geometry["per_device_eval_batch_size"], "16")
    self.assertEqual(geometry["gradient_accumulation_steps"], "8")
    self.assertEqual(geometry["auto_find_batch_size"], "False")

  def test_environment_overrides_are_applied(self):
    geometry = model_manager_mod.perl_batch_geometry(
        env={
            "PERL_PER_DEVICE_TRAIN_BATCH_SIZE": "2",
            "PERL_GRADIENT_ACCUMULATION_STEPS": "16",
        }
    )
    self.assertEqual(geometry["per_device_train_batch_size"], "2")
    self.assertEqual(geometry["gradient_accumulation_steps"], "16")
    # Everything else keeps its default.
    self.assertEqual(geometry["num_generations"], "8")

  def test_blank_overrides_are_ignored(self):
    geometry = model_manager_mod.perl_batch_geometry(
        env={"PERL_NUM_GENERATIONS": "   "}
    )
    self.assertEqual(geometry["num_generations"], "8")

  def test_unrelated_environment_variables_are_ignored(self):
    geometry = model_manager_mod.perl_batch_geometry(
        env={"PERL_RUNS": "99", "NUM_GENERATIONS": "1"}
    )
    self.assertEqual(geometry, model_manager_mod.PERL_BATCH_GEOMETRY)

  def test_the_table_is_not_mutated(self):
    before = dict(model_manager_mod.PERL_BATCH_GEOMETRY)
    model_manager_mod.perl_batch_geometry(env={"PERL_NUM_GENERATIONS": "4"})
    self.assertEqual(model_manager_mod.PERL_BATCH_GEOMETRY, before)


if __name__ == "__main__":
  unittest.main()
