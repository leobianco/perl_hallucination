"""Tests for size-aware DeepSpeed selection and PE-RL batch scaling.

Two failures are being guarded against, and they fail in opposite directions:

  * A 7B campaign silently inherits the 4B launcher and batch geometry, and
    OOMs in the PE-RL stage hours in, with ``auto_find_batch_size=False``
    leaving no runtime escape.
  * A 4B campaign - every campaign this project has already run - is quietly
    moved onto a slower launcher or a different effective batch, invalidating
    comparisons against the numbers in BASELINES_*.md.

So most of these tests assert that Gemma does *not* change.
"""

import os
import tempfile
import unittest

import yaml

from src.orchestrator import accel
from src.orchestrator.config import CampaignConfig
from src.orchestrator import model_manager as model_manager_mod
from src.orchestrator.model_manager import ModelManager
from src.orchestrator.stages.base import CampaignContext
from src.orchestrator.stages.perl_stage import PerlStage
from src.orchestrator.stages.rm_stage import RmStage
from src.orchestrator.stages.sft_stage import SftStage
from src.orchestrator.state import CampaignState
from src.orchestrator.sweep_controller import SweepController


GEMMA = "google/gemma-4-E4B-it"
MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"


class _Captured(Exception):
  """Stops ``execute`` once the sweep configuration has been assembled."""


class EstimateParametersTest(unittest.TestCase):

  def test_reads_the_size_token_of_common_checkpoints(self):
    cases = {
        GEMMA: 4.0,
        "Qwen/Qwen3-4B-Instruct-2507": 4.0,
        MISTRAL: 7.0,
        "google/gemma-3-1b-it": 1.0,
        "meta-llama/Llama-3.1-70B-Instruct": 70.0,
        "Qwen/Qwen3-4B": 4.0,
        "someone/model-1.5b": 1.5,
    }
    for repo_id, expected in cases.items():
      with self.subTest(repo_id=repo_id):
        self.assertEqual(accel.estimate_parameters_b(repo_id), expected)

  def test_reads_the_size_of_a_checkpoint_this_project_published(self):
    # Campaign checkpoints are named <user>/<task>_<stage>_<model>_<details>,
    # so the size token sits in the middle rather than at the end.
    self.assertEqual(
        accel.estimate_parameters_b(
            "leobianco/npov_PERL_Mistral-7B-Instruct-v0.3_S130104_epo0.2"
        ),
        7.0,
    )

  def test_a_mixture_of_experts_is_not_read_as_its_expert_size(self):
    # "8x7B" must not read as 7: the box has to hold every expert.
    self.assertEqual(
        accel.estimate_parameters_b("mistralai/Mixtral-8x7B-Instruct-v0.1"),
        46.7,
    )

  def test_an_unnamed_size_is_no_opinion_rather_than_a_guess(self):
    self.assertIsNone(accel.estimate_parameters_b("leobianco/my_model"))
    self.assertIsNone(accel.estimate_parameters_b(""))
    self.assertIsNone(accel.estimate_parameters_b(None))

  def test_an_unnamed_size_keeps_the_historical_settings(self):
    # The safe direction for "I don't know": behave exactly as the project
    # did before this module existed.
    self.assertEqual(
        accel.select_profile("perl", "leobianco/my_model"), accel.ZERO2
    )
    self.assertEqual(accel.memory_flags("sft", "leobianco/my_model"), {})


class SelectProfileTest(unittest.TestCase):

  def test_a_4b_campaign_is_unchanged_in_every_stage(self):
    for stage in ("sft", "rm", "perl"):
      with self.subTest(stage=stage):
        self.assertEqual(
            accel.deepspeed_config_path(stage, GEMMA),
            "scripts/deepspeed_config.yaml",
        )

  def test_a_7b_campaign_gets_stage_3_in_perl_only(self):
    self.assertEqual(accel.select_profile("sft", MISTRAL), accel.ZERO2)
    self.assertEqual(accel.select_profile("rm", MISTRAL), accel.ZERO2)
    self.assertEqual(accel.select_profile("perl", MISTRAL), accel.ZERO3)

  def test_perl_is_sized_on_the_larger_of_policy_and_reward_model(self):
    # The stage holds both. A small policy with a large reward model is just
    # as tight as the other way round.
    self.assertEqual(
        accel.select_profile("perl", GEMMA, reward_model=MISTRAL), accel.ZERO3
    )
    # ...but the RM sweep itself only ever holds the reward model.
    self.assertEqual(
        accel.select_profile("rm", MISTRAL, reward_model=GEMMA), accel.ZERO2
    )

  def test_two_4b_models_do_not_add_up_into_stage_3(self):
    # Guards the deliberate choice of max() over sum(): summing would flip
    # every existing Gemma campaign onto a launcher it never ran under.
    self.assertEqual(
        accel.select_profile("perl", GEMMA, reward_model=GEMMA), accel.ZERO2
    )

  def test_a_very_large_model_falls_back_to_offloading(self):
    for stage in ("sft", "rm", "perl"):
      with self.subTest(stage=stage):
        self.assertEqual(
            accel.select_profile(stage, "meta-llama/Llama-3.1-70B-Instruct"),
            accel.ZERO3_OFFLOAD,
        )

  def test_an_explicit_profile_overrides_the_estimate(self):
    self.assertEqual(
        accel.deepspeed_config_path("sft", GEMMA, profile=accel.ZERO3),
        "scripts/deepspeed_config_zero3.yaml",
    )
    self.assertEqual(
        accel.deepspeed_config_path("perl", MISTRAL, profile=accel.ZERO2),
        "scripts/deepspeed_config.yaml",
    )

  def test_a_path_is_passed_through_untouched(self):
    self.assertEqual(
        accel.deepspeed_config_path("perl", MISTRAL, profile="my/own.yaml"),
        "my/own.yaml",
    )

  def test_every_named_profile_exists_on_disk(self):
    # A profile that names a missing file fails inside `accelerate launch`,
    # minutes into a stage, long after the campaign committed to it.
    repo_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    for name, path in accel.PROFILE_CONFIGS.items():
      with self.subTest(profile=name):
        self.assertTrue(
            os.path.exists(os.path.join(repo_root, path)),
            f"{name} points at {path}, which does not exist.",
        )

  def test_the_named_profiles_declare_the_stage_they_claim(self):
    repo_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    expected = {accel.ZERO2: 2, accel.ZERO3: 3, accel.ZERO3_OFFLOAD: 3}
    for name, stage in expected.items():
      with self.subTest(profile=name):
        with open(
            os.path.join(repo_root, accel.PROFILE_CONFIGS[name]),
            "r",
            encoding="utf-8",
        ) as handle:
          parsed = yaml.safe_load(handle)
        self.assertEqual(parsed["deepspeed_config"]["zero_stage"], stage)
        # Every profile launches the same box; a mismatch here would start
        # one process per GPU on one profile and not the other.
        self.assertEqual(parsed["num_processes"], accel.NUM_PROCESSES)


class MemoryFlagsTest(unittest.TestCase):

  def test_a_4b_campaign_gets_no_extra_flags(self):
    for stage in ("sft", "rm", "perl"):
      with self.subTest(stage=stage):
        self.assertEqual(accel.memory_flags(stage, GEMMA), {})

  def test_a_7b_campaign_gets_gradient_checkpointing_everywhere(self):
    for stage in ("sft", "rm", "perl"):
      with self.subTest(stage=stage):
        self.assertEqual(
            accel.memory_flags(stage, MISTRAL),
            {"gradient_checkpointing": "True"},
        )


class ScalePerlGeometryTest(unittest.TestCase):

  BASELINE = dict(model_manager_mod.PERL_BATCH_GEOMETRY)

  def test_a_4b_policy_keeps_the_tuned_geometry(self):
    self.assertEqual(
        accel.scale_perl_geometry(self.BASELINE, GEMMA), self.BASELINE
    )

  def test_a_7b_policy_halves_the_micro_batch(self):
    scaled = accel.scale_perl_geometry(self.BASELINE, MISTRAL)
    self.assertEqual(scaled["per_device_train_batch_size"], "2")
    self.assertEqual(scaled["per_device_eval_batch_size"], "8")

  def test_the_effective_batch_is_preserved(self):
    # This is the point of the feature: less memory per micro-step, the same
    # optimisation problem. If this drifts, a 7B sweep is no longer
    # comparable to the 4B one it is being contrasted with.
    scaled = accel.scale_perl_geometry(self.BASELINE, MISTRAL)
    before = (
        int(self.BASELINE["per_device_train_batch_size"])
        * int(self.BASELINE["gradient_accumulation_steps"])
    )
    after = (
        int(scaled["per_device_train_batch_size"])
        * int(scaled["gradient_accumulation_steps"])
    )
    self.assertEqual(before, after)

  def test_the_generation_batch_stays_divisible_by_num_generations(self):
    # TRL's own invariant; violating it aborts the trial at startup.
    scaled = accel.scale_perl_geometry(self.BASELINE, MISTRAL)
    generation_batch = (
        int(scaled["per_device_train_batch_size"])
        * accel.NUM_PROCESSES
        * int(scaled["steps_per_generation"])
    )
    self.assertEqual(generation_batch % int(scaled["num_generations"]), 0)

  def test_a_geometry_that_cannot_be_halved_is_left_alone(self):
    baseline = dict(self.BASELINE, per_device_train_batch_size="1")
    self.assertEqual(
        accel.scale_perl_geometry(baseline, MISTRAL), baseline
    )

  def test_a_geometry_that_would_break_divisibility_is_left_alone(self):
    # 3 prompts per device x 2 processes x 4 steps = 24, divisible by 8;
    # halving to 1 would give 8 -- still fine -- so force the bad case with
    # steps_per_generation=3: 1 x 2 x 3 = 6, which is not.
    baseline = dict(
        self.BASELINE,
        per_device_train_batch_size="2",
        steps_per_generation="3",
        num_generations="8",
    )
    self.assertEqual(accel.scale_perl_geometry(baseline, MISTRAL), baseline)

  def test_a_non_numeric_geometry_is_left_alone(self):
    baseline = dict(self.BASELINE, per_device_train_batch_size="auto")
    self.assertEqual(accel.scale_perl_geometry(baseline, MISTRAL), baseline)


class PerlBatchGeometryTest(unittest.TestCase):
  """The materialization side of the same scaling."""

  def test_the_retrain_scales_exactly_as_the_sweep_does(self):
    self.assertEqual(
        model_manager_mod.perl_batch_geometry(env={}, policy_model=MISTRAL),
        accel.scale_perl_geometry(
            model_manager_mod.PERL_BATCH_GEOMETRY, MISTRAL
        ),
    )

  def test_omitting_the_policy_keeps_the_historical_geometry(self):
    self.assertEqual(
        model_manager_mod.perl_batch_geometry(env={}),
        model_manager_mod.PERL_BATCH_GEOMETRY,
    )

  def test_an_explicit_environment_override_still_wins(self):
    # The operator's own number must not be silently re-derived.
    geometry = model_manager_mod.perl_batch_geometry(
        env={"PERL_PER_DEVICE_TRAIN_BATCH_SIZE": "1"}, policy_model=MISTRAL
    )
    self.assertEqual(geometry["per_device_train_batch_size"], "1")


class ConfigValidationTest(unittest.TestCase):

  def test_the_default_is_auto(self):
    self.assertEqual(CampaignConfig().deepspeed_profile, accel.AUTO)

  def test_a_known_profile_validates(self):
    for profile in accel.VALID_PROFILES:
      with self.subTest(profile=profile):
        CampaignConfig(deepspeed_profile=profile).validate()

  def test_a_path_validates(self):
    CampaignConfig(deepspeed_profile="scripts/whatever.yaml").validate()

  def test_a_typo_is_rejected_before_any_gpu_time_is_spent(self):
    with self.assertRaises(ValueError):
      CampaignConfig(deepspeed_profile="zero_3").validate()
    with self.assertRaises(ValueError):
      CampaignConfig(deepspeed_profile="").validate()


_SWEEP_YAML = {
    "program": "src/perl.py",
    "method": "grid",
    "metric": {"name": "train/rewards/reward_fn/mean", "goal": "maximize"},
    "parameters": {"learning_rate": {"values": [1e-4]}},
    "command": [
        "${env}",
        "accelerate",
        "launch",
        "--config_file=scripts/deepspeed_config.yaml",
        "${program}",
        "--task_name=npov",
        "--seed=1",
        "--dataset_repo_id=someone/else",
        "--model_repo_id=google/gemma-4-E4B-it",
        "--sft_model_path=leobianco/",
        "--reward_model_path=leobianco/",
        "--num_generations=8",
        "--steps_per_generation=16",
        "--per_device_train_batch_size=4",
        "--per_device_eval_batch_size=16",
        "--gradient_accumulation_steps=8",
        "${args}",
    ],
}


class SweepCommandTest(unittest.TestCase):
  """What the trials are actually launched with."""

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmp.cleanup)

  def _yaml_path(self, contents, name="sweep.yaml"):
    path = os.path.join(self._tmp.name, name)
    with open(path, "w", encoding="utf-8") as f:
      yaml.safe_dump(contents, f)
    return path

  def _config(self, base_model, **kwargs):
    config = CampaignConfig.create_default(task_name="npov", dry_run=True)
    config.user = "leobianco"
    config.base_model = base_model
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

  def _perl_command(self, base_model):
    config = self._config(base_model)
    config.perl.sweep_config_path = self._yaml_path(_SWEEP_YAML)
    config.perl.sft_model_path = "leobianco/sft"
    config.perl.reward_model_path = "leobianco/rm"
    return self._captured_command(PerlStage(self._context(config)))

  def _flag(self, command, name):
    prefix = f"--{name}="
    values = [c[len(prefix):] for c in command if c.startswith(prefix)]
    self.assertEqual(len(values), 1, f"{name} appears {len(values)} times")
    return values[0]

  def test_a_4b_perl_sweep_keeps_every_launcher_flag_it_had(self):
    # The stage legitimately rewrites task, seed, dataset and checkpoint
    # paths, so this compares only what the launcher feature owns - and
    # requires it to have touched nothing.
    owned = (
        "config_file",
        "num_generations",
        "steps_per_generation",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
    )
    command = self._perl_command(GEMMA)
    for name in owned:
      with self.subTest(flag=name):
        self.assertEqual(
            self._flag(command, name), self._flag(_SWEEP_YAML["command"], name)
        )
    self.assertNotIn(
        "gradient_checkpointing",
        " ".join(command),
        "A 4B campaign must not silently acquire gradient checkpointing.",
    )

  def test_a_7b_perl_sweep_moves_to_stage_3(self):
    command = self._perl_command(MISTRAL)
    self.assertEqual(
        self._flag(command, "config_file"),
        "scripts/deepspeed_config_zero3.yaml",
    )

  def test_a_7b_perl_sweep_shrinks_the_micro_batch(self):
    command = self._perl_command(MISTRAL)
    self.assertEqual(self._flag(command, "per_device_train_batch_size"), "2")
    self.assertEqual(self._flag(command, "gradient_accumulation_steps"), "16")

  def test_a_7b_sweep_asks_for_gradient_checkpointing(self):
    command = self._perl_command(MISTRAL)
    self.assertEqual(self._flag(command, "gradient_checkpointing"), "True")

  def test_the_expansion_token_stays_last(self):
    # W&B appends the trial's sampled hyperparameters at ``${args}``; a flag
    # inserted after it would be overridden by, or collide with, them.
    command = self._perl_command(MISTRAL)
    self.assertEqual(command[-1], "${args}")

  def test_sft_and_rm_stay_on_stage_2_at_7b(self):
    for stage_cls, attr in ((SftStage, "sft"), (RmStage, "rm")):
      with self.subTest(stage=attr):
        config = self._config(MISTRAL)
        getattr(config, attr).sweep_config_path = self._yaml_path(
            _SWEEP_YAML, name=f"{attr}.yaml"
        )
        command = self._captured_command(stage_cls(self._context(config)))
        self.assertEqual(
            self._flag(command, "config_file"),
            "scripts/deepspeed_config.yaml",
        )
        self.assertEqual(self._flag(command, "gradient_checkpointing"), "True")

  def test_a_forced_profile_reaches_the_sweep(self):
    config = self._config(GEMMA, deepspeed_profile=accel.ZERO3_OFFLOAD)
    config.perl.sweep_config_path = self._yaml_path(_SWEEP_YAML)
    command = self._captured_command(PerlStage(self._context(config)))
    self.assertEqual(
        self._flag(command, "config_file"),
        "scripts/deepspeed_config_zero3_offload.yaml",
    )

  def test_a_sweep_that_already_pins_checkpointing_is_not_doubled(self):
    pinned = dict(_SWEEP_YAML)
    pinned["command"] = [
        c for c in _SWEEP_YAML["command"] if c != "${args}"
    ] + ["--gradient_checkpointing=False", "${args}"]
    config = self._config(MISTRAL)
    config.perl.sweep_config_path = self._yaml_path(pinned, name="pinned.yaml")
    command = self._captured_command(PerlStage(self._context(config)))
    # Deliberate beats derived, and the flag must appear exactly once or
    # HfArgumentParser takes the last one and the intent is anyone's guess.
    self.assertEqual(self._flag(command, "gradient_checkpointing"), "False")


class SweepMaterializationAgreementTest(unittest.TestCase):
  """The retrain must land on the same launcher and batch as the sweep.

  A winner selected under Stage 3 that is retrained under Stage 2 OOMs after
  the sweep has already been paid for - the most expensive possible moment.
  """

  def _plan_command(self, base_model, config):
    manager = ModelManager(dry_run=True)
    plan = manager.build_materialization_command(
        stage_name="perl",
        task_name="npov",
        base_model=base_model,
        best_params={"learning_rate": 1e-4},
        sft_model_path="leobianco/sft",
        reward_model_path="leobianco/rm",
        deepspeed_config=config.deepspeed_config_for("perl"),
        memory_flags=config.memory_flags_for("perl"),
        timestamp="2501010000",
    )
    return plan.command

  def _value(self, command, flag):
    for idx, token in enumerate(command):
      if token == flag:
        return command[idx + 1]
      if token.startswith(f"{flag}="):
        return token.split("=", 1)[1]
    return None

  def test_the_retrain_uses_the_profile_the_sweep_selected(self):
    config = CampaignConfig.create_default(task_name="npov", dry_run=True)
    config.base_model = MISTRAL
    command = self._plan_command(MISTRAL, config)
    self.assertIn(
        "--config_file=scripts/deepspeed_config_zero3.yaml", command
    )

  def test_the_retrain_uses_the_batch_the_sweep_searched(self):
    config = CampaignConfig.create_default(task_name="npov", dry_run=True)
    config.base_model = MISTRAL
    command = self._plan_command(MISTRAL, config)
    self.assertEqual(
        self._value(command, "--per_device_train_batch_size"), "2"
    )
    self.assertEqual(
        self._value(command, "--gradient_accumulation_steps"), "16"
    )
    self.assertEqual(self._value(command, "--gradient_checkpointing"), "True")

  def test_a_4b_retrain_is_unchanged(self):
    config = CampaignConfig.create_default(task_name="npov", dry_run=True)
    config.base_model = GEMMA
    command = self._plan_command(GEMMA, config)
    self.assertIn("--config_file=scripts/deepspeed_config.yaml", command)
    self.assertEqual(
        self._value(command, "--per_device_train_batch_size"), "4"
    )
    self.assertIsNone(self._value(command, "--gradient_checkpointing"))


if __name__ == "__main__":
  unittest.main()
