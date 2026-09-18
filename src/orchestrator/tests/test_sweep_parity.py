"""Parity between a sweep's trials and the retraining of its winner.

The orchestrator never ships a checkpoint produced by a sweep: every
``scripts/sweep_*.yaml`` runs with ``--save_strategy=no``, so a trial writes no
weights at all. Selecting a winner therefore selects a *configuration*, which
``ModelManager.materialize_and_push`` then retrains from scratch. That is only
sound while the retraining command is the trial's command: a single flag that
the sweep pins and the retraining forgets means the published model was trained
on a different problem from the one that was searched.

This module pins that correspondence. It caught the PE-RL stage retraining its
winner with the HuggingFace default schedule (linear, no warmup) while all of
its trials had run on cosine with 10% warmup.

Every difference is declared below with a reason. The allowlists are themselves
tested for staleness, so an exception cannot outlive the situation it was
written for.
"""

import unittest

import yaml

from src.orchestrator import model_manager as model_manager_mod
from src.orchestrator.config import CampaignConfig
from src.orchestrator.model_manager import ModelManager


#: The sweep stages and the YAML each one is registered from.
STAGES = {
    "sft": "scripts/sweep_sft.yaml",
    "rm": "scripts/sweep_rm.yaml",
    "perl": "scripts/sweep_perl.yaml",
}

#: Tokens of a sweep ``command:`` block that are not training flags.
_NON_FLAG_TOKENS = frozenset(
    {"${env}", "${program}", "${args}", "accelerate", "launch"}
)

#: Flags a sweep pins that the retraining deliberately does not, with why.
#:
#: Anything not listed here is a bug: the retraining must reproduce the trial.
SWEEP_ONLY = {
    "--save_only_model": (
        "A no-op in the sweep, which saves nothing at all "
        "(--save_strategy=no). The retraining does save, and keeping the "
        "optimizer state is what lets a materialization that crashed halfway "
        "resume instead of restarting."
    ),
}

#: Flags the retraining adds, with why the sweep has no business setting them.
MATERIALIZATION_ONLY = {
    # Publication: a trial is disposable, the winner is not.
    "--config_file": "accelerate launcher argument, not a training flag.",
    "--output_dir": "Per-materialization checkpoint directory.",
    "--run_name": "W&B run name derived from the published repo id.",
    "--push_to_hub": "Trials publish nothing; the winner is published.",
    "--hub_model_id": "Destination repository of the published winner.",
    # Checkpointing: the whole reason a retraining exists.
    "--save_steps": "Trials save nothing (--save_strategy=no).",
    "--save_total_limit": "Trials save nothing.",
    "--load_best_model_at_end": "Chooses which checkpoint the repo serves.",
    "--metric_for_best_model": "Ranks checkpoints inside the retraining.",
    "--greater_is_better": "Direction of --metric_for_best_model.",
    # Values that equal the training script's own default, pinned here so the
    # two stages that must agree on them cannot drift apart silently.
    "--reward_max_length": (
        "Equals the ScriptArguments default. Pinned explicitly because the "
        "reward model is trained with it in the RM stage and queried with it "
        "in the PE-RL stage; a default change would desynchronise them."
    ),
    "--peft_type": "Equals the ScriptArguments default (LORA).",
    "--task_type": "Equals the ScriptArguments default for the stage.",
    "--bf16": (
        "Redundant with mixed_precision: bf16 in "
        "scripts/deepspeed_config.yaml, which both paths launch with."
    ),
}

#: Flags present on both sides whose *values* may legitimately differ.
VALUE_MAY_DIFFER = {
    "--save_strategy": (
        "'no' in the sweep, because a trial is scored from its W&B history "
        "and its weights are thrown away; the retraining exists precisely to "
        "produce a checkpoint."
    ),
    "--sft_model_path": (
        "A placeholder in the YAML; the orchestrator injects the checkpoint "
        "the SFT stage actually published."
    ),
    "--reward_model_path": (
        "A placeholder in the YAML; the orchestrator injects the checkpoint "
        "the RM stage actually published."
    ),
}


def _load_sweep(path):
  with open(path, "r", encoding="utf-8") as handle:
    return yaml.safe_load(handle)


def _yaml_flags(sweep):
  """Returns the ``--flag: value`` map a sweep pins for every trial."""
  flags = {}
  for arg in sweep.get("command", []) or []:
    if not isinstance(arg, str) or arg in _NON_FLAG_TOKENS:
      continue
    if not arg.startswith("--") or arg.startswith("--config_file"):
      continue
    key, _, value = arg.partition("=")
    flags[key] = value
  return flags


def _command_flags(command):
  """Returns the ``--flag: value`` map of a built materialization command."""
  flags = {}
  index = 0
  while index < len(command):
    token = command[index]
    if not (isinstance(token, str) and token.startswith("--")):
      index += 1
      continue
    if "=" in token:
      key, _, value = token.partition("=")
      flags[key] = value
      index += 1
      continue
    following = command[index + 1] if index + 1 < len(command) else ""
    if isinstance(following, str) and following.startswith("--"):
      flags[token] = ""
      index += 1
      continue
    flags[token] = following
    index += 2
  return flags


class SweepMaterializationParityTest(unittest.TestCase):
  """The retraining of a winner must reproduce the trials it beat."""

  def setUp(self):
    self.config = CampaignConfig.create_default(task_name="npov")
    self.manager = ModelManager(user="leobianco", dry_run=True)
    # A plausible winner. Only the keys a sweep actually searches over are
    # replayed, so the rest of a real W&B run config is irrelevant here.
    self.best_params = {
        "learning_rate": 0.0003,
        "num_train_epochs": 2,
        "lora_r": 16,
        "lora_alpha": 32,
        "beta": 0.05,
        "temperature": 0.7,
        "reward_penalty_alpha": 1.5,
    }

  def _parity(self, stage):
    """Returns (yaml flags, retrain flags, swept parameter names)."""
    sweep = _load_sweep(STAGES[stage])
    swept = set(sweep.get("parameters", {}) or {})
    stage_config = getattr(self.config, stage)
    plan = self.manager.build_materialization_command(
        stage_name=stage,
        task_name=self.config.task_name,
        base_model=self.config.base_model,
        best_params=self.best_params,
        seed=self.config.seed,
        sft_model_path="leobianco/npov_SFT_winner",
        reward_model_path="leobianco/npov_RM_winner",
        tunable_keys=swept,
        checkpoint_policy=stage_config.checkpoint_policy,
        eval_steps=stage_config.materialization_eval_steps,
        timestamp="2601010000",
    )
    return _yaml_flags(sweep), _command_flags(plan.command), swept

  def test_retraining_pins_every_flag_the_sweep_pinned(self):
    for stage in STAGES:
      with self.subTest(stage=stage):
        yaml_flags, retrain_flags, _ = self._parity(stage)
        dropped = set(yaml_flags) - set(retrain_flags) - set(SWEEP_ONLY)
        self.assertEqual(
            dropped,
            set(),
            f"The {stage.upper()} sweep pins {sorted(dropped)} for every "
            "trial, but the retraining of its winner does not. Either pass "
            "the flag in ModelManager.build_materialization_command or add "
            "it to SWEEP_ONLY with a reason.",
        )

  def test_retraining_adds_nothing_undeclared(self):
    for stage in STAGES:
      with self.subTest(stage=stage):
        yaml_flags, retrain_flags, swept = self._parity(stage)
        swept_flags = {f"--{name}" for name in swept}
        extra = (
            set(retrain_flags)
            - set(yaml_flags)
            - set(MATERIALIZATION_ONLY)
            - swept_flags
        )
        self.assertEqual(
            extra,
            set(),
            f"The {stage.upper()} retraining passes {sorted(extra)}, which no "
            "trial ever saw. Remove it or declare it in "
            "MATERIALIZATION_ONLY with a reason.",
        )

  def test_shared_flags_carry_identical_values(self):
    for stage in STAGES:
      with self.subTest(stage=stage):
        yaml_flags, retrain_flags, _ = self._parity(stage)
        for flag in sorted(set(yaml_flags) & set(retrain_flags)):
          if flag in VALUE_MAY_DIFFER:
            continue
          self.assertEqual(
              str(retrain_flags[flag]),
              str(yaml_flags[flag]),
              f"{stage.upper()}: trials ran with {flag}="
              f"{yaml_flags[flag]!r} but the winner is retrained with "
              f"{retrain_flags[flag]!r}.",
          )

  def test_swept_hyperparameters_come_from_the_winner(self):
    # The flags the sweep searches over are the only ones the retraining may
    # take from the W&B run config; everything else is pinned code-side.
    for stage in STAGES:
      with self.subTest(stage=stage):
        _, retrain_flags, swept = self._parity(stage)
        for name in swept:
          self.assertIn(f"--{name}", retrain_flags)
          self.assertEqual(
              str(retrain_flags[f"--{name}"]),
              str(self.best_params[name]),
          )

  # --- Targeted regressions -------------------------------------------

  def test_perl_retraining_uses_the_cosine_schedule_of_its_sweep(self):
    # The original bug: lr_scheduler_type/warmup_ratio are not in
    # sweep_perl.yaml's parameters, so they are never replayed from the run
    # config, and the perl branch did not pin them either. Every trial ran
    # cosine + 10% warmup; the published policy ran linear + none.
    _, retrain_flags, _ = self._parity("perl")
    self.assertEqual(retrain_flags["--lr_scheduler_type"], "cosine")
    self.assertEqual(str(retrain_flags["--warmup_ratio"]), "0.1")

  def test_every_stage_retrains_on_the_schedule_it_searched(self):
    for stage in STAGES:
      with self.subTest(stage=stage):
        yaml_flags, retrain_flags, _ = self._parity(stage)
        for flag in ("--lr_scheduler_type", "--warmup_ratio"):
          self.assertIn(flag, retrain_flags)
          self.assertEqual(str(retrain_flags[flag]), str(yaml_flags[flag]))

  def test_the_reward_model_is_trained_and_queried_at_one_length(self):
    # The RM learns to score (prompt + completion) truncated to this budget
    # and PE-RL queries it with the same budget. If the two ever disagree the
    # reward is computed on text the model was never trained on, which is
    # silent: nothing crashes, the advantage just degrades.
    _, rm_flags, _ = self._parity("rm")
    _, perl_flags, _ = self._parity("perl")
    self.assertEqual(
        rm_flags["--reward_max_length"], perl_flags["--reward_max_length"]
    )
    self.assertEqual(
        str(rm_flags["--reward_max_length"]),
        str(model_manager_mod.REWARD_MAX_LENGTH),
    )

  def test_the_reward_model_retrains_on_its_sweeps_dataset(self):
    # A reward model trained on organic data and one trained on synthetic
    # hallucinations are different classifiers; retraining the winner on the
    # other one would invalidate the ROC-AUC it was selected on.
    yaml_flags, retrain_flags, _ = self._parity("rm")
    self.assertEqual(
        retrain_flags["--dataset_repo_id"], yaml_flags["--dataset_repo_id"]
    )
    for flag in ("--num_organic_hallus_to_keep", "--num_struct_hallus_to_keep"):
      self.assertEqual(str(retrain_flags[flag]), str(yaml_flags[flag]))

  def test_the_reward_model_retrains_as_a_classifier(self):
    _, retrain_flags, _ = self._parity("rm")
    self.assertEqual(retrain_flags["--task_type"], "SEQ_CLS")
    self.assertEqual(retrain_flags["--peft_type"], "LORA")

  def test_materialization_evaluates_as_often_as_the_sweep_did(self):
    # Trials are ranked on their best eval step; a retraining that evaluates
    # less often cannot reach the peak that won.
    for stage in STAGES:
      with self.subTest(stage=stage):
        yaml_flags, retrain_flags, _ = self._parity(stage)
        self.assertEqual(
            str(retrain_flags["--eval_steps"]), str(yaml_flags["--eval_steps"])
        )
        # load_best_model_at_end requires save and eval to be in lockstep.
        self.assertEqual(
            retrain_flags["--save_steps"], retrain_flags["--eval_steps"]
        )
        self.assertEqual(
            retrain_flags["--save_strategy"], retrain_flags["--eval_strategy"]
        )

  def test_the_checkpoint_metric_is_the_metric_the_sweep_ranked_on(self):
    # ``metric_for_best_model`` decides which checkpoint of the retraining is
    # published as "best". Ranking trials on one quantity and checkpoints on
    # another would publish a model the report does not describe.
    for stage in STAGES:
      with self.subTest(stage=stage):
        sweep = _load_sweep(STAGES[stage])
        sweep_metric = sweep["metric"]["name"]
        _, retrain_flags, _ = self._parity(stage)
        checkpoint_metric = retrain_flags["--metric_for_best_model"]
        bare = sweep_metric.split("/", 1)[-1]
        self.assertIn(
            checkpoint_metric,
            (sweep_metric, bare, sweep_metric.replace("train/", "")),
            f"{stage.upper()} ranks trials on {sweep_metric!r} but ranks "
            f"checkpoints on {checkpoint_metric!r}.",
        )
        expected = "True" if sweep["metric"]["goal"] == "maximize" else "False"
        self.assertEqual(retrain_flags["--greater_is_better"], expected)

  def test_the_campaign_goal_agrees_with_the_sweep_yaml(self):
    # The stage ranks trials with the *config's* goal but the *YAML's* metric
    # name; editing only the YAML would silently invert the ranking.
    for stage in STAGES:
      with self.subTest(stage=stage):
        sweep = _load_sweep(STAGES[stage])
        stage_config = getattr(self.config, stage)
        self.assertEqual(stage_config.goal, sweep["metric"]["goal"])
        self.assertEqual(stage_config.metric, sweep["metric"]["name"])

  # --- The allowlists must not rot -------------------------------------

  def _all_flags(self):
    yaml_side, retrain_side, shared = set(), set(), set()
    for stage in STAGES:
      yaml_flags, retrain_flags, _ = self._parity(stage)
      yaml_side |= set(yaml_flags)
      retrain_side |= set(retrain_flags)
      shared |= set(yaml_flags) & set(retrain_flags)
    return yaml_side, retrain_side, shared

  def test_no_stale_sweep_only_exceptions(self):
    yaml_side, retrain_side, _ = self._all_flags()
    for flag in SWEEP_ONLY:
      self.assertIn(flag, yaml_side, f"{flag} is in no sweep YAML any more.")
      self.assertNotIn(
          flag,
          retrain_side,
          f"{flag} is now passed by the retraining; drop it from SWEEP_ONLY.",
      )

  def test_no_stale_materialization_only_exceptions(self):
    _, retrain_side, _ = self._all_flags()
    for flag in MATERIALIZATION_ONLY:
      if flag == "--config_file":
        continue  # Consumed by `accelerate launch`, not by the script.
      self.assertIn(
          flag,
          retrain_side,
          f"{flag} is no longer passed by any retraining; drop it from "
          "MATERIALIZATION_ONLY.",
      )

  def test_no_stale_value_exceptions(self):
    _, _, shared = self._all_flags()
    for flag in VALUE_MAY_DIFFER:
      self.assertIn(
          flag,
          shared,
          f"{flag} is not on both sides any more; drop it from "
          "VALUE_MAY_DIFFER.",
      )


if __name__ == "__main__":
  unittest.main()
