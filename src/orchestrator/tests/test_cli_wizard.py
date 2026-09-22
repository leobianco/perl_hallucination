"""Tests for the interactive campaign setup wizard.

The wizard is exercised head-less through :class:`ScriptedPrompter`, so these
tests cover the real conversation flow (including every cancellation path)
without needing a terminal or ``questionary``.
"""

from __future__ import annotations

import builtins
import io
import os
import tempfile
import unittest
from unittest import mock

from src.orchestrator.cli import console as console_mod
from src.orchestrator.cli import theme as theme_mod
from src.orchestrator import flavors
from src.orchestrator.cli import wizard
from src.orchestrator.config import CampaignConfig


def make_console() -> console_mod.UiConsole:
  """Builds an in-memory, plain-text console for assertions."""
  return console_mod.UiConsole(
      theme=theme_mod.detect_theme(force_color=False, force_ascii=True),
      file=io.StringIO(),
      force_plain=True,
  )


class ValidatorTest(unittest.TestCase):
  """Input validation shared by both prompters."""

  def test_repo_id_accepts_owner_slash_name(self):
    self.assertIsNone(wizard.validate_repo_id("leobianco/npov_SFT_gemma"))

  def test_repo_id_accepts_dots_and_dashes(self):
    self.assertIsNone(wizard.validate_repo_id("org-x/model.v2-b"))

  def test_repo_id_accepts_local_paths(self):
    self.assertIsNone(wizard.validate_repo_id("./checkpoints/sft"))
    self.assertIsNone(wizard.validate_repo_id("/data/sft"))

  def test_repo_id_rejects_empty(self):
    self.assertIn("required", wizard.validate_repo_id(""))
    self.assertIn("required", wizard.validate_repo_id("   "))

  def test_repo_id_rejects_missing_owner(self):
    self.assertIn("Hugging Face", wizard.validate_repo_id("npov_SFT"))

  def test_repo_id_rejects_spaces(self):
    self.assertIsNotNone(wizard.validate_repo_id("owner/my model"))

  def test_positive_int_accepts_digits(self):
    self.assertIsNone(wizard.validate_positive_int("30"))

  def test_positive_int_rejects_non_numeric(self):
    self.assertIn("whole number", wizard.validate_positive_int("thirty"))
    self.assertIn("whole number", wizard.validate_positive_int("3.5"))
    self.assertIn("whole number", wizard.validate_positive_int("-4"))

  def test_positive_int_rejects_zero(self):
    self.assertIn("greater than zero", wizard.validate_positive_int("0"))

  def test_positive_int_respects_maximum(self):
    self.assertIn("too large", wizard.validate_positive_int("99999"))
    self.assertIsNone(wizard.validate_positive_int("99999", maximum=100000))


class EstimateTest(unittest.TestCase):
  """The ETA preview that stops accidental 40 hour launches."""

  def test_dry_run_is_reported_as_minutes(self):
    config = CampaignConfig.create_default(task_name="npov", dry_run=True)
    low, high = wizard.estimate_runtime(config)
    self.assertLess(high, 0.1)
    self.assertIn("min", wizard.format_estimate(low, high))

  def test_estimate_grows_with_budget(self):
    small = wizard.estimate_runtime(
        CampaignConfig.create_default(task_name="npov", sft_runs=1)
    )
    large = wizard.estimate_runtime(
        CampaignConfig.create_default(task_name="npov", sft_runs=50)
    )
    self.assertLess(small[1], large[1])

  def test_estimate_is_a_range(self):
    low, high = wizard.estimate_runtime(
        CampaignConfig.create_default(task_name="npov")
    )
    self.assertLess(low, high)

  def test_dropping_stages_lowers_the_estimate(self):
    full = CampaignConfig.create_default(task_name="npov")
    trimmed = CampaignConfig.create_default(task_name="npov")
    trimmed.stages = ["eval"]
    self.assertLess(
        wizard.estimate_runtime(trimmed)[1], wizard.estimate_runtime(full)[1]
    )

  def test_format_estimate_switches_unit(self):
    self.assertIn("min", wizard.format_estimate(0.1, 0.4))
    self.assertIn("h", wizard.format_estimate(2.0, 5.0))


class EquivalentCommandTest(unittest.TestCase):
  """The copy-pasteable command printed in the review step."""

  def test_defaults_are_omitted(self):
    config = CampaignConfig.create_default(task_name="npov")
    command = wizard.equivalent_command(config)
    self.assertIn("--task npov", command)
    self.assertNotIn("--stages", command)
    self.assertNotIn("--sft-runs", command)
    self.assertNotIn("--dry-run", command)

  def test_non_default_budgets_are_included(self):
    config = CampaignConfig.create_default(
        task_name="npov", sft_runs=5, rm_runs=4, perl_runs=2, dry_run=True
    )
    config.eval.max_eval_samples = 50
    command = wizard.equivalent_command(config)
    self.assertIn("--sft-runs 5", command)
    self.assertIn("--rm-runs 4", command)
    self.assertIn("--perl-runs 2", command)
    self.assertIn("--eval-samples 50", command)
    self.assertIn("--dry-run", command)

  def test_partial_stage_list_is_included(self):
    config = CampaignConfig.create_default(task_name="npov")
    config.stages = ["perl", "eval"]
    self.assertIn("--stages perl,eval", wizard.equivalent_command(config))

  def test_checkpoints_are_quoted(self):
    config = CampaignConfig.create_default(task_name="npov")
    config.perl.sft_model_path = "leobianco/npov_SFT_x"
    config.perl.reward_model_path = "leobianco/npov_RM_x"
    command = wizard.equivalent_command(config)
    self.assertIn('--sft-model "leobianco/npov_SFT_x"', command)
    self.assertIn('--reward-model "leobianco/npov_RM_x"', command)

  def test_auto_checkpoints_are_omitted(self):
    config = CampaignConfig.create_default(task_name="npov")
    config.perl.sft_model_path = "auto"
    self.assertNotIn("--sft-model", wizard.equivalent_command(config))


class BuildConfigTest(unittest.TestCase):
  """Wizard answers to a validated campaign config."""

  def test_answers_are_applied(self):
    answers = wizard.WizardAnswers(
        task="bosch",
        stages=["sft", "eval"],
        sft_runs=7,
        eval_samples=42,
        dry_run=True,
    )
    config = wizard.build_config(answers)
    self.assertEqual(config.task_name, "bosch")
    self.assertEqual(config.stages, ["sft", "eval"])
    self.assertEqual(config.sft.max_runs, 7)
    self.assertEqual(config.eval.max_eval_samples, 42)
    self.assertTrue(config.dry_run)

  def test_stage_order_is_canonical(self):
    answers = wizard.WizardAnswers(stages=["eval", "perl", "sft"])
    self.assertEqual(wizard.build_config(answers).stages,
                     ["sft", "perl", "eval"])

  def test_blank_checkpoints_default_to_auto(self):
    answers = wizard.WizardAnswers(sft_model="", reward_model="")
    config = wizard.build_config(answers)
    self.assertEqual(config.perl.sft_model_path, "auto")
    self.assertEqual(config.perl.reward_model_path, "auto")

  def test_invalid_task_is_rejected(self):
    with self.assertRaises(Exception):
      wizard.build_config(wizard.WizardAnswers(task="not_a_task"))

  def test_summary_lines_cover_every_stage(self):
    config = wizard.build_config(wizard.WizardAnswers(dry_run=True))
    theme = theme_mod.detect_theme(force_color=False, force_ascii=True)
    text = "\n".join(wizard.config_summary_lines(config, theme))
    for label in ("Task", "Stages", "SFT budget", "Eval budget", "Seed"):
      self.assertIn(label, text)
    self.assertIn("DRY-RUN", text)


class PlainPrompterTest(unittest.TestCase):
  """The numbered-menu fallback used when questionary is missing."""

  def prompter(self, inputs):
    self.console = make_console()
    self.inputs = list(inputs)
    return wizard.PlainPrompter(
        self.console, input_fn=lambda _: self.inputs.pop(0)
    )

  def test_select_by_number(self):
    p = self.prompter(["2"])
    self.assertEqual(p.select("pick", [("a", "A"), ("b", "B")]), "b")

  def test_select_by_name(self):
    p = self.prompter(["B"])
    self.assertEqual(p.select("pick", [("a", "A"), ("b", "B")]), "b")

  def test_select_empty_uses_default(self):
    p = self.prompter([""])
    self.assertEqual(
        p.select("pick", [("a", "A"), ("b", "B")], default="b"), "b"
    )

  def test_select_reprompts_on_garbage(self):
    p = self.prompter(["nope", "1"])
    self.assertEqual(p.select("pick", [("a", "A"), ("b", "B")]), "a")
    self.assertIn("Invalid choice", self.console.file.getvalue())

  def test_select_returns_none_on_eof(self):
    def boom(_):
      raise EOFError()

    p = wizard.PlainPrompter(make_console(), input_fn=boom)
    self.assertIsNone(p.select("pick", [("a", "A")]))

  def test_checkbox_keeps_defaults_on_enter(self):
    p = self.prompter([""])
    picked = p.checkbox(
        "stages", [("sft", "SFT", True), ("rm", "RM", False)]
    )
    self.assertEqual(picked, ["sft"])

  def test_checkbox_parses_comma_list_in_click_order(self):
    # The widget is generic - it also asks non-stage questions like the
    # reward-model dataset - so it returns what was clicked. Canonical
    # pipeline ordering is the stage caller's job.
    p = self.prompter(["3, 1"])
    picked = p.checkbox(
        "stages",
        [("sft", "SFT", True), ("rm", "RM", True), ("perl", "PERL", True)],
    )
    self.assertEqual(picked, ["perl", "sft"])

  def test_text_validates_and_reprompts(self):
    p = self.prompter(["zero", "12"])
    value = p.text("trials", default="30",
                   validate=wizard.validate_positive_int)
    self.assertEqual(value, "12")
    self.assertIn("whole number", self.console.file.getvalue())

  def test_text_empty_uses_default(self):
    p = self.prompter([""])
    self.assertEqual(p.text("trials", default="30"), "30")

  def test_confirm_variants(self):
    self.assertTrue(self.prompter(["y"]).confirm("ok?"))
    self.assertFalse(self.prompter(["n"]).confirm("ok?"))
    self.assertTrue(self.prompter([""]).confirm("ok?", default=True))
    self.assertFalse(self.prompter([""]).confirm("ok?", default=False))

  def test_prompts_are_rendered_without_style_tags(self):
    captured = []

    def record(prompt):
      captured.append(prompt)
      return "x"

    p = wizard.PlainPrompter(make_console(), input_fn=record)
    p.text("[bold]repo[/bold] id", default="a/b")
    self.assertIn("repo id", captured[0])
    self.assertNotIn("[bold]", captured[0])
    # Literal brackets that are not style tags must survive.
    self.assertIn("[a/b]", captured[0])

  def test_checkbox_marks_are_visible(self):
    p = self.prompter([""])
    p.checkbox("stages", [("sft", "SFT", True), ("rm", "RM", False)])
    output = self.console.file.getvalue()
    # Regression: "[x]" used to be parsed as a style tag and disappear.
    self.assertIn("[x]", output)
    self.assertIn("[ ]", output)

  def test_confirm_hint_is_visible(self):
    captured = []

    def record(prompt):
      captured.append(prompt)
      return "y"

    wizard.PlainPrompter(make_console(), input_fn=record).confirm("ok?")
    self.assertIn("[Y/n]", captured[0])



class MakePrompterTest(unittest.TestCase):
  """Prompter auto-detection."""

  def test_falls_back_to_plain_without_questionary(self):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
      if name == "questionary":
        raise ImportError("missing")
      return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=fake_import):
      prompter = wizard.make_prompter(make_console())
    self.assertIsInstance(prompter, wizard.PlainPrompter)



class WizardFlowTest(unittest.TestCase):
  """End-to-end conversations driven by scripted answers."""

  def setUp(self):
    super().setUp()
    self.console = make_console()

  def run_wizard(self, answers, save_yaml=False, raw=False):
    """Drives the wizard with ``answers``.

    The base-model question sits between "task" and "stages". Every existing
    flow predates it and does not care about the model, so unless ``raw`` is
    set the default answer is spliced in for them; that keeps this helper the
    single place that knows the question order.
    """
    if not raw:
      answers = list(answers)
      answers.insert(1, wizard.DEFAULT_BASE_MODEL)
    prompter = wizard.ScriptedPrompter(answers)
    config = wizard.run_setup_wizard(
        console=self.console, prompter=prompter, save_yaml=save_yaml
    )
    return config, prompter

  def test_happy_path_with_preset(self):
    # task, stages, preset, dry-run, launch
    config, prompter = self.run_wizard(
        [
            "npov",
            ["sft", "rm", "perl", "eval"],
            ["organic"],
            "quick",
            True,
            True,
        ]
    )
    self.assertIsNotNone(config)
    self.assertEqual(config.task_name, "npov")
    self.assertEqual(config.sft.max_runs, wizard.PRESETS["quick"][0])
    self.assertEqual(config.eval.max_eval_samples,
                     wizard.PRESETS["quick"][3])
    self.assertTrue(config.dry_run)
    self.assertEqual(len(prompter.answers), 0)

  def test_smoke_preset_is_tiny(self):
    config, _ = self.run_wizard(
        ["npov", ["sft"], "smoke", True, True]
    )
    self.assertEqual(config.sft.max_runs, 1)

  def test_custom_budget_asks_per_stage(self):
    # task, stages, preset, sft, rm, perl, eval, dry-run, launch
    config, _ = self.run_wizard(
        [
            "bosch",
            ["sft", "rm", "perl", "eval"],
            ["organic"],
            "custom",
            "3",
            "4",
            "5",
            "60",
            True,
            True,
        ]
    )
    self.assertEqual(config.sft.max_runs, 3)
    self.assertEqual(config.rm.max_runs, 4)
    self.assertEqual(config.perl.max_runs, 5)
    self.assertEqual(config.eval.max_eval_samples, 60)

  def test_custom_budget_skips_unselected_stages(self):
    config, prompter = self.run_wizard(
        ["npov", ["sft", "eval"], "custom", "2", "30", True, True]
    )
    self.assertEqual(config.sft.max_runs, 2)
    self.assertEqual(config.eval.max_eval_samples, 30)
    joined = " ".join(prompter.asked)
    self.assertNotIn("reward model sweep", joined)
    self.assertNotIn("PE-RL sweep", joined)

  def test_perl_without_sft_requires_a_checkpoint(self):
    config, prompter = self.run_wizard(
        [
            "npov",
            ["perl", "eval"],
            "smoke",
            "leobianco/npov_SFT_x",
            "leobianco/npov_RM_x",
            True,
            True,
        ]
    )
    self.assertEqual(config.perl.sft_model_path, "leobianco/npov_SFT_x")
    self.assertEqual(config.perl.reward_model_path, "leobianco/npov_RM_x")
    joined = " ".join(prompter.asked)
    self.assertIn("Existing SFT checkpoint", joined)

  def test_checkpoint_is_not_asked_when_stage_is_present(self):
    _, prompter = self.run_wizard(
        ["npov", ["sft", "rm", "perl"], ["organic"], "smoke", True, True]
    )
    self.assertNotIn(
        "Existing SFT checkpoint", " ".join(prompter.asked)
    )

  def test_invalid_scripted_checkpoint_is_rejected(self):
    with self.assertRaises(AssertionError):
      self.run_wizard(
          ["npov", ["perl"], "smoke", "not-a-repo-id", True, True]
      )

  def test_cancel_at_task(self):
    config, _ = self.run_wizard([None])
    self.assertIsNone(config)
    self.assertIn("cancelled", self.console.file.getvalue())

  def test_cancel_at_stages(self):
    config, _ = self.run_wizard(["npov", None])
    self.assertIsNone(config)
    self.assertIn("cancelled", self.console.file.getvalue())

  def test_empty_stage_selection_aborts(self):
    config, _ = self.run_wizard(["npov", []])
    self.assertIsNone(config)
    self.assertIn("No stages selected", self.console.file.getvalue())

  def test_cancel_at_preset(self):
    config, _ = self.run_wizard(["npov", ["sft"], None])
    self.assertIsNone(config)

  def test_declining_launch_returns_none_but_not_cancelled(self):
    config, _ = self.run_wizard(
        ["npov", ["sft"], "smoke", True, False]
    )
    self.assertIsNone(config)
    self.assertIn("Nothing launched", self.console.file.getvalue())

  def test_review_block_and_command_are_printed(self):
    self.run_wizard(["npov", ["sft", "eval"], "smoke", True, True])
    output = self.console.file.getvalue()
    self.assertIn("Campaign review", output)
    self.assertIn("Equivalent non-interactive command", output)
    self.assertIn("run_campaign.py run", output)

  def test_long_campaign_emits_a_warning(self):
    self.run_wizard(
        [
            "npov",
            ["sft", "rm", "perl", "eval"],
            ["organic"],
            "thorough",
            False,
            False,
        ]
    )
    self.assertIn("may run for", self.console.file.getvalue())

  def test_short_campaign_has_no_warning(self):
    self.run_wizard(["npov", ["sft"], "smoke", True, False])
    self.assertNotIn("may run for", self.console.file.getvalue())

  def test_dataset_question_is_skipped_without_an_rm_sweep(self):
    # With no reward model to train there is no dataset to choose, and
    # offering a fan-out would promise branches we cannot produce from the
    # single --reward-model path.
    _, prompter = self.run_wizard(
        [
            "npov",
            ["perl", "eval"],
            "smoke",
            "leobianco/npov_SFT_x",
            "leobianco/npov_RM_x",
            True,
            True,
        ]
    )
    self.assertNotIn("reward-model training dataset", " ".join(prompter.asked))

  def test_dataset_question_is_asked_when_the_rm_sweep_runs(self):
    _, prompter = self.run_wizard(
        ["npov", ["sft", "rm"], ["organic"], "smoke", True, True]
    )
    self.assertIn("reward-model training dataset", " ".join(prompter.asked))

  def test_choosing_synthetic_only_reaches_the_config(self):
    config, _ = self.run_wizard(
        ["npov", ["sft", "rm"], ["synthetic_struct"], "smoke", True, True]
    )
    self.assertEqual(config.rm_dataset_flavors, ["synthetic_struct"])

  def test_choosing_both_datasets_fans_the_campaign_out(self):
    config, _ = self.run_wizard(
        [
            "npov",
            ["sft", "rm", "perl", "eval"],
            ["synthetic_struct", "organic"],
            "smoke",
            True,
            True,
        ]
    )
    # Normalised into execution order regardless of click order.
    self.assertEqual(
        config.rm_dataset_flavors, ["organic", "synthetic_struct"]
    )
    self.assertEqual(
        [p.stage_id for p in flavors.build_plan(config)],
        [
            "sft",
            "rm:organic",
            "rm:synthetic_struct",
            "perl:organic",
            "perl:synthetic_struct",
            "eval",
        ],
    )

  def test_empty_dataset_selection_aborts(self):
    config, _ = self.run_wizard(["npov", ["sft", "rm"], []])
    self.assertIsNone(config)
    self.assertIn(
        "No reward-model dataset selected", self.console.file.getvalue()
    )

  def test_review_block_names_the_datasets(self):
    self.run_wizard(
        [
            "npov",
            ["sft", "rm", "perl", "eval"],
            ["organic", "synthetic_struct"],
            "smoke",
            True,
            True,
        ]
    )
    self.assertIn("Organic + Synthetic Struct", self.console.file.getvalue())

  def test_estimate_counts_every_branch(self):
    single = wizard.build_config(
        wizard.WizardAnswers(
            task="npov", rm_dataset_flavors=["organic"], preset="thorough",
            sft_runs=60, rm_runs=60, perl_runs=20, eval_samples=2000,
        )
    )
    both = wizard.build_config(
        wizard.WizardAnswers(
            task="npov",
            rm_dataset_flavors=["organic", "synthetic_struct"],
            preset="thorough",
            sft_runs=60, rm_runs=60, perl_runs=20, eval_samples=2000,
        )
    )
    # SFT and eval are shared, so the fan-out is more than 1x but less
    # than 2x. An estimate that ignored it would be plain wrong.
    self.assertGreater(
        wizard.estimate_runtime(both)[1], wizard.estimate_runtime(single)[1]
    )

  def test_yaml_export_writes_a_loadable_file(self):
    with tempfile.TemporaryDirectory() as tmp:
      cwd = os.getcwd()
      os.chdir(tmp)
      try:
        os.makedirs("configs", exist_ok=True)
        config, _ = self.run_wizard(
            ["npov", ["sft"], "smoke", True, True, True], save_yaml=True
        )
        path = os.path.join("configs", "campaign_npov.yaml")
        self.assertTrue(os.path.exists(path))
        reloaded = CampaignConfig.from_yaml(path)
        self.assertEqual(reloaded.task_name, config.task_name)
        self.assertEqual(reloaded.sft.max_runs, config.sft.max_runs)
      finally:
        os.chdir(cwd)

  def test_yaml_export_can_be_declined(self):
    with tempfile.TemporaryDirectory() as tmp:
      cwd = os.getcwd()
      os.chdir(tmp)
      try:
        self.run_wizard(
            ["npov", ["sft"], "smoke", True, False, True], save_yaml=True
        )
        self.assertFalse(os.path.exists(os.path.join("configs")))
      finally:
        os.chdir(cwd)


class BaseModelQuestionTest(unittest.TestCase):
  """The wizard must be able to choose a base model, not only show one."""

  def setUp(self):
    super().setUp()
    self.console = make_console()

  def run_wizard(self, answers):
    prompter = wizard.ScriptedPrompter(answers)
    config = wizard.run_setup_wizard(
        console=self.console, prompter=prompter, save_yaml=False
    )
    return config, prompter

  def test_the_question_is_asked(self):
    _, prompter = self.run_wizard(
        ["npov", wizard.DEFAULT_BASE_MODEL, ["sft"], "smoke", True, True]
    )
    self.assertIn("base model", " ".join(prompter.asked).lower())

  def test_default_answer_leaves_the_project_default(self):
    config, _ = self.run_wizard(
        ["npov", wizard.DEFAULT_BASE_MODEL, ["sft"], "smoke", True, True]
    )
    self.assertEqual(config.base_model, wizard.DEFAULT_BASE_MODEL)

  def test_catalogue_choice_reaches_the_config(self):
    config, _ = self.run_wizard(
        [
            "npov",
            "Qwen/Qwen3-4B-Instruct-2507",
            ["sft"],
            "smoke",
            True,
            True,
        ]
    )
    self.assertEqual(config.base_model, "Qwen/Qwen3-4B-Instruct-2507")

  def test_the_small_models_are_offered(self):
    """Added for campaigns where SFT alone already saturates the autorater."""
    offered = [repo_id for repo_id, _ in wizard.BASE_MODEL_CATALOGUE]
    self.assertIn("Qwen/Qwen2.5-1.5B-Instruct", offered)
    self.assertIn("HuggingFaceTB/SmolLM2-1.7B-Instruct", offered)

  def test_a_small_model_choice_reaches_the_config(self):
    config, _ = self.run_wizard(
        [
            "npov",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            ["sft"],
            "smoke",
            True,
            True,
        ]
    )
    self.assertEqual(
        config.base_model, "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    )

  def test_every_catalogue_id_fits_the_pickers_column(self):
    """The picker pads the repo id to 36 characters before the description.

    ``HuggingFaceTB/SmolLM2-1.7B-Instruct`` is 35, which fits with one space
    to spare. A longer id would not be truncated - it would push the
    description out of alignment on that row only, which looks like a
    rendering bug rather than a too-long name.
    """
    for repo_id, _ in wizard.BASE_MODEL_CATALOGUE:
      with self.subTest(repo_id=repo_id):
        self.assertLessEqual(len(repo_id), 36)

  def test_custom_opens_a_free_text_prompt(self):
    config, prompter = self.run_wizard(
        [
            "npov",
            "custom",
            "some-org/some-model-v9",
            ["sft"],
            "smoke",
            True,
            True,
        ]
    )
    self.assertEqual(config.base_model, "some-org/some-model-v9")
    self.assertIn("Base model repo id", prompter.asked)

  def test_custom_answer_is_validated(self):
    with self.assertRaises(AssertionError):
      self.run_wizard(
          ["npov", "custom", "not-a-repo-id", ["sft"], "smoke", True, True]
      )

  def test_cancelling_the_model_question_aborts(self):
    config, _ = self.run_wizard(["npov", None])
    self.assertIsNone(config)
    self.assertIn("cancelled", self.console.file.getvalue())

  def test_a_non_default_model_is_announced_in_the_review_block(self):
    self.run_wizard(
        [
            "npov",
            "Qwen/Qwen3-4B-Instruct-2507",
            ["sft"],
            "smoke",
            True,
            True,
        ]
    )
    self.assertIn("Qwen/Qwen3-4B-Instruct-2507", self.console.file.getvalue())

  def test_the_equivalent_command_round_trips_the_model(self):
    # Regression: the printed command used to omit the model entirely, so
    # copy-pasting it from a Qwen campaign silently reran it on Gemma.
    config = CampaignConfig.create_default(task_name="npov")
    self.assertNotIn("--base-model", wizard.equivalent_command(config))
    config.base_model = "Qwen/Qwen3-4B-Instruct-2507"
    command = wizard.equivalent_command(config)
    self.assertIn('--base-model "Qwen/Qwen3-4B-Instruct-2507"', command)
    self.assertNotIn("--reward-base-model", command)
    config.reward_base_model = "mistralai/Mistral-7B-Instruct-v0.3"
    self.assertIn(
        '--reward-base-model "mistralai/Mistral-7B-Instruct-v0.3"',
        wizard.equivalent_command(config),
    )

  def test_build_config_ignores_blank_answers(self):
    answers = wizard.WizardAnswers(task="npov")
    self.assertEqual(answers.base_model, "")
    config = wizard.build_config(answers)
    self.assertEqual(config.base_model, wizard.DEFAULT_BASE_MODEL)
    self.assertIsNone(config.reward_base_model)

  def test_default_base_model_tracks_the_dataclass(self):
    self.assertEqual(
        wizard.DEFAULT_BASE_MODEL,
        CampaignConfig.create_default(task_name="npov").base_model,
    )



if __name__ == "__main__":
  unittest.main()
