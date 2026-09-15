"""Tests for the orchestrator command line application.

Every command is exercised through :func:`app.main` with an explicit argv and
an in-memory console, so the assertions are made on exactly the bytes a user
would see in their tmux pane. Campaign execution itself is stubbed: these
tests cover the CLI, not the training pipeline.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from src.orchestrator.cli import app
from src.orchestrator.cli import console as console_mod
from src.orchestrator.cli import theme as theme_mod
from src.orchestrator.config import CampaignConfig
from src.orchestrator.state import CampaignState, StageResult, StageStatus


def make_console() -> console_mod.UiConsole:
  """Builds an in-memory, plain-text console for assertions."""
  return console_mod.UiConsole(
      theme=theme_mod.detect_theme(force_color=False, force_ascii=True),
      file=io.StringIO(),
      force_plain=True,
  )


def write_state(
    root: str,
    task: str = "npov",
    campaign_id: str = "camp_npov_1",
    status: str = "IN_PROGRESS",
    stages=None,
    config_dict=None,
) -> str:
  """Writes a campaign state file below ``root`` and returns its path."""
  state = CampaignState(campaign_id=campaign_id, task_name=task)
  state.status = status
  state.stages_order = ["sft", "rm", "perl", "eval"]
  state.config_dict = config_dict
  for name, result in (stages or {}).items():
    state.stages[name] = result
  path = os.path.join(root, task, f"{campaign_id}_state.json")
  state.save(path)
  return path


class ParserTest(unittest.TestCase):
  """Argument parsing surface."""

  def setUp(self):
    super().setUp()
    self.parser = app.build_parser()

  def test_no_command_is_allowed(self):
    self.assertIsNone(self.parser.parse_args([]).command)

  def test_run_defaults(self):
    args = self.parser.parse_args(["run"])
    self.assertEqual(args.command, "run")
    self.assertEqual(args.task, "npov")
    self.assertIsNone(args.preset)
    self.assertFalse(args.dry_run)

  def test_run_accepts_presets(self):
    self.assertEqual(
        self.parser.parse_args(["run", "--preset", "smoke"]).preset, "smoke"
    )

  def test_run_rejects_unknown_preset(self):
    with contextlib.redirect_stderr(io.StringIO()):
      with self.assertRaises(SystemExit):
        self.parser.parse_args(["run", "--preset", "turbo"])

  def test_run_rejects_unknown_task(self):
    with contextlib.redirect_stderr(io.StringIO()):
      with self.assertRaises(SystemExit):
        self.parser.parse_args(["run", "--task", "not_a_task"])

  def test_global_flags_work_before_and_after_the_command(self):
    self.assertTrue(self.parser.parse_args(["--no-color", "status"]).no_color)
    self.assertTrue(self.parser.parse_args(["status", "--no-color"]).no_color)

  def test_ls_is_an_alias_of_list(self):
    self.assertEqual(self.parser.parse_args(["ls"]).command, "ls")

  def test_status_watch_interval(self):
    args = self.parser.parse_args(["status", "--watch", "--interval", "2.5"])
    self.assertTrue(args.watch)
    self.assertAlmostEqual(args.interval, 2.5)

  def test_short_flags(self):
    args = self.parser.parse_args(["run", "-t", "bosch", "-y"])
    self.assertEqual(args.task, "bosch")
    self.assertTrue(args.yes)


class StateDiscoveryTest(unittest.TestCase):
  """Finding and rehydrating campaign state files."""

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.root = self.tmp.name

  def test_find_state_files_is_empty_by_default(self):
    self.assertEqual(app.find_state_files(root=self.root), [])

  def test_find_state_files_filters_by_task(self):
    write_state(self.root, task="npov", campaign_id="a")
    write_state(self.root, task="bosch", campaign_id="b")
    self.assertEqual(len(app.find_state_files(root=self.root)), 2)
    self.assertEqual(
        len(app.find_state_files("bosch", root=self.root)), 1
    )

  def test_find_latest_state_prefers_the_newest(self):
    old = write_state(self.root, campaign_id="old")
    new = write_state(self.root, campaign_id="new")
    os.utime(old, (1, 1))
    self.assertEqual(app.find_latest_state("npov", root=self.root), new)

  def test_find_latest_state_returns_none_when_absent(self):
    self.assertIsNone(app.find_latest_state("npov", root=self.root))

  def test_config_for_state_uses_persisted_config(self):
    config = CampaignConfig.create_default(task_name="npov", sft_runs=7)
    state = CampaignState(campaign_id="c1", task_name="npov")
    state.config_dict = config.to_dict()
    rebuilt = app.config_for_state(state)
    self.assertEqual(rebuilt.sft.max_runs, 7)
    self.assertEqual(rebuilt.name, "c1")

  def test_config_for_state_falls_back_to_defaults(self):
    state = CampaignState(campaign_id="c1", task_name="bosch")
    state.stages_order = ["sft", "eval"]
    rebuilt = app.config_for_state(state)
    self.assertEqual(rebuilt.task_name, "bosch")
    self.assertEqual(rebuilt.stages, ["sft", "eval"])

  def test_config_for_state_survives_a_corrupt_config_dict(self):
    state = CampaignState(campaign_id="c1", task_name="npov")
    state.config_dict = {"task_name": object()}
    rebuilt = app.config_for_state(state)
    self.assertEqual(rebuilt.task_name, "npov")

  def test_state_progress_counts_terminal_stages(self):
    config = CampaignConfig.create_default(task_name="npov")
    state = CampaignState(campaign_id="c1", task_name="npov")
    state.stages["sft"] = StageResult(status=StageStatus.COMPLETED)
    state.stages["rm"] = StageResult(status=StageStatus.SKIPPED)
    state.stages["perl"] = StageResult(status=StageStatus.RUNNING)
    self.assertEqual(app.state_progress(state, config), (2, 4))


class DiagnosticsTest(unittest.TestCase):
  """The pre-flight ``doctor`` checks."""

  def setUp(self):
    super().setUp()
    self.config = CampaignConfig.create_default(task_name="npov")

  def test_every_check_is_well_formed(self):
    checks = app.run_diagnostics(self.config)
    self.assertTrue(checks)
    for check in checks:
      self.assertEqual(set(check), {"name", "status", "detail"})
      self.assertIn(check["status"], ("ok", "warn", "fail"))

  def test_yaml_is_reported_as_present(self):
    checks = {c["name"]: c for c in app.run_diagnostics(self.config)}
    self.assertEqual(checks["package: yaml"]["status"], "ok")

  def test_missing_credentials_are_reported(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      with mock.patch("os.path.exists", return_value=False):
        checks = {c["name"]: c for c in app.run_diagnostics(self.config)}
    self.assertEqual(checks["W&B credentials"]["status"], "fail")
    self.assertEqual(checks["Hugging Face token"]["status"], "warn")

  def test_tmux_is_detected(self):
    with mock.patch.dict(os.environ, {"TMUX": "/tmp/tmux-1/default"}):
      checks = {c["name"]: c for c in app.run_diagnostics(self.config)}
    self.assertEqual(checks["tmux session"]["status"], "ok")

  def test_missing_sweep_config_is_a_failure(self):
    self.config.sft.sweep_config_path = "/definitely/not/here.yaml"
    checks = {c["name"]: c for c in app.run_diagnostics(self.config)}
    self.assertEqual(checks["sweep config: sft"]["status"], "fail")


class CommandTestBase(unittest.TestCase):
  """Runs commands inside a temporary working directory."""

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.cwd = os.getcwd()
    os.chdir(self.tmp.name)
    self.addCleanup(os.chdir, self.cwd)
    self.console = make_console()

  @property
  def output(self) -> str:
    return self.console.file.getvalue()

  def run_cli(self, argv):
    return app.main(argv, console=self.console)

  def run_cli_json(self, argv):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
      code = app.main(argv, console=self.console)
    return code, buffer.getvalue()


class HelpAndUsageTest(CommandTestBase):
  """Zero-argument invocation and help text."""

  def test_bare_invocation_prints_banner_and_help(self):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
      code = self.run_cli([])
    self.assertEqual(code, app.EXIT_OK)
    self.assertIn("doctor", buffer.getvalue())
    self.assertIn("Tip:", self.output)

  def test_help_exits_cleanly(self):
    with contextlib.redirect_stdout(io.StringIO()):
      with self.assertRaises(SystemExit) as ctx:
        self.run_cli(["--help"])
    self.assertEqual(ctx.exception.code, 0)


class StatusCommandTest(CommandTestBase):
  """``status`` in human, JSON and watch modes."""

  def test_missing_state_is_reported_with_a_hint(self):
    code = self.run_cli(["status", "--task", "npov"])
    self.assertEqual(code, app.EXIT_NOT_FOUND)
    self.assertIn("No campaign state found", self.output)
    self.assertIn("run --task npov", self.output)

  def test_missing_state_in_json_mode(self):
    code, payload = self.run_cli_json(["status", "--task", "npov", "--json"])
    self.assertEqual(code, app.EXIT_NOT_FOUND)
    self.assertEqual(json.loads(payload)["error"], "no_state_file")

  def test_renders_a_known_campaign(self):
    write_state(
        app.CHECKPOINTS_ROOT,
        stages={
            "sft": StageResult(
                status=StageStatus.COMPLETED,
                best_metric_val=0.31,
                model_repo_id="leobianco/npov_SFT",
            )
        },
    )
    code = self.run_cli(["status", "--task", "npov"])
    self.assertEqual(code, app.EXIT_OK)
    self.assertIn("camp_npov_1", self.output)
    self.assertIn("SFT Sweep", self.output)

  def test_json_payload_shape(self):
    write_state(
        app.CHECKPOINTS_ROOT,
        stages={"sft": StageResult(status=StageStatus.COMPLETED)},
    )
    code, payload = self.run_cli_json(["status", "--task", "npov", "--json"])
    self.assertEqual(code, app.EXIT_OK)
    data = json.loads(payload)
    self.assertEqual(data["campaign_id"], "camp_npov_1")
    self.assertEqual(data["stages_completed"], 1)
    self.assertEqual(data["stages_total"], 4)
    self.assertIn("sft", data["stages"])

  def test_explicit_state_file_is_honored(self):
    path = write_state(os.path.join(self.tmp.name, "elsewhere"))
    code = self.run_cli(["status", "--state-file", path])
    self.assertEqual(code, app.EXIT_OK)
    self.assertIn("camp_npov_1", self.output)

  def test_final_metrics_are_shown(self):
    write_state(
        app.CHECKPOINTS_ROOT,
        stages={
            "eval": StageResult(
                status=StageStatus.COMPLETED,
                metrics={"hallucination_rate": 0.12},
            )
        },
    )
    self.run_cli(["status", "--task", "npov"])
    self.assertIn("hallucination_rate", self.output)
    self.assertIn("0.12", self.output)

  def test_watch_mode_stops_after_max_refreshes(self):
    write_state(app.CHECKPOINTS_ROOT)
    code = self.run_cli(
        ["status", "--task", "npov", "--watch", "--interval", "0.5",
         "--max-refreshes", "2"]
    )
    self.assertEqual(code, app.EXIT_OK)
    self.assertEqual(self.output.count("Ctrl-C to exit"), 2)
    self.assertIn("camp_npov_1", self.output)


class ListCommandTest(CommandTestBase):
  """``list``/``ls``."""

  def test_empty_listing_suggests_the_wizard(self):
    code = self.run_cli(["list"])
    self.assertEqual(code, app.EXIT_OK)
    self.assertIn("No campaigns found", self.output)
    self.assertIn("wizard", self.output)

  def test_lists_all_tasks(self):
    write_state(app.CHECKPOINTS_ROOT, task="npov", campaign_id="a")
    write_state(app.CHECKPOINTS_ROOT, task="bosch", campaign_id="b")
    code = self.run_cli(["list"])
    self.assertEqual(code, app.EXIT_OK)
    self.assertIn("a", self.output)
    self.assertIn("b", self.output)

  def test_json_listing(self):
    write_state(
        app.CHECKPOINTS_ROOT,
        stages={"sft": StageResult(status=StageStatus.COMPLETED)},
    )
    code, payload = self.run_cli_json(["list", "--json"])
    self.assertEqual(code, app.EXIT_OK)
    entries = json.loads(payload)
    self.assertEqual(len(entries), 1)
    self.assertEqual(entries[0]["progress"], "1/4")

  def test_alias_ls_behaves_identically(self):
    write_state(app.CHECKPOINTS_ROOT)
    self.assertEqual(self.run_cli(["ls"]), app.EXIT_OK)
    self.assertIn("camp_npov_1", self.output)

  def test_corrupt_state_files_are_skipped(self):
    write_state(app.CHECKPOINTS_ROOT, campaign_id="good")
    bad = os.path.join(app.CHECKPOINTS_ROOT, "npov", "bad_state.json")
    with open(bad, "w", encoding="utf-8") as handle:
      handle.write("{not json")
    code, payload = self.run_cli_json(["list", "--json"])
    self.assertEqual(code, app.EXIT_OK)
    self.assertEqual(len(json.loads(payload)), 1)


class DoctorCommandTest(CommandTestBase):
  """``doctor``."""

  def test_json_mode_lists_checks(self):
    _, payload = self.run_cli_json(["doctor", "--json"])
    checks = json.loads(payload)
    self.assertTrue(any(c["name"] == "package: yaml" for c in checks))

  def test_human_mode_prints_every_check(self):
    self.run_cli(["doctor"])
    self.assertIn("Pre-flight check", self.output)
    self.assertIn("package: yaml", self.output)

  def test_failures_produce_an_error_exit_code(self):
    with mock.patch.object(
        app,
        "run_diagnostics",
        return_value=[{"name": "x", "status": "fail", "detail": "nope"}],
    ):
      code = self.run_cli(["doctor"])
    self.assertEqual(code, app.EXIT_ERROR)
    self.assertIn("blocking issue", self.output)

  def test_warnings_do_not_fail(self):
    with mock.patch.object(
        app,
        "run_diagnostics",
        return_value=[{"name": "x", "status": "warn", "detail": "meh"}],
    ):
      code = self.run_cli(["doctor"])
    self.assertEqual(code, app.EXIT_OK)
    self.assertIn("warning", self.output)

  def test_clean_environment_is_celebrated(self):
    with mock.patch.object(
        app,
        "run_diagnostics",
        return_value=[{"name": "x", "status": "ok", "detail": "fine"}],
    ):
      code = self.run_cli(["doctor"])
    self.assertEqual(code, app.EXIT_OK)
    self.assertIn("healthy", self.output)

  def test_doctor_entity_flag(self):
    with mock.patch.object(app, "run_diagnostics", return_value=[]) as mock_diag:
      self.run_cli(["doctor", "--entity", "my-wandb-team"])
      config_arg = mock_diag.call_args[0][0]
      self.assertEqual(config_arg.wandb_entity, "my-wandb-team")


class RunCommandTest(CommandTestBase):
  """``run`` argument handling, with execution stubbed out."""

  def setUp(self):
    super().setUp()
    self.executed = {}

    def fake_execute(config, console, state=None, plain=False):
      del console, state
      self.executed["config"] = config
      self.executed["plain"] = plain
      return app.EXIT_OK

    patcher = mock.patch.object(app, "_execute_campaign", fake_execute)
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_preset_sets_the_budget(self):
    code = self.run_cli(["run", "--preset", "smoke", "--dry-run", "--yes"])
    self.assertEqual(code, app.EXIT_OK)
    self.assertEqual(self.executed["config"].sft.max_runs, 1)

  def test_explicit_flags_beat_the_preset(self):
    self.run_cli(
        ["run", "--preset", "smoke", "--sft-runs", "9", "--dry-run", "--yes"]
    )
    self.assertEqual(self.executed["config"].sft.max_runs, 9)

  def test_stage_subset_is_normalized(self):
    self.run_cli(
        [
            "run",
            "--stages",
            "eval, SFT",
            "--dry-run",
            "--yes",
        ]
    )
    self.assertEqual(self.executed["config"].stages, ["sft", "eval"])

  def test_missing_upstream_checkpoint_is_rejected(self):
    code = self.run_cli(["run", "--stages", "perl,eval", "--dry-run", "--yes"])
    self.assertEqual(code, app.EXIT_USAGE)
    self.assertIn("--sft-model", self.output)
    self.assertIn("--reward-model", self.output)
    self.assertNotIn("config", self.executed)

  def test_supplying_checkpoints_unblocks_the_run(self):
    code = self.run_cli(
        [
            "run",
            "--stages",
            "perl,eval",
            "--sft-model",
            "leobianco/npov_SFT_x",
            "--reward-model",
            "leobianco/npov_RM_x",
            "--dry-run",
            "--yes",
        ]
    )
    self.assertEqual(code, app.EXIT_OK)
    self.assertEqual(
        self.executed["config"].perl.sft_model_path, "leobianco/npov_SFT_x"
    )

  def test_entity_and_user_flags(self):
    self.run_cli(
        [
            "run",
            "--preset",
            "smoke",
            "--entity",
            "team-alpha",
            "--user",
            "hf-tester",
            "--dry-run",
            "--yes",
        ]
    )
    self.assertEqual(self.executed["config"].wandb_entity, "team-alpha")
    self.assertEqual(self.executed["config"].user, "hf-tester")

  def test_launch_preview_is_printed(self):
    self.run_cli(["run", "--preset", "smoke", "--dry-run", "--yes"])
    self.assertIn("Launching campaign", self.output)
    self.assertIn("Estimated time", self.output)

  def test_missing_config_file(self):
    code = self.run_cli(["run", "--config", "nope.yaml"])
    self.assertEqual(code, app.EXIT_NOT_FOUND)
    self.assertIn("not found", self.output)

  def test_config_file_is_loaded(self):
    config = CampaignConfig.create_default(task_name="bosch", sft_runs=3)
    config.to_yaml("campaign.yaml")
    self.run_cli(["run", "--config", "campaign.yaml", "--dry-run", "--yes"])
    self.assertEqual(self.executed["config"].task_name, "bosch")
    self.assertEqual(self.executed["config"].sft.max_runs, 3)

  def test_plain_flag_disables_the_dashboard(self):
    self.run_cli(["run", "--preset", "smoke", "--dry-run", "--yes", "--plain"])
    self.assertTrue(self.executed["plain"])
    self.assertTrue(self.executed["config"].no_tui)

  def test_no_tui_flag_disables_the_dashboard(self):
    self.run_cli(
        ["run", "--preset", "smoke", "--dry-run", "--yes", "--no-tui"]
    )
    self.assertTrue(self.executed["plain"])

  def test_interactive_flag_delegates_to_the_wizard(self):
    with mock.patch.object(app, "cmd_wizard", return_value=app.EXIT_OK) as w:
      self.run_cli(["run", "--interactive"])
    w.assert_called_once()


class ResumeCommandTest(CommandTestBase):
  """``resume``."""

  def setUp(self):
    super().setUp()
    self.executed = {}

    def fake_execute(config, console, state=None, plain=False):
      del console, plain
      self.executed["config"] = config
      self.executed["state"] = state
      return app.EXIT_OK

    patcher = mock.patch.object(app, "_execute_campaign", fake_execute)
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_missing_state(self):
    code = self.run_cli(["resume", "--task", "npov"])
    self.assertEqual(code, app.EXIT_NOT_FOUND)
    self.assertIn("No campaign state found", self.output)

  def test_resume_plan_is_shown(self):
    write_state(
        app.CHECKPOINTS_ROOT,
        stages={"sft": StageResult(status=StageStatus.COMPLETED)},
    )
    code = self.run_cli(["resume", "--task", "npov", "--yes", "--dry-run"])
    self.assertEqual(code, app.EXIT_OK)
    self.assertIn("Resume plan", self.output)
    self.assertIn("SFT Sweep", self.output)
    self.assertTrue(self.executed["config"].dry_run)
    self.assertIsNotNone(self.executed["state"])

  def test_persisted_config_is_restored(self):
    config = CampaignConfig.create_default(task_name="npov", perl_runs=13)
    write_state(app.CHECKPOINTS_ROOT, config_dict=config.to_dict())
    self.run_cli(["resume", "--task", "npov", "--yes"])
    self.assertEqual(self.executed["config"].perl.max_runs, 13)

  def test_resume_entity_and_user_flags(self):
    write_state(
        app.CHECKPOINTS_ROOT,
        stages={"sft": StageResult(status=StageStatus.COMPLETED)},
    )
    code = self.run_cli([
        "resume",
        "--task",
        "npov",
        "--entity",
        "resumed-team",
        "--user",
        "resumed-user",
        "--yes",
        "--dry-run",
    ])
    self.assertEqual(code, app.EXIT_OK)
    self.assertEqual(self.executed["config"].wandb_entity, "resumed-team")
    self.assertEqual(self.executed["config"].user, "resumed-user")

  def test_completed_campaign_is_not_resumed(self):
    done = {
        name: StageResult(status=StageStatus.COMPLETED)
        for name in ("sft", "rm", "perl", "eval")
    }
    write_state(app.CHECKPOINTS_ROOT, status="COMPLETED", stages=done)
    code = self.run_cli(["resume", "--task", "npov", "--yes"])
    self.assertEqual(code, app.EXIT_OK)
    self.assertIn("already complete", self.output)
    self.assertNotIn("config", self.executed)


class ReportCommandTest(CommandTestBase):
  """``report``."""

  def test_missing_state(self):
    self.assertEqual(
        self.run_cli(["report", "--task", "npov"]), app.EXIT_NOT_FOUND
    )

  def test_artifacts_are_listed(self):
    write_state(app.CHECKPOINTS_ROOT)
    fake_reporter = mock.MagicMock()
    fake_reporter.return_value.generate_all.return_value = {
        "markdown": "reports/x.md"
    }
    with mock.patch(
        "src.orchestrator.reporter.CampaignReporter", fake_reporter
    ):
      code = self.run_cli(["report", "--task", "npov"])
    self.assertEqual(code, app.EXIT_OK)
    self.assertIn("reports/x.md", self.output)


class ErrorHandlingTest(CommandTestBase):
  """Friendly failures and exit codes."""

  def test_unexpected_errors_are_summarized(self):
    with mock.patch.object(app, "cmd_list", side_effect=RuntimeError("boom")):
      code = self.run_cli(["list"])
    self.assertEqual(code, app.EXIT_ERROR)
    self.assertIn("RuntimeError: boom", self.output)
    self.assertIn("--debug", self.output)

  def test_debug_flag_reraises(self):
    with mock.patch.object(app, "cmd_list", side_effect=RuntimeError("boom")):
      with self.assertRaises(RuntimeError):
        self.run_cli(["list", "--debug"])

  def test_keyboard_interrupt_is_graceful(self):
    with mock.patch.object(app, "cmd_list", side_effect=KeyboardInterrupt):
      code = self.run_cli(["list"])
    self.assertEqual(code, app.EXIT_ERROR)
    self.assertIn("Interrupted", self.output)


class ThemeFlagTest(unittest.TestCase):
  """Global appearance flags build the right console."""

  def test_no_color_disables_styling(self):
    captured = {}

    def spy(args, console):
      del args
      captured["theme"] = console.theme
      return app.EXIT_OK

    with mock.patch.object(app, "cmd_doctor", spy):
      app.main(["doctor", "--no-color"])
    self.assertFalse(captured["theme"].use_color)

  def test_ascii_flag_selects_ascii_glyphs(self):
    captured = {}

    def spy(args, console):
      del args
      captured["theme"] = console.theme
      return app.EXIT_OK

    with mock.patch.object(app, "cmd_doctor", spy):
      app.main(["doctor", "--ascii"])
    self.assertFalse(captured["theme"].use_unicode)
    captured["theme"].glyphs.completed.encode("ascii")  # Must not raise.


if __name__ == "__main__":
  unittest.main()
