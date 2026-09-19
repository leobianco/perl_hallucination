"""Tests for powering the VM off at the end of an unattended campaign.

Nothing here may execute a real shutdown, so every test injects either the
command runner or the grace-period waiter. The policy itself
(:func:`shutdown.should_shutdown`) is pure and is tested directly.
"""

from __future__ import annotations

import io
import unittest
from unittest import mock

from src.orchestrator import shutdown
from src.orchestrator.cli import app
from src.orchestrator.cli import console as console_mod
from src.orchestrator.cli import theme as theme_mod
from src.orchestrator.cli import wizard
from src.orchestrator.config import CampaignConfig


def make_console() -> console_mod.UiConsole:
  """Builds an in-memory, plain-text console for assertions."""
  return console_mod.UiConsole(
      theme=theme_mod.detect_theme(force_color=False, force_ascii=True),
      file=io.StringIO(),
      force_plain=True,
  )


class _Config:
  """Minimal stand-in for the fields the shutdown policy reads."""

  def __init__(self, armed=True, dry_run=False, grace=0):
    self.shutdown_when_done = armed
    self.dry_run = dry_run
    self.shutdown_grace_seconds = grace


class PolicyTest(unittest.TestCase):
  """Which outcomes justify powering the machine off."""

  def test_it_is_off_unless_asked_for(self):
    wanted, reason = shutdown.should_shutdown(_Config(armed=False), "COMPLETED")
    self.assertFalse(wanted)
    self.assertIn("shutdown_when_done", reason)

  def test_a_finished_campaign_powers_off(self):
    wanted, reason = shutdown.should_shutdown(_Config(), "COMPLETED")
    self.assertTrue(wanted)
    self.assertIn("COMPLETED", reason)

  def test_a_crashed_campaign_powers_off_too(self):
    # The expensive case: twelve hours booked, dead after two.
    wanted, _ = shutdown.should_shutdown(_Config(), "FAILED")
    self.assertTrue(wanted)

  def test_a_hand_stopped_campaign_does_not(self):
    # Reaching STOPPED requires the stop hotkey or Ctrl-C, so a human is
    # present and pulling the machine from under them is never the intent.
    wanted, reason = shutdown.should_shutdown(_Config(), "STOPPED")
    self.assertFalse(wanted)
    self.assertIn("by hand", reason)

  def test_a_dry_run_never_powers_off(self):
    wanted, reason = shutdown.should_shutdown(
        _Config(dry_run=True), "COMPLETED"
    )
    self.assertFalse(wanted)
    self.assertIn("dry run", reason)

  def test_an_unknown_status_does_not(self):
    for status in ("", "DETACHED", "WHATEVER", None):
      wanted, _ = shutdown.should_shutdown(_Config(), status)
      self.assertFalse(wanted, f"status {status!r} should not power off")

  def test_the_status_is_normalized(self):
    wanted, _ = shutdown.should_shutdown(_Config(), "  completed ")
    self.assertTrue(wanted)

  def test_a_config_without_the_field_is_treated_as_off(self):
    # Old state files rebuild configs that predate the flag.
    wanted, _ = shutdown.should_shutdown(object(), "COMPLETED")
    self.assertFalse(wanted)


class GracePeriodTest(unittest.TestCase):
  """The cancellable countdown."""

  def test_a_zero_grace_period_does_not_wait(self):
    sleeper = mock.Mock()
    self.assertTrue(shutdown._wait_out_grace_period(0, sleeper=sleeper))
    sleeper.assert_not_called()

  def test_it_waits_out_the_whole_period(self):
    ticks = iter([0.0] + [float(i) for i in range(1, 12)])
    elapsed = []
    self.assertTrue(
        shutdown._wait_out_grace_period(
            5,
            sleeper=elapsed.append,
            clock=lambda: next(ticks),
        )
    )
    # Sleeps in one-second steps so Ctrl-C stays responsive, and stops once
    # the period is up rather than overshooting it.
    self.assertTrue(elapsed)
    self.assertTrue(all(0 < step <= 1.0 for step in elapsed))
    self.assertLessEqual(sum(elapsed), 5)

  def test_ctrl_c_cancels_it(self):
    def interrupt(_seconds):
      raise KeyboardInterrupt

    self.assertFalse(
        shutdown._wait_out_grace_period(
            5, sleeper=interrupt, clock=lambda: 0.0
        )
    )

  def test_it_announces_the_countdown(self):
    console = make_console()
    # Two zeros: the first call fixes the deadline, the second is the first
    # reading inside the loop, so the countdown opens at the full period.
    ticks = iter([0.0, 0.0] + [float(i) for i in range(1, 40)])
    shutdown._wait_out_grace_period(
        3, console=console, sleeper=lambda _s: None, clock=lambda: next(ticks)
    )
    text = console.file.getvalue()
    self.assertIn("Ctrl-C to cancel", text)
    self.assertIn("3s", text)



class ExecutionTest(unittest.TestCase):
  """What ``maybe_shutdown`` actually does."""

  def test_it_issues_the_command(self):
    runner = mock.Mock()
    self.assertTrue(
        shutdown.maybe_shutdown(
            _Config(), "COMPLETED", runner=runner, waiter=lambda *a: True
        )
    )
    runner.assert_called_once_with(["sudo", "shutdown", "-h", "now"])

  def test_it_does_nothing_when_disarmed(self):
    runner = mock.Mock()
    self.assertFalse(
        shutdown.maybe_shutdown(
            _Config(armed=False),
            "COMPLETED",
            runner=runner,
            waiter=lambda *a: True,
        )
    )
    runner.assert_not_called()

  def test_a_cancelled_countdown_spares_the_machine(self):
    runner = mock.Mock()
    self.assertFalse(
        shutdown.maybe_shutdown(
            _Config(), "COMPLETED", runner=runner, waiter=lambda *a: False
        )
    )
    runner.assert_not_called()

  def test_the_configured_grace_period_is_honoured(self):
    seen = []

    def waiter(seconds, _console=None):
      seen.append(seconds)
      return True

    shutdown.maybe_shutdown(
        _Config(grace=45), "COMPLETED", runner=mock.Mock(), waiter=waiter
    )
    self.assertEqual(seen, [45])

  def test_a_failing_command_does_not_raise(self):
    # The campaign already succeeded and wrote its report; turning that into
    # a traceback would be worse than a VM that stays up.
    def explode(_cmd):
      raise OSError("sudo: command not found")

    console = make_console()
    self.assertFalse(
        shutdown.maybe_shutdown(
            _Config(),
            "COMPLETED",
            console=console,
            runner=explode,
            waiter=lambda *a: True,
        )
    )
    self.assertIn("Could not power off", console.file.getvalue())

  def test_it_explains_itself_when_armed_but_skipped(self):
    console = make_console()
    shutdown.maybe_shutdown(
        _Config(), "STOPPED", console=console, runner=mock.Mock(),
        waiter=lambda *a: True,
    )
    self.assertIn("Not powering off", console.file.getvalue())

  def test_it_stays_quiet_when_never_armed(self):
    console = make_console()
    shutdown.maybe_shutdown(
        _Config(armed=False), "COMPLETED", console=console,
        runner=mock.Mock(), waiter=lambda *a: True,
    )
    self.assertEqual(console.file.getvalue().strip(), "")


class ConfigTest(unittest.TestCase):
  """The YAML surface."""

  def test_it_is_off_by_default(self):
    config = CampaignConfig.create_default(task_name="ragtruth")
    self.assertFalse(config.shutdown_when_done)
    self.assertEqual(config.shutdown_grace_seconds, 60)
    config.validate()

  def test_it_round_trips_through_the_config_dict(self):
    config = CampaignConfig.create_default(task_name="ragtruth")
    config.shutdown_when_done = True
    config.shutdown_grace_seconds = 5
    restored = CampaignConfig.from_dict(config.to_dict())
    self.assertTrue(restored.shutdown_when_done)
    self.assertEqual(restored.shutdown_grace_seconds, 5)

  def test_a_negative_grace_period_is_rejected(self):
    config = CampaignConfig.create_default(task_name="ragtruth")
    config.shutdown_when_done = True
    config.shutdown_grace_seconds = -1
    with self.assertRaises(ValueError) as ctx:
      config.validate()
    self.assertIn("shutdown_grace_seconds", str(ctx.exception))

  def test_zero_is_allowed(self):
    config = CampaignConfig.create_default(task_name="ragtruth")
    config.shutdown_when_done = True
    config.shutdown_grace_seconds = 0
    config.validate()


class ParserTest(unittest.TestCase):
  """The command line override."""

  def setUp(self):
    super().setUp()
    self.parser = app.build_parser()

  def test_absent_means_defer_to_the_config(self):
    for command in ("run", "resume"):
      args = self.parser.parse_args([command, "--task", "npov"])
      self.assertIsNone(args.shutdown, command)

  def test_it_can_be_armed_and_disarmed(self):
    for command in ("run", "resume"):
      self.assertTrue(
          self.parser.parse_args([command, "--shutdown"]).shutdown, command
      )
      self.assertFalse(
          self.parser.parse_args([command, "--no-shutdown"]).shutdown, command
      )

  def test_the_two_flags_are_mutually_exclusive(self):
    with self.assertRaises(SystemExit):
      self.parser.parse_args(["run", "--shutdown", "--no-shutdown"])


class PreviewTest(unittest.TestCase):
  """The pre-launch review block."""

  def _lines(self, config) -> str:
    theme = theme_mod.detect_theme(force_color=False, force_ascii=True)
    return "\n".join(wizard.config_summary_lines(config, theme))

  def test_an_armed_shutdown_is_announced(self):
    config = CampaignConfig.create_default(task_name="ragtruth")
    config.shutdown_when_done = True
    config.shutdown_grace_seconds = 30
    text = self._lines(config)
    self.assertIn("POWER OFF", text)
    self.assertIn("30s", text)

  def test_nothing_is_said_when_it_is_off(self):
    config = CampaignConfig.create_default(task_name="ragtruth")
    self.assertNotIn("POWER OFF", self._lines(config))


class ExecutionHookTest(unittest.TestCase):
  """``_execute_campaign`` must reach the hook on every exit path."""

  def setUp(self):
    super().setUp()
    self.config = CampaignConfig.create_default(task_name="ragtruth")
    self.console = make_console()

  def _run(self, dashboard_result=None, dashboard_error=None):
    """Runs ``_execute_campaign`` with the dashboard stubbed out."""
    calls = []

    def fake_dashboard(**_kwargs):
      if dashboard_error is not None:
        raise dashboard_error
      return dashboard_result

    def fake_shutdown(_config, status, _console=None):
      calls.append(status)
      return False

    with mock.patch(
        "src.orchestrator.cli.dashboard.run_campaign_with_dashboard",
        fake_dashboard,
    ):
      with mock.patch.object(shutdown, "maybe_shutdown", fake_shutdown):
        try:
          code = app._execute_campaign(self.config, self.console)
        except BaseException as error:  # pylint: disable=broad-except
          # KeyboardInterrupt is not an Exception, and it is one of the
          # paths under test.
          return calls, error
    return calls, code

  def test_a_completed_campaign_reports_completed(self):
    calls, code = self._run({"status": "COMPLETED"})
    self.assertEqual(calls, ["COMPLETED"])
    self.assertEqual(code, app.EXIT_OK)

  def test_a_stopped_campaign_reports_stopped(self):
    calls, _ = self._run({"status": "STOPPED"})
    self.assertEqual(calls, ["STOPPED"])

  def test_a_crash_still_reaches_the_hook(self):
    # The engine re-raises out of the dashboard instead of returning a
    # status, so without the `finally` this path would never power off.
    calls, error = self._run(dashboard_error=RuntimeError("gpu fell over"))
    self.assertEqual(calls, ["FAILED"])
    self.assertIsInstance(error, RuntimeError)

  def test_ctrl_c_is_not_reported_as_a_failure(self):
    calls, error = self._run(dashboard_error=KeyboardInterrupt())
    self.assertEqual(calls, ["STOPPED"])
    self.assertIsInstance(error, KeyboardInterrupt)


if __name__ == "__main__":
  unittest.main()
