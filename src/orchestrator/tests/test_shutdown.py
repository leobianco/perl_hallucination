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


class _FakeClock:
  """A monotonic clock that only advances when something sleeps.

  Lets a five-minute countdown be tested in microseconds, and - the point
  here - lets the test measure *when* each notice was emitted rather than
  merely that it was.
  """

  def __init__(self):
    self.now = 0.0

  def __call__(self) -> float:
    return self.now

  def sleep(self, seconds: float) -> None:
    self.now += seconds


class _TimedConsole:
  """Console double that records the time of every notice."""

  def __init__(self, clock: _FakeClock):
    self._clock = clock
    self.said = []

  def blank(self) -> None:
    pass

  def info(self, message: str) -> None:
    self.said.append((self._clock.now, message))

  def warn(self, message: str) -> None:
    self.said.append((self._clock.now, message))


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

  def test_an_aborted_campaign_does_not_either(self):
    # `[x]` kills the work in flight, so the stage it interrupted raises on
    # its way down and the campaign used to land here as FAILED - i.e. as a
    # crash, which powers the machine off under the person who pressed the
    # key. ABORTED is the status that keeps those two apart.
    wanted, reason = shutdown.should_shutdown(_Config(), "ABORTED")
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

  def test_a_long_countdown_is_never_silent(self):
    """Regression: ``--grace 300`` printed nothing for its first four minutes.

    The announcement schedule stopped at 60s and the loop only announces a
    mark still ahead of it, so nothing was ahead of 300s. An operator
    watching a program whose sole job is to halt the machine saw four
    minutes of nothing, concluded it had hung, and killed it two minutes
    before it would have fired.
    """
    clock = _FakeClock()
    watcher = _TimedConsole(clock)
    self.assertTrue(
        shutdown._wait_out_grace_period(
            300, watcher, sleeper=clock.sleep, clock=clock
        )
    )

    spoken_at = [when for when, _ in watcher.said]
    self.assertTrue(spoken_at, "the countdown said nothing at all")
    # It opens immediately, stating the full period...
    self.assertEqual(spoken_at[0], 0.0)
    self.assertIn("5 min", watcher.said[0][1])
    # ...never goes quiet for longer than one announcement interval...
    gaps = [b - a for a, b in zip(spoken_at, spoken_at[1:])]
    self.assertTrue(
        all(gap <= shutdown._COARSE_INTERVAL for gap in gaps),
        f"silent for {max(gaps):.0f}s between announcements",
    )
    # ...and is still talking on the way out.
    self.assertLessEqual(300 - spoken_at[-1], shutdown._COARSE_INTERVAL)

  def test_a_short_countdown_keeps_counting_in_seconds(self):
    # Minutes would be a silly unit for the 60s default; the tail stays in
    # seconds so the last ten of them are still readable.
    self.assertEqual(shutdown._format_countdown(45), "45s")
    self.assertEqual(shutdown._format_countdown(60), "60s")
    self.assertEqual(shutdown._format_countdown(300), "5 min")

  def test_the_schedule_never_outruns_the_period(self):
    # A mark the countdown can never reach would be announced immediately
    # and then never again, which is the bug this test brackets.
    for period in (1, 5, 59, 60, 61, 130, 300, 3600):
      marks = shutdown._announcement_marks(period)
      self.assertTrue(marks, f"no marks for a {period}s period")
      self.assertLessEqual(max(marks), period)
      self.assertEqual(max(marks), period, "the period itself is a mark")



class ExecutionTest(unittest.TestCase):
  """What ``maybe_shutdown`` actually does."""

  def test_it_issues_the_command(self):
    runner = mock.Mock()
    self.assertTrue(
        shutdown.maybe_shutdown(
            _Config(), "COMPLETED", runner=runner, waiter=lambda *a: True
        )
    )
    runner.assert_called_once_with(list(shutdown.SHUTDOWN_COMMAND))

  def test_it_never_waits_for_a_sudo_password(self):
    # Without -n, a sudoers entry that wants a password turns the last act of
    # a twelve-hour campaign into an invisible prompt on an inherited stdin,
    # and the watcher hangs there until somebody notices the VM is still up.
    self.assertIn("-n", shutdown.SHUTDOWN_COMMAND)

  def test_a_nonzero_exit_is_not_reported_as_success(self):
    # Only exceptions used to be caught, so `sudo` exiting 1 still logged
    # "Shutdown command issued." while the machine happily stayed up.
    class _Result:
      returncode = 1

    console = make_console()
    self.assertFalse(
        shutdown.maybe_shutdown(
            _Config(),
            "COMPLETED",
            console=console,
            runner=lambda _cmd: _Result(),
            waiter=lambda *a: True,
        )
    )
    text = console.file.getvalue()
    self.assertIn("exited 1", text)
    self.assertIn("sudo -n true", text)

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
