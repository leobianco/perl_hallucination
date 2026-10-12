"""Tests for powering the VM off from outside the campaign process.

Nothing here may execute a real shutdown or a real sleep, so every test
injects the runner, the waiter and the clock. The two things worth pinning:

1. The watcher waits for the process to actually be gone. A decision taken
   while the campaign is mid-stage would halt a machine that is working.
2. The watcher does not own the policy. ``COMPLETED``/``FAILED`` power off,
   ``STOPPED``/``PAUSED`` do not, and that is decided by the same function the
   in-process path calls - so the two can never disagree.
"""

import os
import tempfile
import unittest
from unittest import mock

from src.orchestrator import shutdown
from src.orchestrator import shutdown_watcher
from src.orchestrator.state import CampaignState


class IsRunningTest(unittest.TestCase):
  """Liveness detection."""

  def test_the_current_process_is_running(self):
    self.assertTrue(shutdown_watcher.is_running(os.getpid()))

  def test_a_nonexistent_pid_is_not_running(self):
    with mock.patch.object(
        shutdown_watcher.os, "kill", side_effect=ProcessLookupError
    ):
      self.assertFalse(shutdown_watcher.is_running(424242))

  def test_a_foreign_pid_counts_as_running(self):
    """EPERM means the process exists and belongs to somebody else."""
    with mock.patch.object(
        shutdown_watcher.os, "kill", side_effect=PermissionError
    ):
      self.assertTrue(shutdown_watcher.is_running(1))

  def test_nonpositive_pids_are_rejected(self):
    # kill(0, sig) signals the whole process group, and kill(-1, sig) every
    # process the user owns. Neither may be reachable from a liveness check.
    self.assertFalse(shutdown_watcher.is_running(0))
    self.assertFalse(shutdown_watcher.is_running(-1))


class FindCampaignPidsTest(unittest.TestCase):
  """Discovering a campaign without a pid file."""

  def _proc(self, entries):
    """Builds a fake /proc.

    Args:
      entries: Map of pid -> command line string.

    Returns:
      Path to the fixture directory.
    """
    root = tempfile.mkdtemp()
    for pid, cmdline in entries.items():
      os.makedirs(os.path.join(root, str(pid)))
      with open(os.path.join(root, str(pid), "cmdline"), "wb") as handle:
        handle.write(cmdline.replace(" ", "\0").encode("utf-8"))
    os.makedirs(os.path.join(root, "self"))  # non-numeric, must be skipped
    return root

  def test_finds_the_campaign(self):
    root = self._proc({
        11: "python3 scripts/run_campaign.py run --task npov",
        12: "sshd",
    })
    self.assertEqual(
        shutdown_watcher.find_campaign_pids(proc_root=root), [11]
    )

  def test_excludes_itself(self):
    """The watcher's own command line names the script it greps for."""
    root = self._proc({
        11: "python3 scripts/run_campaign.py run --task npov",
        99: "python3 scripts/shutdown_when_done.py --task npov",
    })
    self.assertEqual(
        shutdown_watcher.find_campaign_pids(proc_root=root, exclude_pids=[99]),
        [11],
    )

  def test_no_campaign_is_not_an_error(self):
    root = self._proc({12: "sshd"})
    self.assertEqual(shutdown_watcher.find_campaign_pids(proc_root=root), [])

  def test_a_missing_proc_is_not_an_error(self):
    self.assertEqual(
        shutdown_watcher.find_campaign_pids(proc_root="/nonexistent"), []
    )


class WaitForExitTest(unittest.TestCase):
  """Waiting for a sibling process."""

  def test_returns_when_the_process_goes_away(self):
    calls = {"n": 0}

    def running(_pid):
      calls["n"] += 1
      return calls["n"] < 3

    self.assertTrue(
        shutdown_watcher.wait_for_exit(
            7, poll_seconds=0, sleeper=lambda _s: None, running=running
        )
    )

  def test_gives_up_at_the_bound(self):
    """A watcher that outlives its campaign must not linger for a week."""
    now = {"t": 0.0}

    def clock():
      now["t"] += 10.0
      return now["t"]

    self.assertFalse(
        shutdown_watcher.wait_for_exit(
            7,
            poll_seconds=0,
            max_wait_seconds=20.0,
            sleeper=lambda _s: None,
            clock=clock,
            running=lambda _pid: True,
        )
    )

  def test_a_nonpositive_bound_waits_indefinitely(self):
    calls = {"n": 0}

    def running(_pid):
      calls["n"] += 1
      return calls["n"] < 5

    self.assertTrue(
        shutdown_watcher.wait_for_exit(
            7,
            poll_seconds=0,
            max_wait_seconds=0,
            sleeper=lambda _s: None,
            clock=lambda: 0.0,
            running=running,
        )
    )


class ResolveStatusTest(unittest.TestCase):
  """Reading the recorded outcome."""

  def _state_file(self, status):
    state = CampaignState(campaign_id="c", task_name="npov")
    state.status = status
    path = os.path.join(tempfile.mkdtemp(), "campaign_state.json")
    state.save(path)
    return path

  def test_reads_a_terminal_status(self):
    status, _ = shutdown_watcher.resolve_status(self._state_file("COMPLETED"))
    self.assertEqual(status, "COMPLETED")

  def test_in_progress_is_reported_as_is_by_default(self):
    """A campaign that died without recording an outcome stays up."""
    status, explanation = shutdown_watcher.resolve_status(
        self._state_file("IN_PROGRESS")
    )
    self.assertEqual(status, "IN_PROGRESS")
    self.assertIn("--assume-failed", explanation)
    # And the shared policy declines it, which is the property that matters.
    wanted, _ = shutdown.should_shutdown(
        shutdown_watcher.WatchConfig(), status
    )
    self.assertFalse(wanted)

  def test_assume_failed_promotes_in_progress(self):
    status, _ = shutdown_watcher.resolve_status(
        self._state_file("IN_PROGRESS"), assume_failed=True
    )
    self.assertEqual(status, "FAILED")

  def test_a_missing_state_file_yields_no_status(self):
    status, explanation = shutdown_watcher.resolve_status(None)
    self.assertEqual(status, "")
    self.assertIn("no campaign state file", explanation)

  def test_an_unreadable_state_file_is_not_a_reason_to_power_off(self):
    path = os.path.join(tempfile.mkdtemp(), "campaign_state.json")
    with open(path, "w") as handle:
      handle.write("{not json")
    status, _ = shutdown_watcher.resolve_status(path)
    self.assertEqual(status, "")
    wanted, _ = shutdown.should_shutdown(
        shutdown_watcher.WatchConfig(), status
    )
    self.assertFalse(wanted)


class WatchAndShutdownTest(unittest.TestCase):
  """End to end, with the halt itself stubbed out."""

  def setUp(self):
    super().setUp()
    self.runner = mock.Mock()
    # Grace period elapses instantly; its own countdown is covered by
    # test_shutdown.py.
    self.grace_waiter = lambda *_a, **_k: True

  def _state_file(self, status):
    state = CampaignState(campaign_id="c", task_name="npov")
    state.status = status
    path = os.path.join(tempfile.mkdtemp(), "campaign_state.json")
    state.save(path)
    return path

  def _run(self, status, **kwargs):
    return shutdown_watcher.watch_and_shutdown(
        pid=None,
        state_path=self._state_file(status),
        runner=self.runner,
        grace_waiter=self.grace_waiter,
        **kwargs,
    )

  def test_a_completed_campaign_powers_off(self):
    self.assertTrue(self._run("COMPLETED"))
    self.runner.assert_called_once_with(list(shutdown.SHUTDOWN_COMMAND))

  def test_a_failed_campaign_powers_off(self):
    """The most expensive case to leave running is a crash at hour two."""
    self.assertTrue(self._run("FAILED"))
    self.runner.assert_called_once()

  def test_a_stopped_campaign_stays_up(self):
    """STOPPED means somebody was at the keyboard."""
    self.assertFalse(self._run("STOPPED"))
    self.runner.assert_not_called()

  def test_a_paused_campaign_stays_up(self):
    self.assertFalse(self._run("PAUSED"))
    self.runner.assert_not_called()

  def test_dry_run_never_halts(self):
    self.assertFalse(self._run("COMPLETED", dry_run=True))
    self.runner.assert_not_called()

  def test_it_waits_for_the_process_before_deciding(self):
    """Deciding mid-run would halt a machine that is still working."""
    order = []

    def waiter(pid, **_kwargs):
      order.append(("waited", pid))
      return True

    def runner(cmd):
      order.append(("ran", cmd[0]))

    shutdown_watcher.watch_and_shutdown(
        pid=4242,
        state_path=self._state_file("COMPLETED"),
        waiter=waiter,
        runner=runner,
        grace_waiter=self.grace_waiter,
    )
    self.assertEqual([step[0] for step in order], ["waited", "ran"])

  def test_giving_up_on_the_wait_leaves_the_machine_alone(self):
    result = shutdown_watcher.watch_and_shutdown(
        pid=4242,
        state_path=self._state_file("COMPLETED"),
        waiter=lambda *_a, **_k: False,
        runner=self.runner,
        grace_waiter=self.grace_waiter,
    )
    self.assertFalse(result)
    self.runner.assert_not_called()

  def test_a_broken_halt_does_not_raise(self):
    """Never turn a finished campaign into a traceback."""
    runner = mock.Mock(side_effect=OSError("no sudo for you"))
    result = shutdown_watcher.watch_and_shutdown(
        pid=None,
        state_path=self._state_file("COMPLETED"),
        runner=runner,
        grace_waiter=self.grace_waiter,
    )
    self.assertFalse(result)


class RequireTargetTest(unittest.TestCase):
  """Refusing to act on a state file that may describe a previous run.

  This is not hypothetical: the first manual smoke test of the CLI, with no
  campaign running, found a months-old ``COMPLETED`` state and would have
  powered the machine off on the spot.
  """

  def test_a_live_process_is_enough(self):
    self.assertIsNone(shutdown_watcher.require_target(4242, None))

  def test_nothing_running_is_refused_by_default(self):
    complaint = shutdown_watcher.require_target(None, "/tmp/state.json")
    self.assertIsNotNone(complaint)
    self.assertIn("--decide-now", complaint)

  def test_decide_now_allows_it(self):
    self.assertIsNone(
        shutdown_watcher.require_target(
            None, "/tmp/state.json", decide_now=True
        )
    )

  def test_decide_now_still_needs_a_state_file(self):
    complaint = shutdown_watcher.require_target(None, None, decide_now=True)
    self.assertIsNotNone(complaint)
    self.assertIn("refusing", complaint.lower())


class PolicyParityTest(unittest.TestCase):
  """The watcher must not grow a second, divergent policy."""

  def test_the_watcher_config_satisfies_the_shared_policy(self):
    config = shutdown_watcher.WatchConfig(grace_seconds=5)
    self.assertTrue(config.shutdown_when_done)
    self.assertEqual(config.shutdown_grace_seconds, 5)
    for status in ("COMPLETED", "FAILED"):
      wanted, _ = shutdown.should_shutdown(config, status)
      self.assertTrue(wanted, status)
    for status in ("STOPPED", "PAUSED", "IN_PROGRESS", ""):
      wanted, _ = shutdown.should_shutdown(config, status)
      self.assertFalse(wanted, status)


if __name__ == "__main__":
  unittest.main()
