"""Tests for the streamed subprocess helper used by every long-running stage.

These cover the failure modes that previously went unnoticed in an unattended
campaign: a silent child that never returns, a stop request that arrives while
the child is quiet, and a non-zero exit code.
"""

import sys
import time
import unittest

from src.orchestrator.process import collect_lines
from src.orchestrator.process import stream_subprocess


def _py(code: str):
  """Builds an argv running ``code`` with the current interpreter."""
  return [sys.executable, "-c", code]


class TestStreamSubprocess(unittest.TestCase):

  def test_streams_lines_and_zero_exit(self):
    lines = []
    outcome = stream_subprocess(
        _py("print('alpha'); print('beta')"),
        on_line=collect_lines(lines),
    )
    self.assertEqual(outcome.returncode, 0)
    self.assertTrue(outcome.ok)
    self.assertFalse(outcome.cut_short)
    self.assertEqual(lines, ["alpha", "beta"])
    self.assertEqual(outcome.line_count, 2)

  def test_captures_stderr_into_the_same_stream(self):
    lines = []
    outcome = stream_subprocess(
        _py("import sys; sys.stderr.write('boom\\n')"),
        on_line=collect_lines(lines),
    )
    self.assertEqual(outcome.returncode, 0)
    self.assertIn("boom", lines)

  def test_non_zero_exit_is_reported(self):
    outcome = stream_subprocess(_py("import sys; sys.exit(3)"))
    self.assertEqual(outcome.returncode, 3)
    self.assertFalse(outcome.ok)
    self.assertFalse(outcome.cut_short)

  def test_timeout_kills_a_silent_child(self):
    # The child never prints, so a readline-based loop would hang forever.
    started = time.time()
    outcome = stream_subprocess(
        _py("import time; time.sleep(60)"),
        timeout_s=1.0,
        poll_interval=0.05,
    )
    self.assertTrue(outcome.timed_out)
    self.assertTrue(outcome.cut_short)
    self.assertFalse(outcome.ok)
    self.assertLess(time.time() - started, 30.0)

  def test_stop_request_terminates_a_silent_child(self):
    started = time.time()

    def _stop():
      return time.time() - started > 0.5

    outcome = stream_subprocess(
        _py("import time; time.sleep(60)"),
        stop_requested=_stop,
        poll_interval=0.05,
    )
    self.assertTrue(outcome.interrupted)
    self.assertFalse(outcome.timed_out)
    self.assertLess(time.time() - started, 30.0)

  def test_child_stdin_is_detached(self):
    # The dashboard puts the TTY in cbreak mode; children must not read it.
    lines = []
    outcome = stream_subprocess(
        _py("import sys; print(repr(sys.stdin.read()))"),
        on_line=collect_lines(lines),
    )
    self.assertEqual(outcome.returncode, 0)
    self.assertEqual(lines, ["''"])

  def test_renderer_exceptions_do_not_kill_the_run(self):
    def _boom(_line):
      raise RuntimeError("broken renderer")

    outcome = stream_subprocess(_py("print('x')"), on_line=_boom)
    self.assertEqual(outcome.returncode, 0)
    self.assertEqual(outcome.line_count, 1)


if __name__ == "__main__":
  unittest.main()
