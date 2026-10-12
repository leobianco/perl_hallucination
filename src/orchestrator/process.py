"""Robust subprocess streaming used by every long-running orchestrator stage.

Why this module exists
----------------------
The naive pattern ``while proc.poll() is None: proc.stdout.readline()`` blocks
inside ``readline`` until the child emits a line. During an unattended
overnight campaign that has two nasty consequences:

  * a silently hung trial (GPU deadlock, stalled HF download) is never caught
    by the stage timeout, because the timeout is only checked *between* lines;
  * a stop/advance hotkey is only honored after the next line of output.

Here the reading happens on a daemon thread feeding a queue, so the control
loop ticks on a fixed cadence regardless of how chatty the child is.

The child also gets ``stdin=DEVNULL``: the dashboard puts the real terminal in
cbreak mode to capture hotkeys, and a child inheriting that TTY would either
steal keystrokes or block forever on an interactive prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import queue
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Grace period granted to a terminated child before it is killed outright.
TERMINATE_GRACE_SECONDS = 15.0


@dataclass
class ProcessOutcome:
  """Result of a streamed subprocess execution."""

  returncode: int
  timed_out: bool = False
  interrupted: bool = False
  line_count: int = 0
  #: How many times the child went quiet past ``stall_warning_s``.
  stall_warnings: int = 0

  @property
  def ok(self) -> bool:
    """True when the process exited cleanly and was not cut short."""
    return self.returncode == 0 and not self.timed_out and not self.interrupted

  @property
  def cut_short(self) -> bool:
    """True when we deliberately terminated the child."""
    return self.timed_out or self.interrupted


def stream_subprocess(
    cmd: Sequence[str],
    on_line: Optional[Callable[[str], None]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    timeout_s: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    poll_interval: float = 0.1,
    stall_warning_s: Optional[float] = None,
) -> ProcessOutcome:
  """Runs ``cmd``, streaming merged stdout/stderr line by line.

  Args:
    cmd: Argument vector to execute.
    on_line: Callback receiving each output line (without trailing newline).
    stop_requested: Predicate polled on every tick; when it returns True the
      child is terminated and ``interrupted`` is set on the outcome.
    timeout_s: Wall-clock budget. When exceeded the child is terminated and
      ``timed_out`` is set on the outcome.
    env: Environment for the child (defaults to the current environment).
    cwd: Working directory for the child.
    poll_interval: Seconds between control-loop ticks.
    stall_warning_s: When set, a warning line is emitted through ``on_line``
      each time the child stays silent for this long. Purely informational: the
      process is left running.

  Returns:
    A :class:`ProcessOutcome` describing how the process ended.
  """
  logger.info("Executing: %s", " ".join(cmd))
  process = subprocess.Popen(
      list(cmd),
      stdin=subprocess.DEVNULL,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      bufsize=1,
      universal_newlines=True,
      env=env if env is not None else os.environ.copy(),
      cwd=cwd,
  )

  lines: "queue.Queue[Optional[str]]" = queue.Queue()

  def _reader() -> None:
    try:
      if process.stdout is not None:
        for raw in iter(process.stdout.readline, ""):
          lines.put(raw.rstrip("\n"))
    except Exception:  # pylint: disable=broad-except
      # A closed pipe during termination is expected; never crash the reader.
      pass
    finally:
      lines.put(None)

  reader = threading.Thread(
      target=_reader, name="orchestrator-proc-reader", daemon=True
  )
  reader.start()

  started = time.time()
  outcome = ProcessOutcome(returncode=0)
  reader_done = False
  last_line_at = started
  stall_warnings = 0

  def _drain(block_for: float) -> None:
    nonlocal reader_done, last_line_at
    deadline = time.time() + max(0.0, block_for)
    first = True
    while True:
      try:
        remaining = deadline - time.time()
        if first and remaining > 0:
          item = lines.get(timeout=remaining)
        else:
          item = lines.get_nowait()
      except queue.Empty:
        return
      first = False
      if item is None:
        reader_done = True
        return
      outcome.line_count += 1
      last_line_at = time.time()
      if on_line:
        try:
          on_line(item)
        except Exception:  # pylint: disable=broad-except
          # A broken renderer must never kill a multi-hour campaign.
          pass

  try:
    while True:
      _drain(poll_interval)

      if reader_done and process.poll() is not None:
        break

      if stop_requested is not None and stop_requested():
        logger.info("Stop requested: terminating child process.")
        outcome.interrupted = True
        _terminate(process)
        break

      if timeout_s is not None and (time.time() - started) > timeout_s:
        logger.warning(
            "Process exceeded its %.0fs budget; terminating.", timeout_s
        )
        outcome.timed_out = True
        _terminate(process)
        break

      if stall_warning_s:
        silent_for = time.time() - last_line_at
        if silent_for > stall_warning_s * (stall_warnings + 1):
          stall_warnings += 1
          outcome.stall_warnings = stall_warnings
          message = (
              f"[STALL] No output for {silent_for / 60.0:.0f} min from "
              f"'{cmd[0]}'. Still waiting; the process has not been touched."
          )
          logger.warning(message)
          if on_line:
            try:
              on_line(message)
            except Exception:  # pylint: disable=broad-except
              pass

    # Drain whatever is still buffered after the loop exits.
    while not reader_done:
      before = outcome.line_count
      _drain(0.2)
      if outcome.line_count == before and process.poll() is not None:
        break
  finally:
    if process.poll() is None:
      _terminate(process)
    try:
      if process.stdout is not None:
        process.stdout.close()
    except Exception:  # pylint: disable=broad-except
      pass
    reader.join(timeout=1.0)

  outcome.returncode = process.poll() if process.poll() is not None else 0
  return outcome


def _terminate(process: subprocess.Popen) -> None:
  """Terminates a child, escalating to SIGKILL if it ignores SIGTERM."""
  try:
    process.terminate()
  except Exception:  # pylint: disable=broad-except
    return
  try:
    process.wait(timeout=TERMINATE_GRACE_SECONDS)
  except subprocess.TimeoutExpired:
    logger.warning("Child ignored SIGTERM; sending SIGKILL.")
    try:
      process.kill()
      process.wait(timeout=5)
    except Exception:  # pylint: disable=broad-except
      pass


def collect_lines(outcome_lines: List[str]) -> Callable[[str], None]:
  """Returns an ``on_line`` callback appending to ``outcome_lines``."""

  def _append(line: str) -> None:
    outcome_lines.append(line)

  return _append
