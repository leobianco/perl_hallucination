"""Powering the VM off once an unattended campaign has nothing left to do.

``scripts/perl.sh`` ends with a ``SHUTDOWN`` switch for exactly one reason: a
single training run that finishes at 02:00 leaves an A100 idling until
somebody wakes up. A campaign makes that worse - it is the longest job in the
repo, so it is the one most likely to end while nobody is watching.

The decision is deliberately split from the act. :func:`should_shutdown` is
pure and answers *whether* we power off and *why*; :func:`maybe_shutdown`
performs it behind a cancellable grace period. That split is what makes the
policy testable without a test suite that can halt the developer's machine.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Any, Callable, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)

#: How the VM is halted. ``shutdown -h now`` rather than ``poweroff`` to match
#: ``scripts/perl.sh``, and via ``sudo`` because the campaign does not run as
#: root.
#:
#: ``-n`` is what keeps the promise made below about never hanging. Without it
#: a machine whose sudoers entry wants a password turns the last line of a
#: twelve-hour campaign into an invisible ``[sudo] password for ...`` prompt on
#: an inherited stdin that nobody is watching - the process sits there forever
#: and the VM stays up anyway. With it, sudo fails immediately and we say so.
SHUTDOWN_COMMAND: Tuple[str, ...] = ("sudo", "-n", "shutdown", "-h", "now")

#: Campaign outcomes after which the machine genuinely has no work left.
#:
#: ``FAILED`` is included on purpose. A campaign that dies at hour two is the
#: *most* expensive case to leave running - it burns the remaining ten hours
#: of GPU time producing nothing - and the state file plus the log are already
#: on disk by the time we get here, so nothing is lost by powering off.
TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED"})

#: Outcomes that mean a human intervened. ``STOPPED`` is reached by the stop
#: hotkey or Ctrl-C and ``ABORTED`` by ``[x]``; both require somebody at the
#: keyboard, and pulling the machine out from under that person is never what
#: they meant.
#:
#: ``ABORTED`` exists precisely for this set. An abort kills the work in
#: flight, so the stage it interrupted raises on its way down; recording that
#: as ``FAILED`` made a deliberate keystroke indistinguishable from an OOM,
#: and this policy would then power off a machine whose user had pressed a key
#: three seconds earlier.
ATTENDED_STATUSES = frozenset({"STOPPED", "PAUSED", "ABORTED"})

#: Seconds at which a countdown of a minute or less announces itself. Sparse
#: on purpose: the notice has to survive in a log file that nobody reads live,
#: without producing sixty lines of noise.
_ANNOUNCE_AT = (60, 30, 15, 10, 5, 3, 2, 1)

#: Spacing of the announcements above a minute.
#:
#: Without these the schedule above was the *whole* schedule, so a
#: ``--grace 300`` printed nothing at all for its first four minutes: the loop
#: only announces a mark that is still ahead of it, and nothing in
#: ``_ANNOUNCE_AT`` is ahead of 300s. Four minutes of silence from a program
#: whose only job is to halt the machine reads exactly like a hang, and the
#: one operator who saw it (correctly) killed it. Every grace period now opens
#: with an announcement and never goes quiet for longer than this.
_COARSE_INTERVAL = 60



def should_shutdown(config: Any, status: str) -> Tuple[bool, str]:
  """Decides whether the campaign's outcome warrants powering the VM off.

  Args:
    config: The campaign configuration; read for ``shutdown_when_done`` and
      ``dry_run``.
    status: The campaign's terminal status, e.g. ``COMPLETED``.

  Returns:
    ``(shutdown, reason)``. ``reason`` explains the decision either way and
    is meant to be shown to the user, because a shutdown that does not
    happen is just as surprising as one that does.
  """
  if not getattr(config, "shutdown_when_done", False):
    return False, "shutdown_when_done is off"

  # A rehearsal must never have a side effect the rehearsal cannot undo.
  if getattr(config, "dry_run", False):
    return False, "this is a dry run"

  normalized = (status or "").strip().upper()
  if normalized in ATTENDED_STATUSES:
    return False, (
        f"the campaign was stopped by hand ({normalized}), so somebody is "
        "at the keyboard"
    )
  if normalized not in TERMINAL_STATUSES:
    # Detached runs, or any future status this module has not been taught
    # about. Refusing here costs idle GPU time; guessing costs a live
    # session, and only one of those is recoverable by typing.
    return False, (
        f"the campaign ended in an unrecognized state ({normalized or 'unknown'})"
    )
  return True, f"the campaign finished with status {normalized}"


def _announcement_marks(seconds: float) -> Tuple[int, ...]:
  """Builds the countdown schedule for a grace period of ``seconds``.

  The full period is always a mark, so the countdown opens by stating how
  long it is going to be; the rest is one mark a minute down to the last
  minute, then the fine-grained tail. See ``_COARSE_INTERVAL`` for the
  incident this shape exists to prevent.

  Args:
    seconds: Length of the grace period.

  Returns:
    The marks, ascending. Only marks the period is long enough to reach are
    included, so a short countdown behaves exactly as it always did.
  """
  total = int(seconds)
  marks = {mark for mark in _ANNOUNCE_AT if mark <= total}
  marks.update(range(_COARSE_INTERVAL, total + 1, _COARSE_INTERVAL))
  marks.add(total)
  return tuple(sorted(marks))


def _format_countdown(mark: int) -> str:
  """Renders a mark the way an operator reads it: minutes once it is minutes."""
  if mark >= 120 and mark % 60 == 0:
    return f"{mark // 60} min"
  return f"{mark}s"


def _wait_out_grace_period(
    seconds: float,
    console: Optional[Any] = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
  """Counts down, giving anyone present a chance to abort with Ctrl-C.

  Args:
    seconds: Length of the grace period. Non-positive skips the wait.
    console: Optional UI console for the countdown notices.
    sleeper: Injectable sleep, so tests do not actually wait.
    clock: Injectable monotonic clock, for the same reason.

  Returns:
    True when the period elapsed, False when the user interrupted it.
  """
  if seconds <= 0:
    return True
  deadline = clock() + seconds
  marks = _announcement_marks(seconds)
  announced = set()
  try:
    while True:
      remaining = deadline - clock()
      if remaining <= 0:
        return True
      mark = min((s for s in marks if s >= remaining), default=None)
      if mark is not None and mark not in announced:
        announced.add(mark)
        message = (
            f"Powering off in {_format_countdown(mark)} - press Ctrl-C to"
            " cancel."
        )
        logger.warning(message)
        if console is not None:
          console.warn(message)
      sleeper(min(1.0, remaining))
  except KeyboardInterrupt:
    return False



def maybe_shutdown(
    config: Any,
    status: str,
    console: Optional[Any] = None,
    runner: Optional[Callable[[Sequence[str]], Any]] = None,
    waiter: Optional[Callable[..., bool]] = None,
) -> bool:
  """Powers the VM off if ``config`` asked for it and ``status`` warrants it.

  Never raises. This runs after the campaign has already written its state
  file and its report, so a missing ``sudo`` or a read-only ``shutdown``
  binary must degrade into a warning - turning a successful twelve-hour
  campaign into a traceback would be a far worse outcome than a VM that
  stays up.

  Args:
    config: The campaign configuration.
    status: The campaign's terminal status.
    console: Optional UI console for user-visible notices.
    runner: Injectable command runner, defaulting to ``subprocess.run``.
    waiter: Injectable grace-period implementation, for tests.

  Returns:
    True when the shutdown command was issued successfully.
  """
  wanted, reason = should_shutdown(config, status)
  if not wanted:
    if getattr(config, "shutdown_when_done", False):
      # Only worth saying when the user armed the switch and it did not
      # fire; otherwise this is noise on every single campaign.
      note = f"Not powering off: {reason}."
      logger.info(note)
      if console is not None:
        console.info(note)
    return False

  grace = max(0, int(getattr(config, "shutdown_grace_seconds", 60) or 0))
  opening = f"Shutdown requested: {reason}."
  logger.warning(opening)
  if console is not None:
    console.blank()
    console.warn(opening)

  wait = waiter or _wait_out_grace_period
  if not wait(grace, console):
    cancelled = "Shutdown cancelled; the VM stays up."
    logger.warning(cancelled)
    if console is not None:
      console.info(cancelled)
    return False

  run = runner or subprocess.run
  try:
    outcome = run(list(SHUTDOWN_COMMAND))
  except Exception as error:  # pylint: disable=broad-except
    failed = f"Could not power off the VM ({type(error).__name__}: {error})."
    logger.error(failed)
    if console is not None:
      console.error(failed)
      console.hint("Shut it down by hand: " + " ".join(SHUTDOWN_COMMAND))
    return False

  # Only exceptions used to be caught, so a `sudo` that exited 1 - not a
  # sudoer, or a sudoers entry that wants a password - was still reported as
  # "Shutdown command issued." and the VM quietly stayed up. An injected
  # runner that reports no code at all is taken at its word; that is the test
  # doubles' contract, not a real runner's.
  code = getattr(outcome, "returncode", 0)
  if isinstance(code, int) and code != 0:
    failed = (
        f"Could not power off the VM: {' '.join(SHUTDOWN_COMMAND)} exited"
        f" {code}."
    )
    logger.error(failed)
    if console is not None:
      console.error(failed)
      console.hint(
          "This needs passwordless sudo; `sudo -n true` says whether you"
          " have it."
      )
    return False
  logger.warning("Shutdown command issued.")
  return True
