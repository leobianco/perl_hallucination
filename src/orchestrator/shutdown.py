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
#: root. On a GCP VM the invoking user is normally a passwordless sudoer; when
#: they are not, the command fails and we say so instead of hanging.
SHUTDOWN_COMMAND: Tuple[str, ...] = ("sudo", "shutdown", "-h", "now")

#: Campaign outcomes after which the machine genuinely has no work left.
#:
#: ``FAILED`` is included on purpose. A campaign that dies at hour two is the
#: *most* expensive case to leave running - it burns the remaining ten hours
#: of GPU time producing nothing - and the state file plus the log are already
#: on disk by the time we get here, so nothing is lost by powering off.
TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED"})

#: Outcomes that mean a human intervened. ``STOPPED`` is reached by the stop
#: hotkey or Ctrl-C, both of which require somebody at the keyboard, and
#: pulling the machine out from under that person is never what they meant.
ATTENDED_STATUSES = frozenset({"STOPPED", "PAUSED"})

#: Seconds at which the countdown announces itself. Sparse on purpose: the
#: notice has to survive in a log file that nobody reads live, without
#: producing sixty lines of noise.
_ANNOUNCE_AT = (60, 30, 15, 10, 5, 3, 2, 1)


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
  announced = set()
  try:
    while True:
      remaining = deadline - clock()
      if remaining <= 0:
        return True
      mark = min((s for s in _ANNOUNCE_AT if s >= remaining), default=None)
      if mark is not None and mark not in announced:
        announced.add(mark)
        message = f"Powering off in {mark}s - press Ctrl-C to cancel."
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
    run(list(SHUTDOWN_COMMAND))
  except Exception as error:  # pylint: disable=broad-except
    failed = f"Could not power off the VM ({type(error).__name__}: {error})."
    logger.error(failed)
    if console is not None:
      console.error(failed)
      console.hint("Shut it down by hand: " + " ".join(SHUTDOWN_COMMAND))
    return False
  logger.warning("Shutdown command issued.")
  return True
