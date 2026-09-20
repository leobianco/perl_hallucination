"""Powering the VM off from *outside* the campaign process.

``src.orchestrator.shutdown`` arms the switch from inside the run: the campaign
decides, at the end of its own ``finally``, whether to halt the machine. That
is the right default for a fire-and-forget launch, but it forces the decision
before the campaign starts, and a launch that was not armed cannot be armed
later without killing it.

This module inverts that. A second process watches the first, and powers the
machine off once there is genuinely nothing left to do. It can be started at
any point after the launch, it can be cancelled by killing it, and it never
touches the campaign.

What it deliberately does *not* do is re-decide policy. The question "does this
outcome warrant a shutdown?" is answered by :func:`shutdown.should_shutdown`,
exactly as it is for an in-process run, so the two paths cannot drift apart.
This module only answers the question the in-process hook never has to ask:
*has the campaign stopped running at all?*

The one genuinely new case is a campaign whose process is gone while its state
file still says ``IN_PROGRESS`` - an OOM kill, a ``kill -9``, a closed pane.
The default is to stay up. An unexplained death is precisely when somebody
wants to log in and look, and the cost is asymmetric in the same way the
in-process module already argues: an idle GPU is recoverable by typing, a
machine that vanished mid-investigation is not. ``--assume-failed`` opts into
the other behaviour.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

from src.orchestrator import shutdown
from src.orchestrator.state import CampaignState


logger = logging.getLogger(__name__)

#: Seconds between liveness checks. The watcher may sit here for twelve hours,
#: so the poll is cheap and the interval is generous; nothing downstream cares
#: about half a minute of latency before a power-off.
DEFAULT_POLL_SECONDS = 30.0

#: Hours after which the watcher gives up and leaves the machine alone. A
#: watcher that outlives its campaign - because the campaign was restarted
#: under a new pid, say - must not linger for a week waiting to power off a
#: machine somebody else is now using.
DEFAULT_MAX_WAIT_HOURS = 48.0

#: Substring identifying a campaign process in ``/proc/<pid>/cmdline``.
CAMPAIGN_PATTERN = "run_campaign.py"

#: Status a state file carries while the campaign is still working. Seeing it
#: *after* the process has gone means the campaign died without recording an
#: outcome.
IN_PROGRESS = "IN_PROGRESS"


class WatchConfig:
  """The subset of a campaign config that the shutdown policy reads.

  :func:`shutdown.should_shutdown` accepts anything with these attributes, so
  the watcher hands it this instead of a real
  :class:`~src.orchestrator.config.CampaignConfig`. The watcher has no
  campaign of its own and must not have to fabricate one.

  Attributes:
    shutdown_when_done: Always True - running this program *is* the arming.
    dry_run: When True the policy declines and explains itself, which is how
      ``--dry-run`` reports the decision without acting on it.
    shutdown_grace_seconds: Cancellable countdown before the machine halts.
  """

  def __init__(self, dry_run: bool = False, grace_seconds: int = 60):
    self.shutdown_when_done = True
    self.dry_run = dry_run
    self.shutdown_grace_seconds = grace_seconds


def is_running(pid: int) -> bool:
  """Reports whether a process exists.

  Signal 0 performs the permission and existence checks without delivering
  anything. ``PermissionError`` means the process exists and belongs to
  somebody else, which still counts as running.

  Args:
    pid: Process id to test.

  Returns:
    True when the process exists.
  """
  if pid <= 0:
    return False
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  return True


def find_campaign_pids(
    pattern: str = CAMPAIGN_PATTERN,
    proc_root: str = "/proc",
    exclude_pids: Optional[Sequence[int]] = None,
) -> List[int]:
  """Finds running campaign processes by scanning ``/proc``.

  Used when the caller did not pass ``--pid``. Matching on the command line
  rather than on a pid file keeps the watcher independent of the campaign,
  which is the entire point of running it separately.

  Args:
    pattern: Substring to look for in each process's command line.
    proc_root: Root of the proc filesystem; a parameter so tests can point it
      at a fixture directory.
    exclude_pids: Pids to ignore - at minimum the watcher itself, whose own
      command line mentions the campaign script it is watching for.

  Returns:
    Matching pids, ascending. Empty when nothing matches.
  """
  excluded = set(exclude_pids or ())
  found = []
  try:
    entries = os.listdir(proc_root)
  except OSError:
    return []
  for entry in entries:
    if not entry.isdigit():
      continue
    pid = int(entry)
    if pid in excluded:
      continue
    try:
      with open(
          os.path.join(proc_root, entry, "cmdline"), "rb"
      ) as handle:
        # Arguments are NUL separated; join with spaces so a pattern can span
        # two of them.
        cmdline = handle.read().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except OSError:
      # The process exited between listdir and open, or it belongs to another
      # user. Either way it is not something we can watch.
      continue
    if pattern in cmdline:
      found.append(pid)
  return sorted(found)


def wait_for_exit(
    pid: int,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    max_wait_seconds: float = DEFAULT_MAX_WAIT_HOURS * 3600.0,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    running: Callable[[int], bool] = is_running,
) -> bool:
  """Blocks until a process exits.

  ``os.waitpid`` is not available here: the watcher is a sibling of the
  campaign, not its parent, so it polls instead.

  Args:
    pid: Process to wait for.
    poll_seconds: Interval between liveness checks.
    max_wait_seconds: Upper bound on the wait. Non-positive means no bound.
    sleeper: Injectable sleep, so tests do not wait.
    clock: Injectable monotonic clock, for the same reason.
    running: Injectable liveness predicate.

  Returns:
    True when the process exited, False when the bound elapsed first.
  """
  deadline = None if max_wait_seconds <= 0 else clock() + max_wait_seconds
  while running(pid):
    if deadline is not None and clock() >= deadline:
      return False
    sleeper(poll_seconds)
  return True


def resolve_status(
    state_path: Optional[str], assume_failed: bool = False
) -> Tuple[str, str]:
  """Reads the campaign's recorded outcome from its state file.

  Args:
    state_path: Path to ``campaign_state.json``, or None when no state file
      was found.
    assume_failed: Treat a campaign that left ``IN_PROGRESS`` behind as
      ``FAILED``, which is what the in-process hook assumes when the engine
      raises. Off by default; see the module docstring.

  Returns:
    ``(status, explanation)``. The status is fed to
    :func:`shutdown.should_shutdown`; the explanation is for the log, because
    a shutdown that does not happen is as surprising as one that does.
  """
  if not state_path:
    return "", "no campaign state file was found"
  try:
    state = CampaignState.load(state_path)
  except Exception as error:  # pylint: disable=broad-except
    # An unreadable state file is not a reason to power off a machine.
    return "", f"the state file could not be read ({type(error).__name__})"

  status = str(getattr(state, "status", "") or "").strip().upper()
  if status == IN_PROGRESS:
    if assume_failed:
      return (
          "FAILED",
          "the process is gone while the state still says IN_PROGRESS, and "
          "--assume-failed was given",
      )
    return (
        IN_PROGRESS,
        "the process is gone but the state still says IN_PROGRESS, which "
        "means the campaign died without recording an outcome; pass "
        "--assume-failed to power off anyway",
    )
  return status, f"the state file records {status or 'no status'}"


def require_target(
    pid: Optional[int], state_path: Optional[str], decide_now: bool = False
) -> Optional[str]:
  """Checks that there is something safe to act on.

  The dangerous case this exists for: started with no campaign running, the
  watcher would read the *previous* campaign's state file, find ``COMPLETED``
  and halt the machine on the spot. Waiting for a process that does not exist
  takes no time at all, so "nothing to watch" must be an error rather than an
  immediate power-off.

  Args:
    pid: Process the watcher would wait for, or None if none was found.
    state_path: State file the decision would be read from, or None.
    decide_now: Caller explicitly asked to decide without waiting, which is
      the legitimate version of the same situation.

  Returns:
    An error message, or None when it is safe to proceed.
  """
  if pid:
    return None
  if not decide_now:
    return (
        "No running campaign found. Refusing to act on a state file that may "
        "describe a previous run - a stale COMPLETED would power this machine "
        "off immediately. Start the campaign first, pass --pid, or pass "
        "--decide-now if you really mean 'the campaign is over, decide from "
        "the recorded state'."
    )
  if not state_path:
    return "Nothing to watch and no state to read; refusing to power off."
  return None


def watch_and_shutdown(
    pid: Optional[int] = None,
    state_path: Optional[str] = None,
    grace_seconds: int = 60,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    max_wait_seconds: float = DEFAULT_MAX_WAIT_HOURS * 3600.0,
    assume_failed: bool = False,
    dry_run: bool = False,
    console: Optional[Any] = None,
    waiter: Optional[Callable[..., bool]] = None,
    runner: Optional[Callable[[Sequence[str]], Any]] = None,
    grace_waiter: Optional[Callable[..., bool]] = None,
) -> bool:
  """Waits for the campaign to finish, then applies the shutdown policy.

  Args:
    pid: Campaign process to watch. None skips the wait, which is how a
      caller asks "the campaign is already over, decide now".
    state_path: Path to the campaign state file.
    grace_seconds: Cancellable countdown before the machine halts.
    poll_seconds: Interval between liveness checks.
    max_wait_seconds: Upper bound on the wait.
    assume_failed: See :func:`resolve_status`.
    dry_run: Report the decision without acting on it.
    console: Optional UI console for user-visible notices.
    waiter: Injectable process wait, for tests.
    runner: Injectable command runner, forwarded to
      :func:`shutdown.maybe_shutdown`.
    grace_waiter: Injectable grace-period implementation, forwarded likewise.

  Returns:
    True when the shutdown command was issued.
  """

  def _say(message: str) -> None:
    logger.info(message)
    if console is not None:
      console.info(message)

  if pid:
    _say(f"Watching pid {pid}; will decide once it exits.")
    wait = waiter or wait_for_exit
    exited = wait(
        pid,
        poll_seconds=poll_seconds,
        max_wait_seconds=max_wait_seconds,
    )
    if not exited:
      message = (
          f"Gave up waiting for pid {pid} after "
          f"{max_wait_seconds / 3600.0:.1f}h; the VM stays up."
      )
      logger.warning(message)
      if console is not None:
        console.warn(message)
      return False
    _say(f"Pid {pid} exited.")

  status, explanation = resolve_status(state_path, assume_failed=assume_failed)
  _say(f"Campaign outcome: {explanation}.")

  # The decision itself belongs to the in-process module, so that arming the
  # shutdown from outside can never mean something different from arming it
  # from inside.
  return shutdown.maybe_shutdown(
      WatchConfig(dry_run=dry_run, grace_seconds=grace_seconds),
      status,
      console=console,
      runner=runner,
      waiter=grace_waiter,
  )
