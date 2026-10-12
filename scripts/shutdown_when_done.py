#!/usr/bin/env python3
"""Powers the VM off after a running campaign finishes.

An alternative to arming ``--shutdown`` on the campaign itself: start this in
a second pane at any point during the run, and kill it if you change your
mind. The campaign is never touched.

The shutdown policy is shared with the in-process path
(:func:`src.orchestrator.shutdown.should_shutdown`), so this powers off after
``COMPLETED`` or ``FAILED``, and stays up after ``STOPPED`` or ``PAUSED`` -
those mean somebody was at the keyboard.

Usage:
  # Watch whatever campaign is running, 5 minute cancellable countdown
  python3 scripts/shutdown_when_done.py --task npov --grace 300

  # Report the decision without acting on it
  python3 scripts/shutdown_when_done.py --task npov --dry-run

  # Watch a specific process
  python3 scripts/shutdown_when_done.py --pid 12345 --task npov

  # Also power off when the campaign dies without recording an outcome
  python3 scripts/shutdown_when_done.py --task npov --assume-failed
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orchestrator import shutdown_watcher  # pylint: disable=g-import-not-at-top
from src.orchestrator.cli import app  # pylint: disable=g-import-not-at-top


EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NOT_FOUND = 3


class _Console:
  """Plain stdout console with the surface the shutdown modules expect.

  The campaign's own ``UiConsole`` is deliberately not reused: this program
  may be the only thing still running on a machine that is about to halt, and
  its output has to land in a log file that nobody is watching live.
  """

  def blank(self) -> None:
    print()

  def info(self, message: str) -> None:
    print(message, flush=True)

  def warn(self, message: str) -> None:
    print(f"WARNING: {message}", flush=True)

  def error(self, message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr, flush=True)

  def hint(self, message: str) -> None:
    print(f"  {message}", flush=True)

  def success(self, message: str) -> None:
    print(message, flush=True)


def build_parser() -> argparse.ArgumentParser:
  """Builds the argument parser."""
  parser = argparse.ArgumentParser(
      prog="shutdown_when_done.py",
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
      "--pid", type=int, default=None,
      help=(
          "Campaign process to watch. Discovered automatically when omitted."
      ),
  )
  parser.add_argument(
      "--task", "-t", default="npov",
      help="Task whose campaign state file should be consulted.",
  )
  parser.add_argument(
      "--state-file", default=None,
      help="Explicit path to a campaign state JSON, overriding --task.",
  )
  parser.add_argument(
      "--grace", type=int, default=60,
      help=(
          "Seconds to count down before halting, cancellable with Ctrl-C."
          " 0 powers off immediately."
      ),
  )
  parser.add_argument(
      "--poll", type=float, default=shutdown_watcher.DEFAULT_POLL_SECONDS,
      help="Seconds between liveness checks on the campaign process.",
  )
  parser.add_argument(
      "--max-wait-hours", type=float,
      default=shutdown_watcher.DEFAULT_MAX_WAIT_HOURS,
      help=(
          "Give up watching after this long and leave the machine alone."
          " 0 waits forever."
      ),
  )
  parser.add_argument(
      "--assume-failed", action="store_true",
      help=(
          "Power off even when the campaign died without recording an"
          " outcome (state still IN_PROGRESS). Off by default: an unexplained"
          " death is usually worth logging in to look at."
      ),
  )
  parser.add_argument(
      "--decide-now", action="store_true",
      help=(
          "Decide from the recorded state without waiting for a campaign."
          " Required when nothing is running, because the state file on disk"
          " may describe a previous run."
      ),
  )
  parser.add_argument(
      "--dry-run", action="store_true",
      help="Report the decision without powering anything off.",
  )
  parser.add_argument(
      "--log-file", default=None,
      help="Mirror the watcher's output into this file.",
  )
  return parser


def resolve_pid(explicit, console):
  """Determines which process to watch.

  Args:
    explicit: The ``--pid`` value, or None.
    console: Console for user-visible notices.

  Returns:
    A pid, or None to decide immediately without waiting.
  """
  if explicit:
    if not shutdown_watcher.is_running(explicit):
      console.warn(f"Pid {explicit} is not running.")
      return None
    return explicit

  # The watcher's own command line mentions the campaign script it greps for,
  # so it has to exclude itself - and its parent shell, which may be running
  # this from a one-liner that also names the script.
  candidates = shutdown_watcher.find_campaign_pids(
      exclude_pids=(os.getpid(), os.getppid())
  )
  if not candidates:
    return None
  if len(candidates) > 1:
    console.warn(
        f"Several campaign processes are running ({candidates}); watching the"
        f" oldest, {candidates[0]}. Pass --pid to choose."
    )
  return candidates[0]


def resolve_state_file(args, console):
  """Locates the campaign state file.

  Args:
    args: Parsed arguments.
    console: Console for user-visible notices.

  Returns:
    A path, or None when nothing was found.
  """
  if args.state_file:
    if not os.path.exists(args.state_file):
      console.error(f"State file not found: {args.state_file}")
      return None
    return args.state_file
  path = app.find_latest_state(args.task)
  if not path:
    console.warn(f"No campaign state file found for task '{args.task}'.")
  return path


def main(argv=None) -> int:
  """Entry point.

  Args:
    argv: Argument vector, defaulting to ``sys.argv[1:]``.

  Returns:
    Process exit code.
  """
  args = build_parser().parse_args(argv)
  console = _Console()

  handlers = [logging.StreamHandler(sys.stderr)]
  if args.log_file:
    os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
    handlers.append(logging.FileHandler(args.log_file))
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s %(levelname)s %(message)s",
      handlers=handlers,
  )

  if args.grace < 0:
    console.error("--grace must be >= 0. Use 0 to power off immediately.")
    return EXIT_USAGE

  state_path = resolve_state_file(args, console)
  if args.state_file and state_path is None:
    return EXIT_NOT_FOUND

  pid = resolve_pid(args.pid, console)
  # Waiting for a process that does not exist takes no time, so "nothing is
  # running" would otherwise mean "decide immediately from whatever state file
  # happens to be on disk" - including a previous campaign's COMPLETED.
  complaint = shutdown_watcher.require_target(
      pid, state_path, decide_now=args.decide_now
  )
  if complaint:
    console.error(complaint)
    return EXIT_NOT_FOUND

  if args.dry_run:
    console.info("Dry run: the machine will not be powered off.")

  shutdown_watcher.watch_and_shutdown(
      pid=pid,
      state_path=state_path,
      grace_seconds=args.grace,
      poll_seconds=args.poll,
      max_wait_seconds=args.max_wait_hours * 3600.0,
      assume_failed=args.assume_failed,
      dry_run=args.dry_run,
      console=console,
  )
  return EXIT_OK


if __name__ == "__main__":
  sys.exit(main())
