"""Watching the campaign tree and publishing snapshots as it changes.

This is what makes the published site feel live without a server: a long-lived
process that notices when the orchestrator writes a state file, rebuilds, and
pushes the result. It is campaign-agnostic on purpose - it watches a
*directory*, so it spans campaigns, survives their crashes, and needs no
cooperation from the orchestrator whatsoever.

Three rules govern when it publishes:

1. **A terminal transition publishes immediately.** When a campaign reaches
   ``COMPLETED``/``FAILED``/``STOPPED``/``ABORTED`` the debounce is bypassed.
   This is not a nicety: with ``shutdown_when_done`` armed, the VM powers off
   about a minute later, and the campaign's *result* is the one snapshot that
   must not be lost to a timer.
2. **Everything else is debounced.** Trial counters tick often and matter
   little; a Space is a git repository and each publish is a commit.
3. **An unchanged snapshot is never published.** The content fingerprint is
   compared before uploading, so an idle VM produces no commits at all.

Failures are logged and retried on the next change. A watcher that dies
because Hugging Face returned a 503 would be worse than no watcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import logging
import os
import signal
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.dashboard import build as build_mod
from src.dashboard import index as index_mod
from src.dashboard import publish as publish_mod

logger = logging.getLogger(__name__)

#: How often the campaign tree is re-read. Cheap: a few small JSON files.
DEFAULT_POLL_SECONDS = 5.0

#: Minimum spacing between two publishes of ordinary progress.
DEFAULT_DEBOUNCE_SECONDS = 60.0

#: Per-campaign fingerprint: status, last update, report mtime. Anything that
#: changes one of these is worth a new snapshot; nothing else is.
Signature = Dict[str, Tuple[str, str, float]]


@dataclass
class WatchConfig:
  """Everything the watcher needs to build and publish."""

  root: str = "."
  out_dir: str = "site"
  include_archived: bool = True
  backend: str = "hf_space"
  repo_id: Optional[str] = None
  destination: Optional[str] = None
  private: bool = True
  debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS
  poll_seconds: float = DEFAULT_POLL_SECONDS
  state_file: str = publish_mod.DEFAULT_STATE_FILE
  dry_run: bool = False


@dataclass
class TickOutcome:
  """What one poll decided and, if it acted, what happened."""

  changed: bool = False
  reason: str = ""
  published: Optional[publish_mod.PublishResult] = None
  campaigns: int = 0
  #: Campaigns that reached a terminal status during this tick.
  finished: List[str] = field(default_factory=list)


def _report_mtime(root: str, campaign: index_mod.CampaignSummary) -> float:
  """Returns the modification time of a campaign's report, or 0.0.

  Args:
    root: Repository root.
    campaign: The campaign.

  Returns:
    The mtime, or 0.0 when the campaign has no report on disk.
  """
  if not campaign.report_path:
    return 0.0
  try:
    return os.path.getmtime(os.path.join(root, campaign.report_path))
  except OSError:
    return 0.0


def signature_of(
    campaigns, root: str
) -> Signature:
  """Builds the fingerprint the watcher compares between polls.

  Args:
    campaigns: Parsed campaigns.
    root: Repository root, used to stat reports.

  Returns:
    A mapping from slug to ``(status, updated_at, report_mtime)``.
  """
  return {
      campaign.slug: (
          campaign.status,
          campaign.updated_at or "",
          _report_mtime(root, campaign),
      )
      for campaign in campaigns
  }


def newly_finished(previous: Signature, current: Signature) -> List[str]:
  """Returns campaigns that reached a terminal status since the last poll.

  A campaign that is *first seen* already terminal does not count: that is
  the watcher starting up next to an old campaign, not a campaign finishing.

  Args:
    previous: Fingerprint from the previous poll.
    current: Fingerprint from this poll.

  Returns:
    Slugs that transitioned into a terminal status.
  """
  finished = []
  for slug, (status, _, _) in current.items():
    if status not in index_mod.TERMINAL_STATUSES:
      continue
    before = previous.get(slug)
    if before is not None and before[0] not in index_mod.TERMINAL_STATUSES:
      finished.append(slug)
  return finished


class Watcher:
  """Polls the campaign tree and publishes snapshots when it changes."""

  def __init__(
      self,
      config: WatchConfig,
      clock: Callable[[], float] = time.monotonic,
      builder: Optional[Callable[..., build_mod.BuildResult]] = None,
      publisher: Optional[Callable[..., publish_mod.PublishResult]] = None,
  ):
    """Initializes the watcher.

    Args:
      config: What to watch and where to publish it.
      clock: Monotonic time source, injected by tests.
      builder: Build function, injected by tests.
      publisher: Publish function, injected by tests.
    """
    self.config = config
    self._clock = clock
    self._build = builder or build_mod.build_site
    self._publish = publisher or publish_mod.publish
    self._signature: Optional[Signature] = None
    self._last_publish_at: Optional[float] = None
    self._pending_since: Optional[float] = None
    self._stop = False

  def request_stop(self, *_: Any) -> None:
    """Asks the run loop to exit after the current poll."""
    self._stop = True

  def _should_publish(
      self, now: float, changed: bool, finished: List[str]
  ) -> Tuple[bool, str]:
    """Decides whether this poll warrants a publish.

    Args:
      now: Current monotonic time.
      changed: Whether the fingerprint moved.
      finished: Campaigns that just reached a terminal status.

    Returns:
      A ``(should_publish, reason)`` tuple.
    """
    if self._signature is None:
      return changed, "first snapshot"
    if finished:
      return True, f"{len(finished)} campaign(s) finished"
    if not changed and self._pending_since is None:
      return False, ""
    if self._last_publish_at is None:
      return True, "first snapshot"
    waited = now - self._last_publish_at
    if waited >= self.config.debounce_seconds:
      return True, f"progress, {waited:.0f}s since last publish"
    return False, "debounced"

  def tick(self) -> TickOutcome:
    """Runs one poll: detect changes, then publish if warranted.

    Returns:
      What this poll observed and did.
    """
    now = self._clock()
    campaigns = index_mod.build_index(
        self.config.root, include_archived=self.config.include_archived
    )
    current = signature_of(campaigns, self.config.root)
    previous = self._signature if self._signature is not None else {}
    changed = current != previous
    finished = newly_finished(previous, current)

    if changed and self._pending_since is None:
      self._pending_since = now

    should, reason = self._should_publish(now, changed, finished)
    outcome = TickOutcome(
        changed=changed,
        reason=reason,
        campaigns=len(campaigns),
        finished=finished,
    )
    self._signature = current

    if not should:
      return outcome

    result = self._build(
        root=self.config.root,
        out_dir=self.config.out_dir,
        include_archived=self.config.include_archived,
        campaigns=campaigns,
    )
    if self.config.dry_run:
      logger.info(
          "[DRY-RUN] would publish %s (%s)", result.content_hash[:12], reason
      )
      self._last_publish_at = now
      self._pending_since = None
      return outcome

    published = self._publish(
        site_dir=self.config.out_dir,
        content_hash=result.content_hash,
        backend=self.config.backend,
        repo_id=self.config.repo_id,
        destination=self.config.destination,
        private=self.config.private,
        state_file=self.config.state_file,
    )
    outcome.published = published
    if published.error:
      # Keep `_pending_since` set so the next poll retries rather than
      # waiting for another change that may never come.
      logger.warning("Publish failed: %s", published.error)
      return outcome

    self._last_publish_at = now
    self._pending_since = None
    if published.published:
      logger.info(
          "Published %s campaign(s) to %s (%s)",
          len(campaigns),
          published.url or published.target,
          reason,
      )
    else:
      logger.debug("Nothing to publish: %s", published.reason)
    return outcome

  def run(self, max_ticks: Optional[int] = None) -> int:
    """Polls until stopped.

    Args:
      max_ticks: Stop after this many polls; ``None`` runs forever. Used by
        tests and by ``--once``.

    Returns:
      The number of polls performed.
    """
    ticks = 0
    while not self._stop and (max_ticks is None or ticks < max_ticks):
      try:
        self.tick()
      except Exception as error:  # pylint: disable=broad-except
        # A malformed state file, a full disk, a transient Hub error: log it
        # and keep watching. The watcher outliving its problems is the whole
        # point of running it in a tmux window for days.
        logger.warning("Watch tick failed: %s: %s", type(error).__name__, error)
      ticks += 1
      if self._stop or (max_ticks is not None and ticks >= max_ticks):
        break
      time.sleep(self.config.poll_seconds)
    return ticks


def install_signal_handlers(watcher: Watcher) -> None:
  """Makes SIGINT and SIGTERM stop the watcher cleanly.

  Args:
    watcher: The watcher to stop.
  """
  for received in (signal.SIGINT, signal.SIGTERM):
    try:
      signal.signal(received, watcher.request_stop)
    except (ValueError, OSError):
      # Not the main thread, or a platform without the signal. Ctrl-C still
      # raises KeyboardInterrupt, which the CLI handles.
      pass


def describe_start(config: WatchConfig) -> str:
  """Returns the banner printed when the watcher starts.

  Args:
    config: The watcher's configuration.

  Returns:
    A human-readable multi-line summary.
  """
  target = config.repo_id if config.backend == "hf_space" else config.destination
  started = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
  lines = [
      f"Watching {os.path.abspath(config.root)} for campaign changes.",
      f"  Publishing to {config.backend}:{target}"
      + ("  [DRY-RUN]" if config.dry_run else ""),
      f"  Poll {config.poll_seconds:g}s, debounce {config.debounce_seconds:g}s,"
      " terminal transitions publish immediately.",
      f"  Started {started}. Ctrl-C to stop.",
  ]
  return "\n".join(lines)
