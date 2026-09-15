"""Structured event bus, control signals, and sweep log parsing for the CLI.

The original implementation passed raw log strings from the engine to the UI
and reverse-engineered progress with substring matching. This module replaces
that with a small, typed event stream so that every UI surface (live
dashboard, plain stream, tests) reacts to the *same* facts, while remaining
backwards compatible with the ``live_line_callback(str)`` contract used by the
stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from enum import Enum
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional


class EventType(str, Enum):
  """Types of events published on the campaign event bus."""

  CAMPAIGN_STARTED = "CAMPAIGN_STARTED"
  CAMPAIGN_FINISHED = "CAMPAIGN_FINISHED"
  STAGE_STARTED = "STAGE_STARTED"
  STAGE_SKIPPED = "STAGE_SKIPPED"
  STAGE_COMPLETED = "STAGE_COMPLETED"
  STAGE_FAILED = "STAGE_FAILED"
  TRIAL_STARTED = "TRIAL_STARTED"
  TRIAL_METRIC = "TRIAL_METRIC"
  TRIAL_FINISHED = "TRIAL_FINISHED"
  LOG = "LOG"
  NOTICE = "NOTICE"
  CONTROL = "CONTROL"


@dataclass
class CampaignEvent:
  """A single immutable fact about campaign progress."""

  type: EventType
  stage: Optional[str] = None
  message: str = ""
  payload: Dict[str, Any] = field(default_factory=dict)
  timestamp: float = field(default_factory=time.time)

  @property
  def clock(self) -> str:
    """Wall-clock ``HH:MM:SS`` timestamp of the event."""
    return datetime.datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")

  def to_dict(self) -> Dict[str, Any]:
    """Serializes the event (used by ``--json`` output and tests)."""
    return {
        "type": self.type.value,
        "stage": self.stage,
        "message": self.message,
        "payload": dict(self.payload),
        "timestamp": self.timestamp,
    }


class EventBus:
  """Thread-safe publish/subscribe hub for :class:`CampaignEvent` objects.

  The engine runs in a worker thread while the UI renders on the main thread,
  so both emission and subscription are mutex protected. Subscriber
  exceptions are swallowed on purpose: a broken UI widget must never kill a
  multi-hour training campaign.
  """

  def __init__(self, history_limit: int = 2000):
    self._lock = threading.RLock()
    self._subscribers: List[Callable[[CampaignEvent], None]] = []
    self._history: List[CampaignEvent] = []
    self._history_limit = max(1, int(history_limit))

  def subscribe(
      self, callback: Callable[[CampaignEvent], None], replay: bool = False
  ) -> Callable[[], None]:
    """Registers ``callback``; returns a function that unsubscribes it.

    Args:
      callback: Receives every future event.
      replay: When True, immediately replays the retained history.

    Returns:
      A zero-argument callable removing the subscription.
    """
    with self._lock:
      self._subscribers.append(callback)
      backlog = list(self._history) if replay else []
    for event in backlog:
      self._safe_call(callback, event)

    def _unsubscribe() -> None:
      with self._lock:
        if callback in self._subscribers:
          self._subscribers.remove(callback)

    return _unsubscribe

  def emit(self, event: CampaignEvent) -> CampaignEvent:
    """Publishes ``event`` to all subscribers and retains it in history."""
    with self._lock:
      self._history.append(event)
      if len(self._history) > self._history_limit:
        del self._history[: len(self._history) - self._history_limit]
      subscribers = list(self._subscribers)
    for callback in subscribers:
      self._safe_call(callback, event)
    return event

  def publish(
      self,
      event_type: EventType,
      message: str = "",
      stage: Optional[str] = None,
      **payload: Any,
  ) -> CampaignEvent:
    """Convenience wrapper building and emitting an event in one call."""
    return self.emit(
        CampaignEvent(
            type=event_type, stage=stage, message=message, payload=payload
        )
    )

  def log(self, message: str, stage: Optional[str] = None) -> CampaignEvent:
    """Publishes a plain log line."""
    return self.publish(EventType.LOG, message=message, stage=stage)

  @property
  def history(self) -> List[CampaignEvent]:
    """Returns a snapshot copy of retained events."""
    with self._lock:
      return list(self._history)

  def as_line_callback(
      self, stage: Optional[str] = None
  ) -> Callable[[str], None]:
    """Adapts the bus to the legacy ``live_line_callback(str)`` contract."""

    def _callback(line: str) -> None:
      self.log(line, stage=stage)

    return _callback

  @staticmethod
  def _safe_call(
      callback: Callable[[CampaignEvent], None], event: CampaignEvent
  ) -> None:
    try:
      callback(event)
    except Exception:  # pylint: disable=broad-except
      # A misbehaving renderer must not abort the campaign.
      pass


class ControlSignals:
  """User intent shared between the UI thread and the campaign engine.

  The previous implementation conflated *pause*, *stop sweep* and *advance*
  into a single ``stop_requested`` flag, which meant pressing ``[P]`` silently
  terminated the campaign. Here each intent is distinct:

    * ``pause``  - block before the next stage until resumed (campaign alive).
    * ``advance``- seal the running sweep, promote its current best, continue.
    * ``stop``   - seal the running sweep and end the campaign gracefully.
    * ``abort``  - same as stop, but do not run the reporting step.
  """

  def __init__(self):
    self._lock = threading.RLock()
    self._paused = threading.Event()
    self._resumed = threading.Event()
    self._resumed.set()
    self._stop = threading.Event()
    self._abort = threading.Event()
    self._advance = threading.Event()
    self._listeners: List[Callable[[str], None]] = []

  # --- Intent setters -------------------------------------------------
  def on_change(self, callback: Callable[[str], None]) -> None:
    """Registers a callback invoked with the name of each control action."""
    with self._lock:
      self._listeners.append(callback)

  def _notify(self, action: str) -> None:
    with self._lock:
      listeners = list(self._listeners)
    for listener in listeners:
      try:
        listener(action)
      except Exception:  # pylint: disable=broad-except
        pass

  def pause(self) -> None:
    """Requests a pause at the next safe checkpoint."""
    self._paused.set()
    self._resumed.clear()
    self._notify("pause")

  def resume(self) -> None:
    """Clears a pending pause."""
    self._paused.clear()
    self._resumed.set()
    self._notify("resume")

  def toggle_pause(self) -> bool:
    """Toggles the pause flag; returns True when now paused."""
    if self._paused.is_set():
      self.resume()
      return False
    self.pause()
    return True

  def request_advance(self) -> None:
    """Requests early promotion of the current sweep leader."""
    self._advance.set()
    self._notify("advance")

  def clear_advance(self) -> None:
    """Consumes the advance request (called by the engine after acting)."""
    self._advance.clear()

  def request_stop(self) -> None:
    """Requests a graceful stop of the whole campaign."""
    self._stop.set()
    self.resume()
    self._notify("stop")

  def request_abort(self) -> None:
    """Requests an immediate stop, skipping reporting."""
    self._abort.set()
    self._stop.set()
    self.resume()
    self._notify("abort")

  # --- Intent getters -------------------------------------------------
  @property
  def is_paused(self) -> bool:
    return self._paused.is_set()

  @property
  def advance_requested(self) -> bool:
    return self._advance.is_set()

  @property
  def stop_requested(self) -> bool:
    return self._stop.is_set()

  @property
  def abort_requested(self) -> bool:
    return self._abort.is_set()

  def should_interrupt_stage(self) -> bool:
    """Predicate handed to long-running subprocesses to cut them short."""
    return self._stop.is_set() or self._advance.is_set()

  def wait_while_paused(self, poll_seconds: float = 0.2) -> bool:
    """Blocks while paused. Returns False when a stop was requested."""
    while self._paused.is_set() and not self._stop.is_set():
      self._resumed.wait(poll_seconds)
    return not self._stop.is_set()

  def snapshot(self) -> Dict[str, bool]:
    """Returns the current flag values (handy for tests and the status bar)."""
    return {
        "paused": self.is_paused,
        "advance": self.advance_requested,
        "stop": self.stop_requested,
        "abort": self.abort_requested,
    }


@dataclass
class TrialRecord:
  """A single sweep trial as reconstructed from the agent's output."""

  index: int
  run_id: str
  params: Dict[str, Any] = field(default_factory=dict)
  metric: Optional[float] = None
  state: str = "running"
  started_at: float = field(default_factory=time.time)
  finished_at: Optional[float] = None
  #: True while ``run_id`` is a placeholder synthesized before W&B disclosed
  #: the real id. Such a record is adopted (not duplicated) as soon as the
  #: real id appears on a later line.
  provisional: bool = False

  @property
  def elapsed(self) -> float:
    """Seconds the trial has been running (or ran, once finished)."""
    end = self.finished_at if self.finished_at is not None else time.time()
    return max(0.0, end - self.started_at)


# ``wandb agent`` announces run boundaries in *two* different shapes and the
# parser has to understand both, because which one you get depends on the
# wandb version and on whether the terminal is a TTY:
#
#   1. The ``termlog`` shape, carrying the run id:
#        wandb: Agent Starting Run: k8jd92la with config:
#        wandb: Agent Finished Run: k8jd92la
#   2. The ``logging`` shape emitted by the ``wandb.wandb_agent`` logger,
#      which is what a piped, non-interactive agent actually prints:
#        2026-09-15 11:30:46,457 - wandb.wandb_agent - INFO - Agent starting
#            run with config:
#        2026-09-15 11:45:10,000 - wandb.wandb_agent - INFO - Cleaning up
#            finished run: k8jd92la
#
# Matching only shape (1) is what used to freeze the trial counter at 00/N:
# "Agent starting run with config:" was captured as a run named ``with`` and
# no line ever matched "agent finished run", so no trial ever left the
# ``running`` state.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_LOGGER_PREFIX_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\s-\s[\w.]+\s-\s\w+\s-\s"
)
_TERM_PREFIX_RE = re.compile(r"^wandb:[ ]?")

#: Shape (1): an explicit run id follows the colon.
_RUN_START_ID_RE = re.compile(
    r"agent\s+starting\s+run:\s*([\w\-]+)", re.IGNORECASE
)
#: Shape (2): the run id is not known yet on this line.
_RUN_START_ANON_RE = re.compile(r"agent\s+starting\s+run\b", re.IGNORECASE)
_RUN_FINISH_RE = re.compile(
    r"(?:agent\s+finished\s+run|cleaning\s+up\s+finished\s+run)\s*:?\s*([\w\-]*)",
    re.IGNORECASE,
)
_RUN_FAILED_RE = re.compile(
    r"run\s+([\w\-]+)\s+(?:failed|errored|crashed)", re.IGNORECASE
)
_PARAM_RE = re.compile(r"^\s*\t?\s*([A-Za-z_][\w./-]*)\s*[:=]\s*(.+?)\s*$")
_DRY_TRIAL_RE = re.compile(r"trial\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def strip_log_prefix(text: str) -> str:
  """Removes ANSI colors and the ``wandb:``/logging prefixes from a line.

  Both prefixes can be stacked, so stripping repeats until it is a no-op.

  Args:
    text: A raw line of agent output.

  Returns:
    The line without decoration, with meaningful leading tabs preserved.
  """
  cleaned = _ANSI_RE.sub("", str(text))
  previous = None
  while previous != cleaned:
    previous = cleaned
    cleaned = _LOGGER_PREFIX_RE.sub("", cleaned, count=1)
    cleaned = _TERM_PREFIX_RE.sub("", cleaned, count=1)
  return cleaned


#: Smallest number of path segments an alias may keep. A qualified metric
#: must never degrade to its bare last segment: ``eval/loss`` reduced to
#: ``loss`` matches the HF ``Trainer``'s *training* loss line
#: (``{'loss': 1.90, 'grad_norm': ..., 'epoch': ...}``), which is a different,
#: systematically smaller quantity than the ``eval/loss`` the sweep optimizes.
#: The leaderboard would then rank trials on the wrong number.
_MIN_ALIAS_SEGMENTS = 2


def _metric_aliases(metric_name: str) -> List[str]:
  """Returns the spellings a training script may use for ``metric_name``.

  ``eval/loss`` is logged as ``eval/loss`` by W&B and as ``eval_loss`` by the
  HF ``Trainer``. Long keys are also matched on a suffix, because TRL logs
  ``train/rewards/reward_fn/mean`` as ``rewards/reward_fn/mean`` - but never
  on a suffix short enough to collide with an unrelated metric.

  Args:
    metric_name: The configured metric key.

  Returns:
    Candidate keys, longest first so the most specific one wins.
  """
  target = (metric_name or "").strip()
  if not target:
    return []
  # ``/`` is the real separator whenever it is present. Treating ``_`` as one
  # too would shred a segment that legitimately contains it:
  # ``train/rewards/reward_fn/mean`` would become ``train/rewards/reward/fn/
  # mean``, which matches nothing at all.
  separator = "/" if "/" in target else "_"
  segments = [s for s in target.split(separator) if s]
  if not segments:
    return []

  aliases = set()
  # An unqualified metric (``accuracy``) has nothing to strip; keep it as is.
  shortest = min(_MIN_ALIAS_SEGMENTS, len(segments))
  for start in range(0, len(segments) - shortest + 1):
    suffix = segments[start:]
    aliases.add("/".join(suffix))
    aliases.add("_".join(suffix))
  return sorted(aliases, key=len, reverse=True)


class SweepProgressParser:
  """Reconstructs a live trial leaderboard from sweep agent stdout.

  The parser is deliberately forgiving: unknown lines are ignored, and any
  recognized ``metric_name`` occurrence updates the active trial. It is pure
  (no I/O), which makes it straightforward to unit test against recorded
  agent output.
  """

  def __init__(self, metric_name: str = "", max_trials: int = 0):
    self.metric_name = metric_name or ""
    self.max_trials = int(max_trials or 0)
    self.trials: List[TrialRecord] = []
    self._by_run: Dict[str, TrialRecord] = {}
    self._active: Optional[TrialRecord] = None
    #: Trial closed most recently. A piped child process is block buffered,
    #: so its final output (the ``wandb: Run summary:`` block, the last
    #: ``{'eval_loss': ...}`` line) is often flushed *after* the agent has
    #: already logged "Cleaning up finished run". Without somewhere to put
    #: those lines the trial stays unscored forever.
    self._last_finished: Optional[TrialRecord] = None
    self._reading_params = False
    self._metric_re = self._build_metric_re(self.metric_name)

  @staticmethod
  def _build_metric_re(metric_name: str) -> Optional[re.Pattern]:
    """Compiles a matcher for ``metric_name`` in any of its usual spellings.

    The separator is optional so that the W&B run-summary block
    (``eval/loss 0.285``, no colon) is picked up alongside the usual
    ``eval_loss: 0.285`` / ``'eval_loss': 0.285`` forms.

    Args:
      metric_name: The configured metric key.

    Returns:
      A compiled pattern, or None when no metric is configured.
    """
    aliases = _metric_aliases(metric_name)
    if not aliases:
      return None
    alternation = "|".join(re.escape(alias) for alias in aliases)
    return re.compile(
        r"(?:^|[^\w/])['\"]?(?:" + alternation + r")['\"]?(?![\w/])"
        r"\s*[:=]?\s*(" + _NUMBER + r")\b"
    )

  @property
  def completed_count(self) -> int:
    """Number of trials that reached a terminal state."""
    return sum(1 for trial in self.trials if trial.state != "running")

  @property
  def active_trial(self) -> Optional[TrialRecord]:
    """The trial currently executing, if any."""
    return self._active

  def best_trial(self, goal: str = "minimize") -> Optional[TrialRecord]:
    """Returns the leading trial according to ``goal``."""
    scored = [t for t in self.trials if t.metric is not None]
    if not scored:
      return None
    return (max if str(goal).lower().startswith("max") else min)(
        scored, key=lambda t: t.metric
    )

  def leaderboard(self, goal: str = "minimize", limit: int = 8) -> List[TrialRecord]:
    """Returns trials ordered by metric quality, unscored ones last."""
    scored = [t for t in self.trials if t.metric is not None]
    unscored = [t for t in self.trials if t.metric is None]
    scored.sort(
        key=lambda t: t.metric,
        reverse=str(goal).lower().startswith("max"),
    )
    ordered = scored + unscored
    return ordered[:limit] if limit else ordered

  def feed(self, line: str) -> Optional[TrialRecord]:
    """Consumes one output line; returns the affected trial, if any."""
    if not line:
      return None
    text = strip_log_prefix(line).rstrip()
    if not text:
      return None

    start_id = _RUN_START_ID_RE.search(text)
    if start_id:
      return self._start_trial(start_id.group(1))

    if _RUN_START_ANON_RE.search(text):
      return self._start_trial(None)

    finish_match = _RUN_FINISH_RE.search(text)
    if finish_match:
      return self._finish_trial(finish_match.group(1) or None)

    failed_match = _RUN_FAILED_RE.search(text)
    if failed_match:
      return self._finish_trial(failed_match.group(1), state="failed")

    dry_match = _DRY_TRIAL_RE.search(text)
    if dry_match and "trial" in text.lower():
      index, total = int(dry_match.group(1)), int(dry_match.group(2))
      if total and not self.max_trials:
        self.max_trials = total
      trial = self._start_trial(f"trial-{index}", index=index)
      self._finish_trial(trial.run_id)
      return trial

    # A metric can be logged by the training script long after the config
    # block ended, so it is checked before the (block-scoped) param parsing.
    metric_match = (
        self._metric_re.search(text) if self._metric_re is not None else None
    )
    if metric_match:
      target = self._active
      if target is None:
        # Late arrival: the child's buffered tail drained after the agent
        # announced the cleanup. Score the trial it belongs to, but never
        # overwrite a value that trial already reported - a number seen after
        # the boundary is only trustworthy as a first observation.
        target = self._last_finished
        if target is not None and target.metric is not None:
          target = None
      if target is not None:
        try:
          target.metric = float(metric_match.group(1))
          return target
        except ValueError:
          pass

    if self._active is None:
      return None

    param_match = _PARAM_RE.match(text)
    if param_match:
      key, raw_value = param_match.group(1), param_match.group(2)
      value = _coerce(raw_value)
      if isinstance(value, (int, float, str)) and self._reading_params:
        # Only the indented block right after "Starting Run" holds params.
        self._active.params[key] = value
        return self._active
    else:
      self._reading_params = False
    return None

  def _start_trial(
      self, run_id: Optional[str], index: Optional[int] = None
  ) -> TrialRecord:
    """Opens (or re-opens) the trial identified by ``run_id``.

    Args:
      run_id: The W&B run id, or None when the agent has not disclosed it yet.
      index: Explicit 1-based trial number (used by the dry-run path).

    Returns:
      The trial the subsequent lines belong to.
    """
    if run_id and run_id in self._by_run:
      self._active = self._by_run[run_id]
      self._reading_params = True
      return self._active

    # The logging and termlog shapes both announce the *same* trial. When the
    # real id finally shows up, rename the provisional record instead of
    # double counting it.
    if (
        run_id
        and self._active is not None
        and self._active.provisional
        and self._active.state == "running"
    ):
      trial = self._active
      self._by_run.pop(trial.run_id, None)
      trial.run_id = run_id
      trial.provisional = False
      self._by_run[run_id] = trial
      self._reading_params = True
      return trial

    if (
        run_id is None
        and self._active is not None
        and self._active.state == "running"
    ):
      # A second announcement for a trial that is already open.
      self._reading_params = True
      return self._active

    position = index if index is not None else len(self.trials) + 1
    trial = TrialRecord(
        index=position,
        run_id=run_id or f"run-{position}",
        provisional=run_id is None,
    )
    self.trials.append(trial)
    self._by_run[trial.run_id] = trial
    self._active = trial
    self._reading_params = True
    return trial

  def _finish_trial(
      self, run_id: Optional[str], state: str = "done"
  ) -> Optional[TrialRecord]:
    """Closes a trial, falling back to the active one for unknown ids.

    Args:
      run_id: The W&B run id reported on the closing line, if any.
      state: Terminal state to record.

    Returns:
      The closed trial, or None when there was nothing open.
    """
    trial = (self._by_run.get(run_id) if run_id else None) or self._active
    if trial is None:
      return None
    if run_id and trial.provisional:
      # The closing line is the first place the real id appeared.
      self._by_run.pop(trial.run_id, None)
      trial.run_id = run_id
      trial.provisional = False
      self._by_run[run_id] = trial
    trial.state = state
    trial.finished_at = time.time()
    self._reading_params = False
    self._last_finished = trial
    if self._active is trial:
      self._active = None
    return trial


def _coerce(raw: str) -> Any:
  """Best-effort conversion of a textual config value to a Python scalar."""
  text = str(raw).strip().strip(",")
  lowered = text.lower()
  if lowered in ("true", "false"):
    return lowered == "true"
  if lowered in ("none", "null"):
    return None
  try:
    if re.fullmatch(r"[-+]?\d+", text):
      return int(text)
    return float(text)
  except ValueError:
    return text
