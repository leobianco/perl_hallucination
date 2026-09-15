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

  @property
  def elapsed(self) -> float:
    """Seconds the trial has been running (or ran, once finished)."""
    end = self.finished_at if self.finished_at is not None else time.time()
    return max(0.0, end - self.started_at)


# ``wandb agent`` prints run boundaries in a stable, documented format.
_RUN_START_RE = re.compile(
    r"agent\s+starting\s+run:?\s*([\w\-]+)", re.IGNORECASE
)
_RUN_FINISH_RE = re.compile(
    r"agent\s+finished\s+run:?\s*([\w\-]+)", re.IGNORECASE
)
_PARAM_RE = re.compile(r"^\s*(?:wandb:)?\s*\t?\s*([A-Za-z_][\w./-]*)\s*[:=]\s*(.+?)\s*$")
_DRY_TRIAL_RE = re.compile(r"trial\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


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
    self._reading_params = False

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
    text = str(line).rstrip()

    start_match = _RUN_START_RE.search(text)
    if start_match:
      return self._start_trial(start_match.group(1))

    finish_match = _RUN_FINISH_RE.search(text)
    if finish_match:
      return self._finish_trial(finish_match.group(1))

    dry_match = _DRY_TRIAL_RE.search(text)
    if dry_match and "trial" in text.lower():
      index, total = int(dry_match.group(1)), int(dry_match.group(2))
      if total and not self.max_trials:
        self.max_trials = total
      trial = self._start_trial(f"trial-{index}", index=index)
      self._finish_trial(trial.run_id)
      return trial

    if self._active is None:
      return None

    param_match = _PARAM_RE.match(text)
    if param_match:
      key, raw_value = param_match.group(1), param_match.group(2)
      value = _coerce(raw_value)
      if self.metric_name and key in (
          self.metric_name,
          self.metric_name.split("/")[-1],
      ):
        if isinstance(value, (int, float)):
          self._active.metric = float(value)
          return self._active
      if isinstance(value, (int, float, str)) and self._reading_params:
        # Only the indented block right after "Starting Run" holds params.
        self._active.params[key] = value
        return self._active
    else:
      self._reading_params = False
    return None

  def _start_trial(self, run_id: str, index: Optional[int] = None) -> TrialRecord:
    if run_id in self._by_run:
      self._active = self._by_run[run_id]
      self._reading_params = True
      return self._active
    trial = TrialRecord(
        index=index if index is not None else len(self.trials) + 1,
        run_id=run_id,
    )
    self.trials.append(trial)
    self._by_run[run_id] = trial
    self._active = trial
    self._reading_params = True
    return trial

  def _finish_trial(self, run_id: str) -> Optional[TrialRecord]:
    trial = self._by_run.get(run_id) or self._active
    if trial is None:
      return None
    trial.state = "done"
    trial.finished_at = time.time()
    self._reading_params = False
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
