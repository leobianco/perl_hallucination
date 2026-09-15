"""Live "Mission Control" dashboard for tmux/SSH sessions.

Design notes
------------
The dashboard deliberately uses ``rich.live.Live`` with ``screen=False``
instead of a full-screen alternate-buffer TUI. On a GCP VM driven through
tmux this matters a lot:

  * Log lines scroll into the normal tmux scrollback, so ``Ctrl-b [`` and
    mouse selection keep working; a full-screen app would swallow them.
  * Re-attaching a tmux session or resizing the pane simply re-flows the
    pinned block instead of corrupting the layout.
  * If the campaign is backgrounded with ``nohup``/piped to a file, the exact
    same code path emits clean, timestamped plain text.

The visible result is a docker-compose style UI: a pinned status block at the
bottom (header, DAG, live sweep leaderboard, hotkeys) with logs streaming
above it.
"""

from __future__ import annotations

import collections
import datetime
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from src.orchestrator.cli import renderables
from src.orchestrator.cli import theme as theme_mod
from src.orchestrator.cli.console import UiConsole
from src.orchestrator.cli.events import (
    CampaignEvent,
    ControlSignals,
    EventBus,
    EventType,
    SweepProgressParser,
)
from src.orchestrator.cli.theme import Theme

#: Log verbosity levels cycled with the ``l`` hotkey.
LOG_ALL, LOG_MILESTONES, LOG_OFF = 0, 1, 2
LOG_LEVEL_NAMES = {LOG_ALL: "all", LOG_MILESTONES: "milestones", LOG_OFF: "off"}

#: Substrings that always deserve to be shown, even in milestone mode.
MILESTONE_HINTS = (
    "registered",
    "best ",
    "pushed",
    "published",
    "starting",
    "error",
    "failed",
    "warning",
    "traceback",
    "dry-run",
)


class DashboardModel:
  """UI state derived from the campaign event stream.

  Pure and side-effect free (no printing, no terminal access) so the whole
  dashboard behaviour can be unit tested by feeding synthetic events.
  """

  def __init__(
      self,
      config: Any,
      state: Any,
      controls: Optional[ControlSignals] = None,
      log_capacity: int = 400,
  ):
    self.config = config
    self.state = state
    self.controls = controls or ControlSignals()
    self.started_at = time.time()
    self.finished = False
    self.final_status: Optional[str] = None
    self.current_stage: Optional[str] = None
    self.log_level = LOG_ALL
    self.show_help = False
    self.log_window = 0  # Extra rows requested by the user with +/-.
    self.notices: Deque[str] = collections.deque(maxlen=5)

    self._lock = threading.RLock()
    self._logs: Deque[Tuple[float, Optional[str], str]] = collections.deque(
        maxlen=log_capacity
    )
    self._pending: Deque[Tuple[float, Optional[str], str]] = collections.deque()
    self._parsers: Dict[str, SweepProgressParser] = {}
    self._live: Dict[str, Dict[str, Any]] = {}
    #: Trials each stage had already completed before this process started,
    #: as reported by the engine. The parsers above count only the current
    #: agent, so every number they produce has to be offset by this.
    self._baselines: Dict[str, int] = {}

  # --- Event ingestion ------------------------------------------------
  def on_event(self, event: CampaignEvent) -> None:
    """Folds a campaign event into the UI state."""
    with self._lock:
      if event.type == EventType.CAMPAIGN_STARTED:
        self.started_at = event.timestamp
      elif event.type == EventType.STAGE_STARTED:
        stage = event.stage or ""
        self.current_stage = stage
        parser = SweepProgressParser(
            metric_name=str(event.payload.get("metric") or ""),
            max_trials=int(event.payload.get("max_runs") or 0),
        )
        self._parsers[stage] = parser
        # This parser sees only the agent *this* process launches. On a
        # resume the sweep already has trials, and the engine says how many.
        self._baselines[stage] = int(event.payload.get("trials_baseline") or 0)
        self._live.setdefault(stage, {})["status"] = "RUNNING"
      elif event.type == EventType.TRIAL_FINISHED:
        stage = event.stage or self.current_stage or ""
        live = self._live.setdefault(stage, {})
        index = event.payload.get("index")
        if index is None:
          live["trials_done"] = live.get("trials_done", 0) + 1
        else:
          live["trials_done"] = self._baselines.get(stage, 0) + int(index)
        metric = event.payload.get("metric")
        if metric is not None:
          self._record_metric(stage, float(metric))
      elif event.type == EventType.TRIAL_METRIC:
        metric = event.payload.get("metric")
        if metric is not None:
          self._record_metric(event.stage or self.current_stage or "", float(metric))
      elif event.type in (EventType.STAGE_COMPLETED, EventType.STAGE_SKIPPED):
        stage = event.stage or ""
        live = self._live.setdefault(stage, {})
        live["status"] = (
            "COMPLETED" if event.type == EventType.STAGE_COMPLETED else "SKIPPED"
        )
        if event.payload.get("metric") is not None:
          live["metric_value"] = float(event.payload["metric"])
        if event.payload.get("model_repo_id"):
          live["model_repo_id"] = event.payload["model_repo_id"]
        if self.current_stage == stage:
          self.current_stage = None
      elif event.type == EventType.STAGE_FAILED:
        stage = event.stage or ""
        live = self._live.setdefault(stage, {})
        live["status"] = "FAILED"
        live["error"] = event.message or event.payload.get("error")
      elif event.type == EventType.CAMPAIGN_FINISHED:
        self.finished = True
        self.final_status = event.payload.get("status") or event.message
      elif event.type == EventType.NOTICE:
        self.notices.append(event.message)

      if event.message:
        self._ingest_log(event)

  def _ingest_log(self, event: CampaignEvent) -> None:
    stage = event.stage or self.current_stage
    parser = self._parsers.get(stage or "")
    if parser is not None and event.type == EventType.LOG:
      parser.feed(event.message)
      live = self._live.setdefault(stage or "", {})
      # `live` overrides the persisted counter in build_stage_views, so a
      # bare in-process count here would *hide* the recovered trials rather
      # than add to them.
      done = self._baselines.get(stage or "", 0) + parser.completed_count
      if done:
        live["trials_done"] = done
      if parser.max_trials:
        live["trials_total"] = parser.max_trials
      best = parser.best_trial(self._goal(stage))
      if best is not None and best.metric is not None:
        self._record_metric(stage or "", best.metric, replace=True)
    entry = (event.timestamp, stage, event.message)
    self._logs.append(entry)
    if self._should_display(event):
      self._pending.append(entry)

  def _should_display(self, event: CampaignEvent) -> bool:
    if self.log_level == LOG_OFF:
      return event.type in (EventType.STAGE_FAILED, EventType.NOTICE)
    if self.log_level == LOG_ALL:
      return True
    if event.type != EventType.LOG:
      return True
    lowered = event.message.lower()
    return any(hint in lowered for hint in MILESTONE_HINTS)

  def _record_metric(self, stage: str, metric: float, replace: bool = False) -> None:
    live = self._live.setdefault(stage, {})
    live["metric_value"] = metric
    history: List[float] = live.setdefault("history", [])
    if replace and history and history[-1] == metric:
      return
    history.append(metric)
    if len(history) > 60:
      del history[: len(history) - 60]

  def _goal(self, stage: Optional[str]) -> str:
    stage_cfg = getattr(self.config, stage or "", None)
    return str(getattr(stage_cfg, "goal", "minimize") or "minimize")

  # --- Queries used by the renderer -----------------------------------
  def drain_pending_logs(self, limit: int = 200) -> List[Tuple[float, Optional[str], str]]:
    """Pops buffered log lines that should be printed above the live block."""
    with self._lock:
      out = []
      while self._pending and len(out) < limit:
        out.append(self._pending.popleft())
      return out

  def recent_logs(self, limit: int = 10) -> List[Tuple[float, Optional[str], str]]:
    """Returns the most recent log entries (newest last)."""
    with self._lock:
      return list(self._logs)[-limit:]

  def stage_views(self) -> List[renderables.StageView]:
    """Builds the DAG rows, merging persisted state with live progress."""
    with self._lock:
      live = {stage: dict(values) for stage, values in self._live.items()}
    return renderables.build_stage_views(self.config, self.state, live)

  def active_parser(self) -> Optional[SweepProgressParser]:
    """Returns the parser of the stage currently running, if any."""
    with self._lock:
      if not self.current_stage:
        return None
      return self._parsers.get(self.current_stage)

  @property
  def elapsed(self) -> float:
    """Seconds since the campaign started."""
    return max(0.0, time.time() - self.started_at)

  def cycle_log_level(self) -> str:
    """Advances the log verbosity and returns its new name."""
    self.log_level = (self.log_level + 1) % 3
    return LOG_LEVEL_NAMES[self.log_level]

  def progress_summary(self) -> str:
    """One-line textual summary, used by the plain renderer and tests."""
    views = self.stage_views()
    done = sum(1 for v in views if str(v.status).upper() in ("COMPLETED", "SKIPPED"))
    active = next((v for v in views if v.is_active), None)
    parts = [f"stages {done}/{len(views)}"]
    if active is not None:
      parts.append(
          f"{active.title} {theme_mod.format_count(active.trials_done, active.trials_total)}"
      )
      if active.metric_value is not None:
        parts.append(
            f"best {active.metric_name}={theme_mod.format_metric(active.metric_value)}"
        )
    parts.append(f"elapsed {theme_mod.format_duration(self.elapsed)}")
    return " | ".join(parts)


#: Byte that introduces every ANSI escape sequence (arrow keys, function
#: keys, mouse reports, bracketed paste, terminal status replies).
ESC = "\x1b"

#: Grace period for the remaining bytes of an escape sequence to arrive.
#: Terminals emit them back-to-back, so this only needs to cover scheduling
#: jitter; a lone ``ESC`` (the user pressed Escape) costs at most this wait.
ESCAPE_SEQUENCE_TIMEOUT_S = 0.05


class KeyReader:
  """Non-blocking single-keystroke reader for terminals in cbreak mode.

  Safe by construction: if stdin is not a TTY (nohup, CI, piped input) the
  reader silently does nothing, and the original terminal attributes are
  always restored in :meth:`stop`, including on exceptions.

  Escape sequences are consumed as a unit and never dispatched. Reading them
  byte by byte used to turn *Up arrow* (``ESC [ A``) into the ``[a]`` advance
  hotkey, which sealed a running sweep and killed the campaign; mouse reports
  and terminal replies could likewise synthesize ``[s]``/``[x]``.
  """

  def __init__(
      self,
      callback: Callable[[str], None],
      stream=None,
      escape_timeout_s: float = ESCAPE_SEQUENCE_TIMEOUT_S,
  ):
    self.callback = callback
    self._stream = stream
    self._escape_timeout_s = escape_timeout_s
    self._thread: Optional[threading.Thread] = None
    self._stop = threading.Event()
    self._old_attrs = None
    self._fd = None

  @property
  def enabled(self) -> bool:
    """True when the reader can actually capture keystrokes."""
    stream = self._stream
    if stream is None:
      import sys  # pylint: disable=g-import-not-at-top

      stream = sys.stdin
    try:
      return bool(stream.isatty())
    except Exception:  # pylint: disable=broad-except
      return False

  def start(self) -> bool:
    """Puts the terminal in cbreak mode and starts the listener thread."""
    if not self.enabled:
      return False
    try:
      import sys  # pylint: disable=g-import-not-at-top
      import termios  # pylint: disable=g-import-not-at-top
      import tty  # pylint: disable=g-import-not-at-top
    except ImportError:
      return False
    stream = self._stream or sys.stdin
    try:
      self._fd = stream.fileno()
      self._old_attrs = termios.tcgetattr(self._fd)
      tty.setcbreak(self._fd)
    except Exception:  # pylint: disable=broad-except
      self._old_attrs = None
      return False
    self._thread = threading.Thread(
        target=self._loop, args=(stream,), daemon=True, name="perl-keyreader"
    )
    self._thread.start()
    return True

  def _read_char(self, stream) -> str:
    """Reads one character, bypassing Python's text-layer buffering.

    ``select`` only sees the file descriptor, while ``stream.read(1)`` may
    pull a whole chunk into the ``TextIOWrapper`` buffer. Everything that
    followed an escape sequence in the same chunk would then sit invisible
    until the next keypress. Reading the fd directly keeps ``select`` and the
    data in sync.

    Args:
      stream: Fallback stream used when no file descriptor is available.

    Returns:
      A single character, or the empty string on EOF.
    """
    if self._fd is None:
      return stream.read(1)
    import os  # pylint: disable=g-import-not-at-top

    return os.read(self._fd, 1).decode("utf-8", errors="ignore")

  def _loop(self, stream) -> None:
    import select  # pylint: disable=g-import-not-at-top

    source = stream if self._fd is None else self._fd
    while not self._stop.is_set():
      try:
        ready, _, _ = select.select([source], [], [], 0.2)
        if not ready:
          continue
        char = self._read_char(stream)
        if not char:
          continue
        if char == ESC:
          # Arrow keys, function keys, mouse reports and paste markers all
          # start here. Consume the whole sequence and dispatch nothing.
          self._swallow_escape_sequence(stream, select)
          continue
        if not char.isprintable():
          # Control bytes (Ctrl-*, CR/LF, backspace) are not hotkeys.
          continue
        self.callback(char)
      except Exception:  # pylint: disable=broad-except
        return

  def _swallow_escape_sequence(self, stream, select_mod) -> None:
    """Consumes the remainder of an ANSI escape sequence, discarding it.

    Handles the three shapes a terminal can send after ``ESC``:
    CSI (``ESC [`` params final), SS3 (``ESC O`` final), and a bare
    Alt-chord (``ESC`` + one character). A lone ``ESC`` keypress simply
    times out.

    Args:
      stream: The (cbreak-mode) input stream being read.
      select_mod: The ``select`` module, injected to keep the import local.
    """
    source = stream if self._fd is None else self._fd
    deadline = time.monotonic() + self._escape_timeout_s
    saw_introducer = False
    while not self._stop.is_set():
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        return
      ready, _, _ = select_mod.select([source], [], [], remaining)
      if not ready:
        return
      char = self._read_char(stream)
      if not char:
        return
      if not saw_introducer:
        if char in ("[", "O"):
          saw_introducer = True
          continue
        # Alt-<key> or an unknown two-byte sequence: already consumed.
        return
      # Parameter (0x30-0x3F) and intermediate (0x20-0x2F) bytes continue the
      # sequence; anything else is the final byte that terminates it.
      if "\x20" <= char <= "\x3f":
        continue
      return

  def stop(self) -> None:
    """Restores the terminal attributes and stops the listener."""
    self._stop.set()
    if self._old_attrs is not None and self._fd is not None:
      try:
        import termios  # pylint: disable=g-import-not-at-top

        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
      except Exception:  # pylint: disable=broad-except
        pass
      finally:
        self._old_attrs = None
    if self._thread is not None:
      self._thread.join(timeout=0.5)
      self._thread = None

  def __enter__(self) -> "KeyReader":
    self.start()
    return self

  def __exit__(self, *exc_info) -> None:
    self.stop()


def handle_key(key: str, model: DashboardModel, controls: ControlSignals) -> Optional[str]:
  """Applies a keystroke to the dashboard; returns a notice message.

  Kept as a free function so that keyboard semantics are unit testable
  without spawning a terminal.

  Only a single printable character counts as a hotkey. ``KeyReader`` already
  swallows escape sequences, but this second guard makes sure a stray
  ``ESC``/``[``/control byte from any other caller cannot be mistaken for
  ``[a]``, ``[s]`` or ``[x]``.
  """
  if not key or len(key) != 1 or not key.isprintable():
    return None
  key = key.lower()
  if key == "p":
    return "Paused: no new stage will start." if controls.toggle_pause() else "Resumed."
  if key == "a":
    controls.request_advance()
    return "Advancing: sealing sweep and promoting the current best run."
  if key == "s":
    controls.request_stop()
    return "Stop requested: finishing current stage, then reporting."
  if key == "x":
    controls.request_abort()
    return "Abort requested: stopping without a final report."
  if key == "l":
    return f"Log verbosity: {model.cycle_log_level()}."
  if key == "?":
    model.show_help = not model.show_help
    return None
  if key == "+":
    model.log_window = min(20, model.log_window + 2)
    return None
  if key == "-":
    model.log_window = max(0, model.log_window - 2)
    return None
  if key == "q":
    return "detach"
  return None


class LiveDashboard:
  """Renders a :class:`DashboardModel` as a pinned live block."""

  def __init__(
      self,
      model: DashboardModel,
      console: Optional[UiConsole] = None,
      refresh_per_second: float = 6.0,
  ):
    self.model = model
    self.console = console or UiConsole()
    self.theme: Theme = self.console.theme
    self.refresh_per_second = refresh_per_second

  # --- Rendering ------------------------------------------------------
  def render(self):
    """Builds the pinned renderable (rich) for the current model state."""
    from rich.console import Group  # pylint: disable=g-import-not-at-top

    width = self.console.width
    views = self.model.stage_views()
    blocks: List[Any] = [
        renderables.header_panel(
            self.model.config,
            self.model.state,
            self.theme,
            elapsed_s=self.model.elapsed,
            paused=self.model.controls.is_paused,
            dry_run=bool(getattr(self.model.config, "dry_run", False)),
        ),
        renderables.dag_table(views, self.theme, width=width),
    ]

    parser = self.model.active_parser()
    if parser is not None and parser.trials:
      from rich.panel import Panel  # pylint: disable=g-import-not-at-top
      from rich import box  # pylint: disable=g-import-not-at-top

      stage = self.model.current_stage or ""
      stage_cfg = getattr(self.model.config, stage, None)
      limit = max(3, min(8, 4 + self.model.log_window))
      blocks.append(
          Panel(
              renderables.leaderboard_table(
                  parser.leaderboard(
                      goal=str(getattr(stage_cfg, "goal", "minimize")), limit=limit
                  ),
                  self.theme,
                  metric_name=parser.metric_name
                  or theme_mod.STAGE_METRIC_LABELS.get(stage, "metric"),
                  goal=str(getattr(stage_cfg, "goal", "minimize")),
                  limit=limit,
              ),
              title=self.theme.markup(
                  f"Live leaderboard {self.theme.glyphs.bullet} "
                  f"{theme_mod.STAGE_TITLES.get(stage, stage.upper())}",
                  "heading",
              ),
              border_style=self.theme.style("border") or "none",
              box=box.ROUNDED if self.theme.use_unicode else box.ASCII,
              padding=(0, 1),
          )
      )

    if self.model.show_help:
      blocks.append(renderables.help_panel(self.theme))

    blocks.append(self._footer(width))
    return Group(*blocks)

  def _footer(self, width: int):
    from rich.table import Table  # pylint: disable=g-import-not-at-top

    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=3)
    grid.add_column(justify="right", ratio=2)
    logs = self.model.recent_logs(1)
    activity = logs[-1][2] if logs else "waiting for output…"
    grid.add_row(
        renderables.hotkey_hint(
            self.theme, paused=self.model.controls.is_paused, width=width
        ),
        self.theme.markup(
            theme_mod.truncate(activity, max(20, width // 3)), "muted"
        ),
    )
    return grid

  # --- Log streaming --------------------------------------------------
  def flush_logs(self, printer: Optional[Callable[[str], None]] = None) -> int:
    """Prints buffered log lines above the pinned block.

    Returns:
      The number of lines printed.
    """
    entries = self.model.drain_pending_logs()
    if not entries:
      return 0
    printer = printer or self.console.print
    for timestamp, stage, message in entries:
      printer(self.format_log_line(timestamp, stage, message))
    return len(entries)

  def format_log_line(
      self, timestamp: float, stage: Optional[str], message: str
  ) -> str:
    """Formats one log line with a clock and a colored stage tag."""
    clock = datetime.datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
    tag = (stage or "run").upper()[:4].ljust(4)
    style = "error" if any(
        hint in message.lower() for hint in ("error", "failed", "traceback")
    ) else "value"
    return (
        f"{self.theme.markup(clock, 'muted')} "
        f"{self.theme.markup(tag, 'accent_dim')} "
        f"{self.theme.markup(theme_mod.strip_markup(message), style)}"
    )

  # --- Main loop ------------------------------------------------------
  def run(self, is_done: Callable[[], bool], poll_seconds: float = 0.15) -> None:
    """Renders until ``is_done()`` returns True (blocking)."""
    if not self.console.is_rich:
      self.run_plain(is_done, poll_seconds=poll_seconds)
      return
    from rich.live import Live  # pylint: disable=g-import-not-at-top

    with Live(
        self.render(),
        console=self.console.rich,
        refresh_per_second=self.refresh_per_second,
        transient=False,
        screen=False,
        vertical_overflow="visible",
    ) as live:
      while not is_done():
        self.flush_logs(printer=lambda text: live.console.print(text, markup=True))
        live.update(self.render())
        time.sleep(poll_seconds)
      self.flush_logs(printer=lambda text: live.console.print(text, markup=True))
      live.update(self.render())

  def run_plain(self, is_done: Callable[[], bool], poll_seconds: float = 0.15) -> None:
    """Fallback loop: timestamped log streaming plus periodic progress."""
    last_summary = 0.0
    while not is_done():
      printed = self.flush_logs()
      now = time.time()
      if now - last_summary > 30 or (printed and last_summary == 0.0):
        self.console.print(
            self.theme.markup(f"[progress] {self.model.progress_summary()}", "accent")
        )
        last_summary = now
      time.sleep(poll_seconds)
    self.flush_logs()


def run_campaign_with_dashboard(
    config: Any,
    state: Any = None,
    console: Optional[UiConsole] = None,
    enable_keys: bool = True,
    engine_factory: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
  """Runs a campaign with the live dashboard attached.

  Args:
    config: ``CampaignConfig`` to execute.
    state: Optional pre-loaded ``CampaignState`` (resume).
    console: Output surface; created from terminal capabilities when omitted.
    enable_keys: Whether to capture hotkeys (disabled automatically when
      stdin is not a TTY).
    engine_factory: Injection point for tests; defaults to ``CampaignEngine``.

  Returns:
    The engine result dictionary, augmented with ``ui_status``.
  """
  from src.orchestrator.engine import CampaignEngine  # pylint: disable=g-import-not-at-top

  console = console or UiConsole()
  bus = EventBus()
  controls = ControlSignals()
  factory = engine_factory or CampaignEngine

  # The reporter prints its own scorecard; inside a live region that would
  # collide with the pinned block, so we render it ourselves afterwards.
  reporting = getattr(config, "reporting", None)
  previous_summary_flag = getattr(reporting, "render_console_summary", None)
  if reporting is not None:
    reporting.render_console_summary = False

  engine = factory(config=config, state=state, event_bus=bus, controls=controls)

  model = DashboardModel(config=config, state=engine.state, controls=controls)
  bus.subscribe(model.on_event)
  dashboard = LiveDashboard(model, console=console)

  result: Dict[str, Any] = {}
  error: List[BaseException] = []
  done = threading.Event()
  detached = threading.Event()

  def _worker() -> None:
    try:
      result.update(engine.run() or {})
    except BaseException as exc:  # pylint: disable=broad-except
      error.append(exc)
    finally:
      done.set()

  def _on_key(char: str) -> None:
    notice = handle_key(char, model, controls)
    if notice == "detach":
      # Only the rendering loop is torn down here. The campaign itself keeps
      # running in the worker thread and switches to plain log streaming; the
      # engine thread is a daemon, so returning from this function would kill
      # a multi-hour run outright.
      detached.set()
    elif notice:
      bus.publish(EventType.NOTICE, message=notice)

  def _ui_finished() -> bool:
    return done.is_set() or detached.is_set()

  thread = threading.Thread(target=_worker, name="perl-campaign", daemon=True)
  thread.start()

  reader = KeyReader(_on_key) if enable_keys else None
  keys_active = bool(reader and reader.start())
  if enable_keys and not keys_active:
    bus.publish(
        EventType.NOTICE,
        message="Hotkeys unavailable (stdin is not a TTY); running unattended.",
    )
  try:
    dashboard.run(is_done=_ui_finished)
  except KeyboardInterrupt:
    controls.request_stop()
    console.warn("Interrupt received - stopping after the current stage (Ctrl-C again to abort).")
    try:
      dashboard.run(is_done=_ui_finished)
    except KeyboardInterrupt:
      controls.request_abort()
  finally:
    if reader is not None:
      reader.stop()
    if reporting is not None and previous_summary_flag is not None:
      reporting.render_console_summary = previous_summary_flag

  if detached.is_set() and not done.is_set():
    _stream_plain_until_done(bus, console, controls, done)

  thread.join(timeout=1.0 if model.finished or done.is_set() else 0.1)
  if error:
    raise error[0]

  if previous_summary_flag:
    render_final_scorecard(model, console)
  for notice in list(model.notices):
    console.info(notice)
  result.setdefault("ui_status", model.final_status or "DETACHED")
  return result


def _stream_plain_until_done(
    bus: EventBus,
    console: UiConsole,
    controls: Any,
    done: threading.Event,
    poll_interval: float = 0.5,
) -> None:
  """Keeps the campaign in the foreground with plain (non-live) log output.

  Closing the live region does not stop the engine, so after a detach we must
  block the main thread until the worker finishes. Exiting here would kill the
  daemon worker thread and orphan the ``wandb agent`` subprocess with no state
  updates.

  Args:
    bus: Event bus the engine publishes to.
    console: Output surface used for the plain log stream.
    controls: ``ControlSignals`` used to honour Ctrl-C.
    done: Set by the worker thread once the campaign terminates.
    poll_interval: Seconds between liveness checks.
  """
  console.blank()
  console.info(
      "Dashboard detached. The campaign keeps running here in plain-log mode; "
      "press Ctrl-C to stop after the current stage."
  )

  def _echo(event: CampaignEvent) -> None:
    if event.type in (EventType.LOG, EventType.NOTICE):
      text = event.message
    else:
      text = f"[{event.type.value}] {event.stage or ''} {event.message}".strip()
    if text:
      console.info(f"{event.clock} {text}")

  unsubscribe = bus.subscribe(_echo)
  try:
    while not done.wait(poll_interval):
      pass
  except KeyboardInterrupt:
    controls.request_stop()
    console.warn(
        "Interrupt received - stopping after the current stage "
        "(Ctrl-C again to abort)."
    )
    try:
      while not done.wait(poll_interval):
        pass
    except KeyboardInterrupt:
      controls.request_abort()
      done.wait(30.0)
  finally:
    unsubscribe()


def render_final_scorecard(model: DashboardModel, console: UiConsole) -> None:
  """Prints the end-of-campaign scorecard below the (now closed) live block."""
  console.blank()
  if console.is_rich:
    console.print_renderable(
        renderables.status_table(
            model.config, model.state, console.theme, width=console.width
        )
    )
    return
  console.print_lines(
      renderables.summary_lines(
          model.config, model.state, console.theme, width=console.width
      )
  )
