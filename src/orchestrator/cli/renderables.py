"""Composable renderables shared by the dashboard, the CLI and the reporter.

Everything in this module is a *pure function* of campaign state: it takes
config/state objects and returns either rich renderables (when ``rich`` is
installed) or plain strings. That separation is what makes the terminal UI
testable without a terminal - the unit tests render into a fixed-width
in-memory console and assert on the produced text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.orchestrator import eval_metrics
from src.orchestrator import flavors
from src.orchestrator.cli import theme as theme_mod
from src.orchestrator.cli.theme import Theme


def rich_available() -> bool:
  """Returns True when the optional ``rich`` dependency can be imported."""
  try:
    import rich  # pylint: disable=g-import-not-at-top,unused-import

    return True
  except ImportError:
    return False


@dataclass
class StageView:
  """Everything the UI needs to draw one row of the campaign DAG."""

  key: str
  title: str
  status: str = "PENDING"
  trials_done: int = 0
  trials_total: int = 0
  metric_name: str = ""
  metric_value: Optional[float] = None
  goal: str = "minimize"
  elapsed_s: Optional[float] = None
  model_repo_id: Optional[str] = None
  error: Optional[str] = None
  history: List[float] = field(default_factory=list)
  #: ``complete`` / ``partial`` / ``unknown`` as recorded by the stage, or
  #: None for a stage that has not finished its sweep (or predates the
  #: field).
  sweep_outcome: Optional[str] = None
  warnings: List[str] = field(default_factory=list)

  @property
  def is_active(self) -> bool:
    return str(self.status).upper() == "RUNNING"

  @property
  def is_partial(self) -> bool:
    """True when the stage completed on fewer trials than it asked for.

    This is the state the dashboard used to be unable to express: a PE-RL
    stage that lost its 4th of 5 trials to a dead VM published a model, was
    marked COMPLETED, and was drawn as a full ``05/05`` bar.
    """
    if str(self.status).upper() != "COMPLETED":
      return False
    if self.sweep_outcome == "partial":
      return True
    return bool(self.trials_total) and 0 < self.trials_done < self.trials_total

  @property
  def fraction(self) -> float:
    """Completion ratio in ``[0, 1]``."""
    status = str(self.status).upper()
    if status == "SKIPPED":
      return 1.0
    if not self.trials_total:
      return 1.0 if status == "COMPLETED" else 0.0
    if status == "COMPLETED" and not self.trials_done:
      # Nothing was recorded, which is what every state file written before
      # trial accounting existed looks like. Claiming 0/N for a stage that
      # demonstrably finished would be a worse lie than the old one.
      return 1.0
    return max(0.0, min(1.0, self.trials_done / float(self.trials_total)))


def _parse_iso(value: Optional[str]) -> Optional[datetime.datetime]:
  if not value:
    return None
  try:
    return datetime.datetime.fromisoformat(value)
  except (TypeError, ValueError):
    return None


def stage_elapsed_seconds(stage_result: Any, now: Optional[datetime.datetime] = None) -> Optional[float]:
  """Computes how long a stage ran, from its persisted ISO timestamps."""
  if stage_result is None:
    return None
  start = _parse_iso(getattr(stage_result, "start_time", None))
  if start is None:
    return None
  end = _parse_iso(getattr(stage_result, "end_time", None))
  if end is None:
    end = now or datetime.datetime.now()
  delta = (end - start).total_seconds()
  return max(0.0, delta)


def stage_budget(config: Any, stage_key: str) -> int:
  """Returns the configured trial budget for a stage (0 for non-sweeps)."""
  stage_cfg = getattr(config, stage_key, None)
  return int(getattr(stage_cfg, "max_runs", 0) or 0)


def build_stage_views(
    config: Any,
    state: Any,
    live: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[StageView]:
  """Derives the list of :class:`StageView` rows for a campaign.

  Args:
    config: A ``CampaignConfig``.
    state: A ``CampaignState``.
    live: Optional per-stage live overrides, e.g.
      ``{"sft": {"trials_done": 4, "metric_value": 0.31, "history": [...]}}``.

  Returns:
    One view per configured stage, in execution order. A campaign that
    branches over several reward-model dataset flavors gets one row per
    branch, keyed by the branch id (``rm:synthetic_struct``).
  """
  live = live or {}
  views: List[StageView] = []
  plan = flavors.build_plan(config, theme_mod.STAGE_TITLES)
  state_stages = getattr(state, "stages", {}) or {}

  for planned in plan:
    key = planned.stage_id
    kind = planned.kind
    result = state_stages.get(key)
    status = "PENDING"
    if result is not None:
      raw_status = getattr(result, "status", None)
      status = getattr(raw_status, "value", raw_status) or "PENDING"

    metric_name = theme_mod.STAGE_METRIC_LABELS.get(kind, "metric")
    stage_cfg = getattr(config, kind, None)
    configured_metric = getattr(stage_cfg, "metric", None)
    if configured_metric:
      metric_name = configured_metric

    metric_value = getattr(result, "best_metric_val", None) if result else None
    if kind == "eval" and result is not None:
      metrics = getattr(result, "metrics", {}) or {}
      # Eval scores several policies; the headline number is the PE-RL one.
      # It is only a *preference*: an eval that recorded something else must
      # not end up with an emptier row than one that recorded nothing.
      headline = eval_metrics.headline_metric(metrics)
      if headline is not None:
        metric_value = headline

    view = StageView(
        key=key,
        title=planned.title or key.upper(),
        status=str(status).upper(),
        trials_total=stage_budget(config, kind),
        metric_name=metric_name,
        metric_value=metric_value,
        goal=str(getattr(stage_cfg, "goal", "minimize") or "minimize"),
        elapsed_s=stage_elapsed_seconds(result),
        model_repo_id=getattr(result, "model_repo_id", None) if result else None,
        error=getattr(result, "error_message", None) if result else None,
    )
    # Progress of the *running* campaign, mirrored into the state file.
    # Without this, any reader that is not the process owning the dashboard -
    # a `status --watch` in a second tmux pane, `status --json`, a plain run -
    # would draw a permanently frozen 00/N for the active stage.
    if result is not None:
      persisted_total = int(getattr(result, "trials_total", 0) or 0)
      if persisted_total:
        # The budget the sweep actually committed to, recorded by the process
        # that ran it. It beats `stage_budget`, which may come from a config
        # rebuilt from defaults because the state file predates config
        # persistence, or one the user has edited since.
        view.trials_total = persisted_total
      view.trials_done = min(
          int(getattr(result, "trials_done", 0) or 0), view.trials_total or 0
      )
      view.sweep_outcome = getattr(result, "sweep_outcome", None)
      view.warnings = list(getattr(result, "warnings", None) or [])
    # NOTE: a COMPLETED stage used to have its counter forced to the full
    # budget here. That is how a PE-RL sweep that crashed during its 4th of
    # 5 trials still rendered "05/05" and read as a clean run. The stage now
    # records what W&B says actually finished, and the counter shows it.

    overrides = live.get(key) or {}
    for attr, value in overrides.items():
      if hasattr(view, attr) and value is not None:
        setattr(view, attr, value)
    views.append(view)
  return views


# --------------------------------------------------------------------------
# Text (markup) builders - usable with or without rich.
# --------------------------------------------------------------------------
def _bar_values(view: StageView) -> Tuple[int, int, str]:
  """Returns ``(done, total, counter_text)`` for a stage row.

  Stages without a trial budget (the evaluation stage) would otherwise render
  an empty bar and a meaningless ``0`` counter, so they are displayed as a
  single completed unit once they finish.
  """
  status = str(view.status).upper()
  if not view.trials_total:
    if status in ("COMPLETED", "SKIPPED"):
      return 1, 1, "done"
    if status == "RUNNING":
      return 0, 1, "running"
    if status == "FAILED":
      return 1, 1, "failed"
    return 0, 1, "-"
  done = view.trials_total if view.fraction >= 1.0 else view.trials_done
  return done, view.trials_total, theme_mod.format_count(done, view.trials_total)


def dag_lines(
    views: Sequence[StageView],
    theme: Theme,
    width: int = 100,
    bar_width: int = 0,
) -> List[str]:
  """Renders the campaign DAG as a list of markup strings.

  The layout adapts to ``width`` so the panel stays readable from a narrow
  tmux split up to a full-screen terminal.

  Args:
    views: Stage rows to draw.
    theme: Active theme.
    width: Available terminal width.
    bar_width: Explicit progress bar width (auto when 0).

  Returns:
    One markup string per stage.
  """
  width = max(40, int(width))
  if not bar_width:
    bar_width = 12 if width < 90 else (20 if width < 130 else 28)
  title_width = max(12, min(22, width // 5))
  show_model = width >= 110

  lines: List[str] = []
  for index, view in enumerate(views, start=1):
    marker = theme_mod.status_marker(view.status, theme)
    title = theme_mod.truncate(view.title, title_width).ljust(title_width)
    title_style = "heading" if view.is_active else "value"
    if str(view.status).upper() == "PENDING":
      title_style = "muted"
    bar_style = {
        "COMPLETED": "success",
        "FAILED": "error",
        "RUNNING": "running",
    }.get(str(view.status).upper(), "muted")
    if view.is_partial:
      # Same status, different story: the stage produced a model, but from a
      # search that was cut short. Green would say it all went to plan.
      bar_style = "warning"
    done, total, counter = _bar_values(view)
    bar = theme_mod.progress_bar(done, total, bar_width, theme, bar_style)
    metric = theme_mod.format_metric(view.metric_value)
    metric_cell = f"{view.metric_name}={metric}" if view.metric_value is not None else ""
    elapsed = theme_mod.format_duration(view.elapsed_s) if view.elapsed_s else ""

    parts = [
        f"{marker} {index}.",
        theme.markup(title, title_style),
        bar,
        counter.rjust(7),
    ]
    if view.is_partial:
      parts.append(theme.markup("PARTIAL", "warning"))
    if metric_cell:
      parts.append(theme.markup(metric_cell, "metric"))
    if elapsed:
      parts.append(theme.markup(elapsed, "muted"))
    if show_model and view.model_repo_id:
      parts.append(
          theme.markup(
              theme_mod.truncate(view.model_repo_id, 34), "accent_dim"
          )
      )
    if view.error:
      parts.append(theme.markup(theme_mod.truncate(view.error, 40), "error"))
    lines.append("  ".join(parts))
  return lines


def summary_lines(
    config: Any,
    state: Any,
    theme: Theme,
    width: int = 100,
) -> List[str]:
  """Renders a compact end-of-campaign scorecard as markup strings."""
  views = build_stage_views(config, state)
  lines = [
      theme.markup(
          f"Campaign {getattr(state, 'campaign_id', '?')} "
          f"[{getattr(state, 'status', '?')}]",
          "heading",
      )
  ]
  lines.extend(dag_lines(views, theme, width=width))
  total = sum(v.elapsed_s or 0.0 for v in views)
  if total:
    lines.append(
        theme.markup(
            f"Total stage time: {theme_mod.format_duration(total)}", "muted"
        )
    )
  # The last thing on screen is the last thing remembered. A campaign that
  # lost trials has to say so here, not only in the log scrollback that has
  # long since scrolled away.
  for view in views:
    for warning in view.warnings:
      lines.append(theme.markup(f"! {view.title}: {warning}", "warning"))
  return lines


def hotkey_hint(theme: Theme, paused: bool = False, width: int = 100) -> str:
  """Renders the footer hotkey legend.

  Args:
    theme: Active theme.
    paused: Whether the campaign is currently paused.
    width: Available width; narrow terminals get abbreviated labels.

  Returns:
    A markup string.
  """
  pause_label = "resume" if paused else "pause"
  if width < 88:
    keys: List[Tuple[str, str]] = [
        ("a", "adv"),
        ("p", pause_label[:3]),
        ("s", "stop"),
        ("?", "help"),
        ("q", "quit"),
    ]
  else:
    keys = [
        ("a", "advance w/ best"),
        ("p", pause_label),
        ("s", "stop sweep"),
        ("l", "logs"),
        ("?", "help"),
        ("q", "detach"),
    ]
  rendered = [
      f"{theme.markup(f'[{key}]', 'accent')} {theme.markup(label, 'muted')}"
      for key, label in keys
  ]
  return "  ".join(rendered)


HELP_TEXT: List[Tuple[str, str]] = [
    ("a", "Seal the running sweep, promote its current best run, advance."),
    ("p", "Pause before the next stage (current trial finishes cleanly)."),
    ("s", "Stop the campaign gracefully after the current stage."),
    ("x", "Abort now, skipping report generation."),
    ("l", "Cycle log verbosity: all -> milestones -> off."),
    ("+/-", "Grow or shrink the live log window."),
    ("?", "Toggle this help overlay."),
    ("q", "Detach the dashboard; the campaign keeps running headless."),
]


# --------------------------------------------------------------------------
# Rich renderables (imported lazily so the module stays importable without it).
# --------------------------------------------------------------------------
def dag_table(views: Sequence[StageView], theme: Theme, width: int = 100):
  """Builds a ``rich.table.Table`` for the campaign DAG."""
  from rich.table import Table  # pylint: disable=g-import-not-at-top
  from rich import box  # pylint: disable=g-import-not-at-top

  bar_width = 12 if width < 90 else (18 if width < 130 else 26)
  table = Table(
      box=box.SIMPLE_HEAD if theme.use_unicode else box.ASCII,
      expand=True,
      pad_edge=False,
      show_edge=False,
      header_style=theme.style("muted") or None,
  )
  table.add_column("", width=3, no_wrap=True)
  table.add_column("Stage", ratio=3, no_wrap=True)
  table.add_column("Progress", width=bar_width, no_wrap=True)
  table.add_column("Trials", width=8, justify="right", no_wrap=True)
  table.add_column("Best", ratio=2, justify="right", no_wrap=True)
  table.add_column("Elapsed", width=9, justify="right", no_wrap=True)
  if width >= 110:
    table.add_column("Artifact", ratio=3, no_wrap=True)

  for view in views:
    status = str(view.status).upper()
    bar_style = {
        "COMPLETED": "success",
        "FAILED": "error",
        "RUNNING": "running",
    }.get(status, "muted")
    if view.is_partial:
      bar_style = "warning"
    done, total, counter = _bar_values(view)
    if view.is_partial:
      counter = theme.markup(counter, "warning")
    title_style = "heading" if view.is_active else (
        "muted" if status == "PENDING" else "value"
    )
    metric_cell = "-"
    if view.metric_value is not None:
      metric_cell = theme.markup(
          theme_mod.format_metric(view.metric_value), "metric"
      )
      spark = theme_mod.sparkline(view.history, theme, width=12)
      if spark:
        metric_cell = f"{theme.markup(spark, 'accent_dim')} {metric_cell}"
    row = [
        theme_mod.status_marker(status, theme),
        theme.markup(view.title, title_style),
        theme_mod.progress_bar(done, total, bar_width, theme, bar_style),
        counter,
        metric_cell,
        theme.markup(theme_mod.format_duration(view.elapsed_s), "muted")
        if view.elapsed_s
        else "-",
    ]
    if width >= 110:
      # A cut-short search outranks the artifact name here: the repo id is
      # discoverable elsewhere, the fact that it was trained on a partial
      # sweep is not.
      if view.error:
        artifact, artifact_style = view.error, "error"
      elif view.is_partial:
        artifact = f"PARTIAL: {done}/{total} trials finished"
        artifact_style = "warning"
      else:
        artifact, artifact_style = view.model_repo_id or "", "accent_dim"
      row.append(
          theme.markup(theme_mod.truncate(artifact, 40), artifact_style)
          if artifact
          else ""
      )
    table.add_row(*row)
  return table


def leaderboard_table(
    trials: Sequence[Any],
    theme: Theme,
    metric_name: str = "metric",
    goal: str = "minimize",
    limit: int = 8,
):
  """Builds a ``rich`` leaderboard of sweep trials, best first."""
  from rich.table import Table  # pylint: disable=g-import-not-at-top
  from rich import box  # pylint: disable=g-import-not-at-top

  table = Table(
      box=box.SIMPLE if theme.use_unicode else box.ASCII,
      expand=True,
      pad_edge=False,
      show_edge=False,
      header_style=theme.style("muted") or None,
  )
  table.add_column("#", width=4, justify="right", no_wrap=True)
  table.add_column("Run", ratio=2, no_wrap=True)
  table.add_column("Params", ratio=4, no_wrap=True)
  table.add_column(theme_mod.truncate(metric_name, 14), ratio=2, justify="right", no_wrap=True)
  table.add_column("State", width=8, no_wrap=True)

  ranked = list(trials)[: limit or None]
  for rank, trial in enumerate(ranked):
    params = getattr(trial, "params", {}) or {}
    params_text = ", ".join(
        f"{key}={theme_mod.format_metric(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value}"
        for key, value in list(params.items())[:4]
    )
    state = str(getattr(trial, "state", "")).lower()
    state_style = {"done": "success", "running": "running", "failed": "error"}.get(
        state, "muted"
    )
    crown = theme.glyphs.bullet
    if rank == 0 and getattr(trial, "metric", None) is not None:
      crown = "★" if theme.use_unicode else "*"
    table.add_row(
        theme.markup(crown, "warning") + str(getattr(trial, "index", rank + 1)),
        theme.markup(theme_mod.truncate(str(getattr(trial, "run_id", "?")), 16), "value"),
        theme.markup(theme_mod.truncate(params_text, 46), "muted"),
        theme.markup(theme_mod.format_metric(getattr(trial, "metric", None)), "metric"),
        theme.markup(state or "-", state_style),
    )
  if not ranked:
    table.add_row("", theme.markup("waiting for first trial…", "muted"), "", "", "")
  return table


def status_table(config: Any, state: Any, theme: Theme, width: int = 100):
  """Builds the table used by ``run_campaign.py status``.

  Args:
    config: Campaign configuration.
    state: Campaign state.
    theme: Active theme.
    width: Available terminal width; narrow terminals drop optional columns.

  Returns:
    A ``rich.table.Table``.
  """
  from rich.table import Table  # pylint: disable=g-import-not-at-top
  from rich import box  # pylint: disable=g-import-not-at-top

  views = build_stage_views(config, state)
  compact = width < 100
  table = Table(
      box=box.ROUNDED if theme.use_unicode else box.ASCII,
      expand=False,
      header_style=theme.style("heading") or None,
      border_style=theme.style("border") or None,
      pad_edge=False,
  )
  # No explicit width here: cell padding would otherwise squeeze the glyph
  # column down to zero characters on an 80 column terminal.
  table.add_column("", no_wrap=True)
  table.add_column("Stage", no_wrap=True)
  table.add_column("Status", no_wrap=True)
  table.add_column("Trials", justify="right", no_wrap=True)
  table.add_column("Best", justify="right", no_wrap=True)
  table.add_column("Elapsed", justify="right", no_wrap=True)
  if not compact:
    table.add_column("Artifact", overflow="fold")

  for view in views:
    _, _, counter = _bar_values(view)
    metric = theme_mod.format_metric(view.metric_value)
    if view.metric_value is not None and not compact:
      metric = f"{view.metric_name} = {metric}"
    row = [
        theme_mod.status_marker(view.status, theme),
        theme.markup(theme_mod.truncate(view.title, 24), "value"),
        theme_mod.status_label(view.status, theme),
        counter,
        theme.markup(metric, "metric") if view.metric_value is not None else "-",
        theme_mod.format_duration(view.elapsed_s) if view.elapsed_s else "-",
    ]
    if not compact:
      row.append(
          theme.markup(view.error, "error")
          if view.error
          else (view.model_repo_id or "-")
      )
    table.add_row(*row)
  return table


def header_panel(
    config: Any,
    state: Any,
    theme: Theme,
    elapsed_s: float = 0.0,
    paused: bool = False,
    dry_run: bool = False,
):
  """Builds the pinned dashboard header."""
  from rich.panel import Panel  # pylint: disable=g-import-not-at-top
  from rich.table import Table  # pylint: disable=g-import-not-at-top
  from rich import box  # pylint: disable=g-import-not-at-top

  grid = Table.grid(expand=True)
  grid.add_column(justify="left", ratio=2)
  grid.add_column(justify="center", ratio=1)
  grid.add_column(justify="right", ratio=2)

  badges = []
  if dry_run:
    badges.append(theme.markup(" DRY-RUN ", "warning"))
  if paused:
    badges.append(theme.markup(" PAUSED ", "warning"))
  status = str(getattr(state, "status", "")).upper()
  if status:
    style = {"COMPLETED": "success", "FAILED": "error"}.get(status, "accent")
    badges.append(theme.markup(f" {status} ", style))

  grid.add_row(
      theme.markup(f"Task: {getattr(config, 'task_name', '?')}", "heading")
      + theme.markup(f"  {theme.glyphs.bullet}  ", "muted")
      + theme.markup(str(getattr(config, 'base_model', '')), "accent_dim"),
      " ".join(badges),
      theme.markup(
          f"Elapsed {theme_mod.format_duration(elapsed_s)}", "value"
      ),
  )
  return Panel(
      grid,
      title=theme.markup("Auto-PERL Mission Control", "heading"),
      subtitle=theme.markup(str(getattr(config, "name", "")), "muted"),
      border_style=theme.style("border") or "none",
      box=box.ROUNDED if theme.use_unicode else box.ASCII,
      padding=(0, 1),
  )


def help_panel(theme: Theme):
  """Builds the ``?`` help overlay."""
  from rich.panel import Panel  # pylint: disable=g-import-not-at-top
  from rich.table import Table  # pylint: disable=g-import-not-at-top
  from rich import box  # pylint: disable=g-import-not-at-top

  table = Table.grid(padding=(0, 2))
  table.add_column(justify="right", no_wrap=True)
  table.add_column()
  for key, description in HELP_TEXT:
    table.add_row(theme.markup(key, "accent"), theme.markup(description, "value"))
  return Panel(
      table,
      title=theme.markup("Keyboard controls", "heading"),
      border_style=theme.style("accent") or "none",
      box=box.ROUNDED if theme.use_unicode else box.ASCII,
      padding=(0, 1),
  )


def banner(theme: Theme, subtitle: str = "") -> str:
  """Returns the ASCII/Unicode product banner used on startup."""
  if theme.use_unicode:
    art = [
        "╔═╗┬ ┬┌┬┐┌─┐  ╔═╗╔═╗╦═╗╦  ",
        "╠═╣│ │ │ │ │  ╠═╝║╣ ╠╦╝║  ",
        "╩ ╩└─┘ ┴ └─┘  ╩  ╚═╝╩╚═╩═╝",
    ]
  else:
    art = [
        " _         _          ___ ___ ___ _    ",
        "/_\\ _  _ _| |_ ___   | _ \\ __| _ \\ |   ",
        "/ _ \\ || |  _/ _ \\   |  _/ _||   / |__ ",
        "\\_/ \\_\\_,_|\\__\\___/  |_| |___|_|_\\____|",
    ]
  lines = [theme.markup(line, "accent") for line in art]
  if subtitle:
    lines.append(theme.markup(subtitle, "muted"))
  return "\n".join(lines)
