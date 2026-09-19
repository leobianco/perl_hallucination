"""Visual theme, capability detection, and formatting helpers for the CLI.

This module is intentionally dependency-free (no ``rich`` import at module
level) so that every other CLI module can rely on a single, consistent source
of truth for colors, glyphs, and value formatting, even when the optional
pretty-printing dependencies are unavailable.

Design goals:
  * Respect the de-facto standards ``NO_COLOR`` / ``FORCE_COLOR`` / ``TERM``.
  * Degrade from truecolor -> 16 colors -> no color without losing information.
  * Degrade from Unicode glyphs to pure ASCII when the terminal encoding
    cannot represent box drawing / emoji characters (common on bare tmux
    sessions started with LANG=C on a GCP VM).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import math
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO


# Ordered from best-looking to most compatible.
SPARK_CHARS = "▁▂▃▄▅▆▇█"
ASCII_SPARK_CHARS = ".:-=+*#%"


@dataclass(frozen=True)
class Glyphs:
  """Character set used to draw status markers and bars."""

  pending: str = "○"
  running: str = "▶"
  completed: str = "✔"
  failed: str = "✖"
  skipped: str = "–"
  paused: str = "❙❙"
  bar_full: str = "█"
  bar_empty: str = "░"
  bar_head: str = "▓"
  arrow: str = "→"
  bullet: str = "•"
  spark: str = SPARK_CHARS

  @classmethod
  def ascii(cls) -> "Glyphs":
    """Returns a pure-ASCII glyph set for terminals without Unicode support."""
    return cls(
        pending="o",
        running=">",
        completed="+",
        failed="x",
        skipped="-",
        paused="||",
        bar_full="#",
        bar_empty=".",
        bar_head="#",
        arrow="->",
        bullet="*",
        spark=ASCII_SPARK_CHARS,
    )


@dataclass(frozen=True)
class Theme:
  """Semantic color palette and glyph set.

  Styles are stored as ``rich`` style strings. When ``use_color`` is False,
  :meth:`style` returns an empty string so callers can build markup-free text.
  """

  use_color: bool = True
  use_unicode: bool = True
  glyphs: Glyphs = dataclasses.field(default_factory=Glyphs)

  # Semantic styles.
  accent: str = "bright_cyan"
  accent_dim: str = "cyan"
  heading: str = "bold bright_white"
  muted: str = "grey58"
  success: str = "bold green"
  warning: str = "bold yellow"
  error: str = "bold red"
  running: str = "bold bright_yellow"
  pending: str = "grey58"
  skipped: str = "grey42"
  metric: str = "bold bright_magenta"
  value: str = "bright_white"
  border: str = "cyan"
  border_done: str = "green"
  border_failed: str = "red"

  def style(self, name: str) -> str:
    """Returns the style string for a semantic name, honoring ``use_color``."""
    if not self.use_color:
      return ""
    return getattr(self, name, "")

  def markup(self, text: str, style_name: str) -> str:
    """Wraps ``text`` in rich markup for ``style_name``.

    The text is always escaped first: campaign output legitimately contains
    square brackets (``[DRY-RUN]``, hotkey hints such as ``[a]``) that must
    not be mistaken for style tags.

    Args:
      text: Literal text to display.
      style_name: Semantic style attribute name.

    Returns:
      A markup string, or the escaped text when colors are disabled.
    """
    escaped = escape_markup(text)
    style = self.style(style_name)
    if not style:
      return escaped
    return f"[{style}]{escaped}[/{style}]"


#: Maps ``StageStatus`` values (as plain strings, to avoid a circular import)
#: onto a (glyph attribute, style attribute, human label) triple.
STATUS_PRESENTATION: Dict[str, Any] = {
    "PENDING": ("pending", "pending", "pending"),
    "RUNNING": ("running", "running", "running"),
    "COMPLETED": ("completed", "success", "done"),
    "FAILED": ("failed", "error", "failed"),
    "SKIPPED": ("skipped", "skipped", "skipped"),
    "PAUSED": ("paused", "warning", "paused"),
}

#: Human readable stage titles used across every surface of the CLI.
STAGE_TITLES: Dict[str, str] = {
    "autorater": "Autorater Calibration",
    "sft": "SFT Sweep",
    "rm": "Reward Model Sweep",
    "perl": "PE-RL Sweep",
    "eval": "Final Evaluation",
}

#: Short metric labels shown in compact tables.
STAGE_METRIC_LABELS: Dict[str, str] = {
    "autorater": "judge roc_auc",
    "sft": "eval/loss",
    "rm": "eval/roc_auc",
    "perl": "reward/mean",
    "eval": "halluc.rate",
}


def _env_flag(name: str) -> Optional[bool]:
  """Reads a tri-state boolean environment flag (unset/true/false)."""
  raw = os.environ.get(name)
  if raw is None:
    return None
  if raw.strip() == "":
    # Per the NO_COLOR spec, presence alone is enough.
    return True
  return raw.strip().lower() not in ("0", "false", "no", "off")


def supports_color(stream: Optional[TextIO] = None) -> bool:
  """Detects whether ANSI color should be emitted on ``stream``."""
  if _env_flag("NO_COLOR"):
    return False
  forced = _env_flag("FORCE_COLOR")
  if forced:
    return True
  term = os.environ.get("TERM", "")
  if term in ("dumb", ""):
    # tmux/ssh always set TERM; an empty TERM means a non-interactive pipe.
    if term == "dumb":
      return False
  stream = stream or sys.stdout
  try:
    return bool(stream.isatty())
  except Exception:  # pylint: disable=broad-except
    return False


def supports_unicode(stream: Optional[TextIO] = None) -> bool:
  """Detects whether the output encoding can render the Unicode glyph set."""
  if _env_flag("PERL_ASCII"):
    return False
  stream = stream or sys.stdout
  encoding = getattr(stream, "encoding", None) or ""
  if not encoding:
    encoding = sys.getdefaultencoding()
  try:
    "▁▶✔░".encode(encoding)
  except (UnicodeEncodeError, LookupError):
    return False
  return True


def detect_theme(
    force_color: Optional[bool] = None,
    force_ascii: Optional[bool] = None,
    stream: Optional[TextIO] = None,
) -> Theme:
  """Builds a :class:`Theme` from explicit overrides and terminal capabilities.

  Args:
    force_color: ``True``/``False`` to force color on/off, ``None`` to detect.
    force_ascii: ``True`` to force the ASCII glyph set, ``None`` to detect.
    stream: Output stream inspected for tty-ness and encoding.

  Returns:
    A fully resolved theme.
  """
  use_color = supports_color(stream) if force_color is None else bool(force_color)
  if force_ascii is None:
    use_unicode = supports_unicode(stream)
  else:
    use_unicode = not force_ascii
  glyphs = Glyphs() if use_unicode else Glyphs.ascii()
  return Theme(use_color=use_color, use_unicode=use_unicode, glyphs=glyphs)


def status_marker(status: str, theme: Theme) -> str:
  """Returns the colored glyph for a stage status (e.g. a green check)."""
  glyph_name, style_name, _ = STATUS_PRESENTATION.get(
      str(status).upper(), ("pending", "pending", "pending")
  )
  glyph = getattr(theme.glyphs, glyph_name)
  return theme.markup(glyph, style_name)


def status_label(status: str, theme: Theme) -> str:
  """Returns the colored human label for a stage status."""
  _, style_name, label = STATUS_PRESENTATION.get(
      str(status).upper(), ("pending", "pending", str(status).lower())
  )
  return theme.markup(label, style_name)


def format_duration(seconds: Optional[float]) -> str:
  """Formats a duration as a compact human string.

  Examples: ``"-"``, ``"9s"``, ``"3m 05s"``, ``"2h 07m"``, ``"1d 03h"``.

  Args:
    seconds: Duration in seconds, or ``None``.

  Returns:
    The formatted duration, or ``"-"`` when the input is missing/invalid.
  """
  if seconds is None:
    return "-"
  try:
    seconds = float(seconds)
  except (TypeError, ValueError):
    return "-"
  if math.isnan(seconds) or math.isinf(seconds):
    return "-"
  if seconds < 0:
    seconds = 0.0
  total = int(round(seconds))
  if total < 60:
    return f"{total}s"
  minutes, secs = divmod(total, 60)
  if minutes < 60:
    return f"{minutes}m {secs:02d}s"
  hours, minutes = divmod(minutes, 60)
  if hours < 24:
    return f"{hours}h {minutes:02d}m"
  days, hours = divmod(hours, 24)
  return f"{days}d {hours:02d}h"


def format_metric(value: Optional[float], width: int = 0) -> str:
  """Formats a metric value with a readable, stable number of digits.

  Args:
    value: Metric value of any scalar type.
    width: Optional right-justification width.

  Returns:
    A display string such as ``"0.3120"``, ``"1.25e-04"`` or ``"-"``.
  """
  if value is None:
    return "-"
  if isinstance(value, bool):
    return str(value)
  if isinstance(value, int):
    text = str(value)
  else:
    try:
      number = float(value)
    except (TypeError, ValueError):
      return str(value)
    if math.isnan(number):
      return "nan"
    if math.isinf(number):
      return "inf" if number > 0 else "-inf"
    magnitude = abs(number)
    if magnitude != 0 and (magnitude < 1e-3 or magnitude >= 1e6):
      text = f"{number:.3e}"
    elif magnitude >= 100:
      text = f"{number:.2f}"
    else:
      text = f"{number:.4f}"
  if width:
    return text.rjust(width)
  return text


def format_count(done: Optional[int], total: Optional[int]) -> str:
  """Formats a ``done/total`` trial counter with aligned zero padding."""
  if total is None or total <= 0:
    return "-" if done is None else str(done)
  done = 0 if done is None else max(0, min(int(done), int(total)))
  pad = len(str(int(total)))
  return f"{done:0{pad}d}/{int(total)}"


def progress_bar(
    done: Optional[int],
    total: Optional[int],
    width: int,
    theme: Theme,
    style_name: str = "accent",
) -> str:
  """Renders a text progress bar with rich markup.

  Args:
    done: Completed units (clamped to ``[0, total]``).
    total: Total units. Non-positive totals render an empty bar.
    width: Bar width in characters (minimum 1).
    theme: Active theme supplying glyphs and colors.
    style_name: Semantic style used for the filled portion.

  Returns:
    A string possibly containing rich markup.
  """
  width = max(1, int(width))
  if not total or total <= 0:
    return theme.markup(theme.glyphs.bar_empty * width, "muted")
  done = max(0, min(int(done or 0), int(total)))
  filled = int(round(width * done / float(total)))
  filled = max(0, min(width, filled))
  if 0 < done < total and filled == 0:
    filled = 1
  if done < total and filled == width:
    filled = width - 1
  bar = theme.markup(theme.glyphs.bar_full * filled, style_name)
  empty = theme.markup(theme.glyphs.bar_empty * (width - filled), "muted")
  return f"{bar}{empty}"


def sparkline(values: Sequence[float], theme: Theme, width: int = 0) -> str:
  """Renders a compact sparkline for a metric history.

  Args:
    values: Metric samples in chronological order.
    theme: Active theme (decides the character ramp).
    width: If positive, keeps only the last ``width`` samples.

  Returns:
    A sparkline string, or an empty string when there is nothing to plot.
  """
  numbers: List[float] = []
  for value in values or []:
    try:
      number = float(value)
    except (TypeError, ValueError):
      continue
    if math.isnan(number) or math.isinf(number):
      continue
    numbers.append(number)
  if not numbers:
    return ""
  if width and width > 0:
    numbers = numbers[-width:]
  ramp = theme.glyphs.spark
  low, high = min(numbers), max(numbers)
  if high - low < 1e-12:
    return ramp[len(ramp) // 2] * len(numbers)
  span = high - low
  chars = []
  for number in numbers:
    index = int((number - low) / span * (len(ramp) - 1))
    chars.append(ramp[max(0, min(len(ramp) - 1, index))])
  return "".join(chars)


def truncate(text: str, max_width: int, placeholder: str = "…") -> str:
  """Truncates ``text`` to ``max_width`` characters, keeping the tail readable.

  Repo ids such as ``leobianco/npov_SFT_gemma-4-E2B-it_lr0.003`` are truncated
  in the middle so both the owner and the distinguishing suffix stay visible.

  Args:
    text: Text to shorten.
    max_width: Maximum number of characters allowed.
    placeholder: Marker inserted where characters were removed.

  Returns:
    The shortened text.
  """
  if max_width <= 0:
    return ""
  text = str(text)
  if len(text) <= max_width:
    return text
  if max_width <= len(placeholder):
    return text[:max_width]
  keep = max_width - len(placeholder)
  head = keep // 2
  tail = keep - head
  if tail == 0:
    return text[:head] + placeholder
  return text[:head] + placeholder + text[-tail:]


def escape_markup(text: str) -> str:
  """Escapes ``[`` so arbitrary text is never parsed as a style tag.

  Log lines legitimately contain things like ``[DRY-RUN]`` or ``[RESUME]``;
  without escaping they would either be interpreted by rich as (unknown)
  styles or silently removed by :func:`strip_markup`.
  """
  return str(text).replace("\\", "\\\\").replace("[", "\\[")


# A style tag looks like ``[bold red]`` or ``[/bold red]``: only these
# characters may appear inside, which keeps ``[DRY-RUN]`` from matching.
_TAG_BODY_CHARS = set(
    "abcdefghijklmnopqrstuvwxyz0123456789 _#/.,()-"
)


def _is_style_tag(body: str) -> bool:
  """Heuristic: does ``body`` look like a rich style tag rather than text?

  Style tags are lowercase style names, optionally closing (``/bold``), so
  ``[DRY-RUN]`` (uppercase) and ``[1, 2]`` (starts with a digit) are treated
  as literal text and preserved.
  """
  if not body:
    return False
  if not all(char in _TAG_BODY_CHARS for char in body):
    return False
  first = body[0]
  if not (first.isalpha() or first == "/"):
    return False
  return any(char.isalpha() for char in body)


def strip_markup(text: str) -> str:
  r"""Removes rich style tags from ``text`` and unescapes ``\[`` sequences.

  Args:
    text: Text possibly containing markup.

  Returns:
    The literal text a user should see.
  """
  source = str(text)
  out: List[str] = []
  index = 0
  length = len(source)
  while index < length:
    char = source[index]
    if char == "\\" and index + 1 < length and source[index + 1] == "[":
      out.append("[")
      index += 2
      continue
    if char == "[":
      close = source.find("]", index + 1)
      body = source[index + 1 : close] if close != -1 else ""
      if close != -1 and _is_style_tag(body):
        index = close + 1
        continue
    out.append(char)
    index += 1
  return "".join(out)


def plain(text: str, theme: Optional[Theme] = None) -> str:
  """Returns ``text`` without markup when colors are disabled.

  Args:
    text: Markup string.
    theme: Active theme; ``None`` forces stripping.

  Returns:
    Either the original markup or its plain-text rendering.
  """
  if theme is not None and theme.use_color:
    return text
  return strip_markup(text)


#: The only order in which the stages can legally execute: each sweep
#: consumes the checkpoint produced by the previous one.
PIPELINE_ORDER: List[str] = list(STAGE_TITLES)


def iter_stage_names(stages: Iterable[str]) -> List[str]:
  """Normalizes, de-duplicates and canonically orders a stage list.

  Ordering matters: the engine executes ``config.stages`` sequentially and
  PE-RL consumes the checkpoints produced by SFT and RM. A UI that returns
  stages in click order (e.g. a checkbox answered ``"3, 1"``) would otherwise
  produce an unrunnable campaign.

  Args:
    stages: Raw stage identifiers (possibly padded, duplicated or ``None``).

  Returns:
    Lowercase, de-duplicated stage names in pipeline order. Unknown names are
    preserved after the known ones, in their original relative order.
  """
  seen = set()
  ordered: List[str] = []
  for stage in stages or []:
    if not isinstance(stage, str):
      continue
    key = stage.strip().lower()
    if key and key not in seen:
      seen.add(key)
      ordered.append(key)
  known = [name for name in PIPELINE_ORDER if name in seen]
  unknown = [name for name in ordered if name not in PIPELINE_ORDER]
  return known + unknown

