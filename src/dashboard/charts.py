"""Dependency-free SVG charts for the dashboard's evaluation section.

The dashboard is a static site published to a Hugging Face Space with
``sdk: static``: there is no server and no build step on the other end, and
the page must also open from ``file://``. A plotting library would mean
either a JavaScript bundle to inline into every page or a matplotlib
dependency to rasterise at build time. Neither pays for itself for three
chart shapes, so the charts are emitted as inline SVG strings here.

Every dynamic string goes through :func:`html.escape`; the series names and
category labels originate in campaign state files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import html
import math
from typing import List, Optional, Sequence, Tuple

#: Colours keyed by policy family, chosen to stay distinguishable for the
#: common forms of colour blindness. Anything past the named families cycles
#: through :data:`_FALLBACK_COLORS`.
_NAMED_COLORS = {
    "sft": "#9ca3af",
    "perl": "#2563eb",
    "perl:organic": "#2563eb",
    "perl:synthetic_struct": "#d97706",
}
_FALLBACK_COLORS = ("#059669", "#db2777", "#7c3aed", "#0891b2")

_WIDTH = 640
_HEIGHT = 300
_MARGIN_LEFT = 56
_MARGIN_RIGHT = 16
_MARGIN_TOP = 34
_MARGIN_BOTTOM = 58

Interval = Tuple[float, float]


@dataclass
class BarSeries:
  """One policy's values across the chart's categories.

  Attributes:
    name: Legend label.
    color: CSS colour.
    values: One value per category; None leaves a gap.
    intervals: Optional ``(lo, hi)`` per category, drawn as error bars.
    marked: Categories to flag with a star, e.g. the rollout temperature.
  """

  name: str
  color: str
  values: List[Optional[float]]
  intervals: List[Optional[Interval]] = field(default_factory=list)
  marked: List[bool] = field(default_factory=list)


@dataclass
class TrajectoryPoint:
  """One point of a connected trajectory, with optional 2-D error bars."""

  x: float
  y: float
  label: str
  x_interval: Optional[Interval] = None
  y_interval: Optional[Interval] = None
  marked: bool = False


@dataclass
class TrajectorySeries:
  """One policy's points, connected in the given order."""

  name: str
  color: str
  points: List[TrajectoryPoint]


def series_color(base_label: str, index: int) -> str:
  """Returns a stable colour for a policy.

  Args:
    base_label: Target label without temperature, e.g. ``perl:organic``.
    index: Position of the series, used for families without a named colour.

  Returns:
    A CSS colour string.
  """
  if base_label in _NAMED_COLORS:
    return _NAMED_COLORS[base_label]
  return _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]


def _e(text: str) -> str:
  return html.escape(str(text), quote=True)


def _finite(value: Optional[float]) -> bool:
  return isinstance(value, (int, float)) and math.isfinite(value)


def _nice_step(span: float, target_ticks: int = 5) -> float:
  """Returns a 1/2/5 x 10^k tick step covering ``span`` in ~target ticks."""
  if span <= 0 or not math.isfinite(span):
    return 1.0
  raw = span / target_ticks
  magnitude = 10 ** math.floor(math.log10(raw))
  for multiplier in (1, 2, 5, 10):
    if raw <= multiplier * magnitude:
      return multiplier * magnitude
  return 10 * magnitude


def _axis_range(
    values: Sequence[float], include_zero: bool
) -> Tuple[float, float, float]:
  """Computes a padded, tick-aligned ``(lo, hi, step)`` for the values."""
  finite = [v for v in values if _finite(v)]
  if not finite:
    return 0.0, 1.0, 0.25
  lo, hi = min(finite), max(finite)
  if include_zero:
    lo, hi = min(lo, 0.0), max(hi, 0.0)
  if hi == lo:
    pad = abs(hi) * 0.1 or 1.0
    lo, hi = lo - pad, hi + pad
  step = _nice_step(hi - lo)
  lo = math.floor(lo / step) * step
  hi = math.ceil(hi / step) * step
  return lo, hi, step


def _format_tick(value: float, percent: bool, step: float) -> str:
  if percent:
    decimals = 0 if step * 100 >= 1 else 1
    return f"{value * 100:.{decimals}f}%"
  if step >= 1:
    return f"{value:.0f}"
  decimals = max(0, -int(math.floor(math.log10(step))))
  return f"{value:.{decimals}f}"


def _ticks(lo: float, hi: float, step: float) -> List[float]:
  count = int(round((hi - lo) / step))
  return [lo + i * step for i in range(count + 1)]


def _frame(title: str, body: str, legend: str, aria: str) -> str:
  return (
      f'<figure class="chart"><svg viewBox="0 0 {_WIDTH} {_HEIGHT}" '
      f'role="img" aria-label="{_e(aria)}" preserveAspectRatio="xMidYMid meet">'
      f'<text class="chart-title" x="{_WIDTH / 2:.1f}" y="18" '
      f'text-anchor="middle">{_e(title)}</text>{body}</svg>'
      f"{legend}</figure>"
  )


def _legend(entries: Sequence[Tuple[str, str]], note: str = "") -> str:
  items = "".join(
      f'<span class="legend-item"><span class="swatch" '
      f'style="background:{_e(color)}"></span>{_e(name)}</span>'
      for name, color in entries
  )
  note_html = f'<span class="legend-note">{_e(note)}</span>' if note else ""
  return f'<figcaption class="chart-legend">{items}{note_html}</figcaption>'


def _y_axis(
    lo: float, hi: float, step: float, percent: bool, label: str, y_of
) -> str:
  parts = []
  plot_right = _WIDTH - _MARGIN_RIGHT
  for tick in _ticks(lo, hi, step):
    y = y_of(tick)
    parts.append(
        f'<line class="grid" x1="{_MARGIN_LEFT}" x2="{plot_right}" '
        f'y1="{y:.1f}" y2="{y:.1f}"/>'
        f'<text class="tick" x="{_MARGIN_LEFT - 6}" y="{y + 4:.1f}" '
        f'text-anchor="end">{_e(_format_tick(tick, percent, step))}</text>'
    )
  mid = (_MARGIN_TOP + _HEIGHT - _MARGIN_BOTTOM) / 2
  parts.append(
      f'<text class="axis-label" x="14" y="{mid:.1f}" text-anchor="middle" '
      f'transform="rotate(-90 14 {mid:.1f})">{_e(label)}</text>'
  )
  return "".join(parts)


def grouped_bar_chart(
    title: str,
    categories: Sequence[str],
    series: Sequence[BarSeries],
    y_label: str,
    percent: bool = False,
    x_label: str = "Decoding temperature",
    marker_note: str = "",
) -> str:
  """Renders side-by-side bars per category, with optional error bars.

  Args:
    title: Chart title.
    categories: Category labels along the x axis, e.g. temperatures.
    series: One entry per policy; each holds one value per category.
    y_label: Y-axis label.
    percent: Whether to format ticks as percentages of 1.
    x_label: X-axis label.
    marker_note: Legend text explaining the star, if any series is marked.

  Returns:
    An HTML ``<figure>`` with an inline SVG, or an empty string when there
    is nothing to draw.
  """
  all_values: List[float] = []
  for s in series:
    all_values.extend(v for v in s.values if _finite(v))
    for interval in s.intervals:
      if interval is not None:
        all_values.extend(v for v in interval if _finite(v))
  if not categories or not series or not all_values:
    return ""

  lo, hi, step = _axis_range(all_values, include_zero=True)
  plot_left, plot_right = _MARGIN_LEFT, _WIDTH - _MARGIN_RIGHT
  plot_top, plot_bottom = _MARGIN_TOP, _HEIGHT - _MARGIN_BOTTOM

  def y_of(value: float) -> float:
    return plot_bottom - (value - lo) / (hi - lo) * (plot_bottom - plot_top)

  group_width = (plot_right - plot_left) / len(categories)
  bar_width = group_width * 0.8 / len(series)
  parts = [_y_axis(lo, hi, step, percent, y_label, y_of)]
  zero = y_of(0.0)
  parts.append(
      f'<line class="axis" x1="{plot_left}" x2="{plot_right}" '
      f'y1="{zero:.1f}" y2="{zero:.1f}"/>'
  )
  any_marked = False
  for c, category in enumerate(categories):
    group_left = plot_left + c * group_width + group_width * 0.1
    center = plot_left + (c + 0.5) * group_width
    parts.append(
        f'<text class="tick" x="{center:.1f}" y="{plot_bottom + 16}" '
        f'text-anchor="middle">{_e(category)}</text>'
    )
    for s_index, s in enumerate(series):
      value = s.values[c] if c < len(s.values) else None
      if not _finite(value):
        continue
      x = group_left + s_index * bar_width
      top, bottom = sorted((y_of(value), zero))
      tooltip = f"{s.name} @ {category}: {_format_value(value, percent)}"
      interval = s.intervals[c] if c < len(s.intervals) else None
      if interval is not None:
        tooltip += (
            f" [{_format_value(interval[0], percent)}, "
            f"{_format_value(interval[1], percent)}]"
        )
      parts.append(
          f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar_width * 0.92:.1f}" '
          f'height="{max(bottom - top, 0.5):.1f}" fill="{_e(s.color)}">'
          f"<title>{_e(tooltip)}</title></rect>"
      )
      bar_center = x + bar_width * 0.46
      if interval is not None and all(_finite(v) for v in interval):
        y_lo, y_hi = y_of(interval[0]), y_of(interval[1])
        cap = bar_width * 0.2
        parts.append(
            f'<path class="errbar" d="M{bar_center:.1f},{y_lo:.1f}'
            f"V{y_hi:.1f}M{bar_center - cap:.1f},{y_lo:.1f}"
            f"H{bar_center + cap:.1f}M{bar_center - cap:.1f},{y_hi:.1f}"
            f'H{bar_center + cap:.1f}"/>'
        )
      if c < len(s.marked) and s.marked[c]:
        any_marked = True
        star_y = min(top, y_of(interval[1]) if interval else top) - 4
        parts.append(
            f'<text class="marker" x="{bar_center:.1f}" y="{star_y:.1f}" '
            f'text-anchor="middle">\u2605</text>'
        )
  parts.append(
      f'<text class="axis-label" x="{(plot_left + plot_right) / 2:.1f}" '
      f'y="{_HEIGHT - 20}" text-anchor="middle">{_e(x_label)}</text>'
  )
  legend = _legend(
      [(s.name, s.color) for s in series],
      marker_note if any_marked else "",
  )
  return _frame(title, "".join(parts), legend, title)


def trajectory_plot(
    title: str,
    series: Sequence[TrajectorySeries],
    x_label: str,
    y_label: str,
    x_percent: bool = False,
    y_percent: bool = False,
    marker_note: str = "",
) -> str:
  """Renders connected scatter trajectories, one per policy.

  Used for the quality-versus-faithfulness frontier: each policy traces the
  path its operating point takes as the decoding temperature rises.

  Args:
    title: Chart title.
    series: One trajectory per policy, points in connection order.
    x_label: X-axis label.
    y_label: Y-axis label.
    x_percent: Whether to format x ticks as percentages of 1.
    y_percent: Whether to format y ticks as percentages of 1.
    marker_note: Legend text explaining the ring drawn around marked points.

  Returns:
    An HTML ``<figure>`` with an inline SVG, or an empty string when no
    series has a point.
  """
  xs: List[float] = []
  ys: List[float] = []
  for s in series:
    for p in s.points:
      xs.append(p.x)
      ys.append(p.y)
      if p.x_interval:
        xs.extend(p.x_interval)
      if p.y_interval:
        ys.extend(p.y_interval)
  xs = [v for v in xs if _finite(v)]
  ys = [v for v in ys if _finite(v)]
  if not xs or not ys:
    return ""

  x_lo, x_hi, x_step = _axis_range(xs, include_zero=False)
  y_lo, y_hi, y_step = _axis_range(ys, include_zero=False)
  plot_left, plot_right = _MARGIN_LEFT, _WIDTH - _MARGIN_RIGHT
  plot_top, plot_bottom = _MARGIN_TOP, _HEIGHT - _MARGIN_BOTTOM

  def x_of(value: float) -> float:
    return plot_left + (value - x_lo) / (x_hi - x_lo) * (plot_right - plot_left)

  def y_of(value: float) -> float:
    return plot_bottom - (value - y_lo) / (y_hi - y_lo) * (
        plot_bottom - plot_top
    )

  parts = [_y_axis(y_lo, y_hi, y_step, y_percent, y_label, y_of)]
  for tick in _ticks(x_lo, x_hi, x_step):
    x = x_of(tick)
    parts.append(
        f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{plot_top}" '
        f'y2="{plot_bottom}"/><text class="tick" x="{x:.1f}" '
        f'y="{plot_bottom + 16}" text-anchor="middle">'
        f"{_e(_format_tick(tick, x_percent, x_step))}</text>"
    )
  parts.append(
      f'<line class="axis" x1="{plot_left}" x2="{plot_right}" '
      f'y1="{plot_bottom}" y2="{plot_bottom}"/>'
  )
  any_marked = False
  for s in series:
    points = [p for p in s.points if _finite(p.x) and _finite(p.y)]
    if not points:
      continue
    color = _e(s.color)
    if len(points) > 1:
      path = " ".join(
          f"{'M' if i == 0 else 'L'}{x_of(p.x):.1f},{y_of(p.y):.1f}"
          for i, p in enumerate(points)
      )
      parts.append(
          f'<path class="trajectory" d="{path}" stroke="{color}" fill="none"/>'
      )
    for p in points:
      cx, cy = x_of(p.x), y_of(p.y)
      if p.x_interval and all(_finite(v) for v in p.x_interval):
        parts.append(
            f'<path class="errbar" stroke="{color}" '
            f'd="M{x_of(p.x_interval[0]):.1f},{cy:.1f}'
            f'H{x_of(p.x_interval[1]):.1f}"/>'
        )
      if p.y_interval and all(_finite(v) for v in p.y_interval):
        parts.append(
            f'<path class="errbar" stroke="{color}" '
            f'd="M{cx:.1f},{y_of(p.y_interval[0]):.1f}'
            f'V{y_of(p.y_interval[1]):.1f}"/>'
        )
      tooltip = (
          f"{s.name} @ T={p.label}: "
          f"{_format_value(p.x, x_percent)}, {_format_value(p.y, y_percent)}"
      )
      parts.append(
          f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{color}">'
          f"<title>{_e(tooltip)}</title></circle>"
      )
      if p.marked:
        any_marked = True
        parts.append(
            f'<circle class="ring" cx="{cx:.1f}" cy="{cy:.1f}" r="8" '
            f'stroke="{color}" fill="none"/>'
        )
      parts.append(
          f'<text class="point-label" x="{cx + 7:.1f}" y="{cy - 7:.1f}">'
          f"{_e(p.label)}</text>"
      )
  parts.append(
      f'<text class="axis-label" x="{(plot_left + plot_right) / 2:.1f}" '
      f'y="{_HEIGHT - 20}" text-anchor="middle">{_e(x_label)}</text>'
  )
  legend = _legend(
      [(s.name, s.color) for s in series],
      marker_note if any_marked else "",
  )
  return _frame(title, "".join(parts), legend, title)


def _format_value(value: float, percent: bool) -> str:
  if percent:
    return f"{value * 100:.1f}%"
  return f"{value:.3g}"
