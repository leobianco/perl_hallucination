"""Thin console abstraction that works with or without the ``rich`` package.

Every user-facing print in the orchestrator goes through :class:`UiConsole`.
When ``rich`` is installed the output is styled, width-aware and resizable;
otherwise the very same markup strings are stripped and printed with the
standard library, so piping the CLI into a log file stays perfectly readable.
"""

from __future__ import annotations

import shutil
import sys
from typing import Any, Iterable, List, Optional, TextIO

from src.orchestrator.cli import theme as theme_mod
from src.orchestrator.cli.theme import Theme


class UiConsole:
  """Unified output surface for the orchestrator CLI."""

  def __init__(
      self,
      theme: Optional[Theme] = None,
      file: Optional[TextIO] = None,
      force_plain: bool = False,
      width: Optional[int] = None,
  ):
    self.file = file or sys.stdout
    self.theme = theme or theme_mod.detect_theme(stream=self.file)
    self._forced_width = width
    self._rich_console = None
    if not force_plain:
      self._rich_console = self._make_rich_console()

  def _make_rich_console(self):
    try:
      from rich.console import Console  # pylint: disable=g-import-not-at-top
    except ImportError:
      return None
    kwargs = {
        "file": self.file,
        "no_color": not self.theme.use_color,
        "soft_wrap": False,
        "highlight": False,
        "emoji": False,
    }
    if self._forced_width:
      kwargs["width"] = self._forced_width
    try:
      return Console(**kwargs)
    except Exception:  # pylint: disable=broad-except
      return None

  # --- Capabilities ---------------------------------------------------
  @property
  def is_rich(self) -> bool:
    """True when styled rendering is available."""
    return self._rich_console is not None

  @property
  def rich(self):
    """The underlying ``rich.console.Console`` (may be ``None``)."""
    return self._rich_console

  @property
  def width(self) -> int:
    """Current terminal width, re-read on every access to survive resizes."""
    if self._forced_width:
      return self._forced_width
    if self._rich_console is not None:
      try:
        return max(40, self._rich_console.size.width)
      except Exception:  # pylint: disable=broad-except
        pass
    try:
      return max(40, shutil.get_terminal_size(fallback=(100, 30)).columns)
    except Exception:  # pylint: disable=broad-except
      return 100

  @property
  def height(self) -> int:
    """Current terminal height in rows."""
    if self._rich_console is not None:
      try:
        return max(10, self._rich_console.size.height)
      except Exception:  # pylint: disable=broad-except
        pass
    try:
      return max(10, shutil.get_terminal_size(fallback=(100, 30)).lines)
    except Exception:  # pylint: disable=broad-except
      return 30

  # --- Output ---------------------------------------------------------
  def print(self, markup: str = "") -> None:
    """Prints a markup string, degrading to plain text when needed."""
    if self._rich_console is not None:
      self._rich_console.print(markup, markup=True, highlight=False)
    else:
      print(theme_mod.strip_markup(markup), file=self.file)

  def print_lines(self, lines: Iterable[str]) -> None:
    """Prints a sequence of markup strings."""
    for line in lines:
      self.print(line)

  def print_renderable(self, renderable: Any, fallback_lines: Optional[List[str]] = None) -> None:
    """Prints a rich renderable, falling back to ``fallback_lines``."""
    if self._rich_console is not None and renderable is not None:
      self._rich_console.print(renderable)
      return
    for line in fallback_lines or []:
      self.print(line)

  def rule(self, title: str = "") -> None:
    """Draws a horizontal rule with an optional centered title."""
    if self._rich_console is not None:
      self._rich_console.rule(
          self.theme.markup(title, "accent") if title else "",
          style=self.theme.style("border") or "none",
      )
      return
    fill = "-" if not self.theme.use_unicode else "─"
    width = self.width
    if title:
      text = f" {theme_mod.strip_markup(title)} "
      pad = max(0, width - len(text))
      left = pad // 2
      print(fill * left + text + fill * (pad - left), file=self.file)
    else:
      print(fill * width, file=self.file)

  def blank(self) -> None:
    """Prints an empty line."""
    self.print("")

  def success(self, message: str) -> None:
    """Prints a success message with a check glyph."""
    self.print(
        f"{self.theme.markup(self.theme.glyphs.completed, 'success')} "
        f"{self.theme.markup(message, 'value')}"
    )

  def warn(self, message: str) -> None:
    """Prints a warning message."""
    self.print(
        f"{self.theme.markup('!', 'warning')} "
        f"{self.theme.markup(message, 'warning')}"
    )

  def error(self, message: str) -> None:
    """Prints an error message."""
    self.print(
        f"{self.theme.markup(self.theme.glyphs.failed, 'error')} "
        f"{self.theme.markup(message, 'error')}"
    )

  def info(self, message: str) -> None:
    """Prints an informational message."""
    self.print(
        f"{self.theme.markup(self.theme.glyphs.arrow, 'accent')} "
        f"{self.theme.markup(message, 'value')}"
    )

  def hint(self, message: str) -> None:
    """Prints a dimmed hint / next-step suggestion."""
    self.print(self.theme.markup(f"  {message}", "muted"))

  def panel(self, body_lines: List[str], title: str = "", style: str = "border") -> None:
    """Prints a bordered panel, or an indented block in plain mode."""
    if self._rich_console is not None:
      try:
        from rich.panel import Panel  # pylint: disable=g-import-not-at-top
        from rich.text import Text  # pylint: disable=g-import-not-at-top
        from rich import box  # pylint: disable=g-import-not-at-top

        body = Text.from_markup("\n".join(body_lines))
        self._rich_console.print(
            Panel(
                body,
                title=self.theme.markup(title, "heading") if title else None,
                border_style=self.theme.style(style) or "none",
                box=box.ROUNDED if self.theme.use_unicode else box.ASCII,
                padding=(0, 1),
            )
        )
        return
      except Exception:  # pylint: disable=broad-except
        pass
    if title:
      self.rule(title)
    for line in body_lines:
      self.print(line)
    if title:
      self.rule()
