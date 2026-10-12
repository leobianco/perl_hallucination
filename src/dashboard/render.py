"""Markdown rendering for campaign reports.

The reports the orchestrator writes are table-heavy GitHub-flavoured markdown
with the occasional GitHub alert (``> [!CAUTION]``, raised when the autorater
is too weak to trust). Both are handled here; nothing else in the site needs a
markdown renderer.

``markdown-it-py`` is already pinned in ``requirements.txt`` (it arrives with
``rich``), but it is imported *optionally* so the dashboard behaves like the
rest of this repository: degraded but working when dependencies are missing.
Without it, a report is shown verbatim in a ``<pre>`` block rather than
failing the build.

Raw HTML in the source markdown is **not** rendered. The reports contain none
today, and the published site is a remote artifact, so escaping is the
default rather than a reaction to a future incident.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import re
from typing import List, Optional, Tuple

try:
  from markdown_it import MarkdownIt
except ImportError:  # pragma: no cover - exercised by the degraded-path test.
  MarkdownIt = None  # type: ignore[assignment]

#: GitHub alert kinds, mapped to the CSS class used to colour them.
_ALERT_KINDS = {
    "NOTE": "note",
    "TIP": "tip",
    "IMPORTANT": "important",
    "WARNING": "warning",
    "CAUTION": "caution",
}

_ALERT_RE = re.compile(
    r"<blockquote>\s*<p>\s*\[!(?P<kind>[A-Z]+)\]\s*(?:<br\s*/?>)?\s*",
    re.IGNORECASE,
)

_HEADING_RE = re.compile(
    r"<h(?P<level>[1-6])>(?P<text>.*?)</h(?P=level)>", re.DOTALL
)

_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Heading:
  """One entry of a rendered report's table of contents."""

  level: int
  text: str
  anchor: str


@dataclass(frozen=True)
class RenderedMarkdown:
  """A rendered report plus the navigation extracted from it."""

  html: str
  headings: List[Heading]
  degraded: bool = False

  @property
  def is_empty(self) -> bool:
    """Whether there was nothing to render."""
    return not self.html.strip()


def _slugify(text: str) -> str:
  """Turns heading text into a URL fragment.

  Args:
    text: Heading text, possibly containing inline markup.

  Returns:
    A lowercase, hyphen-separated anchor.
  """
  plain = _TAG_RE.sub("", text)
  plain = html.unescape(plain)
  slug = re.sub(r"[^a-zA-Z0-9\s-]", "", plain).strip().lower()
  slug = re.sub(r"[\s-]+", "-", slug)
  return slug or "section"


def _promote_alerts(rendered: str) -> str:
  """Rewrites GitHub alert blockquotes into styled callouts.

  ``markdown-it`` has no notion of ``> [!CAUTION]``; without this the single
  most important block a report can contain - the warning that the autorater
  was too weak for its numbers to be trusted - renders as an ordinary quote.

  Args:
    rendered: HTML produced by the markdown renderer.

  Returns:
    HTML with alert blockquotes replaced by ``<div class="alert alert-*">``.
  """
  out = []
  cursor = 0
  for match in _ALERT_RE.finditer(rendered):
    kind = match.group("kind").upper()
    if kind not in _ALERT_KINDS:
      continue
    closing = rendered.find("</blockquote>", match.end())
    if closing == -1:
      continue
    body = rendered[match.end() : closing]
    css = _ALERT_KINDS[kind]
    out.append(rendered[cursor : match.start()])
    out.append(
        f'<div class="alert alert-{css}">'
        f'<div class="alert-title">{kind.title()}</div>'
        f"<p>{body}"
        "</div>"
    )
    cursor = closing + len("</blockquote>")
  out.append(rendered[cursor:])
  return "".join(out)


def _anchor_headings(rendered: str) -> Tuple[str, List[Heading]]:
  """Adds ids to headings and collects them into a table of contents.

  Args:
    rendered: HTML produced by the markdown renderer.

  Returns:
    A ``(html, headings)`` tuple. Duplicate slugs are suffixed so that every
    anchor in a long report remains unique and linkable.
  """
  headings: List[Heading] = []
  seen = {}

  def _replace(match: "re.Match[str]") -> str:
    level = int(match.group("level"))
    text = match.group("text")
    anchor = _slugify(text)
    if anchor in seen:
      seen[anchor] += 1
      anchor = f"{anchor}-{seen[anchor]}"
    else:
      seen[anchor] = 1
    headings.append(
        Heading(level=level, text=html.unescape(_TAG_RE.sub("", text)).strip(),
                anchor=anchor)
    )
    return f'<h{level} id="{anchor}">{text}</h{level}>'

  return _HEADING_RE.sub(_replace, rendered), headings


def render_markdown(text: Optional[str]) -> RenderedMarkdown:
  """Renders report markdown to HTML.

  Args:
    text: The markdown source, or None.

  Returns:
    The rendered HTML, its headings, and whether the degraded (no
    ``markdown-it-py``) path was taken.
  """
  if not text:
    return RenderedMarkdown(html="", headings=[])

  if MarkdownIt is None:
    escaped = html.escape(text)
    return RenderedMarkdown(
        html=f'<pre class="raw-markdown">{escaped}</pre>',
        headings=[],
        degraded=True,
    )

  # `linkify` is deliberately off: it needs the separate `linkify-it-py`
  # package, which is not in this repository's lockfile, and markdown-it
  # raises at render time rather than degrading when it is missing. The
  # reports write their links explicitly anyway.
  parser = MarkdownIt("gfm-like", {"html": False, "linkify": False})
  rendered = parser.render(text)
  rendered = _promote_alerts(rendered)
  rendered, headings = _anchor_headings(rendered)
  return RenderedMarkdown(html=rendered, headings=headings)


def render_report_file(path: str) -> RenderedMarkdown:
  """Reads and renders a markdown report from disk.

  Args:
    path: Path to the report file.

  Returns:
    The rendered report, or an empty result when the file cannot be read. A
    report that vanished between indexing and building must not fail a build
    that is otherwise fine.
  """
  try:
    with open(path, "r", encoding="utf-8") as handle:
      return render_markdown(handle.read())
  except OSError:
    return RenderedMarkdown(html="", headings=[])
