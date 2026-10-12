"""Tests for markdown rendering of campaign reports.

The assertions that need a real markdown parser are skipped when
``markdown-it-py`` is absent, so this suite passes both on the bare
interpreter and in the environment the VM actually builds with. The degraded
path itself is tested unconditionally, because it is the one that runs before
``pip install -r requirements.txt`` has finished.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

from src.dashboard import render as render_mod

_HAS_MARKDOWN = render_mod.MarkdownIt is not None
_requires_markdown = unittest.skipUnless(
    _HAS_MARKDOWN, "markdown-it-py is not installed"
)


class EmptyInputTest(unittest.TestCase):
  """Nothing in, nothing out."""

  def test_none_renders_empty(self):
    result = render_mod.render_markdown(None)
    self.assertTrue(result.is_empty)
    self.assertEqual(result.headings, [])

  def test_blank_renders_empty(self):
    self.assertTrue(render_mod.render_markdown("").is_empty)


class DegradedPathTest(unittest.TestCase):
  """Behaviour when markdown-it-py is missing."""

  def test_markdown_is_escaped_into_a_pre_block(self):
    with mock.patch.object(render_mod, "MarkdownIt", None):
      result = render_mod.render_markdown("# Title\n<script>x</script>")
    self.assertTrue(result.degraded)
    self.assertIn("raw-markdown", result.html)
    self.assertNotIn("<script>", result.html)
    self.assertIn("&lt;script&gt;", result.html)


@_requires_markdown
class MarkdownFeatureTest(unittest.TestCase):
  """The constructs the orchestrator's reports actually use."""

  def test_tables_are_rendered(self):
    source = "| A | B |\n| :--- | ---: |\n| 1 | 2 |\n"
    html = render_mod.render_markdown(source).html
    self.assertIn("<table>", html)
    self.assertIn("<td", html)

  def test_raw_html_in_the_source_is_not_executed(self):
    html = render_mod.render_markdown("<script>alert(1)</script>").html
    self.assertNotIn("<script>", html)

  def test_code_and_emphasis_survive(self):
    html = render_mod.render_markdown("`eval/loss` is **low**").html
    self.assertIn("<code>eval/loss</code>", html)
    self.assertIn("<strong>low</strong>", html)

  def test_links_are_preserved(self):
    html = render_mod.render_markdown("[hf](https://huggingface.co/x)").html
    self.assertIn('href="https://huggingface.co/x"', html)


@_requires_markdown
class AlertTest(unittest.TestCase):
  """GitHub alerts, which no markdown parser handles natively."""

  def test_caution_becomes_a_styled_callout(self):
    source = "> [!CAUTION]\n> The judge is too weak to trust.\n"
    html = render_mod.render_markdown(source).html
    self.assertIn('class="alert alert-caution"', html)
    self.assertIn("Caution", html)
    self.assertIn("too weak to trust", html)
    self.assertNotIn("[!CAUTION]", html)

  def test_every_supported_kind(self):
    for kind in ("NOTE", "TIP", "IMPORTANT", "WARNING", "CAUTION"):
      html = render_mod.render_markdown(f"> [!{kind}]\n> body\n").html
      self.assertIn(f"alert-{kind.lower()}", html)

  def test_unknown_kind_stays_a_blockquote(self):
    html = render_mod.render_markdown("> [!BOGUS]\n> body\n").html
    self.assertIn("<blockquote>", html)
    self.assertNotIn("alert-bogus", html)

  def test_ordinary_blockquotes_are_untouched(self):
    html = render_mod.render_markdown("> **Summary**\n> line two\n").html
    self.assertIn("<blockquote>", html)
    self.assertNotIn('class="alert', html)

  def test_multiple_alerts_in_one_document(self):
    source = "> [!NOTE]\n> one\n\ntext\n\n> [!WARNING]\n> two\n"
    html = render_mod.render_markdown(source).html
    self.assertIn("alert-note", html)
    self.assertIn("alert-warning", html)
    self.assertIn("text", html)


@_requires_markdown
class HeadingTest(unittest.TestCase):
  """Anchors and the table of contents."""

  def test_headings_get_ids(self):
    result = render_mod.render_markdown("## Stage Breakdown\n")
    self.assertIn('id="stage-breakdown"', result.html)
    self.assertEqual(result.headings[0].anchor, "stage-breakdown")
    self.assertEqual(result.headings[0].level, 2)

  def test_duplicate_titles_get_distinct_anchors(self):
    result = render_mod.render_markdown("## Same\n\n## Same\n")
    anchors = [heading.anchor for heading in result.headings]
    self.assertEqual(anchors, ["same", "same-2"])

  def test_heading_text_is_stripped_of_markup(self):
    result = render_mod.render_markdown("## 2.1. `eval/loss` results\n")
    self.assertEqual(result.headings[0].text, "2.1. eval/loss results")

  def test_punctuation_is_dropped_from_anchors(self):
    result = render_mod.render_markdown("## 2.2.1 Reward Model (organic)\n")
    self.assertEqual(result.headings[0].anchor, "221-reward-model-organic")


class ReportFileTest(unittest.TestCase):
  """Reading reports off disk."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

  def test_missing_file_renders_empty_instead_of_raising(self):
    result = render_mod.render_report_file(
        os.path.join(self.root, "absent.md")
    )
    self.assertTrue(result.is_empty)

  def test_existing_file_is_rendered(self):
    path = os.path.join(self.root, "report.md")
    with open(path, "w", encoding="utf-8") as handle:
      handle.write("# Campaign\n\nbody\n")
    result = render_mod.render_report_file(path)
    self.assertFalse(result.is_empty)
    self.assertIn("body", result.html)


if __name__ == "__main__":
  unittest.main()
