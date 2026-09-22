"""Tests for static site emission."""

from __future__ import annotations

import datetime
import json
import os
import shutil
import tempfile
import unittest

from src.dashboard import build as build_mod
from src.dashboard import index as index_mod
from src.dashboard.tests import fixtures


def _read(path: str) -> str:
  """Reads a UTF-8 file.

  Args:
    path: File to read.

  Returns:
    Its contents.
  """
  with open(path, "r", encoding="utf-8") as handle:
    return handle.read()


class FormattingTest(unittest.TestCase):
  """The small formatters the tables depend on."""

  def test_durations(self):
    self.assertEqual(build_mod.format_duration(None), "\u2014")
    self.assertEqual(build_mod.format_duration(42), "42s")
    self.assertEqual(build_mod.format_duration(432), "7m12s")
    self.assertEqual(build_mod.format_duration(11040), "3h04m")

  def test_metrics(self):
    self.assertEqual(build_mod.format_metric(None), "\u2014")
    self.assertEqual(build_mod.format_metric(0.28500), "0.285")
    self.assertEqual(build_mod.format_metric(0.0), "0")
    self.assertEqual(build_mod.format_metric(True), "yes")

  def test_timestamps(self):
    self.assertEqual(
        build_mod.format_timestamp("2026-09-21T12:06:08.879030"),
        "2026-09-21 12:06",
    )
    self.assertEqual(build_mod.format_timestamp("nonsense"), "\u2014")


class BuildTest(unittest.TestCase):
  """A full build of a small repository."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.out = os.path.join(self.root, "site")
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
    fixtures.write_campaign(self.root)
    self.result = build_mod.build_site(root=self.root, out_dir=self.out)

  def test_emits_the_expected_files(self):
    for relative in (
        "index.html",
        "README.md",
        os.path.join("data", "index.json"),
        os.path.join("campaign", "npov-campaign-2609211206.html"),
        os.path.join("campaign", "npov-campaign-2609211206.md"),
    ):
      self.assertTrue(
          os.path.isfile(os.path.join(self.out, relative)), relative
      )

  def test_does_not_emit_a_static_directory(self):
    """The CSS and JS are inlined, so a copy on disk would be dead weight."""
    self.assertFalse(os.path.isdir(os.path.join(self.out, "static")))

  def test_reports_what_it_built(self):
    self.assertEqual(self.result.campaign_count, 1)
    self.assertEqual(self.result.page_count, 2)
    self.assertEqual(self.result.in_progress, 0)

  def test_space_readme_configures_a_static_space(self):
    readme = _read(os.path.join(self.out, "README.md"))
    self.assertIn("sdk: static", readme)
    self.assertIn("app_file: index.html", readme)

  def test_index_json_is_machine_readable(self):
    payload = json.loads(_read(os.path.join(self.out, "data", "index.json")))
    self.assertEqual(payload["campaign_count"], 1)
    self.assertIn("generated_at", payload)
    self.assertEqual(
        payload["campaigns"][0]["campaign_id"], "npov_campaign_2609211206"
    )

  def test_list_page_links_to_the_campaign(self):
    html = _read(os.path.join(self.out, "index.html"))
    self.assertIn('href="campaign/npov-campaign-2609211206.html"', html)
    self.assertIn("npov_campaign_2609211206", html)

  def test_list_page_offers_the_facets(self):
    html = _read(os.path.join(self.out, "index.html"))
    for control in ("f-task", "f-status", "f-model", "f-flavor", "f-search"):
      self.assertIn(f'id="{control}"', html)
    self.assertIn('id="f-completed"', html)
    self.assertIn('id="f-archived"', html)
    self.assertIn('id="f-real"', html)

  def test_rows_carry_the_filter_metadata(self):
    html = _read(os.path.join(self.out, "index.html"))
    self.assertIn('data-task="npov"', html)
    self.assertIn('data-status="COMPLETED"', html)
    self.assertIn('data-model="google/gemma-4-E4B-it"', html)
    self.assertIn('data-flavors="organic"', html)
    self.assertIn('data-archived="0"', html)
    self.assertIn('data-dryrun="0"', html)

  def test_list_columns(self):
    """Campaign, task, model, status, RM dataset - and nothing else.

    The metric columns were dropped deliberately: the list is for finding a
    campaign, and four-decimal floats in a narrow column helped nobody.
    """
    html = _read(os.path.join(self.out, "index.html"))
    header = html.split("<thead>")[1].split("</thead>")[0]
    self.assertIn("<th>Campaign</th>", header)
    self.assertIn("<th>Task</th>", header)
    self.assertIn("<th>Model</th>", header)
    self.assertIn("<th>Status</th>", header)
    self.assertIn("RM dataset", header)
    for dropped in ("Halluc", "vs SFT", "Elapsed", "Updated"):
      self.assertNotIn(dropped, header)
    # The model column sits immediately right of the task, as its own cell.
    self.assertIn(
        '<td>npov</td>\n<td class="model">google/gemma-4-E4B-it</td>', html
    )

  def test_a_real_campaign_is_not_flagged_as_a_rehearsal(self):
    listing = _read(os.path.join(self.out, "index.html"))
    self.assertNotIn('<span class="chip chip-dry-run">', listing)

  def test_campaign_page_has_every_section(self):
    html = _read(
        os.path.join(self.out, "campaign", "npov-campaign-2609211206.html")
    )
    for heading in ("Stages", "Evaluation", "Links", "Report", "Configuration"):
      self.assertIn(f"<h3>{heading}</h3>", html)

  def test_campaign_page_carries_the_outbound_links(self):
    html = _read(
        os.path.join(self.out, "campaign", "npov-campaign-2609211206.html")
    )
    self.assertIn("https://wandb.ai/leobianco/new_perl/sweeps/abc123", html)
    self.assertIn(f"https://huggingface.co/{fixtures.PERL_REPO}", html)
    self.assertIn("https://huggingface.co/datasets/leobianco/eval_npov", html)

  def test_campaign_page_shows_stage_degradations(self):
    html = _read(
        os.path.join(self.out, "campaign", "npov-campaign-2609211206.html")
    )
    self.assertIn("Degradations", html)
    self.assertIn("Sweep finished 1/30 trials", html)

  def test_inlines_the_assets_on_every_page(self):
    """A subresource that 401s on a private Space is an unstyled page.

    The failure is silent - no console error the reader would look for - so
    the assets travel inside the document instead of beside it.
    """
    for relative in (
        "index.html",
        os.path.join("campaign", "npov-campaign-2609211206.html"),
    ):
      page = _read(os.path.join(self.out, relative))
      self.assertNotIn("static/style.css", page, relative)
      self.assertNotIn("static/app.js", page, relative)
      # A recognisable line from each asset, rather than just the tag.
      self.assertIn("--bg-raised:", page, relative)
      self.assertIn("auto-perl-filters", page, relative)

  def test_relative_navigation_paths(self):
    detail = _read(
        os.path.join(self.out, "campaign", "npov-campaign-2609211206.html")
    )
    self.assertIn('href="../index.html"', detail)
    self.assertIn('data-index-url="../data/index.json"', detail)
    listing = _read(os.path.join(self.out, "index.html"))
    self.assertIn('data-index-url="data/index.json"', listing)


class EscapingTest(unittest.TestCase):
  """Campaign metadata originates outside this module."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.out = os.path.join(self.root, "site")
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

  def test_markup_in_a_sweep_name_is_escaped(self):
    state = fixtures.campaign_state()
    state["stages"]["sft"]["sweep_name"] = '<img src=x onerror="alert(1)">'
    fixtures.write_campaign(self.root, state=state)
    build_mod.build_site(root=self.root, out_dir=self.out)
    html = _read(
        os.path.join(self.out, "campaign", "npov-campaign-2609211206.html")
    )
    self.assertNotIn("<img src=x", html)
    self.assertIn("&lt;img src=x", html)


class StalenessTest(unittest.TestCase):
  """A snapshot must not present an abandoned campaign as live."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.out = os.path.join(self.root, "site")
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

  def _build_with_age(self, status: str, minutes: int):
    """Builds a site whose only campaign was updated ``minutes`` ago.

    Args:
      status: Campaign status to record.
      minutes: Age of its last update at snapshot time.

    Returns:
      The build result.
    """
    now = datetime.datetime(2026, 9, 22, 12, 0, 0)
    updated = now - datetime.timedelta(minutes=minutes)
    state = fixtures.campaign_state(
        status=status, updated_at=updated.isoformat()
    )
    fixtures.write_campaign(self.root, state=state, with_report=False)
    return build_mod.build_site(
        root=self.root, out_dir=self.out, generated_at=now
    )

  def test_recent_in_progress_campaign_is_live(self):
    result = self._build_with_age("IN_PROGRESS", minutes=2)
    self.assertEqual(result.stale, 0)
    self.assertEqual(result.in_progress, 1)

  def test_old_in_progress_campaign_is_stale(self):
    result = self._build_with_age("IN_PROGRESS", minutes=90)
    self.assertEqual(result.stale, 1)
    html = _read(os.path.join(self.out, "index.html"))
    self.assertIn("the VM may be off", html)

  def test_stale_campaign_page_explains_itself(self):
    self._build_with_age("IN_PROGRESS", minutes=90)
    html = _read(
        os.path.join(self.out, "campaign", "npov-campaign-2609211206.html")
    )
    self.assertIn("not been written to", html)

  def test_completed_campaigns_are_never_stale(self):
    result = self._build_with_age("COMPLETED", minutes=10_000)
    self.assertEqual(result.stale, 0)

  def test_paused_campaigns_are_never_stale(self):
    # PAUSED means somebody is coming back; it is not an abandoned run.
    result = self._build_with_age("PAUSED", minutes=10_000)
    self.assertEqual(result.stale, 0)


class ArchiveTest(unittest.TestCase):
  """Archived campaigns are published but hidden by default."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.out = os.path.join(self.root, "site")
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
    fixtures.write_campaign(self.root)
    fixtures.write_campaign(self.root, archived=True)

  def test_archived_campaigns_are_included_by_default(self):
    result = build_mod.build_site(root=self.root, out_dir=self.out)
    self.assertEqual(result.campaign_count, 2)
    html = _read(os.path.join(self.out, "index.html"))
    self.assertIn('data-archived="1"', html)
    self.assertIn("chip-archived", html)

  def test_archived_campaigns_can_be_excluded(self):
    result = build_mod.build_site(
        root=self.root, out_dir=self.out, include_archived=False
    )
    self.assertEqual(result.campaign_count, 1)


class RebuildTest(unittest.TestCase):
  """The output directory is a snapshot, not an accumulation."""

  def test_pages_of_removed_campaigns_disappear(self):
    root = tempfile.mkdtemp()
    out = os.path.join(root, "site")
    self.addCleanup(shutil.rmtree, root, ignore_errors=True)
    fixtures.write_campaign(root)
    fixtures.write_campaign(
        root, state=fixtures.campaign_state(name="npov_temporary")
    )
    build_mod.build_site(root=root, out_dir=out)
    stale_page = os.path.join(out, "campaign", "npov-temporary.html")
    self.assertTrue(os.path.isfile(stale_page))

    os.remove(
        os.path.join(root, "checkpoints", "npov", "npov_temporary_state.json")
    )
    build_mod.build_site(root=root, out_dir=out)
    self.assertFalse(os.path.isfile(stale_page))


class EmptyRepositoryTest(unittest.TestCase):
  """A repository with no campaigns still builds a usable page."""

  def test_builds_an_empty_list(self):
    root = tempfile.mkdtemp()
    out = os.path.join(root, "site")
    self.addCleanup(shutil.rmtree, root, ignore_errors=True)
    result = build_mod.build_site(root=root, out_dir=out)
    self.assertEqual(result.campaign_count, 0)
    self.assertIn("No campaigns found.", _read(os.path.join(out, "index.html")))


class NoReportTest(unittest.TestCase):
  """In-progress campaigns have no report, and must say so."""

  def test_page_explains_the_missing_report(self):
    root = tempfile.mkdtemp()
    out = os.path.join(root, "site")
    self.addCleanup(shutil.rmtree, root, ignore_errors=True)
    state = fixtures.campaign_state(status="IN_PROGRESS")
    fixtures.write_campaign(root, state=state, with_report=False)
    build_mod.build_site(root=root, out_dir=out)
    html = _read(
        os.path.join(out, "campaign", "npov-campaign-2609211206.html")
    )
    self.assertIn("No markdown report yet", html)
    self.assertFalse(
        os.path.isfile(
            os.path.join(out, "campaign", "npov-campaign-2609211206.md")
        )
    )


class PreloadedCampaignsTest(unittest.TestCase):
  """The builder accepts an index it did not load itself."""

  def test_campaigns_can_be_injected(self):
    root = tempfile.mkdtemp()
    out = os.path.join(root, "site")
    self.addCleanup(shutil.rmtree, root, ignore_errors=True)
    fixtures.write_campaign(root)
    campaigns = index_mod.build_index(root)
    result = build_mod.build_site(
        root=root, out_dir=out, campaigns=campaigns
    )
    self.assertEqual(result.campaign_count, 1)


class DryRunTest(unittest.TestCase):
  """Campaigns the orchestrator only rehearsed.

  A dry run fabricates its sweep and run ids locally. The dashboard still
  renders the links - the naming convention is the same - but every one of
  them points at something that was never created, so the page has to say so
  before the reader wastes a click.
  """

  def _build(self, state):
    """Builds a one-campaign site from a state dict.

    Args:
      state: The campaign state to write.

    Returns:
      The campaign page's HTML and the listing's HTML, in that order.
    """
    root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, root, ignore_errors=True)
    out = os.path.join(root, "site")
    fixtures.write_campaign(root, state=state)
    build_mod.build_site(root=root, out_dir=out)
    return (
        _read(os.path.join(out, "campaign", "npov-campaign-2609211206.html")),
        _read(os.path.join(out, "index.html")),
    )

  def test_the_config_flag_is_enough(self):
    state = fixtures.campaign_state()
    state["config_dict"]["dry_run"] = True
    detail, listing = self._build(state)
    self.assertIn('<span class="chip chip-dry-run">', listing)
    self.assertIn('<span class="chip chip-dry-run">', detail)
    self.assertIn("Dry run.", detail)

  def test_mock_sweep_ids_are_enough(self):
    """States written before ``dry_run`` was persisted still betray it."""
    state = fixtures.campaign_state()
    state["config_dict"].pop("dry_run", None)
    for stage in state["stages"].values():
      if stage.get("sweep_id"):
        stage["sweep_id"] = "leobianco/new_perl/mock_sweep_1789992366"
    detail, _ = self._build(state)
    self.assertIn("Dry run.", detail)

  def test_a_real_campaign_says_nothing(self):
    detail, _ = self._build(fixtures.campaign_state())
    self.assertNotIn("Dry run.", detail)


if __name__ == "__main__":
  unittest.main()
