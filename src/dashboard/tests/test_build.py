"""Tests for static site emission."""

from __future__ import annotations

import datetime
import json
import os
import re
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


class ContentHashTest(unittest.TestCase):
  """What the publisher's skip-if-unchanged guard is allowed to ignore."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
    fixtures.write_campaign(self.root)

  def _hash(self, **kwargs):
    """Builds the site and returns its content hash.

    Args:
      **kwargs: Passed through to ``build_site``.

    Returns:
      The build's content hash.
    """
    out = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, out, ignore_errors=True)
    return build_mod.build_site(root=self.root, out_dir=out, **kwargs).content_hash

  def test_the_clock_is_not_part_of_it(self):
    """Otherwise every poll would look like a change and republish."""
    early = datetime.datetime(2026, 9, 22, 9, 0, tzinfo=datetime.timezone.utc)
    late = datetime.datetime(2026, 9, 22, 18, 0, tzinfo=datetime.timezone.utc)
    self.assertEqual(
        self._hash(generated_at=early), self._hash(generated_at=late)
    )

  def test_the_renderer_is_part_of_it(self):
    """A stylesheet edit changes the site, so it must change the hash.

    The data is identical in both builds; only the presentation moved. Before
    this was folded in, restyling the dashboard produced a visibly different
    site that the publisher skipped as unchanged.
    """
    before = self._hash()
    original = build_mod._asset("style.css")
    build_mod._ASSET_CACHE["style.css"] = original + "\n.kpi { color: red; }\n"
    self.addCleanup(build_mod._ASSET_CACHE.__setitem__, "style.css", original)
    self.assertNotEqual(before, self._hash())


class KpiTest(unittest.TestCase):
  """The headline cards, whose sign convention is easy to get backwards."""

  def _cards(self, state):
    """Renders the KPI strip for a campaign state.

    Args:
      state: The campaign state dict.

    Returns:
      The KPI HTML.
    """
    root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, root, ignore_errors=True)
    fixtures.write_campaign(root, state=state)
    campaign = index_mod.build_index(root)[0]
    return build_mod._kpis(campaign)

  def test_fewer_hallucinations_reads_as_a_negative_and_is_green(self):
    """The stage stores positive-is-better; the card shows the rate's change.

    The fixture improves from 10.5% to 6.2%, so the card must say -4.3% and
    be green. Showing "+0.043" was correct arithmetic and the wrong story.
    """
    html = self._cards(fixtures.campaign_state())
    self.assertIn("-4.3%", html)
    self.assertNotIn("+4.3%", html)
    self.assertIn('kpi-value good', html)
    self.assertNotIn('kpi-value bad', html)

  def test_more_hallucinations_reads_as_a_positive_and_is_red(self):
    state = fixtures.campaign_state()
    metrics = state["stages"]["eval"]["metrics"]
    metrics["delta/hallucination_rate"] = -0.02
    html = self._cards(state)
    self.assertIn("+2.0%", html)
    self.assertIn('kpi-value bad', html)

  def test_rates_are_rendered_as_percentages(self):
    html = self._cards(fixtures.campaign_state())
    self.assertIn("6.2%", html)
    self.assertNotIn("0.062", html)

  def test_every_branch_gets_its_own_pair_of_cards(self):
    """A fan-out campaign has no single "the" PE-RL rate.

    Collapsing the branches would hide the comparison the fan-out exists to
    make, so each flavor is named and carries its own change card.
    """
    state = fixtures.campaign_state()
    metrics = state["stages"]["eval"]["metrics"]
    for key in [k for k in metrics if k.startswith(("perl/", "delta/"))]:
      suffix = key.split("/", 1)[1]
      namespace = key.split("/", 1)[0]
      metrics[f"{namespace}:organic/{suffix}"] = metrics[key]
      del metrics[key]
    metrics["perl:synthetic_struct/hallucination_rate"] = 0.081
    metrics["perl:synthetic_struct/decoding_temperature"] = 0.7
    metrics["delta:synthetic_struct/hallucination_rate"] = 0.024
    html = self._cards(state)
    self.assertIn("PE-RL hallucination rate (organic)", html)
    self.assertIn("PE-RL hallucination rate (synthetic_struct)", html)
    self.assertIn("Change vs SFT (organic)", html)
    self.assertIn("Change vs SFT (synthetic_struct)", html)
    self.assertIn("8.1%", html)


class EvalTableTest(unittest.TestCase):
  """The comparison table, which used to be unreadably wide."""

  def _table(self, extra=None):
    """Builds a campaign page's evaluation table.

    Args:
      extra: Additional eval metrics to record.

    Returns:
      The table HTML.
    """
    state = fixtures.campaign_state()
    state["stages"]["eval"]["metrics"].update(extra or {})
    root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, root, ignore_errors=True)
    fixtures.write_campaign(root, state=state)
    return build_mod._eval_table(index_mod.build_index(root)[0])

  def test_provenance_never_appears(self):
    """Its values are adapter repo ids; they are what made the table wide."""
    html = self._table({
        "perl/provenance_policy_adapter": "leobianco/" + "x" * 90,
        "sft/provenance_policy_adapter": "leobianco/" + "y" * 90,
    })
    self.assertNotIn("provenance", html)
    self.assertNotIn("x" * 90, html)

  def test_the_wide_tail_goes_behind_a_disclosure(self):
    html = self._table({
        "perl/rouge1_precision_median": 0.4,
        "sft/rouge1_precision_median": 0.5,
    })
    main, _, more = html.partition("<details")
    self.assertNotIn("rouge1 precision median", main)
    self.assertIn("rouge1 precision median", more)
    self.assertIn("hallucination rate", main)

  def test_the_temperature_leads_the_table(self):
    html = self._table()
    body = html.split("<tbody>")[1]
    self.assertTrue(
        body.lstrip().startswith("<tr><td>decoding temperature</td>"), body[:120]
    )

  def test_the_superseded_spelling_does_not_duplicate_a_row(self):
    """`bertscore_f1` and `bertscore_f1_mean` are the same measurement.

    A state written across a version boundary can hold both, which rendered
    as two rows labelled "bertscore f1" with different numbers.
    """
    html = self._table({
        "perl/bertscore_f1_mean": 0.9,
        "sft/bertscore_f1_mean": 0.88,
    })
    main = html.partition("<details")[0]
    self.assertEqual(main.count("<td>bertscore f1</td>"), 1)

  def test_mean_suffixes_are_dropped_but_std_is_kept(self):
    html = self._table({
        "perl/bertscore_f1_mean": 0.9,
        "perl/bertscore_f1_std": 0.02,
        "sft/bertscore_f1_mean": 0.88,
        "sft/bertscore_f1_std": 0.03,
    })
    self.assertIn("<td>bertscore f1</td>", html)
    self.assertIn("<td>bertscore f1 std</td>", html)

  def test_both_tables_lead_with_the_baselines_and_pair_the_deltas(self):
    """The disclosure reuses the header, so it must not drift from the main.

    A reader who expands the tail is comparing the same columns; renumbering
    them halfway down the page would be worse than not pairing them at all.
    """
    html = self._table({
        "perl:organic/hallucination_rate": 0.062,
        "delta:organic/hallucination_rate": 0.043,
        "perl:organic/rouge1_precision_median": 0.4,
        "sft/rouge1_precision_median": 0.5,
    })
    expected = [
        "sft",
        "sft@t0.7",
        "perl",
        "delta",
        "perl:organic",
        "delta:organic",
    ]
    main, _, more = html.partition("<details")
    for chunk in (main, more):
      headers = re.findall(r'<th class="num">([^<]*)</th>', chunk)
      self.assertEqual(headers, expected)

  def test_a_delta_also_reads_as_a_share_of_its_baseline(self):
    """0.105 -> 0.062 is +0.043, which is 41% of what SFT was getting wrong.

    The absolute number alone cannot say whether that is a rout or noise.
    """
    html = self._table()
    self.assertIn('0.043<span class="sub">+41.0%</span>', html)

  def test_the_share_sits_only_under_the_delta(self):
    """The policy and baseline columns are levels, not changes."""
    html = self._table()
    row = next(
        line
        for line in html.split("<tr>")
        if line.startswith("<td>hallucination rate</td>")
    )
    self.assertEqual(row.count('class="sub"'), 1)

  def test_each_branch_shows_its_own_share(self):
    """Two branches compared against baselines at different temperatures."""
    html = self._table({
        "sft@t1.0/hallucination_rate": 0.200,
        "sft@t1.0/decoding_temperature": 1.0,
        "perl:synthetic_struct/hallucination_rate": 0.100,
        "perl:synthetic_struct/decoding_temperature": 1.0,
        "delta:synthetic_struct/hallucination_rate": 0.100,
    })
    self.assertIn('0.043<span class="sub">+41.0%</span>', html)
    self.assertIn('0.1<span class="sub">+50.0%</span>', html)

  def test_a_regression_keeps_its_sign(self):
    html = self._table({
        "sft@t0.7/bertscore_f1": 0.900,
        "perl/bertscore_f1": 0.810,
        "delta/bertscore_f1": -0.090,
    })
    self.assertIn('-0.09<span class="sub">-10.0%</span>', html)

  def test_the_disclosure_gets_the_shares_too(self):
    """A metric behind "N more" is read the same way as one above it."""
    html = self._table({
        "sft@t0.7/rouge1_precision_median": 0.50,
        "perl/rouge1_precision_median": 0.55,
        "delta/rouge1_precision_median": 0.05,
    })
    more = html.partition("<details")[2]
    self.assertIn('0.05<span class="sub">+10.0%</span>', more)

  def test_the_legend_explains_the_small_number_once(self):
    html = self._table()
    self.assertEqual(html.count('<p class="subtitle">'), 1)
    self.assertIn("share of the baseline", html)

  def test_a_campaign_without_shares_gets_no_legend(self):
    """Nothing to explain when no column carries one."""
    state = fixtures.campaign_state()
    metrics = state["stages"]["eval"]["metrics"]
    for key in [k for k in metrics if k.startswith("delta/")]:
      del metrics[key]
    root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, root, ignore_errors=True)
    fixtures.write_campaign(root, state=state)
    html = build_mod._eval_table(index_mod.build_index(root)[0])
    self.assertNotIn('<p class="subtitle">', html)


if __name__ == "__main__":
  unittest.main()
