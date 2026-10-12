"""Tests for the temperature-grid evaluation view: SVG charts and layout."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import unittest

from src.dashboard import build as build_mod
from src.dashboard import charts as charts_mod
from src.dashboard import index as index_mod
from src.dashboard.tests import fixtures
from src.orchestrator import eval_metrics

_GRID = (0.0, 0.7, 1.0, 1.25)
_POLICIES = ("sft", "perl:organic", "perl:synthetic_struct")
_ROLLOUT = {"perl:organic": 0.7, "perl:synthetic_struct": 1.0}


def _token(temperature: float) -> str:
  return eval_metrics.format_temperature(temperature)


def grid_eval_metrics() -> dict:
  """Returns flat eval metrics for a 3-policy x 4-temperature campaign."""
  metrics = {}
  for policy in _POLICIES:
    for index, temperature in enumerate(_GRID):
      label = f"{policy}@t{_token(temperature)}"
      offset = 0.1 if policy == "sft" else 0.05
      rate = offset + 0.02 * index
      metrics[f"{label}/hallucination_rate"] = rate
      metrics[f"{label}/faithfulness_rate"] = 1 - rate
      metrics[f"{label}/autorater_scored_count"] = 400
      metrics[f"{label}/reward_hacking_quality"] = 0.8 - 0.03 * index
      metrics[f"{label}/reward_hacking_audit_quality_std"] = 0.18
      metrics[f"{label}/reward_hacking_audit_n_scored"] = 100
      metrics[f"{label}/perplexity_mean"] = 3.0 + index
      metrics[f"{label}/perplexity_std"] = 1.0
      metrics[f"{label}/num_samples"] = 400
      metrics[f"{label}/decoding_temperature"] = temperature
      if policy in _ROLLOUT:
        metrics[f"{label}/rollout_temperature"] = _ROLLOUT[policy]
        flavor = policy.partition(":")[2]
        metrics[f"delta:{flavor}@t{_token(temperature)}/hallucination_rate"] = (
            0.05
        )
  return metrics


def _grid_state() -> dict:
  state = fixtures.campaign_state()
  state["stages"]["eval"]["metrics"] = grid_eval_metrics()
  return state


class ChartsTest(unittest.TestCase):
  """The SVG emitters on their own."""

  def test_bar_chart_draws_one_rect_per_value_and_error_bars(self):
    svg = charts_mod.grouped_bar_chart(
        "Rates",
        ["T = 0", "T = 1"],
        [
            charts_mod.BarSeries(
                "SFT", "#999", [0.1, 0.2], [(0.08, 0.12), None], [False, True]
            ),
            charts_mod.BarSeries("PE-RL", "#00f", [0.05, None]),
        ],
        "Rate",
        percent=True,
        marker_note="star note",
    )
    self.assertEqual(svg.count("<rect"), 3)
    self.assertEqual(svg.count('class="errbar"'), 1)
    self.assertIn("\u2605", svg)
    self.assertIn("star note", svg)
    self.assertIn("%", svg)

  def test_bar_chart_escapes_dynamic_strings(self):
    svg = charts_mod.grouped_bar_chart(
        "<t>", ["<c>"], [charts_mod.BarSeries("<s>", "#000", [1.0])], "<y>"
    )
    self.assertNotIn("<s>", svg)
    self.assertIn("&lt;s&gt;", svg)
    self.assertIn("&lt;c&gt;", svg)

  def test_empty_input_renders_nothing(self):
    self.assertEqual(charts_mod.grouped_bar_chart("x", [], [], "y"), "")
    self.assertEqual(
        charts_mod.grouped_bar_chart(
            "x", ["a"], [charts_mod.BarSeries("s", "#000", [None])], "y"
        ),
        "",
    )
    self.assertEqual(charts_mod.trajectory_plot("x", [], "a", "b"), "")

  def test_trajectory_connects_points_and_rings_the_marked_one(self):
    svg = charts_mod.trajectory_plot(
        "Frontier",
        [
            charts_mod.TrajectorySeries(
                "PE-RL",
                "#00f",
                [
                    charts_mod.TrajectoryPoint(0.8, 0.9, "0"),
                    charts_mod.TrajectoryPoint(
                        0.7, 0.85, "0.7", (0.65, 0.75), (0.8, 0.9), True
                    ),
                ],
            )
        ],
        "quality",
        "faithfulness",
    )
    self.assertEqual(svg.count("<circle"), 3)  # two points + one ring
    self.assertEqual(svg.count('class="trajectory"'), 1)
    self.assertEqual(svg.count('class="errbar"'), 2)

  def test_nice_ticks_cover_the_data(self):
    lo, hi, step = charts_mod._axis_range([0.013, 0.27], include_zero=True)
    self.assertEqual(lo, 0.0)
    self.assertGreaterEqual(hi, 0.27)
    self.assertIn(step, (0.05, 0.1))

  def test_known_policies_get_stable_colors(self):
    self.assertNotEqual(
        charts_mod.series_color("perl:organic", 0),
        charts_mod.series_color("perl:synthetic_struct", 1),
    )
    self.assertEqual(
        charts_mod.series_color("unknown", 0),
        charts_mod.series_color("other", 0),
    )


class GridIndexTest(unittest.TestCase):
  """Target grouping and delta pairing for grid campaigns."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
    path = fixtures.write_campaign(self.root, state=_grid_state())
    self.campaign = index_mod.load_campaign(path, root=self.root)

  def test_grid_is_detected(self):
    self.assertTrue(index_mod.is_temperature_grid(self.campaign.eval_targets))

  def test_legacy_layout_is_not_a_grid(self):
    root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, root, ignore_errors=True)
    legacy = index_mod.load_campaign(fixtures.write_campaign(root), root=root)
    self.assertFalse(index_mod.is_temperature_grid(legacy.eval_targets))

  def test_columns_are_grouped_coldest_first(self):
    temperatures = [t.temperature for t in self.campaign.eval_targets]
    self.assertEqual(temperatures, sorted(temperatures))
    first_group = [
        t.label for t in self.campaign.eval_targets if t.temperature == 0.0
    ]
    self.assertEqual(
        first_group,
        [
            "sft@t0.0",
            "perl:organic@t0.0",
            "delta:organic@t0.0",
            "perl:synthetic_struct@t0.0",
            "delta:synthetic_struct@t0.0",
        ],
    )

  def test_deltas_pair_with_the_policy_at_their_own_temperature(self):
    targets = {t.label: t for t in self.campaign.eval_targets}
    policy = index_mod._policy_for_delta(
        targets["delta:organic@t1.25"], self.campaign.eval_targets
    )
    self.assertEqual(policy.label, "perl:organic@t1.25")

  def test_headline_is_the_coldest_cell(self):
    self.assertAlmostEqual(
        self.campaign.headline["hallucination_rate"], 0.05
    )


class GridBuildTest(unittest.TestCase):
  """A full build of a grid campaign."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.out = os.path.join(self.root, "site")
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
    state = _grid_state()
    fixtures.write_campaign(self.root, state=state)
    build_mod.build_site(root=self.root, out_dir=self.out)
    slug = [
        name
        for name in os.listdir(os.path.join(self.out, "campaign"))
        if name.endswith(".html")
    ][0]
    with open(
        os.path.join(self.out, "campaign", slug), "r", encoding="utf-8"
    ) as handle:
      self.page = handle.read()

  def test_renders_all_charts(self):
    self.assertEqual(self.page.count('<figure class="chart">'), 5)
    for title in (
        "Hallucination rate by decoding temperature",
        "Reward-hacking audit quality by decoding temperature",
        "Quality vs. faithfulness across temperatures",
        "Hallucination-rate reduction vs. SFT",
        "Perplexity by decoding temperature",
    ):
      self.assertIn(title, self.page)

  def test_one_table_per_temperature(self):
    headings = re.findall(
        r'<h4 class="temperature-heading">([^<]*)</h4>', self.page
    )
    self.assertEqual(
        headings,
        [f"Decoding temperature T = {_token(t)}" for t in _GRID],
    )

  def test_rollout_temperature_is_starred(self):
    self.assertIn("\u2605", self.page)

  def test_cells_carry_confidence_intervals(self):
    self.assertIn('title="95% confidence interval"', self.page)

  def test_legacy_campaign_gets_no_charts(self):
    root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, root, ignore_errors=True)
    fixtures.write_campaign(root)
    out = os.path.join(root, "site")
    build_mod.build_site(root=root, out_dir=out)
    for name in os.listdir(os.path.join(out, "campaign")):
      if name.endswith(".html"):
        with open(
            os.path.join(out, "campaign", name), "r", encoding="utf-8"
        ) as handle:
          page = handle.read()
        self.assertNotIn('<figure class="chart">', page)
        self.assertNotIn('<h4 class="temperature-heading"', page)


if __name__ == "__main__":
  unittest.main()
