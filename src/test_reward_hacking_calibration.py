"""Tests for the reward-hacking rubric calibration on healthy references.

The calibration answers one question with data that only holds negatives:
how often does the deployed rule (``quality < --reward_hacking_threshold``)
fire on text known to be healthy? The properties worth pinning:

1. The false-positive rate measured here is exactly the
   ``reward_hacking_rate`` the scoring pipeline would report on the same
   verdicts. A calibration of a differently aggregated score calibrates
   nothing.
2. Each suggested threshold is the *largest* one whose FPR stays within its
   budget: one grid step higher must exceed it.
3. The guards keep demonstration prompts, empty references and duplicates
   out, and the per-arm cap counts usable rows only.
4. The independence test compares labels within the reward-model arm only,
   so the faithful-only SFT arm cannot masquerade as a label effect.
5. The final test set and the autorater set are refused outright.

Everything here is pure Python and runs without the ML stack.
"""

import json
import math
import os
import random
import re
import tempfile
import unittest

from src import reward_hacking_calibration as rhc
from src import utils

_DIMENSIONS = utils.reward_hacking_dimension_keys()


class _FakeProcessor:
  """Mirrors the generic accessors of ``BaseTaskProcessor``."""

  @classmethod
  def reward_hacking_context(cls, entry):
    return entry.get("context", "")

  @classmethod
  def reward_hacking_query(cls, entry):
    return entry.get("question", "")

  @classmethod
  def reward_hacking_response(cls, entry, use_true_label=False):
    keys = ("response", "completion") if use_true_label else ("completion",)
    for key in keys:
      if entry.get(key):
        return entry[key]
    return ""


def _grades(value):
  return {key: value for key in _DIMENSIONS}


def _row(
    index=0,
    source=rhc.SOURCE_RM,
    label=rhc.LABEL_FAITHFUL,
    grades=None,
    context=None,
    query="q",
    response=None,
):
  return rhc.CalibrationRow(
      source=source,
      source_index=index,
      label=label,
      context=f"context {index}" if context is None else context,
      query=query,
      response=f"response {index}" if response is None else response,
      grades=grades,
  )


class VerdictQualityTest(unittest.TestCase):

  def test_it_is_the_unweighted_mean_of_the_dimensions_present(self):
    grades = {"fluency": 1.0, "non_repetition": 0.5}
    self.assertAlmostEqual(rhc.verdict_quality(grades), 0.75)

  def test_unresolved_verdicts_have_no_quality(self):
    self.assertIsNone(rhc.verdict_quality(None))
    self.assertIsNone(rhc.verdict_quality({}))
    self.assertIsNone(rhc.verdict_quality({"vibes": 1.0}))


class FalsePositiveRateMatchesDeployedRateTest(unittest.TestCase):
  """The FPR must be the rate ``reward_hacking_summary`` reports."""

  def test_random_verdicts_agree_with_the_deployed_summary(self):
    rng = random.Random(7)
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    for trial in range(25):
      verdicts = []
      for _ in range(rng.randint(1, 60)):
        roll = rng.random()
        if roll < 0.05:
          verdicts.append(None)
          continue
        keys = _DIMENSIONS if roll > 0.15 else _DIMENSIONS[:2]
        verdicts.append({key: rng.choice(levels) for key in keys})
      threshold = rng.choice([0.5, 0.6, 7 / 12, 0.75])
      qualities = [
          rhc.verdict_quality(v) for v in verdicts if rhc.verdict_quality(v)
          is not None
      ]
      deployed = utils.reward_hacking_summary(verdicts, threshold).get(
          "reward_hacking_rate"
      )
      with self.subTest(trial=trial):
        self.assertEqual(
            rhc.false_positive_rate(qualities, threshold), deployed
        )

  def test_a_quality_equal_to_the_threshold_is_not_flagged(self):
    self.assertEqual(rhc.false_positive_rate([0.5, 0.75], 0.5), 0.0)

  def test_empty_sample_has_no_rate(self):
    self.assertIsNone(rhc.false_positive_rate([], 0.6))


class ResolveLabelTest(unittest.TestCase):

  def test_class_hall_is_read_case_insensitively(self):
    self.assertEqual(
        rhc.resolve_label({"class_hall": " YES "}, rhc.SOURCE_RM),
        rhc.LABEL_HALLUCINATED,
    )
    self.assertEqual(
        rhc.resolve_label({"class_hall": "No"}, rhc.SOURCE_RM),
        rhc.LABEL_FAITHFUL,
    )

  def test_numeric_label_follows_the_repo_convention(self):
    # 1 = faithful, 0 = hallucinated, in every task processor.
    self.assertEqual(
        rhc.resolve_label({"label": 1}, rhc.SOURCE_RM), rhc.LABEL_FAITHFUL
    )
    self.assertEqual(
        rhc.resolve_label({"label": 0}, rhc.SOURCE_RM),
        rhc.LABEL_HALLUCINATED,
    )
    self.assertEqual(
        rhc.resolve_label({"label": "0"}, rhc.SOURCE_RM),
        rhc.LABEL_HALLUCINATED,
    )

  def test_class_hall_wins_over_label(self):
    row = {"class_hall": "Yes", "label": 1}
    self.assertEqual(
        rhc.resolve_label(row, rhc.SOURCE_RM), rhc.LABEL_HALLUCINATED
    )

  def test_booleans_are_not_read_as_labels(self):
    self.assertEqual(
        rhc.resolve_label({"label": True}, rhc.SOURCE_RM), rhc.LABEL_UNKNOWN
    )

  def test_unannotated_sft_rows_are_faithful_by_construction(self):
    self.assertEqual(rhc.resolve_label({}, rhc.SOURCE_SFT), rhc.LABEL_FAITHFUL)
    self.assertEqual(rhc.resolve_label({}, rhc.SOURCE_RM), rhc.LABEL_UNKNOWN)

  def test_an_annotated_sft_row_keeps_its_annotation(self):
    self.assertEqual(
        rhc.resolve_label({"class_hall": "Yes"}, rhc.SOURCE_SFT),
        rhc.LABEL_HALLUCINATED,
    )


class CalibrationSourcesTest(unittest.TestCase):

  def test_defaults_are_the_two_held_out_arms(self):
    self.assertEqual(
        rhc.calibration_sources("leobianco", "ragtruth"),
        [
            (rhc.SOURCE_RM, "leobianco/ragtruth_rm_organic", "test"),
            (rhc.SOURCE_SFT, "leobianco/ragtruth_sft", "test"),
        ],
    )

  def test_overrides_are_honoured(self):
    sources = rhc.calibration_sources(
        "u", "npov", rm_dataset="x/rm", rm_split="dev", sft_split="validation"
    )
    self.assertEqual(
        sources,
        [
            (rhc.SOURCE_RM, "x/rm", "dev"),
            (rhc.SOURCE_SFT, "u/npov_sft", "validation"),
        ],
    )

  def test_an_arm_can_be_switched_off(self):
    self.assertEqual(
        rhc.calibration_sources("u", "npov", sft_dataset="None"),
        [(rhc.SOURCE_RM, "u/npov_rm_organic", "test")],
    )

  def test_switching_both_arms_off_is_refused(self):
    with self.assertRaises(ValueError):
      rhc.calibration_sources("u", "npov", rm_dataset="none", sft_dataset="off")

  def test_the_final_test_set_is_refused(self):
    with self.assertRaisesRegex(ValueError, "final"):
      rhc.calibration_sources(
          "u", "npov", rm_dataset="u/npov_final_test_set"
      )

  def test_the_autorater_set_is_refused(self):
    with self.assertRaisesRegex(ValueError, "autorater"):
      rhc.check_calibration_sources(["u/bosch_autorater"])


class BuildRowsTest(unittest.TestCase):

  def test_rows_carry_the_reference_not_a_completion(self):
    raw = [{
        "context": "ctx",
        "question": "why?",
        "response": "the reference",
        "completion": "a policy completion",
        "class_hall": "Yes",
    }]
    (row,) = rhc.build_rows(raw, rhc.SOURCE_RM, _FakeProcessor)
    self.assertEqual(row.response, "the reference")
    self.assertEqual(row.context, "ctx")
    self.assertEqual(row.query, "why?")
    self.assertEqual(row.label, rhc.LABEL_HALLUCINATED)
    self.assertEqual(row.source_index, 0)
    self.assertEqual(row.raw["completion"], "a policy completion")

  def test_demo_prompt_keys_use_the_processor_accessors(self):
    examples = [
        {"context": "c  1", "question": "q"},
        {"context": "c 1", "question": "q", "is_degenerate": True},
    ]
    self.assertEqual(
        rhc.demo_prompt_keys(examples, _FakeProcessor), {("c 1", "q")}
    )
    self.assertEqual(rhc.demo_prompt_keys(None, _FakeProcessor), set())


class PrepareRowsTest(unittest.TestCase):

  def test_guards_and_their_accounting(self):
    rm_rows = [
        _row(0, response="   "),  # Empty.
        _row(1, context="demo  context", query="demo q"),  # Demo prompt.
        _row(2, label=rhc.LABEL_HALLUCINATED),
        _row(3),
    ]
    sft_rows = [
        # Same text as rm row 2, with whitespace noise: a duplicate.
        _row(0, source=rhc.SOURCE_SFT, context="context  2",
             response="response 2"),
        _row(1, source=rhc.SOURCE_SFT, context="fresh"),
    ]
    kept, accounting = rhc.prepare_rows(
        [(rhc.SOURCE_RM, rm_rows), (rhc.SOURCE_SFT, sft_rows)],
        demo_keys={("demo context", "demo q")},
        max_rows_per_source=None,
        seed=0,
    )
    self.assertEqual(
        [(r.source, r.source_index) for r in kept],
        [(rhc.SOURCE_RM, 2), (rhc.SOURCE_RM, 3), (rhc.SOURCE_SFT, 1)],
    )
    # The copy that survives is the reward-model one, label included.
    self.assertEqual(kept[0].label, rhc.LABEL_HALLUCINATED)
    self.assertEqual(
        accounting[rhc.SOURCE_RM],
        {
            "loaded": 4,
            "dropped_empty_response": 1,
            "dropped_demo_prompt": 1,
            "dropped_duplicate": 0,
            "after_guards": 2,
            "after_subsample": 2,
        },
    )
    self.assertEqual(accounting[rhc.SOURCE_SFT]["dropped_duplicate"], 1)
    self.assertEqual(accounting[rhc.SOURCE_SFT]["after_subsample"], 1)

  def test_the_cap_applies_per_arm_after_the_guards(self):
    rm_rows = [_row(i, response="" if i < 5 else None) for i in range(15)]
    sft_rows = [_row(i, source=rhc.SOURCE_SFT) for i in range(3)]
    kept, accounting = rhc.prepare_rows(
        [(rhc.SOURCE_RM, rm_rows), (rhc.SOURCE_SFT, sft_rows)],
        demo_keys=(),
        max_rows_per_source=8,
        seed=3,
    )
    self.assertEqual(accounting[rhc.SOURCE_RM]["after_guards"], 10)
    self.assertEqual(accounting[rhc.SOURCE_RM]["after_subsample"], 8)
    self.assertEqual(accounting[rhc.SOURCE_SFT]["after_subsample"], 3)
    self.assertEqual(len(kept), 11)

  def test_the_subsample_is_seeded_and_keeps_dataset_order(self):
    rows = [_row(i) for i in range(50)]
    first = rhc.subsample(rows, 10, seed=12345)
    again = rhc.subsample(rows, 10, seed=12345)
    other = rhc.subsample(rows, 10, seed=1)
    indices = [r.source_index for r in first]
    self.assertEqual(indices, sorted(indices))
    self.assertEqual(indices, [r.source_index for r in again])
    self.assertNotEqual(indices, [r.source_index for r in other])
    self.assertEqual(len(rhc.subsample(rows, -1, seed=0)), 50)
    self.assertEqual(len(rhc.subsample(rows, 0, seed=0)), 50)


class SuggestThresholdsTest(unittest.TestCase):

  def test_hand_computed_example(self):
    # 100 rows: 2 at 1/3, 4 at 1/2, 6 at 7/12, the rest at 1.
    qualities = [1 / 3] * 2 + [0.5] * 4 + [7 / 12] * 6 + [1.0] * 88
    by_target = {
        s["target_fpr"]: s for s in rhc.suggest_thresholds(qualities)
    }
    # 1%: even the lowest value holds 2% of the rows, so the threshold sits
    # on it and flags nothing.
    self.assertAlmostEqual(by_target[0.01]["threshold"], 1 / 3)
    self.assertEqual(by_target[0.01]["n_flagged"], 0)
    # 5%: flagging the 1/3s (2%) is fine, adding the 1/2s (6%) is not.
    self.assertAlmostEqual(by_target[0.05]["threshold"], 0.5)
    self.assertEqual(by_target[0.05]["n_flagged"], 2)
    self.assertAlmostEqual(by_target[0.05]["achieved_fpr"], 0.02)
    # 10%: 1/3s and 1/2s (6%) fit; adding the 7/12s (12%) does not.
    self.assertAlmostEqual(by_target[0.10]["threshold"], 7 / 12)
    self.assertEqual(by_target[0.10]["n_flagged"], 6)
    self.assertTrue(by_target[0.10]["reliable"])
    self.assertFalse(by_target[0.01]["reliable"])

  def test_each_threshold_is_the_largest_within_budget(self):
    rng = random.Random(11)
    grid = [i / 12 for i in range(13)]
    for trial in range(40):
      n = rng.randint(1, 400)
      weights = [rng.random() ** 3 for _ in grid]
      qualities = rng.choices(grid, weights=weights, k=n)
      cells = sorted(set(round(q, 9) for q in qualities))
      for suggestion in rhc.suggest_thresholds(qualities):
        alpha = suggestion["target_fpr"]
        threshold = suggestion["threshold"]
        with self.subTest(trial=trial, alpha=alpha):
          fpr = rhc.false_positive_rate(qualities, threshold)
          self.assertLessEqual(fpr, alpha + 1e-12)
          self.assertAlmostEqual(fpr, suggestion["achieved_fpr"])
          self.assertEqual(
              sum(1 for q in qualities if q < threshold),
              suggestion["n_flagged"],
          )
          higher = [c for c in cells if c > round(threshold, 9)]
          if higher:
            next_threshold = min(
                q for q in qualities if round(q, 9) == higher[0]
            )
            self.assertGreater(
                rhc.false_positive_rate(qualities, next_threshold), alpha
            )

  def test_float_noise_does_not_split_a_grid_point(self):
    noisy = [0.1 + 0.2, 0.3, 0.3, 1.0]  # 0.1 + 0.2 == 0.30000000000000004.
    (suggestion,) = rhc.suggest_thresholds(noisy, target_fprs=(0.5,))
    self.assertEqual(suggestion["n_flagged"], 0)
    self.assertAlmostEqual(suggestion["threshold"], 0.3)

  def test_budgets_outside_the_unit_interval_are_refused(self):
    with self.assertRaises(ValueError):
      rhc.suggest_thresholds([0.5], target_fprs=(1.0,))
    with self.assertRaises(ValueError):
      rhc.suggest_thresholds([0.5], target_fprs=(0.0,))

  def test_no_rows_no_suggestions(self):
    self.assertEqual(rhc.suggest_thresholds([]), [])

  def test_the_achieved_rate_carries_a_wilson_interval(self):
    (suggestion,) = rhc.suggest_thresholds([0.0, 1.0, 1.0, 1.0], (0.3,))
    low, high = suggestion["achieved_fpr_ci95"]
    self.assertLessEqual(low, suggestion["achieved_fpr"])
    self.assertGreaterEqual(high, suggestion["achieved_fpr"])


class WilsonIntervalTest(unittest.TestCase):

  def test_it_matches_the_orchestrator_copy(self):
    # The module keeps its own copy to stay independent of the
    # orchestrator package; this pins the two together.
    from src.orchestrator import eval_metrics  # pylint: disable=g-import-not-at-top

    self.assertEqual(rhc.Z_95, eval_metrics.Z_95)
    for proportion, n in [(0.0, 10), (0.02, 500), (0.5, 3), (1.0, 7)]:
      with self.subTest(p=proportion, n=n):
        self.assertEqual(
            rhc.wilson_interval(proportion, n),
            eval_metrics.wilson_interval(proportion, n),
        )

  def test_an_empty_sample_is_refused(self):
    with self.assertRaises(ValueError):
      rhc.wilson_interval(0.0, 0)


class WelchDifferenceTest(unittest.TestCase):

  def test_hand_computed_example(self):
    result = rhc.welch_difference([1.0, 0.5, 0.75], [0.5, 0.25])
    self.assertAlmostEqual(result["delta"], 0.75 - 0.375)
    # Sample variances: 0.0625 and 0.03125.
    expected_se = math.sqrt(0.0625 / 3 + 0.03125 / 2)
    self.assertAlmostEqual(result["se"], expected_se)
    self.assertAlmostEqual(
        result["ci95"][0], result["delta"] - rhc.Z_95 * expected_se
    )
    self.assertTrue(result["excludes_zero"])

  def test_identical_groups_do_not_exclude_zero(self):
    result = rhc.welch_difference([0.5, 1.0], [0.5, 1.0])
    self.assertEqual(result["delta"], 0.0)
    self.assertFalse(result["excludes_zero"])

  def test_small_groups_get_no_interval(self):
    result = rhc.welch_difference([1.0], [0.5, 0.25])
    self.assertAlmostEqual(result["delta"], 0.625)
    self.assertIsNone(result["ci95"])
    self.assertIsNone(result["excludes_zero"])

  def test_an_empty_group_gives_nothing(self):
    self.assertIsNone(rhc.welch_difference([], [1.0]))


class StrataAndIndependenceTest(unittest.TestCase):

  def _rows(self):
    rows = []
    for i in range(6):
      rows.append(_row(i, label=rhc.LABEL_FAITHFUL, grades=_grades(1.0)))
    for i in range(6, 10):
      rows.append(_row(i, label=rhc.LABEL_HALLUCINATED, grades=_grades(0.75)))
    # Faithful SFT rows graded badly: a pooled comparison would read their
    # low grades as a (negative) label effect.
    for i in range(5):
      rows.append(
          _row(i, source=rhc.SOURCE_SFT, label=rhc.LABEL_FAITHFUL,
               grades=_grades(0.25))
      )
    rows.append(_row(99, grades=None))  # Judge failure.
    return rows

  def test_strata_names_and_order(self):
    names = [name for name, _ in rhc.stratify(self._rows())]
    self.assertEqual(
        names,
        [
            "all",
            "label=faithful",
            "label=hallucinated",
            "source=rm_organic",
            "source=rm_organic/label=faithful",
            "source=rm_organic/label=hallucinated",
            "source=sft",
            "source=sft/label=faithful",
        ],
    )

  def test_independence_uses_the_reward_model_arm_only(self):
    result = rhc.independence_tests(self._rows())
    self.assertEqual(result["n_faithful"], 6)
    self.assertEqual(result["n_hallucinated"], 4)
    for key in _DIMENSIONS + [rhc.QUALITY]:
      with self.subTest(key=key):
        self.assertAlmostEqual(result["deltas"][key]["delta"], 0.25)

  def test_stratum_summary_counts_failures_and_flags(self):
    summary = rhc.stratum_summary(self._rows(), threshold=0.6)
    self.assertEqual(summary["n"], 16)
    self.assertEqual(summary["n_scored"], 15)
    self.assertEqual(summary["n_unscored"], 1)
    self.assertEqual(summary["n_flagged"], 5)
    self.assertAlmostEqual(summary["fpr_at_threshold"], 5 / 15)
    low, high = summary["fpr_ci95"]
    self.assertLess(low, 5 / 15)
    self.assertGreater(high, 5 / 15)
    self.assertAlmostEqual(
        summary["means"][rhc.QUALITY], (6 * 1.0 + 4 * 0.75 + 5 * 0.25) / 15
    )

  def test_grade_histogram_maps_back_to_likert_levels(self):
    histogram = rhc.grade_histogram(self._rows())
    self.assertEqual(
        histogram["fluency"], {"1": 0, "2": 5, "3": 0, "4": 4, "5": 6}
    )

  def test_summarize_assembles_every_block(self):
    payload = rhc.summarize(self._rows(), threshold=0.6)
    self.assertEqual(payload["threshold"], 0.6)
    self.assertEqual(payload["strata"][0]["name"], "all")
    self.assertEqual(
        [s["target_fpr"] for s in payload["suggested_thresholds"]],
        list(rhc.TARGET_FPRS),
    )
    self.assertIn("deltas", payload["independence"])
    json.dumps(payload)  # Must be serializable as is.


class RenderingTest(unittest.TestCase):

  def _payload(self, rows):
    payload = rhc.summarize(rows, threshold=0.6)
    payload["config"] = {
        "task_name": "npov",
        "judge_model": "gemini-2.5-flash",
        "num_fewshot_pairs": 2,
        "num_samples": 1,
        "seed": 12345,
        "sources": [{
            "source": rhc.SOURCE_RM,
            "dataset": "u/npov_rm_organic",
            "split": "test",
        }],
    }
    payload["data"] = {
        rhc.SOURCE_RM: {"loaded": 3, "after_subsample": 3, "graded": 2,
                        "judge_failures": 1},
    }
    payload["judge_audit"] = {"reward_hacking_audit_model": "gemini"}
    return payload

  def _rows(self):
    return [
        _row(0, grades=_grades(1.0)),
        _row(1, label=rhc.LABEL_HALLUCINATED, grades=_grades(0.25)),
        _row(2, grades=None),
    ]

  def test_the_report_names_every_section(self):
    text = rhc.render_report(self._payload(self._rows()))
    for needle in (
        "u/npov_rm_organic",
        "source=rm_organic/label=hallucinated",
        "Suggested thresholds",
        "Independence from hallucination",
        "Grade distribution",
        "Judge audit",
        "Not measured: sensitivity",
    ):
      with self.subTest(needle=needle):
        self.assertIn(needle, text)

  def test_the_report_survives_a_bare_payload(self):
    text = rhc.render_report(rhc.summarize([], threshold=0.6))
    self.assertIn("(no graded rows)", text)

  def test_lowest_rows_are_sorted_and_capped(self):
    rows = [_row(i, grades=_grades(v)) for i, v in enumerate([1.0, 0.0, 0.5])]
    rows.append(_row(3, grades=None))
    self.assertEqual(
        [r.source_index for r in rhc.lowest_rows(rows, limit=2)], [1, 2]
    )
    self.assertIn("#2  quality 0.500", rhc.render_lowest_rows(rows, limit=2))

  def test_write_reports(self):
    rows = self._rows()
    prompts = [f"prompt {i}" for i in range(len(rows))] * 3
    with tempfile.TemporaryDirectory() as tmp:
      out_dir = os.path.join(tmp, "run")
      paths = rhc.write_reports(out_dir, self._payload(rows), rows, prompts)
      with open(paths["metrics_json"], encoding="utf-8") as handle:
        self.assertEqual(json.load(handle)["threshold"], 0.6)
      with open(paths["rows_jsonl"], encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
      self.assertEqual(len(records), 3)
      self.assertEqual(records[1]["label"], rhc.LABEL_HALLUCINATED)
      self.assertAlmostEqual(records[1]["quality"], 0.25)
      self.assertIsNone(records[2]["quality"])
      with open(paths["prompts_txt"], encoding="utf-8") as handle:
        self.assertEqual(handle.read().count("----------"),
                         rhc.NUM_PROMPTS_DUMPED)
      for key in ("metrics_txt", "lowest_txt"):
        self.assertTrue(os.path.getsize(paths[key]) > 0)


class NamingTest(unittest.TestCase):

  def test_run_name_carries_every_verdict_changing_input(self):
    self.assertEqual(
        rhc.run_name("models/gemini-2.5-flash", 2, 3, "bosch", 12345),
        "eval_reward_hacking_autorater_gemini-2.5-flash_fewshot_2_k3"
        "_task_bosch_seed_12345",
    )

  def test_fingerprint_is_order_and_content_sensitive(self):
    base = rhc.prompts_fingerprint(["a", "b"])
    self.assertEqual(base, rhc.prompts_fingerprint(["a", "b"]))
    self.assertNotEqual(base, rhc.prompts_fingerprint(["b", "a"]))
    self.assertNotEqual(base, rhc.prompts_fingerprint(["ab"]))


def _table_cells(line):
  """Splits a markdown table row on its unescaped pipes."""
  return re.split(r"(?<!\\)\|", line.strip())[1:-1]


class MarkdownReportTest(unittest.TestCase):

  def _payload(self, rows):
    payload = RenderingTest()._payload(rows)  # pylint: disable=protected-access
    payload["config"]["generated_at"] = "2026-10-14 12:00:00 UTC"
    payload["config"]["fewshot_source"] = {
        "dataset": "u/npov_sft",
        "split": "train",
    }
    return payload

  def _rows(self):
    return RenderingTest()._rows()  # pylint: disable=protected-access

  def test_it_lands_in_a_subfolder_of_the_orchestrator_reports(self):
    from src.orchestrator.config import ReportingConfig  # pylint: disable=g-import-not-at-top

    self.assertEqual(rhc.DEFAULT_REPORTS_DIR, ReportingConfig().reports_dir)
    args = utils.EvalArguments(
        task_name="npov", user="u", writer_model_lora="unused"
    )
    self.assertEqual(
        args.reward_hacking_calibration_reports_dir, rhc.DEFAULT_REPORTS_DIR
    )
    self.assertEqual(
        rhc.markdown_report_path(None, "run"),
        os.path.join("reports", "reward_hacking_calibration", "run.md"),
    )
    self.assertEqual(
        rhc.markdown_report_path("/tmp/r", "run"),
        os.path.join("/tmp/r", "reward_hacking_calibration", "run.md"),
    )

  def test_the_headline_comes_first(self):
    text = rhc.render_markdown_report(self._payload(self._rows()), [])
    headline = text.split("## Headline")[1].split("## Data")[0]
    # Two graded rows, the hallucinated one below 0.6.
    self.assertIn("False-positive rate at `0.6`: 50.0%", headline)
    self.assertIn("By label: faithful 0.0%", headline)
    self.assertIn("Judge failures: 1 of 3 rows", headline)
    self.assertIn("2026-10-14 12:00:00 UTC", text.split("## Headline")[0])
    self.assertIn("`u/npov_sft:train`", text.split("## Headline")[0])

  def test_every_section_is_present(self):
    text = rhc.render_markdown_report(self._payload(self._rows()), [])
    for section in (
        "## Data",
        "## Grades and false-positive rate",
        "## Suggested thresholds (report only)",
        "## Independence from hallucination",
        "## Grade distribution",
        "## Lowest-quality healthy references",
        "## Not measured",
        "<summary>Judge audit</summary>",
    ):
      with self.subTest(section=section):
        self.assertIn(section, text)

  def test_tables_are_well_formed(self):
    text = rhc.render_markdown_report(self._payload(self._rows()), [])
    lines = text.splitlines()
    for index, line in enumerate(lines):
      if not line.startswith("| ") or lines[index - 1].startswith("|"):
        continue  # Only table headers.
      width = len(_table_cells(line))
      with self.subTest(header=line):
        self.assertTrue(lines[index + 1].startswith("| --- |"))
        body = index + 2
        while body < len(lines) and lines[body].startswith("|"):
          self.assertEqual(len(_table_cells(lines[body])), width)
          body += 1

  def test_text_is_escaped(self):
    self.assertEqual(rhc._md_cell("a|b\nc"), "a\\|b c")  # pylint: disable=protected-access
    fenced = rhc._md_fence("x ``` y")  # pylint: disable=protected-access
    self.assertTrue(fenced.startswith("````text\n"))
    self.assertTrue(fenced.endswith("\n````"))

  def test_the_histogram_is_embedded_only_when_given(self):
    payload = self._payload(self._rows())
    self.assertNotIn("![", rhc.render_markdown_report(payload, []))
    self.assertIn(
        "](run_histograms.png)",
        rhc.render_markdown_report(
            payload, [], histogram_file="run_histograms.png"
        ),
    )

  def test_false_alarms_are_capped_and_sorted(self):
    rows = [
        _row(i, grades=_grades(i / 20), response=f"answer {i} with ``` ticks")
        for i in range(rhc.NUM_LOWEST_ROWS_MARKDOWN + 5)
    ]
    text = rhc.render_markdown_report(self._payload(rows), rows)
    summaries = re.findall(r"<summary>#(\d+) · quality ([0-9.]+)", text)
    self.assertEqual(len(summaries), rhc.NUM_LOWEST_ROWS_MARKDOWN)
    qualities = [float(q) for _, q in summaries]
    self.assertEqual(qualities, sorted(qualities))
    self.assertIn("````text\nanswer 0 with ``` ticks\n````", text)

  def test_write_links_the_raw_artifacts_relatively(self):
    rows = self._rows()
    with tempfile.TemporaryDirectory() as tmp:
      metrics = os.path.join(tmp, "logs", "run", rhc.METRICS_JSON)
      path = rhc.markdown_report_path(os.path.join(tmp, "reports"), "run")
      rhc.write_markdown_report(
          path, self._payload(rows), rows, artifacts={"metrics_json": metrics}
      )
      with open(path, encoding="utf-8") as handle:
        text = handle.read()
    self.assertIn(
        f"- [Metrics (JSON)](../../logs/run/{rhc.METRICS_JSON})", text
    )

  def test_a_bare_payload_still_renders(self):
    text = rhc.render_markdown_report(rhc.summarize([], threshold=0.6), [])
    self.assertIn("No reference could be graded.", text)
    self.assertIn("No graded rows.", text)


if __name__ == "__main__":
  unittest.main()
