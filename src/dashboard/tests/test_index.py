"""Tests for campaign discovery and state-file parsing."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from src.dashboard import index as index_mod
from src.dashboard.tests import fixtures


class DiscoveryTest(unittest.TestCase):
  """Globbing of the checkpoints tree."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

  def test_finds_live_and_archived_campaigns(self):
    fixtures.write_campaign(self.root)
    fixtures.write_campaign(
        self.root,
        state=fixtures.campaign_state(name="npov_campaign_old"),
        archived=True,
    )
    self.assertEqual(len(index_mod.discover_state_files(self.root)), 2)

  def test_archived_can_be_excluded(self):
    fixtures.write_campaign(self.root)
    fixtures.write_campaign(
        self.root,
        state=fixtures.campaign_state(name="npov_campaign_old"),
        archived=True,
    )
    found = index_mod.discover_state_files(self.root, include_archived=False)
    self.assertEqual(len(found), 1)

  def test_missing_checkpoints_directory_is_not_an_error(self):
    self.assertEqual(index_mod.discover_state_files(self.root), [])

  def test_ignores_unrelated_json(self):
    directory = os.path.join(self.root, "checkpoints", "npov")
    os.makedirs(directory)
    with open(
        os.path.join(directory, "checkpoints.json"), "w", encoding="utf-8"
    ) as handle:
      handle.write("{}")
    self.assertEqual(index_mod.discover_state_files(self.root), [])


class LoadCampaignTest(unittest.TestCase):
  """Parsing of a single state file."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
    self.path = fixtures.write_campaign(self.root)
    self.campaign = index_mod.load_campaign(self.path, root=self.root)

  def test_parses_identity_and_config(self):
    self.assertIsNotNone(self.campaign)
    self.assertEqual(self.campaign.campaign_id, "npov_campaign_2609211206")
    self.assertEqual(self.campaign.task, "npov")
    self.assertEqual(self.campaign.status, "COMPLETED")
    self.assertEqual(self.campaign.base_model, "google/gemma-4-E4B-it")
    self.assertEqual(self.campaign.seed, 130104)
    self.assertEqual(self.campaign.rm_dataset_flavors, ["organic"])
    self.assertTrue(self.campaign.is_completed)
    self.assertTrue(self.campaign.is_terminal)

  def test_entity_falls_back_to_user(self):
    self.assertIsNone(self.campaign.wandb_entity)
    self.assertEqual(self.campaign.entity, "leobianco")

  def test_stages_keep_campaign_order(self):
    self.assertEqual(
        [stage.stage_id for stage in self.campaign.stages],
        ["autorater", "sft", "rm", "perl", "eval"],
    )

  def test_stage_metric_names_come_from_config(self):
    sft = self.campaign.stage("sft")
    self.assertEqual(sft.metric_name, "eval/loss")
    self.assertEqual(sft.goal, "minimize")
    self.assertEqual(sft.progress_label, "1/1")
    self.assertTrue(sft.is_sweep)

  def test_non_sweep_stages_have_no_trial_counter(self):
    self.assertEqual(self.campaign.stage("eval").progress_label, "\u2014")
    self.assertFalse(self.campaign.stage("autorater").is_sweep)

  def test_elapsed_is_computed_per_stage(self):
    sft = self.campaign.stage("sft")
    self.assertAlmostEqual(sft.elapsed_seconds, 0.341787, places=4)

  def test_warnings_are_prefixed_with_their_stage(self):
    self.assertEqual(
        self.campaign.warnings, ["Reward model: Sweep finished 1/30 trials"]
    )

  def test_report_path_is_resolved_when_the_file_exists(self):
    self.assertEqual(
        self.campaign.report_path,
        os.path.join("reports", "npov_campaign_2609211206_summary.md"),
    )

  def test_report_path_is_none_when_absent(self):
    root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, root, ignore_errors=True)
    path = fixtures.write_campaign(root, with_report=False)
    campaign = index_mod.load_campaign(path, root=root)
    self.assertIsNone(campaign.report_path)

  def test_to_dict_is_json_serializable(self):
    payload = json.dumps(self.campaign.to_dict())
    self.assertIn("npov_campaign_2609211206", payload)


class DryRunTest(unittest.TestCase):
  """Rehearsals must be distinguishable from real runs.

  A dry run records mock sweep ids, which the dashboard happily turns into
  W&B URLs that go nowhere. Detecting the rehearsal is what lets the page
  say so instead of quietly serving dead links.
  """

  def setUp(self):
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

  def _load(self, state):
    path = os.path.join(
        self.root, "checkpoints", "npov", "npov_campaign_2609211206_state.json"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
      json.dump(state, handle)
    return index_mod.load_campaign(path, root=self.root)

  def test_explicit_flag_is_honoured(self):
    state = fixtures.campaign_state()
    state["config_dict"]["dry_run"] = True
    self.assertTrue(self._load(state).is_dry_run)

  def test_mock_sweep_ids_betray_older_rehearsals(self):
    """States written before ``dry_run`` was persisted still have mock ids."""
    state = fixtures.campaign_state()
    state["config_dict"].pop("dry_run", None)
    for stage in state["stages"].values():
      if stage.get("sweep_id"):
        stage["sweep_id"] = "leobianco/new_perl/mock_sweep_1789992366"
    self.assertTrue(self._load(state).is_dry_run)

  def test_a_real_campaign_is_not_flagged(self):
    state = fixtures.campaign_state()
    state["config_dict"]["dry_run"] = False
    for stage in state["stages"].values():
      if stage.get("sweep_id"):
        stage["sweep_id"] = stage["sweep_id"].replace("mock_sweep_", "")
    campaign = self._load(state)
    self.assertFalse(campaign.is_dry_run)
    self.assertFalse(campaign.to_dict()["is_dry_run"])


class ResilienceTest(unittest.TestCase):
  """One bad file must never empty an index of good ones."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

  def test_corrupt_json_yields_none(self):
    directory = os.path.join(self.root, "checkpoints", "npov")
    os.makedirs(directory)
    path = os.path.join(directory, "broken_state.json")
    with open(path, "w", encoding="utf-8") as handle:
      handle.write('{"campaign_id": "half-writt')
    self.assertIsNone(index_mod.load_campaign(path, root=self.root))

  def test_missing_file_yields_none(self):
    self.assertIsNone(
        index_mod.load_campaign(os.path.join(self.root, "nope.json"))
    )

  def test_json_without_campaign_id_yields_none(self):
    directory = os.path.join(self.root, "checkpoints", "npov")
    os.makedirs(directory)
    path = os.path.join(directory, "other_state.json")
    with open(path, "w", encoding="utf-8") as handle:
      json.dump({"hello": "world"}, handle)
    self.assertIsNone(index_mod.load_campaign(path, root=self.root))

  def test_index_skips_the_bad_file_and_keeps_the_good_one(self):
    fixtures.write_campaign(self.root)
    directory = os.path.join(self.root, "checkpoints", "npov")
    with open(
        os.path.join(directory, "broken_state.json"), "w", encoding="utf-8"
    ) as handle:
      handle.write("not json at all")
    campaigns = index_mod.build_index(self.root)
    self.assertEqual(len(campaigns), 1)

  def test_unknown_stage_keys_do_not_crash(self):
    state = fixtures.campaign_state()
    state["stages"]["future_stage"] = {"status": "COMPLETED", "novel": 1}
    path = fixtures.write_campaign(self.root, state=state)
    campaign = index_mod.load_campaign(path, root=self.root)
    self.assertIn("future_stage", [s.stage_id for s in campaign.stages])

  def test_stage_missing_from_order_is_still_shown(self):
    state = fixtures.campaign_state()
    state["stages_order"] = ["sft"]
    path = fixtures.write_campaign(self.root, state=state)
    campaign = index_mod.load_campaign(path, root=self.root)
    ids = [stage.stage_id for stage in campaign.stages]
    self.assertEqual(ids[0], "sft")
    self.assertIn("eval", ids)


class RedactionTest(unittest.TestCase):
  """Secrets must not reach the published site."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

  def test_secret_strings_are_masked(self):
    state = fixtures.campaign_state()
    state["config_dict"]["wandb_api_key"] = "abc123"
    state["config_dict"]["eval"]["gemini_token"] = "xyz"
    path = fixtures.write_campaign(self.root, state=state)
    campaign = index_mod.load_campaign(path, root=self.root)
    self.assertNotIn("abc123", json.dumps(campaign.config))
    self.assertNotIn("xyz", json.dumps(campaign.config))

  def test_numeric_fields_named_like_secrets_are_preserved(self):
    state = fixtures.campaign_state()
    path = fixtures.write_campaign(self.root, state=state)
    campaign = index_mod.load_campaign(path, root=self.root)
    # `max_tokens` matches nothing, but this guards the general principle
    # that only non-empty strings are masked.
    self.assertEqual(campaign.config["eval"]["max_tokens"], 250)


class EvalTargetTest(unittest.TestCase):
  """Reshaping of the eval stage's flat metric keys."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
    path = fixtures.write_campaign(self.root)
    self.campaign = index_mod.load_campaign(path, root=self.root)

  def test_targets_are_split_by_label(self):
    labels = [target.label for target in self.campaign.eval_targets]
    self.assertEqual(labels, ["sft", "sft@t0.7", "perl", "delta"])

  def test_temperature_suffix_is_parsed(self):
    target = next(
        t for t in self.campaign.eval_targets if t.label == "sft@t0.7"
    )
    self.assertEqual(target.temperature, 0.7)
    self.assertEqual(target.base_label, "sft")

  def test_temperature_falls_back_to_the_recorded_metric(self):
    target = next(t for t in self.campaign.eval_targets if t.label == "sft")
    self.assertEqual(target.temperature, 0.0)

  def test_delta_is_flagged_and_follows_its_policy(self):
    """The delta is the policy's column, so it sits beside it, not at the end.

    Baselines lead, because they are what everything else is measured
    against.
    """
    self.assertTrue(self.campaign.eval_targets[-1].is_delta)
    self.assertEqual(self.campaign.eval_targets[-2].label, "perl")
    self.assertFalse(any(t.is_delta for t in self.campaign.eval_targets[:2]))

  def test_a_fan_out_pairs_each_flavor_with_its_own_delta(self):
    """Two branches must not interleave: organic's delta is organic's.

    Sorting deltas last put `delta:organic` next to `delta:synthetic_struct`
    and both a table-width away from the numbers they qualify.
    """
    state = fixtures.campaign_state()
    metrics = state["stages"]["eval"]["metrics"]
    for key in [k for k in metrics if k.startswith(("perl/", "delta/"))]:
      namespace, _, suffix = key.partition("/")
      metrics[f"{namespace}:organic/{suffix}"] = metrics.pop(key)
    metrics["perl:synthetic_struct/hallucination_rate"] = 0.081
    metrics["delta:synthetic_struct/hallucination_rate"] = 0.024
    root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, root, ignore_errors=True)
    path = fixtures.write_campaign(root, state=state)
    campaign = index_mod.load_campaign(path, root=root)
    labels = [target.label for target in campaign.eval_targets]
    self.assertEqual(
        labels,
        [
            "sft",
            "sft@t0.7",
            "perl:organic",
            "delta:organic",
            "perl:synthetic_struct",
            "delta:synthetic_struct",
        ],
    )

  def test_targets_are_attributed_to_their_model(self):
    perl = next(t for t in self.campaign.eval_targets if t.label == "perl")
    self.assertEqual(perl.model_repo_id, fixtures.PERL_REPO)
    sft = next(t for t in self.campaign.eval_targets if t.label == "sft")
    self.assertEqual(sft.model_repo_id, fixtures.SFT_REPO)

  def test_metric_names_exclude_bookkeeping_keys(self):
    self.assertIn("hallucination_rate", self.campaign.eval_metric_names)
    self.assertNotIn("num_samples", self.campaign.eval_metric_names)

  def test_decoding_temperature_leads_the_table(self):
    """It is a setting, not a measurement, but the table must state it.

    Every other row is meaningless without knowing the decoding regime the
    completions were sampled under, so it earns the first row rather than a
    hand-written one bolted on by the renderer.
    """
    names = self.campaign.eval_metric_names
    self.assertEqual(names[0], "decoding_temperature")
    self.assertIn("decoding_temperature", self.campaign.headline_metric_names)

  def test_provenance_and_audit_keys_never_reach_a_table(self):
    """Their values are adapter repo ids, which made the table unreadable."""
    for name in (
        "provenance_sft_adapter",
        "provenance_policy_dir",
        "autorater_score_mean",
    ):
      self.assertFalse(index_mod.is_tabulated_metric(name), name)
    # ... but the handful of audit keys the report does tabulate survive.
    self.assertTrue(index_mod.is_tabulated_metric("autorater_accuracy"))

  def test_headline_metrics_are_separated_from_the_rest(self):
    """The wide tail goes behind a disclosure, not into the main table."""
    self.assertIn("hallucination_rate", self.campaign.headline_metric_names)
    self.assertNotIn("hallucination_rate", self.campaign.secondary_metric_names)
    combined = (
        self.campaign.headline_metric_names
        + self.campaign.secondary_metric_names
    )
    self.assertCountEqual(combined, self.campaign.eval_metric_names)

  def test_headline_reports_the_perl_rate_and_delta(self):
    headline = self.campaign.headline
    self.assertAlmostEqual(headline["hallucination_rate"], 0.062)
    self.assertAlmostEqual(headline["hallucination_delta"], 0.043)

  def test_campaign_without_eval_has_no_targets(self):
    state = fixtures.campaign_state(name="npov_no_eval")
    del state["stages"]["eval"]
    state["stages_order"] = ["sft"]
    path = fixtures.write_campaign(self.root, state=state)
    campaign = index_mod.load_campaign(path, root=self.root)
    self.assertEqual(campaign.eval_targets, [])
    self.assertIsNone(campaign.headline["hallucination_rate"])


class MetricRatioTest(unittest.TestCase):
  """Each delta re-expressed as a share of the baseline it improved on."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

  def _campaign(self, extra=None):
    """Loads a campaign with extra eval metrics.

    Args:
      extra: Additional eval metrics to record.

    Returns:
      The loaded campaign summary.
    """
    state = fixtures.campaign_state()
    state["stages"]["eval"]["metrics"].update(extra or {})
    root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, root, ignore_errors=True)
    path = fixtures.write_campaign(root, state=state)
    return index_mod.load_campaign(path, root=root)

  def _delta(self, campaign, label="delta"):
    """Returns a campaign's delta target by label.

    Args:
      campaign: The campaign.
      label: The delta column's label.

    Returns:
      The matching eval target.
    """
    return next(t for t in campaign.eval_targets if t.label == label)

  def test_the_ratio_uses_the_temperature_matched_baseline(self):
    """0.105 -> 0.062, not 0.091 -> 0.062.

    The fixture has two SFT rows; the policy was scored at 0.7, so the
    greedy row must not be what the percentage is taken against.
    """
    delta = self._delta(self._campaign())
    self.assertAlmostEqual(
        delta.metric_ratios["hallucination_rate"], 0.043 / 0.105
    )

  def test_a_higher_is_better_metric_is_inverted_the_other_way(self):
    delta = self._delta(self._campaign())
    self.assertAlmostEqual(delta.metric_ratios["bertscore_f1"], 0.018 / 0.874)

  def test_each_branch_uses_its_own_baseline(self):
    """The two flavors were scored at different temperatures.

    Organic improves 0.105 -> 0.062 and synthetic 0.200 -> 0.100; the
    absolute deltas differ, and so must the shares.
    """
    campaign = self._campaign({
        "sft@t1.0/hallucination_rate": 0.200,
        "sft@t1.0/decoding_temperature": 1.0,
        "perl:synthetic_struct/hallucination_rate": 0.100,
        "perl:synthetic_struct/decoding_temperature": 1.0,
        "delta:synthetic_struct/hallucination_rate": 0.100,
    })
    self.assertAlmostEqual(
        self._delta(campaign).metric_ratios["hallucination_rate"],
        0.043 / 0.105,
    )
    self.assertAlmostEqual(
        self._delta(campaign, "delta:synthetic_struct").metric_ratios[
            "hallucination_rate"
        ],
        0.5,
    )

  def test_only_delta_columns_carry_a_ratio(self):
    campaign = self._campaign()
    for target in campaign.eval_targets:
      if not target.is_delta:
        self.assertEqual(target.metric_ratios, {}, target.label)

  def test_a_zero_baseline_leaves_the_cell_empty(self):
    """A metric that was already perfect has no share to improve on."""
    campaign = self._campaign({
        "sft@t0.7/autorater_n_dropped": 0,
        "perl/autorater_n_dropped": 0,
        "delta/autorater_n_dropped": 0.0,
    })
    self.assertNotIn(
        "autorater_n_dropped", self._delta(campaign).metric_ratios
    )

  def test_an_orphan_delta_is_left_alone(self):
    """A delta whose policy column is missing has nothing to invert."""
    state = fixtures.campaign_state()
    metrics = state["stages"]["eval"]["metrics"]
    for key in [k for k in metrics if k.startswith("perl/")]:
      del metrics[key]
    path = fixtures.write_campaign(self.root, state=state)
    campaign = index_mod.load_campaign(path, root=self.root)
    self.assertEqual(self._delta(campaign).metric_ratios, {})


class BranchedCampaignTest(unittest.TestCase):
  """Dataset-flavor branches produce ``kind:flavor`` stage ids."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
    state = fixtures.campaign_state(name="ragtruth_branched")
    state["task_name"] = "ragtruth"
    state["config_dict"]["task_name"] = "ragtruth"
    state["config_dict"]["rm_dataset_flavors"] = ["organic", "synthetic_struct"]
    state["stages"]["rm:organic"] = state["stages"].pop("rm")
    state["stages"]["rm:synthetic_struct"] = dict(state["stages"]["rm:organic"])
    state["stages"]["perl:organic"] = state["stages"].pop("perl")
    state["stages_order"] = [
        "sft",
        "rm:organic",
        "rm:synthetic_struct",
        "perl:organic",
        "eval",
    ]
    path = fixtures.write_campaign(self.root, state=state, task="ragtruth")
    self.campaign = index_mod.load_campaign(path, root=self.root)

  def test_stage_id_splits_into_kind_and_flavor(self):
    stage = self.campaign.stage("rm:synthetic_struct")
    self.assertEqual(stage.kind, "rm")
    self.assertEqual(stage.flavor, "synthetic_struct")

  def test_branch_title_names_the_flavor(self):
    self.assertEqual(
        self.campaign.stage("rm:organic").title, "Reward model (organic)"
    )

  def test_stages_of_kind_collects_every_branch(self):
    self.assertEqual(len(self.campaign.stages_of_kind("rm")), 2)

  def test_branch_stage_reads_its_kind_config(self):
    self.assertEqual(
        self.campaign.stage("rm:organic").metric_name, "eval/roc_auc"
    )


class IndexOrderingTest(unittest.TestCase):
  """The list is newest-first and slugs are unique."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

  def test_sorted_by_updated_at_descending(self):
    fixtures.write_campaign(
        self.root,
        state=fixtures.campaign_state(
            name="npov_older", updated_at="2026-09-20T10:00:00"
        ),
    )
    fixtures.write_campaign(
        self.root,
        state=fixtures.campaign_state(
            name="npov_newer", updated_at="2026-09-22T10:00:00"
        ),
    )
    campaigns = index_mod.build_index(self.root)
    self.assertEqual(
        [c.campaign_id for c in campaigns], ["npov_newer", "npov_older"]
    )

  def test_archived_and_live_namesakes_get_distinct_slugs(self):
    fixtures.write_campaign(self.root)
    fixtures.write_campaign(self.root, archived=True)
    campaigns = index_mod.build_index(self.root)
    slugs = {campaign.slug for campaign in campaigns}
    self.assertEqual(len(slugs), 2)
    self.assertEqual(len(campaigns), 2)

  def test_archive_flag_is_set(self):
    fixtures.write_campaign(self.root, archived=True)
    campaign = index_mod.build_index(self.root)[0]
    self.assertTrue(campaign.is_archived)

  def test_slug_is_filesystem_safe(self):
    fixtures.write_campaign(self.root)
    campaign = index_mod.build_index(self.root)[0]
    self.assertRegex(campaign.slug, r"^[A-Za-z0-9-]+$")


if __name__ == "__main__":
  unittest.main()
