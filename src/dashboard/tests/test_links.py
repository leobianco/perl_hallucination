"""Tests for outbound W&B and Hugging Face link construction.

The dataset ids asserted here are **golden values**: they were produced by
:func:`src.utils.build_eval_dataset_repo_id` against a real ``npov`` campaign.
If a change to the naming convention breaks them, that is the point — the
links of every previously published campaign break with them, and this test is
the only thing that says so out loud.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest

from src.dashboard import index as index_mod
from src.dashboard import links as links_mod
from src.dashboard.tests import fixtures

GOLDEN_SFT_GREEDY = (
    "leobianco/eval_npov_SFT_gemma-4-E4B-it_S130104_epo1_lr2_5e_5a90"
    "_gens_T0_wfs0_s12345_mt250_nosft"
)
GOLDEN_SFT_MATCHED = (
    "leobianco/eval_npov_SFT_gemma-4-E4B-it_S130104_epo1_lr2_5_5a90"
    "_gens_T0_7_wfs0_s12345_mt250_nosft"
)
GOLDEN_PERL = (
    "leobianco/eval_npov_PERL_organic_gemma-4-E4B-it_S1301_68fd"
    "_gens_T0_7_wfs0_s12345_mt250_sftaaccba"
)


def _load(root: str, **kwargs) -> index_mod.CampaignSummary:
  """Writes a campaign into ``root`` and returns its parsed summary.

  Args:
    root: Temporary repository root.
    **kwargs: Forwarded to :func:`fixtures.write_campaign`.

  Returns:
    The parsed campaign.
  """
  path = fixtures.write_campaign(root, **kwargs)
  return index_mod.load_campaign(path, root=root)


class WandbLinkTest(unittest.TestCase):
  """Sweep, run and project URLs."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
    self.campaign = _load(self.root)

  def test_qualified_sweep_id_is_used_verbatim(self):
    url = links_mod.sweep_url(self.campaign, self.campaign.stage("sft"))
    self.assertEqual(url, "https://wandb.ai/leobianco/new_perl/sweeps/abc123")

  def test_bare_sweep_id_is_qualified_from_the_campaign(self):
    url = links_mod.sweep_url(self.campaign, self.campaign.stage("rm"))
    self.assertEqual(url, "https://wandb.ai/leobianco/new_perl/sweeps/def456")

  def test_stage_without_a_sweep_has_no_link(self):
    self.assertIsNone(
        links_mod.sweep_url(self.campaign, self.campaign.stage("eval"))
    )

  def test_run_url_points_at_the_winning_trial(self):
    url = links_mod.run_url(self.campaign, self.campaign.stage("perl"))
    self.assertEqual(
        url, "https://wandb.ai/leobianco/new_perl/runs/run_1789992368"
    )

  def test_project_and_eval_project(self):
    self.assertEqual(
        links_mod.project_url(self.campaign),
        "https://wandb.ai/leobianco/new_perl",
    )
    self.assertEqual(
        links_mod.eval_project_url(self.campaign),
        "https://wandb.ai/leobianco/new_perl_eval",
    )

  def test_explicit_entity_overrides_the_user_fallback(self):
    state = fixtures.campaign_state(name="npov_entity")
    state["config_dict"]["wandb_entity"] = "some-team"
    campaign = _load(self.root, state=state)
    self.assertEqual(
        links_mod.project_url(campaign), "https://wandb.ai/some-team/new_perl"
    )
    self.assertEqual(
        links_mod.sweep_url(campaign, campaign.stage("rm")),
        "https://wandb.ai/some-team/new_perl/sweeps/def456",
    )

  def test_qualified_sweep_id_wins_over_the_campaign_entity(self):
    # The recorded path is what W&B actually registered; trusting the config
    # instead would point at a project the sweep never lived in.
    state = fixtures.campaign_state(name="npov_mismatch")
    state["config_dict"]["wandb_entity"] = "some-team"
    campaign = _load(self.root, state=state)
    self.assertEqual(
        links_mod.sweep_url(campaign, campaign.stage("sft")),
        "https://wandb.ai/leobianco/new_perl/sweeps/abc123",
    )


class HubLinkTest(unittest.TestCase):
  """Model and dataset URLs."""

  def test_model_and_files(self):
    self.assertEqual(
        links_mod.model_url("leobianco/npov_SFT"),
        "https://huggingface.co/leobianco/npov_SFT",
    )
    self.assertEqual(
        links_mod.model_files_url("leobianco/npov_SFT"),
        "https://huggingface.co/leobianco/npov_SFT/tree/main",
    )

  def test_dataset(self):
    self.assertEqual(
        links_mod.dataset_url("leobianco/npov_rm_organic"),
        "https://huggingface.co/datasets/leobianco/npov_rm_organic",
    )

  def test_empty_repo_ids_yield_no_link(self):
    self.assertIsNone(links_mod.model_url(None))
    self.assertIsNone(links_mod.dataset_url(""))


class EvalDatasetTest(unittest.TestCase):
  """Recomputation of the completions dataset ids."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
    self.campaign = _load(self.root)

  def _repo_for(self, label: str):
    target = next(
        t for t in self.campaign.eval_targets if t.label == label
    )
    return links_mod.eval_dataset_repo_id(self.campaign, target)

  def test_greedy_sft_baseline(self):
    self.assertEqual(self._repo_for("sft"), GOLDEN_SFT_GREEDY)

  def test_temperature_matched_sft_baseline(self):
    self.assertEqual(self._repo_for("sft@t0.7"), GOLDEN_SFT_MATCHED)

  def test_perl_policy_records_the_stacked_sft_adapter(self):
    repo = self._repo_for("perl")
    self.assertEqual(repo, GOLDEN_PERL)
    self.assertTrue(repo.endswith("_sftaaccba"))

  def test_sft_baseline_stacks_nothing(self):
    self.assertTrue(self._repo_for("sft").endswith("_nosft"))

  def test_delta_column_has_no_dataset(self):
    self.assertIsNone(self._repo_for("delta"))

  def test_target_without_a_model_has_no_dataset(self):
    target = index_mod.EvalTarget(
        label="ghost", base_label="ghost", temperature=0.0, is_delta=False
    )
    self.assertIsNone(
        links_mod.eval_dataset_repo_id(self.campaign, target)
    )


class TrainingDatasetTest(unittest.TestCase):
  """The datasets the stages trained on, mirrored from their sweep commands."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

  def test_unbranched_campaign(self):
    campaign = _load(self.root)
    pairs = dict(
        (label, repo)
        for label, repo in links_mod.training_dataset_repo_ids(campaign)
    )
    self.assertEqual(pairs["SFT training set"], "leobianco/npov_sft")
    self.assertEqual(pairs["PE-RL prompt set"], "leobianco/npov_perl")
    self.assertEqual(
        pairs["Autorater calibration set"], "leobianco/npov_autorater"
    )
    self.assertEqual(
        pairs["Reward model training set (organic)"],
        "leobianco/npov_rm_organic",
    )

  def test_branched_campaign_lists_one_dataset_per_branch(self):
    state = fixtures.campaign_state(name="ragtruth_branched")
    state["task_name"] = "ragtruth"
    state["config_dict"]["task_name"] = "ragtruth"
    state["config_dict"]["rm_dataset_flavors"] = ["organic", "synthetic_struct"]
    state["stages"]["rm:organic"] = state["stages"].pop("rm")
    state["stages"]["rm:synthetic_struct"] = dict(
        state["stages"]["rm:organic"]
    )
    state["stages_order"] = [
        "sft",
        "rm:organic",
        "rm:synthetic_struct",
        "perl",
        "eval",
    ]
    campaign = _load(self.root, state=state, task="ragtruth")
    repos = [repo for _, repo in links_mod.training_dataset_repo_ids(campaign)]
    self.assertIn("leobianco/ragtruth_rm_organic", repos)
    self.assertIn("leobianco/ragtruth_rm_synthetic_struct", repos)

  def test_stage_that_did_not_run_contributes_no_dataset(self):
    state = fixtures.campaign_state(name="npov_no_autorater")
    del state["stages"]["autorater"]
    state["stages_order"] = ["sft", "rm", "perl", "eval"]
    campaign = _load(self.root, state=state)
    labels = [
        label for label, _ in links_mod.training_dataset_repo_ids(campaign)
    ]
    self.assertNotIn("Autorater calibration set", labels)


class LinkGroupTest(unittest.TestCase):
  """Assembly of the detail page's link panel."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
    self.groups = links_mod.link_groups(_load(self.root))

  def test_groups_are_titled_and_ordered(self):
    self.assertEqual(
        [group.title for group in self.groups],
        [
            "Weights & Biases",
            "Hugging Face models",
            "Hugging Face generations",
            "Hugging Face datasets",
        ],
    )

  def test_every_link_has_a_url(self):
    for group in self.groups:
      for link in group.links:
        self.assertTrue(link.url.startswith("https://"), link)

  def test_derived_links_are_flagged(self):
    generations = next(
        g for g in self.groups if g.title == "Hugging Face generations"
    )
    self.assertTrue(all(link.derived for link in generations.links))

  def test_sweep_links_are_not_flagged_as_derived(self):
    wandb = next(g for g in self.groups if g.title == "Weights & Biases")
    self.assertFalse(any(link.derived for link in wandb.links))

  def test_generation_links_note_their_temperature(self):
    generations = next(
        g for g in self.groups if g.title == "Hugging Face generations"
    )
    notes = {link.note for link in generations.links}
    self.assertIn("greedy", notes)
    self.assertIn("T=0.7", notes)

  def test_eval_stage_does_not_duplicate_the_perl_adapter(self):
    models = next(g for g in self.groups if g.title == "Hugging Face models")
    labels = [link.label for link in models.links]
    self.assertNotIn("Evaluation adapter", labels)
    self.assertIn("PE-RL adapter", labels)

  def test_campaign_with_nothing_published_yields_no_groups(self):
    state = fixtures.campaign_state(name="npov_empty", status="IN_PROGRESS")
    state["stages"] = {
        "sft": {"status": "RUNNING", "trials_done": 2, "trials_total": 30}
    }
    state["stages_order"] = ["sft"]
    state["config_dict"]["user"] = ""
    state["config_dict"]["project"] = ""
    state["config_dict"]["eval"]["wandb_project"] = None
    campaign = _load(self.root, state=state, with_report=False)
    self.assertEqual(links_mod.link_groups(campaign), [])


if __name__ == "__main__":
  unittest.main()
