"""Tests for reward-model dataset flavors and the campaign fan-out.

Two properties are worth more than all the rest here:

1. A single-flavor campaign must be *indistinguishable* from a campaign
   written before flavors existed - same stage ids, same state keys, same
   report headings - so that in-flight campaigns resume.
2. A two-flavor campaign must keep the branches separate all the way
   through: two datasets, two reward models, two policies, two deltas, and
   never a value from one branch attributed to the other.
"""

import os
import tempfile
import unittest

from src.orchestrator import eval_metrics
from src.orchestrator import flavors
from src.orchestrator.config import CampaignConfig, VALID_STAGES
from src.orchestrator.model_manager import ModelManager
from src.orchestrator.reporter import CampaignReporter
from src.orchestrator.stages.base import CampaignContext
from src.orchestrator.stages.eval_stage import EvalStage
from src.orchestrator.stages.eval_stage import compute_deltas
from src.orchestrator.stages.perl_stage import PerlStage
from src.orchestrator.stages.rm_stage import RmStage
from src.orchestrator.state import CampaignState
from src.orchestrator.state import StageResult
from src.orchestrator.state import StageStatus
from src.orchestrator.sweep_controller import SweepController


def _config(flavor_list=None, **kwargs):
  config = CampaignConfig.create_default(
      task_name="ragtruth",
      dry_run=True,
      rm_dataset_flavors=flavor_list,
      **kwargs,
  )
  config.user = "leobianco"
  return config


def _context(config, state, state_path=None):
  return CampaignContext(
      config=config,
      state=state,
      sweep_controller=SweepController(dry_run=True),
      model_manager=ModelManager(dry_run=True),
      state_path=state_path,
  )


class VocabularyTest(unittest.TestCase):
  """The flavor vocabulary itself."""

  def test_dataset_repo_id_matches_the_shell_script_convention(self):
    self.assertEqual(
        flavors.dataset_repo_id("leobianco", "ragtruth", flavors.ORGANIC),
        "leobianco/ragtruth_rm_organic",
    )
    self.assertEqual(
        flavors.dataset_repo_id(
            "leobianco", "ragtruth", flavors.SYNTHETIC_STRUCT
        ),
        "leobianco/ragtruth_rm_synthetic_struct",
    )

  def test_synthetic_llm_is_not_selectable(self):
    # It is the one flavor whose split is assembled at load time from two
    # datasets, via knobs this vocabulary cannot express.
    self.assertNotIn("synthetic_llm", flavors.RM_DATASET_FLAVORS)

  def test_normalize_puts_known_flavors_in_execution_order(self):
    self.assertEqual(
        flavors.normalize_flavors([" SYNTHETIC_STRUCT ", "organic", "organic"]),
        [flavors.ORGANIC, flavors.SYNTHETIC_STRUCT],
    )

  def test_normalize_keeps_unknown_entries_so_validation_can_name_them(self):
    self.assertEqual(flavors.normalize_flavors(["banana"]), ["banana"])

  def test_a_bare_string_is_accepted(self):
    self.assertEqual(
        flavors.normalize_flavors("synthetic_struct"),
        [flavors.SYNTHETIC_STRUCT],
    )

  def test_stage_ids_round_trip(self):
    stage_id = flavors.make_stage_id("rm", flavors.SYNTHETIC_STRUCT, True)
    self.assertEqual(stage_id, "rm:synthetic_struct")
    self.assertEqual(
        flavors.split_stage_id(stage_id), ("rm", flavors.SYNTHETIC_STRUCT)
    )
    self.assertEqual(flavors.split_stage_id("rm"), ("rm", None))

  def test_log_tags_are_unchanged_for_unbranched_stages(self):
    self.assertEqual(flavors.log_tag("sft"), "SFT ")
    self.assertEqual(flavors.log_tag("perl"), "PERL")
    self.assertEqual(flavors.log_tag(None), "RUN ")

  def test_log_tags_distinguish_the_branches(self):
    # A plain truncation to four characters turned both PE-RL branches into
    # "PERL", leaving every log line ambiguous about its branch.
    self.assertNotEqual(
        flavors.log_tag("perl:organic"),
        flavors.log_tag("perl:synthetic_struct"),
    )
    self.assertEqual(flavors.log_tag("perl:organic"), "PE:O")
    self.assertEqual(flavors.log_tag("rm:synthetic_struct"), "RM:S")

  def test_log_tags_are_always_the_requested_width(self):
    for stage_id in ("sft", "rm", "perl:organic", "eval", None):
      self.assertEqual(len(flavors.log_tag(stage_id)), 4)


class SingleFlavorIsUnchangedTest(unittest.TestCase):
  """One flavor must leave no trace of the branching machinery."""

  def test_stage_ids_stay_bare(self):
    plan = flavors.build_plan(_config([flavors.ORGANIC]))
    self.assertEqual([p.stage_id for p in plan], list(VALID_STAGES))

  def test_a_lone_synthetic_campaign_also_keeps_bare_ids(self):
    # The suffix disambiguates *between branches*. With one branch there is
    # nothing to disambiguate, and a 'rm:synthetic_struct' key would make
    # the campaign unresumable from a state file written by an older build.
    plan = flavors.build_plan(_config([flavors.SYNTHETIC_STRUCT]))
    self.assertEqual([p.stage_id for p in plan], list(VALID_STAGES))
    self.assertFalse(flavors.is_branched(_config([flavors.SYNTHETIC_STRUCT])))

  def test_titles_stay_bare(self):
    plan = flavors.build_plan(_config(), {"rm": "RM Sweep"})
    titles = {p.kind: p.title for p in plan}
    self.assertEqual(titles["rm"], "RM Sweep")

  def test_default_config_is_organic(self):
    self.assertEqual(
        flavors.campaign_flavors(CampaignConfig.create_default("ragtruth")),
        [flavors.ORGANIC],
    )


class BranchedPlanTest(unittest.TestCase):
  """Two flavors fan the campaign out."""

  def setUp(self):
    super().setUp()
    self.config = _config([flavors.ORGANIC, flavors.SYNTHETIC_STRUCT])

  def test_rm_and_perl_are_duplicated_but_sft_and_eval_are_not(self):
    plan = flavors.build_plan(self.config)
    self.assertEqual(
        [p.stage_id for p in plan],
        [
            "autorater",
            "sft",
            "rm:organic",
            "rm:synthetic_struct",
            "perl:organic",
            "perl:synthetic_struct",
            "eval",
        ],
    )

  def test_expansion_is_kind_major_so_cheap_failures_surface_first(self):
    # Both reward models are trained before either policy: an RM sweep is
    # far cheaper than a PE-RL sweep, and both ROC-AUCs are worth seeing
    # before committing GPU hours to the expensive half.
    ids = [p.stage_id for p in flavors.build_plan(self.config)]
    self.assertLess(ids.index("rm:synthetic_struct"), ids.index("perl:organic"))

  def test_titles_name_the_branch(self):
    plan = flavors.build_plan(self.config, {"rm": "RM Sweep"})
    titles = [p.title for p in plan if p.kind == "rm"]
    self.assertEqual(
        titles, ["RM Sweep (Organic)", "RM Sweep (Synthetic Struct)"]
    )

  def test_branch_ids_ignore_the_stage_selection(self):
    # A `--stages eval` rerun still has to find the branches that earlier
    # invocations produced.
    self.config.stages = ["eval"]
    self.assertEqual(
        flavors.branch_stage_ids(self.config, "perl"),
        ["perl:organic", "perl:synthetic_struct"],
    )


class ConfigValidationTest(unittest.TestCase):
  """Bad flavor configurations fail before any GPU time is spent."""

  def test_unknown_flavor_is_rejected_by_name(self):
    config = _config()
    config.rm_dataset_flavors = ["organic", "synthetic_llm"]
    with self.assertRaises(ValueError) as ctx:
      config.validate()
    self.assertIn("synthetic_llm", str(ctx.exception))

  def test_empty_flavor_list_is_rejected(self):
    config = _config()
    config.rm_dataset_flavors = []
    with self.assertRaises(ValueError):
      config.validate()

  def test_two_flavors_without_an_rm_stage_is_rejected(self):
    # Both branches would fall back to the single reward_model_path
    # override, producing two identical policies reported as a comparison.
    config = _config([flavors.ORGANIC, flavors.SYNTHETIC_STRUCT])
    config.stages = ["perl", "eval"]
    config.perl.sft_model_path = "u/sft"
    config.perl.reward_model_path = "u/rm"
    with self.assertRaises(ValueError) as ctx:
      config.validate()
    self.assertIn("does not run the 'rm' stage", str(ctx.exception))

  def test_one_flavor_without_an_rm_stage_is_fine(self):
    config = _config([flavors.SYNTHETIC_STRUCT])
    config.stages = ["perl", "eval"]
    config.perl.sft_model_path = "u/sft"
    config.perl.reward_model_path = "u/rm"
    config.validate()

  def test_flavors_survive_a_yaml_round_trip(self):
    config = _config([flavors.ORGANIC, flavors.SYNTHETIC_STRUCT])
    with tempfile.TemporaryDirectory() as temp_dir:
      path = os.path.join(temp_dir, "campaign.yaml")
      config.to_yaml(path)
      restored = CampaignConfig.from_yaml(path)
    self.assertEqual(
        restored.rm_dataset_flavors,
        [flavors.ORGANIC, flavors.SYNTHETIC_STRUCT],
    )


class RmDatasetTest(unittest.TestCase):
  """The RM stage trains on the dataset its branch names."""

  def _stage(self, flavor_list, flavor):
    config = _config(flavor_list)
    state = CampaignState(campaign_id="c", task_name="ragtruth")
    stage_id = flavors.stage_id_for_flavor(config, "rm", flavor)
    return RmStage(_context(config, state), flavor=flavor, stage_id=stage_id)

  def test_synthetic_only_campaign_uses_the_synthetic_dataset(self):
    # This is the bug the feature exists for: editing sweep_rm.yaml had no
    # effect because the stage overwrote --dataset_repo_id with the organic
    # dataset on the way past.
    stage = self._stage([flavors.SYNTHETIC_STRUCT], flavors.SYNTHETIC_STRUCT)
    self.assertEqual(
        stage.dataset_repo_id(), "leobianco/ragtruth_rm_synthetic_struct"
    )

  def test_the_sweep_command_is_rewritten_to_that_dataset(self):
    stage = self._stage([flavors.SYNTHETIC_STRUCT], flavors.SYNTHETIC_STRUCT)
    sweep = {"command": ["--dataset_repo_id=leobianco/npov_rm_organic"]}
    _, descriptor = stage.get_sweep_descriptor(sweep)
    self.assertEqual(descriptor, "RM Synthetic Struct")

  def test_the_two_branches_read_different_datasets(self):
    both = [flavors.ORGANIC, flavors.SYNTHETIC_STRUCT]
    self.assertNotEqual(
        self._stage(both, flavors.ORGANIC).dataset_repo_id(),
        self._stage(both, flavors.SYNTHETIC_STRUCT).dataset_repo_id(),
    )

  def test_branch_stage_ids_are_distinct_state_keys(self):
    both = [flavors.ORGANIC, flavors.SYNTHETIC_STRUCT]
    self.assertEqual(
        self._stage(both, flavors.SYNTHETIC_STRUCT).name,
        "rm:synthetic_struct",
    )
    self.assertEqual(self._stage(both, flavors.ORGANIC).name, "rm:organic")


class PerlBranchTest(unittest.TestCase):
  """Each PE-RL branch optimises against its own reward model."""

  def setUp(self):
    super().setUp()
    self.config = _config([flavors.ORGANIC, flavors.SYNTHETIC_STRUCT])
    self.state = CampaignState(campaign_id="c", task_name="ragtruth")
    self.state.stages["sft"] = StageResult(
        status=StageStatus.COMPLETED, model_repo_id="u/sft"
    )
    self.state.stages["rm:organic"] = StageResult(
        status=StageStatus.COMPLETED, model_repo_id="u/rm_organic"
    )
    self.state.stages["rm:synthetic_struct"] = StageResult(
        status=StageStatus.COMPLETED, model_repo_id="u/rm_synstruct"
    )
    self.context = _context(self.config, self.state)

  def test_each_branch_resolves_its_own_reward_model(self):
    self.assertEqual(
        self.context.reward_model_repo_id_for(flavors.ORGANIC), "u/rm_organic"
    )
    self.assertEqual(
        self.context.reward_model_repo_id_for(flavors.SYNTHETIC_STRUCT),
        "u/rm_synstruct",
    )

  def test_branches_are_not_crossed(self):
    stage = PerlStage(
        self.context,
        flavor=flavors.SYNTHETIC_STRUCT,
        stage_id="perl:synthetic_struct",
    )
    _, descriptor = stage.get_sweep_descriptor({})
    self.assertEqual(descriptor, "PERL Synthetic Struct")

  def test_an_explicit_override_still_wins(self):
    self.config.perl.reward_model_path = "u/pinned_rm"
    self.assertEqual(
        self.context.reward_model_repo_id_for(flavors.ORGANIC), "u/pinned_rm"
    )


class RepoNamingTest(unittest.TestCase):
  """Published checkpoints say which dataset produced them."""

  def _plan(self, stage_name, flavor):
    manager = ModelManager(dry_run=True, user="leobianco")
    return manager.build_materialization_command(
        stage_name=stage_name,
        task_name="ragtruth",
        base_model="google/gemma-3-4b-it",
        best_params={"learning_rate": 1e-5},
        seed=42,
        sft_model_path="u/sft",
        reward_model_path="u/rm",
        flavor=flavor,
        timestamp="2401010000",
    )

  def test_rm_repo_carries_the_flavor_tag(self):
    self.assertIn(
        "synstruct", self._plan("rm", flavors.SYNTHETIC_STRUCT).repo_id
    )

  def test_perl_repo_carries_the_flavor_of_its_reward_model(self):
    # The policy is only interpretable next to the reward model that shaped
    # it, so the tag has to survive into the PE-RL repo id too.
    self.assertIn(
        "synstruct", self._plan("perl", flavors.SYNTHETIC_STRUCT).repo_id
    )

  def test_the_two_branches_do_not_collide(self):
    self.assertNotEqual(
        self._plan("perl", flavors.ORGANIC).repo_id,
        self._plan("perl", flavors.SYNTHETIC_STRUCT).repo_id,
    )

  def test_the_rm_retrain_reads_the_branch_dataset(self):
    # The winner's retraining used to be hardcoded to '_rm_organic' in a
    # second place, so a synthetic sweep published an organic-trained model.
    command = self._plan("rm", flavors.SYNTHETIC_STRUCT).command
    index = command.index("--dataset_repo_id")
    self.assertEqual(
        command[index + 1], "leobianco/ragtruth_rm_synthetic_struct"
    )


class BranchedEvaluationTest(unittest.TestCase):
  """One evaluation pass scores every branch against the shared baseline."""

  def _stage(self, flavor_list, policies):
    config = _config(flavor_list)
    config.eval.temperature_grid = [0.0]
    state = CampaignState(campaign_id="c", task_name="ragtruth")
    state.stages["sft"] = StageResult(
        status=StageStatus.COMPLETED, model_repo_id="u/sft"
    )
    for stage_id, repo_id in policies.items():
      state.stages[stage_id] = StageResult(
          status=StageStatus.COMPLETED, model_repo_id=repo_id
      )
    return EvalStage(_context(config, state))

  def _pairs(self, stage):
    """Returns each pass as (label, repo), dropping the temperature.

    The fixture pins a one-point temperature grid and records no
    ``best_params``, so the plan has one pass per policy. See
    test_eval_temperature.py for the temperature-grid behaviour.

    Args:
      stage: The eval stage to resolve.

    Returns:
      One (label, model repo id) pair per pass.
    """
    return [(t.label, t.model_repo_id) for t in stage.resolve_targets()]

  def test_every_branch_is_a_target(self):
    stage = self._stage(
        [flavors.ORGANIC, flavors.SYNTHETIC_STRUCT],
        {
            "perl:organic": "u/policy_org",
            "perl:synthetic_struct": "u/policy_syn",
        },
    )
    self.assertEqual(
        self._pairs(stage),
        [
            ("sft", "u/sft"),
            ("perl:organic", "u/policy_org"),
            ("perl:synthetic_struct", "u/policy_syn"),
        ],
    )

  def test_single_flavor_targets_are_unchanged(self):
    stage = self._stage([flavors.ORGANIC], {"perl": "u/policy"})
    self.assertEqual(
        self._pairs(stage), [("sft", "u/sft"), ("perl", "u/policy")]
    )

  def test_each_branch_gets_its_own_delta_namespace(self):
    deltas = compute_deltas({
        "sft/hallucination_rate": 0.10,
        "perl:organic/hallucination_rate": 0.06,
        "perl:synthetic_struct/hallucination_rate": 0.04,
    })
    self.assertAlmostEqual(deltas["delta:organic/hallucination_rate"], 0.04)
    self.assertAlmostEqual(
        deltas["delta:synthetic_struct/hallucination_rate"], 0.06
    )

  def test_unbranched_deltas_keep_the_plain_namespace(self):
    deltas = compute_deltas({
        "sft/hallucination_rate": 0.10,
        "perl/hallucination_rate": 0.04,
    })
    self.assertEqual(list(deltas), ["delta/hallucination_rate"])

  def test_a_branch_never_borrows_the_other_branch_value(self):
    # Only one branch was scored; the other must produce no delta at all
    # rather than silently reusing the first one's improvement.
    deltas = compute_deltas({
        "sft/hallucination_rate": 0.10,
        "perl:organic/hallucination_rate": 0.06,
    })
    self.assertNotIn("delta:synthetic_struct/hallucination_rate", deltas)

  def test_dry_run_scores_both_branches_in_one_pass(self):
    stage = self._stage(
        [flavors.ORGANIC, flavors.SYNTHETIC_STRUCT],
        {
            "perl:organic": "u/policy_org",
            "perl:synthetic_struct": "u/policy_syn",
        },
    )
    result = stage.execute()
    self.assertEqual(result.status, StageStatus.COMPLETED)
    self.assertIn("perl:organic/hallucination_rate", result.metrics)
    self.assertIn("perl:synthetic_struct/hallucination_rate", result.metrics)
    # The mock is per-kind: a branch must not fall back to the SFT numbers.
    self.assertNotEqual(
        result.metrics["perl:organic/hallucination_rate"],
        result.metrics["sft/hallucination_rate"],
    )


class BranchedMetricViewTest(unittest.TestCase):
  """The report helpers discover branches from the metric map alone."""

  METRICS = {
      "sft/hallucination_rate": 0.10,
      "perl:organic/hallucination_rate": 0.06,
      "perl:synthetic_struct/hallucination_rate": 0.04,
      "delta:organic/hallucination_rate": 0.04,
      "delta:synthetic_struct/hallucination_rate": 0.06,
  }

  def test_targets_are_discovered_and_titled(self):
    self.assertEqual(
        eval_metrics.present_targets(self.METRICS),
        [
            ("sft", "SFT"),
            ("perl:organic", "PE-RL (Organic)"),
            ("perl:synthetic_struct", "PE-RL (Synthetic Struct)"),
        ],
    )

  def test_headline_reports_the_best_branch_not_the_first(self):
    # For a minimized metric the best branch is the lowest. Taking the first
    # would make the headline depend on flavor ordering.
    self.assertAlmostEqual(
        eval_metrics.headline_metric(self.METRICS), 0.04
    )

  def test_headline_maximizes_for_an_upward_metric(self):
    self.assertAlmostEqual(
        eval_metrics.headline_metric(
            {
                "perl:organic/faithfulness_rate": 0.90,
                "perl:synthetic_struct/faithfulness_rate": 0.95,
            },
            name="faithfulness_rate",
        ),
        0.95,
    )

  def test_one_delta_per_policy(self):
    rows = eval_metrics.comparison_rows(self.METRICS)
    name, values, deltas = rows[0]
    self.assertEqual(name, "hallucination_rate")
    self.assertEqual(values, [0.10, 0.06, 0.04])
    self.assertEqual(len(deltas), 2)
    self.assertAlmostEqual(deltas[0], 0.04)
    self.assertAlmostEqual(deltas[1], 0.06)

  def test_legacy_single_model_metrics_still_render(self):
    self.assertAlmostEqual(
        eval_metrics.headline_metric({"hallucination_rate": 0.2}), 0.2
    )


class BranchedReportTest(unittest.TestCase):
  """The markdown report shows every branch."""

  def _report(self, flavor_list):
    config = _config(flavor_list)
    state = CampaignState(campaign_id="c", task_name="ragtruth")
    for stage_id in flavors.branch_stage_ids(config, "rm"):
      state.stages[stage_id] = StageResult(
          status=StageStatus.COMPLETED,
          model_repo_id=f"u/{stage_id}",
          best_metric_val=0.9,
          best_run_id="r",
          best_params={"learning_rate": 1e-5},
      )
    with tempfile.TemporaryDirectory() as temp_dir:
      config.reporting.reports_dir = temp_dir
      path = CampaignReporter(config, state).generate_markdown_report()
      with open(path, "r", encoding="utf-8") as handle:
        return handle.read()

  def test_branched_report_has_one_section_per_reward_model(self):
    content = self._report([flavors.ORGANIC, flavors.SYNTHETIC_STRUCT])
    self.assertIn("Reward Model (RM) - Organic dataset", content)
    self.assertIn("Reward Model (RM) - Synthetic Struct dataset", content)

  def test_unbranched_report_keeps_its_original_heading(self):
    content = self._report([flavors.ORGANIC])
    self.assertIn("### 2.2. Reward Model (RM)", content)
    self.assertNotIn("2.2.1.", content)


if __name__ == "__main__":
  unittest.main()
