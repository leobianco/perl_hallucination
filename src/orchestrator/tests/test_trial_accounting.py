"""Tests for sweep trial accounting and campaign-derived sweep names.

Both regressions these cover were observed on the same RAGTruth campaign: a
PE-RL sweep that lost its 4th of 5 trials to a dead VM reported ``05/05`` and
a clean scorecard, and every campaign for a given task produced a sweep called
``... Sweep #1``.
"""

from __future__ import annotations

import datetime
import os
import tempfile
import unittest
from unittest import mock

from src.orchestrator import naming
from src.orchestrator.cli import renderables
from src.orchestrator.cli import theme as theme_mod
from src.orchestrator.config import CampaignConfig
from src.orchestrator.model_manager import ModelManager
from src.orchestrator.reporter import CampaignReporter
from src.orchestrator.stages.base import CampaignContext
from src.orchestrator.stages.perl_stage import PerlStage
from src.orchestrator.stages.rm_stage import RmStage
from src.orchestrator.stages.sft_stage import SftStage
from src.orchestrator.state import CampaignState, StageResult, StageStatus
from src.orchestrator.sweep_controller import AgentRun, SweepController


def _context(
    temp_dir,
    task_name="ragtruth",
    campaign_id="ragtruth_campaign_2609150825",
    dry_run=False,
):
  """Builds a stage context wired to a temp state file."""
  config = CampaignConfig.create_default(task_name=task_name, dry_run=dry_run)
  config.name = campaign_id
  config.base_model = "google/gemma-4-E2B-it"
  config.state_file = os.path.join(temp_dir, f"{campaign_id}_state.json")
  state = CampaignState(campaign_id=campaign_id, task_name=task_name)
  return CampaignContext(
      config=config,
      state=state,
      sweep_controller=SweepController(dry_run=dry_run),
      model_manager=ModelManager(dry_run=dry_run),
      state_path=config.state_file,
  )


class TokenTest(unittest.TestCase):
  """The hex token that replaced ``Sweep #N``."""

  def test_token_is_six_lowercase_hex_characters(self):
    token = naming.campaign_token("ragtruth_campaign_2609150825")
    self.assertRegex(token, r"^[0-9a-f]{6}$")

  def test_token_is_derived_from_the_campaign_timestamp(self):
    # 2026-09-15 08:25, encoded in the campaign id as %y%m%d%H%M.
    expected = naming.token_from_datetime(
        datetime.datetime(2026, 9, 15, 8, 25)
    )
    self.assertEqual(
        naming.campaign_token("ragtruth_campaign_2609150825"), expected
    )

  def test_campaigns_a_minute_apart_get_different_tokens(self):
    self.assertNotEqual(
        naming.campaign_token("ragtruth_campaign_2609150825"),
        naming.campaign_token("ragtruth_campaign_2609150826"),
    )

  def test_token_sorts_chronologically(self):
    earlier = naming.campaign_token("x_campaign_2601010000")
    later = naming.campaign_token("x_campaign_2612312359")
    self.assertLess(earlier, later)

  def test_falls_back_to_created_at_when_the_id_has_no_stamp(self):
    expected = naming.token_from_datetime(
        datetime.datetime(2026, 3, 4, 5, 6)
    )
    self.assertEqual(
        naming.campaign_token(
            "handwritten-campaign", created_at="2026-03-04T05:06:07"
        ),
        expected,
    )

  def test_resume_of_the_same_campaign_keeps_the_same_token(self):
    # The id is stable across resumes; created_at is not consulted when the
    # id carries a stamp, so a restarted campaign must not scatter its
    # sweeps under a second token.
    first = naming.campaign_token(
        "ragtruth_campaign_2609150825", created_at="2026-09-15T08:25:00"
    )
    second = naming.campaign_token(
        "ragtruth_campaign_2609150825", created_at="2026-09-18T22:10:00"
    )
    self.assertEqual(first, second)


class SweepNameTest(unittest.TestCase):
  """Sweep names carry the token, and all three stages share it."""

  def test_sft_name(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      stage = SftStage(_context(temp_dir, task_name="bosch",
                                campaign_id="bosch_campaign_2609150825",
                                dry_run=True))
      token = naming.campaign_token("bosch_campaign_2609150825")
      self.assertEqual(
          stage.generate_sweep_name({"command": []}),
          f"BOSCH gemma-4-E2B-it SFT Sweep {token}",
      )

  def test_rm_name_organic_and_synthetic(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      stage = RmStage(_context(temp_dir, task_name="bosch",
                               campaign_id="bosch_campaign_2609150825",
                               dry_run=True))
      token = naming.campaign_token("bosch_campaign_2609150825")
      self.assertEqual(
          stage.generate_sweep_name(
              {"command": ["--model_repo_id=google/gemma-3-1b-it"]}
          ),
          f"BOSCH gemma-3-1b-it RM Organic Sweep {token}",
      )
      self.assertEqual(
          stage.generate_sweep_name({
              "command": [
                  "--model_repo_id=google/gemma-3-1b-it",
                  "--dataset_repo_id=leobianco/bosch_rm_synthetic",
              ]
          }),
          f"BOSCH gemma-3-1b-it RM Synthetic Sweep {token}",
      )

  def test_perl_name_organic_and_synthetic(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      stage = PerlStage(_context(temp_dir, task_name="npov",
                                 campaign_id="npov_campaign_2609150825",
                                 dry_run=True))
      token = naming.campaign_token("npov_campaign_2609150825")
      self.assertEqual(
          stage.generate_sweep_name({"command": []}),
          f"NPOV gemma-4-E2B-it PERL Organic Sweep {token}",
      )
      self.assertEqual(
          stage.generate_sweep_name(
              {"command": ["--reward_model_path=leobianco/npov_RM_synthetic"]}
          ),
          f"NPOV gemma-4-E2B-it PERL Synthetic Sweep {token}",
      )

  def test_all_stages_of_one_campaign_share_the_token(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      context = _context(temp_dir, task_name="npov",
                         campaign_id="npov_campaign_2609150825", dry_run=True)
      names = [
          SftStage(context).generate_sweep_name({"command": []}),
          RmStage(context).generate_sweep_name({"command": []}),
          PerlStage(context).generate_sweep_name({"command": []}),
      ]
      tokens = {name.rsplit(" ", 1)[1] for name in names}
      self.assertEqual(len(tokens), 1, names)

  def test_two_campaigns_no_longer_collide(self):
    # The old counter restarted at #1 whenever the state directory was empty
    # or W&B was unreachable, so two machines both produced "Sweep #1".
    with tempfile.TemporaryDirectory() as temp_dir:
      def name_for(campaign_id):
        context = _context(temp_dir, campaign_id=campaign_id, dry_run=True)
        return SftStage(context).generate_sweep_name({"command": []})

      self.assertNotEqual(
          name_for("ragtruth_campaign_2609150825"),
          name_for("ragtruth_campaign_2609181430"),
      )

  def test_explicit_name_in_the_yaml_is_not_overwritten(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      context = _context(temp_dir, dry_run=True)
      stage = SftStage(context)
      sweep_dict = {"name": "hand picked", "command": []}
      stage.resolve_sweep_id(sweep_dict)
      self.assertEqual(sweep_dict["name"], "hand picked")


class AccountForTrialsTest(unittest.TestCase):
  """What the stage records once the agent is back."""

  def _stage(self, temp_dir, finished):
    context = _context(temp_dir)
    context.sweep_controller.count_finished_runs = mock.Mock(
        return_value=finished
    )
    return PerlStage(context)

  def test_full_budget_is_clean(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      stage = self._stage(temp_dir, finished=5)
      execution = stage.account_for_trials(
          sweep_id="e/p/s", budget=5, agent=AgentRun(exit_code=0)
      )
      self.assertEqual(execution.trials_done, 5)
      self.assertEqual(execution.outcome, "complete")
      self.assertFalse(execution.is_partial)
      self.assertEqual(execution.warnings, [])

  def test_lost_trials_are_recorded_and_announced(self):
    # The RAGTruth regression: 3 of 5 finished, the 4th died with the VM.
    with tempfile.TemporaryDirectory() as temp_dir:
      stage = self._stage(temp_dir, finished=3)
      lines = []
      execution = stage.account_for_trials(
          sweep_id="e/p/s",
          budget=5,
          agent=AgentRun(exit_code=137),
          live_line_callback=lines.append,
      )
      self.assertEqual(execution.trials_done, 3)
      self.assertEqual(execution.trials_total, 5)
      self.assertTrue(execution.is_partial)
      self.assertEqual(len(execution.warnings), 1)
      self.assertIn("3 of 5", execution.warnings[0])
      self.assertIn("exited with code 137", execution.warnings[0])
      self.assertTrue(any("[WARNING]" in line for line in lines))

  def test_timeout_is_named_as_the_reason(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      stage = self._stage(temp_dir, finished=2)
      execution = stage.account_for_trials(
          sweep_id="e/p/s",
          budget=5,
          agent=AgentRun(exit_code=0, timed_out=True, timeout_minutes=240),
      )
      self.assertIn("240 minute", execution.warnings[0])

  def test_user_stop_is_named_as_the_reason(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      stage = self._stage(temp_dir, finished=2)
      execution = stage.account_for_trials(
          sweep_id="e/p/s",
          budget=5,
          agent=AgentRun(exit_code=0, interrupted=True),
      )
      self.assertIn("stopped on request", execution.warnings[0])

  def test_unreachable_wandb_says_so_instead_of_guessing(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      stage = self._stage(temp_dir, finished=None)
      stage.state.stages["perl"] = StageResult(
          status=StageStatus.RUNNING, trials_done=4
      )
      execution = stage.account_for_trials(sweep_id="e/p/s", budget=5)
      self.assertEqual(execution.outcome, "unknown")
      self.assertEqual(execution.trials_done, 4)
      self.assertIn("lower bound", execution.warnings[0])

  def test_zero_finished_trials_is_called_out(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      stage = self._stage(temp_dir, finished=0)
      execution = stage.account_for_trials(sweep_id="e/p/s", budget=5)
      self.assertTrue(
          any("reached the 'finished' state" in w for w in execution.warnings)
      )

  def test_accounting_is_persisted_before_materialization(self):
    # Materialization retrains the winner and can take longer than the sweep.
    # A crash in there must not lose the record of what the sweep produced.
    with tempfile.TemporaryDirectory() as temp_dir:
      stage = self._stage(temp_dir, finished=3)
      stage.account_for_trials(sweep_id="e/p/s", budget=5)
      reloaded = CampaignState.load(stage.context.state_path)
      self.assertEqual(reloaded.stages["perl"].trials_done, 3)
      self.assertEqual(reloaded.stages["perl"].trials_total, 5)
      self.assertEqual(reloaded.stages["perl"].sweep_outcome, "partial")
      self.assertTrue(reloaded.stages["perl"].warnings)

  def test_dry_run_does_not_call_wandb_or_warn(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      context = _context(temp_dir, dry_run=True)
      context.sweep_controller.count_finished_runs = mock.Mock(
          side_effect=AssertionError("W&B must not be queried in a dry run")
      )
      execution = PerlStage(context).account_for_trials(
          sweep_id="e/p/s", budget=5
      )
      self.assertEqual(execution.trials_done, 5)
      self.assertEqual(execution.warnings, [])

  def test_no_agent_launched_still_reconciles(self):
    # A resumed stage whose budget was already spent skips the agent, but the
    # count still has to be established so the result is not left at zero.
    with tempfile.TemporaryDirectory() as temp_dir:
      stage = self._stage(temp_dir, finished=5)
      execution = stage.account_for_trials(
          sweep_id="e/p/s", budget=5, agent=None
      )
      self.assertEqual(execution.trials_done, 5)
      self.assertEqual(execution.outcome, "complete")

  def test_run_sweep_trials_passes_the_remaining_budget_to_the_agent(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      context = _context(temp_dir)
      context.sweep_controller.count_finished_runs = mock.Mock(return_value=3)
      context.sweep_controller.run_sweep_agent_detailed = mock.Mock(
          return_value=AgentRun(exit_code=0)
      )
      stage = PerlStage(context)
      stage.config.perl.max_runs = 5
      execution = stage.run_sweep_trials(
          sweep_id="e/p/s", stage_config=stage.config.perl
      )
      kwargs = context.sweep_controller.run_sweep_agent_detailed.call_args[1]
      self.assertEqual(kwargs["max_runs"], 2)
      self.assertEqual(execution.trials_done, 3)


class DryRunAgentTest(unittest.TestCase):
  """The simulated agent must not look like a sweep that lost trials."""

  def test_simulation_covers_the_whole_budget(self):
    controller = SweepController(dry_run=True)
    lines = []
    outcome = controller.run_sweep_agent_detailed(
        sweep_id="e/p/s", max_runs=5, live_line_callback=lines.append
    )
    self.assertEqual(outcome.exit_code, 0)
    self.assertEqual(len([l for l in lines if "Trial" in l]), 5)


class AgentRunTest(unittest.TestCase):
  """The exit code alone cannot distinguish these cases."""

  def test_clean(self):
    self.assertTrue(AgentRun(exit_code=0).clean)
    self.assertIsNone(AgentRun(exit_code=0).describe())

  def test_timeout_is_not_clean_despite_exit_code_zero(self):
    run = AgentRun(exit_code=0, timed_out=True, timeout_minutes=90)
    self.assertFalse(run.clean)
    self.assertTrue(run.cut_short)
    self.assertIn("90 minute", run.describe())

  def test_backwards_compatible_wrapper_returns_the_code(self):
    controller = SweepController(dry_run=True)
    self.assertEqual(controller.run_sweep_agent("e/p/s", max_runs=1), 0)


class StageViewTest(unittest.TestCase):
  """The dashboard must not invent a full progress bar."""

  def _views(self, **stage_fields):
    config = CampaignConfig.create_default(task_name="ragtruth")
    config.stages = ["perl"]
    config.perl.max_runs = 5
    state = CampaignState(campaign_id="c", task_name="ragtruth")
    state.stages["perl"] = StageResult(**stage_fields)
    return {v.key: v for v in renderables.build_stage_views(config, state)}

  def test_completed_partial_stage_shows_the_real_count(self):
    view = self._views(
        status=StageStatus.COMPLETED,
        trials_done=3,
        trials_total=5,
        sweep_outcome="partial",
    )["perl"]
    self.assertEqual(view.trials_done, 3)
    self.assertTrue(view.is_partial)
    self.assertAlmostEqual(view.fraction, 0.6)
    done, total, counter = renderables._bar_values(view)  # pylint: disable=protected-access
    self.assertEqual((done, total), (3, 5))
    self.assertIn("3", counter)

  def test_completed_full_stage_is_not_partial(self):
    view = self._views(
        status=StageStatus.COMPLETED,
        trials_done=5,
        trials_total=5,
        sweep_outcome="complete",
    )["perl"]
    self.assertFalse(view.is_partial)
    self.assertEqual(view.fraction, 1.0)

  def test_legacy_state_without_accounting_still_shows_full(self):
    # State files written before trial accounting existed have no counters;
    # drawing them as 00/05 would be a worse lie than the one being fixed.
    view = self._views(status=StageStatus.COMPLETED)["perl"]
    self.assertFalse(view.is_partial)
    self.assertEqual(view.fraction, 1.0)

  def test_partial_stage_is_labelled_in_the_dag(self):
    view = self._views(
        status=StageStatus.COMPLETED,
        trials_done=3,
        trials_total=5,
        sweep_outcome="partial",
    )["perl"]
    theme = theme_mod.detect_theme(force_color=False, force_ascii=True)
    line = renderables.dag_lines([view], theme, width=120)[0]
    self.assertIn("PARTIAL", line)

  def test_warnings_reach_the_end_of_campaign_scorecard(self):
    config = CampaignConfig.create_default(task_name="ragtruth")
    config.stages = ["perl"]
    config.perl.max_runs = 5
    state = CampaignState(campaign_id="c", task_name="ragtruth")
    state.stages["perl"] = StageResult(
        status=StageStatus.COMPLETED,
        trials_done=3,
        trials_total=5,
        sweep_outcome="partial",
        warnings=["PERL sweep is INCOMPLETE: 3 of 5 trials finished."],
    )
    theme = theme_mod.detect_theme(force_color=False, force_ascii=True)
    text = "\n".join(renderables.summary_lines(config, state, theme))
    self.assertIn("INCOMPLETE", text)


class ReportTest(unittest.TestCase):
  """A partial campaign must not produce a wall of green ticks."""

  def _report(self, **stage_fields):
    with tempfile.TemporaryDirectory() as temp_dir:
      config = CampaignConfig.create_default(task_name="ragtruth")
      config.stages = ["perl"]
      config.perl.max_runs = 5
      config.reporting.reports_dir = temp_dir
      state = CampaignState(campaign_id="c", task_name="ragtruth")
      state.stages["perl"] = StageResult(**stage_fields)
      path = CampaignReporter(config, state).generate_markdown_report()
      with open(path, "r", encoding="utf-8") as f:
        return f.read()

  def test_partial_stage_is_flagged(self):
    text = self._report(
        status=StageStatus.COMPLETED,
        trials_done=3,
        trials_total=5,
        sweep_outcome="partial",
        best_metric_val=0.5,
        warnings=["PERL sweep is INCOMPLETE: 3 of 5 trials finished."],
    )
    self.assertIn("COMPLETED (PARTIAL)", text)
    self.assertIn("3/5", text)
    self.assertIn("Degradations", text)
    self.assertIn("INCOMPLETE", text)

  def test_clean_stage_is_not_flagged(self):
    text = self._report(
        status=StageStatus.COMPLETED,
        trials_done=5,
        trials_total=5,
        sweep_outcome="complete",
        best_metric_val=0.5,
    )
    self.assertIn("✅ COMPLETED", text)
    self.assertNotIn("PARTIAL", text)
    self.assertNotIn("Degradations", text)
    self.assertIn("5/5", text)


class StatePersistenceTest(unittest.TestCase):
  """The new fields survive a round trip and do not leak across attempts."""

  def test_round_trip(self):
    result = StageResult(
        status=StageStatus.COMPLETED,
        trials_done=3,
        trials_total=5,
        sweep_outcome="partial",
        warnings=["something went wrong"],
    )
    restored = StageResult.from_dict(result.to_dict())
    self.assertEqual(restored.sweep_outcome, "partial")
    self.assertEqual(restored.warnings, ["something went wrong"])

  def test_restarting_a_stage_clears_its_previous_warnings(self):
    state = CampaignState(campaign_id="c", task_name="ragtruth")
    state.stages["perl"] = StageResult(
        status=StageStatus.FAILED,
        sweep_outcome="partial",
        warnings=["stale"],
    )
    state.mark_stage_running("perl")
    self.assertEqual(state.stages["perl"].warnings, [])
    self.assertIsNone(state.stages["perl"].sweep_outcome)

  def test_reconciled_count_is_not_overwritten_by_stale_progress(self):
    # record_stage_result carries the previous counters over only when the
    # result has none. The W&B-reconciled 3 must beat the parser's 4.
    state = CampaignState(campaign_id="c", task_name="ragtruth")
    state.stages["perl"] = StageResult(
        status=StageStatus.RUNNING, trials_done=4, trials_total=5
    )
    state.record_stage_result(
        "perl",
        StageResult(
            status=StageStatus.COMPLETED,
            trials_done=3,
            trials_total=5,
            sweep_outcome="partial",
        ),
    )
    self.assertEqual(state.stages["perl"].trials_done, 3)

  def test_reconciled_count_survives_materialization_output(self):
    # Materialization streams through the same line sink as the sweep, so
    # the engine's progress tracker keeps calling update_stage_progress for
    # minutes after the count was established. Its stdout estimate counts
    # crashed trials as done and must not win.
    state = CampaignState(campaign_id="c", task_name="ragtruth")
    state.stages["perl"] = StageResult(
        status=StageStatus.RUNNING,
        trials_done=3,
        trials_total=5,
        sweep_outcome="partial",
    )
    changed = state.update_stage_progress(
        "perl", trials_done=4, trials_total=5
    )
    self.assertFalse(changed)
    self.assertEqual(state.stages["perl"].trials_done, 3)

  def test_metric_still_updates_after_reconciliation(self):
    state = CampaignState(campaign_id="c", task_name="ragtruth")
    state.stages["perl"] = StageResult(
        status=StageStatus.RUNNING, trials_done=3, sweep_outcome="partial"
    )
    self.assertTrue(state.update_stage_progress("perl", best_metric_val=0.9))
    self.assertEqual(state.stages["perl"].best_metric_val, 0.9)

  def test_a_fresh_attempt_unfreezes_the_counters(self):
    state = CampaignState(campaign_id="c", task_name="ragtruth")
    state.stages["perl"] = StageResult(
        status=StageStatus.COMPLETED, trials_done=3, sweep_outcome="partial"
    )
    state.mark_stage_running("perl")
    self.assertTrue(state.update_stage_progress("perl", trials_done=1))
    self.assertEqual(state.stages["perl"].trials_done, 1)

  def test_the_sweep_name_survives_the_end_of_the_stage(self):
    # record_sweep_id stores the name at registration; the stage then
    # returns a fresh StageResult carrying only the id. Dropping the name
    # here is why every report showed a bare sweep id.
    state = CampaignState(campaign_id="c", task_name="ragtruth")
    state.stages["perl"] = StageResult(
        status=StageStatus.RUNNING,
        sweep_id="leobianco/new_perl/ab12cd34",
        sweep_name="RAGTRUTH gemma-4-E4B-it PERL Organic Sweep 35d199",
    )
    state.record_stage_result(
        "perl",
        StageResult(
            status=StageStatus.COMPLETED,
            sweep_id="leobianco/new_perl/ab12cd34",
        ),
    )
    self.assertEqual(
        state.stages["perl"].sweep_name,
        "RAGTRUTH gemma-4-E4B-it PERL Organic Sweep 35d199",
    )


if __name__ == "__main__":
  unittest.main()
