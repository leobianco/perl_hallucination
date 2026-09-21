"""Unit and integration tests for the Auto-PERL orchestrator."""

import os
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from src.orchestrator import shutdown as shutdown_mod
from src.orchestrator.cli import events as events_mod
from src.orchestrator.config import CampaignConfig, VALID_STAGES
from src.orchestrator.engine import CampaignEngine
from src.orchestrator.model_manager import ModelManager
from src.orchestrator.reporter import CampaignReporter
from src.orchestrator.stages.base import CampaignContext
from src.orchestrator.stages.perl_stage import PerlStage
from src.orchestrator.stages.rm_stage import RmStage
from src.orchestrator.stages.sft_stage import SftStage
from src.orchestrator.state import CampaignState
from src.orchestrator.state import StageResult
from src.orchestrator.state import StageStatus
from src.orchestrator.sweep_controller import RunScore
from src.orchestrator.sweep_controller import SweepController


class TestOrchestratorConfig(unittest.TestCase):
  """Tests for configuration serialization, defaults, and validation."""

  def test_defaults(self):
    config = CampaignConfig.create_default(task_name="npov")
    self.assertEqual(config.task_name, "npov")
    self.assertEqual(config.sft.max_runs, 30)
    self.assertEqual(config.rm.max_runs, 30)
    self.assertEqual(config.perl.max_runs, 10)
    self.assertEqual(config.stages, list(VALID_STAGES))

  def test_task_validation(self):
    valid_cfg = CampaignConfig.create_default(task_name="bosch")
    valid_cfg.validate()

    with self.assertRaises(ValueError):
      invalid_cfg = CampaignConfig.create_default(task_name="nonexistent_task")
      invalid_cfg.validate()

  def test_yaml_roundtrip(self):
    config = CampaignConfig.create_default(task_name="npov", sft_runs=15)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
      yaml_path = tf.name

    try:
      config.to_yaml(yaml_path)
      loaded_cfg = CampaignConfig.from_yaml(yaml_path)
      self.assertEqual(loaded_cfg.task_name, config.task_name)
      self.assertEqual(loaded_cfg.sft.max_runs, 15)
    finally:
      if os.path.exists(yaml_path):
        os.remove(yaml_path)


class TestOrchestratorState(unittest.TestCase):
  """Tests for atomic state persistence, resumption, and stage tracking."""

  def test_state_lifecycle(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      state_file = os.path.join(temp_dir, "test_state.json")
      state = CampaignState(campaign_id="test_camp", task_name="npov")

      # Initially no stages completed
      self.assertFalse(state.is_stage_completed("sft"))
      self.assertEqual(state.get_next_pending_stage(["sft", "rm"]), "sft")

      # Mark SFT running
      state.mark_stage_running("sft")
      self.assertEqual(state.stages["sft"].status, StageStatus.RUNNING)

      # Record SFT completed
      res = StageResult(
          status=StageStatus.COMPLETED,
          best_run_id="run_123",
          best_metric_val=0.25,
          model_repo_id="leobianco/npov_SFT_test",
      )
      state.record_stage_result("sft", res)
      state.save(state_file)

      # Reload state from disk
      loaded = CampaignState.load(state_file)
      self.assertTrue(loaded.is_stage_completed("sft"))
      self.assertEqual(loaded.get_model_repo_id("sft"), "leobianco/npov_SFT_test")
      self.assertEqual(loaded.get_next_pending_stage(["sft", "rm"]), "rm")

  def test_stage_result_inherits_live_best_metric(self):
    # A stage reports its *outcome*. When the post-sweep W&B query comes back
    # empty - or the stage failed before reaching it - the best score mirrored
    # while the sweep ran is the only one there is, and blanking it leaves a
    # visibly-executed stage showing "-" in the Best column.
    state = CampaignState(campaign_id="c", task_name="npov")
    state.mark_stage_running("sft")
    state.update_stage_progress(
        "sft", trials_done=7, trials_total=15, best_metric_val=0.31
    )

    state.record_stage_result(
        "sft", StageResult(status=StageStatus.FAILED, error_message="boom")
    )
    self.assertAlmostEqual(state.stages["sft"].best_metric_val, 0.31)
    self.assertEqual(state.stages["sft"].trials_done, 7)

  def test_stage_result_keeps_its_own_best_metric(self):
    state = CampaignState(campaign_id="c", task_name="npov")
    state.mark_stage_running("sft")
    state.update_stage_progress("sft", best_metric_val=0.31)

    state.record_stage_result(
        "sft",
        StageResult(
            status=StageStatus.COMPLETED,
            best_run_id="run_xyz",
            best_metric_val=0.28,
        ),
    )
    # The authoritative value from the W&B API wins over the parsed one.
    self.assertAlmostEqual(state.stages["sft"].best_metric_val, 0.28)
    self.assertEqual(state.stages["sft"].best_run_id, "run_xyz")


class TestResumeBudget(unittest.TestCase):
  """`wandb agent --count N` bounds an agent, not a sweep.

  Asking a resumed stage for the full `max_runs` runs a second full budget
  on top of the trials already paid for: stop at 6 of 10, resume, get 16.
  """

  def make_stage(self, finished):
    config = CampaignConfig.create_default(
        task_name="npov", sft_runs=10, dry_run=True
    )
    controller = mock.MagicMock()
    controller.count_finished_runs.return_value = finished
    context = CampaignContext(
        config=config,
        state=CampaignState(campaign_id="c", task_name="npov"),
        sweep_controller=controller,
        model_manager=mock.MagicMock(),
    )
    return SftStage(context)

  def test_only_the_missing_trials_are_requested(self):
    stage = self.make_stage(finished=6)
    self.assertEqual(stage.remaining_runs("sweep1", 10), 4)

  def test_a_spent_budget_requests_nothing(self):
    stage = self.make_stage(finished=10)
    self.assertEqual(stage.remaining_runs("sweep1", 10), 0)

  def test_an_overshot_budget_does_not_go_negative(self):
    stage = self.make_stage(finished=14)
    self.assertEqual(stage.remaining_runs("sweep1", 10), 0)

  def test_a_fresh_sweep_gets_the_whole_budget(self):
    stage = self.make_stage(finished=0)
    self.assertEqual(stage.remaining_runs("sweep1", 10), 10)

  def test_an_unknown_count_never_shrinks_the_search(self):
    # Overshooting costs GPU hours; undershooting silently gives the user a
    # smaller hyperparameter search than they asked for. Prefer the former.
    stage = self.make_stage(finished=None)
    self.assertEqual(stage.remaining_runs("sweep1", 10), 10)

  def test_the_user_is_told_what_is_being_skipped(self):
    stage = self.make_stage(finished=6)
    lines = []
    stage.remaining_runs("sweep1", 10, lines.append)
    self.assertTrue(any("6/10" in line and "4" in line for line in lines))


class TestSealedSweepIsReactivated(unittest.TestCase):
  """Aborting a campaign seals its sweep; resuming has to unseal it.

  A sealed sweep still resolves through the W&B API, so an existence check
  alone says "reuse it" and the agent then dies with "Sweep <id> is not
  running" - while the trials already paid for sit there, unusable.
  """

  def make_stage(self, running, resume_ok=True):
    config = CampaignConfig.create_default(
        task_name="npov", sft_runs=10, dry_run=True
    )
    controller = mock.MagicMock()
    controller.sweep_exists.return_value = True
    controller.sweep_is_running.return_value = running
    controller.resume_sweep.return_value = resume_ok
    state = CampaignState(campaign_id="c", task_name="npov")
    state.mark_stage_running("sft")
    state.stages["sft"].sweep_id = "sweep-abc"
    context = CampaignContext(
        config=config,
        state=state,
        sweep_controller=controller,
        model_manager=mock.MagicMock(),
    )
    return SftStage(context), controller

  def test_a_stopped_sweep_is_reactivated_before_reuse(self):
    stage, controller = self.make_stage(running=False)
    lines = []
    sweep_id = stage.resolve_sweep_id({"program": "x"}, lines.append)
    self.assertEqual(sweep_id, "sweep-abc")
    controller.resume_sweep.assert_called_once_with("sweep-abc")
    controller.create_sweep.assert_not_called()
    self.assertTrue(any("reactivated" in line for line in lines))

  def test_a_live_sweep_is_left_alone(self):
    stage, controller = self.make_stage(running=True)
    stage.resolve_sweep_id({"program": "x"})
    controller.resume_sweep.assert_not_called()

  def test_an_unknown_state_is_left_alone(self):
    # Never poke a sweep whose state could not be read; the agent launch is
    # a cheaper way to find out than a wrong guess.
    stage, controller = self.make_stage(running=None)
    stage.resolve_sweep_id({"program": "x"})
    controller.resume_sweep.assert_not_called()

  def test_a_failed_reactivation_tells_the_user_what_to_do(self):
    stage, _ = self.make_stage(running=False, resume_ok=False)
    lines = []
    stage.resolve_sweep_id({"program": "x"}, lines.append)
    self.assertTrue(
        any("wandb sweep --resume" in line for line in lines),
        f"no actionable remedy in {lines}",
    )


class TestCrashedTrialsDoNotConsumeBudget(unittest.TestCase):
  """A trial that crashed produced no model, so it bought the user nothing.

  `[x]` kills the trial in flight, so counting the wreckage would make every
  interruption quietly cost one trial of the search the user configured.
  """

  def _controller_seeing(self, states):
    fake_wandb = mock.MagicMock()
    fake_sweep = mock.MagicMock()
    fake_sweep.runs = [mock.MagicMock(state=s) for s in states]
    fake_wandb.Api.return_value.sweep.return_value = fake_sweep
    with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
      ctrl = SweepController(entity="e", project="p", dry_run=False)
      return ctrl.count_finished_runs("sweep-abc")

  def test_only_successful_trials_are_counted(self):
    count = self._controller_seeing(
        ["finished", "finished", "crashed", "failed", "killed", "running"]
    )
    self.assertEqual(count, 2)

  def test_state_casing_is_irrelevant(self):
    self.assertEqual(self._controller_seeing(["Finished", "FINISHED"]), 2)


class TestResumeBaseline(unittest.TestCase):
  """A resumed counter must never appear to go backwards.

  The new agent counts from zero. Without a baseline the dashboard shows
  `00/10` for a sweep that already has six trials, which reads as "it threw
  my work away" - the single most alarming thing the UI can do.
  """

  def make_engine(self, persisted, counted, sweep_id="sweep-abc"):
    config = CampaignConfig.create_default(
        task_name="npov", sft_runs=10, dry_run=True
    )
    state = CampaignState(campaign_id="c", task_name="npov")
    state.mark_stage_running("sft")
    state.stages["sft"].sweep_id = sweep_id
    state.stages["sft"].trials_done = persisted
    engine = CampaignEngine(config=config, state=state)
    engine.sweep_controller = mock.MagicMock()
    engine.sweep_controller.count_finished_runs.return_value = counted
    return engine

  def test_wandb_wins_over_a_stale_state_file(self):
    # The previous process died before flushing its progress.
    engine = self.make_engine(persisted=0, counted=6)
    self.assertEqual(engine._resume_baseline("sft"), 6)

  def test_the_state_file_wins_when_it_knows_more(self):
    # W&B may not have reconciled a just-finished run yet.
    engine = self.make_engine(persisted=6, counted=4)
    self.assertEqual(engine._resume_baseline("sft"), 6)

  def test_an_unreachable_api_falls_back_to_disk(self):
    engine = self.make_engine(persisted=6, counted=None)
    self.assertEqual(engine._resume_baseline("sft"), 6)

  def test_a_raising_api_never_breaks_the_campaign(self):
    engine = self.make_engine(persisted=6, counted=0)
    engine.sweep_controller.count_finished_runs.side_effect = RuntimeError("x")
    self.assertEqual(engine._resume_baseline("sft"), 6)

  def test_a_stage_without_a_sweep_does_not_call_wandb(self):
    engine = self.make_engine(persisted=0, counted=6, sweep_id=None)
    self.assertEqual(engine._resume_baseline("sft"), 0)
    engine.sweep_controller.count_finished_runs.assert_not_called()

  def test_every_sweep_stage_publishes_its_baseline(self):
    # The dashboard builds its own parser from this payload, so the offset
    # has to reach it for RM and PE-RL exactly as it does for SFT.
    for stage in ("sft", "rm", "perl"):
      with self.subTest(stage=stage):
        config = CampaignConfig.create_default(task_name="npov", dry_run=True)
        state = CampaignState(campaign_id="c", task_name="npov")
        state.mark_stage_running(stage)
        state.stages[stage].sweep_id = f"sweep-{stage}"
        state.stages[stage].trials_done = 6
        engine = CampaignEngine(config=config, state=state)
        engine.sweep_controller = mock.MagicMock()
        engine.sweep_controller.count_finished_runs.return_value = 6

        payload = engine._stage_payload(stage)

        self.assertEqual(payload["trials_baseline"], 6)

  def test_the_baseline_is_established_once_per_stage(self):
    # Every consumer must see the same number, and each lookup is a W&B
    # round trip.
    engine = self.make_engine(persisted=0, counted=6)
    for _ in range(4):
      engine._resume_baseline("sft")
    engine._stage_payload("sft")
    self.assertEqual(
        engine.sweep_controller.count_finished_runs.call_count, 1
    )


class TestBestRunIsTagged(unittest.TestCase):
  """The winner is tagged in W&B so it is findable in the web UI."""

  def _sweep_with(self, runs):
    fake_wandb = mock.MagicMock()
    fake_sweep = mock.MagicMock()
    fake_sweep.runs = runs
    fake_wandb.Api.return_value.sweep.return_value = fake_sweep
    return fake_wandb

  def _run(self, run_id, tags=()):
    run = mock.MagicMock()
    run.id = run_id
    run.tags = list(tags)
    return run

  def test_the_winner_gets_both_tags_and_a_note(self):
    winner = self._run("win1")
    fake_wandb = self._sweep_with([winner, self._run("other")])
    with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
      ctrl = SweepController(entity="e", project="p", dry_run=False)
      ok = ctrl.mark_best_run(
          sweep_id="s", run_id="win1", stage_name="rm",
          metric_name="eval/roc_auc", metric_value=0.8812,
      )
    self.assertTrue(ok)
    self.assertEqual(winner.tags, ["best", "best-rm"])
    self.assertIn("eval/roc_auc=0.88120", winner.notes)
    winner.update.assert_called_once()

  def test_the_previous_winner_is_demoted(self):
    # Otherwise a re-run leaves a pile of runs all claiming to be best.
    old = self._run("old1", tags=["best", "best-sft", "keep-me"])
    winner = self._run("new1")
    fake_wandb = self._sweep_with([old, winner])
    with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
      ctrl = SweepController(entity="e", project="p", dry_run=False)
      ctrl.mark_best_run(
          sweep_id="s", run_id="new1", stage_name="sft",
          metric_name="eval/loss", metric_value=0.3,
      )
    # Unrelated tags on the demoted run are left alone.
    self.assertEqual(old.tags, ["keep-me"])
    old.update.assert_called_once()
    self.assertEqual(winner.tags, ["best", "best-sft"])

  def test_an_untouched_run_is_not_rewritten(self):
    # One API write per changed run; a 30-trial sweep must not issue 30.
    bystander = self._run("other", tags=["something"])
    winner = self._run("win1")
    fake_wandb = self._sweep_with([bystander, winner])
    with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
      ctrl = SweepController(entity="e", project="p", dry_run=False)
      ctrl.mark_best_run(
          sweep_id="s", run_id="win1", stage_name="perl",
          metric_name="train/rewards/reward_fn/mean", metric_value=0.7,
      )
    bystander.update.assert_not_called()

  def test_tagging_is_idempotent(self):
    winner = self._run("win1", tags=["best", "best-sft"])
    fake_wandb = self._sweep_with([winner])
    with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
      ctrl = SweepController(entity="e", project="p", dry_run=False)
      ctrl.mark_best_run(
          sweep_id="s", run_id="win1", stage_name="sft",
          metric_name="eval/loss", metric_value=0.3,
      )
    self.assertEqual(winner.tags, ["best", "best-sft"])

  def test_a_failure_is_reported_but_never_raises(self):
    # This runs between the sweep and materialization. A cosmetic tag must
    # not be able to throw away hours of GPU time.
    fake_wandb = mock.MagicMock()
    fake_wandb.Api.side_effect = RuntimeError("network is down")
    lines = []
    with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
      ctrl = SweepController(entity="e", project="p", dry_run=False)
      ok = ctrl.mark_best_run(
          sweep_id="s", run_id="win1", stage_name="sft",
          metric_name="eval/loss", metric_value=0.3,
          live_line_callback=lines.append,
      )
    self.assertFalse(ok)
    self.assertTrue(any("could not tag" in l.lower() for l in lines), lines)

  def test_a_missing_run_id_is_a_no_op(self):
    ctrl = SweepController(entity="e", project="p", dry_run=True)
    self.assertFalse(ctrl.mark_best_run(
        sweep_id="s", run_id="", stage_name="sft", metric_name="eval/loss"
    ))

  def test_every_sweep_stage_tags_its_winner(self):
    for stage_name, stage_cls in (
        ("sft", SftStage), ("rm", RmStage), ("perl", PerlStage)
    ):
      with self.subTest(stage=stage_name):
        config = CampaignConfig.create_default(task_name="npov", dry_run=True)
        state = CampaignState(campaign_id="c", task_name="npov")
        # PE-RL needs its predecessors' artefacts before it will run.
        for done, repo in (("sft", "u/npov_SFT"), ("rm", "u/npov_RM")):
          state.record_stage_result(done, StageResult(
              status=StageStatus.COMPLETED, model_repo_id=repo
          ))
        controller = mock.MagicMock()
        controller.create_sweep.return_value = "sweep-x"
        controller.sweep_exists.return_value = False
        # `remaining_runs` lives on the stage; this is what it really calls.
        controller.count_finished_runs.return_value = 0
        controller.run_sweep_agent.return_value = 0
        controller.fetch_best_run_details.return_value = RunScore(
            run_id="win1",
            value=0.3,
            params={"lora_r": 8},
            final_value=0.3,
            selection=getattr(config, stage_name).selection_strategy,
        )
        model_manager = mock.MagicMock()
        model_manager.materialize_and_push.return_value = "u/out"
        context = CampaignContext(
            config=config,
            state=state,
            sweep_controller=controller,
            model_manager=model_manager,
        )

        stage_cls(context).execute()

        controller.mark_best_run.assert_called_once()
        kwargs = controller.mark_best_run.call_args.kwargs
        self.assertEqual(kwargs["stage_name"], stage_name)
        self.assertEqual(kwargs["run_id"], "win1")
        # Each stage must rank its trials the way its config says, not the
        # way whichever stage was edited last says.
        self.assertEqual(
            controller.fetch_best_run_details.call_args.kwargs["selection"],
            getattr(config, stage_name).selection_strategy,
        )
        # ... and hand the matching checkpoint policy to the retraining.
        self.assertEqual(
            model_manager.materialize_and_push.call_args.kwargs[
                "checkpoint_policy"
            ],
            getattr(config, stage_name).checkpoint_policy,
        )


class TestStaleErrorIsCleared(unittest.TestCase):
  """A resumed stage must not wear the previous attempt's failure."""

  def test_re_running_a_failed_stage_clears_its_error(self):
    state = CampaignState(campaign_id="c", task_name="npov")
    state.mark_stage_running("sft")
    state.update_stage_progress("sft", trials_done=6, trials_total=10)
    state.stages["sft"].sweep_id = "sweep-abc"
    state.mark_stage_failed("sft", "Materialization ... was interrupted.")

    state.mark_stage_running("sft")

    self.assertIsNone(state.stages["sft"].error_message)
    # Progress and sweep identity still hold; only the verdict is stale.
    self.assertEqual(state.stages["sft"].trials_done, 6)
    self.assertEqual(state.stages["sft"].sweep_id, "sweep-abc")


class TestSweepController(unittest.TestCase):
  """Tests for sweep registration, bounded execution, and best-run query in dry-run mode."""

  def test_dry_run_sweep_flow(self):
    ctrl = SweepController(entity="leobianco", project="test", dry_run=True)
    sweep_id = ctrl.create_sweep({"method": "bayes"})
    self.assertIn("mock_sweep", sweep_id)

    # Test agent execution with max_runs=2
    exit_code = ctrl.run_sweep_agent(sweep_id, max_runs=2)
    self.assertEqual(exit_code, 0)

    # Test best-run extraction
    run_id, val, params = ctrl.fetch_best_run(sweep_id, "eval/loss", "minimize")
    self.assertIsNotNone(run_id)
    self.assertLess(val, 1.0)
    self.assertIn("learning_rate", params)

  def test_resolve_entity_precedence(self):
    # 1. Explicit entity (when not 'auto')
    ctrl1 = SweepController(entity="explicit_team", dry_run=True)
    self.assertEqual(ctrl1.resolve_entity(), "explicit_team")

    # 2. Environment variable
    with mock.patch.dict(os.environ, {"WANDB_ENTITY": "env_user"}):
      ctrl2 = SweepController(entity=None, dry_run=True)
      self.assertEqual(ctrl2.resolve_entity(), "env_user")

    # 3. Auto-detected default entity from wandb.Api()
    fake_wandb = mock.MagicMock()
    fake_api = mock.MagicMock()
    fake_api.default_entity = "api_user"
    fake_wandb.Api.return_value = fake_api
    with mock.patch.dict(os.environ, {}, clear=True):
      os.environ.pop("WANDB_ENTITY", None)
      with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
        ctrl3 = SweepController(entity=None, dry_run=False)
        self.assertEqual(ctrl3.resolve_entity(), "api_user")

    # 4. Fallback to 'leobianco'
    with mock.patch.dict(os.environ, {}, clear=True):
      os.environ.pop("WANDB_ENTITY", None)
      ctrl4 = SweepController(entity=None, dry_run=True)
      self.assertEqual(ctrl4.resolve_entity(), "leobianco")


class TestModelManager(unittest.TestCase):
  """Tests for model materialization in dry-run mode."""

  def test_dry_run_materialize(self):
    mgr = ModelManager(user="leobianco", dry_run=True)
    repo_id = mgr.materialize_and_push(
        stage_name="sft",
        task_name="npov",
        base_model="google/gemma-4-E2B-it",
        best_params={"learning_rate": 0.003, "lora_r": 8},
        seed=130104,
    )
    self.assertIn("leobianco/npov_SFT_gemma-4-E2B-it", repo_id)


class TestReporter(unittest.TestCase):
  """Tests for scientific report generation."""

  def test_markdown_report_generation(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      config = CampaignConfig.create_default(task_name="npov", dry_run=True)
      config.reporting.reports_dir = temp_dir
      state = CampaignState(campaign_id="test_camp", task_name="npov")

      # Populate state with completed SFT and RM results
      state.stages["sft"] = StageResult(
          status=StageStatus.COMPLETED,
          sweep_id="sweep_sft_123",
          best_run_id="run_1",
          best_metric_val=0.312,
          model_repo_id="leobianco/npov_SFT_test",
          best_params={"learning_rate": 0.002, "lora_r": 8},
      )
      state.stages["eval"] = StageResult(
          status=StageStatus.COMPLETED,
          model_repo_id="leobianco/npov_PERL_test",
          metrics={"hallucination_rate": 0.052, "win_rate_vs_base": 0.76},
      )

      reporter = CampaignReporter(config, state)
      md_path = reporter.generate_markdown_report()

      self.assertTrue(os.path.exists(md_path))
      with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

      self.assertIn("Executive Performance Scorecard", content)
      self.assertIn("leobianco/npov_SFT_test", content)
      self.assertIn("hallucination_rate", content)


class TestFullEngineDryRun(unittest.TestCase):
  """End-to-end integration test of CampaignEngine in dry-run mode."""

  def test_complete_dry_run_campaign(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      state_file = os.path.join(temp_dir, "test_camp_state.json")
      reports_dir = os.path.join(temp_dir, "reports")

      config = CampaignConfig.create_default(
          task_name="npov",
          sft_runs=2,
          rm_runs=2,
          perl_runs=2,
          dry_run=True,
      )
      config.state_file = state_file
      config.reporting.reports_dir = reports_dir
      config.no_tui = True

      logs = []

      def log_cb(line):
        logs.append(line)

      engine = CampaignEngine(config=config, live_line_callback=log_cb)
      res = engine.run()

      self.assertEqual(res["status"], "COMPLETED")
      self.assertTrue(os.path.exists(state_file))

      # Verify all 4 stages completed
      final_state = CampaignState.load(state_file)
      self.assertTrue(final_state.is_stage_completed("sft"))
      self.assertTrue(final_state.is_stage_completed("rm"))
      self.assertTrue(final_state.is_stage_completed("perl"))
      self.assertTrue(final_state.is_stage_completed("eval"))

      # Verify model IDs were chained
      sft_model = final_state.get_model_repo_id("sft")
      rm_model = final_state.get_model_repo_id("rm")
      perl_model = final_state.get_model_repo_id("perl")
      self.assertIsNotNone(sft_model)
      self.assertIsNotNone(rm_model)
      self.assertIsNotNone(perl_model)

      # Verify Markdown report created
      md_report = res["artifacts"].get("markdown_report")
      self.assertTrue(os.path.exists(md_report))


class TestEngineEvents(unittest.TestCase):
  """The structured event stream published by the engine."""

  def setUp(self):
    super().setUp()
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.config = CampaignConfig.create_default(
        task_name="npov", sft_runs=1, rm_runs=1, perl_runs=1, dry_run=True
    )
    self.config.state_file = os.path.join(self.temp.name, "camp_state.json")
    self.config.reporting.reports_dir = os.path.join(self.temp.name, "reports")
    self.config.no_tui = True

  def run_engine(self, **kwargs):
    bus = events_mod.EventBus()
    seen = []
    bus.subscribe(seen.append)
    engine = CampaignEngine(config=self.config, event_bus=bus, **kwargs)
    result = engine.run()
    return result, seen, engine

  def types_of(self, seen):
    return [event.type for event in seen]

  def test_campaign_lifecycle_events_are_published(self):
    _, seen, _ = self.run_engine()
    types = self.types_of(seen)
    self.assertEqual(types[0], events_mod.EventType.CAMPAIGN_STARTED)
    self.assertEqual(types[-1], events_mod.EventType.CAMPAIGN_FINISHED)
    self.assertEqual(seen[-1].payload["status"], "COMPLETED")

  def test_every_stage_reports_start_and_completion(self):
    _, seen, _ = self.run_engine()
    started = {
        e.stage for e in seen
        if e.type == events_mod.EventType.STAGE_STARTED
    }
    completed = {
        e.stage for e in seen
        if e.type == events_mod.EventType.STAGE_COMPLETED
    }
    self.assertEqual(started, set(VALID_STAGES))
    self.assertEqual(completed, set(VALID_STAGES))

  def test_stage_started_carries_sweep_metadata(self):
    _, seen, _ = self.run_engine()
    sft = next(
        e for e in seen
        if e.type == events_mod.EventType.STAGE_STARTED and e.stage == "sft"
    )
    self.assertEqual(sft.payload["max_runs"], 1)
    self.assertTrue(sft.payload["metric"])
    self.assertIn(sft.payload["goal"], ("minimize", "maximize"))

  def test_stage_completed_carries_artifacts(self):
    _, seen, _ = self.run_engine()
    sft = next(
        e for e in seen
        if e.type == events_mod.EventType.STAGE_COMPLETED and e.stage == "sft"
    )
    self.assertIsNotNone(sft.payload["model_repo_id"])
    self.assertIn("duration_s", sft.payload)

  def test_stage_logs_are_tagged_with_their_stage(self):
    _, seen, _ = self.run_engine()
    logs = [e for e in seen if e.type == events_mod.EventType.LOG]
    self.assertTrue(logs)
    self.assertTrue(all(e.stage for e in logs))

  def test_legacy_line_callback_still_receives_text(self):
    lines = []
    _, _, _ = self.run_engine(live_line_callback=lines.append)
    self.assertTrue(lines)
    self.assertTrue(any("Starting Campaign" in line for line in lines))

  def test_config_is_persisted_for_resume(self):
    self.run_engine()
    state = CampaignState.load(self.config.state_file)
    self.assertIsNotNone(state.config_dict)
    self.assertEqual(state.stages_order, list(VALID_STAGES))
    restored = CampaignConfig.from_dict(state.config_dict)
    self.assertEqual(restored.sft.max_runs, 1)
    self.assertTrue(restored.dry_run)

  def test_stage_durations_are_persisted(self):
    self.run_engine()
    state = CampaignState.load(self.config.state_file)
    sft = state.stages["sft"]
    # Regression: start_time used to be dropped, so the UI showed "-".
    self.assertIsNotNone(sft.start_time)
    self.assertIsNotNone(sft.end_time)

  def test_resuming_skips_completed_stages(self):
    self.run_engine()
    bus = events_mod.EventBus()
    seen = []
    bus.subscribe(seen.append)
    state = CampaignState.load(self.config.state_file)
    CampaignEngine(config=self.config, state=state, event_bus=bus).run()
    skipped = {
        e.stage for e in seen
        if e.type == events_mod.EventType.STAGE_SKIPPED
    }
    self.assertEqual(skipped, set(VALID_STAGES))
    self.assertFalse(
        any(e.type == events_mod.EventType.STAGE_STARTED for e in seen)
    )


class TestEngineControls(unittest.TestCase):
  """Pause, stop and abort semantics."""

  def setUp(self):
    super().setUp()
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.config = CampaignConfig.create_default(
        task_name="npov", sft_runs=1, rm_runs=1, perl_runs=1, dry_run=True
    )
    self.config.state_file = os.path.join(self.temp.name, "camp_state.json")
    self.config.reporting.reports_dir = os.path.join(self.temp.name, "reports")
    self.config.no_tui = True

  def test_stop_before_the_first_stage_still_reports(self):
    controls = events_mod.ControlSignals()
    controls.request_stop()
    result = CampaignEngine(config=self.config, controls=controls).run()
    self.assertEqual(result["status"], "STOPPED")
    self.assertIn("artifacts", result)
    state = CampaignState.load(self.config.state_file)
    self.assertEqual(state.status, "STOPPED")

  def test_abort_skips_reporting_and_says_it_was_an_abort(self):
    controls = events_mod.ControlSignals()
    controls.request_abort()
    result = CampaignEngine(config=self.config, controls=controls).run()
    # Not STOPPED: `[s]` and `[x]` differ in what they kill, so they must
    # differ in what they record. Not FAILED either - see below.
    self.assertEqual(result["status"], "ABORTED")
    self.assertNotIn("artifacts", result)
    state = CampaignState.load(self.config.state_file)
    self.assertEqual(state.status, "ABORTED")

  def test_an_abort_that_kills_a_stage_is_not_a_crash(self):
    """The regression that powered off an attended VM.

    Aborting *inside* a stage kills the materialization, so the stage raises
    on its way down and the campaign used to record FAILED - a terminal
    status, which the shutdown policy happily acts on while the operator who
    pressed the key is still sitting there.
    """
    controls = events_mod.ControlSignals()
    engine = CampaignEngine(config=self.config, controls=controls)

    def explode(*_args, **_kwargs):
      # Abort first, exactly as the hotkey would, then fail the way a
      # terminated materialization subprocess does.
      controls.request_abort()
      raise RuntimeError("Materialization of the best perl model was interrupted")

    with mock.patch.object(CampaignEngine, "_instantiate_stage") as factory:
      factory.return_value.execute.side_effect = explode
      result = engine.run()

    self.assertEqual(result["status"], "ABORTED")
    state = CampaignState.load(self.config.state_file)
    self.assertEqual(state.status, "ABORTED")
    # The *stage* still failed, and still holds what a resume needs.
    self.assertEqual(
        state.stages[state.current_stage].status, StageStatus.FAILED
    )
    # And the policy leaves the machine alone.
    armed = types.SimpleNamespace(shutdown_when_done=True, dry_run=False)
    wanted, reason = shutdown_mod.should_shutdown(armed, state.status)
    self.assertFalse(wanted)
    self.assertIn("by hand", reason)

  def test_pause_does_not_end_the_campaign(self):
    controls = events_mod.ControlSignals()
    controls.pause()
    engine = CampaignEngine(config=self.config, controls=controls)
    outcome = {}

    def worker():
      outcome["result"] = engine.run()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    time.sleep(0.2)
    # Still alive and parked, not stopped.
    self.assertTrue(thread.is_alive())
    self.assertNotIn("result", outcome)
    controls.resume()
    thread.join(timeout=30)
    self.assertFalse(thread.is_alive())
    self.assertEqual(outcome["result"]["status"], "COMPLETED")

  def test_advance_flag_is_cleared_between_stages(self):
    controls = events_mod.ControlSignals()
    controls.request_advance()
    CampaignEngine(config=self.config, controls=controls).run()
    self.assertFalse(controls.advance_requested)

  def test_legacy_stop_callback_is_honored(self):
    result = CampaignEngine(
        config=self.config, stop_requested_callback=lambda: True
    ).run()
    self.assertEqual(result["status"], "STOPPED")


#: A recorded two-trial transcript in the shape a *piped* (non-TTY) wandb
#: agent really emits. Hand-written fixtures hid a parser bug once already,
#: so this is kept verbatim.
AGENT_TRANSCRIPT = [
    "wandb: Starting wandb agent",
    "2026-09-15 11:30:46,123 - wandb.wandb_agent - INFO - Running runs: []",
    "2026-09-15 11:30:46,457 - wandb.wandb_agent - INFO - Agent starting run"
    " with config:",
    "2026-09-15 11:30:46,457 - wandb.wandb_agent - INFO - \tlearning_rate:"
    " 0.0003",
    "wandb: Agent Starting Run: k8jd92la with config:",
    "wandb:   eval_loss: 0.42",
    "2026-09-15 11:45:10,000 - wandb.wandb_agent - INFO - Cleaning up finished"
    " run: k8jd92la",
    "2026-09-15 11:45:11,000 - wandb.wandb_agent - INFO - Agent starting run"
    " with config:",
    "wandb: Agent Starting Run: p2mq71zz with config:",
    "wandb:   eval_loss: 0.31",
    "2026-09-15 11:58:02,000 - wandb.wandb_agent - INFO - Cleaning up finished"
    " run: p2mq71zz",
]


class TestProgressIsMirroredToDisk(unittest.TestCase):
  """Sweep progress must be readable by a process that is not the engine.

  The dashboard parses agent output in memory, which is invisible to a
  ``status --watch`` running in a second tmux pane. These tests assert on the
  *reloaded* state file, never on engine internals, because the file is the
  entire contract between the two panes.
  """

  def setUp(self):
    super().setUp()
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.config = CampaignConfig.create_default(
        task_name="npov", sft_runs=15, dry_run=True
    )
    self.config.state_file = os.path.join(self.temp.name, "camp_state.json")
    self.config.reporting.reports_dir = os.path.join(self.temp.name, "reports")
    self.config.no_tui = True
    self.engine = CampaignEngine(config=self.config)
    self.engine.state.mark_stage_running("sft")
    self.engine.state.save(self.engine.state_path)

  @property
  def sink(self):
    """The stage's line sink.

    The engine builds exactly one callback per stage execution, and the
    parser behind it is stateful, so tests must reuse it rather than build a
    fresh one per line.
    """
    if getattr(self, "_sink", None) is None:
      self._sink = self.engine._stage_line_callback("sft")  # pylint: disable=protected-access
    return self._sink

  def feed(self, lines):
    for line in lines:
      self.sink(line)

  def reload(self):
    """Reads the state file back, as the watching pane would."""
    return CampaignState.load(self.engine.state_path).stages["sft"]

  def test_trial_counts_reach_the_state_file(self):
    self.feed(AGENT_TRANSCRIPT)
    result = self.reload()
    self.assertEqual(result.trials_done, 2)
    self.assertEqual(result.trials_total, 15)

  def test_counter_advances_one_trial_at_a_time(self):
    observed = []
    for line in AGENT_TRANSCRIPT:
      self.feed([line])
      observed.append(self.reload().trials_done)
    # Never goes backwards, and ends where the transcript ends.
    self.assertEqual(observed, sorted(observed))
    self.assertEqual(observed[-1], 2)
    self.assertIn(1, observed)

  def test_best_metric_is_mirrored_for_a_minimize_goal(self):
    self.feed(AGENT_TRANSCRIPT)
    self.assertAlmostEqual(self.reload().best_metric_val, 0.31)

  def test_a_finished_trial_is_written_through_immediately(self):
    # Trial boundaries bypass the throttle: they are rare and are exactly
    # what the watching pane is waiting for.
    self.feed(AGENT_TRANSCRIPT[:7])
    self.assertEqual(self.reload().trials_done, 1)

  def test_noise_does_not_rewrite_the_state_file(self):
    self.feed(AGENT_TRANSCRIPT)
    before = os.path.getmtime(self.engine.state_path)
    for _ in range(500):
      self.feed(["2026-09-15 12:00:00,000 - wandb - INFO - some chatter"])
    self.assertEqual(os.path.getmtime(self.engine.state_path), before)

  def test_a_resumed_stage_does_not_restart_the_counter(self):
    # A resumed stage spawns a fresh agent counting from zero. The pane must
    # not appear to lose the trials a previous attempt already paid for.
    # A resume is a new *process*, so this builds a second engine over the
    # same state file rather than a second sink on the same engine.
    self.feed(AGENT_TRANSCRIPT)
    self.assertEqual(self.reload().trials_done, 2)

    resumed_engine = CampaignEngine(config=self.config)
    resumed = resumed_engine._stage_line_callback("sft")  # pylint: disable=protected-access
    for line in AGENT_TRANSCRIPT[:7]:
      resumed(line)

    reloaded = CampaignState.load(resumed_engine.state_path).stages["sft"]
    self.assertEqual(reloaded.trials_done, 3)

  def test_tracking_never_breaks_the_log_stream(self):
    # The progress mirror is cosmetic; if it explodes, the campaign and its
    # log must carry on regardless.
    seen = []
    with mock.patch.object(
        self.engine, "_persist_progress", side_effect=RuntimeError("boom")
    ):
      sink = self.engine._stage_line_callback("sft")  # pylint: disable=protected-access
      self.engine.bus.subscribe(seen.append)
      sink("wandb: Agent Starting Run: abc123 with config:")
    self.assertTrue(seen)

  def test_progress_is_tracked_per_stage(self):
    self.engine.state.mark_stage_running("rm")
    rm_sink = self.engine._stage_line_callback("rm")  # pylint: disable=protected-access
    self.feed(AGENT_TRANSCRIPT)
    for line in AGENT_TRANSCRIPT[:7]:
      rm_sink(line)
    stages = CampaignState.load(self.engine.state_path).stages
    self.assertEqual(stages["sft"].trials_done, 2)
    self.assertEqual(stages["rm"].trials_done, 1)

  def notices(self):
    """NOTICE messages published on the bus so far."""
    return [
        event.message
        for event in self.engine.bus.history
        if event.type == events_mod.EventType.NOTICE
    ]

  def test_unscored_trials_are_announced_once(self):
    # A metric that never appears is the difference between "still warming
    # up" and "your metric name is wrong", and the UI cannot tell them apart.
    scoreless = [
        line for line in AGENT_TRANSCRIPT if "eval_loss" not in line
    ]
    self.engine.bus.subscribe(lambda _event: None)
    self.feed(scoreless)
    matching = [n for n in self.notices() if "no 'eval/loss'" in n]
    self.assertEqual(len(matching), 1)
    self.assertIn("SFT", matching[0])

  def test_no_warning_when_trials_are_scored(self):
    self.feed(AGENT_TRANSCRIPT)
    self.assertEqual([n for n in self.notices() if "no 'eval/loss'" in n], [])


# NOTE: sweep-name generation moved to tests/test_trial_accounting.py when the
# "Sweep #N" counter was replaced by a campaign-derived hex token. The class
# that lived here asserted "... SFT Sweep #1" and the incrementing of that
# counter across state files, neither of which exists any more.

if __name__ == "__main__":
  unittest.main()

