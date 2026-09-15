"""Unit and integration tests for the Auto-PERL orchestrator."""

import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from src.orchestrator.cli import events as events_mod
from src.orchestrator.config import CampaignConfig
from src.orchestrator.engine import CampaignEngine
from src.orchestrator.model_manager import ModelManager
from src.orchestrator.reporter import CampaignReporter
from src.orchestrator.stages.base import CampaignContext
from src.orchestrator.state import CampaignState
from src.orchestrator.state import StageResult
from src.orchestrator.state import StageStatus
from src.orchestrator.sweep_controller import SweepController


class TestOrchestratorConfig(unittest.TestCase):
  """Tests for configuration serialization, defaults, and validation."""

  def test_defaults(self):
    config = CampaignConfig.create_default(task_name="npov")
    self.assertEqual(config.task_name, "npov")
    self.assertEqual(config.sft.max_runs, 30)
    self.assertEqual(config.rm.max_runs, 30)
    self.assertEqual(config.perl.max_runs, 10)
    self.assertEqual(config.stages, ["sft", "rm", "perl", "eval"])

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
    self.assertEqual(started, {"sft", "rm", "perl", "eval"})
    self.assertEqual(completed, {"sft", "rm", "perl", "eval"})

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
    self.assertEqual(state.stages_order, ["sft", "rm", "perl", "eval"])
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
    self.assertEqual(skipped, {"sft", "rm", "perl", "eval"})
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

  def test_abort_skips_reporting(self):
    controls = events_mod.ControlSignals()
    controls.request_abort()
    result = CampaignEngine(config=self.config, controls=controls).run()
    self.assertEqual(result["status"], "STOPPED")
    self.assertNotIn("artifacts", result)

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
    self.feed(AGENT_TRANSCRIPT)
    self.assertEqual(self.reload().trials_done, 2)
    resumed = self.engine._stage_line_callback("sft")  # pylint: disable=protected-access
    for line in AGENT_TRANSCRIPT[:7]:
      resumed(line)
    self.assertEqual(self.reload().trials_done, 3)

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


class TestSweepNaming(unittest.TestCase):
  """Tests for descriptive sweep name generation across SFT, RM, and PE-RL."""

  def test_sft_sweep_name_generation(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      config = CampaignConfig.create_default(task_name="bosch", dry_run=True)
      config.state_file = os.path.join(temp_dir, "bosch_state.json")
      state = CampaignState(campaign_id="test_camp", task_name="bosch")
      context = CampaignContext(
          config=config,
          state=state,
          sweep_controller=SweepController(dry_run=True),
          model_manager=ModelManager(dry_run=True),
      )
      from src.orchestrator.stages.sft_stage import SftStage  # pylint: disable=g-import-not-at-top

      stage = SftStage(context)
      name = stage.generate_sweep_name({"command": []})
      self.assertEqual(name, "BOSCH gemma-4-E2B-it SFT Sweep #1")

  def test_rm_sweep_name_organic_and_synthetic(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      config = CampaignConfig.create_default(task_name="bosch", dry_run=True)
      config.state_file = os.path.join(temp_dir, "bosch_state.json")
      state = CampaignState(campaign_id="test_camp", task_name="bosch")
      context = CampaignContext(
          config=config,
          state=state,
          sweep_controller=SweepController(dry_run=True),
          model_manager=ModelManager(dry_run=True),
      )
      from src.orchestrator.stages.rm_stage import RmStage  # pylint: disable=g-import-not-at-top

      stage = RmStage(context)
      # Default organic
      name_org = stage.generate_sweep_name({"command": []})
      self.assertEqual(name_org, "BOSCH gemma-3-1b-it RM Organic Sweep #1")

      # Synthetic dataset
      name_synth = stage.generate_sweep_name({
          "command": ["--dataset_repo_id=leobianco/bosch_rm_synthetic"]
      })
      self.assertEqual(name_synth, "BOSCH gemma-3-1b-it RM Synthetic Sweep #1")

  def test_perl_sweep_name_organic_and_synthetic(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      config = CampaignConfig.create_default(task_name="npov", dry_run=True)
      config.state_file = os.path.join(temp_dir, "npov_state.json")
      state = CampaignState(campaign_id="test_camp", task_name="npov")
      context = CampaignContext(
          config=config,
          state=state,
          sweep_controller=SweepController(dry_run=True),
          model_manager=ModelManager(dry_run=True),
      )
      from src.orchestrator.stages.perl_stage import PerlStage  # pylint: disable=g-import-not-at-top

      stage = PerlStage(context)
      name_org = stage.generate_sweep_name({"command": []})
      self.assertEqual(name_org, "NPOV gemma-4-E2B-it PERL Organic Sweep #1")

      name_synth = stage.generate_sweep_name({
          "command": ["--reward_model_path=leobianco/npov_RM_synthetic"]
      })
      self.assertEqual(name_synth, "NPOV gemma-4-E2B-it PERL Synthetic Sweep #1")

  def test_sweep_number_increments_with_existing_sweeps(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      config = CampaignConfig.create_default(task_name="bosch", dry_run=True)
      config.state_file = os.path.join(temp_dir, "bosch", "camp2_state.json")
      os.makedirs(os.path.join(temp_dir, "bosch"), exist_ok=True)

      # Create prior state file with an existing SFT sweep
      prior_state = CampaignState(campaign_id="camp1", task_name="bosch")
      prior_state.stages["sft"] = StageResult(
          status=StageStatus.COMPLETED,
          sweep_id="sweep_sft_prior_1",
          sweep_name="BOSCH gemma-4-E2B-it SFT Sweep #1",
      )
      prior_state.save(os.path.join(temp_dir, "bosch", "camp1_state.json"))

      current_state = CampaignState(campaign_id="camp2", task_name="bosch")
      context = CampaignContext(
          config=config,
          state=current_state,
          sweep_controller=SweepController(dry_run=True),
          model_manager=ModelManager(dry_run=True),
      )
      from src.orchestrator.stages.sft_stage import SftStage  # pylint: disable=g-import-not-at-top

      stage = SftStage(context)
      name = stage.generate_sweep_name({"command": []})
      self.assertEqual(name, "BOSCH gemma-4-E2B-it SFT Sweep #2")

      # Add another state file with Sweep #2 and verify it increments to #3
      prior_state_2 = CampaignState(campaign_id="camp2_done", task_name="bosch")
      prior_state_2.stages["sft"] = StageResult(
          status=StageStatus.COMPLETED,
          sweep_id="sweep_sft_prior_2",
          sweep_name="BOSCH gemma-4-E2B-it SFT Sweep #2",
      )
      prior_state_2.save(os.path.join(temp_dir, "bosch", "camp2_done_state.json"))

      name_3 = stage.generate_sweep_name({"command": []})
      self.assertEqual(name_3, "BOSCH gemma-4-E2B-it SFT Sweep #3")


if __name__ == "__main__":
  unittest.main()

