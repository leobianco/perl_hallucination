"""Campaign execution engine coordinating the linear scientific DAG."""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from typing import Any, Callable, Dict, Optional

from src.orchestrator.cli.events import ControlSignals, EventBus, EventType
from src.orchestrator.cli.events import SweepProgressParser
from src.orchestrator.config import CampaignConfig
from src.orchestrator import flavors
from src.orchestrator import logging_setup
from src.orchestrator.model_manager import ModelManager
from src.orchestrator.reporter import CampaignReporter
from src.orchestrator.stages.base import BaseStage, CampaignContext
from src.orchestrator.stages.eval_stage import EvalStage
from src.orchestrator.stages.perl_stage import PerlStage
from src.orchestrator.stages.rm_stage import RmStage
from src.orchestrator.stages.sft_stage import SftStage
from src.orchestrator.state import CampaignState, StageResult, StageStatus
from src.orchestrator.sweep_controller import SweepController

logger = logging.getLogger(__name__)

#: Free disk below this (GiB) triggers a warning before each stage.
LOW_DISK_WARNING_GB = 25.0

#: Minimum seconds between state-file writes triggered by an improving metric
#: alone. A finished trial always writes immediately; this only throttles the
#: noisy intra-trial updates.
PROGRESS_SAVE_INTERVAL_S = 5.0

#: How many trials may finish without the configured metric ever appearing
#: before the campaign says so. One is enough to be suspicious, but a first
#: trial that crashes early legitimately has no metric, so wait for a second.
UNSCORED_TRIALS_BEFORE_WARNING = 2


class CampaignEngine:
  """Coordinates DAG execution, state persistence, error handling, and reporting."""

  def __init__(
      self,
      config: CampaignConfig,
      state: Optional[CampaignState] = None,
      live_line_callback: Optional[Callable[[str], None]] = None,
      stop_requested_callback: Optional[Callable[[], bool]] = None,
      event_bus: Optional[EventBus] = None,
      controls: Optional[ControlSignals] = None,
  ):
    self.config = config
    self.config.validate()

    state_path = config.state_file or f"./checkpoints/{config.task_name}/{config.name}_state.json"
    self.state_path = state_path

    # Opened before anything else so even constructor failures land on disk.
    self.log_path = logging_setup.configure_campaign_logging(
        config.name, log_dir=config.robustness.log_dir
    )

    if state:
      self.state = state
    else:
      self.state = CampaignState.load_or_create(
          filepath=state_path,
          campaign_id=config.name,
          task_name=config.task_name,
      )

    wandb_ent = config.wandb_entity or os.environ.get("WANDB_ENTITY")
    self.sweep_controller = SweepController(
        entity=wandb_ent,
        project=config.project,
        dry_run=config.dry_run,
        robustness=config.robustness,
    )
    if self.sweep_controller.entity and not self.config.wandb_entity:
      self.config.wandb_entity = self.sweep_controller.entity
    self.model_manager = ModelManager(
        user=config.user,
        dry_run=config.dry_run,
        robustness=config.robustness,
    )
    self.reporter = CampaignReporter(self.config, self.state)

    # Structured event stream. A legacy ``live_line_callback`` is bridged onto
    # the bus so existing callers (and the plain runner) keep working.
    self.bus = event_bus or EventBus()
    self.controls = controls or ControlSignals()
    self.live_line_callback = live_line_callback
    if live_line_callback is not None:
      self.bus.subscribe(
          lambda event: (
              live_line_callback(event.message)
              if event.message and event.type
              in (EventType.LOG, EventType.NOTICE, EventType.STAGE_STARTED,
                  EventType.STAGE_COMPLETED, EventType.STAGE_FAILED,
                  EventType.STAGE_SKIPPED, EventType.CAMPAIGN_STARTED)
              else None
          )
      )
    self._external_stop = stop_requested_callback
    self.stop_requested_callback = self._should_interrupt_stage

    # Live sweep progress, mirrored to the state file. The dashboard keeps its
    # own parser for the pane it draws in; this one exists for every *other*
    # reader - `status --watch` in a second tmux pane, `status --json | jq`,
    # and plain/nohup runs that have no dashboard at all.
    self._progress_parsers: Dict[str, Any] = {}
    self._progress_lock = threading.Lock()
    self._progress_saved_at = 0.0
    #: Trials each stage had already completed when this process took over.
    #: Cached because every consumer of the counter has to agree on it, and
    #: establishing it costs a W&B round trip.
    self._resume_baselines: Dict[str, int] = {}

    # Context shared across all stages
    self.context = CampaignContext(
        config=self.config,
        state=self.state,
        sweep_controller=self.sweep_controller,
        model_manager=self.model_manager,
        state_path=self.state_path,
        abort_requested_callback=self._should_abort_stage,
    )

  def _stage_line_callback(self, stage_name: str) -> Callable[[str], None]:
    """Line sink for a stage: event bus, durable log, and progress mirror."""
    downstream = logging_setup.tee_line_callback(
        self.log_path, self.bus.as_line_callback(stage_name)
    )
    track = self._progress_tracker(stage_name)

    def callback(line: str) -> None:
      downstream(line)
      track(line)

    return callback

  def _resume_baseline(self, stage_name: str) -> int:
    """Returns the trials this stage had already completed before now.

    Two sources disagree and both can be wrong. The state file only knows
    what the previous process managed to flush before it died, so a campaign
    killed mid-trial - or one whose stage crashed before the first progress
    write - reports too few. W&B knows the truth but may be unreachable, and
    the stage may not have registered a sweep yet.

    The larger of the two wins: a counter that jumps back to ``00/10`` after
    a resume is the single most alarming thing the dashboard can do, and it
    is exactly what a user sees when the disk value is stale.

    Args:
      stage_name: Stage about to run.

    Returns:
      Number of trials to add to the new agent's own count. Never negative.
    """
    if stage_name in self._resume_baselines:
      return self._resume_baselines[stage_name]
    baseline = self._compute_resume_baseline(stage_name)
    self._resume_baselines[stage_name] = baseline
    return baseline

  def _compute_resume_baseline(self, stage_name: str) -> int:
    """Establishes the resume baseline; see :meth:`_resume_baseline`.

    Args:
      stage_name: Stage about to run.

    Returns:
      Trials already completed, from the state file or W&B.
    """
    previous = self.state.stages.get(stage_name)
    baseline = int(getattr(previous, "trials_done", 0) or 0)
    sweep_id = getattr(previous, "sweep_id", None)
    if not sweep_id:
      return baseline
    try:
      counted = self.sweep_controller.count_finished_runs(sweep_id)
    except Exception:  # pylint: disable=broad-exception-caught
      # A cosmetic counter must never be able to stop a campaign.
      logger.debug("Resume baseline lookup failed for %s", stage_name,
                   exc_info=True)
      return baseline
    if counted is None:
      return baseline
    return max(baseline, int(counted))

  def _progress_tracker(self, stage_name: str) -> Callable[[str], None]:
    """Builds the line sink that mirrors sweep progress into the state file.

    Args:
      stage_name: Stage whose output will be fed to the returned callable.

    Returns:
      A callable consuming one output line at a time. It never raises: a
      cosmetic progress counter must not be able to kill a running campaign.
    """
    payload = self._stage_payload(stage_name)
    goal = payload.get("goal") or "minimize"
    parser = SweepProgressParser(
        metric_name=payload.get("metric") or "",
        max_trials=int(payload.get("max_runs") or 0),
    )
    self._progress_parsers[stage_name] = parser

    # A resumed stage starts a *new* agent, whose own counter restarts at
    # zero even though the sweep already has finished trials. The state file
    # describes the campaign, not one agent process, so the counter is
    # offset by what a previous attempt already achieved - otherwise the
    # watching pane would appear to go backwards on every resume.
    baseline = self._resume_baseline(stage_name)
    #: A silent "-" in the Best column is indistinguishable from "the sweep
    #: has not scored anything yet". Say it once, out loud, naming the key
    #: that was searched for, so a metric renamed in the sweep YAML is a
    #: two-second diagnosis instead of an evening of squinting at logs.
    unscored_warned = [False]

    def track(line: str) -> None:
      try:
        parser.feed(line)
        best = parser.best_trial(goal)
        if (
            best is None
            and not unscored_warned[0]
            and parser.completed_count >= UNSCORED_TRIALS_BEFORE_WARNING
        ):
          unscored_warned[0] = True
          self.bus.publish(
              EventType.NOTICE,
              message=(
                  f"{stage_name.upper()}: {parser.completed_count} trials have"
                  f" finished but no '{parser.metric_name}' value appeared in"
                  " the agent output, so the leaderboard and the Best column"
                  " stay empty. The final ranking still uses the W&B API;"
                  " only the live view is affected. Check that the metric"
                  " name matches what the training script logs."
              ),
              stage=stage_name,
          )
        self._persist_progress(
            stage_name,
            trials_done=baseline + parser.completed_count,
            trials_total=parser.max_trials,
            best_metric_val=best.metric if best is not None else None,
        )
      except Exception:  # pylint: disable=broad-exception-caught
        logger.debug("Progress tracking failed for %s", stage_name, exc_info=True)

    return track

  def _persist_progress(
      self,
      stage_name: str,
      trials_done: int,
      trials_total: int,
      best_metric_val: Optional[float],
  ) -> None:
    """Writes progress to the state file, throttled by time.

    A trial finishing is rare and important, so it is written immediately. A
    metric improving mid-trial is neither, and the agent can emit thousands of
    lines a minute - those writes are rate limited to keep the campaign from
    turning into a disk-thrashing loop.

    Args:
      stage_name: Stage being tracked.
      trials_done: Trials that reached a terminal state.
      trials_total: Trial budget.
      best_metric_val: Best metric seen so far, or None.
    """
    with self._progress_lock:
      result = self.state.stages.get(stage_name)
      trial_completed = (
          result is None or int(trials_done) != int(result.trials_done)
      )
      changed = self.state.update_stage_progress(
          stage_name,
          trials_done=trials_done,
          trials_total=trials_total,
          best_metric_val=best_metric_val,
      )
      if not changed:
        return
      now = time.time()
      if not trial_completed and (
          now - self._progress_saved_at < PROGRESS_SAVE_INTERVAL_S
      ):
        return
      self._progress_saved_at = now
      try:
        self.state.save(self.state_path)
      except OSError as exc:
        logger.debug("Could not persist progress for %s: %s", stage_name, exc)

  def _warn_on_low_disk(self, stage_name: str) -> None:
    """Publishes a notice when free disk space is running out.

    Every trial writes a LoRA checkpoint; a 30-trial sweep fills a disk
    quietly and then a push fails at the worst possible moment. This does not
    block the stage - only the user can decide what to delete.

    Args:
      stage_name: Stage about to start, used in the message.
    """
    try:
      free_gb = shutil.disk_usage(".").free / (1024**3)
    except OSError:
      return
    if free_gb >= LOW_DISK_WARNING_GB:
      return
    self.bus.publish(
        EventType.NOTICE,
        stage=stage_name,
        message=(
            f"Only {free_gb:.0f} GB of disk left before stage "
            f"'{stage_name}'. Checkpoints may fail to write; consider "
            "clearing ./checkpoints."
        ),
    )
    logger.warning("Low disk space before stage %s: %.0f GB", stage_name, free_gb)

  def _should_interrupt_stage(self) -> bool:
    """Predicate handed to sweep agents so hotkeys can cut a sweep short."""
    if self._external_stop is not None and self._external_stop():
      return True
    return self.controls.should_interrupt_stage()

  def _should_abort_stage(self) -> bool:
    """Predicate for work that must survive ``[a]`` advance and ``[s]`` stop.

    Advancing or stopping seals the *sweep*; the winning configuration still
    has to be retrained and pushed, otherwise the stage dies with
    "Materialization ... was interrupted". Only an explicit abort (``[x]``,
    double Ctrl-C, or the legacy caller-supplied stop callback) is allowed to
    kill that final training subprocess.
    """
    if self._external_stop is not None and self._external_stop():
      return True
    return self.controls.abort_requested

  def _instantiate_stage(self, planned: flavors.PlannedStage) -> BaseStage:
    """Instantiates the concrete stage implementation for one planned branch.

    Args:
      planned: The branch to run. ``kind`` picks the implementation;
        ``flavor`` and ``stage_id`` tell it which branch it is.

    Returns:
      The stage instance.

    Raises:
      ValueError: On an unknown stage kind.
    """
    classes = {
        "sft": SftStage,
        "rm": RmStage,
        "perl": PerlStage,
        "eval": EvalStage,
    }
    stage_class = classes.get(planned.kind)
    if stage_class is None:
      raise ValueError(f"Unknown stage name: {planned.kind}")
    return stage_class(
        self.context, flavor=planned.flavor, stage_id=planned.stage_id
    )

  def _stage_payload(self, stage_id: str) -> Dict[str, Any]:
    """Metadata describing a stage, published with ``STAGE_STARTED``.

    ``trials_baseline`` travels with the event because the dashboard builds
    its *own* parser from this payload, and that parser counts only what the
    current agent process does. Without the offset the pane would show the
    recovered ``06/10`` until the first new trial landed and then drop to
    ``01/10`` - the resumed sweep looking like it had thrown the work away.

    Args:
      stage_name: Stage being described.

    Returns:
      Payload dictionary for the event bus.
    """
    kind, _ = flavors.split_stage_id(stage_id)
    stage_cfg = getattr(self.config, kind, None)
    return {
        "metric": getattr(stage_cfg, "metric", None),
        "goal": getattr(stage_cfg, "goal", None),
        "max_runs": getattr(stage_cfg, "max_runs", 0),
        "trials_baseline": self._resume_baseline(stage_id),
    }

  def run(self) -> Dict[str, Any]:
    """Executes the complete experimental campaign DAG.

    Returns:
        Dictionary of generated report artifacts and final status.
    """
    logger.info("=== Launching Auto-PERL Campaign: %s ===", self.config.name)
    logger.info("Task: %s | Base Model: %s", self.config.task_name, self.config.base_model)
    logger.info("Stages to run: %s", self.config.stages)

    self.bus.publish(
        EventType.CAMPAIGN_STARTED,
        message=(
            f"Starting Campaign: {self.config.name} (Task: {self.config.task_name})"
        ),
        campaign_id=self.config.name,
        task=self.config.task_name,
        stages=list(self.config.stages),
        dry_run=self.config.dry_run,
    )

    self.state.status = "IN_PROGRESS"
    # Persisting the config makes ``resume`` reproduce the original campaign
    # (budgets, stage selection, checkpoint overrides) instead of guessing.
    self.state.config_dict = self.config.to_dict()
    plan = flavors.build_plan(self.config)
    self.state.stages_order = [planned.stage_id for planned in plan]
    self.state.save(self.state_path)

    interrupted = False
    try:
      for planned in plan:
        stage_name = planned.stage_id
        # The plan's title already carries the flavor when the campaign
        # branches, so every operator-facing line says *which* branch it is.
        stage_label = planned.title or stage_name.upper()
        # Check if already finished in previous run (resumption)
        if self.state.is_stage_completed(stage_name):
          logger.info("Stage '%s' is already completed. Skipping.", stage_name)
          self.bus.publish(
              EventType.STAGE_SKIPPED,
              stage=stage_name,
              message=(
                  f"[RESUME] Skipping already completed stage: {stage_label}"
              ),
              model_repo_id=self.state.get_model_repo_id(stage_name),
              metric=getattr(self.state.stages.get(stage_name), "best_metric_val", None),
          )
          continue

        # Honor a pause request before committing to a new stage.
        if self.controls.is_paused:
          self.bus.publish(
              EventType.NOTICE,
              message=f"Paused before stage '{stage_label}'. Press [p] to resume.",
          )
          self.state.status = "PAUSED"
          self.state.save(self.state_path)
          self.controls.wait_while_paused()
          if not self.controls.stop_requested:
            self.state.status = "IN_PROGRESS"
            self.state.save(self.state_path)

        # Check early stop request (hotkey or legacy callback).
        if self.controls.stop_requested or (
            self._external_stop is not None and self._external_stop()
        ):
          logger.info("Campaign stopped by user before starting %s", stage_name)
          interrupted = True
          self.state.status = "STOPPED"
          self.state.save(self.state_path)
          self.bus.publish(
              EventType.NOTICE,
              message=f"Campaign stopped by user before stage '{stage_label}'.",
          )
          break

        self._warn_on_low_disk(stage_name)
        # The baseline belongs to a stage *attempt*. Dropping it here means a
        # stage re-entered within one process re-establishes it instead of
        # reusing an offset that is now too low.
        self._resume_baselines.pop(stage_name, None)
        self.state.mark_stage_running(stage_name)
        self.state.save(self.state_path)
        started = time.time()
        self.bus.publish(
            EventType.STAGE_STARTED,
            stage=stage_name,
            message=f"=== Starting {stage_label} stage ===",
            **self._stage_payload(stage_name),
        )

        stage_instance = self._instantiate_stage(planned)
        try:
          result: StageResult = stage_instance.execute(
              live_line_callback=self._stage_line_callback(stage_name),
              stop_requested_callback=self._should_interrupt_stage,
          )
          self.state.record_stage_result(stage_name, result)
          self.state.save(self.state_path)

          if result.status == StageStatus.FAILED:
            raise RuntimeError(f"Stage '{stage_name}' failed: {result.error_message}")

          # An ``[a]`` advance request only applies to the stage it interrupted.
          self.controls.clear_advance()
          self.bus.publish(
              EventType.STAGE_COMPLETED,
              stage=stage_name,
              message=(
                  f"{stage_label} finished in "
                  f"{time.time() - started:.0f}s"
                  + (
                      f" (best={result.best_metric_val:.5f})"
                      if result.best_metric_val is not None
                      else ""
                  )
              ),
              metric=result.best_metric_val,
              model_repo_id=result.model_repo_id,
              duration_s=time.time() - started,
          )

        except Exception as stage_err:
          logger.error("Error executing stage '%s': %s", stage_name, stage_err)
          # Preserve the sweep id and any partial metrics: a resume must be
          # able to rejoin the sweep instead of paying for it a second time.
          self.state.mark_stage_failed(stage_name, str(stage_err))
          self.state.status = "FAILED"
          self.state.save(self.state_path)
          self.bus.publish(
              EventType.STAGE_FAILED,
              stage=stage_name,
              message=f"Stage '{stage_name}' failed: {stage_err}",
              error=str(stage_err),
          )
          raise

      if interrupted:
        self.bus.publish(
            EventType.CAMPAIGN_FINISHED,
            message="Campaign stopped before completion.",
            status="STOPPED",
        )
        if self.controls.abort_requested:
          return {"status": "STOPPED", "state_file": self.state_path}
        artifacts = self.reporter.generate_all()
        return {
            "status": "STOPPED",
            "artifacts": artifacts,
            "state_file": self.state_path,
          "log_file": self.log_path,
        }

      # All stages completed
      self.state.status = "COMPLETED"
      self.state.current_stage = None
      self.state.save(self.state_path)
      logger.info("All campaign stages completed successfully!")

      # Generate scientific report
      artifacts = self.reporter.generate_all()
      self.bus.publish(
          EventType.CAMPAIGN_FINISHED,
          message="Campaign completed successfully.",
          status="COMPLETED",
          artifacts=artifacts,
      )
      return {
          "status": "COMPLETED",
          "artifacts": artifacts,
          "state_file": self.state_path,
          "log_file": self.log_path,
      }

    except Exception as e:
      logger.error("Campaign failed: %s", e)
      self.state.status = "FAILED"
      self.state.save(self.state_path)
      self.bus.publish(
          EventType.CAMPAIGN_FINISHED,
          message=f"Campaign failed: {e}",
          status="FAILED",
          error=str(e),
      )
      raise
