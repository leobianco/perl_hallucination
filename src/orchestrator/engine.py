"""Campaign execution engine coordinating the linear scientific DAG."""

from __future__ import annotations

import logging
import os
import shutil
import time
from typing import Any, Callable, Dict, Optional

from src.orchestrator.cli.events import ControlSignals, EventBus, EventType
from src.orchestrator.config import CampaignConfig
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
    """Line sink for a stage: the event bus plus the durable campaign log."""
    return logging_setup.tee_line_callback(
        self.log_path, self.bus.as_line_callback(stage_name)
    )

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

  def _instantiate_stage(self, stage_name: str) -> BaseStage:
    """Instantiates the concrete stage implementation by name."""
    if stage_name == "sft":
      return SftStage(self.context)
    elif stage_name == "rm":
      return RmStage(self.context)
    elif stage_name == "perl":
      return PerlStage(self.context)
    elif stage_name == "eval":
      return EvalStage(self.context)
    else:
      raise ValueError(f"Unknown stage name: {stage_name}")

  def _stage_payload(self, stage_name: str) -> Dict[str, Any]:
    """Metadata describing a stage, published with ``STAGE_STARTED``."""
    stage_cfg = getattr(self.config, stage_name, None)
    return {
        "metric": getattr(stage_cfg, "metric", None),
        "goal": getattr(stage_cfg, "goal", None),
        "max_runs": getattr(stage_cfg, "max_runs", 0),
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
    self.state.stages_order = list(self.config.stages)
    self.state.save(self.state_path)

    interrupted = False
    try:
      for stage_name in self.config.stages:
        # Check if already finished in previous run (resumption)
        if self.state.is_stage_completed(stage_name):
          logger.info("Stage '%s' is already completed. Skipping.", stage_name)
          self.bus.publish(
              EventType.STAGE_SKIPPED,
              stage=stage_name,
              message=f"[RESUME] Skipping already completed stage: {stage_name}",
              model_repo_id=self.state.get_model_repo_id(stage_name),
              metric=getattr(self.state.stages.get(stage_name), "best_metric_val", None),
          )
          continue

        # Honor a pause request before committing to a new stage.
        if self.controls.is_paused:
          self.bus.publish(
              EventType.NOTICE,
              message=f"Paused before stage '{stage_name}'. Press [p] to resume.",
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
              message=f"Campaign stopped by user before stage '{stage_name}'.",
          )
          break

        self._warn_on_low_disk(stage_name)
        self.state.mark_stage_running(stage_name)
        self.state.save(self.state_path)
        started = time.time()
        self.bus.publish(
            EventType.STAGE_STARTED,
            stage=stage_name,
            message=f"=== Starting {stage_name.upper()} stage ===",
            **self._stage_payload(stage_name),
        )

        stage_instance = self._instantiate_stage(stage_name)
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
                  f"{stage_name.upper()} finished in "
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
