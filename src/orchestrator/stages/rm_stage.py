"""Reward Model (RM) sweep and model materialization stage."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple
import yaml
from src.orchestrator.stages.base import BaseStage
from src.orchestrator.state import StageResult, StageStatus

logger = logging.getLogger(__name__)


class RmStage(BaseStage):
  """Orchestrates the Reward Model hyperparameter sweep and pushes the winning model."""

  @property
  def name(self) -> str:
    return "rm"

  def get_sweep_descriptor(self, sweep_dict: Dict[str, Any]) -> Tuple[str, str]:
    rm_model = "google/gemma-3-1b-it"
    dataset = f"{self.config.user}/{self.config.task_name}_rm_organic"
    cmd = sweep_dict.get("command", [])
    if isinstance(cmd, list):
      for arg in cmd:
        if isinstance(arg, str):
          if arg.startswith("--model_repo_id="):
            rm_model = arg.split("=", 1)[1]
          elif arg.startswith("--dataset_repo_id="):
            dataset = arg.split("=", 1)[1]
    model_short = rm_model.rstrip("/").split("/")[-1]
    is_synth = any(k in dataset.lower() for k in ("synthetic", "llm", "erased"))
    data_type = "Synthetic" if is_synth else "Organic"
    return model_short, f"RM {data_type}"

  def execute(
      self,
      live_line_callback: Optional[Callable[[str], None]] = None,
      stop_requested_callback: Optional[Callable[[], bool]] = None,
  ) -> StageResult:
    cfg = self.config.rm
    if not cfg.enabled:
      logger.info("Reward Model Stage is disabled. Skipping.")
      return StageResult(status=StageStatus.SKIPPED)

    yaml_path = cfg.sweep_config_path or "scripts/sweep_rm.yaml"
    if not os.path.exists(yaml_path) and not self.config.dry_run:
      raise FileNotFoundError(f"RM sweep config file not found: {yaml_path}")

    logger.info("=== Starting Reward Model Stage for task: %s ===", self.config.task_name)
    if live_line_callback:
      live_line_callback(f"=== Starting Reward Model Stage for task: {self.config.task_name} ===")

    # Load and adjust sweep configuration
    sweep_dict = {}
    rm_base_model = "google/gemma-3-1b-it"
    if os.path.exists(yaml_path):
      with open(yaml_path, "r", encoding="utf-8") as f:
        sweep_dict = yaml.safe_load(f)

    # Inject task name, dataset repo, model repo, and seed overrides
    expected_dataset = f"{self.config.user}/{self.config.task_name}_rm_organic"
    if "command" in sweep_dict and isinstance(sweep_dict["command"], list):
      cmd_list = sweep_dict["command"]
      has_dataset = False
      for idx, arg in enumerate(cmd_list):
        if isinstance(arg, str):
          if arg.startswith("--task_name="):
            cmd_list[idx] = f"--task_name={self.config.task_name}"
          elif arg.startswith("--seed="):
            cmd_list[idx] = f"--seed={self.config.seed}"
          elif arg.startswith("--dataset_repo_id="):
            cmd_list[idx] = f"--dataset_repo_id={expected_dataset}"
            has_dataset = True
          elif arg.startswith("--model_repo_id="):
            rm_base_model = arg.split("=", 1)[1]

      if not has_dataset:
        cmd_list.append(f"--dataset_repo_id={expected_dataset}")

    # Apply parameter search overrides if specified
    if cfg.parameter_overrides and "parameters" in sweep_dict:
      for param_name, param_spec in cfg.parameter_overrides.items():
        if isinstance(param_spec, list):
          sweep_dict["parameters"][param_name] = {"values": param_spec}
        elif isinstance(param_spec, dict):
          sweep_dict["parameters"][param_name] = param_spec

    # Align metric name with sweep YAML if not overridden
    yaml_metric = sweep_dict.get("metric", {}).get("name")
    target_metric = yaml_metric or cfg.metric

    # 1. Register Sweep (or resume the one an interrupted attempt created)
    sweep_id = self.resolve_sweep_id(sweep_dict, live_line_callback)
    sweep_name = sweep_dict.get("name")
    name_str = f" ({sweep_name})" if sweep_name else ""
    logger.info("Registered RM Sweep: %s%s", sweep_id, name_str)
    if live_line_callback:
      live_line_callback(f"Registered RM Sweep: {sweep_id}{name_str}")

    # 2. Run Sweep Agent for the trials still missing from the budget. On a
    # resume this is fewer than max_runs; see BaseStage.remaining_runs.
    remaining = self.remaining_runs(sweep_id, cfg.max_runs, live_line_callback)
    if remaining:
      exit_code = self.sweep_controller.run_sweep_agent(
          sweep_id=sweep_id,
          max_runs=remaining,
          timeout_minutes=cfg.timeout_minutes,
          live_line_callback=live_line_callback,
          stop_requested_callback=stop_requested_callback,
      )
      if exit_code != 0 and live_line_callback:
        live_line_callback(
            f"[WARNING] RM sweep agent exited with code {exit_code}; scoring "
            "the trials that did finish."
        )

    # 3. Query Best Run (maximizing eval/roc_auc)
    best_run_id, best_val, best_params = self.sweep_controller.fetch_best_run(
        sweep_id=sweep_id,
        metric_name=target_metric,
        goal=cfg.goal,
    )
    logger.info("Best RM Run: %s (%s=%.5f)", best_run_id, target_metric, best_val)
    if live_line_callback:
      live_line_callback(f"Best RM Run: {best_run_id} ({target_metric}={best_val:.5f})")

    # Make the winner findable in the W&B web UI. Best-effort by design.
    self.sweep_controller.mark_best_run(
        sweep_id=sweep_id,
        run_id=best_run_id,
        stage_name="rm",
        metric_name=target_metric,
        metric_value=best_val,
        live_line_callback=live_line_callback,
    )

    # 4. Materialize and Push to Hugging Face Hub
    # NOTE: the sweep-agent predicate is deliberately *not* reused here; see
    # BaseStage.materialization_stop_callback.
    if live_line_callback:
      live_line_callback("Materializing and pushing best RM to Hugging Face...")
    rm_repo_id = self.model_manager.materialize_and_push(
        stage_name="rm",
        task_name=self.config.task_name,
        base_model=rm_base_model,
        best_params=best_params,
        seed=self.config.seed,
        live_line_callback=live_line_callback,
        tunable_keys=self.tunable_keys(sweep_dict),
        stop_requested_callback=self.materialization_stop_callback(),
    )

    logger.info("RM model pushed to: %s", rm_repo_id)
    if live_line_callback:
      live_line_callback(f"RM model published to Hub: {rm_repo_id}")

    return StageResult(
        status=StageStatus.COMPLETED,
        sweep_id=sweep_id,
        best_run_id=best_run_id,
        best_metric_val=best_val,
        best_params=best_params,
        model_repo_id=rm_repo_id,
        metrics={"eval_roc_auc": best_val},
    )
