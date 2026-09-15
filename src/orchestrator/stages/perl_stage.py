"""Parameter-Efficient Reinforcement Learning (PE-RL / RLOO) sweep and model materialization stage."""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional
import yaml
from src.orchestrator.stages.base import BaseStage
from src.orchestrator.state import StageResult, StageStatus

logger = logging.getLogger(__name__)


class PerlStage(BaseStage):
  """Orchestrates the PE-RL (RLOO) hyperparameter sweep and pushes the winning model."""

  @property
  def name(self) -> str:
    return "perl"

  def execute(
      self,
      live_line_callback: Optional[Callable[[str], None]] = None,
      stop_requested_callback: Optional[Callable[[], bool]] = None,
  ) -> StageResult:
    cfg = self.config.perl
    if not cfg.enabled:
      logger.info("PE-RL Stage is disabled. Skipping.")
      return StageResult(status=StageStatus.SKIPPED)

    yaml_path = cfg.sweep_config_path or "scripts/sweep_perl.yaml"
    if not os.path.exists(yaml_path) and not self.config.dry_run:
      raise FileNotFoundError(f"PE-RL sweep config file not found: {yaml_path}")

    # Resolve SFT and RM model checkpoints
    sft_model = self.context.sft_model_repo_id
    rm_model = self.context.reward_model_repo_id

    if not self.config.dry_run:
      if not sft_model:
        raise ValueError(
            "PE-RL stage requires an SFT model checkpoint. None found in state or config."
        )
      if not rm_model:
        raise ValueError(
            "PE-RL stage requires a Reward Model checkpoint. None found in state or config."
        )

    logger.info("=== Starting PE-RL Stage for task: %s ===", self.config.task_name)
    logger.info("  Using SFT Model: %s", sft_model)
    logger.info("  Using Reward Model: %s", rm_model)
    if live_line_callback:
      live_line_callback(f"=== Starting PE-RL Stage for task: {self.config.task_name} ===")
      live_line_callback(f"  Using SFT Model: {sft_model}")
      live_line_callback(f"  Using Reward Model: {rm_model}")

    # Load base sweep YAML
    sweep_dict = {}
    if os.path.exists(yaml_path):
      with open(yaml_path, "r", encoding="utf-8") as f:
        sweep_dict = yaml.safe_load(f)

    # Dynamic Dependency Injection: inject SFT & RM model paths, dataset repo, model repo, task_name, and seed
    expected_dataset = f"{self.config.user}/{self.config.task_name}_perl"
    if "command" in sweep_dict and isinstance(sweep_dict["command"], list):
      cmd_list = sweep_dict["command"]
      has_sft = False
      has_rm = False
      has_dataset = False
      has_model = False

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
            cmd_list[idx] = f"--model_repo_id={self.config.base_model}"
            has_model = True
          elif arg.startswith("--sft_model_path="):
            cmd_list[idx] = f"--sft_model_path={sft_model}"
            has_sft = True
          elif arg.startswith("--reward_model_path="):
            cmd_list[idx] = f"--reward_model_path={rm_model}"
            has_rm = True

      # Append if not found in template
      if not has_dataset:
        cmd_list.append(f"--dataset_repo_id={expected_dataset}")
      if not has_model:
        cmd_list.append(f"--model_repo_id={self.config.base_model}")
      if not has_sft and sft_model:
        cmd_list.append(f"--sft_model_path={sft_model}")
      if not has_rm and rm_model:
        cmd_list.append(f"--reward_model_path={rm_model}")

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

    # 1. Register Sweep
    sweep_id = self.sweep_controller.create_sweep(sweep_dict)
    logger.info("Registered PE-RL Sweep: %s", sweep_id)
    if live_line_callback:
      live_line_callback(f"Registered PE-RL Sweep: {sweep_id}")

    # 2. Run Sweep Agent with max_runs bound (default: 10)
    self.sweep_controller.run_sweep_agent(
        sweep_id=sweep_id,
        max_runs=cfg.max_runs,
        timeout_minutes=cfg.timeout_minutes,
        live_line_callback=live_line_callback,
        stop_requested_callback=stop_requested_callback,
    )

    # 3. Query Best Run (maximizing rewards/reward_fn/mean)
    best_run_id, best_val, best_params = self.sweep_controller.fetch_best_run(
        sweep_id=sweep_id,
        metric_name=target_metric,
        goal=cfg.goal,
    )
    logger.info("Best PE-RL Run: %s (%s=%.5f)", best_run_id, target_metric, best_val)
    if live_line_callback:
      live_line_callback(f"Best PE-RL Run: {best_run_id} ({target_metric}={best_val:.5f})")

    # 4. Materialize and Push to Hugging Face Hub
    if live_line_callback:
      live_line_callback("Materializing and pushing best PE-RL policy to Hugging Face...")
    perl_repo_id = self.model_manager.materialize_and_push(
        stage_name="perl",
        task_name=self.config.task_name,
        base_model=self.config.base_model,
        best_params=best_params,
        seed=self.config.seed,
        sft_model_path=sft_model,
        reward_model_path=rm_model,
        live_line_callback=live_line_callback,
    )
    logger.info("PE-RL model pushed to: %s", perl_repo_id)
    if live_line_callback:
      live_line_callback(f"PE-RL model published to Hub: {perl_repo_id}")

    return StageResult(
        status=StageStatus.COMPLETED,
        sweep_id=sweep_id,
        best_run_id=best_run_id,
        best_metric_val=best_val,
        best_params=best_params,
        model_repo_id=perl_repo_id,
        metrics={"rewards/reward_fn/mean": best_val},
    )
