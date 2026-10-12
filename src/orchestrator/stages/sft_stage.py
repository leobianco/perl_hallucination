"""Supervised Fine-Tuning (SFT) sweep and model materialization stage."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple
import yaml
from src.orchestrator.stages.base import BaseStage
from src.orchestrator.state import StageResult, StageStatus

logger = logging.getLogger(__name__)


class SftStage(BaseStage):
  """Orchestrates the SFT hyperparameter sweep and pushes the winning model."""

  @property
  def kind(self) -> str:
    return "sft"

  def get_sweep_descriptor(self, sweep_dict: Dict[str, Any]) -> Tuple[str, str]:
    model = self.config.base_model
    cmd = sweep_dict.get("command", [])
    if isinstance(cmd, list):
      for arg in cmd:
        if isinstance(arg, str) and arg.startswith("--model_repo_id="):
          model = arg.split("=", 1)[1]
    model_short = model.rstrip("/").split("/")[-1]
    return model_short, "SFT"

  def execute(
      self,
      live_line_callback: Optional[Callable[[str], None]] = None,
      stop_requested_callback: Optional[Callable[[], bool]] = None,
  ) -> StageResult:
    cfg = self.config.sft
    if not cfg.enabled:
      logger.info("SFT Stage is disabled. Skipping.")
      return StageResult(status=StageStatus.SKIPPED)

    yaml_path = cfg.sweep_config_path or "scripts/sweep_sft.yaml"
    if not os.path.exists(yaml_path) and not self.config.dry_run:
      raise FileNotFoundError(f"SFT sweep config file not found: {yaml_path}")

    logger.info("=== Starting SFT Stage for task: %s ===", self.config.task_name)
    if live_line_callback:
      live_line_callback(f"=== Starting SFT Stage for task: {self.config.task_name} ===")

    # Load and adjust sweep configuration
    sweep_dict = {}
    if os.path.exists(yaml_path):
      with open(yaml_path, "r", encoding="utf-8") as f:
        sweep_dict = yaml.safe_load(f)

    # Inject task name, dataset repo, model repo, and seed overrides
    expected_dataset = f"{self.config.user}/{self.config.task_name}_sft"
    if "command" in sweep_dict and isinstance(sweep_dict["command"], list):
      cmd_list = sweep_dict["command"]
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

      if not has_dataset:
        cmd_list.append(f"--dataset_repo_id={expected_dataset}")
      if not has_model:
        cmd_list.append(f"--model_repo_id={self.config.base_model}")

    # Must come after the model is pinned above: the launcher profile and the
    # memory flags are derived from which model this sweep will actually
    # train, not from whatever the YAML shipped with.
    self.apply_launcher_settings(sweep_dict, live_line_callback)

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
    logger.info("Registered SFT Sweep: %s%s", sweep_id, name_str)
    if live_line_callback:
      live_line_callback(f"Registered SFT Sweep: {sweep_id}{name_str}")

    # 2. Run the trials still missing from the budget and establish, against
    # W&B, how many of them actually finished. See
    # BaseStage.account_for_trials: a sweep that loses trials to a crash
    # must not report like one that ran clean.
    execution = self.run_sweep_trials(
        sweep_id=sweep_id,
        stage_config=cfg,
        live_line_callback=live_line_callback,
        stop_requested_callback=stop_requested_callback,
    )

    # 3. Query Best Run. Under selection_strategy="best" this is the trial
    # with the lowest eval loss at *any* eval step, which is the checkpoint
    # the materialization below will publish.
    winner = self.select_winner(
        sweep_id=sweep_id,
        stage_config=cfg,
        metric_name=target_metric,
        live_line_callback=live_line_callback,
    )

    # 4. Materialize and Push to Hugging Face Hub
    # NOTE: the sweep-agent predicate is deliberately *not* reused here; see
    # BaseStage.materialization_stop_callback.
    if live_line_callback:
      live_line_callback("Materializing and pushing best SFT model to Hugging Face...")
    sft_repo_id = self.model_manager.materialize_and_push(
        stage_name="sft",
        task_name=self.config.task_name,
        base_model=self.config.base_model,
        best_params=winner.params,
        seed=self.config.seed,
        live_line_callback=live_line_callback,
        tunable_keys=self.tunable_keys(sweep_dict),
        stop_requested_callback=self.materialization_stop_callback(),
        deepspeed_config=self.config.deepspeed_config_for(self.kind),
        memory_flags=self.config.memory_flags_for(self.kind),
        **self.materialization_checkpoint_kwargs(cfg),
    )

    logger.info("SFT model pushed to: %s", sft_repo_id)
    if live_line_callback:
      live_line_callback(f"SFT model published to Hub: {sft_repo_id}")

    return StageResult(
        status=StageStatus.COMPLETED,
        sweep_id=sweep_id,
        model_repo_id=sft_repo_id,
        metrics={"eval_loss": winner.value},
        **self.stage_result_selection_fields(winner),
        **execution.stage_result_fields(
            extra_warnings=self.selection_warnings(winner)
        ),
    )
