"""Reward Model (RM) sweep and model materialization stage."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple
import yaml
from src.orchestrator import flavors
from src.orchestrator.stages.base import BaseStage
from src.orchestrator.state import StageResult, StageStatus

logger = logging.getLogger(__name__)


class RmStage(BaseStage):
  """Orchestrates the Reward Model hyperparameter sweep and pushes the winning model."""

  @property
  def kind(self) -> str:
    return "rm"

  def dataset_repo_id(self) -> str:
    """Returns the Hub dataset this branch trains its reward model on.

    This is the single source of truth for the choice, and it is applied to
    both the sweep trials and the winner's retraining. It used to be the
    literal ``{user}/{task}_rm_organic`` in two separate places, which is
    why editing ``--dataset_repo_id`` in ``scripts/sweep_rm.yaml`` had no
    effect: the stage overwrote it on the way past.
    """
    return flavors.dataset_repo_id(
        user=self.config.user,
        task_name=self.config.task_name,
        flavor=self.flavor or flavors.campaign_flavors(self.config)[0],
    )

  def get_sweep_descriptor(self, sweep_dict: Dict[str, Any]) -> Tuple[str, str]:
    rm_model = "google/gemma-4-E4B-it"
    dataset = self.dataset_repo_id()
    cmd = sweep_dict.get("command", [])
    if isinstance(cmd, list):
      for arg in cmd:
        if isinstance(arg, str):
          if arg.startswith("--model_repo_id="):
            rm_model = arg.split("=", 1)[1]
          elif arg.startswith("--dataset_repo_id="):
            dataset = arg.split("=", 1)[1]
    model_short = rm_model.rstrip("/").split("/")[-1]
    # Prefer the flavor this branch was built for: two branches of the same
    # campaign must not both end up named "RM Synthetic". The keyword sniff
    # stays as the fallback for a dataset pinned in the YAML by hand.
    data_type = flavors.flavor_title(self.flavor)
    if not data_type:
      is_synth = any(
          k in dataset.lower() for k in ("synthetic", "llm", "erased")
      )
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
    rm_base_model = "google/gemma-4-E4B-it"
    if os.path.exists(yaml_path):
      with open(yaml_path, "r", encoding="utf-8") as f:
        sweep_dict = yaml.safe_load(f)

    # Inject task name, dataset repo, model repo, and seed overrides
    expected_dataset = self.dataset_repo_id()
    logger.info("  Reward model dataset: %s", expected_dataset)
    if live_line_callback:
      live_line_callback(f"  Reward model dataset: {expected_dataset}")
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
    # with the highest ROC-AUC at *any* eval step. A reward model is a
    # classifier we then hand to PE-RL as a reward, so a lucky peak is worth
    # watching: BaseStage.select_winner reports the gap to the final value.
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
      live_line_callback("Materializing and pushing best RM to Hugging Face...")
    rm_repo_id = self.model_manager.materialize_and_push(
        stage_name="rm",
        flavor=self.flavor,
        task_name=self.config.task_name,
        base_model=rm_base_model,
        best_params=winner.params,
        seed=self.config.seed,
        live_line_callback=live_line_callback,
        tunable_keys=self.tunable_keys(sweep_dict),
        stop_requested_callback=self.materialization_stop_callback(),
        **self.materialization_checkpoint_kwargs(cfg),
    )

    logger.info("RM model pushed to: %s", rm_repo_id)
    if live_line_callback:
      live_line_callback(f"RM model published to Hub: {rm_repo_id}")

    return StageResult(
        status=StageStatus.COMPLETED,
        sweep_id=sweep_id,
        model_repo_id=rm_repo_id,
        metrics={"eval_roc_auc": winner.value},
        **self.stage_result_selection_fields(winner),
        **execution.stage_result_fields(
            extra_warnings=self.selection_warnings(winner)
        ),
    )
