"""Parameter-Efficient Reinforcement Learning (PE-RL / RLOO) sweep and model materialization stage."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple
import yaml
from src.orchestrator import flavors
from src.orchestrator.stages.base import BaseStage
from src.orchestrator.state import StageResult, StageStatus

logger = logging.getLogger(__name__)


class PerlStage(BaseStage):
  """Orchestrates the PE-RL (RLOO) hyperparameter sweep and pushes the winning model."""

  @property
  def kind(self) -> str:
    return "perl"

  def get_sweep_descriptor(self, sweep_dict: Dict[str, Any]) -> Tuple[str, str]:
    model = self.config.base_model
    rm_model = (
        self.context.reward_model_repo_id_for(self.flavor)
        or getattr(self.config.perl, "reward_model_path", None)
        or ""
    )
    cmd = sweep_dict.get("command", [])
    if isinstance(cmd, list):
      for arg in cmd:
        if isinstance(arg, str):
          if arg.startswith("--model_repo_id="):
            model = arg.split("=", 1)[1]
          elif arg.startswith("--reward_model_path="):
            rm_model = arg.split("=", 1)[1]
    model_short = model.rstrip("/").split("/")[-1]
    # See RmStage.get_sweep_descriptor: the branch's own flavor beats a
    # keyword sniff of the reward model's repo id.
    data_type = flavors.flavor_title(self.flavor)
    if not data_type:
      is_synth = any(
          k in str(rm_model).lower() for k in ("synthetic", "llm", "erased")
      )
      data_type = "Synthetic" if is_synth else "Organic"
    return model_short, f"PERL {data_type}"

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
    # This branch's own reward model, not "the" reward model: a two-flavor
    # campaign has one per branch and crossing them would silently compare
    # the wrong pair.
    rm_model = self.context.reward_model_repo_id_for(self.flavor)

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

    # 1. Register Sweep (or resume the one an interrupted attempt created)
    sweep_id = self.resolve_sweep_id(sweep_dict, live_line_callback)
    sweep_name = sweep_dict.get("name")
    name_str = f" ({sweep_name})" if sweep_name else ""
    logger.info("Registered PE-RL Sweep: %s%s", sweep_id, name_str)
    if live_line_callback:
      live_line_callback(f"Registered PE-RL Sweep: {sweep_id}{name_str}")

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

    # 3. Query Best Run. Unlike SFT and RM this stage keeps
    # selection_strategy="final": the reward is a *training* signal logged
    # every step over 8 sampled generations, so its maximum is reached
    # early, before the policy stabilises. Ranking on that peak selects the
    # luckiest batch of completions rather than the best configuration.
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
      live_line_callback("Materializing and pushing best PE-RL policy to Hugging Face...")
    perl_repo_id = self.model_manager.materialize_and_push(
        stage_name="perl",
        flavor=self.flavor,
        task_name=self.config.task_name,
        base_model=self.config.base_model,
        best_params=winner.params,
        seed=self.config.seed,
        sft_model_path=sft_model,
        reward_model_path=rm_model,
        live_line_callback=live_line_callback,
        tunable_keys=self.tunable_keys(sweep_dict),
        stop_requested_callback=self.materialization_stop_callback(),
        **self.materialization_checkpoint_kwargs(cfg),
    )

    logger.info("PE-RL model pushed to: %s", perl_repo_id)
    if live_line_callback:
      live_line_callback(f"PE-RL model published to Hub: {perl_repo_id}")

    return StageResult(
        status=StageStatus.COMPLETED,
        sweep_id=sweep_id,
        model_repo_id=perl_repo_id,
        metrics={"rewards/reward_fn/mean": winner.value},
        **self.stage_result_selection_fields(winner),
        **execution.stage_result_fields(
            extra_warnings=self.selection_warnings(winner)
        ),
    )
