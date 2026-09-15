"""W&B Sweep lifecycle manager, execution agent, and best-run query controller."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SweepController:
  """Controls W&B hyperparameter sweep registration, agent execution, and best-run extraction."""

  def __init__(
      self,
      entity: str = "leobianco",
      project: str = "new_perl",
      dry_run: bool = False,
  ):
    self.entity = entity
    self.project = project
    self.dry_run = dry_run

  def create_sweep(
      self,
      sweep_config: Dict[str, Any],
      project_override: Optional[str] = None,
  ) -> str:
    """Registers a new sweep configuration with the W&B backend."""
    proj = project_override or self.project
    if self.dry_run:
      mock_id = f"mock_sweep_{int(time.time())}"
      logger.info(
          "[DRY-RUN] Registered simulated sweep: %s/%s/%s",
          self.entity,
          proj,
          mock_id,
      )
      return f"{self.entity}/{proj}/{mock_id}"

    try:
      import wandb  # pylint: disable=g-import-not-at-top

      sweep_id = wandb.sweep(sweep_config, entity=self.entity, project=proj)
      # Ensure sweep_id has full path
      if "/" not in str(sweep_id):
        return f"{self.entity}/{proj}/{sweep_id}"
      return str(sweep_id)
    except Exception as e:
      logger.error("Failed to register W&B sweep: %s", e)
      raise

  def run_sweep_agent(
      self,
      sweep_id: str,
      max_runs: int = 30,
      timeout_minutes: int = 240,
      live_line_callback: Optional[Callable[[str], None]] = None,
      stop_requested_callback: Optional[Callable[[], bool]] = None,
  ) -> int:
    """Executes the W&B sweep agent, bound by max_runs and timeout.

    Args:
        sweep_id: Full path to the sweep (entity/project/id) or short ID.
        max_runs: Strict upper limit of trials to execute before terminating.
        timeout_minutes: Maximum wall-clock time in minutes allowed for the
          sweep.
        live_line_callback: Optional callback receiving live stdout lines.
        stop_requested_callback: Optional predicate checking if user requested
          early stop.

    Returns:
        Exit code of the agent process.
    """
    if self.dry_run:
      logger.info(
          "[DRY-RUN] Simulating sweep agent for %s with max_runs=%d",
          sweep_id,
          max_runs,
      )
      for i in range(1, min(max_runs + 1, 4)):
        msg = (
            f"[DRY-RUN] Trial {i}/{max_runs} executed successfully (metric simulated)."
        )
        if live_line_callback:
          live_line_callback(msg)
        time.sleep(0.3)
      return 0

    # Format full sweep path
    full_sweep_id = sweep_id
    if "/" not in sweep_id:
      full_sweep_id = f"{self.entity}/{self.project}/{sweep_id}"

    cmd = ["wandb", "agent", "--count", str(max_runs), full_sweep_id]
    logger.info("Launching sweep agent: %s", " ".join(cmd))

    timeout_seconds = timeout_minutes * 60
    start_time = time.time()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    try:
      while process.poll() is None:
        if stop_requested_callback and stop_requested_callback():
          logger.info("Early stop requested by user. Terminating sweep agent...")
          process.terminate()
          try:
            process.wait(timeout=10)
          except subprocess.TimeoutExpired:
            process.kill()
          break

        if time.time() - start_time > timeout_seconds:
          logger.warning(
              "Sweep exceeded timeout threshold of %d minutes. Terminating...",
              timeout_minutes,
          )
          process.terminate()
          try:
            process.wait(timeout=15)
          except subprocess.TimeoutExpired:
            process.kill()
          break

        line = process.stdout.readline() if process.stdout else ""
        if line:
          clean_line = line.rstrip()
          if live_line_callback:
            live_line_callback(clean_line)
        else:
          time.sleep(0.1)

      # Drain any remaining lines in buffer
      if process.stdout:
        for remaining in process.stdout.readlines():
          clean = remaining.rstrip()
          if clean and live_line_callback:
            live_line_callback(clean)

      return process.returncode or 0
    except Exception as e:
      logger.error("Error during sweep agent execution: %s", e)
      if process.poll() is None:
        process.kill()
      raise
    finally:
      self.stop_sweep(full_sweep_id)

  def stop_sweep(self, sweep_id: str) -> None:
    """Closes and seals a sweep on the W&B backend, transitioning state to STOPPED."""
    if self.dry_run:
      logger.info("[DRY-RUN] Sealed sweep: %s", sweep_id)
      return

    full_sweep_id = sweep_id.strip()
    parts = full_sweep_id.split("/")
    if len(parts) == 1:
      full_sweep_id = f"{self.entity}/{self.project}/{parts[0]}"
    elif len(parts) == 2:
      full_sweep_id = f"{self.entity}/{parts[0]}/{parts[1]}"

    # Attempt 1: Via wandb.Api
    try:
      import wandb  # pylint: disable=g-import-not-at-top

      api = wandb.Api()
      sweep = api.sweep(full_sweep_id)
      if hasattr(sweep, "stop") and callable(sweep.stop):
        sweep.stop()
        logger.info("Successfully stopped sweep via W&B API: %s", full_sweep_id)
        return
    except Exception as e:
      logger.debug("API sweep.stop() did not succeed (%s), falling back to CLI", e)

    # Attempt 2: Via wandb CLI
    try:
      subprocess.run(
          ["wandb", "sweep", "--stop", full_sweep_id],
          check=False,
          capture_output=True,
          timeout=15,
      )
      logger.info("Executed wandb sweep --stop for %s", full_sweep_id)
    except Exception as e:
      logger.warning("Could not stop sweep via CLI: %s", e)

  def fetch_best_run(
      self,
      sweep_id: str,
      metric_name: str,
      goal: str = "minimize",
  ) -> Tuple[str, float, Dict[str, Any]]:
    """Retrieves the best run, its metric value, and configuration from the sweep.

    Args:
        sweep_id: Full path to sweep.
        metric_name: Target metric key (e.g. 'eval/loss', 'eval/roc_auc',
          'rewards/reward_fn/mean').
        goal: 'minimize' or 'maximize'.

    Returns:
        Tuple of (best_run_id, best_metric_val, best_hyperparameters_dict).
    """
    if self.dry_run:
      mock_run_id = f"run_{int(time.time())}"
      mock_metric = 0.285 if goal == "minimize" else 0.965
      mock_params = {
          "learning_rate": 0.0025,
          "num_train_epochs": 1,
          "lora_r": 8,
          "lora_alpha": 16,
          "beta": 0.05,
          "temperature": 0.7,
      }
      logger.info(
          "[DRY-RUN] Fetched mock best run %s with %s=%f",
          mock_run_id,
          metric_name,
          mock_metric,
      )
      return mock_run_id, mock_metric, mock_params

    try:
      import wandb  # pylint: disable=g-import-not-at-top

      api = wandb.Api()
      full_sweep_id = sweep_id.strip()
      parts = full_sweep_id.split("/")
      if len(parts) == 1:
        full_sweep_id = f"{self.entity}/{self.project}/{parts[0]}"
      elif len(parts) == 2:
        full_sweep_id = f"{self.entity}/{parts[0]}/{parts[1]}"

      sweep = api.sweep(full_sweep_id)
      runs = list(sweep.runs)

      if not runs:
        raise ValueError(f"No runs found in sweep {full_sweep_id}")

      def _extract_metric(run_summary: Dict[str, Any], target: str) -> Optional[float]:
        candidates = [
            target,
            target.replace("/", "_"),
            target.replace("_", "/"),
        ]
        if "train/" in target:
          without_train = target.replace("train/", "")
          candidates.extend([without_train, without_train.replace("/", "_")])
        if "eval/" in target:
          without_eval = target.replace("eval/", "")
          candidates.extend([without_eval, without_eval.replace("/", "_")])
        candidates.append(target.split("/")[-1])

        for cand in candidates:
          if cand in run_summary:
            v = run_summary[cand]
            if v is not None and isinstance(v, (int, float)):
              try:
                val_float = float(v)
                if val_float == val_float and abs(val_float) != float("inf"):
                  return val_float
              except (ValueError, TypeError):
                continue
        return None

      finished_runs: List[Tuple[Any, float]] = []
      other_runs: List[Tuple[Any, float]] = []

      for run in runs:
        val = _extract_metric(run.summary, metric_name)
        if val is not None:
          if run.state == "finished":
            finished_runs.append((run, val))
          elif run.state in ("running", "failed"):
            other_runs.append((run, val))

      # Strictly prioritize finished runs over failed or running ones
      valid_runs = finished_runs if finished_runs else other_runs

      if not valid_runs:
        logger.warning(
            "No runs in sweep %s logged metric '%s'. Inspecting first available run.",
            full_sweep_id,
            metric_name,
        )
        fallback_run = runs[0]
        return fallback_run.id, 0.0, fallback_run.config

      # Sort by metric
      reverse = goal == "maximize"
      valid_runs.sort(key=lambda x: x[1], reverse=reverse)
      best_run, best_val = valid_runs[0]

      logger.info(
          "Best run for %s is %s with %s=%.5f",
          sweep_id,
          best_run.id,
          metric_name,
          best_val,
      )
      return best_run.id, best_val, best_run.config

    except Exception as e:
      logger.error("Failed to query best run from W&B API: %s", e)
      raise
