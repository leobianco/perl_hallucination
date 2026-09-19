"""Autorater calibration: how much the Gemini judge can be trusted.

Every headline number this campaign produces - hallucination rate,
faithfulness rate, the SFT-to-PE-RL delta - is a count of verdicts handed
down by a Gemini judge against a fixed decision threshold. The threshold
used to be a constant (``0.1025``) carried in the configuration, fitted by
hand at some point on some task with some number of few-shot examples. A
campaign consumed it without ever checking that it still separated anything.

This stage closes that loop. It scores the human-labelled set
``{user}/{task}_autorater`` with the same judge, the same few-shot count and
the same seed the final evaluation will use, fits the threshold that
maximises the ROC operating point, and hands it downstream.

It runs *first*, before SFT, for two reasons. It depends on nothing the
campaign trains, so there is no ordering constraint to respect; and if the
judge turns out to be unable to separate the labels, that is worth knowing
before several GPU-hours are spent producing policies it cannot grade.

A weak judge is reported, not fatal. ``min_autorater_auc`` is advisory:
below it the stage raises a warning that follows the campaign into the
dashboard and the final report, and the campaign continues.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from src.orchestrator import eval_metrics
from src.orchestrator.process import stream_subprocess
from src.orchestrator.retry import run_with_retries
from src.orchestrator.stages.base import BaseStage
from src.orchestrator.state import StageResult, StageStatus
from src.utils import autorater_eval_metrics_path

logger = logging.getLogger(__name__)

#: Metric namespace. Mirrors the ``sft/``-style prefixes the evaluation
#: stage writes, so a reader of the state file can tell at a glance which
#: stage produced a number. Defined in :mod:`eval_metrics` alongside the
#: report's rendering of it, because the report is often built from a state
#: file alone, with no stage instance in sight.
PREFIX = eval_metrics.AUTORATER_PREFIX


class AutoraterStage(BaseStage):
  """Measures the judge against human labels and fits its threshold."""

  @property
  def kind(self) -> str:
    return "autorater"

  def execute(
      self,
      live_line_callback: Optional[Callable[[str], None]] = None,
      stop_requested_callback: Optional[Callable[[], bool]] = None,
  ) -> StageResult:
    cfg = self.config.eval
    if not cfg.enabled:
      # Calibration exists to serve the evaluation. With no evaluation to
      # serve it is a Gemini bill for a number nobody reads.
      logger.info(
          "Evaluation is disabled, so autorater calibration has nothing to "
          "calibrate for. Skipping."
      )
      return StageResult(status=StageStatus.SKIPPED)

    dataset_labels = self.labels_dataset()
    logger.info(
        "=== Calibrating the %s autorater against %s ===",
        cfg.evaluator_model,
        dataset_labels,
    )
    if live_line_callback:
      live_line_callback(
          f"Judge: {cfg.evaluator_model} "
          f"({cfg.evaluator_num_fewshot}-shot, k={cfg.autorater_num_samples})"
      )
      live_line_callback(f"Labelled set: {dataset_labels}")

    if self.config.dry_run:
      return self._dry_run_result(live_line_callback)

    command = self.build_command()
    run_with_retries(
        lambda: self._run_subprocess(
            command, live_line_callback, stop_requested_callback
        ),
        description="autorater calibration",
        attempts=self.config.robustness.eval_attempts,
        base_delay_s=self.config.robustness.retry_base_delay_s,
        on_notice=live_line_callback,
        stop_requested=stop_requested_callback,
    )

    calibration = self.load_calibration()
    warnings = self.quality_warnings(calibration)
    for warning in warnings:
      logger.warning(warning)
      if live_line_callback:
        live_line_callback(warning)

    threshold = calibration.get("best_threshold")
    roc_auc = calibration.get("roc_auc")
    if live_line_callback:
      live_line_callback(
          f"Fitted threshold {threshold!r} at ROC-AUC {roc_auc:.4f}; "
          "the evaluation stage will score at this operating point."
      )

    return StageResult(
        status=StageStatus.COMPLETED,
        best_metric_val=roc_auc,
        metrics={f"{PREFIX}/{key}": value for key, value in calibration.items()},
        warnings=warnings,
    )

  # --- Inputs -----------------------------------------------------------

  def labels_dataset(self) -> str:
    """Returns the human-labelled set the judge is scored against."""
    return f"{self.config.user}/{self.config.task_name}_autorater"

  # --- Command construction ---------------------------------------------

  def build_command(self) -> List[str]:
    """Builds the ``--mode autoratereval`` invocation.

    Every judge setting is taken from ``config.eval`` rather than from a
    configuration of its own. A threshold fitted with four few-shot examples
    does not transfer to a judge given two, so letting the two stages
    disagree would produce a calibration that silently does not apply to the
    scoring it was fitted for.

    Returns:
      The argument vector.
    """
    cfg = self.config.eval
    cmd = [
        "python3",
        "-m",
        "src.evaluator",
        "--mode",
        "autoratereval",
        "--task_name",
        self.config.task_name,
        "--user",
        self.config.user,
        "--seed",
        str(cfg.seed),
        "--max_eval_samples",
        str(cfg.max_eval_samples),
        "--eval_batch_size",
        str(cfg.eval_batch_size),
        "--max_workers",
        str(cfg.max_workers),
        "--dataset_labels",
        self.labels_dataset(),
        "--dataset_labels_split",
        "test",
        # Required by the argument parser but unused by this mode, which
        # never generates anything: calibration runs before any policy
        # exists, so there is no adapter to name here.
        "--writer_model_lora",
        self.config.base_model,
        "--evaluator_model",
        cfg.evaluator_model,
        "--use_gemini",
        str(cfg.use_gemini),
        "--evaluator_num_fewshot",
        str(cfg.evaluator_num_fewshot),
        "--autorater_num_samples",
        str(cfg.autorater_num_samples),
        "--evaluate_evaluator",
        "True",
        # The incumbent constant. This mode fits its own and reports both,
        # so passing it costs nothing and keeps the run reproducible from
        # the command line alone.
        "--threshold",
        str(cfg.threshold),
    ]
    if os.environ.get("GEMINI_API_KEY"):
      cmd.extend(["--gemini_api_key", os.environ["GEMINI_API_KEY"]])
    return cmd

  # --- Results ----------------------------------------------------------

  def calibration_path(self) -> str:
    """Returns the JSON file the calibration pipeline writes."""
    cfg = self.config.eval
    return autorater_eval_metrics_path(
        evaluator_model=cfg.evaluator_model,
        dataset_labels=self.labels_dataset(),
        evaluator_num_fewshot=cfg.evaluator_num_fewshot,
        seed=cfg.seed,
    )

  def load_calibration(self) -> Dict[str, Any]:
    """Reads the fitted threshold and the judge's quality metrics.

    Returns:
      The parsed calibration.

    Raises:
      RuntimeError: When the file is missing, unreadable, or carries no
        usable threshold. Continuing would mean scoring the whole campaign
        at a threshold nobody chose.
    """
    path = self.calibration_path()
    if not os.path.isfile(path):
      raise RuntimeError(
          f"Autorater calibration finished but wrote no metrics to {path}. "
          "The judge calls likely failed; inspect the log above."
      )
    try:
      with open(path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    except (OSError, ValueError) as error:
      raise RuntimeError(
          f"Could not read the autorater calibration {path}: {error}"
      ) from error
    if not isinstance(loaded, dict):
      raise RuntimeError(f"Autorater calibration {path} is not a JSON object.")
    threshold = loaded.get("best_threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
      raise RuntimeError(
          f"Autorater calibration {path} carries no numeric "
          f"'best_threshold' (got {threshold!r})."
      )
    return loaded

  def quality_warnings(self, calibration: Dict[str, Any]) -> List[str]:
    """Flags a judge whose verdicts should not be taken at face value.

    Args:
      calibration: The parsed calibration metrics.

    Returns:
      Human-readable warnings, empty when the judge looks sound. They are
      attached to the stage result, so they reach the dashboard and survive
      into the final report.
    """
    warnings: List[str] = []
    floor = self.config.eval.min_autorater_auc
    roc_auc = calibration.get("roc_auc")
    if isinstance(roc_auc, (int, float)) and roc_auc < floor:
      warnings.append(
          f"[WARNING] Autorater ROC-AUC is {roc_auc:.4f}, below the "
          f"{floor:.2f} floor. The judge separates hallucinated from "
          "faithful completions poorly, so every hallucination rate and "
          "every SFT-to-PE-RL delta downstream carries more error than the "
          "report's precision suggests. Treat the comparison as indicative."
      )
    if calibration.get("degenerate_roc"):
      warnings.append(
          "[WARNING] The autorater's ROC curve collapsed to a single "
          "operating point: its scores carry no ranking information beyond "
          "the binary Yes/No verdict, and the fitted threshold sits inside "
          "the saturated cluster where a 1e-5 difference flips a label."
      )
    return warnings

  # --- Plumbing ---------------------------------------------------------

  def _dry_run_result(
      self, live_line_callback: Optional[Callable[[str], None]]
  ) -> StageResult:
    """Produces a plausible calibration without calling Gemini."""
    logger.info("[DRY-RUN] Simulating autorater calibration...")
    if live_line_callback:
      live_line_callback("[DRY-RUN] Scoring the labelled set with the judge...")
    time.sleep(0.4)
    calibration = {
        "roc_auc": 0.913,
        "best_threshold": 0.1025,
        "tpr_at_best_threshold": 0.884,
        "fpr_at_best_threshold": 0.121,
        "accuracy_at_best_threshold": 0.881,
        "balanced_accuracy": 0.881,
        "precision_at_best_threshold": 0.872,
        "scored_samples": self.config.eval.max_eval_samples,
        "unscored_samples": 0,
        "degenerate_roc": False,
    }
    if live_line_callback:
      live_line_callback(
          "[DRY-RUN] Fitted threshold 0.1025 at ROC-AUC 0.9130."
      )
    return StageResult(
        status=StageStatus.COMPLETED,
        best_metric_val=calibration["roc_auc"],
        metrics={f"{PREFIX}/{k}": v for k, v in calibration.items()},
    )

  def _run_subprocess(
      self,
      cmd: List[str],
      live_line_callback: Optional[Callable[[str], None]],
      stop_requested_callback: Optional[Callable[[], bool]] = None,
  ) -> None:
    """Runs the calibration subprocess.

    Args:
      cmd: Argument vector.
      live_line_callback: Log sink for the streamed output.
      stop_requested_callback: Predicate used to cut the run short.

    Raises:
      RuntimeError: If the calibration is interrupted or exits non-zero.
    """
    env = os.environ.copy()
    env.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    env.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

    outcome = stream_subprocess(
        cmd,
        on_line=live_line_callback,
        stop_requested=stop_requested_callback,
        env=env,
        timeout_s=self.config.eval.timeout_minutes * 60,
        stall_warning_s=self.config.robustness.stall_warning_minutes * 60,
    )

    if outcome.interrupted:
      raise RuntimeError(
          "Autorater calibration was interrupted before it completed."
      )
    if outcome.timed_out:
      raise RuntimeError(
          "Autorater calibration exceeded its "
          f"{self.config.eval.timeout_minutes} minute budget."
      )
    if outcome.returncode != 0:
      raise RuntimeError(
          "Autorater calibration failed with exit code "
          f"{outcome.returncode}"
      )
