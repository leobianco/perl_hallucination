"""Evaluation and autorating stage (generation + Gemini scoring)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import Any, Callable, Dict, Optional
from src.orchestrator.stages.base import BaseStage
from src.orchestrator.state import StageResult, StageStatus
from src.utils import build_eval_dataset_repo_id

logger = logging.getLogger(__name__)


class EvalStage(BaseStage):
  """Orchestrates test completion generation and multi-metric autorating."""

  @property
  def name(self) -> str:
    return "eval"

  def execute(
      self,
      live_line_callback: Optional[Callable[[str], None]] = None,
      stop_requested_callback: Optional[Callable[[], bool]] = None,
  ) -> StageResult:
    cfg = self.config.eval
    if not cfg.enabled:
      logger.info("Evaluation Stage is disabled. Skipping.")
      return StageResult(status=StageStatus.SKIPPED)

    # Determine model to evaluate
    model_to_eval = (
        self.context.perl_model_repo_id
        or self.context.sft_model_repo_id
        or self.config.base_model
    )

    logger.info("=== Starting Evaluation Stage for task: %s ===", self.config.task_name)
    logger.info("  Evaluating Model: %s", model_to_eval)
    if live_line_callback:
      live_line_callback(f"=== Starting Evaluation Stage for task: {self.config.task_name} ===")
      live_line_callback(f"  Evaluating Model: {model_to_eval}")

    if self.config.dry_run:
      logger.info("[DRY-RUN] Simulating generation and scoring...")
      if live_line_callback:
        live_line_callback("[DRY-RUN] Generating completions on test set...")
        live_line_callback("[DRY-RUN] Scoring completions with Gemini 2.5 Flash autorater...")
        live_line_callback("[DRY-RUN] Computing BertScore and Perplexity...")
      time.sleep(1.0)
      mock_metrics = {
          "hallucination_rate": 0.048,
          "win_rate_vs_base": 0.785,
          "bertscore_f1": 0.892,
          "perplexity": 7.84,
          "num_samples": cfg.max_eval_samples,
      }
      return StageResult(
          status=StageStatus.COMPLETED,
          model_repo_id=model_to_eval,
          metrics=mock_metrics,
      )

    # --- Phase 1: Generation ---
    if live_line_callback:
      live_line_callback("Step 1/2: Generating test completions with vLLM / HuggingFace...")

    gen_cmd = [
        "python3",
        "-m",
        "src.evaluator",
        "--mode",
        "generate",
        "--task_name",
        self.config.task_name,
        "--user",
        self.config.user,
        "--seed",
        str(self.config.seed),
        "--max_eval_samples",
        str(cfg.max_eval_samples),
        "--dataset_labels",
        f"{self.config.user}/{self.config.task_name}_autorater",
        "--dataset_labels_split",
        "test",
        "--dataset_prompts",
        f"{self.config.user}/{self.config.task_name}_final_test_set",
        "--dataset_prompts_split",
        "test",
        "--writer_model_base",
        self.config.base_model,
        "--writer_model_lora",
        model_to_eval,
        "--max_tokens",
        str(cfg.max_tokens),
        "--temperature",
        str(cfg.temperature),
        "--top_p",
        str(cfg.top_p),
        "--top_k",
        str(cfg.top_k),
        "--writer_num_fewshot",
        str(cfg.writer_num_fewshot),
    ]

    self._run_subprocess(gen_cmd, live_line_callback)

    # --- Phase 2: Scoring ---
    if live_line_callback:
      live_line_callback("Step 2/2: Scoring completions with Gemini autorater & metrics...")

    score_cmd = [
        "python3",
        "-m",
        "src.evaluator",
        "--mode",
        "score",
        "--task_name",
        self.config.task_name,
        "--user",
        self.config.user,
        "--seed",
        str(self.config.seed),
        "--max_eval_samples",
        str(cfg.max_eval_samples),
        "--eval_batch_size",
        str(cfg.eval_batch_size),
        "--max_workers",
        str(cfg.max_workers),
        "--dataset_labels",
        f"{self.config.user}/{self.config.task_name}_autorater",
        "--dataset_labels_split",
        "test",
        "--writer_model_base",
        self.config.base_model,
        "--writer_model_lora",
        model_to_eval,
        "--temperature",
        str(cfg.temperature),
        "--writer_num_fewshot",
        str(cfg.writer_num_fewshot),
        "--run_autorater",
        "True",
        "--evaluator_model",
        cfg.evaluator_model,
        "--use_gemini",
        str(cfg.use_gemini),
        "--evaluator_num_fewshot",
        str(cfg.evaluator_num_fewshot),
        "--threshold",
        str(cfg.threshold),
        "--compute_bertscore",
        str(cfg.compute_bertscore),
        "--bertscore_model",
        cfg.bertscore_model,
        "--compute_perplexity",
        str(cfg.compute_perplexity),
        "--fluency_model",
        cfg.fluency_model,
        "--log_to_wandb",
        str(cfg.log_to_wandb),
        "--wandb_project",
        cfg.wandb_project,
    ]

    if cfg.overwrite_scores:
      score_cmd.extend(["--overwrite_scores", "True"])
    if cfg.scores_checkpoint_path:
      score_cmd.extend(["--scores_checkpoint_path", cfg.scores_checkpoint_path])
    if os.environ.get("GEMINI_API_KEY"):
      score_cmd.extend(["--gemini_api_key", os.environ["GEMINI_API_KEY"]])

    captured_metrics = self._run_subprocess(score_cmd, live_line_callback)

    # Load metrics from saved summary JSON if produced by GenerationMetricsEvaluator
    expected_dataset_repo = build_eval_dataset_repo_id(
        user=self.config.user,
        writer_model_lora=model_to_eval,
        temperature=cfg.temperature,
        writer_num_fewshot=cfg.writer_num_fewshot,
        task_name=self.config.task_name,
    )
    expected_dataset_name = expected_dataset_repo.split("/")[-1]
    summary_file = os.path.join("logs", "eval", f"{expected_dataset_name}_summary.json")
    if os.path.isfile(summary_file):
      try:
        with open(summary_file, "r", encoding="utf-8") as f:
          file_metrics = json.load(f)
        if isinstance(file_metrics, dict):
          captured_metrics.update(file_metrics)
      except Exception as e:
        logger.warning("Could not load eval summary JSON %s: %s", summary_file, e)

    return StageResult(
        status=StageStatus.COMPLETED,
        model_repo_id=model_to_eval,
        metrics=captured_metrics,
    )

  def _run_subprocess(
      self,
      cmd: list[str],
      live_line_callback: Optional[Callable[[str], None]],
  ) -> Dict[str, Any]:
    """Runs an evaluator subprocess and captures logged metrics."""
    logger.info("Executing evaluation command: %s", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    env.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    metrics = {}
    while process.poll() is None:
      line = process.stdout.readline() if process.stdout else ""
      if line:
        clean = line.rstrip()
        if live_line_callback:
          live_line_callback(clean)
        # Parse output metrics lines if printed
        if "eval/" in clean or "metric:" in clean or "Score:" in clean:
          parts = clean.split(":")
          if len(parts) == 2:
            k, v = parts[0].strip(), parts[1].strip()
            try:
              metrics[k] = float(v)
            except ValueError:
              pass
      else:
        time.sleep(0.1)

    # Drain any remaining output lines
    if process.stdout:
      for remaining in process.stdout.readlines():
        clean = remaining.rstrip()
        if clean:
          if live_line_callback:
            live_line_callback(clean)
          if "eval/" in clean or "metric:" in clean or "Score:" in clean:
            parts = clean.split(":")
            if len(parts) == 2:
              k, v = parts[0].strip(), parts[1].strip()
              try:
                metrics[k] = float(v)
              except ValueError:
                pass

    if process.returncode != 0:
      raise RuntimeError(
          f"Evaluation command failed with exit code {process.returncode}"
      )

    return metrics
