"""Evaluation and autorating stage (generation + Gemini scoring).

The stage evaluates the trained policies produced by the campaign - the best
SFT checkpoint and the best PE-RL checkpoint - so the report can show what the
RL step actually bought. The raw base model is deliberately *not* evaluated:
it is not a policy this campaign produced, and generating + autorating it
would cost a third of the eval budget for a number that never changes.

Each target goes through the same two phases as ``scripts/evaluator.sh``:
generation of test completions, then Gemini autorating plus BertScore and
perplexity.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.orchestrator.process import stream_subprocess
from src.orchestrator.retry import run_with_retries
from src.orchestrator.stages.base import BaseStage
from src.orchestrator.state import StageResult, StageStatus
from src.utils import build_eval_dataset_repo_id

logger = logging.getLogger(__name__)

#: Evaluated policies, in report order. ``sft`` is the RL baseline.
TARGET_LABELS: Tuple[str, ...] = ("sft", "perl")

#: Bookkeeping entries that are identical by construction; a delta on them is
#: pure noise in the report.
NO_DELTA = frozenset({"num_samples", "num_examples", "seed"})

#: Metrics where a *lower* value is better; used to sign the SFT->PE-RL delta.
LOWER_IS_BETTER = frozenset({
    "hallucination_rate",
    "perplexity",
    "eval_loss",
    "loss",
})


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

    targets = self.resolve_targets()
    primary_model = targets[-1][1]

    logger.info(
        "=== Starting Evaluation Stage for task: %s ===", self.config.task_name
    )
    if live_line_callback:
      live_line_callback(
          f"=== Starting Evaluation Stage for task: {self.config.task_name} ==="
      )
      for label, repo_id in targets:
        live_line_callback(f"  Target [{label}]: {repo_id}")

    if self.config.dry_run:
      return self._dry_run_result(targets, primary_model, live_line_callback)

    # Metrics already recorded by an earlier attempt: autorating is billed per
    # call, so an interrupted campaign must not pay for the same model twice.
    metrics: Dict[str, Any] = dict(self.previous_metrics())

    for index, (label, repo_id) in enumerate(targets, start=1):
      if self.is_target_scored(label, metrics):
        message = (
            f"[RESUME] {label.upper()} ({repo_id}) was already scored; "
            "keeping the recorded metrics."
        )
        logger.info(message)
        if live_line_callback:
          live_line_callback(message)
        continue

      target_metrics = self._evaluate_target(
          label=label,
          model_repo_id=repo_id,
          position=index,
          total=len(targets),
          live_line_callback=live_line_callback,
          stop_requested_callback=stop_requested_callback,
      )
      metrics.update(target_metrics)
      # Checkpoint after every model: the next one may take hours.
      self.record_partial_metrics(target_metrics)

    metrics.update(compute_deltas(metrics))

    return StageResult(
        status=StageStatus.COMPLETED,
        model_repo_id=primary_model,
        metrics=metrics,
    )

  # --- Target resolution ----------------------------------------------

  def resolve_targets(self) -> List[Tuple[str, str]]:
    """Lists the (label, repo id) pairs to evaluate, in report order.

    Returns:
      One entry per available trained policy. The last entry is the primary
      model (PE-RL when it exists, otherwise SFT).

    Raises:
      ValueError: When the campaign produced no policy to evaluate. Falling
        back to the base model would silently report base-model numbers as if
        they were the campaign's results.
    """
    candidates = {
        "sft": self.context.sft_model_repo_id,
        "perl": self.context.perl_model_repo_id,
    }
    targets: List[Tuple[str, str]] = []
    seen = set()
    for label in TARGET_LABELS:
      repo_id = candidates.get(label)
      if not repo_id or repo_id in seen:
        continue
      seen.add(repo_id)
      targets.append((label, repo_id))

    if not targets:
      raise ValueError(
          "Evaluation has no trained policy to score: neither an SFT nor a "
          "PE-RL checkpoint is available. Run the 'sft'/'perl' stages first, "
          "or point perl.sft_model_path / perl.reward_model_path at existing "
          "checkpoints. The base model is intentionally not evaluated."
      )
    return targets

  def previous_metrics(self) -> Dict[str, Any]:
    """Returns metrics persisted by an earlier attempt of this stage."""
    previous = self.state.stages.get(self.name)
    if previous is None or previous.status == StageStatus.COMPLETED:
      return {}
    return dict(previous.metrics or {})

  def is_target_scored(self, label: str, metrics: Dict[str, Any]) -> bool:
    """True when ``metrics`` already holds results for ``label``."""
    prefix = f"{label}/"
    return any(str(key).startswith(prefix) for key in metrics)

  def record_partial_metrics(self, metrics: Dict[str, Any]) -> None:
    """Persists per-target metrics immediately so a crash stays resumable."""
    self.state.merge_stage_metrics(self.name, metrics)
    if self.context.state_path:
      self.state.save(self.context.state_path)

  # --- Evaluation -------------------------------------------------------

  def _evaluate_target(
      self,
      label: str,
      model_repo_id: str,
      position: int,
      total: int,
      live_line_callback: Optional[Callable[[str], None]],
      stop_requested_callback: Optional[Callable[[], bool]],
  ) -> Dict[str, Any]:
    """Generates and scores completions for a single policy.

    Args:
      label: Short target name ('sft' or 'perl'), used as the metric prefix.
      model_repo_id: LoRA adapter repo to evaluate.
      position: 1-based index of this target, for progress messages.
      total: Total number of targets.
      live_line_callback: Log sink.
      stop_requested_callback: Predicate used to cut the evaluation short.

    Returns:
      Metrics for this target, prefixed with ``{label}/``.
    """
    cfg = self.config.eval
    prefix = f"[{label.upper()} {position}/{total}]"

    if live_line_callback:
      live_line_callback(
          f"{prefix} Step 1/2: generating test completions for {model_repo_id}"
      )

    gen_cmd = self._generation_command(model_repo_id)
    run_with_retries(
        lambda: self._run_subprocess(
            gen_cmd, live_line_callback, stop_requested_callback
        ),
        description=f"{label} completion generation",
        attempts=self.config.robustness.eval_attempts,
        base_delay_s=self.config.robustness.retry_base_delay_s,
        on_notice=live_line_callback,
        stop_requested=stop_requested_callback,
    )

    if live_line_callback:
      live_line_callback(
          f"{prefix} Step 2/2: scoring with {cfg.evaluator_model} "
          "+ BertScore + perplexity"
      )

    score_cmd = self._scoring_command(model_repo_id)
    captured = run_with_retries(
        lambda: self._run_subprocess(
            score_cmd, live_line_callback, stop_requested_callback
        ),
        description=f"{label} autorating",
        attempts=self.config.robustness.eval_attempts,
        base_delay_s=self.config.robustness.retry_base_delay_s,
        on_notice=live_line_callback,
        stop_requested=stop_requested_callback,
    )

    summary = self._load_summary(model_repo_id)
    captured.update(summary)
    return {f"{label}/{key}": value for key, value in captured.items()}

  def _summary_path(self, model_repo_id: str) -> Tuple[str, str]:
    """Returns the (completions repo id, summary json path) for a target."""
    cfg = self.config.eval
    repo = build_eval_dataset_repo_id(
        user=self.config.user,
        writer_model_lora=model_repo_id,
        temperature=cfg.temperature,
        writer_num_fewshot=cfg.writer_num_fewshot,
        task_name=self.config.task_name,
        # Must mirror `_generation_command`: the completions repo name encodes
        # whether the SFT adapter was stacked, so passing a different value
        # here would send the stage looking for another run's dataset.
        sft_model_path=self._stacked_sft_repo_id(model_repo_id),
    )
    name = repo.split("/")[-1]
    return repo, os.path.join("logs", "eval", f"{name}_summary.json")

  def _stacked_sft_repo_id(self, model_repo_id: str) -> Optional[str]:
    """Returns the SFT adapter stacked under ``model_repo_id``, if any.

    Evaluating the SFT checkpoint itself stacks nothing: it is already the
    single adapter being served.

    Args:
      model_repo_id: The adapter being evaluated.

    Returns:
      The SFT repo ID, or None when nothing is stacked.
    """
    sft_repo = self.context.sft_model_repo_id
    if sft_repo and model_repo_id != sft_repo:
      return sft_repo
    return None

  def _load_summary(self, model_repo_id: str) -> Dict[str, Any]:
    """Loads the metrics JSON written by the scoring pipeline.

    Args:
      model_repo_id: The evaluated adapter.

    Returns:
      The parsed metrics dictionary.

    Raises:
      RuntimeError: If the summary is missing or unreadable. Reporting an
        empty metric table as a success is the one outcome an overnight run
        cannot afford.
    """
    repo, summary_file = self._summary_path(model_repo_id)
    if not os.path.isfile(summary_file):
      raise RuntimeError(
          "Scoring finished but no evaluation summary was written to "
          f"{summary_file}. The completions dataset ({repo}) or the autorater "
          "step likely failed; inspect the scoring logs above."
      )
    try:
      with open(summary_file, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    except (OSError, ValueError) as error:
      raise RuntimeError(
          f"Could not read the evaluation summary {summary_file}: {error}"
      ) from error
    if not isinstance(loaded, dict):
      raise RuntimeError(
          f"Evaluation summary {summary_file} is not a JSON object."
      )
    return loaded

  # --- Command construction ---------------------------------------------

  def _generation_command(self, model_repo_id: str) -> List[str]:
    """Builds the ``--mode generate`` invocation for one policy."""
    cfg = self.config.eval
    cmd = [
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
        str(cfg.seed),
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
        model_repo_id,
    ]
    sft_repo = self._stacked_sft_repo_id(model_repo_id)
    if sft_repo:
      cmd.extend(["--sft_model_path", sft_repo])
    cmd.extend([
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
    ])
    return cmd

  def _scoring_command(self, model_repo_id: str) -> List[str]:
    """Builds the ``--mode score`` invocation for one policy."""
    cfg = self.config.eval
    cmd = [
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
        str(cfg.seed),
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
        model_repo_id,
    ]
    # Scoring derives the completions repo from the same inputs as generation,
    # SFT stacking included; without this flag it would load the '_nosft'
    # dataset of the same checkpoint.
    sft_repo = self._stacked_sft_repo_id(model_repo_id)
    if sft_repo:
      cmd.extend(["--sft_model_path", sft_repo])
    cmd.extend([
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
        "--evaluate_evaluator",
        "False",
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
    ])
    if cfg.overwrite_scores:
      cmd.extend(["--overwrite_scores", "True"])
    if cfg.scores_checkpoint_path:
      cmd.extend(["--scores_checkpoint_path", cfg.scores_checkpoint_path])
    if os.environ.get("GEMINI_API_KEY"):
      cmd.extend(["--gemini_api_key", os.environ["GEMINI_API_KEY"]])
    return cmd

  # --- Plumbing ---------------------------------------------------------

  def _dry_run_result(
      self,
      targets: List[Tuple[str, str]],
      primary_model: str,
      live_line_callback: Optional[Callable[[str], None]],
  ) -> StageResult:
    """Produces plausible per-target metrics without touching the GPU."""
    logger.info("[DRY-RUN] Simulating generation and scoring...")
    mock_per_target = {
        "sft": {
            "hallucination_rate": 0.091,
            "faithfulness_rate": 0.909,
            "bertscore_f1": 0.874,
            "perplexity": 8.41,
        },
        "perl": {
            "hallucination_rate": 0.048,
            "faithfulness_rate": 0.952,
            "bertscore_f1": 0.892,
            "perplexity": 7.84,
        },
    }
    metrics: Dict[str, Any] = {}
    for label, repo_id in targets:
      if live_line_callback:
        live_line_callback(f"[DRY-RUN] Generating completions for {repo_id}...")
        live_line_callback(f"[DRY-RUN] Scoring {label} with Gemini autorater...")
      time.sleep(0.4)
      values = dict(mock_per_target.get(label, mock_per_target["sft"]))
      values["num_samples"] = self.config.eval.max_eval_samples
      metrics.update({f"{label}/{k}": v for k, v in values.items()})
    metrics.update(compute_deltas(metrics))
    return StageResult(
        status=StageStatus.COMPLETED,
        model_repo_id=primary_model,
        metrics=metrics,
    )

  def _run_subprocess(
      self,
      cmd: List[str],
      live_line_callback: Optional[Callable[[str], None]],
      stop_requested_callback: Optional[Callable[[], bool]] = None,
  ) -> Dict[str, Any]:
    """Runs an evaluator subprocess and captures metrics printed on stdout.

    Args:
      cmd: Argument vector of the evaluator invocation.
      live_line_callback: Log sink for the streamed output.
      stop_requested_callback: Predicate used to cut the evaluation short.

    Returns:
      Metrics scraped from the process output (the authoritative values come
      from the summary JSON written by the evaluator).

    Raises:
      RuntimeError: If the evaluator is interrupted or exits non-zero.
    """
    env = os.environ.copy()
    env.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    env.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

    metrics: Dict[str, Any] = {}

    def _handle(line: str) -> None:
      if live_line_callback:
        live_line_callback(line)
      if "eval/" in line or "metric:" in line or "Score:" in line:
        parts = line.split(":")
        if len(parts) == 2:
          key, value = parts[0].strip(), parts[1].strip()
          try:
            metrics[key] = float(value)
          except ValueError:
            pass

    outcome = stream_subprocess(
        cmd,
        on_line=_handle,
        stop_requested=stop_requested_callback,
        env=env,
        timeout_s=self.config.eval.timeout_minutes * 60,
        stall_warning_s=self.config.robustness.stall_warning_minutes * 60,
    )

    if outcome.interrupted:
      raise RuntimeError("Evaluation was interrupted before it completed.")
    if outcome.timed_out:
      raise RuntimeError(
          "Evaluation exceeded its "
          f"{self.config.eval.timeout_minutes} minute budget."
      )
    if outcome.returncode != 0:
      raise RuntimeError(
          f"Evaluation command failed with exit code {outcome.returncode}"
      )

    return metrics


def compute_deltas(metrics: Dict[str, Any]) -> Dict[str, Any]:
  """Adds ``delta/<metric>`` entries comparing PE-RL against SFT.

  The delta is signed so that a *positive* value always means PE-RL improved
  on SFT, whatever the direction of the underlying metric.

  Args:
    metrics: Metric map with ``sft/`` and ``perl/`` prefixed keys.

  Returns:
    The additional delta entries (empty when either side is missing).
  """
  deltas: Dict[str, Any] = {}
  for key, perl_value in metrics.items():
    if not str(key).startswith("perl/"):
      continue
    bare = str(key)[len("perl/"):]
    if bare in NO_DELTA:
      continue
    sft_value = metrics.get(f"sft/{bare}")
    if not isinstance(perl_value, (int, float)) or isinstance(perl_value, bool):
      continue
    if not isinstance(sft_value, (int, float)) or isinstance(sft_value, bool):
      continue
    improvement = float(perl_value) - float(sft_value)
    if bare in LOWER_IS_BETTER:
      improvement = -improvement
    deltas[f"delta/{bare}"] = improvement
  return deltas
