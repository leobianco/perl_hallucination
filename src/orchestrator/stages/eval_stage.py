"""Evaluation and autorating stage (generation + Gemini scoring).

The stage evaluates the trained policies produced by the campaign - the best
SFT checkpoint and the best PE-RL checkpoint - so the report can show what the
RL step actually bought. The raw base model is deliberately *not* evaluated:
it is not a policy this campaign produced, and generating + autorating it
would cost a third of the eval budget for a number that never changes.

Each target goes through the same two phases as ``scripts/evaluator.sh``:
generation of test completions, then Gemini autorating plus BertScore and
perplexity.

A target is a (policy, temperature) pair rather than just a policy. PE-RL
searches its rollout temperature, and a policy is scored at the temperature it
was trained to sample at; the SFT baseline is additionally re-scored at each
of those temperatures so that no delta ever spans two decoding regimes. See
``EvalStageConfig.match_perl_rollout_temperature``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.orchestrator.process import stream_subprocess
from src.orchestrator.retry import run_with_retries
from src.orchestrator import eval_metrics
from src.orchestrator import flavors
from src.orchestrator.stages.base import BaseStage
from src.orchestrator.state import StageResult, StageStatus
from src.utils import build_eval_dataset_repo_id

logger = logging.getLogger(__name__)

#: Evaluated policies, in report order. ``sft`` is the RL baseline. A campaign
#: that branches over reward-model dataset flavors replaces the bare ``perl``
#: entry with one ``perl:<flavor>`` target per branch.
TARGET_LABELS: Tuple[str, ...] = ("sft", "perl")

#: Bookkeeping entries that are identical by construction; a delta on them is
#: pure noise in the report.
NO_DELTA = eval_metrics.NO_DELTA

#: Metrics where a *lower* value is better; used to sign the SFT->PE-RL delta.
LOWER_IS_BETTER = eval_metrics.LOWER_IS_BETTER


@dataclasses.dataclass(frozen=True)
class EvalTarget:
  """One generation-and-scoring pass: a policy at a decoding temperature.

  Attributes:
    label: Metric namespace and display key, e.g. ``sft``, ``sft@t0.6`` or
      ``perl:organic``.
    model_repo_id: The LoRA adapter to serve.
    temperature: Sampling temperature for generation. Part of the completions
      repo name, so two targets differing only here keep separate datasets and
      neither has to be regenerated when the other is.
  """

  label: str
  model_repo_id: str
  temperature: float


class EvalStage(BaseStage):
  """Orchestrates test completion generation and multi-metric autorating."""

  @property
  def kind(self) -> str:
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
    primary_model = targets[-1].model_repo_id

    logger.info(
        "=== Starting Evaluation Stage for task: %s ===", self.config.task_name
    )
    if live_line_callback:
      live_line_callback(
          f"=== Starting Evaluation Stage for task: {self.config.task_name} ==="
      )
      for target in targets:
        # No brackets: the console strips bracketed spans as style tags,
        # which used to swallow the "[sft]" label entirely.
        live_line_callback(
            f"  Target {target.label}: {target.model_repo_id} "
            f"at temperature {target.temperature}"
        )
      live_line_callback(f"  {self.threshold_provenance()}")
      for line in self.temperature_notes(targets):
        live_line_callback(f"  {line}")

    if self.config.dry_run:
      return self._dry_run_result(targets, primary_model, live_line_callback)

    # Metrics already recorded by an earlier attempt: autorating is billed per
    # call, so an interrupted campaign must not pay for the same model twice.
    metrics: Dict[str, Any] = dict(self.previous_metrics())

    for index, target in enumerate(targets, start=1):
      if self.is_target_scored(target.label, metrics):
        message = (
            f"[RESUME] {target.label.upper()} ({target.model_repo_id}) was "
            "already scored; keeping the recorded metrics."
        )
        logger.info(message)
        if live_line_callback:
          live_line_callback(message)
        continue

      target_metrics = self._evaluate_target(
          target=target,
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

  def target_temperature(self, flavor: Optional[str]) -> float:
    """Returns the temperature one PE-RL branch should be scored at.

    Args:
      flavor: The branch's reward-model dataset flavor, or None.

    Returns:
      The branch's recorded rollout temperature when matching is enabled and
      the value was recorded; otherwise the configured evaluation
      temperature.
    """
    cfg = self.config.eval
    if not cfg.match_perl_rollout_temperature:
      return float(cfg.temperature)
    recorded = self.context.perl_rollout_temperature_for(flavor)
    return float(cfg.temperature) if recorded is None else recorded

  def resolve_targets(self) -> List[EvalTarget]:
    """Lists the passes to run, in report order.

    The label doubles as the metric namespace. For a campaign that branched
    over reward-model dataset flavors there is one PE-RL policy per branch,
    each namespaced by its branch id (``perl:synthetic_struct``) so the report
    can put them side by side.

    Each PE-RL branch is scored at its own rollout temperature. Whenever that
    differs from ``eval.temperature`` the SFT baseline is scheduled a second
    time at the same temperature, tagged ``sft@t0.6``, so that the branch's
    delta is measured against a baseline decoded identically and reflects the
    weights alone. Distinct temperatures are de-duplicated: two branches that
    happened to win at the same temperature share one extra baseline.

    Returns:
      One entry per (policy, temperature) pass. Baselines come first, so the
      last entry remains the primary model - the last PE-RL branch when any
      exists, otherwise SFT.

    Raises:
      ValueError: When the campaign produced no policy to evaluate. Falling
        back to the base model would silently report base-model numbers as if
        they were the campaign's results.
    """
    cfg = self.config.eval
    base_temperature = float(cfg.temperature)
    sft_repo_id = self.context.sft_model_repo_id

    baselines: List[EvalTarget] = []
    if sft_repo_id:
      baselines.append(
          EvalTarget(
              label=eval_metrics.BASELINE_LABEL,
              model_repo_id=sft_repo_id,
              temperature=base_temperature,
          )
      )

    # Branches come from the flavor list, not from ``config.stages``: a
    # `--stages eval` rerun over a finished campaign has no 'perl' entry
    # there but still has the policies to score.
    policies: List[EvalTarget] = []
    seen = {sft_repo_id} if sft_repo_id else set()
    for flavor in flavors.campaign_flavors(self.config):
      repo_id = self.context.perl_model_repo_id_for(flavor)
      if not repo_id or repo_id in seen:
        continue
      seen.add(repo_id)
      policies.append(
          EvalTarget(
              label=flavors.stage_id_for_flavor(self.config, "perl", flavor),
              model_repo_id=repo_id,
              temperature=self.target_temperature(flavor),
          )
      )

    # One matched baseline per distinct policy temperature. Keyed on the
    # formatted token rather than the float so that this agrees exactly with
    # the pairing `eval_metrics.baseline_label_for` will do later.
    if sft_repo_id:
      already = {eval_metrics.format_temperature(base_temperature)}
      for policy in policies:
        token = eval_metrics.format_temperature(policy.temperature)
        if token in already:
          continue
        already.add(token)
        baselines.append(
            EvalTarget(
                label=eval_metrics.make_target_label(
                    eval_metrics.BASELINE_LABEL, policy.temperature
                ),
                model_repo_id=sft_repo_id,
                temperature=policy.temperature,
            )
        )

    targets = baselines + policies
    if not targets:
      raise ValueError(
          "Evaluation has no trained policy to score: neither an SFT nor a "
          "PE-RL checkpoint is available. Run the 'sft'/'perl' stages first, "
          "or point perl.sft_model_path / perl.reward_model_path at existing "
          "checkpoints. The base model is intentionally not evaluated."
      )
    return targets

  def temperature_notes(self, targets: List[EvalTarget]) -> List[str]:
    """Explains the decoding temperatures in play, for the log.

    Args:
      targets: The resolved passes.

    Returns:
      Zero or more lines. Empty when every target shares one temperature,
      which is the case for a campaign with matching switched off and for one
      whose policies happened to train at ``eval.temperature``.
    """
    cfg = self.config.eval
    if not cfg.match_perl_rollout_temperature:
      return [
          "Temperature matching is off: every target is scored at "
          f"{cfg.temperature}."
      ]
    policies = [t for t in targets if not eval_metrics.is_baseline(t.label)]
    unmatched = [
        t.label
        for t in policies
        if self.context.perl_rollout_temperature_for(
            flavors.split_stage_id(t.label)[1]
        )
        is None
    ]
    notes = []
    extra = [
        t
        for t in targets
        if eval_metrics.is_baseline(t.label)
        and t.label != eval_metrics.BASELINE_LABEL
    ]
    if extra:
      notes.append(
          "Scoring the SFT baseline again at "
          + ", ".join(str(t.temperature) for t in extra)
          + " so each PE-RL delta compares like with like."
      )
    if unmatched:
      notes.append(
          "No rollout temperature was recorded for "
          + ", ".join(unmatched)
          + f"; falling back to {cfg.temperature}. The sweep pinned "
          "temperature outside its parameters block, so it was never logged "
          "per trial."
      )
    return notes

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
      target: EvalTarget,
      position: int,
      total: int,
      live_line_callback: Optional[Callable[[str], None]],
      stop_requested_callback: Optional[Callable[[], bool]],
  ) -> Dict[str, Any]:
    """Generates and scores completions for a single policy.

    Args:
      target: The policy and the temperature to sample it at.
      position: 1-based index of this target, for progress messages.
      total: Total number of targets.
      live_line_callback: Log sink.
      stop_requested_callback: Predicate used to cut the evaluation short.

    Returns:
      Metrics for this target, prefixed with ``{target.label}/`` and
      including the temperature the completions were drawn at.
    """
    cfg = self.config.eval
    label = target.label
    model_repo_id = target.model_repo_id
    prefix = f"[{label.upper()} {position}/{total}]"

    if live_line_callback:
      live_line_callback(
          f"{prefix} Step 1/2: generating test completions for "
          f"{model_repo_id} at temperature {target.temperature}"
      )

    gen_cmd = self._generation_command(model_repo_id, target.temperature)
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

    score_cmd = self._scoring_command(model_repo_id, target.temperature)
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

    summary = self._load_summary(model_repo_id, target.temperature)
    captured.update(summary)
    # Recorded per target, not per campaign: with matching on, different rows
    # of the same table were sampled differently, and the report has to be
    # able to say so without the config in hand.
    captured[eval_metrics.DECODING_TEMPERATURE_KEY] = target.temperature
    return {f"{label}/{key}": value for key, value in captured.items()}

  def _summary_path(
      self, model_repo_id: str, temperature: Optional[float] = None
  ) -> Tuple[str, str]:
    """Returns the (completions repo id, summary json path) for a target.

    Args:
      model_repo_id: The adapter being evaluated.
      temperature: The temperature its completions were sampled at. Defaults
        to ``eval.temperature``. Part of the repo name, so the SFT baseline's
        greedy and matched runs resolve to different datasets instead of
        overwriting one another.

    Returns:
      The completions repo id and the path of the summary JSON.
    """
    cfg = self.config.eval
    if temperature is None:
      temperature = cfg.temperature
    repo = build_eval_dataset_repo_id(
        user=self.config.user,
        writer_model_lora=model_repo_id,
        temperature=temperature,
        writer_num_fewshot=cfg.writer_num_fewshot,
        task_name=self.config.task_name,
        # Must mirror `_generation_command`: the completions repo name encodes
        # whether the SFT adapter was stacked, so passing a different value
        # here would send the stage looking for another run's dataset.
        sft_model_path=self._stacked_sft_repo_id(model_repo_id),
        seed=cfg.seed,
        max_tokens=cfg.max_tokens,
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

  def _load_summary(
      self, model_repo_id: str, temperature: float
  ) -> Dict[str, Any]:
    """Loads the metrics JSON written by the scoring pipeline.

    Args:
      model_repo_id: The evaluated adapter.
      temperature: The temperature its completions were sampled at.

    Returns:
      The parsed metrics dictionary.

    Raises:
      RuntimeError: If the summary is missing or unreadable. Reporting an
        empty metric table as a success is the one outcome an overnight run
        cannot afford.
    """
    repo, summary_file = self._summary_path(model_repo_id, temperature)
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

  def _generation_command(
      self, model_repo_id: str, temperature: Optional[float] = None
  ) -> List[str]:
    """Builds the ``--mode generate`` invocation for one policy.

    Args:
      model_repo_id: The adapter to serve.
      temperature: Sampling temperature. Defaults to ``eval.temperature``.
        Part of the completions repo name, so ``_scoring_command`` and
        ``_summary_path`` must be given the same value.

    Returns:
      The argument vector.
    """
    cfg = self.config.eval
    if temperature is None:
      temperature = cfg.temperature
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
        str(temperature),
        "--top_p",
        str(cfg.top_p),
        "--top_k",
        str(cfg.top_k),
        "--writer_num_fewshot",
        str(cfg.writer_num_fewshot),
    ])
    return cmd

  def _scoring_command(
      self, model_repo_id: str, temperature: Optional[float] = None
  ) -> List[str]:
    """Builds the ``--mode score`` invocation for one policy.

    Args:
      model_repo_id: The adapter to score.
      temperature: The temperature its completions were sampled at. Defaults
        to ``eval.temperature``; it is part of the completions repo name, so
        it must match what generation used or scoring reads another run's
        dataset.

    Returns:
      The argument vector.
    """
    cfg = self.config.eval
    if temperature is None:
      temperature = cfg.temperature
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
        # Part of the completions repo name, so scoring must be told the same
        # value generation used; the pipeline default (128) is not it.
        "--max_tokens",
        str(cfg.max_tokens),
        "--temperature",
        str(temperature),
        "--writer_num_fewshot",
        str(cfg.writer_num_fewshot),
        "--run_autorater",
        "True",
        "--autorater_num_samples",
        str(cfg.autorater_num_samples),
        "--evaluator_model",
        cfg.evaluator_model,
        "--use_gemini",
        str(cfg.use_gemini),
        "--evaluator_num_fewshot",
        str(cfg.evaluator_num_fewshot),
        "--evaluate_evaluator",
        "False",
        "--threshold",
        str(self.scoring_threshold()),
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

  def scoring_threshold(self) -> float:
    """Returns the score above which a completion counts as hallucinated.

    Prefers the threshold the ``autorater`` stage fitted on the labelled set
    with this exact judge and few-shot count. ``eval.threshold`` is the
    fallback for campaigns that do not run calibration, and is a constant
    measured once, elsewhere, on a configuration nobody recorded.

    Returns:
      The threshold the scoring pass will use.
    """
    fitted = self.context.calibrated_threshold
    if fitted is None:
      return self.config.eval.threshold
    return fitted

  def threshold_provenance(self) -> str:
    """Returns a one-line account of where the threshold came from."""
    fitted = self.context.calibrated_threshold
    if fitted is None:
      return (
          f"Decision threshold {self.config.eval.threshold} taken from the "
          "configuration; no autorater calibration was run."
      )
    return (
        f"Decision threshold {fitted!r} fitted by the autorater "
        "calibration stage."
    )

  # --- Plumbing ---------------------------------------------------------

  def _dry_run_result(
      self,
      targets: List[EvalTarget],
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
    for target in targets:
      if live_line_callback:
        live_line_callback(
            f"[DRY-RUN] Generating completions for {target.model_repo_id} "
            f"at temperature {target.temperature}..."
        )
        live_line_callback(
            f"[DRY-RUN] Scoring {target.label} with Gemini autorater..."
        )
      time.sleep(0.4)
      # Labels are 'perl:synthetic_struct' or 'sft@t0.6'; the mock is per
      # *kind*, so both branches and matched baselines resolve to a number.
      kind = eval_metrics.target_kind(target.label)
      values = dict(mock_per_target.get(kind, mock_per_target["sft"]))
      values["num_samples"] = self.config.eval.max_eval_samples
      # Sampling costs faithfulness, so a rehearsal that ignored temperature
      # would show a matched baseline scoring identically to the greedy one
      # and make the extra pass look pointless.
      penalty = 0.02 * float(target.temperature)
      if "hallucination_rate" in values:
        values["hallucination_rate"] = round(
            values["hallucination_rate"] + penalty, 4
        )
      if "faithfulness_rate" in values:
        values["faithfulness_rate"] = round(
            values["faithfulness_rate"] - penalty, 4
        )
      values[eval_metrics.DECODING_TEMPERATURE_KEY] = target.temperature
      metrics.update({f"{target.label}/{k}": v for k, v in values.items()})
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
  """Adds ``delta/<metric>`` entries comparing each PE-RL policy against SFT.

  The delta is signed so that a *positive* value always means PE-RL improved
  on SFT, whatever the direction of the underlying metric.

  A campaign that branched over reward-model dataset flavors has one PE-RL
  policy per branch. Each gets its own namespace - ``delta:organic/``,
  ``delta:synthetic_struct/`` - mirroring the ``perl:<flavor>/`` keys it was
  derived from. A single-flavor campaign keeps the plain ``delta/``.

  Each policy is compared against the SFT row sampled at its *own* decoding
  temperature, so the difference is the weights rather than the decode. With
  temperature matching off, or on metrics from an older run, that resolves to
  the one and only ``sft`` row and the behaviour is unchanged.

  Args:
    metrics: Metric map with ``sft/`` and ``perl/``-ish prefixed keys.

  Returns:
    The additional delta entries (empty when either side is missing).
  """
  deltas: Dict[str, Any] = {}
  policies = [
      label
      for label in eval_metrics.target_labels(metrics)
      if not eval_metrics.is_baseline(label)
  ]
  for label in policies:
    prefix = f"{label}/"
    namespace = eval_metrics.delta_label(label)
    baseline = eval_metrics.baseline_label_for(metrics, label)
    if baseline is None:
      continue
    for key, perl_value in metrics.items():
      if not str(key).startswith(prefix):
        continue
      bare = str(key)[len(prefix):]
      if bare in NO_DELTA:
        continue
      sft_value = metrics.get(f"{baseline}/{bare}")
      if not isinstance(perl_value, (int, float)) or isinstance(
          perl_value, bool
      ):
        continue
      if not isinstance(sft_value, (int, float)) or isinstance(
          sft_value, bool
      ):
        continue
      improvement = float(perl_value) - float(sft_value)
      if bare in LOWER_IS_BETTER:
        improvement = -improvement
      deltas[f"{namespace}/{bare}"] = improvement
  return deltas
