"""Abstract base class and shared context for all campaign stages."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from src.orchestrator import naming
from src.orchestrator.config import CampaignConfig, SweepStageConfig
from src.orchestrator.model_manager import ModelManager
from src.orchestrator.state import CampaignState, StageResult, StageStatus
from src.orchestrator.sweep_controller import AgentRun, RunScore, SweepController


logger = logging.getLogger(__name__)


@dataclass
class CampaignContext:
  """Shared runtime execution context passed across stages."""

  config: CampaignConfig
  state: CampaignState
  sweep_controller: SweepController
  model_manager: ModelManager
  #: Where the engine persists ``state``; lets stages checkpoint a sweep id
  #: as soon as it exists, so a crash mid-sweep is resumable.
  state_path: Optional[str] = None
  #: Predicate that is True only for a *hard* abort (``[x]`` / Ctrl-C twice).
  #: Stages use it for the post-sweep materialization, which must survive the
  #: ``[a]`` advance and ``[s]`` stop intents that cut the sweep agent short.
  abort_requested_callback: Optional[Callable[[], bool]] = None

  @property
  def sft_model_repo_id(self) -> Optional[str]:
    """Resolves the SFT model repo ID from state or configuration overrides."""
    if self.config.perl.sft_model_path and self.config.perl.sft_model_path != "auto":
      return self.config.perl.sft_model_path
    return self.state.get_model_repo_id("sft")

  @property
  def reward_model_repo_id(self) -> Optional[str]:
    """Resolves the Reward Model repo ID from state or configuration overrides."""
    if (
        self.config.perl.reward_model_path
        and self.config.perl.reward_model_path != "auto"
    ):
      return self.config.perl.reward_model_path
    return self.state.get_model_repo_id("rm")

  @property
  def perl_model_repo_id(self) -> Optional[str]:
    """Resolves the final PE-RL model repo ID from state."""
    return self.state.get_model_repo_id("perl")


@dataclass(frozen=True)
class SweepExecution:
  """What a sweep actually delivered, as opposed to what it was asked for.

  A sweep can end for several reasons that are not failures - the wall-clock
  budget expires, the user advances with the best run so far, the machine
  running the agent dies and the agent process with it. In all of those the
  stage carries on: some trials did finish, so there *is* a best
  configuration to retrain and publish, and refusing to use it would throw
  away hours of GPU time.

  What must not happen is the campaign then reporting the result as though
  the whole search had run. Ranking 3 configurations is not ranking 5, and
  the difference has to survive into the state file, the dashboard and the
  report.

  Attributes:
    trials_done: Trials that finished successfully, as counted by W&B after
      the agent returned. This is the authoritative number; the live counter
      is parsed from agent stdout and counts crashed trials too.
    trials_total: The trial budget the stage was configured with.
    outcome: ``complete``, ``partial`` or ``unknown``.
    warnings: Human-readable notes explaining a non-``complete`` outcome.
  """

  trials_done: int
  trials_total: int
  outcome: str = "complete"
  warnings: List[str] = field(default_factory=list)

  @property
  def is_partial(self) -> bool:
    """True when fewer trials finished than the stage asked for."""
    return self.outcome == "partial"

  def stage_result_fields(
      self, extra_warnings: Sequence[str] = ()
  ) -> Dict[str, Any]:
    """Returns the ``StageResult`` fields recording this accounting.

    Args:
      extra_warnings: Further degradations discovered after the trials ran,
        typically from winner selection. They are appended rather than
        merged by the caller because both sets land in the same
        ``StageResult.warnings`` list, and a plain ``**`` splat of two dicts
        would silently drop one of them.

    Returns:
      The fields to splat into a ``StageResult``.
    """
    return {
        "trials_done": self.trials_done,
        "trials_total": self.trials_total,
        "sweep_outcome": self.outcome,
        "warnings": list(self.warnings) + [w for w in extra_warnings if w],
    }




class BaseStage(abc.ABC):
  """Abstract interface for a campaign execution stage."""

  def __init__(self, context: CampaignContext):
    self.context = context
    self.config = context.config
    self.state = context.state
    self.sweep_controller = context.sweep_controller
    self.model_manager = context.model_manager

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """Stage identifier (e.g. 'sft', 'rm', 'perl', 'eval')."""

  @abc.abstractmethod
  def execute(
      self,
      live_line_callback: Optional[Callable[[str], None]] = None,
      stop_requested_callback: Optional[Callable[[], bool]] = None,
  ) -> StageResult:
    """Executes the stage logic and returns the resulting StageResult."""

  # --- Shared helpers -------------------------------------------------
  def tunable_keys(self, sweep_dict: Dict[str, Any]) -> Set[str]:
    """Returns the hyperparameter names this sweep actually searches over.

    Only these may be replayed on the materialization command line: the rest
    of a W&B run config is the training-argument and model-config dump, which
    the training scripts' argument parsers reject.

    Args:
      sweep_dict: The sweep configuration that was registered.

    Returns:
      The set of parameter names, or an empty set when the sweep declares
      none (the model manager then falls back to its own allowlist).
    """
    params = sweep_dict.get("parameters")
    if isinstance(params, dict):
      return {str(key) for key in params}
    return set()

  def materialization_stop_callback(self) -> Optional[Callable[[], bool]]:
    """Returns the predicate that may interrupt the winner's retraining.

    The predicate handed to the *sweep agent* is True for ``[a]`` advance and
    ``[s]`` stop, because both mean "seal the sweep now". Reusing it for the
    materialization made the retraining subprocess die on its very first
    poll, raising "Materialization ... was interrupted before the checkpoint
    could be pushed" - a non-retryable error that failed the whole campaign
    precisely when the user asked to *keep going* with the current best run.

    Only a hard abort may kill the retraining; when no abort predicate was
    supplied (e.g. a stage constructed directly in a test), materialization
    simply runs to completion.
    """
    return self.context.abort_requested_callback

  def select_winner(
      self,
      sweep_id: str,
      stage_config: SweepStageConfig,
      metric_name: str,
      live_line_callback: Optional[Callable[[str], None]] = None,
  ) -> RunScore:
    """Picks the sweep's winning trial and explains how it was picked.

    Under ``selection_strategy="best"`` a trial is scored on the peak of its
    logged history rather than on wherever it happened to stop. That matches
    what is actually shipped: the materialization publishes the best-scoring
    checkpoint, so ranking configurations by their last eval would penalise
    exactly the overfitting tail that ``--load_best_model_at_end`` discards.

    Args:
      sweep_id: Sweep to score.
      stage_config: This stage's configuration.
      metric_name: Metric to rank on.
      live_line_callback: Optional sink for the explanation lines.

    Returns:
      The winning trial's score.
    """
    winner = self.sweep_controller.fetch_best_run_details(
        sweep_id=sweep_id,
        metric_name=metric_name,
        goal=stage_config.goal,
        live_line_callback=live_line_callback,
        selection=stage_config.selection_strategy,
        # None on every stage that does not rank on a window; the query
        # ignores it there, but it still has to be an int.
        window=stage_config.selection_window or 1,
    )
    label = self.name.upper()
    logger.info(
        "Best %s Run: %s (%s=%.5f, %s)",
        label,
        winner.run_id,
        metric_name,
        winner.value,
        winner.describe_selection(),
    )
    if live_line_callback:
      live_line_callback(
          f"Best {label} Run: {winner.run_id} "
          f"({metric_name}={winner.value:.5f}; {winner.describe_selection()})"
      )
      # The gap between the peak and the end of the run is the part of the
      # score that comes from early stopping. Surfacing it is what stops a
      # lucky spike on a noisy metric from passing as a result.
      if winner.step is not None and winner.final_value is not None:
        drift = abs(winner.value - winner.final_value)
        live_line_callback(
            f"  Early stopping recovered {drift:.5f} of {metric_name} versus "
            f"the end of the run; the published checkpoint is the step-"
            f"{winner.step} one."
        )
      # A winner drawn from a shrunken field is still a winner, but it is a
      # weaker claim, and the operator watching the log is the person best
      # placed to decide whether to rerun the sweep.
      pool = winner.describe_pool()
      if pool:
        live_line_callback(f"[WARNING] {label}: {pool}")

    self.sweep_controller.mark_best_run(
        sweep_id=sweep_id,
        run_id=winner.run_id,
        stage_name=self.name,
        metric_name=metric_name,
        metric_value=winner.value,
        live_line_callback=live_line_callback,
    )
    return winner

  def selection_warnings(self, winner: RunScore) -> List[str]:
    """Returns the degradations that winner selection itself discovered.

    Kept separate from the trial-accounting warnings because the two are
    independent: a sweep can run all of its trials and still rank only some
    of them (a crashed run is excluded), and a sweep can lose trials and
    still rank every run that exists.

    Args:
      winner: The selected trial.

    Returns:
      Zero or one warning line, ready to append to ``StageResult.warnings``.
    """
    pool = winner.describe_pool()
    return [f"{self.name.upper()}: {pool}"] if pool else []

  def materialization_checkpoint_kwargs(
      self, stage_config: SweepStageConfig
  ) -> Dict[str, Any]:
    """Returns the checkpoint-selection kwargs for ``materialize_and_push``."""
    return {
        "checkpoint_policy": stage_config.checkpoint_policy,
        "eval_steps": stage_config.materialization_eval_steps,
    }

  def stage_result_selection_fields(self, winner: RunScore) -> Dict[str, Any]:
    """Returns the ``StageResult`` fields that record how ``winner`` was chosen."""
    return {
        "best_run_id": winner.run_id,
        "best_metric_val": winner.value,
        "best_params": winner.params,
        "selection_strategy": winner.selection,
        "selection_step": winner.step,
        "selection_window": winner.window_points,
        "final_metric_val": winner.final_value,
    }

  def remaining_runs(
      self,
      sweep_id: str,
      max_runs: int,
      live_line_callback: Optional[Callable[[str], None]] = None,
  ) -> int:
    """Returns how many trials still need to run to reach ``max_runs``.

    ``wandb agent --count N`` bounds a single agent process, not the sweep.
    Passing the full budget to a resumed stage runs a second full budget on
    top of the trials already paid for: stopping at 6 of 10 and resuming
    would produce 16 trials, not 10.

    The count comes from W&B, which is the only party that knows the real
    total. When it cannot be established the budget is left untouched -
    overshooting costs GPU hours, but undershooting would silently shrink
    the search the user asked for.

    Args:
      sweep_id: The sweep being executed.
      max_runs: The configured trial budget for the whole stage.
      live_line_callback: Optional sink for an explanatory line.

    Returns:
      Trials left to run; 0 when the budget is already spent.
    """
    budget = max(0, int(max_runs or 0))
    done = self.sweep_controller.count_finished_runs(sweep_id)
    if done is None or done <= 0:
      return budget

    remaining = max(0, budget - int(done))
    if live_line_callback:
      if remaining:
        live_line_callback(
            f"Sweep already has {done}/{budget} finished trials; running the"
            f" remaining {remaining}."
        )
      else:
        live_line_callback(
            f"Sweep already has {done}/{budget} finished trials; skipping the"
            " agent and scoring the trials that exist."
        )
    return remaining

  def run_sweep_trials(
      self,
      sweep_id: str,
      stage_config: SweepStageConfig,
      live_line_callback: Optional[Callable[[str], None]] = None,
      stop_requested_callback: Optional[Callable[[], bool]] = None,
  ) -> SweepExecution:
    """Runs the sweep agent and reports what the sweep actually produced.

    Args:
      sweep_id: The sweep to execute.
      stage_config: This stage's configuration (budget and timeout).
      live_line_callback: Optional sink for agent output and notices.
      stop_requested_callback: Predicate that cuts the agent short.

    Returns:
      The accounting for this sweep; see :class:`SweepExecution`.
    """
    budget = max(0, int(stage_config.max_runs or 0))
    remaining = self.remaining_runs(sweep_id, budget, live_line_callback)
    agent: Optional[AgentRun] = None
    if remaining:
      agent = self.sweep_controller.run_sweep_agent_detailed(
          sweep_id=sweep_id,
          max_runs=remaining,
          timeout_minutes=stage_config.timeout_minutes,
          live_line_callback=live_line_callback,
          stop_requested_callback=stop_requested_callback,
      )
    return self.account_for_trials(
        sweep_id=sweep_id,
        budget=budget,
        agent=agent,
        live_line_callback=live_line_callback,
    )

  def account_for_trials(
      self,
      sweep_id: str,
      budget: int,
      agent: Optional[AgentRun] = None,
      live_line_callback: Optional[Callable[[str], None]] = None,
  ) -> SweepExecution:
    """Establishes how many trials the sweep really finished, and says so.

    The live counter shown while the stage runs is parsed from the agent's
    stdout, and it counts a trial as "done" the moment the agent stops
    talking about it - including when the trial crashed, and including
    trials the agent never got to report on because it died mid-way. It is
    good enough for a progress bar and worthless as a record.

    So once the agent is back, the count is re-established from W&B, which
    is the only party that knows how many trials reached ``finished``. That
    number is what the stage carries into its result, and any shortfall is
    turned into a warning that follows the campaign all the way to the
    report. A stage that lost two trials to a dead VM used to be
    indistinguishable from one that ran perfectly.

    Args:
      sweep_id: The sweep that was executed.
      budget: Trials the stage was configured to run.
      agent: Outcome of the agent process, when one was launched.
      live_line_callback: Optional sink for the warnings.

    Returns:
      The accounting for this sweep.
    """
    label = self.name.upper()
    warnings: List[str] = []

    if self.config.dry_run:
      # No W&B to ask, and the simulated agent runs the whole budget.
      return SweepExecution(
          trials_done=budget, trials_total=budget, outcome="complete"
      )

    finished = self.sweep_controller.count_finished_runs(sweep_id)
    reason = agent.describe() if agent is not None else None

    if finished is None:
      # Refuse to invent a number. The previously persisted progress is a
      # lower bound parsed from stdout, and saying so is better than either
      # a confident lie or a blank.
      previous = self.state.stages.get(self.name)
      trials_done = int(getattr(previous, "trials_done", 0) or 0)
      outcome = "unknown"
      warnings.append(
          f"W&B could not be asked how many {label} trials finished, so the"
          f" trial count below ({trials_done}/{budget}) is a lower bound"
          " parsed from the agent's output and may be wrong."
      )
    else:
      trials_done = int(finished)
      if budget and trials_done < budget:
        outcome = "partial"
        because = f" because {reason}" if reason else ""
        warnings.append(
            f"{label} sweep is INCOMPLETE: {trials_done} of {budget} trials"
            f" finished{because}. The winner was chosen from those"
            f" {trials_done} trials only, so this stage searched less of the"
            " hyperparameter space than the campaign asked for. Re-run the"
            " campaign with `resume` to run the missing"
            f" {budget - trials_done}."
        )
      else:
        outcome = "complete"

    if trials_done == 0 and budget:
      # Nothing to rank. Left to the winner query to fail on, deliberately:
      # a sweep whose runs all crashed can still have logged a usable
      # metric, and failing here would throw that away.
      warnings.append(
          f"No {label} trial reached the 'finished' state. Any winner below"
          " comes from a run W&B does not consider complete."
      )

    execution = SweepExecution(
        trials_done=trials_done,
        trials_total=budget,
        outcome=outcome,
        warnings=warnings,
    )
    self.record_trial_accounting(execution)

    for warning in warnings:
      logger.warning("%s", warning)
      if live_line_callback:
        live_line_callback(f"[WARNING] {warning}")
    return execution

  def record_trial_accounting(self, execution: SweepExecution) -> None:
    """Persists ``execution`` immediately, before materialization starts.

    Materializing the winner retrains it from scratch and can take longer
    than the sweep did. A crash in there must not lose the record of what
    the sweep produced.

    Args:
      execution: The accounting to persist.
    """
    result = self.state.stages.get(self.name)
    if result is None:
      result = StageResult(status=StageStatus.RUNNING)
      self.state.stages[self.name] = result
    for key, value in execution.stage_result_fields().items():
      setattr(result, key, value)
    if self.context.state_path:
      try:
        self.state.save(self.context.state_path)
      except OSError as exc:
        logger.warning("Could not persist trial accounting: %s", exc)

  def get_sweep_descriptor(self, sweep_dict: Dict[str, Any]) -> Tuple[str, str]:
    """Returns (model_short_name, stage_type_string) for this stage.

    Args:
      sweep_dict: Sweep configuration dictionary.

    Returns:
      Tuple of (short model name, stage type string).
    """
    del sweep_dict  # Default implementation does not inspect command args.
    model_short = self.config.base_model.rstrip("/").split("/")[-1]
    return model_short, self.name.upper()

  def campaign_sweep_token(self) -> str:
    """Returns the hex token shared by every sweep of this campaign.

    See :mod:`src.orchestrator.naming` for why the old ``#1`` counter had to
    go.

    Returns:
      A short lowercase hex string derived from the campaign's start time.
    """
    return naming.campaign_token(
        campaign_id=self.state.campaign_id or self.config.name,
        created_at=getattr(self.state, "created_at", None),
    )

  def generate_sweep_name(self, sweep_dict: Dict[str, Any]) -> str:
    """Constructs a descriptive human-readable W&B sweep name.

    Format: [TASK NAME] [MODEL NAME & SIZE] [STAGE / DATA TYPE] [Sweep TOKEN]
    Example: BOSCH gemma-4-E2B-it SFT Sweep 35d2a7
             BOSCH gemma-3-1b-it RM Organic Sweep 35d2a7
             BOSCH gemma-4-E2B-it PERL Organic Sweep 35d2a7

    The trailing token is the same for all three stages of one campaign, so
    the sweeps that belong together can be found together in the W&B UI.

    Args:
      sweep_dict: Sweep configuration dictionary.

    Returns:
      Descriptive sweep name formatted string.
    """
    task_str = self.config.task_name.upper()
    model_short, stage_type_str = self.get_sweep_descriptor(sweep_dict)
    prefix = f"{task_str} {model_short} {stage_type_str}"
    return f"{prefix} Sweep {self.campaign_sweep_token()}"

  def resolve_sweep_id(
      self,
      sweep_dict: Dict[str, Any],
      live_line_callback: Optional[Callable[[str], None]] = None,
  ) -> str:
    """Reuses the sweep of an interrupted attempt, or registers a new one.

    Registering a fresh sweep after a crash silently discards every trial the
    previous attempt paid for. When the persisted state still holds a sweep id
    for this stage and the stage never completed, the agent is pointed back at
    that sweep instead.

    The recorded sweep is verified first. A sweep deleted from the W&B UI
    between two runs would otherwise be handed to ``wandb agent``, which
    fails the whole campaign with a confusing "sweep not found" - even
    though there is nothing to salvage and a new sweep is exactly what the
    user wants. Verification is tri-state on purpose: when W&B cannot be
    reached the sweep is *kept*, because abandoning a live sweep on a
    transient error is far more expensive than a failed agent launch.

    Args:
      sweep_dict: Sweep configuration to register when there is nothing to
        resume.
      live_line_callback: Optional log sink.

    Returns:
      The sweep id to run the agent against.
    """
    previous = self.state.stages.get(self.name)
    if (
        previous is not None
        and previous.sweep_id
        and previous.status != StageStatus.COMPLETED
    ):
      name_hint = f" ({previous.sweep_name})" if previous.sweep_name else ""
      still_exists = self.sweep_controller.sweep_exists(previous.sweep_id)
      if still_exists is False:
        message = (
            f"Recorded {self.name.upper()} sweep {previous.sweep_id}"
            f"{name_hint} no longer exists on W&B (deleted?). "
            "Registering a new sweep."
        )
        logger.warning(message)
        if live_line_callback:
          live_line_callback(f"[WARNING] {message}")
        # Drop the stale pointer so a later crash does not resurrect it.
        previous.sweep_id = None
        previous.sweep_name = None
        if self.context.state_path:
          try:
            self.state.save(self.context.state_path)
          except OSError as exc:
            logger.warning("Could not persist cleared sweep id: %s", exc)
      else:
        if still_exists is None:
          logger.warning(
              "Could not confirm sweep %s exists; reusing it rather than "
              "risking the loss of completed trials.",
              previous.sweep_id,
          )
        # Existing is not the same as usable. Aborting a campaign seals its
        # sweep (stop_sweep), and a sealed sweep still resolves through the
        # API but rejects every agent with "Sweep <id> is not running".
        # Reactivate it, because its finished trials are exactly what this
        # resume is trying to keep.
        if self.sweep_controller.sweep_is_running(previous.sweep_id) is False:
          reactivated = self.sweep_controller.resume_sweep(previous.sweep_id)
          notice = (
              f"{self.name.upper()} sweep {previous.sweep_id} was stopped; "
              + (
                  "reactivated it."
                  if reactivated
                  else "it could NOT be reactivated automatically. Resume it "
                  "in the W&B UI (or `wandb sweep --resume "
                  f"{previous.sweep_id}`) and run this command again."
              )
          )
          logger.warning(notice)
          if live_line_callback:
            live_line_callback(
                notice if reactivated else f"[WARNING] {notice}"
            )
        message = (
            f"Resuming existing {self.name.upper()} sweep "
            f"{previous.sweep_id}{name_hint} instead of starting a new one."
        )
        logger.info(message)
        if live_line_callback:
          live_line_callback(message)
        return previous.sweep_id

    # If sweep_dict doesn't already have a descriptive name, generate one
    if not sweep_dict.get("name"):
      sweep_name = self.generate_sweep_name(sweep_dict)
      sweep_dict["name"] = sweep_name
    else:
      sweep_name = sweep_dict["name"]

    message = f"Registering sweep '{sweep_name}' on W&B..."
    logger.info(message)
    if live_line_callback:
      live_line_callback(message)

    sweep_id = self.sweep_controller.create_sweep(sweep_dict)
    self.record_sweep_id(sweep_id, sweep_name=sweep_name)
    return sweep_id

  def record_sweep_id(
      self, sweep_id: str, sweep_name: Optional[str] = None
  ) -> None:
    """Persists ``sweep_id`` and optional ``sweep_name`` immediately so a crash stays resumable."""
    result = self.state.stages.get(self.name)
    if result is None:
      result = StageResult(status=StageStatus.RUNNING)
      self.state.stages[self.name] = result
    result.sweep_id = sweep_id
    if sweep_name:
      result.sweep_name = sweep_name
    if self.context.state_path:
      try:
        self.state.save(self.context.state_path)
      except OSError as exc:
        logger.warning("Could not persist sweep id to state file: %s", exc)

