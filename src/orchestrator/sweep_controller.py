"""W&B Sweep lifecycle manager, execution agent, and best-run query controller."""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import subprocess
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.orchestrator.config import RobustnessConfig
from src.orchestrator.process import stream_subprocess
from src.orchestrator.retry import run_with_retries

logger = logging.getLogger(__name__)

#: Upper bound on the number of history rows read per trial.
#:
#: PE-RL logs every optimizer step (``--logging_steps=1``), so a long trial
#: can hold thousands of rows. The eval metrics we actually rank on are
#: logged every 10-50 steps, which stays far below this; the cap only exists
#: so that a misconfigured sweep cannot turn the post-sweep query into a
#: multi-minute download.
MAX_HISTORY_ROWS = 20000


@dataclasses.dataclass(frozen=True)
class RunScore:
  """How one sweep trial scored, and how that score was arrived at.

  Attributes:
    run_id: The W&B run id.
    value: The score the trial is ranked on.
    params: The run's config, i.e. its hyperparameters.
    final_value: The last value the trial logged for the metric. Equal to
      ``value`` under ``selection="final"``.
    step: The step at which ``value`` was observed, when that is a peak
      strictly better than ``final_value``. None otherwise, including when
      the trial simply ended at its best point.
    selection: The strategy that produced ``value`` ('final' or 'best').
    from_history: True when ``value`` came from the logged history rather
      than from ``run.summary``. False under ``selection="best"`` means the
      history was unreadable and the summary was used as a fallback.
  """

  run_id: str
  value: float
  params: Dict[str, Any] = dataclasses.field(default_factory=dict)
  final_value: Optional[float] = None
  step: Optional[int] = None
  selection: str = "final"
  from_history: bool = False

  def describe_selection(self) -> str:
    """Returns a short human-readable account of how ``value`` was chosen."""
    if self.selection != "best":
      return "final logged value"
    if not self.from_history:
      return "final logged value; history unavailable"
    if self.step is None:
      return "best logged value, which is also the final one"
    tail = ""
    if self.final_value is not None:
      tail = f" (final was {self.final_value:.5f})"
    return f"best logged value, at step {self.step}{tail}"


def _metric_key_candidates(target: str) -> List[str]:
  """Returns the key spellings a training script may have used for ``target``.

  The sweep YAML names a metric the way W&B displays it (``eval/loss``), but
  the Hugging Face Trainer logs ``eval_loss`` and TRL prefixes some keys with
  ``train/``. Ranking must not depend on which of those a script happens to
  emit.

  Args:
    target: Metric name as configured.

  Returns:
    Candidate keys, most specific first, without duplicates.
  """
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

  seen = set()
  unique = []
  for cand in candidates:
    if cand not in seen:
      seen.add(cand)
      unique.append(cand)
  return unique


def _coerce_float(value: Any) -> Optional[float]:
  """Returns ``value`` as a finite float, or None when it is not usable.

  NaN and infinity are rejected: a diverged trial logging ``inf`` would
  otherwise win a ``maximize`` sweep outright. Booleans are rejected too,
  since ``True`` would silently score as 1.0.

  Args:
    value: A raw value from a run summary or history row.

  Returns:
    The finite float, or None.
  """
  if value is None or isinstance(value, bool):
    return None
  if not isinstance(value, (int, float)):
    return None
  try:
    as_float = float(value)
  except (ValueError, TypeError):
    return None
  if not math.isfinite(as_float):
    return None
  return as_float


def _resolve_metric_key(
    source: Mapping[str, Any], target: str
) -> Optional[str]:
  """Returns the key under which ``source`` actually holds ``target``.

  Args:
    source: A run summary (or any mapping of logged keys).
    target: Metric name as configured.

  Returns:
    The matching key, or None when the metric is absent or unusable.
  """
  for candidate in _metric_key_candidates(target):
    try:
      present = candidate in source
    except TypeError:
      return None
    if present and _coerce_float(source[candidate]) is not None:
      return candidate
  return None


def _iter_history_rows(run: Any, key: str) -> Optional[Sequence[Any]]:
  """Reads the run's logged rows for ``key``.

  ``scan_history`` is exact, unlike ``history()`` which subsamples to ~500
  points and could therefore miss the very peak we are looking for. The
  ``_step`` column is requested alongside so the peak can be reported, but
  a backend that rejects the pair - or that keeps ``_step`` out of these
  rows, which makes the qualified query return *nothing* - must not cost us
  the scan. An empty result is therefore treated as a miss and retried on
  the bare key; silently accepting it would downgrade the trial to
  final-step scoring without anybody noticing.

  Args:
    run: A ``wandb`` run object.
    key: The exact logged key to read.

  Returns:
    The non-empty list of rows, or None when no history could be read.
  """
  scan = getattr(run, "scan_history", None)
  if not callable(scan):
    return None
  for keys in ([key, "_step"], [key]):
    try:
      rows = list(scan(keys=keys))
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.debug("scan_history(%s) failed for run %s: %s", keys, run, e)
      continue
    if rows:
      return rows
  return None


def _scan_metric_history(
    run: Any, key: str, goal: str
) -> Optional[Tuple[float, Optional[int]]]:
  """Finds a trial's best logged value of ``key`` over all of its steps.

  This is the early-stopping view of a trial: the score of the checkpoint
  that ``--load_best_model_at_end`` would keep, rather than the score at
  whichever step the run happened to end on.

  Args:
    run: A ``wandb`` run object.
    key: The exact key to read, as resolved by :func:`_resolve_metric_key`.
    goal: 'minimize' or 'maximize'.

  Returns:
    ``(best_value, step)`` - ``step`` is None when the row carried no
    ``_step``. None when the history could not be read or held no usable
    value, which the caller must treat as "fall back to the summary".
  """
  rows = _iter_history_rows(run, key)
  if not rows:
    return None

  maximize = goal == "maximize"
  best_value: Optional[float] = None
  best_step: Optional[int] = None
  for index, row in enumerate(rows):
    if index >= MAX_HISTORY_ROWS:
      logger.warning(
          "Stopped scanning the history of run %s after %d rows; the peak "
          "of '%s' is taken over that prefix only.",
          getattr(run, "id", "?"),
          MAX_HISTORY_ROWS,
          key,
      )
      break
    if not isinstance(row, Mapping):
      continue
    value = _coerce_float(row.get(key))
    if value is None:
      continue
    if best_value is None:
      is_better = True
    elif maximize:
      is_better = value > best_value
    else:
      is_better = value < best_value
    if is_better:
      best_value = value
      step = row.get("_step")
      best_step = int(step) if isinstance(step, (int, float)) else None

  if best_value is None:
    return None
  return best_value, best_step



class SweepController:
  """Controls W&B hyperparameter sweep registration, agent execution, and best-run extraction."""

  def __init__(
      self,
      entity: Optional[str] = None,
      project: str = "new_perl",
      dry_run: bool = False,
      robustness: Optional[RobustnessConfig] = None,
  ):
    self.entity = entity
    self.project = project
    self.dry_run = dry_run
    # Every W&B call below crosses the network during a day-long run, so they
    # all go through the shared retry policy.
    self.robustness = robustness or RobustnessConfig()

  def resolve_entity(self) -> str:
    """Resolves the active W&B entity with auto-detection fallback.

    Precedence:
    1. Explicitly configured entity (if provided and != 'auto')
    2. WANDB_ENTITY environment variable
    3. wandb.Api().default_entity (authenticated account default from .netrc / WANDB_API_KEY)
    4. Fallback to 'leobianco'
    """
    if self.entity and self.entity != "auto":
      return self.entity
    env_entity = os.environ.get("WANDB_ENTITY")
    if env_entity and env_entity.strip():
      self.entity = env_entity.strip()
      return self.entity
    if not self.dry_run:
      try:
        import wandb  # pylint: disable=g-import-not-at-top

        api = wandb.Api()
        default_ent = getattr(api, "default_entity", None)
        if default_ent and str(default_ent).strip():
          self.entity = str(default_ent).strip()
          logger.info(
              "Auto-detected W&B entity from credentials: %s", self.entity
          )
          return self.entity
      except Exception as e:
        logger.debug("Could not auto-detect W&B default_entity: %s", e)
    self.entity = self.entity or "leobianco"
    return self.entity

  def create_sweep(
      self,
      sweep_config: Dict[str, Any],
      project_override: Optional[str] = None,
      live_line_callback: Optional[Callable[[str], None]] = None,
  ) -> str:
    """Registers a new sweep configuration with the W&B backend.

    Args:
      sweep_config: Parsed sweep YAML.
      project_override: Register in a different project than the default.
      live_line_callback: Optional sink notified about retries.

    Returns:
      The fully qualified sweep id.
    """
    proj = project_override or self.project
    entity = self.resolve_entity()
    if self.dry_run:
      mock_id = f"mock_sweep_{int(time.time())}"
      logger.info(
          "[DRY-RUN] Registered simulated sweep: %s/%s/%s",
          entity,
          proj,
          mock_id,
      )
      return f"{entity}/{proj}/{mock_id}"

    def _register() -> str:
      import wandb  # pylint: disable=g-import-not-at-top

      active_entity = self.resolve_entity()
      sweep_id = wandb.sweep(sweep_config, entity=active_entity, project=proj)
      # Ensure sweep_id has full path
      if "/" not in str(sweep_id):
        return f"{active_entity}/{proj}/{sweep_id}"
      return str(sweep_id)

    try:
      return run_with_retries(
          _register,
          description="W&B sweep registration",
          attempts=self.robustness.api_attempts,
          base_delay_s=self.robustness.retry_base_delay_s,
          max_delay_s=self.robustness.max_delay_s,
          on_notice=live_line_callback,
      )
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
        Exit code of the agent process (0 when the agent was deliberately cut
        short by the timeout or by a user stop request).
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
    entity = self.resolve_entity()
    full_sweep_id = sweep_id
    if "/" not in sweep_id:
      full_sweep_id = f"{entity}/{self.project}/{sweep_id}"

    cmd = ["wandb", "agent", "--count", str(max_runs), full_sweep_id]
    logger.info("Launching sweep agent: %s", " ".join(cmd))

    try:
      outcome = stream_subprocess(
          cmd,
          on_line=live_line_callback,
          stop_requested=stop_requested_callback,
          timeout_s=timeout_minutes * 60,
          stall_warning_s=self.robustness.stall_warning_minutes * 60,
      )
    except Exception as e:
      logger.error("Error during sweep agent execution: %s", e)
      self.stop_sweep(full_sweep_id)
      raise

    if outcome.timed_out:
      msg = (
          f"Sweep exceeded its {timeout_minutes} minute budget; sealing the "
          "sweep and promoting the best trial completed so far."
      )
      logger.warning(msg)
      if live_line_callback:
        live_line_callback(f"[WARNING] {msg}")
    elif outcome.interrupted:
      logger.info("Sweep agent stopped early on user request.")
    elif outcome.returncode != 0:
      msg = (
          f"wandb agent exited with code {outcome.returncode}. The sweep will "
          "still be scored on whatever trials finished."
      )
      logger.warning(msg)
      if live_line_callback:
        live_line_callback(f"[WARNING] {msg}")

    self.stop_sweep(full_sweep_id)
    # A deliberate termination is not a failure of the campaign.
    return 0 if outcome.cut_short else outcome.returncode


  def qualify_sweep_id(self, sweep_id: str) -> str:
    """Expands a bare or half-qualified sweep id into ``entity/project/id``.

    Args:
      sweep_id: Sweep id in any of the accepted shapes.

    Returns:
      The fully qualified sweep path.
    """
    entity = self.resolve_entity()
    full_sweep_id = str(sweep_id).strip()
    parts = full_sweep_id.split("/")
    if len(parts) == 1:
      return f"{entity}/{self.project}/{parts[0]}"
    if len(parts) == 2:
      return f"{entity}/{parts[0]}/{parts[1]}"
    return full_sweep_id

  def sweep_exists(self, sweep_id: str) -> Optional[bool]:
    """Checks whether ``sweep_id`` is still present on the W&B backend.

    Deliberately tri-state. A resume must distinguish "the sweep was
    deleted" (safe to register a fresh one) from "W&B is unreachable right
    now" - discarding a live sweep on a transient network blip would throw
    away every trial it has already paid for.

    Args:
      sweep_id: Sweep id in any accepted shape.

    Returns:
      True when the sweep is present, False when the backend positively
      reports it missing, and None when existence could not be determined.
    """
    if not sweep_id:
      return False
    if self.dry_run:
      return True

    full_sweep_id = self.qualify_sweep_id(sweep_id)
    try:
      import wandb  # pylint: disable=g-import-not-at-top

      api = wandb.Api()
      sweep = api.sweep(full_sweep_id)
      return sweep is not None
    except Exception as e:  # pylint: disable=broad-exception-caught
      text = f"{type(e).__name__}: {e}".lower()
      missing_hints = (
          "could not find sweep",
          "not found",
          "does not exist",
          "no sweep",
          "404",
      )
      if any(hint in text for hint in missing_hints):
        logger.info("Sweep %s no longer exists on W&B: %s", full_sweep_id, e)
        return False
      logger.warning(
          "Could not verify whether sweep %s still exists: %s",
          full_sweep_id,
          e,
      )
      return None

  def count_finished_runs(self, sweep_id: str) -> Optional[int]:
    """Counts the trials a sweep has already completed *successfully*.

    ``wandb agent --count N`` is a budget for *that agent process*, not for
    the sweep. A resumed stage that asked for the full ``max_runs`` again
    would happily run a second full budget on top of what it already paid
    for. W&B is the only place that knows the real total, since trials may
    also have been run by an agent this campaign never saw.

    Only ``finished`` runs are counted. A crashed, failed or killed trial
    consumed GPU time but produced no candidate model, and "10 trials" in a
    campaign config means "10 models to choose the best from", not "10
    attempts". Counting the wreckage would silently shrink the search - and
    since aborting the campaign (``[x]``) kills the trial in flight, every
    interruption would otherwise cost the user a trial.

    Args:
      sweep_id: Sweep id in any accepted shape.

    Returns:
      The number of successfully completed runs, or None when the count
      could not be established (the caller must then not reduce its budget).
    """
    if not sweep_id:
      return None
    if self.dry_run:
      return 0

    try:
      import wandb  # pylint: disable=g-import-not-at-top

      api = wandb.Api()
      sweep = api.sweep(self.qualify_sweep_id(sweep_id))
      return sum(
          1
          for run in sweep.runs
          if str(getattr(run, "state", "")).lower() == "finished"
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning(
          "Could not count finished runs for sweep %s: %s", sweep_id, e
      )
      return None

  def stop_sweep(self, sweep_id: str) -> None:
    """Closes and seals a sweep on the W&B backend, transitioning state to STOPPED."""
    if self.dry_run:
      logger.info("[DRY-RUN] Sealed sweep: %s", sweep_id)
      return

    full_sweep_id = self.qualify_sweep_id(sweep_id)

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

  def resume_sweep(self, sweep_id: str) -> bool:
    """Reactivates a sweep that was sealed, so agents are accepted again.

    The counterpart of :meth:`stop_sweep`. Aborting a campaign seals its
    sweep; without this, the next resume hands a stopped sweep to
    ``wandb agent`` and dies with "Sweep <id> is not running" while the
    trials it already paid for sit there, reachable but unusable.

    Args:
      sweep_id: Sweep id in any accepted shape.

    Returns:
      True when the sweep is believed to be accepting agents afterwards.
      False when it could not be reactivated - the caller should say so
      rather than launch an agent that is certain to fail.
    """
    if self.dry_run:
      logger.info("[DRY-RUN] Resumed sweep: %s", sweep_id)
      return True

    full_sweep_id = self.qualify_sweep_id(sweep_id)
    try:
      completed = subprocess.run(
          ["wandb", "sweep", "--resume", full_sweep_id],
          check=False,
          capture_output=True,
          timeout=30,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Could not resume sweep %s: %s", full_sweep_id, e)
      return False

    if completed.returncode == 0:
      logger.info("Resumed sweep %s", full_sweep_id)
      return True

    logger.warning(
        "wandb sweep --resume %s exited %d: %s",
        full_sweep_id,
        completed.returncode,
        (completed.stderr or b"").decode("utf-8", "replace").strip(),
    )
    return False

  def sweep_is_running(self, sweep_id: str) -> Optional[bool]:
    """Checks whether a sweep is in a state that accepts new agents.

    Existence is not enough: a stopped or finished sweep is still fetchable
    through the API but rejects every agent.

    Args:
      sweep_id: Sweep id in any accepted shape.

    Returns:
      True when the sweep accepts agents, False when it positively does
      not, and None when the state could not be determined.
    """
    if not sweep_id:
      return False
    if self.dry_run:
      return True

    try:
      import wandb  # pylint: disable=g-import-not-at-top

      api = wandb.Api()
      sweep = api.sweep(self.qualify_sweep_id(sweep_id))
      state = str(getattr(sweep, "state", "") or "").lower()
      if not state:
        return None
      # W&B reports PENDING/RUNNING for live sweeps and
      # STOPPED/FINISHED/CANCELED/CRASHED for sealed ones.
      return state in ("running", "pending")
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Could not read state of sweep %s: %s", sweep_id, e)
      return None

  def mark_best_run(
      self,
      sweep_id: str,
      run_id: str,
      stage_name: str,
      metric_name: str,
      metric_value: Optional[float] = None,
      live_line_callback: Optional[Callable[[str], None]] = None,
  ) -> bool:
    """Tags the winning run so it stands out in the W&B web UI.

    W&B has no "pin" concept, but tags are rendered as chips in the runs
    table and are filterable, which is the closest thing. Two tags are
    applied: a generic ``best`` and a stage-qualified ``best-sft`` /
    ``best-rm`` / ``best-perl``, because one project holds the sweeps of
    every stage and every task.

    Tags are stripped from any *other* run in the same sweep that still
    carries them, so a re-run that picks a different winner leaves exactly
    one tagged run rather than a growing pile of former champions.

    This is cosmetic. It runs after hours of sweeping and immediately before
    materialization, so it is written to be incapable of failing the stage:
    every error is swallowed and reported.

    Args:
      sweep_id: Sweep the run belongs to.
      run_id: The winning run.
      stage_name: Stage that produced it, used for the qualified tag.
      metric_name: Metric the winner was chosen on, recorded in the note.
      metric_value: Its value, recorded in the note.
      live_line_callback: Optional sink for a confirmation line.

    Returns:
      True when the winner was tagged, False when nothing was changed.
    """
    if not sweep_id or not run_id:
      return False

    stage_tag = f"best-{stage_name.lower()}"
    tags = ("best", stage_tag)

    if self.dry_run:
      logger.info("[DRY-RUN] Tagged run %s with %s", run_id, list(tags))
      if live_line_callback:
        live_line_callback(f"[DRY-RUN] Tagged best run {run_id}: {', '.join(tags)}")
      return True

    try:
      import wandb  # pylint: disable=g-import-not-at-top

      api = wandb.Api()
      sweep = api.sweep(self.qualify_sweep_id(sweep_id))

      demoted = 0
      winner = None
      for run in sweep.runs:
        if run.id == run_id:
          winner = run
          continue
        existing = list(run.tags or [])
        remaining = [tag for tag in existing if tag not in tags]
        if len(remaining) != len(existing):
          run.tags = remaining
          run.update()
          demoted += 1

      if winner is None:
        # The winner came from this very sweep, so this means the listing is
        # stale rather than that the run is gone. Fetch it directly.
        winner = api.run(f"{self.resolve_entity()}/{self.project}/{run_id}")

      winner.tags = sorted(set(list(winner.tags or [])) | set(tags))
      value = "unknown" if metric_value is None else f"{metric_value:.5f}"
      winner.notes = (
          f"Selected by Auto-PERL as the best {stage_name.upper()} trial "
          f"({metric_name}={value})."
      )
      winner.update()

      logger.info(
          "Tagged run %s as %s (%d previous winner(s) untagged)",
          run_id,
          stage_tag,
          demoted,
      )
      if live_line_callback:
        live_line_callback(
            f"Tagged best run {run_id} in W&B as '{stage_tag}'"
            + (f"; untagged {demoted} previous winner(s)." if demoted else ".")
        )
      return True
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Could not tag best run %s: %s", run_id, e)
      if live_line_callback:
        live_line_callback(
            f"[WARNING] Could not tag the best run in W&B ({e}). The campaign"
            " is unaffected; this only changes how the sweep looks in the UI."
        )
      return False

  def fetch_best_run(
      self,
      sweep_id: str,
      metric_name: str,
      goal: str = "minimize",
      live_line_callback: Optional[Callable[[str], None]] = None,
      selection: str = "final",
  ) -> Tuple[str, float, Dict[str, Any]]:
    """Retrieves the best run, its metric value, and configuration from the sweep.

    Thin wrapper over :meth:`fetch_best_run_details` kept for callers that
    only need the triple.

    Args:
        sweep_id: Full path to sweep.
        metric_name: Target metric key (e.g. 'eval/loss', 'eval/roc_auc',
          'rewards/reward_fn/mean').
        goal: 'minimize' or 'maximize'.
        live_line_callback: Optional sink notified about retries.
        selection: 'final' to score each trial on its last logged value,
          'best' to score it on the extremum over its whole history.

    Returns:
        Tuple of (best_run_id, best_metric_val, best_hyperparameters_dict).
    """
    score = self.fetch_best_run_details(
        sweep_id=sweep_id,
        metric_name=metric_name,
        goal=goal,
        live_line_callback=live_line_callback,
        selection=selection,
    )
    return score.run_id, score.value, score.params

  def fetch_best_run_details(
      self,
      sweep_id: str,
      metric_name: str,
      goal: str = "minimize",
      live_line_callback: Optional[Callable[[str], None]] = None,
      selection: str = "final",
  ) -> RunScore:
    """Retrieves the winning run of a sweep together with how it was scored.

    Args:
        sweep_id: Full path to sweep.
        metric_name: Target metric key.
        goal: 'minimize' or 'maximize'.
        live_line_callback: Optional sink notified about retries.
        selection: See :meth:`fetch_best_run`.

    Returns:
        The winning :class:`RunScore`.

    Raises:
        ValueError: When the sweep is empty or no run logged ``metric_name``.
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
          "[DRY-RUN] Fetched mock best run %s with %s=%f (selection=%s)",
          mock_run_id,
          metric_name,
          mock_metric,
          selection,
      )
      # The simulated trial "peaks" slightly above where it ends, so a
      # dry run exercises the same reporting path as a real ``best`` pick.
      final_metric = 0.300 if goal == "minimize" else 0.950
      chose_peak = selection == "best"
      return RunScore(
          run_id=mock_run_id,
          value=mock_metric if chose_peak else final_metric,
          params=mock_params,
          final_value=final_metric,
          step=120 if chose_peak else None,
          selection=selection,
          from_history=chose_peak,
      )

    # This query happens right after hours of sweeping: a transient API error
    # here would throw away the whole stage, so it is worth retrying. A
    # genuinely missing metric raises a message flagged non-retryable.
    return run_with_retries(
        lambda: self._fetch_best_run_once(
            sweep_id, metric_name, goal, selection, live_line_callback
        ),
        description="W&B best-run query",
        attempts=self.robustness.api_attempts,
        base_delay_s=self.robustness.retry_base_delay_s,
        max_delay_s=self.robustness.max_delay_s,
        on_notice=live_line_callback,
    )

  def _score_run(
      self,
      run: Any,
      metric_name: str,
      goal: str,
      selection: str,
  ) -> Optional[RunScore]:
    """Scores a single trial according to ``selection``.

    Args:
      run: A ``wandb`` run object.
      metric_name: Metric the sweep optimises.
      goal: 'minimize' or 'maximize'.
      selection: 'final' or 'best'.

    Returns:
      The run's :class:`RunScore`, or None when it never logged the metric.
    """
    summary = getattr(run, "summary", {}) or {}
    key = _resolve_metric_key(summary, metric_name)
    if key is None:
      return None
    final_value = _coerce_float(summary.get(key))
    if final_value is None:
      return None

    base = RunScore(
        run_id=run.id,
        value=final_value,
        params=dict(getattr(run, "config", {}) or {}),
        final_value=final_value,
        selection=selection,
    )
    if selection != "best":
      return base

    peak = _scan_metric_history(run, key, goal)
    if peak is None:
      # The summary value is a real observation from this trial, just not
      # necessarily its best one. Falling back to it keeps the sweep
      # scoreable; failing here would discard hours of GPU time because an
      # API for a *refinement* was unavailable.
      logger.warning(
          "Could not read the logged history of run %s for '%s'; scoring it "
          "on its final value instead.",
          run.id,
          key,
      )
      return base

    peak_value, peak_step = peak
    # A trial whose peak is its last point is not a "best" pick in any
    # meaningful sense; reporting no step keeps that honest.
    improved = (
        peak_value < final_value if goal == "minimize" else peak_value > final_value
    )
    return dataclasses.replace(
        base,
        value=peak_value,
        step=peak_step if improved else None,
        from_history=True,
    )

  def _fetch_best_run_once(
      self,
      sweep_id: str,
      metric_name: str,
      goal: str,
      selection: str = "final",
      live_line_callback: Optional[Callable[[str], None]] = None,
  ) -> RunScore:
    """Single attempt of :meth:`fetch_best_run_details` (see it for the contract)."""
    try:
      import wandb  # pylint: disable=g-import-not-at-top

      api = wandb.Api()
      full_sweep_id = self.qualify_sweep_id(sweep_id)

      sweep = api.sweep(full_sweep_id)
      runs = list(sweep.runs)

      if not runs:
        raise ValueError(f"No runs found in sweep {full_sweep_id}")

      finished_runs: List[RunScore] = []
      other_runs: List[RunScore] = []

      for run in runs:
        score = self._score_run(run, metric_name, goal, selection)
        if score is None:
          continue
        if run.state == "finished":
          finished_runs.append(score)
        elif run.state in ("running", "failed"):
          other_runs.append(score)

      # Strictly prioritize finished runs over failed or running ones
      valid_runs = finished_runs if finished_runs else other_runs

      if not valid_runs:
        # Returning a sentinel here used to be silent data corruption: the
        # stage would materialize and publish a checkpoint trained with an
        # arbitrary run's hyperparameters and report a metric of 0.0.
        observed_keys = sorted(
            {
                str(key)
                for run in runs[:5]
                for key in dict(run.summary).keys()
                if not str(key).startswith("_")
            }
        )
        raise ValueError(
            f"No run in sweep {full_sweep_id} logged the metric "
            f"'{metric_name}' ({len(runs)} run(s) inspected). "
            "Check that the training script logs this key, or fix "
            "'metric.name' in the sweep YAML. Observed summary keys: "
            f"{observed_keys[:40]}"
        )

      # Sort by metric
      reverse = goal == "maximize"
      valid_runs.sort(key=lambda score: score.value, reverse=reverse)
      best = valid_runs[0]

      logger.info(
          "Best run for %s is %s with %s=%.5f (%s)",
          sweep_id,
          best.run_id,
          metric_name,
          best.value,
          best.describe_selection(),
      )
      if live_line_callback and selection == "best":
        if not any(score.from_history for score in valid_runs):
          live_line_callback(
              "[WARNING] Trials were meant to be ranked on their best "
              "evaluation step, but no logged history could be read; they "
              "were ranked on their final value instead."
          )
      return best

    except Exception as e:
      logger.error("Failed to query best run from W&B API: %s", e)
      raise
