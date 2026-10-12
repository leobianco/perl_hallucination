"""Model and checkpoint management: local retention, materialization, and HF Hub publishing."""

from __future__ import annotations

import dataclasses
import datetime
import fnmatch
import json
import logging
import os
import shutil
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src import checkpoint_publication
from src.orchestrator import accel
from src.orchestrator import flavors
from src.orchestrator import naming
from src.orchestrator.config import RobustnessConfig
from src.orchestrator.process import stream_subprocess
from src.orchestrator.retry import run_with_retries
from src.utils import sanitize_hf_repo_id

logger = logging.getLogger(__name__)


#: Hyperparameters that may be re-injected into the materialization command.
#:
#: ``wandb`` run configs do not only contain the sweep's search space: the
#: Hugging Face ``WandbCallback`` merges the *entire* ``TrainingArguments``
#: dump and the model's ``config.json`` into ``run.config``. Forwarding all of
#: that back as CLI flags (``--vocab_size``, ``--rope_theta``, ...) makes
#: ``HfArgumentParser`` abort with "some specified arguments are not used",
#: which would kill the campaign right after a multi-hour sweep. Stages pass
#: the sweep's own ``parameters:`` keys; this constant is the fallback.
DEFAULT_TUNABLE_KEYS: Set[str] = {
    "learning_rate",
    "num_train_epochs",
    "weight_decay",
    "warmup_ratio",
    "lr_scheduler_type",
    "min_lr_ratio",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "beta",
    "temperature",
    "num_generations",
    "max_completion_length",
    "sft_data_fraction",
    "num_fewshot",
}

#: Token budget the reward model uses to score (prompt + completion).
#:
#: Deliberately NOT in ``DEFAULT_TUNABLE_KEYS``: this is a correctness setting,
#: not a hyperparameter to sweep. The reward model must be trained and queried
#: with the same value, so both the ``rm`` and the ``perl`` stage read this one
#: constant. See ``_resolve_reward_max_length`` in ``src/pipelines.py`` for why
#: the previous hardcoded 512 silently zeroed the RLOO advantage on RAGTruth.
REWARD_MAX_LENGTH: int = 2048

#: Asymmetric multiplier on negative rewards in the PE-RL reward function.
#:
#: Pinned, and deliberately NOT tunable, for the same class of reason as
#: REWARD_MAX_LENGTH: it changes what the metric *means*. ``reward_fn`` scales
#: negative logit differences by this factor, so it rescales
#: ``rewards/reward_fn/mean`` - the quantity the PE-RL sweep maximizes.
#: Searching over it ranked trials on mutually incomparable scales and gave a
#: Bayesian optimizer a free way to raise the objective by weakening the
#: hallucination penalty. Both the sweep YAML and the winner's retraining pin
#: this value, so the two cannot drift apart; a stale ``reward_penalty_alpha``
#: in an older winner's W&B config is now ignored rather than replayed.
REWARD_PENALTY_ALPHA: float = 1.0

#: Default minimum learning rate at the end of the PE-RL cosine schedule, as a
#: fraction of the post-warmup peak learning rate (25%). Matches
#: ``--min_lr_ratio=0.25`` in ``scripts/sweep_perl.yaml`` and
#: ``_DEFAULT_PERL_MIN_LR_RATIO`` in ``src/pipelines.py``.
PERL_MIN_LR_RATIO: float = 0.25

#: Batch geometry for PE-RL winner retraining.
#:
#: These are NOT part of the PE-RL search space, so they are not replayed from
#: the winner's run config and have to be pinned to whatever every trial of
#: ``scripts/sweep_perl.yaml`` used. The defaults below are the ones a ~4B
#: policy was tuned with.
#:
#: They are overridable because they are the only thing in this command that
#: depends on the *size* of the model rather than on the experiment: a 7B
#: policy does not fit this geometry, and ``auto_find_batch_size=False``
#: removes the runtime escape hatch. Changing them changes the effective
#: batch size and therefore the optimisation problem, which is why this is a
#: deliberate, explicit override and not an automatic fallback.
#:
#: Override with ``PERL_<UPPERCASED_KEY>``, e.g.
#: ``PERL_PER_DEVICE_TRAIN_BATCH_SIZE=2 PERL_GRADIENT_ACCUMULATION_STEPS=16``.
#:
#: Note for RLOO/GRPO: the number of prompts in a generation batch must stay
#: divisible by ``num_generations``. Halve ``per_device_train_batch_size`` and
#: double ``gradient_accumulation_steps`` together rather than one of the two.
PERL_BATCH_GEOMETRY: Dict[str, str] = {
    "num_generations": "8",
    "num_iterations": "1",
    "steps_per_generation": "16",
    "per_device_train_batch_size": "4",
    "per_device_eval_batch_size": "16",
    "gradient_accumulation_steps": "8",
    "auto_find_batch_size": "False",
}


def perl_batch_geometry(
    env: Optional[Dict[str, str]] = None,
    policy_model: Optional[str] = None,
    reward_model: Optional[str] = None,
) -> Dict[str, str]:
  """Returns the PE-RL batch geometry for a given policy.

  Args:
    env: Environment mapping to read overrides from. Defaults to ``os.environ``.
    policy_model: Base model of the policy. When given, a model at or above
      :data:`src.orchestrator.accel.CHECKPOINT_THRESHOLD_B` halves the
      micro-batch and doubles the accumulation, exactly as the sweep stage
      does - the retraining has to reproduce the trial that won, so the two
      scalings are the same function applied to the same baseline.
    reward_model: Base model of the reward model, when it differs.

  Returns:
    A copy of :data:`PERL_BATCH_GEOMETRY` with the size scaling and then any
    ``PERL_<UPPERCASED_KEY>`` entries substituted. Values are passed through
    verbatim so that ``HfArgumentParser`` - not this function - owns their
    validation.

  The environment override is applied *last*, so an operator who pins
  ``PERL_PER_DEVICE_TRAIN_BATCH_SIZE`` still wins over the automatic choice.
  """
  source = os.environ if env is None else env
  geometry = dict(PERL_BATCH_GEOMETRY)
  if policy_model:
    geometry = accel.scale_perl_geometry(geometry, policy_model, reward_model)
  for key in geometry:
    override = source.get(f"PERL_{key.upper()}")
    if override is not None and str(override).strip():
      geometry[key] = str(override).strip()
      logger.info(
          "PE-RL batch geometry: --%s overridden to %s", key, geometry[key]
      )
  return geometry


#: Handshake file of a PE-RL run with continual evaluation, written into its
#: output directory. Mirrors
#: ``src.continual_eval.CONTINUAL_EVAL_STATUS_FILENAME`` (not imported here:
#: that module pulls in transformers and matplotlib);
#: ``src/test_continual_eval.py`` keeps the two in sync.
CONTINUAL_EVAL_STATUS_FILENAME = "continual_eval_status.json"

#: Lightweight per-step adapter archive directory inside a continual-eval trial.
STEP_ADAPTERS_DIRNAME = "adapters"
#: Preserved best constrained continual-eval checkpoint directory.
BEST_CONTINUAL_CHECKPOINT_DIRNAME = "best_continual_eval_checkpoint"
BEST_CONTINUAL_CHECKPOINT_META_FILENAME = (
    "continual_eval_best_checkpoint.json"
)

#: Patterns excluded when uploading a completed sweep trial directory directly
#: to Hugging Face Hub so only the promoted root adapter, ``last/`` adapter,
#: ``checkpoints.json``, continual-eval history, and Pareto plots are pushed.
TRIAL_CHECKPOINT_UPLOAD_IGNORE_PATTERNS: Tuple[str, ...] = tuple(
    list(checkpoint_publication.CHECKPOINT_UPLOAD_IGNORE_PATTERNS)
    + [
        "checkpoint-*",
        "checkpoint-*/*",
        f"{STEP_ADAPTERS_DIRNAME}",
        f"{STEP_ADAPTERS_DIRNAME}/*",
        f"{BEST_CONTINUAL_CHECKPOINT_DIRNAME}",
        f"{BEST_CONTINUAL_CHECKPOINT_DIRNAME}/*",
        "_best_checkpoint",
        "_best_checkpoint/*",
        BEST_CONTINUAL_CHECKPOINT_META_FILENAME,
        CONTINUAL_EVAL_STATUS_FILENAME,
    ]
)

_PUBLISHABLE_WEIGHT_FILES = (
    "adapter_config.json",
    "config.json",
    "adapter_model.safetensors",
    "model.safetensors",
    "pytorch_model.bin",
)


def _dir_has_publishable_weights(path: Optional[str]) -> bool:
  """Returns True when ``path`` is a directory containing model/adapter files."""
  if not path or not os.path.isdir(path):
    return False
  try:
    files = os.listdir(path)
  except OSError:
    return False
  return any(
      f in _PUBLISHABLE_WEIGHT_FILES or f.endswith((".safetensors", ".bin"))
      for f in files
  )


def _is_ignored_checkpoint_entry(name: str) -> bool:
  """Returns True when ``name`` matches training-state ignore patterns."""
  if name in ("ref", BEST_CONTINUAL_CHECKPOINT_META_FILENAME):
    return True
  for pattern in checkpoint_publication.CHECKPOINT_UPLOAD_IGNORE_PATTERNS:
    clean_pat = pattern.split("/", 1)[0]
    if fnmatch.fnmatch(name, clean_pat):
      return True
  return False


def _copy_publishable_checkpoint_files(src_dir: str, dst_dir: str) -> None:
  """Copies publishable model/tokenizer files from ``src_dir`` into ``dst_dir``."""
  os.makedirs(dst_dir, exist_ok=True)
  if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
    return
  for entry in os.listdir(src_dir):
    if _is_ignored_checkpoint_entry(entry):
      continue
    src_path = os.path.join(src_dir, entry)
    dst_path = os.path.join(dst_dir, entry)
    if os.path.isfile(src_path):
      shutil.copy2(src_path, dst_path)


def continual_eval_training_completed(output_dir: str) -> Optional[bool]:
  """Reads whether a run with continual evaluation has finished training.

  Args:
    output_dir: The training run's output directory.

  Returns:
    None when the run has no continual-evaluation state (it ran without
    continual evaluation, or never started); otherwise whether training
    completed. An unreadable state reads as not completed.
  """
  path = os.path.join(output_dir, CONTINUAL_EVAL_STATUS_FILENAME)
  if not os.path.exists(path):
    return None
  try:
    with open(path, encoding="utf-8") as handle:
      status = json.load(handle)
  except (OSError, ValueError) as e:
    logger.warning("Unreadable continual-evaluation status %s: %s", path, e)
    return False
  return isinstance(status, dict) and status.get("training_completed") is True


@dataclasses.dataclass(frozen=True)
class MaterializationPlan:
  """Everything needed to retrain and publish a sweep's winning trial.

  Attributes:
    repo_id: Hugging Face repository the retrained model is pushed to.
    command: The full ``accelerate launch`` argument vector.
    output_dir: Local directory the run writes its checkpoints to.
    injected: Winning hyperparameters that were replayed on the command line.
    skipped: Keys of the W&B run config that were deliberately not replayed.
  """

  repo_id: str
  command: List[str]
  output_dir: str
  injected: Dict[str, Any] = dataclasses.field(default_factory=dict)
  skipped: List[str] = dataclasses.field(default_factory=list)


class ModelManager:
  """Manages model checkpoint materialization, Hugging Face uploads, and local retention."""

  def __init__(
      self,
      user: str = "leobianco",
      dry_run: bool = False,
      deepspeed_config: str = "scripts/deepspeed_config.yaml",
      robustness: Optional[RobustnessConfig] = None,
  ):
    self.user = user
    self.dry_run = dry_run
    self.deepspeed_config = deepspeed_config
    # Materialization ends with a Hugging Face push, the single most
    # failure-prone moment of the whole campaign.
    self.robustness = robustness or RobustnessConfig()


  def verify_model_exists(self, repo_or_path: str) -> bool:
    """Verifies whether a model checkpoint exists locally or on Hugging Face Hub."""
    if self.dry_run:
      return True

    # Check local directory first
    if os.path.exists(repo_or_path):
      return True

    # Check Hugging Face Hub
    try:
      from huggingface_hub import HfApi  # pylint: disable=g-import-not-at-top

      api = HfApi()
      return api.repo_exists(repo_id=repo_or_path, repo_type="model")
    except Exception as e:
      logger.debug("HfApi repo check failed for '%s': %s", repo_or_path, e)
      return False

  def upload_local_checkpoint(
      self,
      folder_path: str,
      repo_id: str,
      commit_message: str = "Auto-PERL automated checkpoint upload",
      ignore_patterns: Optional[Sequence[str]] = None,
  ) -> str:
    """Pushes a local directory containing checkpoint weights directly to Hugging Face Hub."""
    if self.dry_run:
      logger.info(
          "[DRY-RUN] Simulating direct folder upload from %s to %s",
          folder_path,
          repo_id,
      )
      return repo_id

    def _upload() -> str:
      from huggingface_hub import HfApi  # pylint: disable=g-import-not-at-top

      api = HfApi()
      api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
      upload_kwargs: Dict[str, Any] = {
          "folder_path": folder_path,
          "repo_id": repo_id,
          "repo_type": "model",
          "commit_message": commit_message,
      }
      if ignore_patterns is not None:
        upload_kwargs["ignore_patterns"] = list(ignore_patterns)
      api.upload_folder(**upload_kwargs)
      return repo_id

    try:
      run_with_retries(
          _upload,
          description=f"Hugging Face upload for {repo_id}",
          attempts=self.robustness.api_attempts,
          base_delay_s=self.robustness.retry_base_delay_s,
          max_delay_s=self.robustness.max_delay_s,
      )
      logger.info("Successfully pushed folder %s to Hugging Face: %s", folder_path, repo_id)
      return repo_id
    except Exception as e:
      logger.error("Failed to upload folder to Hugging Face Hub: %s", e)
      raise

  @staticmethod
  def _trial_has_any_publishable_weights(trial_dir: str) -> bool:
    """Returns True when ``trial_dir`` or one of its checkpoint archives has weights."""
    if _dir_has_publishable_weights(trial_dir):
      return True
    if _dir_has_publishable_weights(
        os.path.join(trial_dir, BEST_CONTINUAL_CHECKPOINT_DIRNAME)
    ):
      return True
    adapters_root = os.path.join(trial_dir, STEP_ADAPTERS_DIRNAME)
    if os.path.isdir(adapters_root):
      try:
        for entry in os.listdir(adapters_root):
          if _dir_has_publishable_weights(os.path.join(adapters_root, entry)):
            return True
      except OSError:
        pass
    try:
      for entry in os.listdir(trial_dir):
        if entry.startswith("checkpoint-") and _dir_has_publishable_weights(
            os.path.join(trial_dir, entry)
        ):
          return True
    except OSError:
      pass
    return False

  @staticmethod
  def _trial_matches_run_id(trial_dir: str, run_id: str) -> bool:
    """Returns True when ``trial_dir`` metadata records ``run_id``."""
    status_path = os.path.join(trial_dir, CONTINUAL_EVAL_STATUS_FILENAME)
    if os.path.isfile(status_path):
      try:
        with open(status_path, encoding="utf-8") as handle:
          status = json.load(handle)
        if isinstance(status, dict) and status.get("wandb_run_id") == run_id:
          return True
      except (OSError, ValueError):
        pass
    id_file = os.path.join(trial_dir, "wandb_run_id.txt")
    if os.path.isfile(id_file):
      try:
        with open(id_file, encoding="utf-8") as handle:
          if handle.read().strip() == run_id:
            return True
      except OSError:
        pass
    return False

  def find_completed_trial_checkpoint(
      self,
      task_name: str,
      stage_name: str,
      run_id: Optional[str],
      base_dir: str = "./checkpoints",
  ) -> Optional[str]:
    """Locates a completed local sweep trial checkpoint directory for ``run_id``.

    Checks ``<base_dir>/<task_name>/<stage_name>/<run_id>`` first, then scans
    sibling directories under ``<base_dir>/<task_name>/<stage_name>`` whose
    ``continual_eval_status.json`` or ``wandb_run_id.txt`` matches ``run_id``.
    Incomplete or paused continual-evaluation trials
    (``continual_eval_training_completed(dir) is False``) are refused.

    Args:
      task_name: Task identifier (e.g. ``'npov'``).
      stage_name: Stage name (e.g. ``'perl'``).
      run_id: W&B run ID of the winning sweep trial.
      base_dir: Root checkpoint directory (default ``'./checkpoints'``).

    Returns:
      Path to the completed trial checkpoint directory, or None if unavailable.
    """
    if not run_id or not str(run_id).strip():
      return None
    clean_run_id = str(run_id).strip()
    stage_dir = os.path.join(base_dir, task_name, stage_name)
    if not os.path.isdir(stage_dir):
      return None

    candidates: List[str] = []
    direct = os.path.join(stage_dir, clean_run_id)
    if os.path.isdir(direct):
      candidates.append(direct)
    try:
      for entry in sorted(os.listdir(stage_dir)):
        sub = os.path.join(stage_dir, entry)
        if sub == direct or not os.path.isdir(sub):
          continue
        if self._trial_matches_run_id(sub, clean_run_id):
          candidates.append(sub)
    except OSError:
      pass

    for cand in candidates:
      if continual_eval_training_completed(cand) is False:
        logger.warning(
            "Skipping local trial directory %s for run %s: continual "
            "evaluation training is not completed.",
            cand,
            clean_run_id,
        )
        continue
      if self._trial_has_any_publishable_weights(cand):
        return cand
    return None

  def prepare_trial_checkpoint_for_upload(
      self,
      trial_dir: str,
      winner_step: Optional[int] = None,
  ) -> str:
    """Ensures ``trial_dir`` root holds the selected step's publishable adapter weights.

    When ``winner_step`` is provided and a matching per-step archive exists in
    ``adapters/step-<winner_step>``, ``checkpoint-<winner_step>``, or
    ``best_continual_eval_checkpoint``, promotes those publishable files to the
    root of ``trial_dir`` and updates ``checkpoints.json`` if present. If the
    root has no publishable weights yet, promotes from
    ``best_continual_eval_checkpoint`` or the latest available step archive.

    Args:
      trial_dir: Completed trial checkpoint directory.
      winner_step: Optional optimizer step selected by the sweep controller.

    Returns:
      ``trial_dir`` ready for upload.
    """
    target_step = checkpoint_publication.coerce_step(winner_step)
    promoted_src: Optional[str] = None
    promoted_step: Optional[int] = target_step

    if target_step is not None:
      step_adapter = os.path.join(
          trial_dir, STEP_ADAPTERS_DIRNAME, f"step-{target_step}"
      )
      step_ckpt = os.path.join(trial_dir, f"checkpoint-{target_step}")
      best_dir = os.path.join(trial_dir, BEST_CONTINUAL_CHECKPOINT_DIRNAME)
      best_meta = os.path.join(best_dir, BEST_CONTINUAL_CHECKPOINT_META_FILENAME)
      best_meta_step: Optional[int] = None
      if os.path.isfile(best_meta):
        try:
          with open(best_meta, encoding="utf-8") as handle:
            meta = json.load(handle)
          if isinstance(meta, dict):
            best_meta_step = checkpoint_publication.coerce_step(
                meta.get("step")
            )
        except (OSError, ValueError):
          best_meta_step = None

      if _dir_has_publishable_weights(step_adapter):
        promoted_src = step_adapter
      elif _dir_has_publishable_weights(step_ckpt):
        promoted_src = step_ckpt
      elif (
          best_meta_step == target_step
          and _dir_has_publishable_weights(best_dir)
      ):
        promoted_src = best_dir

    if promoted_src is None and not _dir_has_publishable_weights(trial_dir):
      best_dir = os.path.join(trial_dir, BEST_CONTINUAL_CHECKPOINT_DIRNAME)
      if _dir_has_publishable_weights(best_dir):
        promoted_src = best_dir
      else:
        # Fall back to the highest-numbered step in adapters/ or checkpoint-*
        best_found_step = -1
        adapters_root = os.path.join(trial_dir, STEP_ADAPTERS_DIRNAME)
        if os.path.isdir(adapters_root):
          try:
            for entry in os.listdir(adapters_root):
              if entry.startswith("step-"):
                s_val = checkpoint_publication.coerce_step(entry[5:])
                cand = os.path.join(adapters_root, entry)
                if (
                    s_val is not None
                    and s_val > best_found_step
                    and _dir_has_publishable_weights(cand)
                ):
                  best_found_step = s_val
                  promoted_src = cand
                  promoted_step = s_val
          except OSError:
            pass
        if promoted_src is None:
          try:
            for entry in os.listdir(trial_dir):
              s_val = checkpoint_publication.step_from_checkpoint_dir(entry)
              cand = os.path.join(trial_dir, entry)
              if (
                  s_val is not None
                  and s_val > best_found_step
                  and _dir_has_publishable_weights(cand)
              ):
                best_found_step = s_val
                promoted_src = cand
                promoted_step = s_val
          except OSError:
            pass

    if promoted_src is not None:
      _copy_publishable_checkpoint_files(promoted_src, trial_dir)

    if promoted_step is not None:
      manifest_path = os.path.join(
          trial_dir, checkpoint_publication.CHECKPOINT_MANIFEST_FILENAME
      )
      if os.path.isfile(manifest_path):
        try:
          with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
          if isinstance(manifest, dict):
            manifest["default_step"] = int(promoted_step)
            ckpts = manifest.get("checkpoints")
            default_key = manifest.get("default") or "best"
            if isinstance(ckpts, dict) and isinstance(
                ckpts.get(default_key), dict
            ):
              ckpts[default_key]["step"] = int(promoted_step)
            with open(manifest_path, "w", encoding="utf-8") as handle:
              json.dump(manifest, handle, indent=2, sort_keys=True)
        except (OSError, ValueError):
          pass

    return trial_dir

  def prune_inferior_checkpoints(
      self, base_dir: str, keep_run_dir: str
  ) -> None:
    """Deletes losing trial checkpoint directories to conserve disk space."""
    if self.dry_run or not os.path.exists(base_dir):
      return

    try:
      for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path) and full_path != keep_run_dir:
          shutil.rmtree(full_path, ignore_errors=True)
          logger.info("Evicted sub-optimal checkpoint directory: %s", full_path)
    except Exception as e:
      logger.warning("Error while pruning checkpoints in %s: %s", base_dir, e)

  #: Metric ``--load_best_model_at_end`` ranks checkpoints on, per stage,
  #: as ``(metric_for_best_model, greater_is_better)``. These mirror the
  #: ``metric:`` block of the corresponding sweep YAML - the trial and the
  #: checkpoint inside it must be judged by the same quantity.
  BEST_MODEL_METRICS: Dict[str, Tuple[str, str]] = {
      "sft": ("loss", "False"),
      "rm": ("roc_auc", "True"),
      "perl": ("rewards/reward_fn/mean", "True"),
  }

  #: Eval/save cadence used when the caller does not supply one. None means
  #: "once per epoch", which is all the SFT retraining ever did.
  DEFAULT_EVAL_STEPS: Dict[str, Optional[int]] = {
      "sft": None,
      "rm": 50,
      "perl": 24,
  }

  def _checkpointing_flags(
      self,
      stage_name: str,
      checkpoint_policy: str,
      eval_steps: Optional[int],
  ) -> List[str]:
    """Builds the eval/save/best-model flags of a materialization run.

    Args:
      stage_name: 'sft', 'rm' or 'perl'.
      checkpoint_policy: 'best' or 'final'; see ``materialize_and_push``.
      eval_steps: Requested eval/save cadence in optimizer steps, or None
        for the stage default.

    Returns:
      The command-line flags, ready to extend the training command.

    Raises:
      ValueError: On an unknown ``checkpoint_policy``.
    """
    if checkpoint_policy not in ("best", "final"):
      raise ValueError(
          f"Unknown checkpoint_policy '{checkpoint_policy}'; expected 'best' "
          "or 'final'."
      )

    cadence = eval_steps
    if cadence is None:
      cadence = self.DEFAULT_EVAL_STEPS.get(stage_name)
    cadence = int(cadence) if cadence else None

    flags: List[str] = []
    if cadence:
      # ``load_best_model_at_end`` requires the two strategies to match and
      # save_steps to be a multiple of eval_steps; keeping them literally
      # equal is the only way that cannot silently drift.
      flags.extend([
          "--eval_strategy", "steps",
          "--eval_steps", str(cadence),
          "--save_strategy", "steps",
          "--save_steps", str(cadence),
      ])
    else:
      flags.extend([
          "--eval_strategy", "epoch",
          "--save_strategy", "epoch",
      ])

    # The best-model metric is declared in *both* policies, not just "best".
    # Without it TrainerState.best_model_checkpoint is never populated, and
    # the best checkpoint would be neither protected from save_total_limit
    # rotation nor publishable as the companion of the final one.
    metric, greater = self.BEST_MODEL_METRICS.get(stage_name, ("loss", "False"))
    flags.extend([
        "--metric_for_best_model", metric,
        "--greater_is_better", greater,
    ])

    if checkpoint_policy == "final":
      # The last checkpoint is what lands at the repository root. Evaluation
      # still runs and the best checkpoint is still tracked, archived and
      # published under the "best/" subfolder - it just does not become the
      # default the Hub hands out.
      flags.extend(["--load_best_model_at_end", "False"])
    else:
      flags.extend(["--load_best_model_at_end", "True"])
    return flags

  def build_materialization_command(
      self,
      stage_name: str,
      task_name: str,
      base_model: str,
      best_params: Dict[str, Any],
      seed: int = 130104,
      sft_model_path: Optional[str] = None,
      reward_model_path: Optional[str] = None,
      tunable_keys: Optional[Iterable[str]] = None,
      checkpoint_policy: str = "best",
      eval_steps: Optional[int] = None,
      timestamp: Optional[str] = None,
      flavor: Optional[str] = None,
      deepspeed_config: Optional[str] = None,
      memory_flags: Optional[Dict[str, str]] = None,
      continual_eval_flags: Optional[Dict[str, str]] = None,
  ) -> MaterializationPlan:
    """Builds the retraining command for a sweep's winning configuration.

    Kept free of side effects so the command can be asserted on. The training
    flags it emits must reproduce the trial that won the sweep: every flag the
    sweep YAML pins is pinned here to the same value, and only the
    checkpointing and publication flags differ (the sweep saves nothing, see
    ``--save_strategy=no`` in ``scripts/sweep_*.yaml``, except the checkpoints
    a PE-RL trial's continual evaluation pauses and resumes from). That
    correspondence is enforced by ``TestSweepMaterializationParity``; read its
    allowlists before adding or removing a flag below.

    Args:
        stage_name: 'sft', 'rm', or 'perl'.
        task_name: 'npov', 'bosch', etc.
        base_model: Foundation model repo ID.
        best_params: Winning hyperparameter dictionary from the sweep.
        seed: Random seed.
        sft_model_path: Required for PE-RL stage.
        reward_model_path: Required for PE-RL stage.
        tunable_keys: Keys of ``best_params`` that are genuine hyperparameters
          of the training script (normally the sweep's ``parameters:`` block).
        checkpoint_policy: See :meth:`materialize_and_push`.
        eval_steps: See :meth:`materialize_and_push`.
        timestamp: Overrides the ``%y%m%d%H%M`` stamp of the repo id; only
          meant for tests that need a stable name.
        deepspeed_config: ``accelerate launch --config_file`` argument. None
          keeps the manager's default. It must be the same profile the sweep
          ran under - see ``CampaignConfig.deepspeed_config_for`` - because a
          winner selected under ZeRO Stage 3 may simply not fit under Stage 2.
        memory_flags: Extra ``flag -> value`` training arguments the base
          model's size calls for (gradient checkpointing, today). The sweep
          passes the same ones; see ``BaseStage.apply_launcher_settings``.
        continual_eval_flags: ``flag -> value`` arguments that turn on
          continual autorater evaluation (PE-RL). The sweep trials ran with
          the same ones; see ``PerlStage.continual_eval_flags``.

    Returns:
        The :class:`MaterializationPlan` describing the run.

    Raises:
        ValueError: If the stage is unknown or PE-RL dependencies are missing.
    """
    allowed_keys = {str(k) for k in (tunable_keys or DEFAULT_TUNABLE_KEYS)}

    if timestamp is None:
      timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")
    model_short = base_model.split("/")[-1]

    # Compute clean repo identifier
    lr_val = float(best_params.get("learning_rate", 1e-4))
    formatted_lr = f"{lr_val:.1e}"
    epochs_val = float(best_params.get("num_train_epochs", 1))
    formatted_epochs = f"{epochs_val:.2g}"
    lora_r = int(best_params.get("lora_r", 8))
    # Which reward-model dataset this checkpoint belongs to. On an RM repo it
    # names the data it was trained on; on a PE-RL repo it names the reward
    # model that scored it. Without it, two branches of the same campaign
    # differ only by timestamp and become indistinguishable a week later.
    slug = flavors.flavor_slug(flavor)
    flavor_tag = f"{slug}_" if slug else ""
    # An unbranched campaign passes no flavor; the historical default is the
    # organic dataset, which is what every campaign trained on before the
    # choice existed.
    rm_flavor = flavor or flavors.ORGANIC

    if stage_name == "sft":
      details = (
          f"S{seed}_epo{formatted_epochs}_lr{formatted_lr}_r{lora_r}"
      )
      stage_tag = "SFT"
      script_path = "src/writer_sft.py"
    elif stage_name == "rm":
      details = (
          f"S{seed}_epo{formatted_epochs}_lr{formatted_lr}_r{lora_r}"
      )
      stage_tag = "RM"
      script_path = "src/reward_model.py"
    elif stage_name == "perl":
      beta_val = float(best_params.get("beta", 0.05))
      formatted_beta = (
          f"{beta_val:.2g}" if beta_val >= 0.001 else f"{beta_val:.1e}"
      )
      # From the pinned constant, never from best_params: the retrain below
      # passes REWARD_PENALTY_ALPHA, so reading a stale alpha out of an old
      # winner's W&B config here would put a suffix on the repo name that
      # the model inside does not match.
      alpha_suffix = (
          f"_a{REWARD_PENALTY_ALPHA}" if REWARD_PENALTY_ALPHA != 1.0 else ""
      )
      details = (
          f"S{seed}_epo{formatted_epochs}_lr{formatted_lr}_"
          f"beta{formatted_beta}_r{lora_r}{alpha_suffix}"
      )
      stage_tag = "PERL"
      script_path = "src/perl.py"
    else:
      raise ValueError(f"Unknown stage for materialization: {stage_name}")

    # SFT is shared by every branch of a campaign, so its checkpoint carries
    # no flavor: tagging it would claim a provenance it does not have.
    repo_id, shortened = naming.fit_model_repo_id(
        user=self.user,
        task=task_name,
        stage=stage_tag,
        model=model_short,
        details=details,
        timestamp=timestamp,
        flavor_slug="" if stage_name == "sft" else flavor_tag.rstrip("_"),
    )
    if shortened:
      logger.info(
          "Repo id for %s shortened to fit Hugging Face's 96-character "
          "limit: %s",
          stage_name,
          repo_id,
      )

    # Belt and braces: the fitter bounds the length, this bounds the alphabet
    # (periods, stray punctuation) and is a no-op on a well-formed name.
    repo_id = sanitize_hf_repo_id(repo_id)

    if self.dry_run:
      logger.info(
          "[DRY-RUN] Simulating materialization for %s -> %s",
          stage_name,
          repo_id,
      )

    output_dir = f"./checkpoints/{task_name}/{stage_name}/{repo_id}"


    cmd = [
        "accelerate",
        "launch",
        f"--config_file={deepspeed_config or self.deepspeed_config}",
        script_path,
        "--task_name",
        task_name,
        "--seed",
        str(seed),
        "--report_to",
        "wandb",
        "--run_name",
        repo_id,
        "--output_dir",
        output_dir,
        "--push_to_hub",
        "True",
        "--hub_model_id",
        repo_id,
        "--model_repo_id",
        base_model,
        "--do_train",
        "True",
        "--do_eval",
        "True",
        # Pinned rather than left to the HuggingFace default, because every
        # scripts/sweep_*.yaml pins it; a default that changes upstream would
        # otherwise silently make the retrain log on a different cadence than
        # the trials it is reproducing.
        "--logging_strategy",
        "steps",
        "--bf16",
        "True",
        "--save_total_limit",
        "2",
    ]

    # Checkpointing is decided in one place for all three stages: which
    # checkpoint is published, and how finely it can be located. Leaving it
    # scattered across the per-stage blocks is what let the SFT retraining
    # evaluate once per epoch while its sweep ranked trials on every-10-step
    # evals - the selected peak was then unreachable.
    cmd.extend(
        self._checkpointing_flags(stage_name, checkpoint_policy, eval_steps)
    )

    # Inject stage-specific flags and defaults matching scripts/*.sh and sweep configs
    if stage_name == "sft":
      cmd.extend([
          "--dataset_repo_id",
          f"{self.user}/{task_name}_sft",
          "--task_type",
          "CAUSAL_LM",
          "--peft_type",
          "LORA",
          "--per_device_train_batch_size",
          "16",
          "--per_device_eval_batch_size",
          "32",
          "--auto_find_batch_size",
          "True",
          "--lr_scheduler_type",
          "cosine",
          "--warmup_ratio",
          "0.1",
          "--logging_steps",
          "1",
          "--eval_on_start",
          "True",
      ])
    elif stage_name == "rm":
      cmd.extend([
          # Must match what RmStage handed the sweep: the winner is retrained
          # here from scratch, so a mismatch would publish a model trained on
          # a different dataset than the one the sweep ranked.
          "--dataset_repo_id",
          flavors.dataset_repo_id(self.user, task_name, rm_flavor),
          "--task_type",
          "SEQ_CLS",
          "--peft_type",
          "LORA",
          "--per_device_train_batch_size",
          "16",
          "--per_device_eval_batch_size",
          "32",
          "--auto_find_batch_size",
          "True",
          "--lr_scheduler_type",
          "cosine",
          "--warmup_ratio",
          "0.1",
          "--logging_steps",
          "1",
          "--eval_on_start",
          "True",
          "--num_organic_hallus_to_keep",
          "0",
          "--num_struct_hallus_to_keep",
          "0",
          # Must equal the value passed to the PE-RL stage below: this model
          # is trained here and queried there.
          "--reward_max_length",
          str(REWARD_MAX_LENGTH),
      ])
    elif stage_name == "perl":
      if not sft_model_path or not reward_model_path:
        if not self.dry_run:
          raise ValueError(
              "PE-RL stage requires both sft_model_path and reward_model_path"
          )
        # A rehearsal has no published checkpoints to point at. This used to
        # be unreachable because the dry-run short circuit came before the
        # command was built; keep the rehearsal printable rather than
        # failing it on a dependency that only a real run can have.
        sft_model_path = sft_model_path or f"{self.user}/DRY_RUN_sft"
        reward_model_path = reward_model_path or f"{self.user}/DRY_RUN_rm"
      cmd.extend([
          "--dataset_repo_id",
          f"{self.user}/{task_name}_perl",
          "--sft_model_path",
          sft_model_path,
          "--reward_model_path",
          reward_model_path,
          "--task_type",
          "CAUSAL_LM",
          "--peft_type",
          "LORA",
          # The schedule is NOT part of the PE-RL search space, so it is never
          # replayed from the winner's run config; it has to be pinned here to
          # the value every trial of scripts/sweep_perl.yaml used. Omitting it
          # silently retrained the winner on the HuggingFace defaults (linear
          # decay, no warmup) - a different optimisation problem from the one
          # that was searched, and at num_train_epochs=0.2 the warmup alone is
          # a large fraction of the run.
          "--lr_scheduler_type",
          "cosine",
          "--warmup_ratio",
          "0.1",
          "--min_lr_ratio",
          str(PERL_MIN_LR_RATIO),
          "--max_completion_length",
          "256",
          "--reward_max_length",
          str(REWARD_MAX_LENGTH),
          # Must equal the value scripts/sweep_perl.yaml pins for every
          # trial: it rescales the reward the trials were ranked on, so a
          # retrain that used a different one would not reproduce the run
          # that won. No longer replayed from the winner's config - see
          # REWARD_PENALTY_ALPHA.
          "--reward_penalty_alpha",
          str(REWARD_PENALTY_ALPHA),
      ])
      # Size-dependent, not experiment-dependent: see PERL_BATCH_GEOMETRY.
      # Passing the policy makes a large model shrink the micro-batch here
      # exactly as it did in the sweep, so the retrain reproduces the trial.
      for flag, value in perl_batch_geometry(policy_model=base_model).items():
        cmd.extend([f"--{flag}", value])
      cmd.extend([
          "--logging_steps",
          "1",
          "--log_completions",
          "True",
          "--eval_on_start",
          "True",
      ])

    # Memory settings the base model's size calls for. Appended rather than
    # folded into the per-stage blocks above because they are a property of
    # the checkpoint being trained, not of the stage training it - and skipped
    # when the stage already pins them, so an explicit value always wins.
    for flag, value in (memory_flags or {}).items():
      if f"--{flag}" not in cmd:
        cmd.extend([f"--{flag}", str(value)])

    # Continual evaluation, exactly as in the sweep trials, so the retrained
    # winner logs the same autorater curves as the trial that won.
    for flag, value in (continual_eval_flags or {}).items():
      if f"--{flag}" not in cmd:
        cmd.extend([f"--{flag}", str(value)])

    # Inject the winning hyperparameters (scalars only, allowlisted).
    #
    # The W&B run config of a Hugging Face Trainer run is a merge of the sweep
    # parameters, the full TrainingArguments dump and the model's config.json.
    # Only the declared tunables may be replayed on the command line.
    injected: Dict[str, Any] = {}
    skipped: List[str] = []
    for k, v in best_params.items():
      key = str(k)
      if key.startswith("_") or "wandb" in key.lower() or v is None:
        continue
      if isinstance(v, (dict, list, tuple, set, bool)):
        # Booleans are never part of a search space here, and forwarding them
        # would silently flip explicitly configured flags above.
        skipped.append(key)
        continue
      if key not in allowed_keys:
        skipped.append(key)
        continue
      flag = f"--{key}"
      if flag in cmd:
        if (
            key == "min_lr_ratio"
            and tunable_keys is not None
            and key in allowed_keys
        ):
          idx = cmd.index(flag)
          if idx + 1 < len(cmd):
            cmd[idx + 1] = str(v)
            injected[key] = v
        continue
      cmd.extend([flag, str(v)])
      injected[key] = v

    return MaterializationPlan(
        repo_id=repo_id,
        command=cmd,
        output_dir=output_dir,
        injected=injected,
        skipped=skipped,
    )

  def materialize_and_push(
      self,
      stage_name: str,
      task_name: str,
      base_model: str,
      best_params: Dict[str, Any],
      seed: int = 130104,
      sft_model_path: Optional[str] = None,
      reward_model_path: Optional[str] = None,
      live_line_callback: Optional[Callable[[str], None]] = None,
      tunable_keys: Optional[Iterable[str]] = None,
      stop_requested_callback: Optional[Callable[[], bool]] = None,
      checkpoint_policy: str = "best",
      eval_steps: Optional[int] = None,
      flavor: Optional[str] = None,
      deepspeed_config: Optional[str] = None,
      memory_flags: Optional[Dict[str, str]] = None,
      continual_eval_flags: Optional[Dict[str, str]] = None,
      winner_run_id: Optional[str] = None,
      winner_step: Optional[int] = None,
      checkpoints_base_dir: str = "./checkpoints",
  ) -> str:
    """Publishes the winning sweep trial to HF Hub, restoring from local disk when available or retraining.

    Args:
        stage_name: 'sft', 'rm', or 'perl'.
        task_name: 'npov', 'bosch', etc.
        base_model: Foundation model repo ID.
        best_params: Winning hyperparameter dictionary from the sweep.
        seed: Random seed.
        sft_model_path: Required for PE-RL stage.
        reward_model_path: Required for PE-RL stage.
        live_line_callback: Callback for streaming logs.
        tunable_keys: Keys of ``best_params`` that are genuine hyperparameters
          of the training script (normally the sweep's ``parameters:`` block).
          Everything else in the W&B run config is ignored, because it also
          contains the full ``TrainingArguments`` dump and the model config.
        stop_requested_callback: Predicate polled while training; when it
          returns True the training subprocess is terminated.
        checkpoint_policy: ``"best"`` publishes the best-scoring checkpoint
          (``--load_best_model_at_end``); ``"final"`` publishes the last one.
          This must agree with how the sweep ranked its trials - see
          ``SweepStageConfig.checkpoint_policy``.
        eval_steps: Eval/save cadence in optimizer steps. Only used under
          ``checkpoint_policy="best"``, where it bounds how finely the best
          checkpoint can be located; it should mirror the sweep YAML's
          ``--eval_steps``. None keeps the stage's historical cadence.
        deepspeed_config: The launcher the sweep ran under. None keeps the
          manager's default; see :meth:`build_materialization_command`.
        memory_flags: Size-derived training flags the sweep also passed.
        continual_eval_flags: Continual-evaluation flags the sweep also
          passed; see :meth:`build_materialization_command`.
        winner_run_id: Optional W&B run ID of the winning sweep trial. When a
          completed local checkpoint directory exists for this trial, its
          adapter weights are promoted and uploaded directly to Hugging Face Hub
          without launching a redundant retraining run.
        winner_step: Optional optimizer step of the winning checkpoint within
          the winning trial.
        checkpoints_base_dir: Root directory where sweep trials store local
          checkpoints (default ``'./checkpoints'``).

    Returns:
        The uploaded Hugging Face model repository ID.

    Raises:
        ValueError: If the stage is unknown or PE-RL dependencies are missing.
        RuntimeError: If the training subprocess fails or is interrupted
          before the checkpoint is pushed.
    """
    plan = self.build_materialization_command(
        stage_name=stage_name,
        task_name=task_name,
        base_model=base_model,
        best_params=best_params,
        seed=seed,
        sft_model_path=sft_model_path,
        reward_model_path=reward_model_path,
        tunable_keys=tunable_keys,
        checkpoint_policy=checkpoint_policy,
        eval_steps=eval_steps,
        flavor=flavor,
        deepspeed_config=deepspeed_config,
        memory_flags=memory_flags,
        continual_eval_flags=continual_eval_flags,
    )
    repo_id = plan.repo_id
    output_dir = plan.output_dir
    cmd = plan.command

    if self.dry_run:
      if live_line_callback:
        live_line_callback(
            f"[DRY-RUN] Training {stage_name} with best hyperparams..."
        )
        live_line_callback(
            f"[DRY-RUN] Checkpoint pushed to HuggingFace Hub: {repo_id}"
        )
      return repo_id

    if winner_run_id:
      trial_dir = self.find_completed_trial_checkpoint(
          task_name=task_name,
          stage_name=stage_name,
          run_id=winner_run_id,
          base_dir=checkpoints_base_dir,
      )
      if trial_dir is not None:
        self.prepare_trial_checkpoint_for_upload(
            trial_dir, winner_step=winner_step
        )
        coerced_step = checkpoint_publication.coerce_step(winner_step)
        step_note = (
            f" (step {coerced_step})" if coerced_step is not None else ""
        )
        logger.info(
            "Restoring winning %s checkpoint from local sweep trial %s%s "
            "at %s (skipping materialization retraining).",
            stage_name,
            winner_run_id,
            step_note,
            trial_dir,
        )
        if live_line_callback:
          live_line_callback(
              f"Restoring winning {stage_name} adapter from local trial "
              f"{winner_run_id}{step_note} ({trial_dir}) — skipping "
              "materialization retraining..."
          )
        commit_msg = (
            f"Publish winning {stage_name} sweep trial {winner_run_id}"
            f"{step_note}"
        )
        self.upload_local_checkpoint(
            trial_dir,
            repo_id,
            commit_message=commit_msg,
            ignore_patterns=TRIAL_CHECKPOINT_UPLOAD_IGNORE_PATTERNS,
        )
        logger.info(
            "Successfully published restored %s trial checkpoint to: %s",
            stage_name,
            repo_id,
        )
        return repo_id
      logger.info(
          "No completed local checkpoint found for %s trial %s; falling back "
          "to materialization retraining.",
          stage_name,
          winner_run_id,
      )

    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        "Materializing %s with hyperparameters %s (%d config keys ignored)",
        stage_name,
        plan.injected,
        len(plan.skipped),
    )
    if live_line_callback:
      live_line_callback(
          "Retraining winner with: "
          + ", ".join(f"{k}={v}" for k, v in sorted(plan.injected.items()))
      )
    logger.info("Launching materialization: %s", " ".join(cmd))

    def _train_once() -> None:
      outcome = stream_subprocess(
          cmd,
          on_line=live_line_callback,
          stop_requested=stop_requested_callback,
          stall_warning_s=self.robustness.stall_warning_minutes * 60,
      )
      if outcome.interrupted:
        raise RuntimeError(
            f"Materialization of the best {stage_name} model was interrupted "
            "before the checkpoint could be pushed."
        )
      if outcome.returncode != 0:
        # With continual evaluation the run stops at every evaluation step,
        # leaving a checkpoint behind each time. Those are intermediate
        # policies: publishing one as the trained model would be silently
        # wrong. Until training completes, fail (a retry resumes from the
        # last pause); after that, only the final model at the root of
        # output_dir may be salvaged.
        training_completed = continual_eval_training_completed(output_dir)
        if training_completed is False:
          raise RuntimeError(
              f"Materialization failed for {stage_name} with exit code "
              f"{outcome.returncode} before training completed; its "
              "continual-evaluation checkpoints are intermediate policies "
              "and are not published. A retry resumes from the last one."
          )
        # If the training completed and saved weights locally, but the push_to_hub
        # failed at the very end due to a transient error, salvage the checkpoint
        # by uploading the local folder directly with retries.
        candidate_dirs = [output_dir]
        # A finished continual-evaluation run's checkpoint-* folders are its
        # pause checkpoints, i.e. earlier policies, and only the root holds the
        # final one: they are salvage candidates only without continual
        # evaluation (no status file).
        if training_completed is None and os.path.exists(output_dir):
          try:
            for entry in os.listdir(output_dir):
              sub = os.path.join(output_dir, entry)
              if os.path.isdir(sub) and entry.startswith("checkpoint-"):
                candidate_dirs.append(sub)
          except OSError:
            pass

        for c_dir in candidate_dirs:
          try:
            files = os.listdir(c_dir)
          except OSError:
            continue
          if any(f.endswith((".safetensors", ".bin")) for f in files):
            logger.warning(
                "Materialization subprocess exited with %d, but valid checkpoint found at %s. "
                "Attempting direct upload to Hugging Face Hub with backoff...",
                outcome.returncode,
                c_dir,
            )
            if live_line_callback:
              live_line_callback(
                  f"[RETRY] Process exited with {outcome.returncode}; "
                  f"recovering checkpoint from {c_dir} via direct HF upload..."
              )
            try:
              self.upload_local_checkpoint(c_dir, repo_id)
              return
            except Exception as upload_err:
              logger.error("Direct checkpoint recovery upload failed: %s", upload_err)
              break

        raise RuntimeError(
            f"Materialization failed for {stage_name} with exit code "
            f"{outcome.returncode}"
        )

    # A transient Hub 5xx at the push step would otherwise discard the entire
    # sweep. Deterministic failures (bad flags) are not retried: the message
    # is matched against retry.NON_RETRYABLE_MARKERS, and "interrupted" is
    # explicitly one of them so a stop request takes effect at once.
    run_with_retries(
        _train_once,
        description=f"{stage_name} materialization",
        attempts=self.robustness.materialize_attempts,
        base_delay_s=self.robustness.retry_base_delay_s,
        max_delay_s=self.robustness.max_delay_s,
        on_notice=live_line_callback,
        stop_requested=stop_requested_callback,
    )

    logger.info("Successfully materialized and pushed model: %s", repo_id)
    return repo_id

