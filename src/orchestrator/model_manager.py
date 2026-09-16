"""Model and checkpoint management: local retention, materialization, and HF Hub publishing."""

from __future__ import annotations

import datetime
import logging
import os
import shutil
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


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
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "beta",
    "temperature",
    "reward_penalty_alpha",
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
      api.upload_folder(
          folder_path=folder_path,
          repo_id=repo_id,
          repo_type="model",
          commit_message=commit_message,
      )
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
      "perl": 50,
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
  ) -> str:
    """Executes a single training run with winning hyperparameters and pushes to HF Hub.

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

    Returns:
        The uploaded Hugging Face model repository ID.

    Raises:
        ValueError: If the stage is unknown or PE-RL dependencies are missing.
        RuntimeError: If the training subprocess fails or is interrupted
          before the checkpoint is pushed.
    """

    allowed_keys = {str(k) for k in (tunable_keys or DEFAULT_TUNABLE_KEYS)}

    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")
    model_short = base_model.split("/")[-1]

    # Compute clean repo identifier
    lr_val = float(best_params.get("learning_rate", 1e-4))
    formatted_lr = f"{lr_val:.1e}"
    epochs_val = float(best_params.get("num_train_epochs", 1))
    formatted_epochs = f"{epochs_val:.2g}"
    lora_r = int(best_params.get("lora_r", 8))

    if stage_name == "sft":
      repo_id = (
          f"{self.user}/{task_name}_SFT_{model_short}_S{seed}_"
          f"epo{formatted_epochs}_lr{formatted_lr}_r{lora_r}_{timestamp}"
      )
      script_path = "src/writer_sft.py"
    elif stage_name == "rm":
      repo_id = (
          f"{self.user}/{task_name}_RM_{model_short}_S{seed}_"
          f"epo{formatted_epochs}_lr{formatted_lr}_r{lora_r}_{timestamp}"
      )
      script_path = "src/reward_model.py"
    elif stage_name == "perl":
      beta_val = float(best_params.get("beta", 0.05))
      formatted_beta = (
          f"{beta_val:.2g}" if beta_val >= 0.001 else f"{beta_val:.1e}"
      )
      alpha_val = float(best_params.get("reward_penalty_alpha", 1.0))
      alpha_suffix = f"_a{alpha_val}" if alpha_val != 1.0 else ""
      repo_id = (
          f"{self.user}/{task_name}_PERL_{model_short}_S{seed}_"
          f"epo{formatted_epochs}_lr{formatted_lr}_beta{formatted_beta}_"
          f"r{lora_r}{alpha_suffix}_{timestamp}"
      )
      script_path = "src/perl.py"
    else:
      raise ValueError(f"Unknown stage for materialization: {stage_name}")

    # Enforce Hugging Face Hub constraints (length <= 96, no dots, valid characters)
    repo_id = sanitize_hf_repo_id(repo_id)

    if self.dry_run:
      logger.info("[DRY-RUN] Simulating materialization for %s -> %s", stage_name, repo_id)
      if live_line_callback:
        live_line_callback(f"[DRY-RUN] Training {stage_name} with best hyperparams...")
        live_line_callback(f"[DRY-RUN] Checkpoint pushed to HuggingFace Hub: {repo_id}")
      return repo_id

    output_dir = f"./checkpoints/{task_name}/{stage_name}/{repo_id}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "accelerate",
        "launch",
        f"--config_file={self.deepspeed_config}",
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
          "--dataset_repo_id",
          f"{self.user}/{task_name}_rm_organic",
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
        raise ValueError(
            "PE-RL stage requires both sft_model_path and reward_model_path"
        )
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
          "--max_completion_length",
          "256",
          "--reward_max_length",
          str(REWARD_MAX_LENGTH),
          "--num_generations",
          "8",
          "--num_iterations",
          "1",
          "--steps_per_generation",
          "16",
          "--per_device_train_batch_size",
          "4",
          "--per_device_eval_batch_size",
          "16",
          "--gradient_accumulation_steps",
          "8",
          "--auto_find_batch_size",
          "False",
          "--logging_steps",
          "1",
          "--log_completions",
          "True",
          "--eval_on_start",
          "True",
      ])

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
        continue
      cmd.extend([flag, str(v)])
      injected[key] = v

    logger.info(
        "Materializing %s with hyperparameters %s (%d config keys ignored)",
        stage_name,
        injected,
        len(skipped),
    )
    if live_line_callback:
      live_line_callback(
          f"Retraining winner with: "
          + ", ".join(f"{k}={v}" for k, v in sorted(injected.items()))
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
        # If the training completed and saved weights locally, but the push_to_hub
        # failed at the very end due to a transient error, salvage the checkpoint
        # by uploading the local folder directly with retries.
        candidate_dirs = [output_dir]
        if os.path.exists(output_dir):
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

