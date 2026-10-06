"""Tests for src.continual_eval: continual autorater evaluation in PE-RL."""

import dataclasses
import importlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import types
from typing import Any, Union, get_args, get_origin, get_type_hints
import unittest
from unittest.mock import MagicMock, call, patch


# Installs lightweight mocks when the GPU ML packages are not installed.
def _ensure_mock_modules() -> None:
  if "torch" not in sys.modules:
    torch_mod = types.ModuleType("torch")
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.float32 = "float32"
    torch_mod.cuda = MagicMock()
    torch_mod.cuda.is_available = MagicMock(return_value=False)
    torch_mod.device = MagicMock(return_value="cpu")
    torch_mod.no_grad = MagicMock()
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = object
    torch_mod.nn = nn_mod
    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod

  for mod_name in [
      "datasets",
      "evaluate",
      "google",
      "google.genai",
      "google.genai.types",
      "huggingface_hub",
      "peft",
      "scipy",
      "scipy.special",
      "sklearn",
      "sklearn.metrics",
      "tqdm",
      "trl",
      "vllm",
      "vllm.lora",
      "vllm.lora.request",
  ]:
    if mod_name not in sys.modules:
      sys.modules[mod_name] = MagicMock()

  if "transformers" not in sys.modules:
    tf_mod = types.ModuleType("transformers")

    class DummyTrainerCallback:
      pass

    class DummyTrainingArguments:

      def __init__(self, **kwargs):
        for k, v in kwargs.items():
          setattr(self, k, v)

    tf_mod.TrainerCallback = DummyTrainerCallback
    tf_mod.TrainingArguments = DummyTrainingArguments
    tf_mod.TrainerControl = MagicMock
    tf_mod.TrainerState = MagicMock
    tf_mod.AutoModelForCausalLM = MagicMock()
    tf_mod.AutoModelForSequenceClassification = MagicMock()
    tf_mod.AutoTokenizer = MagicMock()
    tf_mod.DataCollatorWithPadding = MagicMock()
    tf_mod.HfArgumentParser = MagicMock()
    tf_mod.LogitsProcessor = object
    tf_mod.LogitsProcessorList = list
    tf_mod.Trainer = MagicMock()
    tf_mod.set_seed = MagicMock()

    tf_utils = types.ModuleType("transformers.trainer_utils")
    tf_utils.get_last_checkpoint = MagicMock(return_value=None)
    tf_mod.trainer_utils = tf_utils

    sys.modules["transformers"] = tf_mod
    sys.modules["transformers.trainer_utils"] = tf_utils


_ensure_mock_modules()

from src import continual_eval as continual_eval_mod  # pylint: disable=g-import-not-at-top
from src.continual_eval import (  # pylint: disable=g-import-not-at-top
    CONTINUAL_EVAL_DEFAULTS,
    CONTINUAL_EVAL_HISTORY_FILENAME,
    CONTINUAL_EVAL_STATUS_FILENAME,
    CONTINUAL_EVAL_TIMEOUT_MINUTES,
    CONTINUAL_WORKER_ENV,
    HF_DEFAULT_WANDB_PROJECT,
    PAUSED_SUMMARY_KEY,
    TRL_DEFAULT_ROLLOUT_TEMPERATURE,
    ContinualEvalStatus,
    ContinualEvalStopCallback,
    build_continual_eval_commands,
    build_segment_command,
    compute_pareto_frontier,
    load_continual_eval_history,
    log_pareto_frontiers_to_wandb,
    parse_cli_flag_map,
    render_pareto_frontier_figure,
    run_perl_with_continual_eval,
    should_coordinate_continual_eval,
    update_continual_eval_history,
)
from src.metrics import GenerationMetricsEvaluator  # pylint: disable=g-import-not-at-top
from src.orchestrator import model_manager as orchestrator_model_manager  # pylint: disable=g-import-not-at-top
from src.orchestrator import sweep_controller as orchestrator_sweep_controller  # pylint: disable=g-import-not-at-top
from src.orchestrator.config import CampaignConfig, EvalStageConfig  # pylint: disable=g-import-not-at-top
from src.orchestrator.model_manager import ModelManager  # pylint: disable=g-import-not-at-top
from src.orchestrator.stages.base import CampaignContext  # pylint: disable=g-import-not-at-top
from src.orchestrator.stages.perl_stage import PerlStage  # pylint: disable=g-import-not-at-top
from src.orchestrator.state import CampaignState, StageResult, StageStatus  # pylint: disable=g-import-not-at-top
from src.orchestrator.sweep_controller import SweepController  # pylint: disable=g-import-not-at-top
from src.utils import (  # pylint: disable=g-import-not-at-top
    EvalArguments,
    ScriptArguments,
    build_eval_dataset_repo_id,
    looks_like_rl_checkpoint,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PERL_SCRIPT = os.path.join(_REPO_ROOT, "src", "perl.py")

#: `src.evaluator --mode score` flag carrying each continual-eval setting.
_SCORE_FLAG_OF_SETTING = {
    "seed": "--seed",
    "max_samples": "--max_eval_samples",
    "max_tokens": "--max_tokens",
    "evaluator_model": "--evaluator_model",
    "use_gemini": "--use_gemini",
    "num_fewshot": "--evaluator_num_fewshot",
    "autorater_num_samples": "--autorater_num_samples",
    "threshold": "--threshold",
    "run_reward_hacking": "--run_reward_hacking_autorater",
    "reward_hacking_num_fewshot": "--reward_hacking_num_fewshot",
    "reward_hacking_threshold": "--reward_hacking_threshold",
    "max_workers": "--max_workers",
    "batch_size": "--eval_batch_size",
    "compute_bertscore": "--compute_bertscore",
    "compute_perplexity": "--compute_perplexity",
}


def _flag_value(cmd: list[str], flag: str) -> str:
  """Returns the value that follows ``flag`` in ``cmd``."""
  return cmd[cmd.index(flag) + 1]


#: Values Hugging Face's ``string_to_bool`` accepts for a boolean flag.
_HF_BOOL_STRINGS = {
    "yes": True,
    "true": True,
    "t": True,
    "y": True,
    "1": True,
    "no": False,
    "false": False,
    "f": False,
    "n": False,
    "0": False,
}


def _parse_evaluator_command(cmd: list[str]) -> EvalArguments:
  """Parses an ``src.evaluator`` command like its ``HfArgumentParser`` would.

  ``transformers`` is mocked in this module, so this mirrors what the real
  parser enforces: every flag names an ``EvalArguments`` field, every value
  converts to that field's type (booleans as in ``string_to_bool``), and every
  field without a default is given.

  Args:
    cmd: ``[python, "-m", "src.evaluator", "--flag", "value", ...]``.

  Returns:
    The parsed arguments.

  Raises:
    AssertionError: If ``src.evaluator`` would reject ``cmd``.
  """
  if cmd[1:3] != ["-m", "src.evaluator"]:
    raise AssertionError(f"Not an evaluator command: {cmd}")
  argv = cmd[3:]
  if len(argv) % 2:
    raise AssertionError(f"Every flag needs exactly one value: {argv}")
  hints = get_type_hints(EvalArguments)
  values: dict[str, Any] = {}
  for flag, raw in zip(argv[::2], argv[1::2]):
    name = flag[2:] if flag.startswith("--") else flag
    if name not in hints:
      raise AssertionError(f"src.evaluator rejects {flag!r}")
    if name in values:
      raise AssertionError(f"{flag} is passed twice")
    field_type = hints[name]
    if get_origin(field_type) in (Union, types.UnionType):
      (field_type,) = [t for t in get_args(field_type) if t is not type(None)]
    if field_type is bool:
      if raw.lower() not in _HF_BOOL_STRINGS:
        raise AssertionError(f"{flag} expects a boolean, got {raw!r}")
      values[name] = _HF_BOOL_STRINGS[raw.lower()]
      continue
    try:
      values[name] = field_type(raw)
    except ValueError as e:
      raise AssertionError(
          f"{flag} expects {field_type.__name__}, got {raw!r}"
      ) from e
  missing = [
      f.name
      for f in dataclasses.fields(EvalArguments)
      if f.name not in values
      and f.default is dataclasses.MISSING
      and f.default_factory is dataclasses.MISSING
  ]
  if missing:
    raise AssertionError(f"src.evaluator requires {missing}")
  return EvalArguments(**values)


def _parse_perl_flag(name: str, raw: str) -> Any:
  """Converts ``--name raw`` like ``src/perl.py``'s ``HfArgumentParser`` would.

  Args:
    name: Flag name without the leading dashes.
    raw: Its value on the command line.

  Returns:
    The value of the ``ScriptArguments`` field.

  Raises:
    AssertionError: If a training segment would reject the flag.
  """
  hints = get_type_hints(ScriptArguments)
  if name not in hints:
    raise AssertionError(f"src/perl.py rejects --{name}")
  field_type = hints[name]
  if get_origin(field_type) in (Union, types.UnionType):
    (field_type,) = [t for t in get_args(field_type) if t is not type(None)]
  if field_type is bool:
    if raw.lower() not in _HF_BOOL_STRINGS:
      raise AssertionError(f"--{name} expects a boolean, got {raw!r}")
    return _HF_BOOL_STRINGS[raw.lower()]
  try:
    return field_type(raw)
  except ValueError as e:
    raise AssertionError(
        f"--{name} expects {field_type.__name__}, got {raw!r}"
    ) from e


def _is_training_segment(cmd: list[str]) -> bool:
  return _PERL_SCRIPT in cmd


def _stages(cmds: list[list[str]]) -> list[str]:
  """Names each command: "train", or the evaluator's --mode."""
  return [
      "train" if _is_training_segment(c) else _flag_value(c, "--mode")
      for c in cmds
  ]


def _write_adapter_checkpoint(checkpoint_dir: str) -> str:
  """Writes the files of a saved LoRA adapter to ``checkpoint_dir``."""
  os.makedirs(checkpoint_dir, exist_ok=True)
  for name, content in (
      ("adapter_config.json", '{"r": 8}'),
      ("adapter_model.safetensors", "weights"),
  ):
    with open(os.path.join(checkpoint_dir, name), "w", encoding="utf-8") as f:
      f.write(content)
  return checkpoint_dir


def _shell_defaults(path: str) -> dict[str, str]:
  """Parses the top-level ``NAME=value`` assignments of a shell script.

  ``NAME="${NAME:-value}"`` yields ``value``. The first assignment wins.

  Args:
    path: Shell script to parse.

  Returns:
    The default value of every assigned variable.
  """
  defaults: dict[str, str] = {}
  with open(path, encoding="utf-8") as f:
    for line in f:
      match = re.match(r"([A-Z][A-Z0-9_]*)=(.*)$", line.rstrip())
      if not match:
        continue
      name, value = match.groups()
      value = value.strip().strip('"').strip("'")
      fallback = re.fullmatch(r"\$\{" + name + r":-(.*)\}", value)
      if fallback:
        value = fallback.group(1)
      defaults.setdefault(name, value)
  return defaults


class MockDataset:
  """Minimal dataset mock for GenerationMetricsEvaluator tests."""

  def __init__(self, data: dict[str, list[Any]]):
    self._data = dict(data)
    self.column_names = list(self._data.keys())

  def __getitem__(self, key: str) -> list[Any]:
    return self._data[key]

  def __len__(self) -> int:
    if not self._data:
      return 0
    return len(next(iter(self._data.values())))


class TestContinualEvalStatusAndCallback(unittest.TestCase):
  """Tests for ContinualEvalStatus persistence and ContinualEvalStopCallback."""

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)
    super().tearDown()

  def test_status_round_trip(self):
    status = ContinualEvalStatus(
        paused_for_eval=True,
        training_completed=False,
        current_step=25,
        checkpoint_dir=os.path.join(self.temp_dir, "checkpoint-25"),
        evaluated_steps=[0],
        wandb_run_id="run_abc123",
    )
    saved_path = status.save(self.temp_dir)
    self.assertTrue(os.path.isfile(saved_path))
    self.assertEqual(
        os.path.basename(saved_path), CONTINUAL_EVAL_STATUS_FILENAME
    )

    loaded = ContinualEvalStatus.load(self.temp_dir)
    self.assertIsNotNone(loaded)
    self.assertTrue(loaded.paused_for_eval)
    self.assertFalse(loaded.training_completed)
    self.assertEqual(loaded.current_step, 25)
    self.assertEqual(
        loaded.checkpoint_dir, os.path.join(self.temp_dir, "checkpoint-25")
    )
    self.assertEqual(loaded.evaluated_steps, [0])
    self.assertEqual(loaded.wandb_run_id, "run_abc123")

  def test_status_load_tolerates_corrupt_files_and_unknown_fields(self):
    path = ContinualEvalStatus.status_path(self.temp_dir)
    with open(path, "w", encoding="utf-8") as f:
      f.write("{not json")
    self.assertEqual(
        ContinualEvalStatus.load(self.temp_dir), ContinualEvalStatus()
    )

    with open(path, "w", encoding="utf-8") as f:
      json.dump(
          {
              "current_step": 25,
              "evaluated_steps": [0, "25"],
              "field_of_a_newer_version": 1,
          },
          f,
      )
    loaded = ContinualEvalStatus.load(self.temp_dir)
    self.assertEqual(loaded.current_step, 25)
    self.assertEqual(loaded.evaluated_steps, [0, 25])

  def test_callback_stops_on_new_intermediate_step(self):
    cb = ContinualEvalStopCallback(evaluated_steps=[0])
    args = types.SimpleNamespace(output_dir=self.temp_dir)
    state = types.SimpleNamespace(
        global_step=0, max_steps=100, is_world_process_zero=True
    )
    control = types.SimpleNamespace(should_training_stop=False)

    cb.on_train_begin(args, state, control)
    self.assertEqual(cb.resumed_step, 0)

    # Step 25 checkpoint saved -> should trigger stop
    state.global_step = 25
    cb.on_save(args, state, control)
    self.assertTrue(control.should_training_stop)
    self.assertTrue(cb.stopped_for_continual_eval)
    self.assertEqual(cb.stopped_step, 25)

  def test_callback_ignores_resumed_step_and_already_evaluated_steps(self):
    cb = ContinualEvalStopCallback(evaluated_steps=[0, 25])
    args = types.SimpleNamespace(output_dir=self.temp_dir)
    state = types.SimpleNamespace(
        global_step=25, max_steps=100, is_world_process_zero=True
    )
    control = types.SimpleNamespace(should_training_stop=False)

    cb.on_train_begin(args, state, control)
    self.assertEqual(cb.resumed_step, 25)

    # Saving at the resumed step (25) must not re-stop
    cb.on_save(args, state, control)
    self.assertFalse(control.should_training_stop)
    self.assertFalse(cb.stopped_for_continual_eval)

    # Saving at step 50 -> new intermediate step -> stops
    state.global_step = 50
    cb.on_save(args, state, control)
    self.assertTrue(control.should_training_stop)
    self.assertTrue(cb.stopped_for_continual_eval)
    self.assertEqual(cb.stopped_step, 50)

  def test_callback_does_not_stop_early_at_max_steps(self):
    cb = ContinualEvalStopCallback(evaluated_steps=[0, 25, 50, 75])
    args = types.SimpleNamespace(output_dir=self.temp_dir)
    state = types.SimpleNamespace(
        global_step=75, max_steps=100, is_world_process_zero=True
    )
    control = types.SimpleNamespace(should_training_stop=False)

    cb.on_train_begin(args, state, control)
    state.global_step = 100
    cb.on_save(args, state, control)
    self.assertFalse(control.should_training_stop)
    self.assertFalse(cb.stopped_for_continual_eval)

  def test_callback_leaves_an_already_requested_stop_to_the_trainer(self):
    # Another callback (early stopping, the end of the last epoch, ...) already
    # ends training at this save: it is the end of the run, not a pause, so the
    # trainer's end-of-training flow (best model, Hub push) must still run.
    cb = ContinualEvalStopCallback(evaluated_steps=[0])
    args = types.SimpleNamespace(output_dir=self.temp_dir)
    state = types.SimpleNamespace(
        global_step=0, max_steps=100, is_world_process_zero=True
    )
    control = types.SimpleNamespace(should_training_stop=False)
    cb.on_train_begin(args, state, control)

    state.global_step = 25
    control.should_training_stop = True
    cb.on_save(args, state, control)
    self.assertTrue(control.should_training_stop)
    self.assertFalse(cb.stopped_for_continual_eval)
    self.assertIsNone(cb.stopped_step)


class TestParetoFrontierAndHistory(unittest.TestCase):
  """Tests for Pareto frontier computation, plotting, and WandB logging."""

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)
    super().tearDown()

  def test_pareto_frontier_minimize_both_rates(self):
    history = [
        {
            "step": 0,
            "hallucination_rate": 0.40,
            "reward_hacking_rate": 0.10,
            "reward_hacking_quality": 0.85,
        },
        {
            "step": 25,
            "hallucination_rate": 0.25,
            "reward_hacking_rate": 0.15,
            "reward_hacking_quality": 0.80,
        },
        {
            "step": 50,
            "hallucination_rate": 0.30,
            "reward_hacking_rate": 0.20,
            "reward_hacking_quality": 0.70,
        },  # Dominated by step 25
        {
            "step": 75,
            "hallucination_rate": 0.15,
            "reward_hacking_rate": 0.25,
            "reward_hacking_quality": 0.65,
        },
    ]

    frontier = compute_pareto_frontier(
        history,
        x_key="hallucination_rate",
        y_key="reward_hacking_rate",
        minimize_x=True,
        minimize_y=True,
    )
    frontier_steps = [pt["step"] for pt in frontier]
    # Sorted by hallucination_rate ascending: step 75 (0.15), step 25 (0.25),
    # step 0 (0.40).
    # Step 50 (0.30, 0.20) is strictly dominated by step 25 (0.25, 0.15).
    self.assertEqual(frontier_steps, [75, 25, 0])

  def test_pareto_frontier_minimize_hallucination_maximize_quality(self):
    history = [
        {
            "step": 0,
            "hallucination_rate": 0.40,
            "reward_hacking_quality": 0.88,
        },
        {
            "step": 25,
            "hallucination_rate": 0.22,
            "reward_hacking_quality": 0.82,
        },
        {
            "step": 50,
            "hallucination_rate": 0.28,
            "reward_hacking_quality": 0.75,
        },  # Dominated by step 25 (higher hallu AND lower quality)
        {
            "step": 75,
            "hallucination_rate": 0.12,
            "reward_hacking_quality": 0.70,
        },
    ]

    frontier = compute_pareto_frontier(
        history,
        x_key="hallucination_rate",
        y_key="reward_hacking_quality",
        minimize_x=True,
        minimize_y=False,
    )
    frontier_steps = [pt["step"] for pt in frontier]
    self.assertEqual(frontier_steps, [75, 25, 0])

  def test_pareto_frontier_keeps_tied_points(self):
    history = [
        {"step": 50, "hallucination_rate": 0.2, "reward_hacking_rate": 0.1},
        {"step": 25, "hallucination_rate": 0.2, "reward_hacking_rate": 0.1},
        # Same reward hacking rate, more hallucinations: dominated.
        {"step": 75, "hallucination_rate": 0.3, "reward_hacking_rate": 0.1},
        # No hallucination rate (e.g. the judge failed): ignored.
        {"step": 100, "hallucination_rate": None, "reward_hacking_rate": 0.0},
    ]
    frontier = compute_pareto_frontier(
        history, x_key="hallucination_rate", y_key="reward_hacking_rate"
    )
    # Identical points do not dominate each other: both are optimal, in step
    # order.
    self.assertEqual([p["step"] for p in frontier], [25, 50])

    history_path = os.path.join(self.temp_dir, CONTINUAL_EVAL_HISTORY_FILENAME)
    for point in history:
      updated = update_continual_eval_history(
          history_path, step=point["step"], summary=point
      )
    flags = {item["step"]: item["is_pareto_rate"] for item in updated}
    self.assertEqual(flags, {25: True, 50: True, 75: False, 100: False})

  def test_update_history_and_log_pareto_frontiers_to_wandb(self):
    history_path = os.path.join(self.temp_dir, CONTINUAL_EVAL_HISTORY_FILENAME)
    update_continual_eval_history(
        history_path,
        step=25,
        summary={
            "hallucination_rate": 0.25,
            "faithfulness_rate": 0.75,
            "reward_hacking_rate": 0.15,
            "reward_hacking_quality": 0.82,
        },
    )
    history = update_continual_eval_history(
        history_path,
        step=0,
        summary={
            "hallucination_rate": 0.40,
            "faithfulness_rate": 0.60,
            "reward_hacking_rate": 0.08,
            "reward_hacking_quality": 0.90,
        },
    )
    # History must be sorted by step ascending
    self.assertEqual([h["step"] for h in history], [0, 25])
    loaded = load_continual_eval_history(history_path)
    self.assertEqual([h["step"] for h in loaded], [0, 25])

    # Render figure
    png_path = os.path.join(self.temp_dir, "pareto_test.png")
    rendered = render_pareto_frontier_figure(
        history,
        x_key="hallucination_rate",
        y_key="reward_hacking_rate",
        minimize_x=True,
        minimize_y=True,
        title="Test Pareto Frontier",
        x_label="Hallucination Rate (lower is better)",
        y_label="Reward Hacking Rate (lower is better)",
        output_path=png_path,
    )
    if rendered is not None:
      self.assertTrue(os.path.isfile(png_path))

    # Log to mocked WandB
    mock_wandb = MagicMock()
    mock_wandb.summary = {}
    mock_wandb.Table = MagicMock(side_effect=lambda columns, data: {
        "columns": columns,
        "data": data,
    })
    mock_wandb.Image = MagicMock(side_effect=lambda p, caption=None: {
        "path": p,
        "caption": caption,
    })
    mock_wandb.plot = MagicMock()
    mock_wandb.plot.scatter = MagicMock(return_value="scatter_plot_obj")

    log_pareto_frontiers_to_wandb(
        mock_wandb,
        history,
        output_dir=self.temp_dir,
        eval_step=25,
    )
    mock_wandb.log.assert_called_once()
    logged_payload = mock_wandb.log.call_args[0][0]
    self.assertEqual(logged_payload["train/global_step"], 25)
    self.assertIn("eval/continual_pareto_table", logged_payload)
    self.assertIn(
        "eval/pareto_hallucination_vs_reward_hacking_rate",
        logged_payload,
    )
    self.assertIn(
        "eval/pareto_hallucination_vs_reward_hacking_quality",
        logged_payload,
    )
    self.assertEqual(logged_payload["eval/pareto_rate_frontier_count"], 2)
    self.assertEqual(logged_payload["eval/pareto_quality_frontier_count"], 2)
    self.assertAlmostEqual(
        logged_payload["eval/pareto_best_hallucination_rate"], 0.25
    )
    self.assertAlmostEqual(
        logged_payload["eval/pareto_best_reward_hacking_rate"], 0.08
    )
    self.assertAlmostEqual(
        logged_payload["eval/pareto_best_reward_hacking_quality"], 0.90
    )


class TestMetricsContinualEvalIntegration(unittest.TestCase):
  """Tests the continual-eval support of save_and_log_results."""

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)
    super().tearDown()

  def _dataset(self) -> MockDataset:
    return MockDataset({
        "prompt": ["Prompt 1", "Prompt 2"],
        "completion": ["Completion 1", "Completion 2"],
        "token_length": [12, 15],
        "distinct_2": [0.9, 0.85],
        "repetition_rate": [0.05, 0.1],
    })

  def _evaluator(self) -> GenerationMetricsEvaluator:
    return GenerationMetricsEvaluator(
        compute_bertscore_metric=False,
        compute_perplexity_metric=False,
    )

  @patch("src.metrics.wandb")
  def test_save_and_log_results_resumes_wandb_run_and_logs_pareto(
      self, mock_wandb
  ):
    mock_wandb.run = None
    mock_wandb.summary = {}
    mock_wandb.Table = MagicMock(return_value="mock_table")
    mock_wandb.Image = MagicMock(return_value="mock_image")
    mock_wandb.plot = MagicMock()
    mock_wandb.plot.scatter = MagicMock(return_value="mock_scatter")

    dataset = self._dataset()
    summary = {
        "reward_hacking_rate": 0.12,
        "reward_hacking_quality": 0.84,
    }
    history_path = os.path.join(self.temp_dir, CONTINUAL_EVAL_HISTORY_FILENAME)

    evaluator = self._evaluator()
    evaluator.save_and_log_results(
        dataset,
        summary,
        output_dir=self.temp_dir,
        dataset_name="continual_step_25",
        autorater_scores=[0.9, 0.2],
        threshold=0.5,
        log_to_wandb=True,
        wandb_project="perl_proj",
        wandb_run_name="npov_PERL_run",
        wandb_run_id="perl_run_123",
        wandb_entity="test_entity",
        eval_step=25,
        continual_eval_history_path=history_path,
    )

    mock_wandb.init.assert_called_once_with(
        project="perl_proj",
        name="npov_PERL_run",
        entity="test_entity",
        id="perl_run_123",
        resume="allow",
    )
    mock_wandb.define_metric.assert_any_call("train/global_step")
    mock_wandb.define_metric.assert_any_call(
        "eval/*", step_metric="train/global_step"
    )
    # Verify continual_eval_history.json was written with step 25
    history = load_continual_eval_history(history_path)
    self.assertEqual(len(history), 1)
    self.assertEqual(history[0]["step"], 25)
    self.assertAlmostEqual(history[0]["hallucination_rate"], 0.5)
    self.assertAlmostEqual(history[0]["reward_hacking_rate"], 0.12)
    self.assertAlmostEqual(history[0]["reward_hacking_quality"], 0.84)
    mock_wandb.finish.assert_called_once()

  @patch("src.metrics.wandb")
  def test_attaching_to_the_perl_run_keeps_its_name_and_config(
      self, mock_wandb
  ):
    mock_wandb.run = None
    mock_wandb.summary = {}
    history_path = os.path.join(self.temp_dir, CONTINUAL_EVAL_HISTORY_FILENAME)

    with patch(
        "src.continual_eval.log_pareto_frontiers_to_wandb"
    ) as mock_log_pareto:
      self._evaluator().save_and_log_results(
          self._dataset(),
          {"reward_hacking_rate": 0.1, "reward_hacking_quality": 0.8},
          output_dir=os.path.join(self.temp_dir, "scores"),
          dataset_name="npov_checkpoint-25_completions",
          autorater_scores=[0.9, 0.05],
          threshold=0.1025,
          log_to_wandb=True,
          wandb_project="huggingface",
          wandb_run_id="perl_run_123",
          eval_step=25,
          continual_eval_history_path=history_path,
      )

    # No `name`: the dataset name would otherwise rename the PE-RL run.
    mock_wandb.init.assert_called_once_with(
        project="huggingface", id="perl_run_123", resume="allow"
    )
    mock_log_pareto.assert_called_once()
    # The Pareto PNGs go next to the history (the PE-RL output directory), not
    # to a directory relative to the evaluator's working directory.
    self.assertEqual(
        mock_log_pareto.call_args.kwargs["output_dir"],
        os.path.dirname(os.path.abspath(history_path)),
    )
    self.assertEqual(mock_log_pareto.call_args.kwargs["eval_step"], 25)
    mock_wandb.finish.assert_called_once()

  @patch("src.metrics.wandb")
  def test_standalone_evaluation_run_is_named_after_the_dataset(
      self, mock_wandb
  ):
    mock_wandb.run = None
    mock_wandb.summary = {}

    self._evaluator().save_and_log_results(
        self._dataset(),
        {},
        output_dir=self.temp_dir,
        dataset_name="npov_final_completions",
        autorater_scores=[0.9],
        threshold=0.5,
        log_to_wandb=True,
        wandb_project="new_perl_eval",
    )

    mock_wandb.init.assert_called_once_with(
        project="new_perl_eval",
        name="npov_final_completions",
        config={"dataset_name": "npov_final_completions", "threshold": 0.5},
    )
    mock_wandb.finish.assert_not_called()


class TestCheckpointHelpersAndCoordinator(unittest.TestCase):
  """Tests checkpoint-N detection, repo ID naming, and the coordinator loop."""

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)
    super().tearDown()

  def test_looks_like_rl_checkpoint_on_local_checkpoint_subdirs(self):
    self.assertTrue(
        looks_like_rl_checkpoint(
            "./checkpoints/npov/perl/npov_PERL_gemma_S130104/checkpoint-25"
        )
    )
    self.assertTrue(
        looks_like_rl_checkpoint(
            "/tmp/checkpoints/ragtruth/perl/my_run/checkpoint-0"
        )
    )
    self.assertFalse(
        looks_like_rl_checkpoint(
            "./checkpoints/npov/sft/npov_SFT_gemma_S130104/checkpoint-25"
        )
    )
    self.assertFalse(
        looks_like_rl_checkpoint(
            "/home/leobianco/new_perl/checkpoints/npov/sft/my_run/"
            "checkpoint-50"
        )
    )

  def test_build_eval_dataset_repo_id_distinguishes_checkpoint_steps(self):
    repo_0 = build_eval_dataset_repo_id(
        user="leobianco",
        writer_model_lora=(
            "./checkpoints/npov/perl/npov_PERL_gemma_S130104/checkpoint-0"
        ),
        temperature=0.7,
        writer_num_fewshot=0,
        task_name="npov",
        sft_model_path="leobianco/npov_SFT_gemma",
    )
    repo_25 = build_eval_dataset_repo_id(
        user="leobianco",
        writer_model_lora=(
            "./checkpoints/npov/perl/npov_PERL_gemma_S130104/checkpoint-25"
        ),
        temperature=0.7,
        writer_num_fewshot=0,
        task_name="npov",
        sft_model_path="leobianco/npov_SFT_gemma",
    )
    self.assertNotEqual(repo_0, repo_25)
    self.assertLessEqual(len(repo_0), 96)
    self.assertLessEqual(len(repo_25), 96)

  def test_build_continual_eval_commands_uses_rollout_temperature_and_sft_stack(
      self,
  ):
    cli_args = [
        "--task_name",
        "npov",
        "--dataset_repo_id",
        "leobianco/npov_perl",
        "--model_repo_id",
        "google/gemma-4-E4B-it",
        "--sft_model_path",
        "leobianco/npov_SFT_gemma",
        "--output_dir",
        self.temp_dir,
        "--temperature",
        "0.7",
        "--continual_eval",
        "True",
        "--continual_eval_threshold",
        "0.125",
    ]
    flags = parse_cli_flag_map(cli_args)
    ckpt_dir = os.path.join(self.temp_dir, "checkpoint-25")
    status = ContinualEvalStatus(
        paused_for_eval=True,
        training_completed=False,
        current_step=25,
        checkpoint_dir=ckpt_dir,
        wandb_run_id="wandb_xyz",
        rollout_temperature=0.7,
    )
    gen_cmd, score_cmd = build_continual_eval_commands(
        flags=flags,
        status=status,
        output_dir=self.temp_dir,
    )
    self.assertIn("--mode", gen_cmd)
    self.assertEqual(gen_cmd[gen_cmd.index("--mode") + 1], "generate")
    self.assertEqual(
        gen_cmd[gen_cmd.index("--writer_model_lora") + 1], ckpt_dir
    )
    self.assertEqual(
        gen_cmd[gen_cmd.index("--sft_model_path") + 1],
        "leobianco/npov_SFT_gemma",
    )
    self.assertEqual(gen_cmd[gen_cmd.index("--temperature") + 1], "0.7")
    self.assertEqual(
        gen_cmd[gen_cmd.index("--dataset_prompts") + 1],
        "leobianco/npov_final_test_set",
    )

    self.assertEqual(score_cmd[score_cmd.index("--mode") + 1], "score")
    self.assertEqual(
        score_cmd[score_cmd.index("--wandb_run_id") + 1], "wandb_xyz"
    )
    self.assertEqual(score_cmd[score_cmd.index("--eval_step") + 1], "25")
    self.assertEqual(score_cmd[score_cmd.index("--threshold") + 1], "0.125")
    self.assertEqual(
        score_cmd[score_cmd.index("--run_reward_hacking_autorater") + 1],
        "True",
    )

  def test_should_coordinate_continual_eval(self):
    args = [
        "--task_name", "npov", "--do_train", "True", "--continual_eval", "True"
    ]
    self.assertTrue(should_coordinate_continual_eval(args, env={}))
    self.assertFalse(
        should_coordinate_continual_eval(
            args, env={"PERL_CONTINUAL_EVAL_WORKER": "1"}
        )
    )
    self.assertFalse(
        should_coordinate_continual_eval(
            ["--do_train", "True", "--continual_eval", "False"], env={}
        )
    )
    # `TrainingArguments.do_train` defaults to False, and PERL never trains
    # without it: there is nothing to coordinate.
    self.assertFalse(
        should_coordinate_continual_eval(
            ["--task_name", "npov", "--continual_eval", "True"], env={}
        )
    )
    self.assertTrue(
        should_coordinate_continual_eval(
            ["--continual_eval=true", "--do_train=1"], env={}
        )
    )

  def test_run_perl_with_continual_eval_end_to_end_loop(self):
    cli_args = [
        "--task_name",
        "npov",
        "--dataset_repo_id",
        "leobianco/npov_perl",
        "--model_repo_id",
        "google/gemma-4-E4B-it",
        "--sft_model_path",
        "leobianco/npov_SFT_gemma",
        "--output_dir",
        self.temp_dir,
        "--temperature",
        "0.7",
        "--do_train",
        "True",
        "--continual_eval",
        "True",
    ]
    env = {
        "PATH": "/usr/bin",
        "GEMINI_API_KEY": "secret-key",
        "RANK": "0",
        "WORLD_SIZE": "1",
    }

    planned_steps = [(0, False), (25, False), (50, True)]
    train_segment_idx = [0]
    executed_cmds: list[list[str]] = []

    def fake_runner(cmd, sub_env):
      executed_cmds.append(list(cmd))
      self.assertNotIn("secret-key", " ".join(cmd))
      if _is_training_segment(cmd):
        self.assertEqual(sub_env.get(CONTINUAL_WORKER_ENV), "1")
        self.assertNotIn("RANK", sub_env)
        self.assertNotIn("WORLD_SIZE", sub_env)
        step, is_done = planned_steps[train_segment_idx[0]]
        train_segment_idx[0] += 1
        if step == 50:
          # Resumes from the last paused (and scored) checkpoint.
          self.assertEqual(
              _flag_value(cmd, "--resume_from_checkpoint"),
              os.path.join(self.temp_dir, "checkpoint-25"),
          )
        else:
          # checkpoint-0 holds no optimizer state: segment 2 starts fresh.
          self.assertNotIn("--resume_from_checkpoint", cmd)
        if step > 0:
          # Later segments resume the WandB run the first segment created.
          self.assertEqual(sub_env.get("WANDB_RUN_ID"), "run_continual_test")
          self.assertEqual(sub_env.get("WANDB_RESUME"), "allow")
          self.assertEqual(sub_env.get("WANDB_PROJECT"), "perl_project")
        else:
          self.assertNotIn("WANDB_RUN_ID", sub_env)
        ckpt_dir = _write_adapter_checkpoint(
            os.path.join(self.temp_dir, f"checkpoint-{step}")
        )
        # What rank 0 of the segment records when it pauses / completes.
        status = ContinualEvalStatus.load(self.temp_dir)
        status.paused_for_eval = True
        status.training_completed = is_done
        status.current_step = step
        status.checkpoint_dir = ckpt_dir
        status.rollout_temperature = 0.7
        status.wandb_run_id = "run_continual_test"
        status.wandb_project = "perl_project"
        status.save(self.temp_dir)
        return 0
      else:
        # src.evaluator --mode generate or --mode score
        self.assertIn("src.evaluator", cmd)
        self.assertNotIn(CONTINUAL_WORKER_ENV, sub_env)
        # Credentials reach the judge through the environment only.
        self.assertEqual(sub_env.get("GEMINI_API_KEY"), "secret-key")
        step = planned_steps[train_segment_idx[0] - 1][0]
        self.assertEqual(
            _flag_value(cmd, "--writer_model_lora"),
            os.path.join(self.temp_dir, f"checkpoint-{step}"),
        )
        self.assertEqual(_flag_value(cmd, "--temperature"), "0.7")
        if _flag_value(cmd, "--mode") == "score":
          self.assertEqual(_flag_value(cmd, "--eval_step"), str(step))
          self.assertEqual(
              _flag_value(cmd, "--wandb_run_id"), "run_continual_test"
          )
          self.assertEqual(_flag_value(cmd, "--wandb_project"), "perl_project")
        return 0

    rc = run_perl_with_continual_eval(
        cli_args,
        env=env,
        runner=fake_runner,
    )
    self.assertEqual(rc, 0)
    # 3 training segments + 3 * (generate + score) = 9 subprocess invocations
    self.assertEqual(len(executed_cmds), 9)
    self.assertEqual(
        _stages(executed_cmds),
        ["train", "generate", "score"] * 3,
    )
    final_status = ContinualEvalStatus.load(self.temp_dir)
    self.assertIsNotNone(final_status)
    self.assertTrue(final_status.training_completed)
    self.assertFalse(final_status.paused_for_eval)
    self.assertEqual(final_status.evaluated_steps, [0, 25, 50])
    self.assertEqual(final_status.segment_idx, 3)


class TestBuildContinualEvalCommands(unittest.TestCase):
  """Tests the generate / score commands built for a paused checkpoint."""

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.mkdtemp()
    self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoint-25")

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)
    super().tearDown()

  def _build(self, extra_args=(), env=None, **status_fields):
    argv = [
        "--task_name",
        "npov",
        "--dataset_repo_id",
        "leobianco/npov_perl",
        "--output_dir",
        self.temp_dir,
        *extra_args,
    ]
    fields = {
        "paused_for_eval": True,
        "current_step": 25,
        "checkpoint_dir": self.checkpoint_dir,
    }
    fields.update(status_fields)
    return build_continual_eval_commands(
        flags=parse_cli_flag_map(argv),
        status=ContinualEvalStatus(**fields),
        output_dir=self.temp_dir,
        env={} if env is None else env,
    )

  def test_judge_settings_default_to_the_final_evaluation(self):
    self.assertEqual(set(_SCORE_FLAG_OF_SETTING), set(CONTINUAL_EVAL_DEFAULTS))
    gen_cmd, score_cmd = self._build()
    for key, flag in _SCORE_FLAG_OF_SETTING.items():
      with self.subTest(setting=key):
        self.assertEqual(
            _flag_value(score_cmd, flag), CONTINUAL_EVAL_DEFAULTS[key]
        )
    for flag in ("--seed", "--max_eval_samples", "--max_tokens"):
      with self.subTest(flag=flag):
        self.assertEqual(
            _flag_value(gen_cmd, flag), _flag_value(score_cmd, flag)
        )
    self.assertEqual(_flag_value(score_cmd, "--threshold"), "0.1025")
    self.assertEqual(_flag_value(score_cmd, "--compute_perplexity"), "False")

  def test_continual_eval_flags_override_the_defaults(self):
    overrides = {
        "seed": "7",
        "max_samples": "50",
        "max_tokens": "128",
        "evaluator_model": "gemini-2.5-pro",
        "use_gemini": "False",
        "num_fewshot": "4",
        "autorater_num_samples": "3",
        "threshold": "0.2",
        "run_reward_hacking": "False",
        "reward_hacking_num_fewshot": "1",
        "reward_hacking_threshold": "0.5",
        "max_workers": "8",
        "batch_size": "16",
        "compute_bertscore": "False",
        "compute_perplexity": "True",
    }
    self.assertEqual(set(overrides), set(CONTINUAL_EVAL_DEFAULTS))
    extra_args = []
    for key, value in overrides.items():
      extra_args.extend([f"--continual_eval_{key}", value])
    gen_cmd, score_cmd = self._build(extra_args=extra_args)
    for key, flag in _SCORE_FLAG_OF_SETTING.items():
      with self.subTest(setting=key):
        self.assertEqual(_flag_value(score_cmd, flag), overrides[key])
    self.assertEqual(_flag_value(gen_cmd, "--max_tokens"), "128")
    self.assertEqual(_flag_value(gen_cmd, "--max_eval_samples"), "50")

  def test_scores_attach_to_wandb_only_with_a_recorded_run(self):
    env = {"WANDB_PROJECT": "env_project", "WANDB_ENTITY": "env_entity"}
    _, score_cmd = self._build(env=env)
    self.assertEqual(_flag_value(score_cmd, "--log_to_wandb"), "False")
    self.assertNotIn("--wandb_run_id", score_cmd)
    self.assertNotIn("--wandb_project", score_cmd)

    _, score_cmd = self._build(
        env=env,
        wandb_run_id="run123",
        wandb_project="perl_project",
        wandb_entity="perl_entity",
    )
    self.assertEqual(_flag_value(score_cmd, "--log_to_wandb"), "True")
    self.assertEqual(_flag_value(score_cmd, "--wandb_run_id"), "run123")
    self.assertEqual(_flag_value(score_cmd, "--wandb_project"), "perl_project")
    self.assertEqual(_flag_value(score_cmd, "--wandb_entity"), "perl_entity")

    _, score_cmd = self._build(env=env, wandb_run_id="run123")
    self.assertEqual(_flag_value(score_cmd, "--wandb_project"), "env_project")
    self.assertEqual(_flag_value(score_cmd, "--wandb_entity"), "env_entity")

    # Hugging Face's WandbCallback logs to "huggingface" without WANDB_PROJECT.
    _, score_cmd = self._build(wandb_run_id="run123")
    self.assertEqual(
        _flag_value(score_cmd, "--wandb_project"), HF_DEFAULT_WANDB_PROJECT
    )
    self.assertNotIn("--wandb_entity", score_cmd)

  def test_no_secret_and_no_run_name_on_the_command_line(self):
    gen_cmd, score_cmd = self._build(
        env={"GEMINI_API_KEY": "super-secret-key"},
        wandb_run_id="run123",
        wandb_run_name="npov_PERL_run",
    )
    for cmd in (gen_cmd, score_cmd):
      self.assertNotIn("--gemini_api_key", cmd)
      self.assertNotIn("super-secret-key", " ".join(cmd))
    # Scoring attaches to the PE-RL run by id and must not rename it.
    self.assertNotIn("--wandb_run_name", score_cmd)

  def test_rollout_temperature_prefers_the_recorded_one(self):
    gen_cmd, score_cmd = self._build(
        extra_args=["--temperature", "0.7"], rollout_temperature=0.9
    )
    self.assertEqual(_flag_value(gen_cmd, "--temperature"), "0.9")
    # The scorer derives the completions dataset name from the temperature.
    self.assertEqual(_flag_value(score_cmd, "--temperature"), "0.9")

    gen_cmd, _ = self._build(extra_args=["--temperature", "0.7"])
    self.assertEqual(_flag_value(gen_cmd, "--temperature"), "0.7")

    gen_cmd, _ = self._build()
    self.assertEqual(
        _flag_value(gen_cmd, "--temperature"),
        str(TRL_DEFAULT_ROLLOUT_TEMPERATURE),
    )

  def test_missing_sft_adapter_is_allowed_explicitly(self):
    for cmd in self._build():
      self.assertNotIn("--sft_model_path", cmd)
      self.assertEqual(_flag_value(cmd, "--allow_missing_sft_adapter"), "True")

  def test_optional_reward_hacking_model_and_context_window(self):
    gen_cmd, score_cmd = self._build()
    self.assertNotIn("--reward_hacking_model", score_cmd)
    self.assertNotIn("--max_model_len", gen_cmd)

    gen_cmd, score_cmd = self._build(
        extra_args=[
            "--continual_eval_reward_hacking_model",
            "gemini-2.5-pro",
            "--continual_eval_max_model_len",
            "8192",
        ]
    )
    self.assertEqual(
        _flag_value(score_cmd, "--reward_hacking_model"), "gemini-2.5-pro"
    )
    self.assertEqual(_flag_value(gen_cmd, "--max_model_len"), "8192")

  def test_scores_land_in_the_history_of_the_output_dir(self):
    _, score_cmd = self._build()
    self.assertEqual(
        _flag_value(score_cmd, "--continual_eval_history_path"),
        os.path.join(self.temp_dir, CONTINUAL_EVAL_HISTORY_FILENAME),
    )
    self.assertEqual(_flag_value(score_cmd, "--eval_step"), "25")

  def test_requires_a_checkpoint(self):
    with self.assertRaises(ValueError):
      self._build(checkpoint_dir=None)

  def test_the_evaluator_accepts_every_flag(self):
    with_every_option = {
        "extra_args": [
            "--sft_model_path",
            "leobianco/npov_sft",
            "--continual_eval_reward_hacking_model",
            "gemini-2.5-pro",
            "--continual_eval_max_model_len",
            "8192",
        ],
        "wandb_run_id": "run123",
        "wandb_project": "perl_project",
        "wandb_entity": "perl_entity",
    }
    for variant, kwargs in (
        ("defaults", {}),
        ("every_option", with_every_option),
    ):
      gen_cmd, score_cmd = self._build(**kwargs)
      for mode, cmd in (("generate", gen_cmd), ("score", score_cmd)):
        with self.subTest(variant=variant, mode=mode):
          self.assertEqual(_parse_evaluator_command(cmd).mode, mode)

  def test_generation_and_scoring_agree_on_the_completions_dataset(self):
    # The scorer finds the completions by re-deriving the dataset name that
    # the generator pushed them to.
    for extra_args in ([], ["--sft_model_path", "leobianco/npov_sft"]):
      with self.subTest(extra_args=extra_args):
        repo_ids = set()
        for cmd in self._build(extra_args=extra_args, rollout_temperature=0.7):
          args = _parse_evaluator_command(cmd)
          repo_ids.add(
              build_eval_dataset_repo_id(
                  user=args.user,
                  writer_model_lora=args.writer_model_lora,
                  temperature=args.temperature,
                  writer_num_fewshot=args.writer_num_fewshot,
                  task_name=args.task_name,
                  sft_model_path=args.sft_model_path,
                  seed=args.seed,
                  max_tokens=args.max_tokens,
              )
          )
        self.assertEqual(len(repo_ids), 1, repo_ids)


class TestContinualEvalCoordinator(unittest.TestCase):
  """Tests the stop -> evaluate -> resume loop of the coordinator."""

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.mkdtemp()
    self.argv = [
        "--task_name",
        "npov",
        "--dataset_repo_id",
        "leobianco/npov_perl",
        "--output_dir",
        self.temp_dir,
        "--temperature",
        "0.7",
        "--do_train",
        "True",
        "--continual_eval",
        "True",
    ]

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)
    super().tearDown()

  def _launch_config(self) -> str:
    path = os.path.join(self.temp_dir, "deepspeed_config.yaml")
    with open(path, "w", encoding="utf-8") as f:
      f.write("distributed_type: DEEPSPEED\nnum_processes: 2\n")
    return path

  def _record_segment_outcome(
      self,
      step: int,
      *,
      paused: bool,
      completed: bool = False,
      write_checkpoint: bool = True,
  ) -> str:
    """Records what rank 0 of a training segment writes when it exits."""
    checkpoint_dir = os.path.join(self.temp_dir, f"checkpoint-{step}")
    if write_checkpoint:
      _write_adapter_checkpoint(checkpoint_dir)
    status = ContinualEvalStatus.load(self.temp_dir)
    status.current_step = step
    status.checkpoint_dir = checkpoint_dir if write_checkpoint else None
    status.paused_for_eval = paused
    status.training_completed = completed
    status.rollout_temperature = 0.7
    status.save(self.temp_dir)
    return checkpoint_dir

  def _set_evaluated_steps(self, steps: list[int]) -> None:
    status = ContinualEvalStatus.load(self.temp_dir)
    status.evaluated_steps = list(steps)
    status.save(self.temp_dir)

  def test_build_segment_command(self):
    self.assertEqual(
        build_segment_command(["--a", "1"], None, 1234),
        [sys.executable, _PERL_SCRIPT, "--a", "1"],
    )
    self.assertEqual(
        build_segment_command(["--a", "1"], "/cfg.yaml", 1234),
        [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--config_file=/cfg.yaml",
            "--main_process_port=1234",
            _PERL_SCRIPT,
            "--a",
            "1",
        ],
    )

  def test_ranks_other_than_zero_exit_under_a_launcher(self):
    runner = MagicMock(return_value=0)
    rc = run_perl_with_continual_eval(
        self.argv,
        env={
            "LOCAL_RANK": "1",
            "RANK": "1",
            "WORLD_SIZE": "2",
            "LOCAL_WORLD_SIZE": "2",
        },
        runner=runner,
    )
    self.assertEqual(rc, 0)
    runner.assert_not_called()
    self.assertFalse(
        os.path.exists(ContinualEvalStatus.status_path(self.temp_dir))
    )

  def test_distributed_launcher_requires_a_launch_config(self):
    runner = MagicMock(return_value=0)
    for env in (
        {"LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "2"},
        {
            "LOCAL_RANK": "0",
            "RANK": "0",
            "WORLD_SIZE": "1",
            "ACCELERATE_USE_DEEPSPEED": "true",
        },
    ):
      with self.subTest(env=env):
        with self.assertRaisesRegex(ValueError, "continual_eval_launch_config"):
          run_perl_with_continual_eval(self.argv, env=env, runner=runner)
    runner.assert_not_called()

  def test_multi_node_launch_is_rejected(self):
    argv = [*self.argv, "--continual_eval_launch_config", self._launch_config()]
    with self.assertRaisesRegex(ValueError, "single-node"):
      run_perl_with_continual_eval(
          argv,
          env={
              "LOCAL_RANK": "0",
              "RANK": "0",
              "WORLD_SIZE": "4",
              "LOCAL_WORLD_SIZE": "2",
          },
          runner=MagicMock(return_value=0),
      )

  def test_missing_launch_config_file_is_rejected(self):
    argv = [
        *self.argv,
        "--continual_eval_launch_config",
        os.path.join(self.temp_dir, "missing.yaml"),
    ]
    with self.assertRaises(FileNotFoundError):
      run_perl_with_continual_eval(
          argv, env={}, runner=MagicMock(return_value=0)
      )

  def test_segments_are_relaunched_with_accelerate_and_a_clean_env(self):
    config = self._launch_config()
    argv = [*self.argv, "--continual_eval_launch_config", config]
    launcher_env = {
        "PATH": "/usr/bin",
        "PYTHONPATH": "/extra/site",
        "LOCAL_RANK": "0",
        "RANK": "0",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
        "GROUP_RANK": "0",
        "ROLE_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        "TORCHELASTIC_RUN_ID": "abc",
        "TORCHELASTIC_USE_AGENT_STORE": "True",
        "ACCELERATE_USE_DEEPSPEED": "true",
    }
    calls = []

    def runner(cmd, env):
      calls.append((list(cmd), dict(env)))
      self._record_segment_outcome(
          50, paused=False, completed=True, write_checkpoint=False
      )
      return 0

    self.assertEqual(
        run_perl_with_continual_eval(argv, env=launcher_env, runner=runner), 0
    )
    self.assertEqual(len(calls), 1)
    cmd, env = calls[0]
    self.assertEqual(
        cmd[:3], [sys.executable, "-m", "accelerate.commands.launch"]
    )
    self.assertEqual(cmd[3], f"--config_file={os.path.abspath(config)}")
    self.assertRegex(cmd[4], r"^--main_process_port=\d+$")
    self.assertEqual(cmd[5:], [_PERL_SCRIPT, *argv])
    # A nested rendezvous inheriting these hangs on the parent's agent store.
    for var in (
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_USE_AGENT_STORE",
    ):
      self.assertNotIn(var, env)
    self.assertEqual(env[CONTINUAL_WORKER_ENV], "1")
    self.assertEqual(
        env["PYTHONPATH"].split(os.pathsep), [_REPO_ROOT, "/extra/site"]
    )
    self.assertEqual(env["PATH"], "/usr/bin")

  def test_plain_python_segment_without_a_launch_config(self):
    calls = []

    def runner(cmd, env):
      del env
      calls.append(list(cmd))
      self._record_segment_outcome(
          50, paused=False, completed=True, write_checkpoint=False
      )
      return 0

    self.assertEqual(
        run_perl_with_continual_eval(self.argv, env={}, runner=runner), 0
    )
    self.assertEqual(calls, [[sys.executable, _PERL_SCRIPT, *self.argv]])

  def test_pending_evaluation_runs_before_training_resumes(self):
    # The coordinator was restarted while step 25 awaited its evaluation.
    ckpt_25 = self._record_segment_outcome(25, paused=True)
    self._set_evaluated_steps([0])
    calls = []

    def runner(cmd, env):
      calls.append(list(cmd))
      if _is_training_segment(cmd):
        self.assertEqual(env[CONTINUAL_WORKER_ENV], "1")
        self.assertEqual(_flag_value(cmd, "--resume_from_checkpoint"), ckpt_25)
        self._record_segment_outcome(
            50, paused=False, completed=True, write_checkpoint=False
        )
      else:
        self.assertNotIn(CONTINUAL_WORKER_ENV, env)
      return 0

    self.assertEqual(
        run_perl_with_continual_eval(self.argv, env={}, runner=runner), 0
    )
    self.assertEqual(_stages(calls), ["generate", "score", "train"])
    status = ContinualEvalStatus.load(self.temp_dir)
    self.assertEqual(status.evaluated_steps, [0, 25])
    self.assertTrue(status.training_completed)
    self.assertFalse(status.paused_for_eval)

  def test_failed_evaluation_is_retried_before_any_training(self):
    self._record_segment_outcome(25, paused=True)

    def failing_runner(cmd, env):
      del env
      self.assertFalse(_is_training_segment(cmd))
      return 1 if _flag_value(cmd, "--mode") == "score" else 0

    with self.assertRaisesRegex(RuntimeError, "scoring failed at step 25"):
      run_perl_with_continual_eval(self.argv, env={}, runner=failing_runner)
    status = ContinualEvalStatus.load(self.temp_dir)
    self.assertTrue(status.paused_for_eval)
    self.assertNotIn(25, status.evaluated_steps)
    self.assertIn("scoring", status.eval_error)

    calls = []

    def runner(cmd, env):
      del env
      calls.append(list(cmd))
      if _is_training_segment(cmd):
        self._record_segment_outcome(50, paused=True, completed=True)
      return 0

    self.assertEqual(
        run_perl_with_continual_eval(self.argv, env={}, runner=runner), 0
    )
    self.assertEqual(
        _stages(calls), ["generate", "score", "train", "generate", "score"]
    )
    status = ContinualEvalStatus.load(self.temp_dir)
    self.assertEqual(status.evaluated_steps, [25, 50])
    self.assertIsNone(status.eval_error)
    self.assertTrue(status.training_completed)

  def test_failed_segment_raises(self):
    with self.assertRaisesRegex(RuntimeError, "exited with code 3"):
      run_perl_with_continual_eval(
          self.argv, env={}, runner=lambda cmd, env: 3
      )

  def test_segment_that_makes_no_progress_is_not_relaunched(self):
    runner = MagicMock(return_value=0)
    with self.assertRaisesRegex(RuntimeError, "refusing to relaunch"):
      run_perl_with_continual_eval(self.argv, env={}, runner=runner)
    self.assertEqual(runner.call_count, 1)

  def test_missing_resume_checkpoint_raises(self):
    ContinualEvalStatus(
        current_step=25,
        checkpoint_dir=os.path.join(self.temp_dir, "checkpoint-25"),
        evaluated_steps=[0, 25],
    ).save(self.temp_dir)
    runner = MagicMock(return_value=0)
    with self.assertRaisesRegex(RuntimeError, "Cannot resume PE-RL from step"):
      run_perl_with_continual_eval(self.argv, env={}, runner=runner)
    runner.assert_not_called()

  def test_missing_paused_checkpoint_fails_its_evaluation(self):
    ContinualEvalStatus(
        paused_for_eval=True,
        current_step=25,
        checkpoint_dir=os.path.join(self.temp_dir, "checkpoint-25"),
    ).save(self.temp_dir)
    runner = MagicMock(return_value=0)
    with self.assertRaisesRegex(RuntimeError, "is missing"):
      run_perl_with_continual_eval(self.argv, env={}, runner=runner)
    runner.assert_not_called()
    self.assertIn("missing", ContinualEvalStatus.load(self.temp_dir).eval_error)

  def test_completed_run_returns_without_launching_anything(self):
    self._record_segment_outcome(50, paused=False, completed=True)
    self._set_evaluated_steps([0, 25, 50])
    runner = MagicMock(return_value=0)
    self.assertEqual(
        run_perl_with_continual_eval(self.argv, env={}, runner=runner), 0
    )
    runner.assert_not_called()

  def _record_wandb_run(self) -> None:
    """Records the W&B run the first segment created, as its rank 0 does."""
    status = ContinualEvalStatus.load(self.temp_dir)
    status.wandb_run_id = "trial1"
    status.wandb_project = "perl_project"
    status.wandb_entity = "perl_entity"
    status.save(self.temp_dir)

  def test_resumed_segments_leave_the_sweep_context(self):
    # What `wandb agent` hands a sweep trial.
    agent_env = {
        "WANDB_SWEEP_ID": "sweep42",
        "WANDB_RUN_ID": "trial1",
        "WANDB_PROJECT": "perl_project",
    }
    seen = []

    def runner(cmd, env):
      seen.append((_stages([cmd])[0], dict(env)))
      if _is_training_segment(cmd):
        if [stage for stage, _ in seen].count("train") == 1:
          self._record_segment_outcome(0, paused=True)
          self._record_wandb_run()
        else:
          self._record_segment_outcome(50, paused=True, completed=True)
      return 0

    self.assertEqual(
        run_perl_with_continual_eval(self.argv, env=agent_env, runner=runner),
        0,
    )
    self.assertEqual(
        [stage for stage, _ in seen],
        ["train", "generate", "score", "train", "generate", "score"],
    )
    first, resumed = [env for stage, env in seen if stage == "train"]
    # Only the first segment registers the run with the sweep ...
    self.assertEqual(first["WANDB_SWEEP_ID"], "sweep42")
    self.assertEqual(first["WANDB_RUN_ID"], "trial1")
    # ... later ones resume it, as the scorer attaches to it, outside it.
    self.assertNotIn("WANDB_SWEEP_ID", resumed)
    self.assertEqual(resumed["WANDB_RUN_ID"], "trial1")
    self.assertEqual(resumed["WANDB_RESUME"], "allow")
    self.assertEqual(resumed["WANDB_PROJECT"], "perl_project")
    self.assertEqual(resumed["WANDB_ENTITY"], "perl_entity")
    for stage, env in seen:
      if stage != "train":
        with self.subTest(stage=stage):
          self.assertNotIn("WANDB_SWEEP_ID", env)

  def test_timed_out_evaluation_stays_pending_and_is_retried(self):
    self._record_segment_outcome(25, paused=True)

    def stalled_runner(cmd, env):
      del env
      raise subprocess.TimeoutExpired(cmd, 90 * 60.0)

    with self.assertRaisesRegex(
        RuntimeError, "generation timed out at step 25 after 90 minutes"
    ):
      run_perl_with_continual_eval(self.argv, env={}, runner=stalled_runner)
    status = ContinualEvalStatus.load(self.temp_dir)
    self.assertTrue(status.paused_for_eval)
    self.assertNotIn(25, status.evaluated_steps)
    self.assertIn("timed out", status.eval_error)

    calls = []

    def runner(cmd, env):
      del env
      calls.append(list(cmd))
      if _is_training_segment(cmd):
        self._record_segment_outcome(50, paused=True, completed=True)
      return 0

    self.assertEqual(
        run_perl_with_continual_eval(self.argv, env={}, runner=runner), 0
    )
    self.assertEqual(
        _stages(calls), ["generate", "score", "train", "generate", "score"]
    )
    self.assertIsNone(ContinualEvalStatus.load(self.temp_dir).eval_error)

  def test_only_evaluation_phases_are_time_bounded(self):
    calls = []

    def fake_run_subprocess(cmd, env, timeout_s=None):
      del env
      calls.append((_stages([cmd])[0], timeout_s))
      if _is_training_segment(cmd):
        self._record_segment_outcome(50, paused=True, completed=True)
      return 0

    argv = [*self.argv, "--continual_eval_timeout_minutes", "90"]
    with patch.object(
        continual_eval_mod, "_run_subprocess", side_effect=fake_run_subprocess
    ):
      self.assertEqual(run_perl_with_continual_eval(argv, env={}), 0)
    self.assertEqual(
        calls, [("train", None), ("generate", 5400.0), ("score", 5400.0)]
    )

  def test_malformed_timeout_fails_before_anything_runs(self):
    argv = [*self.argv, "--continual_eval_timeout_minutes", "soon"]
    with patch.object(continual_eval_mod, "_run_subprocess") as run:
      with self.assertRaisesRegex(ValueError, "continual_eval_timeout_minutes"):
        run_perl_with_continual_eval(argv, env={})
    run.assert_not_called()
    self.assertFalse(
        os.path.exists(ContinualEvalStatus.status_path(self.temp_dir))
    )


class TestContinualEvalDefaultsParity(unittest.TestCase):
  """The judge defaults are repeated outside src.continual_eval; all agree."""

  #: `scripts/evaluator.sh` (final evaluation) variable of each setting.
  _FINAL_EVAL_VARIABLES = {
      "seed": "SEED",
      "max_samples": "MAX_EVAL_SAMPLES",
      "max_tokens": "MAX_TOKENS",
      "evaluator_model": "EVALUATOR_MODEL",
      "use_gemini": "USE_GEMINI",
      "num_fewshot": "EVALUATOR_NUM_FEWSHOT",
      "autorater_num_samples": "AUTORATER_NUM_SAMPLES",
      "threshold": "THRESHOLD",
      "run_reward_hacking": "RUN_REWARD_HACKING_AUTORATER",
      "reward_hacking_num_fewshot": "REWARD_HACKING_NUM_FEWSHOT",
      "reward_hacking_threshold": "REWARD_HACKING_THRESHOLD",
      "max_workers": "MAX_WORKERS",
      "batch_size": "EVAL_BATCH_SIZE",
      "compute_bertscore": "COMPUTE_BERTSCORE",
  }

  def test_script_arguments_defaults(self):
    fields = ScriptArguments.__dataclass_fields__
    for key, value in CONTINUAL_EVAL_DEFAULTS.items():
      with self.subTest(setting=key):
        self.assertEqual(str(fields[f"continual_eval_{key}"].default), value)

  def test_perl_sh_defaults(self):
    defaults = _shell_defaults(os.path.join(_REPO_ROOT, "scripts", "perl.sh"))
    for key, value in CONTINUAL_EVAL_DEFAULTS.items():
      with self.subTest(setting=key):
        self.assertEqual(defaults[f"CONTINUAL_EVAL_{key.upper()}"], value)

  def test_perl_sh_keeps_continual_eval_opt_in(self):
    defaults = _shell_defaults(os.path.join(_REPO_ROOT, "scripts", "perl.sh"))
    self.assertEqual(defaults["CONTINUAL_EVAL"], "False")
    self.assertIs(
        ScriptArguments.__dataclass_fields__["continual_eval"].default, False
    )

  def test_perl_sh_pauses_on_rloo_generation_boundaries(self):
    defaults = _shell_defaults(os.path.join(_REPO_ROOT, "scripts", "perl.sh"))
    eval_steps = int(defaults["EVAL_STEPS"])
    # Continual evaluation forces save_steps = eval_steps anyway; keeping
    # them equal keeps a hand-run without it saving at the same steps.
    self.assertEqual(int(defaults["SAVE_STEPS"]), eval_steps)
    cycle = int(defaults["STEPS_PER_GENERATION"]) * int(
        defaults["NUM_ITERATIONS"]
    )
    accumulation = int(defaults["GRADIENT_ACCUMULATION_STEPS"])
    # The orchestrator doubles the accumulation for >= 6B policies.
    for steps in (accumulation, 2 * accumulation):
      with self.subTest(gradient_accumulation_steps=steps):
        self.assertEqual(eval_steps * steps % cycle, 0)

  def test_perl_sh_passes_every_setting_as_a_known_flag(self):
    with open(
        os.path.join(_REPO_ROOT, "scripts", "perl.sh"), encoding="utf-8"
    ) as f:
      script = f.read()
    for key in CONTINUAL_EVAL_DEFAULTS:
      with self.subTest(setting=key):
        self.assertIn(
            f'--continual_eval_{key} "$CONTINUAL_EVAL_{key.upper()}"', script
        )
    self.assertIn('--continual_eval_launch_config "$DEEPSPEED_CONFIG"', script)
    # Training segments parse their argv with HfArgumentParser, which rejects
    # flags that are not dataclass fields.
    fields = ScriptArguments.__dataclass_fields__
    for flag in sorted(set(re.findall(r"--(continual_eval\w*)", script))):
      with self.subTest(flag=flag):
        self.assertIn(flag, fields)

  def test_final_evaluation_settings(self):
    defaults = _shell_defaults(
        os.path.join(_REPO_ROOT, "scripts", "evaluator.sh")
    )
    # compute_perplexity is the one deliberate difference: it reloads the base
    # model at every evaluated step and is on neither Pareto frontier.
    self.assertEqual(
        set(self._FINAL_EVAL_VARIABLES) | {"compute_perplexity"},
        set(CONTINUAL_EVAL_DEFAULTS),
    )
    for key, variable in self._FINAL_EVAL_VARIABLES.items():
      with self.subTest(setting=key):
        if key == "max_samples":
          expected = str(
              int(
                  round(
                      int(defaults[variable])
                      * continual_eval_mod.CONTINUAL_EVAL_SAMPLE_FRACTION
                  )
              )
          )
          self.assertEqual(CONTINUAL_EVAL_DEFAULTS[key], expected)
        else:
          self.assertEqual(defaults[variable], CONTINUAL_EVAL_DEFAULTS[key])


class TestEvalPhaseTimeout(unittest.TestCase):
  """Each evaluation phase is bounded, like a phase of the final evaluation."""

  # pylint: disable=protected-access

  def test_budget_parsing(self):
    parse = continual_eval_mod._eval_phase_timeout_s
    default = CONTINUAL_EVAL_TIMEOUT_MINUTES * 60.0
    self.assertEqual(parse({}), default)
    self.assertEqual(parse({"continual_eval_timeout_minutes": ""}), default)
    self.assertEqual(parse({"continual_eval_timeout_minutes": "90"}), 5400.0)
    self.assertEqual(parse({"continual_eval_timeout_minutes": " 0.5 "}), 30.0)
    for disabled in ("0", "-5", "inf", "nan"):
      with self.subTest(value=disabled):
        self.assertIsNone(parse({"continual_eval_timeout_minutes": disabled}))
    with self.assertRaisesRegex(ValueError, "must be a number"):
      parse({"continual_eval_timeout_minutes": "soon"})

  def test_exit_code_is_returned(self):
    rc = continual_eval_mod._run_subprocess(
        [sys.executable, "-c", "import sys; sys.exit(3)"],
        dict(os.environ),
        timeout_s=60.0,
    )
    self.assertEqual(rc, 3)

  def test_overrunning_subprocess_is_terminated(self):
    started = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
      process = real_popen(*args, **kwargs)
      started.append(process)
      return process

    with patch.object(
        continual_eval_mod.subprocess, "Popen", side_effect=recording_popen
    ):
      with self.assertRaises(subprocess.TimeoutExpired):
        continual_eval_mod._run_subprocess(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            dict(os.environ),
            timeout_s=0.5,
        )
    (process,) = started
    self.assertEqual(process.returncode, -signal.SIGTERM)

  def _fake_popen(self, wait_side_effect):
    process = MagicMock()
    process.wait.side_effect = wait_side_effect
    popen = MagicMock()
    popen.return_value.__enter__.return_value = process
    popen.return_value.__exit__.return_value = False
    return popen, process

  def test_subprocess_that_ignores_sigterm_is_killed(self):
    cmd = ["evaluator"]
    popen, process = self._fake_popen([
        subprocess.TimeoutExpired(cmd, 60.0),
        subprocess.TimeoutExpired(cmd, 30.0),
        -signal.SIGKILL,
    ])
    with patch.object(continual_eval_mod.subprocess, "Popen", popen):
      with self.assertRaises(subprocess.TimeoutExpired) as ctx:
        continual_eval_mod._run_subprocess(cmd, {}, timeout_s=60.0)
    # The phase's own timeout, not the grace period's.
    self.assertEqual(ctx.exception.timeout, 60.0)
    self.assertEqual(
        process.mock_calls,
        [
            call.wait(timeout=60.0),
            call.terminate(),
            call.wait(timeout=continual_eval_mod._TERMINATE_GRACE_SECONDS),
            call.kill(),
            call.wait(),
        ],
    )

  def test_an_interrupt_terminates_the_subprocess_gracefully(self):
    grace = continual_eval_mod._TERMINATE_GRACE_SECONDS
    popen, process = self._fake_popen([KeyboardInterrupt(), 0])
    with patch.object(continual_eval_mod.subprocess, "Popen", popen):
      with self.assertRaises(KeyboardInterrupt):
        continual_eval_mod._run_subprocess(["accelerate"], {})
    self.assertEqual(
        process.mock_calls,
        [call.wait(timeout=None), call.terminate(), call.wait(timeout=grace)],
    )

  def test_an_interrupted_subprocess_is_killed_if_it_lingers(self):
    grace = continual_eval_mod._TERMINATE_GRACE_SECONDS
    for second in (
        subprocess.TimeoutExpired("accelerate", grace),
        KeyboardInterrupt(),  # Ctrl-C pressed again.
    ):
      with self.subTest(second=type(second).__name__):
        popen, process = self._fake_popen([KeyboardInterrupt(), second])
        with patch.object(continual_eval_mod.subprocess, "Popen", popen):
          # The first interrupt is what propagates.
          with self.assertRaises(KeyboardInterrupt):
            continual_eval_mod._run_subprocess(["accelerate"], {})
        self.assertEqual(
            process.mock_calls,
            [
                call.wait(timeout=None),
                call.terminate(),
                call.wait(timeout=grace),
                call.kill(),
            ],
        )

  def test_an_interrupt_reaches_a_real_subprocess_as_sigterm(self):
    # The child never sees the SIGINT, like a worker blocked in a collective
    # or one in a session of its own: only the coordinator can stop it.
    started = []
    real_popen = subprocess.Popen
    main_thread = threading.main_thread().ident

    def popen_then_interrupt(*args, **kwargs):
      process = real_popen(*args, **kwargs)
      started.append(process)
      self.addCleanup(process.kill)
      timer = threading.Timer(
          0.5, signal.pthread_kill, args=(main_thread, signal.SIGINT)
      )
      self.addCleanup(timer.cancel)
      timer.start()
      return process

    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    self.addCleanup(signal.signal, signal.SIGINT, previous)
    with patch.object(
        continual_eval_mod.subprocess, "Popen", side_effect=popen_then_interrupt
    ):
      with self.assertRaises(KeyboardInterrupt):
        continual_eval_mod._run_subprocess(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            dict(os.environ),
        )
    (process,) = started
    self.assertEqual(process.returncode, -signal.SIGTERM)


class TestOrchestratorContinualEvalWiring(unittest.TestCase):
  """What the orchestrator passes each PE-RL run reaches the coordinator."""

  # pylint: disable=protected-access

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

  def _perl_stage(self, calibrated_threshold=None, **eval_overrides):
    config = CampaignConfig.create_default(task_name="npov", dry_run=True)
    for key, value in eval_overrides.items():
      setattr(config.eval, key, value)
    state = CampaignState(campaign_id="c", task_name="npov")
    if calibrated_threshold is not None:
      state.stages["autorater"] = StageResult(
          status=StageStatus.COMPLETED,
          metrics={"autorater/best_threshold": calibrated_threshold},
      )
    return PerlStage(
        CampaignContext(
            config=config,
            state=state,
            sweep_controller=SweepController(dry_run=True),
            model_manager=ModelManager(dry_run=True),
        )
    )

  def test_shared_constants_stay_in_sync(self):
    self.assertEqual(
        orchestrator_model_manager.CONTINUAL_EVAL_STATUS_FILENAME,
        CONTINUAL_EVAL_STATUS_FILENAME,
    )
    self.assertEqual(
        orchestrator_sweep_controller.CONTINUAL_EVAL_PAUSED_KEY,
        PAUSED_SUMMARY_KEY,
    )
    # One budget per evaluation phase, continual or final.
    timeout_field = ScriptArguments.__dataclass_fields__[
        "continual_eval_timeout_minutes"
    ]
    self.assertEqual(timeout_field.default, CONTINUAL_EVAL_TIMEOUT_MINUTES)
    self.assertEqual(
        float(EvalStageConfig().timeout_minutes), CONTINUAL_EVAL_TIMEOUT_MINUTES
    )

  def test_every_flag_is_a_perl_argument(self):
    for overrides in (
        {},
        {"reward_hacking_model": "gemini-2.5-pro", "max_model_len": 8192},
    ):
      with self.subTest(**overrides):
        flags = self._perl_stage(**overrides).continual_eval_flags()
        for name, raw in flags.items():
          with self.subTest(flag=name):
            _parse_perl_flag(name, raw)
        # Every judge setting is the campaign's, none perl.sh's default.
        self.assertLessEqual(
            {f"continual_eval_{key}" for key in CONTINUAL_EVAL_DEFAULTS},
            set(flags),
        )
        self.assertIs(
            _parse_perl_flag("continual_eval", flags["continual_eval"]), True
        )
        self.assertEqual(
            "continual_eval_reward_hacking_model" in flags, bool(overrides)
        )
        self.assertEqual(
            "continual_eval_max_model_len" in flags, bool(overrides)
        )

  def test_threshold_is_the_calibrated_one(self):
    self.assertEqual(
        self._perl_stage(calibrated_threshold=0.0).continual_eval_threshold(),
        0.0,
    )
    stage = self._perl_stage()
    self.assertEqual(
        stage.continual_eval_threshold(), stage.config.eval.threshold
    )

  def test_campaign_settings_reach_the_scorer(self):
    stage = self._perl_stage(
        calibrated_threshold=0.37,
        max_eval_samples=200,
        evaluator_model="gemini-x",
        timeout_minutes=45,
    )
    argv = [
        "--task_name=npov",
        "--dataset_repo_id=leobianco/npov_perl",
        "--do_train=True",
        f"--output_dir={self.temp_dir}",
        "--temperature=0.7",
        *[f"--{k}={v}" for k, v in stage.continual_eval_flags().items()],
    ]
    self.assertTrue(should_coordinate_continual_eval(argv, {}))
    flags = parse_cli_flag_map(argv)
    self.assertEqual(
        continual_eval_mod._eval_phase_timeout_s(flags), 45 * 60.0
    )
    gen_cmd, score_cmd = build_continual_eval_commands(
        flags=flags,
        status=ContinualEvalStatus(
            paused_for_eval=True,
            current_step=50,
            checkpoint_dir=os.path.join(self.temp_dir, "checkpoint-50"),
            rollout_temperature=0.7,
        ),
        output_dir=self.temp_dir,
        env={},
    )
    _parse_evaluator_command(gen_cmd)
    _parse_evaluator_command(score_cmd)
    self.assertEqual(_flag_value(score_cmd, "--threshold"), "0.37")
    self.assertEqual(_flag_value(score_cmd, "--evaluator_model"), "gemini-x")
    self.assertEqual(_flag_value(score_cmd, "--compute_perplexity"), "False")
    for cmd in (gen_cmd, score_cmd):
      self.assertEqual(_flag_value(cmd, "--max_eval_samples"), "50")


class TestConstrainedCheckpointSelection(unittest.TestCase):
  """Constrained continual-eval checkpoint selection and root promotion."""

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.mkdtemp(prefix="test_constrained_ckpt_")

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)
    super().tearDown()

  def test_excludes_step_zero_and_enforces_reward_hacking_ceiling(self):
    history = [
        # Step 0 (untrained SFT baseline): low hallu, rh_rate=0.04 -> ceiling = max(0.10, 0.04+0.05) = 0.10
        {
            "step": 0,
            "hallucination_rate": 0.05,
            "reward_hacking_rate": 0.04,
            "reward_hacking_quality": 0.90,
        },
        # Step 24: lower hallu than step 48, but violates ceiling (0.18 > 0.10)
        {
            "step": 24,
            "hallucination_rate": 0.08,
            "reward_hacking_rate": 0.18,
            "reward_hacking_quality": 0.70,
        },
        # Step 48: satisfies ceiling (0.07 <= 0.10)
        {
            "step": 48,
            "hallucination_rate": 0.12,
            "reward_hacking_rate": 0.07,
            "reward_hacking_quality": 0.85,
        },
        # Step 72 (final): satisfies ceiling (0.09 <= 0.10) but higher hallu (0.15)
        {
            "step": 72,
            "hallucination_rate": 0.15,
            "reward_hacking_rate": 0.09,
            "reward_hacking_quality": 0.82,
        },
    ]
    sel = continual_eval_mod.select_best_continual_checkpoint(history)
    self.assertIsNotNone(sel)
    assert sel is not None
    # Step 0 is excluded, step 24 is rejected by ceiling, step 48 wins over 72
    self.assertEqual(sel.step, 48)
    self.assertAlmostEqual(sel.hallucination_rate, 0.12)
    self.assertAlmostEqual(sel.reward_hacking_rate, 0.07)
    self.assertAlmostEqual(sel.ceiling, 0.10)
    self.assertTrue(sel.met_ceiling)

  def test_step_zero_only_returns_none(self):
    history = [
        {
            "step": 0,
            "hallucination_rate": 0.10,
            "reward_hacking_rate": 0.02,
            "reward_hacking_quality": 0.90,
        }
    ]
    self.assertIsNone(
        continual_eval_mod.select_best_continual_checkpoint(history)
    )

  def test_fallback_when_all_trained_checkpoints_exceed_ceiling(self):
    history = [
        {
            "step": 0,
            "hallucination_rate": 0.20,
            "reward_hacking_rate": 0.02,
            "reward_hacking_quality": 0.90,
        },
        {
            "step": 24,
            "hallucination_rate": 0.10,
            "reward_hacking_rate": 0.30,
            "reward_hacking_quality": 0.60,
        },
        {
            "step": 48,
            "hallucination_rate": 0.14,
            "reward_hacking_rate": 0.15,
            "reward_hacking_quality": 0.75,
        },
    ]
    sel = continual_eval_mod.select_best_continual_checkpoint(history)
    self.assertIsNotNone(sel)
    assert sel is not None
    self.assertFalse(sel.met_ceiling)
    # Minimizes reward_hacking_rate when all step > 0 exceed ceiling
    self.assertEqual(sel.step, 48)

  def test_preserve_and_finalize_promotes_best_to_root_and_saves_last(self):
    # Create step-24 and step-48 checkpoints; step 24 wins and is preserved
    ckpt24 = os.path.join(self.temp_dir, "checkpoint-24")
    os.makedirs(ckpt24, exist_ok=True)
    with open(os.path.join(ckpt24, "adapter_model.safetensors"), "wb") as f:
      f.write(b"weights-step-24")
    with open(os.path.join(ckpt24, "optimizer.pt"), "wb") as f:
      f.write(b"opt-24")

    history = [
        {
            "step": 0,
            "hallucination_rate": 0.25,
            "reward_hacking_rate": 0.03,
            "reward_hacking_quality": 0.88,
        },
        {
            "step": 24,
            "hallucination_rate": 0.11,
            "reward_hacking_rate": 0.05,
            "reward_hacking_quality": 0.86,
        },
    ]
    with open(
        os.path.join(self.temp_dir, "continual_eval_history.json"),
        "w",
        encoding="utf-8",
    ) as f:
      json.dump(history, f)
    sel24 = continual_eval_mod.preserve_best_continual_checkpoint(
        self.temp_dir, step=24, checkpoint_dir=ckpt24
    )
    self.assertIsNotNone(sel24)
    # Simulate HF Trainer save_total_limit deleting checkpoint-24 later, and
    # saving final step-48 weights at root + checkpoint-48
    shutil.rmtree(ckpt24)
    ckpt48 = os.path.join(self.temp_dir, "checkpoint-48")
    os.makedirs(ckpt48, exist_ok=True)
    for target in (self.temp_dir, ckpt48):
      with open(os.path.join(target, "adapter_model.safetensors"), "wb") as f:
        f.write(b"weights-step-48")

    history.append({
        "step": 48,
        "hallucination_rate": 0.19,
        "reward_hacking_rate": 0.06,
        "reward_hacking_quality": 0.84,
    })
    with open(
        os.path.join(self.temp_dir, "continual_eval_history.json"),
        "w",
        encoding="utf-8",
    ) as f:
      json.dump(history, f)

    status48 = ContinualEvalStatus(
        paused_for_eval=True,
        current_step=48,
        checkpoint_dir=ckpt48,
        training_completed=True,
    )
    continual_eval_mod.finalize_continual_eval_checkpoints(
        output_dir=self.temp_dir,
        flags={"push_to_hub": "False"},
        status=status48,
    )

    # Root now has step-24 weights, and last/ has step-48 weights!
    with open(
        os.path.join(self.temp_dir, "adapter_model.safetensors"), "rb"
    ) as f:
      self.assertEqual(f.read(), b"weights-step-24")
    with open(
        os.path.join(self.temp_dir, "last", "adapter_model.safetensors"), "rb"
    ) as f:
      self.assertEqual(f.read(), b"weights-step-48")
    with open(
        os.path.join(self.temp_dir, "checkpoints.json"),
        "r",
        encoding="utf-8",
    ) as f:
      manifest = json.load(f)
    self.assertEqual(manifest["default"], "best")
    self.assertEqual(manifest["default_step"], 24)
    self.assertIsNone(manifest["checkpoints"]["best"]["subfolder"])
    self.assertEqual(manifest["checkpoints"]["last"]["subfolder"], "last")
    self.assertEqual(manifest["checkpoints"]["last"]["step"], 48)


class TestPerlEntryPoint(unittest.TestCase):
  """src/perl.py dispatches to the coordinator or to a training segment."""

  _ARGV = ["--task_name", "npov", "--do_train", "True", "--continual_eval",
           "True"]

  def _env_without_worker_marker(self) -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != CONTINUAL_WORKER_ENV}

  def test_coordinator_never_imports_the_training_stack(self):
    from src import perl  # pylint: disable=g-import-not-at-top

    # A None entry makes every `import src.pipelines` raise ImportError.
    with patch.dict(sys.modules, {"src.pipelines": None}), patch.dict(
        os.environ, self._env_without_worker_marker(), clear=True
    ):
      perl = importlib.reload(perl)
      with patch.object(
          perl, "run_perl_with_continual_eval", return_value=0
      ) as run:
        perl.main(list(self._ARGV))
    run.assert_called_once_with(list(self._ARGV))

  def test_failed_coordination_exits_with_its_code(self):
    from src import perl  # pylint: disable=g-import-not-at-top

    with patch.dict(
        os.environ, self._env_without_worker_marker(), clear=True
    ), patch.object(perl, "run_perl_with_continual_eval", return_value=3):
      with self.assertRaises(SystemExit) as ctx:
        perl.main(list(self._ARGV))
    self.assertEqual(ctx.exception.code, 3)

  def test_training_segment_runs_the_pipeline(self):
    from src import perl  # pylint: disable=g-import-not-at-top

    fake_pipelines = types.ModuleType("src.pipelines")
    fake_pipelines.PERLPipeline = MagicMock()
    with patch.dict(sys.modules, {"src.pipelines": fake_pipelines}), patch.dict(
        os.environ, {CONTINUAL_WORKER_ENV: "1"}
    ), patch.object(perl, "run_perl_with_continual_eval") as run:
      perl.main(list(self._ARGV))
    run.assert_not_called()
    fake_pipelines.PERLPipeline.return_value.run.assert_called_once_with(
        *self._ARGV
    )


if __name__ == "__main__":
  unittest.main()
