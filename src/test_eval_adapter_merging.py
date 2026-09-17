"""Unit tests for evaluation-time SFT + PE-RL LoRA adapter combination."""

import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch


class FakeTensor:
  """Lightweight 2D matrix tensor for testing exact LoRA concatenation math."""

  def __init__(self, data, dtype="float32", device="cpu"):
    # data is a list of lists of floats
    self.data = [[float(x) for x in row] for row in data]
    self.dtype = dtype
    self.device = device

  @property
  def shape(self):
    if not self.data:
      return (0, 0)
    return (len(self.data), len(self.data[0]))

  def float(self):
    return FakeTensor(self.data, dtype="float32", device=self.device)

  def to(self, dtype):
    return FakeTensor(self.data, dtype=str(dtype), device=self.device)

  def contiguous(self):
    return self

  def __mul__(self, scalar):
    return FakeTensor(
        [[x * float(scalar) for x in row] for row in self.data],
        dtype=self.dtype,
        device=self.device,
    )

  def __rmul__(self, scalar):
    return self.__mul__(scalar)

  def matmul(self, other):
    """Matrix multiplication: self @ other."""
    rows_a, cols_a = self.shape
    rows_b, cols_b = other.shape
    assert cols_a == rows_b, f"Shape mismatch: {self.shape} vs {other.shape}"
    result = []
    for i in range(rows_a):
      row = []
      for j in range(cols_b):
        val = sum(self.data[i][k] * other.data[k][j] for k in range(cols_a))
        row.append(val)
      result.append(row)
    return FakeTensor(result, dtype=self.dtype, device=self.device)

  def __add__(self, other):
    rows, cols = self.shape
    assert (rows, cols) == other.shape
    return FakeTensor(
        [
            [self.data[i][j] + other.data[i][j] for j in range(cols)]
            for i in range(rows)
        ],
        dtype=self.dtype,
        device=self.device,
    )


def _fake_cat(tensors, dim=0):
  if dim == 0:
    combined = []
    for t in tensors:
      combined.extend(t.data)
    return FakeTensor(combined, dtype=tensors[0].dtype, device=tensors[0].device)
  elif dim == 1:
    rows = tensors[0].shape[0]
    combined = []
    for i in range(rows):
      row = []
      for t in tensors:
        row.extend(t.data[i])
      combined.append(row)
    return FakeTensor(combined, dtype=tensors[0].dtype, device=tensors[0].device)
  raise ValueError(f"Unsupported dim: {dim}")


def _fake_zeros(shape, dtype="float32", device="cpu"):
  rows, cols = shape
  return FakeTensor(
      [[0.0 for _ in range(cols)] for _ in range(rows)],
      dtype=dtype,
      device=device,
  )


# Ensure mock torch / transformers exist if running in environment without PyTorch
if "torch" not in sys.modules:
  try:
    import torch
  except ImportError:
    import pickle

    def _mock_save(obj, path):
      with open(path, "wb") as f:
        f.write(pickle.dumps(obj))

    def _mock_load(path, map_location=None):
      del map_location
      with open(path, "rb") as f:
        return pickle.loads(f.read())

    torch_mod = types.ModuleType("torch")
    torch_mod.__path__ = []
    torch_mod.Tensor = FakeTensor
    torch_mod.cat = _fake_cat
    torch_mod.zeros = _fake_zeros
    torch_mod.save = _mock_save
    torch_mod.load = _mock_load
    torch_mod.nn = types.ModuleType("torch.nn")
    torch_mod.nn.Module = object
    torch_mod.bfloat16 = "bfloat16"
    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = torch_mod.nn

if "transformers" not in sys.modules or not hasattr(
    sys.modules["transformers"], "__path__"
):
  transformers_mod = types.ModuleType("transformers")
  transformers_mod.__path__ = []
  transformers_mod.PreTrainedModel = object
  transformers_mod.PretrainedConfig = object
  transformers_mod.TrainerCallback = object
  transformers_mod.TrainerState = object
  transformers_mod.TrainerControl = object
  transformers_mod.TrainingArguments = object
  transformers_mod.Trainer = object
  transformers_mod.DataCollatorWithPadding = MagicMock()
  transformers_mod.HfArgumentParser = MagicMock()
  transformers_mod.set_seed = MagicMock()
  transformers_mod.AutoTokenizer = MagicMock()
  transformers_mod.AutoModelForCausalLM = MagicMock()
  transformers_mod.AutoModelForSequenceClassification = MagicMock()
  transformers_mod.LogitsProcessor = object
  transformers_mod.LogitsProcessorList = list
  sys.modules["transformers"] = transformers_mod
  sys.modules["transformers.trainer_utils"] = types.ModuleType(
      "transformers.trainer_utils"
  )
  sys.modules["transformers.trainer_utils"].get_last_checkpoint = MagicMock(
      return_value=None
  )

for mod_name in [
    "transformers.configuration_utils",
    "transformers.integrations",
    "transformers.integrations.heterogeneity",
    "transformers.integrations.heterogeneity.configuration_utils",
    "datasets",
    "trl",
    "peft",
    "accelerate",
    "wandb",
    "vllm",
    "vllm.lora",
    "vllm.lora.request",
    "evaluate",
    "scipy",
    "scipy.special",
    "sklearn",
    "sklearn.metrics",
    "google",
    "google.genai",
    "google.genai.types",
    "huggingface_hub",
    "matplotlib",
    "matplotlib.pyplot",
    "numpy",
    "pandas",
    "nltk",
    "tqdm",
]:
  if mod_name not in sys.modules:
    sys.modules[mod_name] = MagicMock()

import torch  # pylint: disable=g-import-not-at-top
from src.orchestrator.config import CampaignConfig  # pylint: disable=g-import-not-at-top
from src.orchestrator.model_manager import ModelManager  # pylint: disable=g-import-not-at-top
from src.orchestrator.stages.base import CampaignContext  # pylint: disable=g-import-not-at-top
from src.orchestrator.stages.eval_stage import EvalStage  # pylint: disable=g-import-not-at-top
from src.orchestrator.state import CampaignState  # pylint: disable=g-import-not-at-top
from src.orchestrator.state import StageResult  # pylint: disable=g-import-not-at-top
from src.orchestrator.state import StageStatus  # pylint: disable=g-import-not-at-top
from src.orchestrator.sweep_controller import SweepController  # pylint: disable=g-import-not-at-top
from src.pipelines import EvaluationGenerationPipeline  # pylint: disable=g-import-not-at-top
from src.utils import EvalArguments  # pylint: disable=g-import-not-at-top


class TestEvalAdapterMerging(unittest.TestCase):
  """Tests for combining SFT and RL LoRA adapters at evaluation time."""

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.mkdtemp()
    self.sft_dir = os.path.join(self.temp_dir, "sft_adapter")
    self.rl_dir = os.path.join(self.temp_dir, "rl_adapter")
    os.makedirs(self.sft_dir, exist_ok=True)
    os.makedirs(self.rl_dir, exist_ok=True)

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)
    shutil.rmtree(os.path.join("checkpoints", "eval_combined_lora"), ignore_errors=True)
    super().tearDown()

  def _create_adapter(self, folder, r, alpha, a_matrix, b_matrix, extra_cfg=None):
    cfg = {
        "r": r,
        "lora_alpha": alpha,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "v_proj"],
    }
    if extra_cfg:
      cfg.update(extra_cfg)
    with open(os.path.join(folder, "adapter_config.json"), "w", encoding="utf-8") as f:
      json.dump(cfg, f)

    state_dict = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": FakeTensor(a_matrix),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": FakeTensor(b_matrix),
    }
    import torch

    torch.save(state_dict, os.path.join(folder, "adapter_model.bin"))

  def test_combine_lora_adapters_exact_math_and_power_of_two_padding(self):
    """Combined adapter delta equals s1*B1*A1 + s2*B2*A2 with power-of-2 rank padding."""
    # SFT adapter: r1=2, alpha1=4 -> s1 = 2.0
    # d_in=3, d_out=2
    a1 = [[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]]  # 2 x 3
    b1 = [[2.0, -0.5], [1.0, 3.0]]  # 2 x 2
    self._create_adapter(self.sft_dir, r=2, alpha=4, a_matrix=a1, b_matrix=b1)

    # RL adapter: r2=3, alpha2=9 -> s2 = 3.0
    # r_sum = 2 + 3 = 5 -> padded to r_target = 8 (next power of 2 >= 8)
    a2 = [[0.2, 1.0, -0.5], [1.5, 0.0, 2.0], [-1.0, 1.0, 1.0]]  # 3 x 3
    b2 = [[0.5, 1.0, -2.0], [-1.0, 2.0, 0.5]]  # 2 x 3
    self._create_adapter(self.rl_dir, r=3, alpha=9, a_matrix=a2, b_matrix=b2)

    pipeline = EvaluationGenerationPipeline.__new__(EvaluationGenerationPipeline)
    import torch

    with patch.object(torch, "cat", side_effect=_fake_cat), patch.object(
        torch, "zeros", side_effect=_fake_zeros
    ):
      out_dir, r_target = pipeline._combine_lora_adapters(
          sft_adapter_dir=self.sft_dir,
          rl_adapter_dir=self.rl_dir,
          base_model_name_or_path="google/gemma-4-E4B-it",
      )

    self.assertEqual(r_target, 8)
    with open(os.path.join(out_dir, "adapter_config.json"), "r", encoding="utf-8") as f:
      combined_cfg = json.load(f)
    self.assertEqual(combined_cfg["r"], 8)
    self.assertEqual(combined_cfg["lora_alpha"], 8)
    self.assertEqual(combined_cfg["base_model_name_or_path"], "google/gemma-4-E4B-it")
    self.assertTrue(combined_cfg["sft_merged_into_lora"])

    combined_state = torch.load(os.path.join(out_dir, "adapter_model.bin"))
    a_comb = combined_state["base_model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    b_comb = combined_state["base_model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    self.assertEqual(a_comb.shape, (8, 3))
    self.assertEqual(b_comb.shape, (2, 8))

    # Expected delta: s1 * B1 @ A1 + s2 * B2 @ A2
    delta_sft = (FakeTensor(b1) * 2.0).matmul(FakeTensor(a1))
    delta_rl = (FakeTensor(b2) * 3.0).matmul(FakeTensor(a2))
    expected_delta = delta_sft + delta_rl

    # Combined delta: (alpha_target / r_target) * B_comb @ A_comb = 1.0 * B_comb @ A_comb
    actual_delta = b_comb.matmul(a_comb)

    for i in range(2):
      for j in range(3):
        self.assertAlmostEqual(
            actual_delta.data[i][j],
            expected_delta.data[i][j],
            places=5,
            msg=f"Mismatch at ({i}, {j})",
        )

  def test_setup_model_combines_when_sft_and_rl_are_distinct(self):
    """EvaluationGenerationPipeline.setup_model combines adapters when sft_model_path is given."""
    pipeline = EvaluationGenerationPipeline.__new__(EvaluationGenerationPipeline)
    pipeline.args = EvalArguments(
        task_name="npov",
        user="leobianco",
        writer_model_base="google/gemma-4-E4B-it",
        writer_model_lora="leobianco/npov_PERL_ckpt",
        sft_model_path="leobianco/npov_SFT_ckpt",
    )
    with patch.object(
        pipeline,
        "_resolve_lora_adapter_path",
        side_effect=[
            (True, self.rl_dir),
            (True, self.sft_dir),
        ],
    ), patch.object(
        pipeline,
        "_combine_lora_adapters",
        return_value=("/tmp/combined_adapter", 32),
    ) as mock_combine:
      pipeline.setup_model()
      mock_combine.assert_called_once_with(
          sft_adapter_dir=self.sft_dir,
          rl_adapter_dir=self.rl_dir,
          base_model_name_or_path="google/gemma-4-E4B-it",
      )
      self.assertEqual(pipeline.lora_path, "/tmp/combined_adapter")
      self.assertEqual(pipeline.max_lora_rank, 64)

  def test_setup_model_skips_combination_when_evaluating_sft_itself(self):
    """When writer_model_lora == sft_model_path, no double combination occurs."""
    pipeline = EvaluationGenerationPipeline.__new__(EvaluationGenerationPipeline)
    pipeline.args = EvalArguments(
        task_name="npov",
        user="leobianco",
        writer_model_base="google/gemma-4-E4B-it",
        writer_model_lora="leobianco/npov_SFT_ckpt",
        sft_model_path="leobianco/npov_SFT_ckpt",
    )
    with patch.object(
        pipeline,
        "_resolve_lora_adapter_path",
        return_value=(True, self.sft_dir),
    ), patch.object(
        pipeline,
        "_combine_lora_adapters",
    ) as mock_combine:
      pipeline.setup_model()
      mock_combine.assert_not_called()
      self.assertEqual(pipeline.lora_path, self.sft_dir)

  def test_eval_stage_generation_command_passes_sft_model_path_for_perl_only(self):
    """EvalStage passes --sft_model_path when evaluating PE-RL, but not when evaluating SFT."""
    config = CampaignConfig.create_default(task_name="npov", dry_run=True)
    state = CampaignState(campaign_id="test_camp", task_name="npov")
    state.stages["sft"] = StageResult(
        status=StageStatus.COMPLETED, model_repo_id="leobianco/npov_SFT_winner"
    )
    state.stages["perl"] = StageResult(
        status=StageStatus.COMPLETED, model_repo_id="leobianco/npov_PERL_winner"
    )
    ctx = CampaignContext(
        config=config,
        state=state,
        sweep_controller=SweepController(dry_run=True),
        model_manager=ModelManager(dry_run=True),
    )
    stage = EvalStage(ctx)

    sft_cmd = stage._generation_command("leobianco/npov_SFT_winner")
    self.assertNotIn("--sft_model_path", sft_cmd)

    perl_cmd = stage._generation_command("leobianco/npov_PERL_winner")
    self.assertIn("--sft_model_path", perl_cmd)
    idx = perl_cmd.index("--sft_model_path")
    self.assertEqual(perl_cmd[idx + 1], "leobianco/npov_SFT_winner")


if __name__ == "__main__":
  unittest.main()
