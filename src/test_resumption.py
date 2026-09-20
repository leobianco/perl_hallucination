"""Unit tests for checkpoint and WandB run resumption."""

import contextlib
import json
import os
import shutil
import sys
import tempfile
import types
from typing import Any
import unittest
from unittest.mock import MagicMock, patch

if "torch" not in sys.modules or not hasattr(sys.modules["torch"], "__path__"):
  if "torch" not in sys.modules:
    torch = types.ModuleType("torch")
    torch.__path__ = []
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.Module = object
    torch.nn.Linear = MagicMock()
    torch.nn.Identity = MagicMock()
    torch.nn.BCEWithLogitsLoss = object
    torch.nn.CrossEntropyLoss = object
    torch.nn.MSELoss = object
    torch.no_grad = contextlib.nullcontext
    torch.device = lambda x: x
    torch.cuda = MagicMock()
    torch.cuda.is_available = lambda: False
    torch.tensor = lambda x, **kw: MagicMock()
    torch.randn = lambda *x: MagicMock()
    torch.bernoulli = lambda x: MagicMock()
    torch.softmax = lambda x, **kw: MagicMock()
    torch.zeros_like = lambda x: MagicMock()
    torch.sort = lambda x, **kw: (MagicMock(), MagicMock())
    torch.cumsum = lambda x, **kw: MagicMock()
    torch.multinomial = lambda x, **kw: MagicMock()
    torch.full = lambda *x, **kw: MagicMock()
    torch.ones = lambda *x, **kw: MagicMock()
    torch.zeros = lambda *x, **kw: MagicMock()
    torch.where = lambda *x: MagicMock()
    torch.stack = lambda *x, **kw: MagicMock(
        tolist=lambda: [[10, 11], [12, 13]]
    )
    torch.long = "long"
    torch.bool = "bool"
    torch.bfloat16 = "bfloat16"
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch.nn

# Ensure third-party modules are mocked if not installed in local environment
if "transformers" not in sys.modules or not hasattr(
    sys.modules["transformers"], "__path__"
):
  transformers_mod = types.ModuleType("transformers")
  transformers_mod.TrainerCallback = object
  transformers_mod.TrainerControl = object
  transformers_mod.TrainerState = object
  transformers_mod.TrainingArguments = object
  transformers_mod.Trainer = object
  transformers_mod.AutoModelForCausalLM = MagicMock()
  transformers_mod.AutoModelForSequenceClassification = MagicMock()
  transformers_mod.AutoTokenizer = MagicMock()
  transformers_mod.DataCollatorWithPadding = MagicMock()
  transformers_mod.HfArgumentParser = MagicMock()
  transformers_mod.set_seed = MagicMock()
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
    "trl",
    "peft",
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
    "tqdm",
]:
  if mod_name not in sys.modules:
    sys.modules[mod_name] = MagicMock()


class _TestMockDataset:

  def __init__(self, data: dict[str, list[Any]]):
    self._data = dict(data)
    self.column_names = list(data.keys())

  def __getitem__(self, key: str):
    return self._data[key]

  def __len__(self):
    first_col = next(iter(self._data.values()))
    return len(first_col)

  def remove_columns(self, column_name: str):
    new_data = {k: v for k, v in self._data.items() if k != column_name}
    return _TestMockDataset(new_data)

  def add_column(self, column_name: str, values: list[Any]):
    new_data = dict(self._data)
    new_data[column_name] = values
    return _TestMockDataset(new_data)


class DatasetDict(dict):

  def push_to_hub(self, repo_id, **kwargs):
    pass

  def save_to_disk(self, path, **kwargs):
    pass


class Dataset:

  @classmethod
  def from_dict(cls, d):
    return _TestMockDataset(d)


datasets_mod = types.ModuleType("datasets")
datasets_mod.Dataset = Dataset
datasets_mod.DatasetDict = DatasetDict
datasets_mod.Value = MagicMock()
datasets_mod.load_dataset = MagicMock()
datasets_mod.concatenate_datasets = MagicMock()
sys.modules["datasets"] = datasets_mod

if "torch" not in sys.modules or not hasattr(sys.modules["torch"], "__path__"):
  torch_mod = types.ModuleType("torch")
  torch_mod.nn = types.ModuleType("torch.nn")
  torch_mod.nn.Module = object
  torch_mod.nn.BCEWithLogitsLoss = MagicMock()
  torch_mod.nn.CrossEntropyLoss = MagicMock()
  torch_mod.nn.MSELoss = MagicMock()
  torch_mod.bfloat16 = "bfloat16"
  torch_mod.float16 = "float16"
  torch_mod.device = MagicMock()
  torch_mod.cuda = MagicMock()
  torch_mod.cuda.is_available = MagicMock(return_value=False)
  sys.modules["torch"] = torch_mod
  sys.modules["torch.nn"] = torch_mod.nn

from src.pipelines import (
    DPOPipeline,
    PERLPipeline,
    Pipeline,
    RewardModelPipeline,
    SFTPipeline,
    WandbResumptionCallback,
    _push_to_hub_with_retry,
    parse_hf_repo_reference,
)


class DummyPipeline(Pipeline):
  """Dummy concrete pipeline for testing resumption helper methods."""

  def setup_arguments(self, *cli_args, **cli_kwargs) -> None:
    pass

  def setup_tokenizer(self) -> None:
    pass

  def load_data(self) -> None:
    pass

  def process_data(self) -> None:
    pass

  def setup_model(self) -> None:
    pass

  def setup_trainer(self) -> None:
    pass

  def run_and_save(self) -> None:
    pass


class TestParseHfRepoReference(unittest.TestCase):
  """Tests for parse_hf_repo_reference utility."""

  def test_simple_repo_id(self):
    repo_id, subfolder, rev = parse_hf_repo_reference("leobianco/npov_PERL")
    self.assertEqual(repo_id, "leobianco/npov_PERL")
    self.assertIsNone(subfolder)
    self.assertIsNone(rev)

  def test_repo_with_subfolder(self):
    repo_id, subfolder, rev = parse_hf_repo_reference(
        "leobianco/npov_PERL/checkpoint-50"
    )
    self.assertEqual(repo_id, "leobianco/npov_PERL")
    self.assertEqual(subfolder, "checkpoint-50")
    self.assertIsNone(rev)

  def test_repo_with_colon_subfolder(self):
    repo_id, subfolder, rev = parse_hf_repo_reference(
        "leobianco/npov_PERL:checkpoint-50"
    )
    self.assertEqual(repo_id, "leobianco/npov_PERL")
    self.assertEqual(subfolder, "checkpoint-50")
    self.assertIsNone(rev)

  def test_repo_with_revision(self):
    repo_id, subfolder, rev = parse_hf_repo_reference(
        "leobianco/npov_PERL@v1.0"
    )
    self.assertEqual(repo_id, "leobianco/npov_PERL")
    self.assertIsNone(subfolder)
    self.assertEqual(rev, "v1.0")

  def test_repo_url(self):
    repo_id, subfolder, rev = parse_hf_repo_reference(
        "https://huggingface.co/leobianco/npov_PERL/checkpoint-100"
    )
    self.assertEqual(repo_id, "leobianco/npov_PERL")
    self.assertEqual(subfolder, "checkpoint-100")
    self.assertIsNone(rev)

  def test_hf_protocol_url(self):
    repo_id, subfolder, rev = parse_hf_repo_reference(
        "hf://leobianco/npov_PERL"
    )
    self.assertEqual(repo_id, "leobianco/npov_PERL")
    self.assertIsNone(subfolder)
    self.assertIsNone(rev)

  def test_hf_tree_url(self):
    repo_id, subfolder, rev = parse_hf_repo_reference(
        "https://huggingface.co/leobianco/npov_PERL/tree/main/checkpoint-50"
    )
    self.assertEqual(repo_id, "leobianco/npov_PERL")
    self.assertEqual(subfolder, "checkpoint-50")
    self.assertEqual(rev, "main")


class TestWandbResumptionCallback(unittest.TestCase):
  """Tests for WandbResumptionCallback."""

  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)

  def test_on_train_begin_saves_wandb_run_id(self):
    """Verify that on_train_begin saves the WandB run ID to output_dir and uploads to Hub."""
    callback = WandbResumptionCallback()
    mock_args = MagicMock()
    mock_args.output_dir = self.temp_dir
    mock_args.push_to_hub = True
    mock_args.hub_model_id = "leobianco/test_hub_model"

    mock_state = MagicMock()
    mock_state.is_world_process_zero = True
    mock_state.global_step = 0

    mock_control = MagicMock()

    mock_wandb = MagicMock()
    mock_wandb.run.id = "wandb_test_run_12345"

    mock_hf_api = MagicMock()
    with (
        patch("src.pipelines.wandb", mock_wandb),
        patch("src.pipelines.HfApi", return_value=mock_hf_api),
    ):
      callback.on_train_begin(mock_args, mock_state, mock_control)

    id_file = os.path.join(self.temp_dir, "wandb_run_id.txt")
    self.assertTrue(os.path.exists(id_file))
    with open(id_file, "r") as f:
      self.assertEqual(f.read().strip(), "wandb_test_run_12345")

    mock_hf_api.upload_file.assert_called_once()

  def test_on_save_writes_run_id_inside_checkpoint_dir(self):
    """Verify that on_save writes the WandB run ID inside the checkpoint folder and uploads to Hub."""
    callback = WandbResumptionCallback()
    mock_args = MagicMock()
    mock_args.output_dir = self.temp_dir
    mock_args.push_to_hub = True
    mock_args.hub_model_id = "leobianco/test_hub_model"

    checkpoint_dir = os.path.join(self.temp_dir, "checkpoint-50")
    os.makedirs(checkpoint_dir, exist_ok=True)

    mock_state = MagicMock()
    mock_state.is_world_process_zero = True
    mock_state.global_step = 50

    mock_control = MagicMock()

    mock_wandb = MagicMock()
    mock_wandb.run.id = "wandb_test_run_50"

    mock_hf_api = MagicMock()
    with (
        patch("src.pipelines.wandb", mock_wandb),
        patch("src.pipelines.HfApi", return_value=mock_hf_api),
    ):
      callback.on_save(mock_args, mock_state, mock_control)

    id_file = os.path.join(checkpoint_dir, "wandb_run_id.txt")
    self.assertTrue(os.path.exists(id_file))
    with open(id_file, "r") as f:
      self.assertEqual(f.read().strip(), "wandb_test_run_50")

    mock_hf_api.upload_file.assert_called_once()


class TestPipelineResumptionHelpers(unittest.TestCase):
  """Tests for Pipeline resumption helper methods."""

  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()
    self.pipeline = DummyPipeline()
    self.mock_training_args = MagicMock()
    self.mock_training_args.output_dir = self.temp_dir
    self.pipeline.training_args = self.mock_training_args

    # Clean any pre-existing WANDB env vars
    os.environ.pop("WANDB_RUN_ID", None)
    os.environ.pop("WANDB_RESUME", None)

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)
    os.environ.pop("WANDB_RUN_ID", None)
    os.environ.pop("WANDB_RESUME", None)

  def test_setup_wandb_resumption_restores_env(self):
    """Verify that _setup_wandb_resumption restores WANDB_RUN_ID and WANDB_RESUME from local dir."""
    id_file = os.path.join(self.temp_dir, "wandb_run_id.txt")
    with open(id_file, "w") as f:
      f.write("wandb_restored_id_999\n")

    self.mock_training_args.resume_from_checkpoint = True
    self.pipeline._setup_wandb_resumption()

    self.assertEqual(os.environ.get("WANDB_RUN_ID"), "wandb_restored_id_999")
    self.assertEqual(os.environ.get("WANDB_RESUME"), "allow")

  def test_setup_wandb_resumption_from_checkpoint_dir(self):
    """Verify that _setup_wandb_resumption finds wandb_run_id.txt in checkpoint dir."""
    ckpt_dir = os.path.join(self.temp_dir, "checkpoint-100")
    os.makedirs(ckpt_dir, exist_ok=True)
    id_file = os.path.join(ckpt_dir, "wandb_run_id.txt")
    with open(id_file, "w") as f:
      f.write("ckpt_wandb_id_888\n")

    self.mock_training_args.resume_from_checkpoint = ckpt_dir
    self.pipeline._setup_wandb_resumption()

    self.assertEqual(os.environ.get("WANDB_RUN_ID"), "ckpt_wandb_id_888")
    self.assertEqual(os.environ.get("WANDB_RESUME"), "allow")

  def test_setup_wandb_resumption_from_hf_hub(self):
    """Verify that _setup_wandb_resumption fetches wandb_run_id.txt from Hugging Face Hub."""
    hf_id_file = os.path.join(self.temp_dir, "hf_wandb_run_id.txt")
    with open(hf_id_file, "w") as f:
      f.write("hf_wandb_run_id_777\n")

    self.mock_training_args.resume_from_checkpoint = "leobianco/npov_PERL_run"
    self.mock_training_args.output_dir = None

    with patch("src.pipelines.hf_hub_download", return_value=hf_id_file):
      self.pipeline._setup_wandb_resumption()

    self.assertEqual(os.environ.get("WANDB_RUN_ID"), "hf_wandb_run_id_777")
    self.assertEqual(os.environ.get("WANDB_RESUME"), "allow")

  def test_setup_wandb_resumption_ignores_when_resume_is_none_or_false(self):
    """Verify that _setup_wandb_resumption does not restore run ID when resume_from_checkpoint is None/False even if id file exists."""
    id_file = os.path.join(self.temp_dir, "wandb_run_id.txt")
    with open(id_file, "w") as f:
      f.write("stale_run_id_12345\n")

    for val in (None, False, "False", "false", "None", "none", "no"):
      os.environ.pop("WANDB_RUN_ID", None)
      os.environ.pop("WANDB_RESUME", None)
      self.mock_training_args.resume_from_checkpoint = val
      self.pipeline._setup_wandb_resumption()
      self.assertIsNone(os.environ.get("WANDB_RUN_ID"))
      self.assertIsNone(os.environ.get("WANDB_RESUME"))

  def test_setup_wandb_resumption_ignores_auto_in_sweep(self):
    """Verify that auto resumption does not restore old run ID when running in a W&B sweep."""
    id_file = os.path.join(self.temp_dir, "wandb_run_id.txt")
    with open(id_file, "w") as f:
      f.write("sweep_stale_id_999\n")

    os.environ["WANDB_SWEEP_ID"] = "test_sweep_123"
    try:
      self.mock_training_args.resume_from_checkpoint = True
      self.pipeline._setup_wandb_resumption()
      self.assertIsNone(os.environ.get("WANDB_RUN_ID"))
      self.assertIsNone(os.environ.get("WANDB_RESUME"))
    finally:
      os.environ.pop("WANDB_SWEEP_ID", None)

  def test_resolve_resume_checkpoint_none(self):
    """Verify that None/False returns None."""
    self.mock_training_args.resume_from_checkpoint = None
    self.assertIsNone(self.pipeline._resolve_resume_checkpoint())

    self.mock_training_args.resume_from_checkpoint = False
    self.assertIsNone(self.pipeline._resolve_resume_checkpoint())

    self.mock_training_args.resume_from_checkpoint = "False"
    self.assertIsNone(self.pipeline._resolve_resume_checkpoint())

  def test_resolve_resume_checkpoint_explicit_path(self):
    """Verify that an explicit existing checkpoint path is returned."""
    ckpt_dir = os.path.join(self.temp_dir, "checkpoint-200")
    os.makedirs(ckpt_dir, exist_ok=True)

    with patch("src.pipelines.get_last_checkpoint", return_value=None):
      self.mock_training_args.resume_from_checkpoint = ckpt_dir
      resolved = self.pipeline._resolve_resume_checkpoint()
      self.assertEqual(resolved, ckpt_dir)

  def test_resolve_resume_checkpoint_auto(self):
    """Verify that auto/True resolves to the latest checkpoint."""
    ckpt_100 = os.path.join(self.temp_dir, "checkpoint-100")
    os.makedirs(ckpt_100, exist_ok=True)

    with patch("src.pipelines.get_last_checkpoint", return_value=ckpt_100):
      self.mock_training_args.resume_from_checkpoint = "True"
      resolved = self.pipeline._resolve_resume_checkpoint()
      self.assertEqual(resolved, ckpt_100)

  def test_resolve_resume_checkpoint_from_hf_hub_repo(self):
    """Verify that a Hugging Face Hub repo path is downloaded and resolved."""
    downloaded_hf_dir = os.path.join(self.temp_dir, "hf_download")
    os.makedirs(downloaded_hf_dir, exist_ok=True)

    self.mock_training_args.resume_from_checkpoint = "leobianco/npov_PERL_run"
    self.mock_training_args.output_dir = None

    with patch.object(
        self.pipeline, "_download_hf_checkpoint", return_value=downloaded_hf_dir
    ) as mock_download:
      resolved = self.pipeline._resolve_resume_checkpoint()
      mock_download.assert_called_once_with(
          "leobianco/npov_PERL_run", None, None
      )
      self.assertEqual(resolved, downloaded_hf_dir)

  def test_resolve_resume_checkpoint_from_hf_hub_subfolder(self):
    """Verify that a Hugging Face Hub subfolder path is downloaded and resolved."""
    downloaded_hf_dir = os.path.join(
        self.temp_dir, "hf_download", "checkpoint-50"
    )
    os.makedirs(downloaded_hf_dir, exist_ok=True)

    self.mock_training_args.resume_from_checkpoint = (
        "leobianco/npov_PERL_run/checkpoint-50"
    )
    self.mock_training_args.output_dir = None

    with patch.object(
        self.pipeline, "_download_hf_checkpoint", return_value=downloaded_hf_dir
    ) as mock_download:
      resolved = self.pipeline._resolve_resume_checkpoint()
      mock_download.assert_called_once_with(
          "leobianco/npov_PERL_run", "checkpoint-50", None
      )
      self.assertEqual(resolved, downloaded_hf_dir)

  def test_download_hf_checkpoint_with_nested_checkpoint(self):
    """Verify that _download_hf_checkpoint detects nested checkpoint-XXX directories."""
    downloaded_dir = os.path.join(self.temp_dir, "hf_repo")
    nested_ckpt = os.path.join(downloaded_dir, "checkpoint-150")
    os.makedirs(nested_ckpt, exist_ok=True)

    with (
        patch("src.pipelines.snapshot_download", return_value=downloaded_dir),
        patch("src.pipelines.get_last_checkpoint", return_value=nested_ckpt),
    ):
      resolved = self.pipeline._download_hf_checkpoint(
          "leobianco/npov_PERL_run"
      )
      self.assertEqual(resolved, nested_ckpt)


class TestPipelineRunAndSaveResumption(unittest.TestCase):
  """Test that pipeline run_and_save passes resume_from_checkpoint to trainer."""

  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)

  def test_perl_pipeline_run_and_save_calls_train_with_checkpoint(self):
    pipeline = PERLPipeline()
    pipeline.training_args = MagicMock()
    pipeline.training_args.do_train = True
    pipeline.training_args.output_dir = self.temp_dir

    ckpt_dir = os.path.join(self.temp_dir, "checkpoint-50")
    os.makedirs(ckpt_dir, exist_ok=True)
    pipeline.training_args.resume_from_checkpoint = ckpt_dir

    mock_trainer = MagicMock()
    pipeline.trainer = mock_trainer

    with patch("src.pipelines.get_last_checkpoint", return_value=None):
      pipeline.run_and_save()
      mock_trainer.train.assert_called_once_with(
          resume_from_checkpoint=ckpt_dir
      )
      mock_trainer.push_to_hub.assert_called_once()

  def test_sft_pipeline_run_and_save_calls_train_with_checkpoint(self):
    pipeline = SFTPipeline()
    pipeline.training_args = MagicMock()
    pipeline.training_args.output_dir = self.temp_dir
    pipeline.training_args.resume_from_checkpoint = None
    pipeline.training_args.push_to_hub = True

    mock_trainer = MagicMock()
    pipeline.trainer = mock_trainer

    pipeline.run_and_save()
    mock_trainer.train.assert_called_once_with(resume_from_checkpoint=None)
    mock_trainer.push_to_hub.assert_called_once()


class TestPipelinePushToHubRetries(unittest.TestCase):
  """Push operations across all pipelines must retry on transient 500s."""

  def test_push_to_hub_with_retry_succeeds_after_transient_500(self):
    calls = []

    def _flaky():
      calls.append(1)
      if len(calls) < 3:
        raise RuntimeError("500 Internal Server Error: transient hub blip")
      return "uploaded"

    res = _push_to_hub_with_retry(_flaky, attempts=3, base_delay_s=0.0)
    self.assertEqual(res, "uploaded")
    self.assertEqual(len(calls), 3)

  def test_push_to_hub_with_retry_raises_after_exhaustion(self):
    calls = []

    def _broken():
      calls.append(1)
      raise RuntimeError("500 Internal Server Error")

    with self.assertRaises(RuntimeError) as ctx:
      _push_to_hub_with_retry(_broken, attempts=3, base_delay_s=0.0)
    self.assertIn("500", str(ctx.exception))
    self.assertEqual(len(calls), 3)

  def test_sft_pipeline_push_retries_transient_error(self):
    pipeline = SFTPipeline()
    pipeline.training_args = MagicMock(push_to_hub=True)
    pipeline._resolve_resume_checkpoint = MagicMock(return_value=None)

    calls = []

    def _flaky_push():
      calls.append(1)
      if len(calls) < 2:
        raise RuntimeError("500 Hub Blip")
      return None

    mock_trainer = MagicMock()
    mock_trainer.push_to_hub.side_effect = _flaky_push
    pipeline.trainer = mock_trainer

    with patch("time.sleep"):
      pipeline.run_and_save()
    self.assertEqual(len(calls), 2)

  def test_reward_model_pipeline_push_retries_transient_error(self):
    pipeline = RewardModelPipeline()
    pipeline.training_args = MagicMock(
        push_to_hub=True, hub_model_id="test/rm", output_dir=None
    )
    pipeline._resolve_resume_checkpoint = MagicMock(return_value=None)
    pipeline.trainer = MagicMock()

    calls = []

    def _flaky_push(*_args, **_kwargs):
      calls.append(1)
      if len(calls) < 2:
        raise RuntimeError("500 Hub Blip")
      return None

    pipeline.model = MagicMock()
    pipeline.model.push_to_hub.side_effect = _flaky_push

    with patch("time.sleep"):
      pipeline.run_and_save()
    self.assertEqual(len(calls), 2)

  def test_perl_pipeline_push_retries_transient_error(self):
    pipeline = PERLPipeline()
    pipeline.training_args = MagicMock(push_to_hub=True, do_train=True)
    pipeline._resolve_resume_checkpoint = MagicMock(return_value=None)

    calls = []

    def _flaky_push():
      calls.append(1)
      if len(calls) < 2:
        raise RuntimeError("500 Hub Blip")
      return None

    mock_trainer = MagicMock()
    mock_trainer.push_to_hub.side_effect = _flaky_push
    pipeline.trainer = mock_trainer

    with patch("time.sleep"):
      pipeline.run_and_save()
    self.assertEqual(len(calls), 2)

  def test_dpo_pipeline_push_retries_transient_error(self):
    pipeline = DPOPipeline()
    pipeline.training_args = MagicMock(push_to_hub=True, do_train=True)
    pipeline._resolve_resume_checkpoint = MagicMock(return_value=None)

    calls = []

    def _flaky_push():
      calls.append(1)
      if len(calls) < 2:
        raise RuntimeError("500 Hub Blip")
      return None

    mock_trainer = MagicMock()
    mock_trainer.push_to_hub.side_effect = _flaky_push
    pipeline.trainer = mock_trainer

    with patch("time.sleep"):
      pipeline.run_and_save()
    self.assertEqual(len(calls), 2)


class TestPipelineArgumentSetup(unittest.TestCase):
  """Test that pipeline setup_arguments correctly handles arguments without conflicts."""

  def test_perl_pipeline_setup_arguments(self):
    pipeline = PERLPipeline()
    mock_script_args = MagicMock()
    mock_script_args.task_name = "bosch"
    mock_training_args = MagicMock()
    mock_training_args.seed = 130104
    mock_training_args.learning_rate = 2e-5
    mock_training_args.beta = 0.05
    mock_training_args.temperature = 0.3
    mock_training_args.num_train_epochs = 0.2
    mock_training_args.max_completion_length = 256
    mock_training_args.run_name = "test_perl_run"

    with patch("src.pipelines.HfArgumentParser") as mock_parser_cls:
      mock_parser = MagicMock()
      mock_parser.parse_args_into_dataclasses.return_value = (
          mock_script_args,
          mock_training_args,
      )
      mock_parser_cls.return_value = mock_parser

      pipeline.setup_arguments(
          "--task_name=bosch",
          "--dataset_repo_id=leobianco/bosch_perl",
          "--model_repo_id=google/gemma-4-E2B-it",
          "--max_completion_length=256",
          "--num_generations=4",
          "--num_iterations=1",
          "--steps_per_generation=16",
          "--learning_rate=2e-5",
          "--beta=0.05",
          "--temperature=0.3",
      )
      self.assertEqual(pipeline.args.task_name, "bosch")
      self.assertEqual(pipeline.training_args.max_completion_length, 256)
      self.assertEqual(pipeline.training_args.learning_rate, 2e-5)

  def test_perl_pipeline_setup_arguments_with_lora(self):
    pipeline = PERLPipeline()
    mock_script_args = MagicMock()
    mock_script_args.task_name = "npov"
    mock_training_args = MagicMock()
    mock_training_args.seed = 130104
    mock_training_args.learning_rate = 2e-5
    mock_training_args.beta = 0.05
    mock_training_args.temperature = 0.3
    mock_training_args.num_train_epochs = 1.0
    mock_training_args.run_name = None

    with patch("src.pipelines.HfArgumentParser") as mock_parser_cls:
      mock_parser = MagicMock()
      mock_parser.parse_args_into_dataclasses.return_value = (
          mock_script_args,
          mock_training_args,
      )
      mock_parser_cls.return_value = mock_parser

      pipeline.setup_arguments(
          "--task_name=npov",
          "--dataset_repo_id=leobianco/npov_perl",
          "--model_repo_id=google/gemma-4-E4B-it",
          "--lora_r=4",
          "--lora_alpha=8",
          "--lora_dropout=0.05",
      )
      self.assertEqual(pipeline._lora_args.lora_r, 4)
      self.assertEqual(pipeline._lora_args.lora_alpha, 8)
      self.assertEqual(pipeline._lora_args.lora_dropout, 0.05)
      import peft
      peft.LoraConfig.assert_called_with(
          task_type="CAUSAL_LM",
          peft_type="LORA",
          r=4,
          lora_alpha=8,
          lora_dropout=0.05,
      )
      self.assertIn("r4", pipeline.training_args.run_name)

  def test_perl_pipeline_setup_arguments_with_gradient_accumulation(self):
    pipeline = PERLPipeline()
    mock_script_args = MagicMock()
    mock_script_args.task_name = "npov"
    mock_training_args = MagicMock()
    mock_training_args.seed = 130104
    mock_training_args.learning_rate = 2e-5
    mock_training_args.beta = 0.05
    mock_training_args.temperature = 0.7
    mock_training_args.num_train_epochs = 1.0
    mock_training_args.gradient_accumulation_steps = 4
    mock_training_args.steps_per_generation = 16
    mock_training_args.run_name = None

    with patch("src.pipelines.HfArgumentParser") as mock_parser_cls:
      mock_parser = MagicMock()
      mock_parser.parse_args_into_dataclasses.return_value = (
          mock_script_args,
          mock_training_args,
      )
      mock_parser_cls.return_value = mock_parser

      pipeline.setup_arguments(
          "--task_name=npov",
          "--dataset_repo_id=leobianco/npov_perl",
          "--model_repo_id=google/gemma-4-E2B-it",
          "--gradient_accumulation_steps=4",
          "--steps_per_generation=16",
          "--temperature=0.7",
      )
      self.assertEqual(pipeline.training_args.gradient_accumulation_steps, 4)
      self.assertIn("gas4", pipeline.training_args.run_name)
      self.assertIn("T0.7", pipeline.training_args.run_name)

  def test_perl_pipeline_setup_arguments_with_reward_penalty_alpha(self):
    pipeline = PERLPipeline()
    mock_script_args = MagicMock()
    mock_script_args.task_name = "npov"
    mock_script_args.reward_penalty_alpha = 2.0
    mock_training_args = MagicMock()
    mock_training_args.seed = 130104
    mock_training_args.learning_rate = 2e-5
    mock_training_args.beta = 0.05
    mock_training_args.temperature = 0.7
    mock_training_args.num_train_epochs = 1.0
    mock_training_args.gradient_accumulation_steps = 8
    mock_training_args.steps_per_generation = 16
    mock_training_args.run_name = None

    with patch("src.pipelines.HfArgumentParser") as mock_parser_cls:
      mock_parser = MagicMock()
      mock_parser.parse_args_into_dataclasses.return_value = (
          mock_script_args,
          mock_training_args,
      )
      mock_parser_cls.return_value = mock_parser

      pipeline.setup_arguments(
          "--task_name=npov",
          "--dataset_repo_id=leobianco/npov_perl",
          "--model_repo_id=google/gemma-4-E2B-it",
          "--reward_penalty_alpha=2.0",
      )
      self.assertEqual(pipeline.args.reward_penalty_alpha, 2.0)
      self.assertIn("a2.0", pipeline.training_args.run_name)

  def test_perl_pipeline_setup_arguments_default_alpha_no_suffix(self):
    pipeline = PERLPipeline()
    mock_script_args = MagicMock()
    mock_script_args.task_name = "npov"
    mock_script_args.reward_penalty_alpha = 1.0
    mock_training_args = MagicMock()
    mock_training_args.seed = 130104
    mock_training_args.learning_rate = 2e-5
    mock_training_args.beta = 0.05
    mock_training_args.temperature = 0.7
    mock_training_args.num_train_epochs = 1.0
    mock_training_args.gradient_accumulation_steps = 1
    mock_training_args.steps_per_generation = 16
    mock_training_args.run_name = None

    with patch("src.pipelines.HfArgumentParser") as mock_parser_cls:
      mock_parser = MagicMock()
      mock_parser.parse_args_into_dataclasses.return_value = (
          mock_script_args,
          mock_training_args,
      )
      mock_parser_cls.return_value = mock_parser

      pipeline.setup_arguments(
          "--task_name=npov",
          "--dataset_repo_id=leobianco/npov_perl",
          "--model_repo_id=google/gemma-4-E2B-it",
      )
      self.assertEqual(pipeline.args.reward_penalty_alpha, 1.0)
      self.assertNotIn("_a", pipeline.training_args.run_name)

  def test_perl_pipeline_setup_arguments_invalid_steps_per_gen(self):
    pipeline = PERLPipeline()
    mock_script_args = MagicMock()
    mock_script_args.task_name = "npov"
    mock_training_args = MagicMock()
    mock_training_args.seed = 130104
    mock_training_args.learning_rate = 2e-5
    mock_training_args.beta = 0.05
    mock_training_args.temperature = 0.7
    mock_training_args.num_train_epochs = 1.0
    mock_training_args.gradient_accumulation_steps = 5
    mock_training_args.steps_per_generation = 16
    mock_training_args.run_name = None

    with patch("src.pipelines.HfArgumentParser") as mock_parser_cls:
      mock_parser = MagicMock()
      mock_parser.parse_args_into_dataclasses.return_value = (
          mock_script_args,
          mock_training_args,
      )
      mock_parser_cls.return_value = mock_parser

      with self.assertRaises(ValueError) as ctx:
        pipeline.setup_arguments(
            "--task_name=npov",
            "--dataset_repo_id=leobianco/npov_perl",
            "--model_repo_id=google/gemma-4-E2B-it",
            "--gradient_accumulation_steps=5",
            "--steps_per_generation=16",
        )
      self.assertIn("steps_per_generation (16) must be an integer multiple",
                    str(ctx.exception))

  def test_warmup_steps_configuration_with_gradient_accumulation(self):
    pipeline = PERLPipeline()
    pipeline.training_args = MagicMock()
    pipeline.training_args.warmup_ratio = 0.1
    pipeline.training_args.warmup_steps = 0
    pipeline.training_args.per_device_train_batch_size = 8
    pipeline.training_args.gradient_accumulation_steps = 4
    pipeline.training_args.num_train_epochs = 1
    pipeline.training_args.world_size = 2
    pipeline.data = {"train": list(range(640))}

    pipeline._configure_warmup_steps()
    self.assertEqual(pipeline.training_args.warmup_steps, 1)
    self.assertEqual(pipeline.training_args.warmup_ratio, 0.0)

  def test_perl_pipeline_setup_model_direct_from_base(self):
    pipeline = PERLPipeline()
    pipeline.args = MagicMock()
    pipeline.args.reward_model_path = "leobianco/reward_model"
    pipeline.args.sft_model_path = None
    pipeline.args.model_repo_id = "google/gemma-4-E4B-it"
    pipeline.training_args = MagicMock()
    pipeline.training_args.reward_model_path = None
    pipeline.training_args.sft_model_path = None
    pipeline.tokenizer = MagicMock()
    pipeline.tokenizer.pad_token_id = 0
    pipeline._lora_args = MagicMock()
    pipeline._lora_args.lora_r = 8
    pipeline._lora_args.lora_alpha = 16
    pipeline._lora_args.lora_dropout = 0.0
    pipeline._lora_args.task_type = "CAUSAL_LM"
    pipeline._lora_args.peft_type = "LORA"

    mock_policy_base = MagicMock()
    mock_fresh_peft = MagicMock()

    with patch(
        "src.pipelines.AutoModelForSequenceClassification.from_pretrained"
    ), patch("src.pipelines.AutoTokenizer.from_pretrained"), patch(
        "src.pipelines.AutoModelForCausalLM.from_pretrained",
        return_value=mock_policy_base,
    ), patch(
        "src.pipelines.get_peft_model", return_value=mock_fresh_peft
    ) as mock_get_peft:
      pipeline.setup_model()
      mock_get_peft.assert_called_once_with(
          mock_policy_base, pipeline._lora_config
      )
      self.assertEqual(pipeline.policy, mock_fresh_peft)

  def test_perl_pipeline_setup_model_with_sft_lora_merge(self):
    pipeline = PERLPipeline()
    pipeline.args = MagicMock()
    pipeline.args.reward_model_path = "leobianco/reward_model"
    pipeline.args.model_repo_id = "google/gemma-4-E4B-it"
    pipeline.training_args = MagicMock()
    pipeline.training_args.reward_model_path = None
    pipeline.training_args.sft_model_path = None
    pipeline.tokenizer = MagicMock()
    pipeline.tokenizer.pad_token_id = 0

    temp_sft_dir = tempfile.mkdtemp()
    try:
      config_path = os.path.join(temp_sft_dir, "adapter_config.json")
      with open(config_path, "w") as f:
        json.dump({"r": 8, "lora_alpha": 16}, f)

      pipeline.args.sft_model_path = temp_sft_dir
      pipeline._lora_args = MagicMock()
      pipeline._lora_args.lora_r = 8
      pipeline._lora_args.lora_alpha = 16
      pipeline._lora_args.lora_dropout = 0.0
      pipeline._lora_args.task_type = "CAUSAL_LM"
      pipeline._lora_args.peft_type = "LORA"

      mock_policy_base = MagicMock()
      mock_sft_peft = MagicMock()
      mock_merged_base = MagicMock()
      mock_sft_peft.merge_and_unload.return_value = mock_merged_base
      mock_fresh_peft = MagicMock()

      with patch(
          "src.pipelines.AutoModelForSequenceClassification.from_pretrained"
      ), patch("src.pipelines.AutoTokenizer.from_pretrained"), patch(
          "src.pipelines.AutoModelForCausalLM.from_pretrained",
          return_value=mock_policy_base,
      ), patch(
          "src.pipelines.PeftModel.from_pretrained", return_value=mock_sft_peft
      ) as mock_peft_from_pretrained, patch(
          "src.pipelines.get_peft_model", return_value=mock_fresh_peft
      ) as mock_get_peft:
        pipeline.setup_model()
        mock_peft_from_pretrained.assert_called_once_with(
            mock_policy_base, temp_sft_dir
        )
        mock_sft_peft.merge_and_unload.assert_called_once()
        mock_get_peft.assert_called_once_with(
            mock_merged_base, pipeline._lora_config
        )
        self.assertEqual(pipeline.policy, mock_fresh_peft)
    finally:
      shutil.rmtree(temp_sft_dir, ignore_errors=True)

  def test_perl_pipeline_setup_model_warns_on_rank_mismatch(self):
    pipeline = PERLPipeline()
    pipeline.args = MagicMock()
    pipeline.args.reward_model_path = "leobianco/reward_model"
    pipeline.args.model_repo_id = "google/gemma-4-E4B-it"
    pipeline.training_args = MagicMock()
    pipeline.training_args.reward_model_path = None
    pipeline.training_args.sft_model_path = None
    pipeline.tokenizer = MagicMock()
    pipeline.tokenizer.pad_token_id = 0

    temp_sft_dir = tempfile.mkdtemp()
    try:
      config_path = os.path.join(temp_sft_dir, "adapter_config.json")
      with open(config_path, "w") as f:
        json.dump({"r": 8, "lora_alpha": 16}, f)

      pipeline.args.sft_model_path = temp_sft_dir
      pipeline._lora_args = MagicMock()
      pipeline._lora_args.lora_r = 4
      pipeline._lora_args.lora_alpha = 8
      pipeline._lora_args.lora_dropout = 0.0
      pipeline._lora_args.task_type = "CAUSAL_LM"
      pipeline._lora_args.peft_type = "LORA"

      mock_policy_base = MagicMock()
      mock_sft_peft = MagicMock()
      mock_merged_base = MagicMock()
      mock_sft_peft.merge_and_unload.return_value = mock_merged_base

      with patch(
          "src.pipelines.AutoModelForSequenceClassification.from_pretrained"
      ), patch("src.pipelines.AutoTokenizer.from_pretrained"), patch(
          "src.pipelines.AutoModelForCausalLM.from_pretrained",
          return_value=mock_policy_base,
      ), patch(
          "src.pipelines.PeftModel.from_pretrained", return_value=mock_sft_peft
      ), patch(
          "src.pipelines.get_peft_model"
      ), patch(
          "builtins.print"
      ) as mock_print:
        pipeline.setup_model()
        warning_printed = any(
            "[WARNING] LoRA Rank Mismatch Detected in PE-RL:" in str(call)
            for call in mock_print.call_args_list
        )
        self.assertTrue(warning_printed)
    finally:
      shutil.rmtree(temp_sft_dir, ignore_errors=True)

  def test_perl_pipeline_process_data_zero_fewshot(self):
    pipeline = PERLPipeline()
    pipeline.args = MagicMock()
    pipeline.args.num_fewshot = 0
    mock_dataset = _TestMockDataset({"prompt": ["p1", "p2"]})
    pipeline.data = {"train": mock_dataset}
    pipeline.process_data()
    self.assertEqual(pipeline.data["train"]["prompt"], ["p1", "p2"])


class TestRewardPenaltyAlpha(unittest.TestCase):
  """Unit tests for asymmetric reward penalty alpha scaling."""

  def test_reward_penalty_alpha_applied_to_negative_diffs(self):
    pipeline = PERLPipeline()
    pipeline.args = MagicMock()
    pipeline.args.reward_penalty_alpha = 2.0
    pipeline.reward_model = MagicMock()
    pipeline.reward_tokenizer = MagicMock()
    pipeline.policy = MagicMock()
    pipeline.training_args = MagicMock()
    pipeline.data = {"train": [], "test": None}
    pipeline.tokenizer = MagicMock()

    mock_diff = MagicMock()
    mock_where_result = MagicMock()
    mock_where_result.cpu.return_value.tolist.return_value = [1.5, -2.0]

    with patch("src.pipelines.RLOOTrainer") as mock_rloo_cls, patch(
        "src.pipelines.torch.where", return_value=mock_where_result
    ) as mock_torch_where:
      pipeline.setup_trainer()
      reward_fn = mock_rloo_cls.call_args.kwargs["reward_funcs"]

      mock_logits = MagicMock()
      mock_logits.__getitem__.side_effect = lambda idx: (
          mock_diff if idx[1] in (0, 1) else MagicMock()
      )
      mock_diff.__sub__.return_value = mock_diff
      mock_diff.__lt__.return_value = MagicMock()
      mock_diff.__mul__.return_value = MagicMock()
      pipeline.reward_model.return_value.logits = mock_logits

      rewards = reward_fn(["prompt"], ["completion"])
      self.assertEqual(rewards, [1.5, -2.0])
      mock_torch_where.assert_called_once()

  def test_reward_penalty_alpha_identity_when_1(self):
    pipeline = PERLPipeline()
    pipeline.args = MagicMock()
    pipeline.args.reward_penalty_alpha = 1.0
    pipeline.reward_model = MagicMock()
    pipeline.reward_tokenizer = MagicMock()
    pipeline.policy = MagicMock()
    pipeline.training_args = MagicMock()
    pipeline.data = {"train": [], "test": None}
    pipeline.tokenizer = MagicMock()

    mock_diff = MagicMock()
    mock_diff.cpu.return_value.tolist.return_value = [1.5, -1.0]

    with patch("src.pipelines.RLOOTrainer") as mock_rloo_cls, patch(
        "src.pipelines.torch.where"
    ) as mock_torch_where:
      pipeline.setup_trainer()
      reward_fn = mock_rloo_cls.call_args.kwargs["reward_funcs"]

      mock_logits = MagicMock()
      mock_logits.__getitem__.return_value = mock_diff
      mock_diff.__sub__.return_value = mock_diff
      pipeline.reward_model.return_value.logits = mock_logits

      rewards = reward_fn(["prompt"], ["completion"])
      self.assertEqual(rewards, [1.5, -1.0])
      mock_torch_where.assert_not_called()

  def test_asymmetric_reward_penalty_math(self):
    diffs = [1.5, 0.2, 0.0, -0.5, -1.2]
    alpha = 2.0
    expected = [1.5, 0.2, 0.0, -1.0, -2.4]
    computed = [alpha * d if d < 0 else d for d in diffs]
    for c, e in zip(computed, expected):
      self.assertAlmostEqual(c, e)


class TestRewardMaxLength(unittest.TestCase):
  """Tests for the reward-model token budget and its truncation side.

  The PE-RL scoring path used to tokenize prompt+completion at
  ``max_length=512`` with the default right truncation. RAGTruth prompts
  alone exceed 512 tokens in the overwhelming majority of rows, so the
  completion was cut off entirely: every rollout of a prompt scored
  identically, the RLOO advantage was exactly zero, and only the KL term was
  optimized. Nothing crashed. These tests make that regression loud.
  """

  def test_resolver_returns_configured_value(self):
    from src.pipelines import _resolve_reward_max_length

    args = MagicMock()
    args.reward_max_length = 4096
    self.assertEqual(_resolve_reward_max_length(args), 4096)

  def test_resolver_falls_back_on_unusable_values(self):
    """A test double or a nonsensical limit must not silently shrink."""
    from src.pipelines import _DEFAULT_REWARD_MAX_LENGTH
    from src.pipelines import _resolve_reward_max_length

    for bad in (None, 0, -1, True, "2048", MagicMock()):
      args = MagicMock()
      args.reward_max_length = bad
      self.assertEqual(
          _resolve_reward_max_length(args),
          _DEFAULT_REWARD_MAX_LENGTH,
          f"unexpected budget for {bad!r}",
      )

  def test_default_budget_leaves_room_for_the_completion(self):
    """The budget must exceed what scripts/perl.sh can generate."""
    from src.pipelines import _DEFAULT_REWARD_MAX_LENGTH

    # scripts/perl.sh caps generation at 256 tokens.
    self.assertGreater(_DEFAULT_REWARD_MAX_LENGTH, 256)
    # And the old value is exactly the bug being prevented.
    self.assertNotEqual(_DEFAULT_REWARD_MAX_LENGTH, 512)

  def _build_perl_pipeline(self, budget):
    pipeline = PERLPipeline()
    pipeline.args = MagicMock()
    pipeline.args.reward_penalty_alpha = 1.0
    pipeline.args.reward_max_length = budget
    pipeline.reward_model = MagicMock()
    pipeline.reward_tokenizer = MagicMock()
    pipeline.policy = MagicMock()
    pipeline.training_args = MagicMock()
    pipeline.data = {"train": [], "test": None}
    pipeline.tokenizer = MagicMock()
    return pipeline

  def test_reward_fn_tokenizes_at_the_configured_budget(self):
    pipeline = self._build_perl_pipeline(4096)

    mock_diff = MagicMock()
    mock_diff.cpu.return_value.tolist.return_value = [0.5]

    with patch("src.pipelines.RLOOTrainer") as mock_rloo_cls, patch(
        "src.pipelines.torch.where"
    ):
      pipeline.setup_trainer()
      reward_fn = mock_rloo_cls.call_args.kwargs["reward_funcs"]

      mock_logits = MagicMock()
      mock_logits.__getitem__.return_value = mock_diff
      mock_diff.__sub__.return_value = mock_diff
      pipeline.reward_model.return_value.logits = mock_logits

      reward_fn(["prompt"], ["completion"])

    kwargs = pipeline.reward_tokenizer.call_args.kwargs
    self.assertEqual(kwargs["max_length"], 4096)
    self.assertTrue(kwargs["truncation"])

  def test_reward_fn_truncates_from_the_left(self):
    """The completion lives at the tail and is the only part that varies.

    Right truncation drops it, which is precisely how the reward signal went
    flat without any error being raised.
    """
    pipeline = self._build_perl_pipeline(2048)

    with patch("src.pipelines.RLOOTrainer"), patch("src.pipelines.torch.where"):
      pipeline.setup_trainer()

    self.assertEqual(pipeline.reward_tokenizer.truncation_side, "left")

  def test_training_and_scoring_share_one_budget(self):
    """Train/serve skew here is silent, so pin the parity explicitly.

    The reward model is trained by `RewardModelPipeline.process_data` and
    queried by `PERLPipeline`. If the two tokenize differently the model is
    trained on one view of the text and queried on another.
    """
    from src.pipelines import RewardModelPipeline
    from src.pipelines import _REWARD_TRUNCATION_SIDE

    rm = RewardModelPipeline()
    rm.args = MagicMock()
    rm.args.reward_max_length = 4096
    rm.tokenizer = MagicMock()
    rm.tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
    rm.training_args = MagicMock()
    rm.training_args.fp16 = False
    rm.training_args.bf16 = True
    # process_data builds a LoRA config and runs the synthetic-augmentation
    # hook before it tokenizes.
    rm._lora_args = MagicMock()
    rm._llm_synth_args = MagicMock()

    class _Split:

      # `process_data` drops a stale `labels` column before tokenizing, so
      # the double has to answer this even though the test is about the
      # tokenization budget.
      column_names = ["prompt", "label"]

      def map(self, fn, batched=False):
        del batched
        fn({"prompt": ["text"]})
        return self

      def set_format(self, _):
        return self

      # `process_data` fails loudly on an empty training split.
      def __len__(self):
        return 1

      @property
      def features(self):
        return {"label": None}

      def cast(self, _):
        return self

    rm.data = {"train": _Split()}

    with patch("src.pipelines.get_task_processor") as mock_get_processor, patch(
        "src.pipelines.Value"
    ):
      mock_get_processor.return_value.augment_training_split.side_effect = (
          lambda split, *a, **k: split
      )
      rm.process_data()

    self.assertEqual(rm.tokenizer.truncation_side, _REWARD_TRUNCATION_SIDE)
    self.assertEqual(rm.tokenizer.call_args.kwargs["max_length"], 4096)


class TestBestAndLastCheckpointSaving(unittest.TestCase):
  """Tests for preserving both the best reward checkpoint and the very last step checkpoint locally."""

  def test_callback_bumps_save_total_limit_to_2_and_triggers_final_save(self):
    from src.pipelines import WandbResumptionCallback

    cb = WandbResumptionCallback()
    args = MagicMock()
    args.load_best_model_at_end = True
    args.save_strategy = "steps"
    args.save_total_limit = 1
    args.do_eval = True
    args.eval_strategy = "steps"
    args.output_dir = None

    state = MagicMock()
    state.is_world_process_zero = True
    state.max_steps = 220
    state.global_step = 220

    control = MagicMock()
    control.should_save = False
    control.should_evaluate = False
    control.should_training_stop = False

    cb.on_train_begin(args, state, control)
    self.assertEqual(args.save_total_limit, 2)

    cb.on_step_end(args, state, control)
    self.assertTrue(control.should_save)
    self.assertTrue(control.should_evaluate)

  def test_load_best_model_wrapper_saves_last_step_and_prunes_intermediates(self):
    pipeline = PERLPipeline()
    with tempfile.TemporaryDirectory() as tmpdir:
      best_dir = os.path.join(tmpdir, "checkpoint-150")
      mid_dir = os.path.join(tmpdir, "checkpoint-200")
      os.makedirs(best_dir)
      os.makedirs(mid_dir)
      with open(os.path.join(best_dir, "adapter_config.json"), "w") as f:
        f.write("{}")
      with open(os.path.join(mid_dir, "adapter_config.json"), "w") as f:
        f.write("{}")

      args = MagicMock()
      args.output_dir = tmpdir
      args.load_best_model_at_end = True
      args.save_strategy = "steps"
      args.save_total_limit = 2

      state = MagicMock()
      state.global_step = 220
      state.is_world_process_zero = True
      state.best_model_checkpoint = best_dir

      trainer = MagicMock()
      trainer.args = args
      trainer.state = state
      orig_load_called = []

      def fake_orig_load():
        orig_load_called.append(True)

      trainer._load_best_model = fake_orig_load

      def fake_save_model(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
          f.write('{"step": 220}')

      trainer.save_model.side_effect = fake_save_model

      pipeline.trainer = trainer
      pipeline.training_args = args
      pipeline._configure_best_and_last_checkpoint_saving()

      # Trigger wrapped _load_best_model
      trainer._load_best_model()

      self.assertEqual(len(orig_load_called), 1)
      last_dir = os.path.join(tmpdir, "checkpoint-220")
      self.assertTrue(os.path.isdir(best_dir), "Best checkpoint (150) must be preserved")
      self.assertTrue(os.path.isdir(last_dir), "Last step checkpoint (220) must be saved and preserved")
      self.assertFalse(os.path.exists(mid_dir), "Intermediate checkpoint (200) must be pruned")

  def test_resolve_lora_adapter_path_excludes_ref_subfolder(self):
    pipeline = PERLPipeline()
    with tempfile.TemporaryDirectory() as tmpdir:
      # Simulate a parent directory where get_last_checkpoint returns None
      # and recursive glob finds both checkpoint-220/adapter_config.json and checkpoint-220/ref/adapter_config.json
      ckpt_dir = os.path.join(tmpdir, "nested", "checkpoint-220")
      ref_dir = os.path.join(ckpt_dir, "ref")
      os.makedirs(ref_dir)
      with open(os.path.join(ckpt_dir, "adapter_config.json"), "w") as f:
        f.write("{}")
      with open(os.path.join(ref_dir, "adapter_config.json"), "w") as f:
        f.write("{}")

      with patch("src.pipelines.get_last_checkpoint", return_value=None):
        enable_lora, resolved = pipeline._resolve_lora_adapter_path(tmpdir)
        self.assertTrue(enable_lora)
        self.assertEqual(resolved, ckpt_dir)


class TestBestCheckpointArchiver(unittest.TestCase):
  """The best checkpoint must survive save_total_limit rotation."""

  def _args_and_state(self, output_dir, best_dir, step):
    args = MagicMock()
    args.output_dir = output_dir
    state = MagicMock()
    state.is_world_process_zero = True
    state.best_model_checkpoint = best_dir
    state.best_global_step = step
    state.best_metric = 0.42
    return args, state

  def _make_checkpoint(self, path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "adapter_config.json"), "w") as f:
      f.write("{}")
    with open(os.path.join(path, "adapter_model.safetensors"), "w") as f:
      f.write("weights")
    # Optimizer state is what we must NOT carry around.
    with open(os.path.join(path, "optimizer.pt"), "w") as f:
      f.write("x" * 1024)

  def test_archive_survives_deletion_of_the_original(self):
    from src.pipelines import BestCheckpointArchiver

    with tempfile.TemporaryDirectory() as tmpdir:
      best_dir = os.path.join(tmpdir, "checkpoint-150")
      self._make_checkpoint(best_dir)
      args, state = self._args_and_state(tmpdir, best_dir, 150)

      archiver = BestCheckpointArchiver()
      archiver.on_save(args, state, MagicMock())

      # Rotation removes the original checkpoint.
      shutil.rmtree(best_dir)

      self.assertIsNotNone(archiver.archive_dir)
      self.assertTrue(os.path.isdir(archiver.archive_dir))
      self.assertEqual(archiver.archived_step, 150)
      self.assertTrue(
          os.path.isfile(
              os.path.join(archiver.archive_dir, "adapter_model.safetensors")
          )
      )

  def test_optimizer_state_is_not_archived(self):
    from src.pipelines import BestCheckpointArchiver

    with tempfile.TemporaryDirectory() as tmpdir:
      best_dir = os.path.join(tmpdir, "checkpoint-150")
      self._make_checkpoint(best_dir)
      args, state = self._args_and_state(tmpdir, best_dir, 150)

      archiver = BestCheckpointArchiver()
      archiver.on_save(args, state, MagicMock())

      self.assertFalse(
          os.path.exists(os.path.join(archiver.archive_dir, "optimizer.pt"))
      )

  def test_a_new_best_replaces_the_previous_archive(self):
    from src.pipelines import BestCheckpointArchiver

    with tempfile.TemporaryDirectory() as tmpdir:
      first = os.path.join(tmpdir, "checkpoint-100")
      second = os.path.join(tmpdir, "checkpoint-200")
      self._make_checkpoint(first)
      self._make_checkpoint(second)
      with open(os.path.join(second, "marker.json"), "w") as f:
        f.write("{}")

      archiver = BestCheckpointArchiver()
      args, state = self._args_and_state(tmpdir, first, 100)
      archiver.on_save(args, state, MagicMock())

      args, state = self._args_and_state(tmpdir, second, 200)
      archiver.on_save(args, state, MagicMock())

      self.assertEqual(archiver.archived_step, 200)
      self.assertTrue(
          os.path.isfile(os.path.join(archiver.archive_dir, "marker.json"))
      )

  def test_a_missing_best_checkpoint_is_not_an_error(self):
    from src.pipelines import BestCheckpointArchiver

    with tempfile.TemporaryDirectory() as tmpdir:
      args, state = self._args_and_state(tmpdir, None, None)
      archiver = BestCheckpointArchiver()
      archiver.on_save(args, state, MagicMock())  # must not raise
      self.assertIsNone(archiver.archive_dir)


class TestPublishBestAndLast(unittest.TestCase):
  """Both checkpoints reach the Hub; only the default sits at the root."""

  def _pipeline(self, tmpdir, load_best_model_at_end, global_step=200):
    pipeline = PERLPipeline()
    args = MagicMock()
    args.output_dir = tmpdir
    args.push_to_hub = True
    args.hub_model_id = "leobianco/npov_PERL"
    args.load_best_model_at_end = load_best_model_at_end
    args.metric_for_best_model = "rewards/reward_fn/mean"
    args.greater_is_better = True
    pipeline.training_args = args

    state = MagicMock()
    state.is_world_process_zero = True
    state.global_step = global_step
    state.best_metric = 1.75
    state.best_model_checkpoint = None
    trainer = MagicMock()
    trainer.state = state
    pipeline.trainer = trainer
    return pipeline

  def _make_checkpoint(self, path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "adapter_config.json"), "w") as f:
      f.write("{}")

  def test_perl_publishes_the_best_checkpoint_under_a_subfolder(self):
    from src.pipelines import BestCheckpointArchiver

    with tempfile.TemporaryDirectory() as tmpdir:
      self._make_checkpoint(os.path.join(tmpdir, "checkpoint-200"))
      archive = os.path.join(tmpdir, BestCheckpointArchiver.ARCHIVE_DIRNAME)
      self._make_checkpoint(archive)

      pipeline = self._pipeline(tmpdir, load_best_model_at_end=False)
      archiver = BestCheckpointArchiver()
      archiver.archive_dir = archive
      archiver.archived_step = 120
      pipeline._best_archiver = archiver

      with patch("src.pipelines.HfApi") as mock_api:
        pipeline._publish_best_and_last(description="PERL")

      api = mock_api.return_value
      api.upload_folder.assert_called_once()
      kwargs = api.upload_folder.call_args.kwargs
      self.assertEqual(kwargs["path_in_repo"], "best")
      self.assertEqual(kwargs["folder_path"], archive)
      self.assertIn("optimizer.pt", kwargs["ignore_patterns"])

      manifest_path = os.path.join(tmpdir, "checkpoints.json")
      self.assertTrue(os.path.isfile(manifest_path))
      with open(manifest_path) as f:
        manifest = json.load(f)
      self.assertEqual(manifest["default"], "last")
      self.assertEqual(manifest["default_step"], 200)
      self.assertEqual(manifest["checkpoints"]["best"]["subfolder"], "best")
      self.assertIsNone(manifest["checkpoints"]["last"]["subfolder"])

  def test_sft_style_run_publishes_the_last_checkpoint_under_a_subfolder(self):
    from src.pipelines import BestCheckpointArchiver

    with tempfile.TemporaryDirectory() as tmpdir:
      last_dir = os.path.join(tmpdir, "checkpoint-200")
      self._make_checkpoint(last_dir)
      archive = os.path.join(tmpdir, BestCheckpointArchiver.ARCHIVE_DIRNAME)
      self._make_checkpoint(archive)

      pipeline = self._pipeline(tmpdir, load_best_model_at_end=True)
      archiver = BestCheckpointArchiver()
      archiver.archive_dir = archive
      archiver.archived_step = 120
      pipeline._best_archiver = archiver

      with patch("src.pipelines.HfApi") as mock_api:
        pipeline._publish_best_and_last(description="SFT")

      kwargs = mock_api.return_value.upload_folder.call_args.kwargs
      self.assertEqual(kwargs["path_in_repo"], "last")
      self.assertEqual(kwargs["folder_path"], last_dir)

      with open(os.path.join(tmpdir, "checkpoints.json")) as f:
        manifest = json.load(f)
      self.assertEqual(manifest["default"], "best")
      self.assertEqual(manifest["default_step"], 120)

  def test_nothing_is_uploaded_twice_when_the_best_is_the_last(self):
    from src.pipelines import BestCheckpointArchiver

    with tempfile.TemporaryDirectory() as tmpdir:
      last_dir = os.path.join(tmpdir, "checkpoint-200")
      self._make_checkpoint(last_dir)
      archive = os.path.join(tmpdir, BestCheckpointArchiver.ARCHIVE_DIRNAME)
      self._make_checkpoint(archive)

      pipeline = self._pipeline(tmpdir, load_best_model_at_end=True)
      archiver = BestCheckpointArchiver()
      archiver.archive_dir = archive
      archiver.archived_step = 200  # the peak *is* the final step
      pipeline._best_archiver = archiver

      with patch("src.pipelines.HfApi") as mock_api:
        pipeline._publish_best_and_last(description="SFT")

      mock_api.return_value.upload_folder.assert_not_called()
      with open(os.path.join(tmpdir, "checkpoints.json")) as f:
        manifest = json.load(f)
      for name in ("best", "last"):
        self.assertIsNone(manifest["checkpoints"][name]["subfolder"])
        self.assertTrue(manifest["checkpoints"][name]["available"])

  def test_a_failed_companion_upload_does_not_raise(self):
    from src.pipelines import BestCheckpointArchiver

    with tempfile.TemporaryDirectory() as tmpdir:
      self._make_checkpoint(os.path.join(tmpdir, "checkpoint-200"))
      archive = os.path.join(tmpdir, BestCheckpointArchiver.ARCHIVE_DIRNAME)
      self._make_checkpoint(archive)

      pipeline = self._pipeline(tmpdir, load_best_model_at_end=False)
      archiver = BestCheckpointArchiver()
      archiver.archive_dir = archive
      archiver.archived_step = 120
      pipeline._best_archiver = archiver

      with patch("src.pipelines.HfApi") as mock_api:
        mock_api.return_value.upload_folder.side_effect = RuntimeError("500")
        with patch("src.pipelines.time.sleep"):
          # The root checkpoint was already pushed; the run must not fail.
          pipeline._publish_best_and_last(description="PERL")

      with open(os.path.join(tmpdir, "checkpoints.json")) as f:
        manifest = json.load(f)
      # The manifest must not advertise a subfolder that does not exist.
      self.assertIsNone(manifest["checkpoints"]["best"]["subfolder"])
      self.assertFalse(manifest["checkpoints"]["best"]["available"])

  def test_nothing_is_published_when_push_to_hub_is_off(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      self._make_checkpoint(os.path.join(tmpdir, "checkpoint-200"))
      pipeline = self._pipeline(tmpdir, load_best_model_at_end=False)
      pipeline.training_args.push_to_hub = False

      with patch("src.pipelines.HfApi") as mock_api:
        pipeline._publish_best_and_last(description="PERL")

      mock_api.return_value.upload_folder.assert_not_called()
      mock_api.return_value.upload_file.assert_not_called()

  def test_only_the_main_process_publishes(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      self._make_checkpoint(os.path.join(tmpdir, "checkpoint-200"))
      pipeline = self._pipeline(tmpdir, load_best_model_at_end=False)
      pipeline.trainer.state.is_world_process_zero = False

      with patch("src.pipelines.HfApi") as mock_api:
        pipeline._publish_best_and_last(description="PERL")

      mock_api.return_value.upload_folder.assert_not_called()


if __name__ == "__main__":
  unittest.main()

