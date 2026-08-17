"""Unit tests for checkpoint and WandB run resumption."""

import os
import shutil
import sys
import tempfile
from typing import Any
import unittest
from unittest.mock import MagicMock, patch

import contextlib
import types

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
    torch.stack = lambda *x, **kw: MagicMock(tolist=lambda: [[10, 11], [12, 13]])
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
    PERLPipeline,
    Pipeline,
    SFTPipeline,
    WandbResumptionCallback,
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
    repo_id, subfolder, rev = parse_hf_repo_reference("leobianco/npov_PERL/checkpoint-50")
    self.assertEqual(repo_id, "leobianco/npov_PERL")
    self.assertEqual(subfolder, "checkpoint-50")
    self.assertIsNone(rev)

  def test_repo_with_colon_subfolder(self):
    repo_id, subfolder, rev = parse_hf_repo_reference("leobianco/npov_PERL:checkpoint-50")
    self.assertEqual(repo_id, "leobianco/npov_PERL")
    self.assertEqual(subfolder, "checkpoint-50")
    self.assertIsNone(rev)

  def test_repo_with_revision(self):
    repo_id, subfolder, rev = parse_hf_repo_reference("leobianco/npov_PERL@v1.0")
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
    repo_id, subfolder, rev = parse_hf_repo_reference("hf://leobianco/npov_PERL")
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
    with patch("src.pipelines.wandb", mock_wandb), patch("src.pipelines.HfApi", return_value=mock_hf_api):
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
    with patch("src.pipelines.wandb", mock_wandb), patch("src.pipelines.HfApi", return_value=mock_hf_api):
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

    with patch.object(self.pipeline, "_download_hf_checkpoint", return_value=downloaded_hf_dir) as mock_download:
      resolved = self.pipeline._resolve_resume_checkpoint()
      mock_download.assert_called_once_with("leobianco/npov_PERL_run", None, None)
      self.assertEqual(resolved, downloaded_hf_dir)

  def test_resolve_resume_checkpoint_from_hf_hub_subfolder(self):
    """Verify that a Hugging Face Hub subfolder path is downloaded and resolved."""
    downloaded_hf_dir = os.path.join(self.temp_dir, "hf_download", "checkpoint-50")
    os.makedirs(downloaded_hf_dir, exist_ok=True)

    self.mock_training_args.resume_from_checkpoint = "leobianco/npov_PERL_run/checkpoint-50"
    self.mock_training_args.output_dir = None

    with patch.object(self.pipeline, "_download_hf_checkpoint", return_value=downloaded_hf_dir) as mock_download:
      resolved = self.pipeline._resolve_resume_checkpoint()
      mock_download.assert_called_once_with("leobianco/npov_PERL_run", "checkpoint-50", None)
      self.assertEqual(resolved, downloaded_hf_dir)

  def test_download_hf_checkpoint_with_nested_checkpoint(self):
    """Verify that _download_hf_checkpoint detects nested checkpoint-XXX directories."""
    downloaded_dir = os.path.join(self.temp_dir, "hf_repo")
    nested_ckpt = os.path.join(downloaded_dir, "checkpoint-150")
    os.makedirs(nested_ckpt, exist_ok=True)

    with patch("src.pipelines.snapshot_download", return_value=downloaded_dir), \
         patch("src.pipelines.get_last_checkpoint", return_value=nested_ckpt):
      resolved = self.pipeline._download_hf_checkpoint("leobianco/npov_PERL_run")
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
      mock_trainer.train.assert_called_once_with(resume_from_checkpoint=ckpt_dir)
      mock_trainer.push_to_hub.assert_called_once()

  def test_sft_pipeline_run_and_save_calls_train_with_checkpoint(self):
    pipeline = SFTPipeline()
    pipeline.training_args = MagicMock()
    pipeline.training_args.output_dir = self.temp_dir
    pipeline.training_args.resume_from_checkpoint = None

    mock_trainer = MagicMock()
    pipeline.trainer = mock_trainer

    pipeline.run_and_save()
    mock_trainer.train.assert_called_once_with(resume_from_checkpoint=None)
    mock_trainer.push_to_hub.assert_called_once()


if __name__ == "__main__":
  unittest.main()
