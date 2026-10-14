"""Wiring tests for ``evaluator.sh <task> reward_hacking_autorater_eval``.

The statistics are covered by ``src/test_reward_hacking_calibration.py``.
These tests pin what the pipeline feeds them, with a stubbed judge:

1. It reads the two held-out arms and the demonstration split, and nothing
   else: never a final test set, never an autorater set.
2. It grades the *reference*, not whatever ``completion`` column a row
   happens to carry.
3. Its judge is the deployed judge: the preamble and demonstrations of its
   prompts are byte-identical to the scoring pipeline's for the same
   arguments.
4. It refuses to degrade silently (no Gemini, no demonstrations).
5. ``evaluator.py`` and ``evaluator.sh`` route the mode, and the script
   forwards the same rubric settings as ``score``.
"""

import contextlib
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


def _pickle_save(obj, path):
  with open(path, "wb") as handle:
    pickle.dump(obj, handle)


def _pickle_load(path, map_location=None):
  del map_location
  with open(path, "rb") as handle:
    return pickle.load(handle)


# Stubs are only installed for modules that are missing, and they are shaped
# like the ones the other test modules install (a pass-through tqdm, a torch
# that can save and load), so suites sharing a process keep working.
if "torch" not in sys.modules:
  try:
    import torch  # pylint: disable=g-import-not-at-top,unused-import
  except ImportError:
    _torch = types.ModuleType("torch")
    _torch.__path__ = []
    _torch.nn = types.ModuleType("torch.nn")
    _torch.nn.Module = object
    _torch.bfloat16 = "bfloat16"
    _torch.Tensor = object
    _torch.save = _pickle_save
    _torch.load = _pickle_load
    _torch.no_grad = contextlib.nullcontext
    _torch.device = lambda name: name
    _torch.cuda = mock.MagicMock()
    _torch.cuda.is_available = lambda: False
    sys.modules["torch"] = _torch
    sys.modules["torch.nn"] = _torch.nn

if "tqdm" not in sys.modules:
  _tqdm = types.ModuleType("tqdm")
  _tqdm.tqdm = lambda iterable=None, *args, **kwargs: iterable
  sys.modules["tqdm"] = _tqdm

if "transformers" not in sys.modules or not hasattr(
    sys.modules["transformers"], "__path__"
):
  _transformers = types.ModuleType("transformers")
  _transformers.__path__ = []
  _transformers.PreTrainedModel = object
  _transformers.PretrainedConfig = object
  _transformers.TrainerCallback = object
  _transformers.TrainerState = object
  _transformers.TrainerControl = object
  _transformers.TrainingArguments = object
  _transformers.Trainer = object
  _transformers.DataCollatorWithPadding = mock.MagicMock()
  _transformers.HfArgumentParser = mock.MagicMock()
  _transformers.set_seed = mock.MagicMock()
  _transformers.AutoTokenizer = mock.MagicMock()
  _transformers.AutoModelForCausalLM = mock.MagicMock()
  _transformers.AutoModelForSequenceClassification = mock.MagicMock()
  _transformers.LogitsProcessor = object
  _transformers.LogitsProcessorList = list
  sys.modules["transformers"] = _transformers
  sys.modules["transformers.trainer_utils"] = types.ModuleType(
      "transformers.trainer_utils"
  )
  sys.modules["transformers.trainer_utils"].get_last_checkpoint = (
      mock.MagicMock(return_value=None)
  )

for _name in [
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
]:
  if _name not in sys.modules:
    sys.modules[_name] = mock.MagicMock()

from src import evaluator  # pylint: disable=g-import-not-at-top
from src import pipelines  # pylint: disable=g-import-not-at-top
from src import reward_hacking_calibration as rhc  # pylint: disable=g-import-not-at-top
from src import utils  # pylint: disable=g-import-not-at-top

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "evaluator.sh")
_GRADED_MARKER = "--- RESPONSE TO GRADE ---"
_DEMO_MARKER = "--- GRADED EXAMPLES ---"


class _FakeHubDataset:
  """Just enough of ``datasets.Dataset`` for the code under test."""

  def __init__(self, rows):
    self._rows = [dict(row) for row in rows]

  def __len__(self):
    return len(self._rows)

  def __getitem__(self, index):
    return self._rows[index]

  def __iter__(self):
    return iter([dict(row) for row in self._rows])

  @property
  def column_names(self):
    return list(self._rows[0]) if self._rows else []

  def shuffle(self, seed=None):
    del seed  # Deterministic: the demonstrations are the first rows.
    return self

  def map(self, function, fn_kwargs=None):
    return _FakeHubDataset(
        [function(dict(row), **(fn_kwargs or {})) for row in self._rows]
    )


class _FakeDatasetFactory:

  @staticmethod
  def from_dict(columns):
    keys = list(columns)
    size = len(columns[keys[0]]) if keys else 0
    return _FakeHubDataset(
        [{key: columns[key][i] for key in keys} for i in range(size)]
    )


def _npov_row(query, response=None, completion=None, **extra):
  row = {
      "topic": "t",
      "user_query": query,
      "perspective_1": f"pro arguments for {query}",
      "perspective_1_name": "Pro",
      "perspective_2": f"con arguments for {query}",
      "perspective_2_name": "Con",
  }
  if response is not None:
    row["npov_response"] = response
  if completion is not None:
    row["completion"] = completion
  row.update(extra)
  return row


def _hub():
  """NPOV-shaped Hub content, keyed by (repo, split)."""
  demo = _npov_row(
      "demo question",
      completion="Demo gold sentence one. Demo gold sentence two.",
  )
  return {
      ("u/npov_rm_organic", "test"): [
          _npov_row("q-faithful", response="Faithful reference.",
                    completion="DECOY COMPLETION", class_hall="No", label=1),
          _npov_row("q-hallucinated", response="HALLUCINATED reference.",
                    completion="DECOY COMPLETION", class_hall="Yes", label=0),
          # Shares its prompt with the demonstration: must be dropped.
          _npov_row("demo question", response="Other answer to the demo.",
                    class_hall="No", label=1),
          _npov_row("q-empty", response="", class_hall="No", label=1),
      ],
      ("u/npov_sft", "test"): [
          _npov_row("q-sft", completion="SFT reference."),
          # Same text as the faithful reward-model row: a duplicate.
          _npov_row("q-faithful", completion="Faithful reference."),
      ],
      ("u/npov_sft", "train"): [demo],
  }


def _args(**overrides):
  values = dict(
      task_name="npov",
      user="u",
      writer_model_lora="unused",
      mode=rhc.MODE,
      evaluator_model="gemini-2.5-flash",
      use_gemini=True,
      seed=12345,
      max_eval_samples=1000,
      reward_hacking_num_fewshot=1,
      reward_hacking_threshold=0.6,
  )
  values.update(overrides)
  return utils.EvalArguments(**values)


class _Judge:
  """Stands in for ``gemini_score_rubric_dataset``."""

  def __init__(self, pipe):
    self.pipe = pipe
    self.prompts = None
    self.checkpoint_path = None

  def __call__(self, client, dataset, script_args):
    del client
    self.prompts = [
        dataset[i]["reward_hacking_prompt"] for i in range(len(dataset))
    ]
    self.checkpoint_path = script_args.reward_hacking_checkpoint_path
    keys = utils.reward_hacking_dimension_keys()
    verdicts = []
    for prompt in self.prompts:
      graded = prompt.split(_GRADED_MARKER)[-1]
      if "JUDGE FAILS" in graded:
        verdicts.append(None)
      elif "HALLUCINATED" in graded:
        verdicts.append({key: 0.5 for key in keys})
      else:
        verdicts.append({key: 1.0 for key in keys})
    self.pipe.reward_hacking_stats = {
        "reward_hacking_audit_model": script_args.evaluator_model
    }
    return verdicts


class CalibrationPipelineTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.mkdtemp()
    self.cwd = os.getcwd()
    os.chdir(self.tmp)
    self.addCleanup(os.chdir, self.cwd)
    self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
    self.loaded = []
    self.hub = _hub()
    patcher = mock.patch.object(pipelines, "Dataset", _FakeDatasetFactory)
    patcher.start()
    self.addCleanup(patcher.stop)

  def _load(self, repo, split=None):
    self.loaded.append((repo, split))
    return _FakeHubDataset(self.hub[(repo, split)])

  def _pipeline(self, **overrides):
    pipe = pipelines.EvaluationRewardHackingAutoraterPipeline()
    pipe.args = _args(**overrides)
    pipe.safe_load_dataset = self._load
    pipe.create_gemini_client = mock.MagicMock()
    judge = _Judge(pipe)
    pipe.gemini_score_rubric_dataset = judge
    return pipe, judge

  def _run(self, pipe):
    with contextlib.redirect_stdout(io.StringIO()):
      pipe.load_data()
      pipe.process_data()
      pipe.run_and_save()
    run_dir = os.path.join(
        "logs",
        rhc.run_name(
            pipe.args.evaluator_model,
            pipe.args.reward_hacking_num_fewshot,
            1,
            pipe.args.task_name,
            pipe.args.seed,
        ),
    )
    with open(os.path.join(run_dir, rhc.METRICS_JSON), encoding="utf-8") as f:
      return run_dir, json.load(f)

  def test_end_to_end(self):
    pipe, judge = self._pipeline()
    run_dir, metrics = self._run(pipe)

    # 1. Exactly the two held-out arms and the demonstration split.
    self.assertEqual(
        self.loaded,
        [
            ("u/npov_rm_organic", "test"),
            ("u/npov_sft", "test"),
            ("u/npov_sft", "train"),
        ],
    )

    # 2. The guards, per arm.
    rm_counts = metrics["data"][rhc.SOURCE_RM]
    sft_counts = metrics["data"][rhc.SOURCE_SFT]
    self.assertEqual(rm_counts["dropped_empty_response"], 1)
    self.assertEqual(rm_counts["dropped_demo_prompt"], 1)
    self.assertEqual(rm_counts["after_subsample"], 2)
    self.assertEqual(rm_counts["graded"], 2)
    self.assertEqual(sft_counts["dropped_duplicate"], 1)
    self.assertEqual(sft_counts["after_subsample"], 1)
    self.assertEqual(sft_counts["judge_failures"], 0)

    # 3. The reference is graded, with the demonstrations in front of it.
    self.assertEqual(len(judge.prompts), 3)
    graded = [p.split(_GRADED_MARKER)[-1] for p in judge.prompts]
    self.assertTrue(all("DECOY" not in block for block in graded))
    self.assertIn("Faithful reference.", graded[0])
    self.assertIn("SFT reference.", graded[2])
    for prompt in judge.prompts:
      self.assertIn(_DEMO_MARKER, prompt)
      self.assertIn("Demo gold sentence one.", prompt)

    # 4. The report: one false alarm (the hallucinated row) out of three.
    overall = metrics["strata"][0]
    self.assertEqual(overall["name"], "all")
    self.assertEqual(overall["n_scored"], 3)
    self.assertAlmostEqual(overall["fpr_at_threshold"], 1 / 3)
    names = [s["name"] for s in metrics["strata"]]
    self.assertIn("source=rm_organic/label=hallucinated", names)
    self.assertEqual(
        metrics["config"]["fewshot_source"],
        {"dataset": "u/npov_sft", "split": "train"},
    )
    self.assertEqual(metrics["config"]["num_demonstrations"], 2)
    self.assertEqual(
        [s["dataset"] for s in metrics["config"]["sources"]],
        ["u/npov_rm_organic", "u/npov_sft"],
    )
    self.assertEqual(
        metrics["judge_audit"]["reward_hacking_audit_model"],
        "gemini-2.5-flash",
    )
    for name in (rhc.METRICS_TXT, rhc.ROWS_JSONL, rhc.LOWEST_TXT,
                 rhc.PROMPTS_TXT):
      self.assertTrue(os.path.exists(os.path.join(run_dir, name)), name)

    # 5. The resume checkpoint is keyed by the exact prompt list.
    fingerprint = rhc.prompts_fingerprint(judge.prompts)
    self.assertEqual(
        judge.checkpoint_path,
        os.path.join(run_dir, f"rubric_checkpoint_{fingerprint}.json"),
    )

    # 6. The markdown report, next to the orchestrator's reports.
    name = os.path.basename(run_dir)
    report = os.path.join("reports", "reward_hacking_calibration", f"{name}.md")
    with open(report, encoding="utf-8") as handle:
      text = handle.read()
    self.assertIn("False-positive rate at `0.6`: 33.3%", text)
    self.assertIn(f"(../../logs/{name}/{rhc.METRICS_JSON})", text)
    self.assertIn("UTC", metrics["config"]["generated_at"])
    # No figure was plotted (matplotlib is stubbed), so none is embedded.
    self.assertNotIn("![", text)

  def test_the_reports_folder_can_be_moved(self):
    pipe, _ = self._pipeline(reward_hacking_calibration_reports_dir="elsewhere")
    run_dir, _ = self._run(pipe)
    name = os.path.basename(run_dir)
    self.assertTrue(
        os.path.exists(
            os.path.join("elsewhere", rhc.REPORT_SUBDIR, f"{name}.md")
        )
    )
    self.assertFalse(os.path.exists("reports"))

  def test_the_histogram_is_copied_next_to_the_report(self):
    pipe, _ = self._pipeline()

    def fake_plot(path, threshold):
      del threshold
      with open(path, "wb") as handle:
        handle.write(b"PNG")

    pipe._save_calibration_histograms = fake_plot  # pylint: disable=protected-access
    run_dir, _ = self._run(pipe)
    name = os.path.basename(run_dir)
    folder = os.path.join("reports", rhc.REPORT_SUBDIR)
    with open(os.path.join(folder, f"{name}_histograms.png"), "rb") as handle:
      self.assertEqual(handle.read(), b"PNG")
    with open(os.path.join(folder, f"{name}.md"), encoding="utf-8") as handle:
      text = handle.read()
    self.assertIn(f"]({name}_histograms.png)", text)
    self.assertIn(f"(../../logs/{name}/{rhc.HISTOGRAMS_PNG})", text)

  def test_judge_failures_are_counted_not_graded(self):
    self.hub[("u/npov_sft", "test")][0]["completion"] = "JUDGE FAILS here."
    pipe, _ = self._pipeline()
    _, metrics = self._run(pipe)
    self.assertEqual(metrics["data"][rhc.SOURCE_SFT]["judge_failures"], 1)
    self.assertEqual(metrics["data"][rhc.SOURCE_SFT]["graded"], 0)
    self.assertEqual(metrics["strata"][0]["n_unscored"], 1)
    self.assertAlmostEqual(metrics["strata"][0]["fpr_at_threshold"], 0.5)

  def test_zero_shot_skips_the_demonstration_split(self):
    pipe, judge = self._pipeline(reward_hacking_num_fewshot=0)
    _, metrics = self._run(pipe)
    self.assertNotIn(("u/npov_sft", "train"), self.loaded)
    self.assertTrue(all(_DEMO_MARKER not in p for p in judge.prompts))
    self.assertIsNone(metrics["config"]["fewshot_source"])
    # Without demonstrations no prompt is dropped as a demonstration prompt.
    self.assertEqual(metrics["data"][rhc.SOURCE_RM]["dropped_demo_prompt"], 0)

  def test_an_arm_can_be_switched_off(self):
    pipe, _ = self._pipeline(reward_hacking_calibration_sft_dataset="none")
    _, metrics = self._run(pipe)
    self.assertNotIn(("u/npov_sft", "test"), self.loaded)
    self.assertEqual(list(metrics["data"]), [rhc.SOURCE_RM])

  def test_no_demonstrations_is_fatal_not_zero_shot(self):
    self.hub[("u/npov_sft", "train")] = []
    pipe, _ = self._pipeline()
    with contextlib.redirect_stdout(io.StringIO()):
      with self.assertRaisesRegex(ValueError, "zero-shot"):
        pipe.load_data()

  def test_the_local_evaluator_is_refused_before_loading(self):
    pipe, _ = self._pipeline(use_gemini=False)
    with self.assertRaisesRegex(ValueError, "Gemini"):
      pipe.load_data()
    self.assertEqual(self.loaded, [])

  def test_the_final_test_set_is_refused_before_loading(self):
    pipe, _ = self._pipeline(
        reward_hacking_calibration_rm_dataset="u/npov_final_test_set"
    )
    with self.assertRaisesRegex(ValueError, "final"):
      pipe.load_data()
    self.assertEqual(self.loaded, [])

  def test_the_judge_matches_the_scoring_pipeline(self):
    # The calibration is only meaningful for the judge that is deployed:
    # same preamble, same demonstrations, drawn from the same split.
    pipe, judge = self._pipeline()
    self._run(pipe)
    calibration_loads = list(self.loaded)

    self.loaded = []
    scoring = pipelines.EvaluationScoringPipeline()
    scoring.args = _args(mode="score")
    scoring.safe_load_dataset = self._load
    scoring.val_data = _FakeHubDataset(
        [_npov_row("q-faithful", completion="A policy completion.")]
    )
    with contextlib.redirect_stdout(io.StringIO()):
      scoring._build_reward_hacking_prompts(  # pylint: disable=protected-access
          pipelines.get_task_processor("npov")
      )
    self.assertEqual(self.loaded, [("u/npov_sft", "train")])
    self.assertEqual(calibration_loads[-1], ("u/npov_sft", "train"))

    scoring_prompt = scoring.val_data[0]["reward_hacking_prompt"]
    calibration_prompt = judge.prompts[0]
    self.assertEqual(
        scoring_prompt.split(_GRADED_MARKER)[0],
        calibration_prompt.split(_GRADED_MARKER)[0],
    )
    # Same context and request; only the graded text differs.
    self.assertIn("A policy completion.", scoring_prompt)
    self.assertIn("Faithful reference.", calibration_prompt)


class FewshotSourceTest(unittest.TestCase):

  def test_default_and_override(self):
    self.assertEqual(
        utils.reward_hacking_fewshot_source(_args()), ("u/npov_sft", "train")
    )
    self.assertEqual(
        utils.reward_hacking_fewshot_source(
            _args(reward_hacking_fewshot_dataset="x/y",
                  reward_hacking_fewshot_split="dev")
        ),
        ("x/y", "dev"),
    )

  def test_calibration_argument_defaults(self):
    args = _args()
    self.assertIsNone(args.reward_hacking_calibration_rm_dataset)
    self.assertEqual(args.reward_hacking_calibration_rm_split, "test")
    self.assertIsNone(args.reward_hacking_calibration_sft_dataset)
    self.assertEqual(args.reward_hacking_calibration_sft_split, "test")


class EvaluatorDispatchTest(unittest.TestCase):

  def test_the_mode_wins_over_evaluate_evaluator(self):
    parser = mock.MagicMock()
    parser.parse_args_into_dataclasses.return_value = [
        _args(evaluate_evaluator=True)
    ]
    with mock.patch.object(
        evaluator, "HfArgumentParser", return_value=parser
    ), mock.patch.object(evaluator, "set_seed"), mock.patch.object(
        evaluator, "EvaluationRewardHackingAutoraterPipeline"
    ) as calibration, mock.patch.object(
        evaluator, "EvaluationAutoraterPipeline"
    ) as autorater:
      evaluator.main()
    calibration.assert_called_once_with()
    calibration.return_value.run.assert_called_once_with()
    autorater.assert_not_called()


def _branch(script, mode):
  match = re.search(
      r'\[ "\$2" == "' + re.escape(mode) + r'" \]; then\n(.*?)\n(?:elif|else)',
      script,
      re.DOTALL,
  )
  return match.group(1) if match else None


class EvaluatorScriptTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    with open(_SCRIPT, encoding="utf-8") as handle:
      cls.script = handle.read()

  def test_the_mode_has_a_branch(self):
    branch = _branch(self.script, rhc.MODE)
    self.assertIsNotNone(branch)
    self.assertIn(f'--mode "{rhc.MODE}"', branch)
    self.assertIn(rhc.MODE, self.script.split("Invalid mode")[-1])

  def test_credentials_stay_off_argv(self):
    self.assertNotIn("--gemini_api_key", _branch(self.script, rhc.MODE))

  def test_it_forwards_the_rubric_settings_of_score(self):
    calibration = _branch(self.script, rhc.MODE)
    score = _branch(self.script, "score")
    for flag in (
        "--evaluator_model",
        "--use_gemini",
        "--seed",
        "--autorater_num_samples",
        "--max_workers",
        "--reward_hacking_model",
        "--reward_hacking_num_fewshot",
        "--reward_hacking_threshold",
    ):
      with self.subTest(flag=flag):
        self.assertIn(flag, score)
        self.assertIn(flag, calibration)

  @unittest.skipIf(shutil.which("bash") is None, "bash is not installed")
  def test_the_script_parses(self):
    result = subprocess.run(
        ["bash", "-n", _SCRIPT], capture_output=True, text=True, check=False
    )
    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
  unittest.main()
