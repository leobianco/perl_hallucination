"""Tests for the best/last checkpoint publication rules.

These live outside ``src/orchestrator/tests`` on purpose: the module under
test is deliberately free of training dependencies, so this suite runs
anywhere, unlike the pipeline tests in ``src/test_resumption.py``.

Run with::

  python3 -m unittest src.test_checkpoint_publication -v
"""

import unittest

from src import checkpoint_publication as cp


class TestStepFromCheckpointDir(unittest.TestCase):
  """The step has to be recoverable from a directory name."""

  def test_plain_directory_name(self):
    self.assertEqual(cp.step_from_checkpoint_dir("checkpoint-120"), 120)

  def test_full_path_with_trailing_slash(self):
    self.assertEqual(
        cp.step_from_checkpoint_dir("/tmp/run/checkpoint-40/"), 40
    )

  def test_non_checkpoint_directories_have_no_step(self):
    for name in ("", None, "_best_checkpoint", "checkpoint-", "ref", "output"):
      with self.subTest(name=name):
        self.assertIsNone(cp.step_from_checkpoint_dir(name))

  def test_the_archive_directory_is_not_mistaken_for_a_checkpoint(self):
    # _best_checkpoint/ sits next to checkpoint-N/ inside output_dir and must
    # never be picked up as the final-step checkpoint.
    self.assertIsNone(
        cp.step_from_checkpoint_dir("/tmp/run/_best_checkpoint")
    )


class TestCoerceStep(unittest.TestCase):
  """Publication runs after the push and must never crash on odd state."""

  def test_plain_integers_pass_through(self):
    self.assertEqual(cp.coerce_step(200), 200)

  def test_numeric_strings_and_floats_are_accepted(self):
    self.assertEqual(cp.coerce_step("150"), 150)
    self.assertEqual(cp.coerce_step(150.0), 150)

  def test_zero_and_negative_steps_are_rejected(self):
    # global_step == 0 means nothing was trained; there is no checkpoint.
    self.assertIsNone(cp.coerce_step(0))
    self.assertIsNone(cp.coerce_step(-5))

  def test_none_and_booleans_are_rejected(self):
    self.assertIsNone(cp.coerce_step(None))
    self.assertIsNone(cp.coerce_step(True))

  def test_an_uncomparable_object_does_not_raise(self):
    # A MagicMock trainer state used to crash `global_step > 0` with a
    # TypeError, taking down a run whose model had already been pushed.
    from unittest.mock import MagicMock  # pylint: disable=g-import-not-at-top

    self.assertIsNone(cp.coerce_step(MagicMock()))
    self.assertIsNone(cp.coerce_step(object()))


class TestPlanPublication(unittest.TestCase):
  """Which checkpoint is served by default, and which one is a companion."""

  def test_best_at_root_publishes_the_last_as_companion(self):
    plan = cp.plan_publication(
        root_is_best=True,
        best_dir="/o/_best_checkpoint",
        best_step=120,
        last_dir="/o/checkpoint-200",
        last_step=200,
    )
    self.assertEqual(plan.default_name, "best")
    self.assertEqual(plan.default_step, 120)
    self.assertEqual(plan.companion_name, "last")
    self.assertEqual(plan.companion_dir, "/o/checkpoint-200")
    self.assertEqual(plan.companion_step, 200)
    self.assertFalse(plan.duplicate)

  def test_final_at_root_publishes_the_best_as_companion(self):
    # This is the PE-RL case: the stable last policy is the default, the
    # peak-reward adapter stays available for comparison.
    plan = cp.plan_publication(
        root_is_best=False,
        best_dir="/o/_best_checkpoint",
        best_step=50,
        last_dir="/o/checkpoint-200",
        last_step=200,
    )
    self.assertEqual(plan.default_name, "last")
    self.assertEqual(plan.default_step, 200)
    self.assertEqual(plan.companion_name, "best")
    self.assertEqual(plan.companion_dir, "/o/_best_checkpoint")
    self.assertEqual(plan.companion_step, 50)

  def test_the_same_checkpoint_is_not_uploaded_twice(self):
    plan = cp.plan_publication(
        root_is_best=True,
        best_dir="/o/checkpoint-200",
        best_step=200,
        last_dir="/o/checkpoint-200",
        last_step=200,
    )
    self.assertTrue(plan.duplicate)
    self.assertIsNone(plan.companion_name)
    self.assertIsNone(plan.companion_dir)

  def test_a_missing_companion_directory_yields_nothing_to_upload(self):
    plan = cp.plan_publication(
        root_is_best=False,
        best_dir=None,
        best_step=None,
        last_dir="/o/checkpoint-200",
        last_step=200,
    )
    self.assertIsNone(plan.companion_name)
    self.assertEqual(plan.default_name, "last")
    self.assertEqual(plan.default_step, 200)

  def test_unknown_steps_do_not_count_as_duplicates(self):
    # Two checkpoints whose steps could not be determined are not evidence
    # that they are the same checkpoint.
    plan = cp.plan_publication(
        root_is_best=True,
        best_dir="/o/_best_checkpoint",
        best_step=None,
        last_dir="/o/checkpoint-200",
        last_step=None,
    )
    self.assertFalse(plan.duplicate)
    self.assertEqual(plan.companion_name, "last")


class TestBuildManifest(unittest.TestCase):
  """The manifest has to tell the truth about what is actually on the Hub."""

  def _plan(self, root_is_best=True):
    return cp.plan_publication(
        root_is_best=root_is_best,
        best_dir="/o/_best_checkpoint",
        best_step=120,
        last_dir="/o/checkpoint-200",
        last_step=200,
    )

  def test_root_checkpoint_has_no_subfolder(self):
    manifest = cp.build_manifest(
        self._plan(root_is_best=True),
        best_step=120,
        last_step=200,
        companion_published=True,
    )
    self.assertEqual(manifest["default"], "best")
    self.assertIsNone(manifest["checkpoints"]["best"]["subfolder"])
    self.assertEqual(manifest["checkpoints"]["best"]["step"], 120)

  def test_companion_checkpoint_records_its_subfolder(self):
    manifest = cp.build_manifest(
        self._plan(root_is_best=True),
        best_step=120,
        last_step=200,
        companion_published=True,
    )
    self.assertEqual(manifest["checkpoints"]["last"]["subfolder"], "last")
    self.assertEqual(manifest["checkpoints"]["last"]["step"], 200)
    self.assertTrue(manifest["checkpoints"]["last"]["available"])

  def test_a_failed_companion_upload_is_not_advertised(self):
    # Claiming a subfolder that does not exist would send every later load
    # of repo:last into a 404.
    manifest = cp.build_manifest(
        self._plan(root_is_best=True),
        best_step=120,
        last_step=200,
        companion_published=False,
    )
    self.assertIsNone(manifest["checkpoints"]["last"]["subfolder"])
    self.assertFalse(manifest["checkpoints"]["last"]["available"])
    # The root checkpoint is unaffected by the companion failing.
    self.assertTrue(manifest["checkpoints"]["best"]["available"])

  def test_perl_style_manifest_defaults_to_the_last_checkpoint(self):
    manifest = cp.build_manifest(
        self._plan(root_is_best=False),
        best_step=120,
        last_step=200,
        companion_published=True,
        metric_for_best_model="rewards/reward_fn/mean",
        greater_is_better=True,
        best_metric=1.75,
    )
    self.assertEqual(manifest["default"], "last")
    self.assertEqual(manifest["default_step"], 200)
    self.assertIsNone(manifest["checkpoints"]["last"]["subfolder"])
    self.assertEqual(manifest["checkpoints"]["best"]["subfolder"], "best")
    self.assertEqual(
        manifest["metric_for_best_model"], "rewards/reward_fn/mean"
    )
    self.assertTrue(manifest["greater_is_better"])
    self.assertEqual(manifest["best_metric"], 1.75)

  def test_a_duplicate_checkpoint_is_available_under_both_names(self):
    plan = cp.plan_publication(
        root_is_best=True,
        best_dir="/o/checkpoint-200",
        best_step=200,
        last_dir="/o/checkpoint-200",
        last_step=200,
    )
    manifest = cp.build_manifest(
        plan, best_step=200, last_step=200, companion_published=False
    )
    for name in ("best", "last"):
      with self.subTest(name=name):
        self.assertTrue(manifest["checkpoints"][name]["available"])
        self.assertIsNone(manifest["checkpoints"][name]["subfolder"])
        self.assertEqual(manifest["checkpoints"][name]["step"], 200)


class TestUploadIgnorePatterns(unittest.TestCase):
  """Optimizer state must never reach the Hub; weights always must."""

  def test_optimizer_and_rng_state_are_excluded(self):
    for name in ("optimizer.pt", "scheduler.pt", "rng_state*.pth"):
      with self.subTest(name=name):
        self.assertIn(name, cp.CHECKPOINT_UPLOAD_IGNORE_PATTERNS)

  def test_adapter_weights_and_configs_are_not_excluded(self):
    import fnmatch  # pylint: disable=g-import-not-at-top

    keep = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "trainer_state.json",
        "config.json",
    )
    for name in keep:
      with self.subTest(name=name):
        matched = any(
            fnmatch.fnmatch(name, pattern)
            for pattern in cp.CHECKPOINT_UPLOAD_IGNORE_PATTERNS
        )
        self.assertFalse(matched, f"{name} would be excluded from the upload")


if __name__ == "__main__":
  unittest.main()
