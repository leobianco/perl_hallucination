"""Tests for fitting checkpoint repo ids into Hugging Face's length limit.

The property under test is not "the name is <= 96 characters" - the old
tail-truncating code satisfied that too. It is that the name is <= 96
characters *and still identifies what it names*: the timestamp that makes it
unique, the flavor that says which reward model it came from, and enough of
the task and the base model to tell two checkpoints apart by eye.
"""

import unittest

from src.orchestrator import naming
from src.orchestrator.config import VALID_TASKS
from src.orchestrator.model_manager import ModelManager


#: The longest names the repository can currently produce.
_WORST_CASE_TASK = "ragtruth-summarization"
_MODEL = "gemma-4-E4B-it"
_STAMP = "2509191204"
_PERL_DETAILS = "S12345_epo3_lr1.0e-04_beta0.05_r16"
_SWEEP_DETAILS = "S12345_epo3_lr1.0e-04_r16"


class ShrinkTokenTest(unittest.TestCase):
  """The readable-abbreviation primitive."""

  def test_a_token_that_already_fits_is_untouched(self):
    self.assertEqual(naming._shrink_token("ragtruth", 20), "ragtruth")

  def test_trailing_segments_are_spent_before_the_leading_one(self):
    # The leading segment is the family name; losing it first would turn
    # 'ragtruth-qa' into 'ragtrut-qa', which is unreadable *and* no shorter
    # than dropping a character of the variant.
    self.assertEqual(
        naming._shrink_token("ragtruth-summarization", 13), "ragtruth-summ"
    )

  def test_the_leading_segment_is_protected(self):
    # Asking for 8 cannot be honoured without eating 'ragtruth', which is
    # the family name and the thing a reader recognises. The token comes
    # back longer than requested instead; the caller's floor is what keeps
    # that from mattering.
    self.assertEqual(
        naming._shrink_token("ragtruth-summarization", 8), "ragtruth-s"
    )

  def test_the_leading_segment_gives_way_when_explicitly_allowed(self):
    self.assertEqual(
        naming._shrink_token("ragtruth-summarization", 8, keep_head=False),
        "ragtru-s",
    )

  def test_a_single_segment_token_shrinks_from_the_right(self):
    self.assertEqual(naming._shrink_token("summarization", 5), "summa")

  def test_every_known_task_stays_distinct_at_the_floor(self):
    # The whole point of spending trailing segments first. If two tasks
    # collided here, two campaigns would publish to the same repo id.
    floored = [
        naming._shrink_token(task, naming.TASK_MIN_LEN) for task in VALID_TASKS
    ]
    self.assertEqual(len(set(floored)), len(floored), floored)


class FitModelRepoIdTest(unittest.TestCase):
  """Assembly of a bounded, still-informative repo id."""

  def _fit(self, task=_WORST_CASE_TASK, stage="PERL", model=_MODEL,
           details=_PERL_DETAILS, flavor_slug="synstruct"):
    return naming.fit_model_repo_id(
        user="leobianco",
        task=task,
        stage=stage,
        model=model,
        details=details,
        timestamp=_STAMP,
        flavor_slug=flavor_slug,
    )

  def test_a_short_name_is_returned_verbatim_and_unflagged(self):
    repo_id, shortened = self._fit(task="ragtruth")
    self.assertFalse(shortened)
    self.assertEqual(
        repo_id,
        "leobianco/ragtruth_PERL_synstruct_gemma-4-E4B-it_S12345_epo3_"
        "lr1.0e-04_beta0.05_r16_2509191204",
    )

  def test_the_worst_case_name_fits(self):
    repo_id, shortened = self._fit()
    self.assertTrue(shortened)
    self.assertLessEqual(len(repo_id), naming.HF_REPO_ID_MAX)

  def test_the_timestamp_always_survives(self):
    # This is the regression. Tail truncation removed the timestamp, so two
    # campaigns on the same task and hyperparameters collided.
    for task in VALID_TASKS:
      for stage, details in (("RM", _SWEEP_DETAILS), ("PERL", _PERL_DETAILS)):
        for slug in ("organic", "synstruct"):
          repo_id, _ = self._fit(
              task=task, stage=stage, details=details, flavor_slug=slug
          )
          with self.subTest(task=task, stage=stage, flavor=slug):
            self.assertTrue(repo_id.endswith(_STAMP), repo_id)

  def test_the_flavor_tag_always_survives(self):
    repo_id, _ = self._fit()
    self.assertIn("_synstruct_", repo_id)

  def test_the_stage_tag_always_survives(self):
    # `_RL_CHECKPOINT_MARKERS` in src/utils.py recognises an RL adapter by
    # finding 'perl' in its name; losing it would make the evaluator stop
    # stacking the SFT adapter underneath.
    repo_id, _ = self._fit()
    self.assertIn("_PERL_", repo_id)

  def test_the_hyperparameters_always_survive(self):
    repo_id, _ = self._fit()
    self.assertIn(_PERL_DETAILS, repo_id)

  def test_the_model_is_only_shortened_after_the_task_bottoms_out(self):
    # No name this repository currently produces needs the model shortened,
    # so the base model stays fully readable in every one of them.
    repo_id, _ = self._fit()
    self.assertIn(_MODEL, repo_id)

  def test_the_task_stays_recognisable(self):
    repo_id, _ = self._fit()
    self.assertIn("ragtruth-", repo_id)

  def test_the_model_gives_way_once_the_task_has_nothing_left(self):
    repo_id, shortened = self._fit(
        task=_WORST_CASE_TASK,
        model="a-very-long-base-model-name-indeed",
        details=_PERL_DETAILS + "_extra_padding_to_force_the_issue",
    )
    self.assertTrue(shortened)
    self.assertLessEqual(len(repo_id), naming.HF_REPO_ID_MAX)
    self.assertTrue(repo_id.endswith(_STAMP), repo_id)

  def test_distinct_inputs_stay_distinct_after_fitting(self):
    seen = {}
    for task in VALID_TASKS:
      for model in ("gemma-4-E4B-it", "gemma-3-27b-it"):
        for stage, details in (("RM", _SWEEP_DETAILS), ("PERL", _PERL_DETAILS)):
          for slug in ("organic", "synstruct"):
            key = (task, model, stage, slug)
            repo_id, _ = self._fit(
                task=task, stage=stage, model=model,
                details=details, flavor_slug=slug,
            )
            self.assertNotIn(
                repo_id, seen, f"{key} collides with {seen.get(repo_id)}"
            )
            seen[repo_id] = key


class MaterializationRepoIdTest(unittest.TestCase):
  """The repo ids the model manager actually emits."""

  def _repo_id(self, stage_name, task_name, flavor=None):
    manager = ModelManager(dry_run=True)
    manager.user = "leobianco"
    plan = manager.build_materialization_command(
        stage_name=stage_name,
        task_name=task_name,
        base_model=f"google/{_MODEL}",
        best_params={"learning_rate": 1e-4, "num_train_epochs": 3, "lora_r": 16},
        seed=12345,
        sft_model_path="leobianco/sft",
        reward_model_path="leobianco/rm",
        timestamp=_STAMP,
        flavor=flavor,
    )
    return plan.repo_id

  def test_every_stage_and_task_fits_the_hub_limit(self):
    for task in VALID_TASKS:
      for stage in ("sft", "rm", "perl"):
        for flavor in (None, "organic", "synthetic_struct"):
          repo_id = self._repo_id(stage, task, flavor)
          with self.subTest(task=task, stage=stage, flavor=flavor):
            self.assertLessEqual(len(repo_id), naming.HF_REPO_ID_MAX, repo_id)
            self.assertTrue(repo_id.endswith(_STAMP), repo_id)

  def test_sft_carries_no_flavor_tag(self):
    # SFT is shared by every branch, so tagging it would claim a provenance
    # it does not have.
    repo_id = self._repo_id("sft", "ragtruth", flavor="synthetic_struct")
    self.assertNotIn("synstruct", repo_id)

  def test_branches_of_one_campaign_do_not_collide(self):
    organic = self._repo_id("perl", _WORST_CASE_TASK, flavor="organic")
    synthetic = self._repo_id(
        "perl", _WORST_CASE_TASK, flavor="synthetic_struct"
    )
    self.assertNotEqual(organic, synthetic)


if __name__ == "__main__":
  unittest.main()
