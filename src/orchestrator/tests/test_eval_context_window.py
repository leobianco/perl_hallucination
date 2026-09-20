"""Tests for capping the vLLM context window during evaluation.

vLLM sizes its KV cache from whatever context the checkpoint *declares*, and a
declaration is not a promise that the cache fits: ``Qwen3-4B-Instruct-2507``
advertises 262144 tokens, which is roughly 38 GB of KV cache for a single
sequence. The engine then either refuses to start or serves one request at a
time, and a campaign that trained fine dies in its last stage.

Three properties matter here.

1. A campaign that sets ``eval.max_model_len`` passes it to generation.
2. A campaign that does not set it produces the command it always produced -
   the flag is absent entirely, not present with an empty or ``None`` value,
   which vLLM would reject.
3. Only generation carries it. Scoring runs the autorater and the fluency
   model through plain Transformers and never starts an engine, so the flag
   there would be noise that outlives the reason for it.
"""

import unittest

from src.orchestrator.config import CampaignConfig
from src.orchestrator.model_manager import ModelManager
from src.orchestrator.stages.base import CampaignContext
from src.orchestrator.stages.eval_stage import EvalStage
from src.orchestrator.state import CampaignState
from src.orchestrator.state import StageResult
from src.orchestrator.state import StageStatus
from src.orchestrator.sweep_controller import SweepController


def _stage(max_model_len=None):
  """Builds an EvalStage over a campaign with one finished SFT run.

  Args:
    max_model_len: Value for ``eval.max_model_len``. None leaves the default.

  Returns:
    The stage, ready to build commands.
  """
  config = CampaignConfig.create_default(task_name="npov", dry_run=True)
  config.user = "leobianco"
  if max_model_len is not None:
    config.eval.max_model_len = max_model_len
  state = CampaignState(campaign_id="c", task_name="npov")
  state.stages["sft"] = StageResult(
      status=StageStatus.COMPLETED, model_repo_id="leobianco/sft"
  )
  context = CampaignContext(
      config=config,
      state=state,
      sweep_controller=SweepController(dry_run=True),
      model_manager=ModelManager(dry_run=True),
  )
  return EvalStage(context)


class EvalContextWindowTest(unittest.TestCase):
  """``eval.max_model_len`` on its way to ``src.evaluator``."""

  def test_default_is_unset(self):
    """Campaigns that never needed a cap must keep vLLM's own default."""
    self.assertIsNone(CampaignConfig.create_default(
        task_name="npov", dry_run=True
    ).eval.max_model_len)

  def test_generation_omits_the_flag_when_unset(self):
    cmd = _stage()._generation_command("leobianco/policy")
    self.assertNotIn("--max_model_len", cmd)

  def test_generation_forwards_the_cap(self):
    cmd = _stage(max_model_len=8192)._generation_command("leobianco/policy")
    self.assertIn("--max_model_len", cmd)
    # Adjacency, not mere presence: argparse reads the value positionally.
    self.assertEqual(cmd[cmd.index("--max_model_len") + 1], "8192")

  def test_scoring_never_carries_the_cap(self):
    """Scoring starts no engine, so the cap has nothing to act on there."""
    cmd = _stage(max_model_len=8192)._scoring_command("leobianco/policy")
    self.assertNotIn("--max_model_len", cmd)

  def test_cap_survives_a_yaml_round_trip(self):
    """The cap has to come back on `resume`, which rebuilds from the dict."""
    config = CampaignConfig.create_default(task_name="npov", dry_run=True)
    config.eval.max_model_len = 4096
    restored = CampaignConfig.from_dict(config.to_dict())
    self.assertEqual(restored.eval.max_model_len, 4096)


if __name__ == "__main__":
  unittest.main()
