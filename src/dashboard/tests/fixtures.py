"""Shared fixtures: synthetic campaign trees that mirror real state files.

The shapes here are copied from a real ``npov`` campaign
(``checkpoints/npov/npov_campaign_2609211206_state.json``) so that the golden
link tests assert against ids the pipeline genuinely produced.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

SFT_REPO = "leobianco/npov_SFT_gemma-4-E4B-it_S130104_epo1_lr2_5e-03_r8_2609211206"
RM_REPO = (
    "leobianco/npov_RM_organic_gemma-4-E4B-it_S130104_epo1_lr2_5e-03_r8_2609211206"
)
PERL_REPO = (
    "leobianco/npov_PERL_organic_gemma-4-E4B-it_S130104_epo1_lr2_5e-03"
    "_beta0_05_r8_2609211206"
)


def campaign_config(name: str = "npov_campaign_2609211206") -> Dict[str, Any]:
  """Returns a config_dict shaped like the orchestrator's serialization.

  Args:
    name: Campaign name, which also names its report file.

  Returns:
    A configuration mapping.
  """
  return {
      "name": name,
      "task_name": "npov",
      "seed": 130104,
      "user": "leobianco",
      "project": "new_perl",
      "wandb_entity": None,
      "base_model": "google/gemma-4-E4B-it",
      "reward_base_model": None,
      "stages": ["autorater", "sft", "rm", "perl", "eval"],
      "rm_dataset_flavors": ["organic"],
      "sft": {"metric": "eval/loss", "goal": "minimize", "max_runs": 30},
      "rm": {"metric": "eval/roc_auc", "goal": "maximize", "max_runs": 30},
      "perl": {
          "metric": "train/rewards/reward_fn/mean",
          "goal": "maximize",
          "max_runs": 10,
      },
      "eval": {
          "evaluator_model": "gemini-2.5-flash",
          "seed": 12345,
          "max_eval_samples": 20,
          "writer_num_fewshot": 0,
          "max_tokens": 250,
          "temperature": 0.0,
          "wandb_project": "new_perl_eval",
          "threshold": 0.1025,
      },
      "reporting": {"generate_markdown": True, "reports_dir": "reports"},
      "state_file": f"./checkpoints/npov/{name}_state.json",
  }


def campaign_state(
    name: str = "npov_campaign_2609211206",
    status: str = "COMPLETED",
    updated_at: str = "2026-09-21T12:06:08.879030",
) -> Dict[str, Any]:
  """Returns a full campaign state dict with all five stages.

  Args:
    name: Campaign id.
    status: Campaign status.
    updated_at: Last-update timestamp, used for index ordering.

  Returns:
    A state mapping ready to be JSON-dumped.
  """
  return {
      "campaign_id": name,
      "task_name": "npov",
      "status": status,
      "current_stage": None,
      "stages_order": ["autorater", "sft", "rm", "perl", "eval"],
      "created_at": "2026-09-21T12:06:06.000000",
      "updated_at": updated_at,
      "config_dict": campaign_config(name),
      "stages": {
          "autorater": {
              "status": "COMPLETED",
              "best_metric_val": 0.913,
              "metrics": {
                  "autorater/roc_auc": 0.913,
                  "autorater/best_threshold": 0.1025,
              },
              "start_time": "2026-09-21T12:06:06.213035",
              "end_time": "2026-09-21T12:06:06.624270",
              "warnings": [],
          },
          "sft": {
              "status": "COMPLETED",
              "sweep_id": "leobianco/new_perl/abc123",
              "sweep_name": "NPOV gemma-4-E4B-it SFT Sweep 35f436",
              "best_run_id": "run_1789992366",
              "best_metric_val": 0.285,
              "final_metric_val": 0.3,
              "model_repo_id": SFT_REPO,
              "metrics": {"eval_loss": 0.285},
              "trials_done": 1,
              "trials_total": 1,
              "selection_strategy": "best",
              "selection_step": 120,
              "sweep_outcome": "complete",
              "start_time": "2026-09-21T12:06:06.628815",
              "end_time": "2026-09-21T12:06:06.970602",
              "warnings": [],
          },
          "rm": {
              "status": "COMPLETED",
              "sweep_id": "def456",
              "best_run_id": "run_1789992367",
              "best_metric_val": 0.965,
              "model_repo_id": RM_REPO,
              "metrics": {"eval_roc_auc": 0.965},
              "trials_done": 1,
              "trials_total": 1,
              "start_time": "2026-09-21T12:06:06.975082",
              "end_time": "2026-09-21T12:06:07.311475",
              "warnings": ["Sweep finished 1/30 trials"],
          },
          "perl": {
              "status": "COMPLETED",
              "sweep_id": "leobianco/new_perl/ghi789",
              "best_run_id": "run_1789992368",
              "best_metric_val": 0.9575,
              "model_repo_id": PERL_REPO,
              "metrics": {"rewards/reward_fn/mean": 0.9575},
              "trials_done": 1,
              "trials_total": 1,
              "selection_strategy": "final_window",
              "selection_window": 10,
              "start_time": "2026-09-21T12:06:07.315574",
              "end_time": "2026-09-21T12:06:07.651819",
              "warnings": [],
          },
          "eval": {
              "status": "COMPLETED",
              "model_repo_id": PERL_REPO,
              "metrics": {
                  "sft/hallucination_rate": 0.091,
                  "sft/bertscore_f1": 0.874,
                  "sft/num_samples": 20,
                  "sft/decoding_temperature": 0.0,
                  "sft@t0.7/hallucination_rate": 0.105,
                  "sft@t0.7/bertscore_f1": 0.874,
                  "sft@t0.7/decoding_temperature": 0.7,
                  "perl/hallucination_rate": 0.062,
                  "perl/bertscore_f1": 0.892,
                  "perl/decoding_temperature": 0.7,
                  "delta/hallucination_rate": 0.043,
                  "delta/bertscore_f1": 0.018,
              },
              "start_time": "2026-09-21T12:06:07.655493",
              "end_time": "2026-09-21T12:06:08.879030",
              "warnings": [],
          },
      },
  }


def write_campaign(
    root: str,
    state: Optional[Dict[str, Any]] = None,
    task: str = "npov",
    archived: bool = False,
    with_report: bool = True,
    report_body: str = "# Campaign report\n\nAll good.\n",
) -> str:
  """Writes a campaign state file (and optionally its report) under ``root``.

  Args:
    root: Repository root to populate.
    state: State mapping; defaults to :func:`campaign_state`.
    task: Task directory to write into.
    archived: Whether to place the file under ``archive/``.
    with_report: Whether to also write the markdown report.
    report_body: Contents of that report.

  Returns:
    The absolute path of the written state file.
  """
  state = state or campaign_state()
  name = state["campaign_id"]
  directory = os.path.join(root, "checkpoints", task)
  if archived:
    directory = os.path.join(directory, "archive")
  os.makedirs(directory, exist_ok=True)
  suffix = "_state_20260921120000.json" if archived else "_state.json"
  path = os.path.join(directory, f"{name}{suffix}")
  with open(path, "w", encoding="utf-8") as handle:
    json.dump(state, handle)

  if with_report:
    reports = os.path.join(root, "reports")
    os.makedirs(reports, exist_ok=True)
    with open(
        os.path.join(reports, f"{name}_summary.md"), "w", encoding="utf-8"
    ) as handle:
      handle.write(report_body)
  return path
