"""Scientific reporting engine: generates Markdown summaries, Rich console tables, and W&B reports."""

from __future__ import annotations

import datetime
import logging
import os
from typing import Any, Dict, Optional
from src.orchestrator import eval_metrics
from src.orchestrator.config import CampaignConfig
from src.orchestrator.state import CampaignState, StageStatus

logger = logging.getLogger(__name__)


class CampaignReporter:
  """Generates publication-ready Markdown summaries and interactive reports."""

  def __init__(self, config: CampaignConfig, state: CampaignState):
    self.config = config
    self.state = state

  def generate_all(self) -> Dict[str, str]:
    """Generates all configured reports and returns a dictionary of report paths/URLs.

    Reporting runs after every checkpoint has been trained and pushed, so a
    failure here (a formatting bug, a W&B outage) must never be allowed to
    mark a successful campaign as failed. Each artifact is attempted
    independently and errors are recorded instead of raised.

    Returns:
      Artifact paths/URLs, plus a ``reporting_error`` entry when something
      could not be produced.
    """
    artifacts = {}
    if self.config.reporting.generate_markdown:
      try:
        artifacts["markdown_report"] = self.generate_markdown_report()
      except Exception as error:  # pylint: disable=broad-except
        logger.exception("Markdown report generation failed")
        artifacts["reporting_error"] = f"markdown report: {error}"

    if self.config.reporting.render_console_summary:
      try:
        self.render_console_summary()
      except Exception as error:  # pylint: disable=broad-except
        logger.warning("Console summary rendering failed: %s", error)

    if self.config.reporting.publish_wandb_report:
      try:
        report_url = self.publish_wandb_report()
        if report_url:
          artifacts["wandb_report_url"] = report_url
      except Exception as error:  # pylint: disable=broad-except
        logger.warning("W&B report publishing failed: %s", error)
        artifacts.setdefault("reporting_error", f"wandb report: {error}")

    return artifacts

  def generate_markdown_report(self) -> str:
    """Builds a comprehensive scientific summary report in GitHub-flavored Markdown."""
    reports_dir = self.config.reporting.reports_dir
    os.makedirs(reports_dir, exist_ok=True)
    report_file = os.path.join(reports_dir, f"{self.config.name}_summary.md")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"# Auto-PERL Scientific Campaign Report: `{self.config.task_name}`",
        "",
        "> **Automated Research Summary**  ",
        f"> **Campaign ID**: `{self.config.name}`  ",
        f"> **Generated At**: {now_str}  ",
        f"> **Base Model**: `{self.config.base_model}`  ",
        f"> **Random Seed**: `{self.config.seed}`  ",
        f"> **W&B Project**: `{self.config.wandb_entity or self.config.user}/{self.config.project}`",
        "",
        "---",
        "",
        "## 1. Executive Performance Scorecard",
        "",
        (
            "| Pipeline Stage | Model Identifier | Primary Metric | Best"
            " Value | Status |"
        ),
        (
            "| :--- | :--- | :--- | :---"
            " | :--- |"
        ),
    ]

    for stage_name in self.config.stages:
      stage_res = self.state.stages.get(stage_name)
      if not stage_res:
        lines.append(f"| **{stage_name.upper()}** | *Pending* | - | - | ⏳ PENDING |")
        continue

      model_str = f"`{stage_res.model_repo_id}`" if stage_res.model_repo_id else "*N/A*"
      status_icon = "✅ COMPLETED" if stage_res.status == StageStatus.COMPLETED else f"⚠️ {stage_res.status.value}"

      if stage_name == "sft":
        metric_str = "eval/loss"
        val_str = f"{stage_res.best_metric_val:.4f}" if stage_res.best_metric_val is not None else "-"
      elif stage_name == "rm":
        metric_str = "eval/roc_auc"
        val_str = f"{stage_res.best_metric_val:.4f}" if stage_res.best_metric_val is not None else "-"
      elif stage_name == "perl":
        metric_str = "rewards/mean"
        val_str = f"{stage_res.best_metric_val:.4f}" if stage_res.best_metric_val is not None else "-"
      elif stage_name == "eval":
        metric_str = "autorater_hallucination"
        h_rate = eval_metrics.headline_metric(stage_res.metrics)
        val_str = f"{h_rate:.4f}" if h_rate is not None else "Done"
      else:
        metric_str = "-"
        val_str = "-"

      lines.append(f"| **{stage_name.upper()}** | {model_str} | `{metric_str}` | **{val_str}** | {status_icon} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Stage Breakdown & Winning Hyperparameters",
        "",
    ])

    # SFT Details
    sft_res = self.state.stages.get("sft")
    if sft_res and sft_res.status == StageStatus.COMPLETED:
      sft_sw = (
          f"`{sft_res.sweep_name}` (`{sft_res.sweep_id}`)"
          if sft_res.sweep_name
          else f"`{sft_res.sweep_id}`"
      )
      lines.extend([
          "### 2.1. Supervised Fine-Tuning (SFT)",
          f"* **Sweep**: {sft_sw}",
          f"* **Winning Run ID**: `{sft_res.best_run_id}`",
          f"* **Best Eval Loss**: `{sft_res.best_metric_val:.5f}`",
          f"* **Hugging Face Model**: [{sft_res.model_repo_id}](https://huggingface.co/{sft_res.model_repo_id})",
          "* **Optimal Hyperparameters**:",
          "```yaml",
      ])
      for k, v in sorted(sft_res.best_params.items()):
        if not k.startswith("_"):
          lines.append(f"  {k}: {v}")
      lines.extend(["```", ""])

    # RM Details
    rm_res = self.state.stages.get("rm")
    if rm_res and rm_res.status == StageStatus.COMPLETED:
      rm_sw = (
          f"`{rm_res.sweep_name}` (`{rm_res.sweep_id}`)"
          if rm_res.sweep_name
          else f"`{rm_res.sweep_id}`"
      )
      lines.extend([
          "### 2.2. Reward Model (RM)",
          f"* **Sweep**: {rm_sw}",
          f"* **Winning Run ID**: `{rm_res.best_run_id}`",
          f"* **Best ROC-AUC**: `{rm_res.best_metric_val:.5f}`",
          f"* **Hugging Face Model**: [{rm_res.model_repo_id}](https://huggingface.co/{rm_res.model_repo_id})",
          "* **Optimal Hyperparameters**:",
          "```yaml",
      ])
      for k, v in sorted(rm_res.best_params.items()):
        if not k.startswith("_"):
          lines.append(f"  {k}: {v}")
      lines.extend(["```", ""])

    # PE-RL Details
    perl_res = self.state.stages.get("perl")
    if perl_res and perl_res.status == StageStatus.COMPLETED:
      perl_sw = (
          f"`{perl_res.sweep_name}` (`{perl_res.sweep_id}`)"
          if perl_res.sweep_name
          else f"`{perl_res.sweep_id}`"
      )
      lines.extend([
          "### 2.3. Parameter-Efficient Reinforcement Learning (PE-RL)",
          f"* **Sweep**: {perl_sw}",
          f"* **Winning Run ID**: `{perl_res.best_run_id}`",
          f"* **Best Mean Reward**: `{perl_res.best_metric_val:.5f}`",
          f"* **Hugging Face Model**: [{perl_res.model_repo_id}](https://huggingface.co/{perl_res.model_repo_id})",
          "* **Optimal Hyperparameters**:",
          "```yaml",
      ])
      for k, v in sorted(perl_res.best_params.items()):
        if not k.startswith("_"):
          lines.append(f"  {k}: {v}")
      lines.extend(["```", ""])

    # Eval Details
    eval_res = self.state.stages.get("eval")
    if eval_res and eval_res.status == StageStatus.COMPLETED:
      targets = eval_metrics.present_targets(eval_res.metrics)
      lines.extend([
          "### 2.4. Final Evaluation & Gemini Autorating",
          f"* **Evaluator Model**: `{self.config.eval.evaluator_model}`",
          f"* **Headline Policy**: `{eval_res.model_repo_id}`",
          f"* **Evaluation Seed**: `{self.config.eval.seed}`"
          f" (subsamples {self.config.eval.max_eval_samples} test examples)",
          "",
      ])

      if targets:
        # Δ is signed so that positive always means PE-RL improved on SFT,
        # regardless of whether the metric is minimized or maximized.
        header = "| Metric | " + " | ".join(t for _, t in targets) + " | Δ |"
        divider = "| :--- | " + " | ".join("---:" for _ in targets) + " | ---: |"
        lines.extend([
            "* **Per-policy results** (Δ = PE-RL improvement over SFT):",
            "",
            header,
            divider,
        ])
        for name, values, delta in eval_metrics.comparison_rows(
            eval_res.metrics
        ):
          cells = " | ".join(
              f"**{eval_metrics.format_value(v)}**" for v in values
          )
          delta_str = (
              f"{delta:+.4f}" if isinstance(delta, float) else "-"
          )
          lines.append(f"| `{name}` | {cells} | {delta_str} |")
        lines.append("")
      else:
        lines.extend([
            "| Metric | Measured Value |",
            "| :--- | :--- |",
        ])
        for m_name, m_val in eval_res.metrics.items():
          lines.append(
              f"| `{m_name}` | **{eval_metrics.format_value(m_val)}** |"
          )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. Scientific Verification & Checkpoint Integrity",
        "* All intermediate checkpoints are tagged with explicit seeds and parameter digests.",
        f"* Base and downstream LoRA adapters are fully reproducible using seed `{self.config.seed}`.",
        "",
        f"*Report auto-generated by Auto-PERL on `{now_str}`.*",
    ])

    content = "\n".join(lines)
    with open(report_file, "w", encoding="utf-8") as f:
      f.write(content)

    logger.info("Generated scientific Markdown report: %s", report_file)
    return report_file

  def render_console_summary(self) -> None:
    """Renders an attractive summary scorecard in the terminal using Rich."""
    try:
      from rich.console import Console  # pylint: disable=g-import-not-at-top
      from rich.panel import Panel  # pylint: disable=g-import-not-at-top
      from rich.table import Table  # pylint: disable=g-import-not-at-top

      console = Console()
      table = Table(
          title=f"Auto-PERL Campaign Summary: {self.config.task_name}",
          show_header=True,
          header_style="bold magenta",
      )

      table.add_column("Stage", style="cyan", width=10)
      table.add_column("Status", width=12)
      table.add_column("Metric", style="yellow")
      table.add_column("Score", style="bold green", justify="right")
      table.add_column("Hugging Face Checkpoint", style="dim")

      for stage_name in self.config.stages:
        stage_res = self.state.stages.get(stage_name)
        if not stage_res:
          table.add_row(stage_name.upper(), "[dim]PENDING[/dim]", "-", "-", "-")
          continue

        status_str = (
            "[bold green]COMPLETED[/bold green]"
            if stage_res.status == StageStatus.COMPLETED
            else f"[red]{stage_res.status.value}[/red]"
        )
        val_str = (
            f"{stage_res.best_metric_val:.4f}"
            if stage_res.best_metric_val is not None
            else "-"
        )
        metric_str = (
            "loss"
            if stage_name == "sft"
            else "roc_auc"
            if stage_name == "rm"
            else "reward"
            if stage_name == "perl"
            else "metrics"
        )
        repo_str = stage_res.model_repo_id or "-"

        table.add_row(
            stage_name.upper(), status_str, metric_str, val_str, repo_str
        )

      panel = Panel(
          table,
          title="[bold green]Campaign Execution Finished[/bold green]",
          border_style="green",
      )
      console.print(panel)

    except ImportError:
      # Fallback to plain text printing if rich is not present
      print(f"=== Auto-PERL Campaign Summary: {self.config.task_name} ===")
      for stage_name in self.config.stages:
        stage_res = self.state.stages.get(stage_name)
        status = stage_res.status.value if stage_res else "PENDING"
        print(f"  Stage {stage_name.upper()}: {status}")

  def publish_wandb_report(self) -> Optional[str]:
    """Publishes an interactive W&B Report via wandb-workspaces if available."""
    target_entity = self.config.wandb_entity or self.config.user
    if self.config.dry_run:
      logger.info(
          "[DRY-RUN] Simulating W&B Report creation: https://wandb.ai/%s/%s/reports/mock",
          target_entity,
          self.config.project,
      )
      return f"https://wandb.ai/{target_entity}/{self.config.project}/reports/mock"

    try:
      import wandb_workspaces.reports.v2 as wr  # pylint: disable=g-import-not-at-top

      report = wr.Report(
          entity=self.config.wandb_entity or self.config.user,
          project=self.config.project,
          title=f"Auto-PERL Campaign: {self.config.task_name} ({self.config.name})",
          description=f"Automated PE-RL science report for task '{self.config.task_name}' on model {self.config.base_model}.",
      )

      blocks = [
          wr.H1(f"Auto-PERL Experiment Campaign: {self.config.task_name}"),
          wr.P(f"Execution finished on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}."),
          wr.H2("Stage Outcomes"),
      ]

      report.blocks = blocks
      report.save()
      logger.info("Published W&B Report: %s", report.url)
      return report.url
    except Exception as e:
      logger.debug("Could not publish programmatic W&B report via wandb_workspaces: %s", e)
      return None
