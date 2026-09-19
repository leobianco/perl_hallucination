# Auto-PERL: Automated Scientific Campaign Orchestrator

A modular, zero-lockin research orchestrator for Parameter-Efficient Reinforcement Learning (PE-RL) experiments on Gemma and other language models.

---

## 🚀 Key Features

* **End-to-End Automation**: Automatically chains **SFT Sweep** $\to$ **Reward Model Sweep** $\to$ **PE-RL (RLOO) Sweep** $\to$ **Final Evaluation** with Gemini 2.5 Flash autorater.
* **Automatic Sweep Bounds (`max_runs`)**: Enforces strict run budgets per stage (defaults: SFT=30, RM=30, PE-RL=10) and automatically seals sweeps on W&B (`STOPPED`), preventing runaway cloud jobs.
* **Dynamic Checkpoint Wiring**: Captures winning model checkpoints from SFT and RM sweeps and dynamically injects them into downstream PE-RL sweeps without manual copy-pasting.
* **Zero-Lockin & Non-Invasive**: Resides entirely in `src/orchestrator/`. Does not modify a single line of your core training scripts (`src/writer_sft.py`, `src/reward_model.py`, `src/perl.py`, `src/evaluator.py`). If removed, your existing codebase remains 100% intact.
* **Crash-Resilient & Resumable**: Atomic state persistence (`campaign_state.json`). If interrupted or preempted on a cloud VM, resume seamlessly with `resume`. The full configuration is persisted alongside the state, so a resumed campaign reproduces the original budgets and stage selection.
* **Pinned Mission Control dashboard**: A `rich` live block pinned to the bottom of the pane (header, DAG, live sweep leaderboard, hotkeys) with logs streaming *above* it. It deliberately does **not** use a full-screen alternate-buffer TUI, so `tmux` scrollback, copy-mode and `Ctrl-b [` keep working, re-attaching or resizing simply re-flows the layout, and piping to a file produces clean plain text.
* **Degrades gracefully**: Works with no color, ASCII-only glyphs, narrow panes, no TTY, and even before `pip install -r requirements.txt` has finished (`rich`/`questionary` are optional at import time).
* **Pre-flight `doctor`**: Checks packages, W&B/HF credentials, sweep YAMLs, writable dirs, disk space, GPU and tmux *before* you commit to a multi-hour run.
* **Automated Scientific Reports**: Generates publication-ready Markdown summaries with performance tables, metric deltas, and interactive W&B reports.

---

## 📦 Architecture

```
src/orchestrator/
├── __init__.py               # Package public API
├── config.py                 # Dataclasses & YAML serialization (CampaignConfig, StageConfig)
├── state.py                  # Atomic state persistence & crash resumption (CampaignState)
├── sweep_controller.py       # W&B Sweep registration, agent controller, and best-run query
├── model_manager.py          # Model materialization, local retention, and HF Hub publishing
├── reporter.py               # Markdown report generator & Rich console scorecards
├── engine.py                 # Linear DAG orchestrator (CampaignEngine), event/control aware
├── cli/
│   ├── __init__.py
│   ├── app.py                # argparse CLI: run / wizard / status / resume / report / list / doctor
│   ├── theme.py              # Terminal capability detection, glyphs, colors, formatting
│   ├── events.py             # Typed event bus, control signals, sweep log parser
│   ├── renderables.py        # DAG rows, leaderboard, status table, banners (rich + plain)
│   ├── console.py            # UiConsole: one markup dialect, rich or plain
│   ├── dashboard.py          # Pinned live dashboard, hotkeys, final scorecard
│   ├── wizard.py             # Interactive setup wizard (questionary, with a plain fallback)
│   └── tui.py                # Backwards-compatible shim over the dashboard
└── tests/
    ├── test_orchestrator.py  # Config/state/engine unit & dry-run integration tests
    ├── test_cli_ui.py        # Theme, events, renderables, console, dashboard
    ├── test_cli_wizard.py    # Wizard flows, validators, prompters
    └── test_cli_app.py       # Argument parsing, commands, diagnostics, exit codes
```

---

## ⚡ Quickstart

### 0. Pre-flight check (do this first)
```bash
python3 scripts/run_campaign.py doctor --task npov
```

### 1. Zero-Boilerplate Campaign (Runs full pipeline with defaults)
```bash
python3 scripts/run_campaign.py run --task npov
```
Runs SFT (30 trials) $\to$ RM (30 trials) $\to$ PE-RL (10 trials) $\to$ Evaluation $\to$ Report.

### 2. Guided Interactive Setup Wizard
```bash
python3 scripts/run_campaign.py wizard
```
Prompts for task, stages, budget preset and upstream checkpoints, then shows a
review panel with a runtime estimate and the equivalent non-interactive command.

### 3. Budget presets (or explicit overrides)
```bash
python3 scripts/run_campaign.py run --task npov --preset quick
python3 scripts/run_campaign.py run --task npov --sft-runs 10 --rm-runs 10 --perl-runs 5
```

| Preset | Trials (SFT/RM/PE-RL) | Eval samples | Intent |
| :--- | :--- | :--- | :--- |
| `smoke` | 1 / 1 / 1 | 20 | Wiring check (minutes) |
| `quick` | 5 / 5 / 3 | 200 | Fast signal (a few hours) |
| `standard` | 30 / 30 / 10 | 1000 | Project default |
| `thorough` | 60 / 60 / 20 | 2000 | Publication run |

### 4. Running Sub-Pipelines (Plug in Existing Checkpoints)
If SFT and RM were already trained, run only PE-RL and Evaluation by passing the checkpoints directly:
```bash
python3 scripts/run_campaign.py run --task npov \
  --stages perl,eval \
  --sft-model "leobianco/npov_SFT_gemma-4-E4B-it_..." \
  --reward-model "leobianco/npov_RM_gemma-4-E4B-it_..."
```
Omitting a required checkpoint is caught before launch, not three hours in.

### 4b. Calibrating the autorater (and what happens if it is weak)

Every hallucination number this orchestrator reports is a count of verdicts
from a Gemini judge against a decision threshold. The `autorater` stage runs
**first**, before any GPU time is spent, and fits that threshold instead of
inheriting a constant:

```bash
python3 scripts/run_campaign.py run --task ragtruth --stages autorater
```

It scores `{user}/{task}_autorater` (test split) with the same judge, the
same few-shot count and the same seed the final evaluation will use, then
reports ROC-AUC, the fitted threshold, TPR/FPR, accuracy, precision and how
saturated the judge's scores are. Section 2.0 of the report shows all of it.

Two consequences worth knowing:

* **The final evaluation scores at the fitted threshold**, not at
  `eval_stage.threshold`. Campaigns calibrated at different thresholds are
  not directly comparable with each other, nor with the numbers in
  `BASELINES_*.md`, which were all scored at the old constant `0.1025`. Drop
  `autorater` from `stages` to keep that constant.
* **A weak judge warns, it does not stop.** Below `min_autorater_auc`
  (default `0.85`) the stage raises a warning that appears in the dashboard
  and as a `[!CAUTION]` block at the top of the report. The campaign
  continues: the rates are still computable, they simply carry a wider error
  bar than the report's decimals suggest.

The only knob worth touching is `evaluator_num_fewshot`, and 2 is the value
this repository has settled on. Note that calibration deliberately has no
configuration of its own - it reads `eval_stage`, because a threshold fitted
against a 4-shot judge does not apply to a 2-shot one.

### 5. Choosing the reward model's training dataset

The reward model can be trained on human-labelled hallucinations
(`organic`) or on hallucinations injected by structured perturbation
(`synthetic_struct`). The orchestrator picks the dataset itself - editing
`--dataset_repo_id` in `scripts/sweep_rm.yaml` has no effect, because the RM
stage rewrites that flag on the way past. Choose it here instead:

```bash
python3 scripts/run_campaign.py run --task ragtruth --rm-datasets synthetic_struct
```

The wizard asks the same question whenever the `rm` stage is selected.

**Naming two datasets fans the campaign out.** The reward model *is* the
experiment, so two datasets are two independent branches, not one sweep with
two options:

```bash
python3 scripts/run_campaign.py run --task ragtruth --rm-datasets organic,synthetic_struct
```

| | Single dataset | Two datasets |
| :--- | :--- | :--- |
| Stage ids | `sft`, `rm`, `perl`, `eval` | `sft`, `rm:organic`, `rm:synthetic_struct`, `perl:organic`, `perl:synthetic_struct`, `eval` |
| SFT | once | once (shared by both branches) |
| Evaluation | once | once, scoring **both** policies against the shared SFT baseline |
| Report | `2.2.`, `2.3.` | `2.2.1/2.2.2`, `2.3.1/2.3.2`, one Δ column per branch |

Both reward models are trained before either policy: an RM sweep is far
cheaper than a PE-RL sweep, so a failure in the cheap half surfaces before
the expensive half has begun, and both ROC-AUCs are visible before any
policy training starts.

Every published checkpoint carries its dataset in the repo id
(`..._RM_synstruct_...`, `..._PERL_synstruct_...`), so a reward model and
the policy it shaped stay identifiable a month later.

> [!NOTE]
> `synthetic_llm` is deliberately not selectable. Its training split is
> assembled at load time from two other datasets via
> `--num_organic_hallus_to_keep` / `--num_struct_hallus_to_keep`; exposing it
> as a plain flavor would hide those knobs and silently train on a mixture
> nobody asked for. Use `scripts/reward_model.sh` directly for that case.

Two datasets require the `rm` stage: without it both branches would fall
back to the single `--reward-model` override and produce two identical
policies reported as a comparison. That is rejected before launch.

### 6. Pre-flight Dry-Run (Verify wiring without GPU allocation)
```bash
python3 scripts/run_campaign.py run --task npov --preset smoke --dry-run
```

### 7. Resuming vs. Restarting a Campaign

A campaign is identified by its **name**, and its name determines its **state
file**:

```
./checkpoints/<task>/<name>_state.json
```

If you do not set `name:` in your YAML, one is generated from the clock
(`bosch_campaign_2609151127`), so every launch is a brand-new campaign and
this section never concerns you. **If you *do* pin `name:` in your config,
every `run` points at the same state file** — and the orchestrator then has to
know whether you meant "continue that campaign" or "start it over".

It will not guess. The two answers are expensive in opposite directions:

| You meant | If the orchestrator guesses wrong |
|---|---|
| **Continue** | It re-runs finished stages and re-burns hours of GPU time. |
| **Start over** | It attaches to stale sweep ids and silently discards a campaign's results. |

So `run` behaves as follows:

| Situation | Behaviour |
|---|---|
| No state file exists | Starts a new campaign. Normal case, nothing to decide. |
| State file exists, `--resume` given | Continues it. Prints what it adopted. |
| State file exists, `--fresh` given | Archives the old state, starts over. |
| State file exists, no flag, **interactive TTY** | Prompts `[r]esume / [f]resh / [a]bort`. |
| State file exists, no flag, **no TTY** (piped, `nohup`, CI) | **Refuses**, exit code `2`, and tells you the three ways out. |
| No state file, `--resume` given | **Refuses**, exit code `3` (`Nothing to resume`). |

```bash
# Continue where you left off.
python3 scripts/run_campaign.py run --task bosch --config configs/campaign_bosch.yaml --resume

# Throw the old attempt away and start clean.
python3 scripts/run_campaign.py run --task bosch --config configs/campaign_bosch.yaml --fresh

# Equivalent to --resume, and the preferred spelling when you are only resuming.
python3 scripts/run_campaign.py resume --task bosch
```

`--fresh` and `--resume` are mutually exclusive; argparse rejects passing both.

> **`--yes` does not answer this question.** `--yes` only skips the *launch*
> confirmation. With an existing state file and no `--resume`/`--fresh`, the
> run is still refused — `--yes` must never be able to destroy a campaign.

#### `--fresh` archives, it never deletes

The old state file is **moved**, not removed:

```
./checkpoints/<task>/archive/<name>_state_<YYYYmmddHHMMSS>.json
```

It is the only record of which sweeps a campaign registered and which
checkpoints it pushed, so it is worth keeping. `status`, `list` and `resume`
glob one directory level deep, so archived campaigns disappear from those
listings but remain on disk for you to inspect or restore by moving the file
back.

#### Recipe: I deleted the W&B sweep from the web UI

You do not need `--fresh` for this. On resume, the orchestrator probes each
recorded sweep id before reusing it:

* **Sweep is gone (404)** → the stale pointer is cleared and a new sweep is
  registered automatically. Completed stages are still skipped.
* **W&B is unreachable** (503, network drop) → the sweep id is *kept*, because
  abandoning a live sweep over a transient error would throw away paid-for
  trials.

Use `--fresh` when you want to discard the campaign's *results*, not merely
its sweeps.

#### A resumed sweep only runs the trials it still owes

`max_runs` is a budget for the **whole stage**, not per attempt. Before
launching an agent, the stage asks W&B how many trials the sweep has already
finished and requests only the difference:

```
Sweep already has 6/10 finished trials; running the remaining 4.
```

If the budget is already spent the agent is skipped entirely and the stage
goes straight to picking its winner. If W&B cannot be reached the full budget
is used unchanged - overshooting costs GPU hours, but undershooting would
silently give you a smaller search than you asked for.

**Only trials in W&B state `finished` count against the budget.** A
`crashed`, `failed` or `killed` trial burned GPU time but produced no
candidate model, so it is retried. `10 trials` means "10 models to choose the
best from", not "10 attempts". This matters in practice: aborting with `[x]`
kills the trial in flight, so the opposite rule would make every interruption
quietly cost you one point of your hyperparameter search.

#### What the dashboard shows right after a resume

| Column | Behaviour |
| --- | --- |
| `Trials` | Starts from the trials the sweep **already** has, not from `00`. The new agent counts from zero internally; the displayed number is offset by the larger of the state file's counter and W&B's `finished` count, so it can never appear to go backwards. |
| `Artifact` | The previous attempt's failure (e.g. a red *"Materialization ... was interrupted"*) is cleared the moment the stage re-enters `RUNNING`. It described a run that is over. Seeing it persist on a healthy resumed stage is a bug, not a live error. |
| `Best` | Carried over from the previous attempt until the new agent reports something better. |

Progress, sweep id and best-so-far survive a resume; only the *verdict* of the
failed attempt is discarded.

Both panes derive the counter from the same baseline, which the engine
establishes once per stage attempt and ships with the `STAGE_STARTED` event.
If the launching pane and `status --watch` ever disagree about the trial
count, that is a bug worth reporting - they are no longer allowed to.


### 8. Monitoring from a second tmux pane
```bash
python3 scripts/run_campaign.py status --task npov --watch
python3 scripts/run_campaign.py status --task npov --json | jq .stages_completed
python3 scripts/run_campaign.py list
```

The watcher is a **separate process**. It cannot see the launching pane's
in-memory log parser, so everything it shows is read from the state file,
which the running campaign mirrors its progress into:

| Shown in the watch pane | Refreshed |
|---|---|
| Stage statuses, elapsed times, model repo ids | On every stage transition |
| `NN/total` trial counter | The moment a trial finishes |
| Best metric so far | At most every 5s while a trial runs |
| Final eval metrics | When the eval stage completes |

The per-trial leaderboard is **not** mirrored - it stays exclusive to the
launching pane's dashboard, because writing every trial's parameters to disk
on every refresh is not worth the I/O.

Three behaviours worth knowing:

* **You can start the watcher first.** With `--watch` it waits for a campaign
  to appear rather than exiting, so the natural `split-window` → `watch` →
  `run` order works.
* **It follows the newest campaign** for the task, re-resolved on every
  refresh. Pass `--state-file` to pin it to one campaign instead.
* **It never dies on a transient read.** A campaign archived by
  `run --fresh`, or a half-written file, produces a message and a retry on
  the next refresh - not a traceback in a pane you stopped looking at.

> [!NOTE]
> On resume the counter continues from where the previous attempt stopped
> rather than restarting at zero, because a new `wandb agent` counts only its
> own trials while the state file tracks the whole stage.

#### Finding the winner in the W&B web UI

When a stage picks its winner, that run is tagged in W&B:

| Tag | Meaning |
|---|---|
| `best` | This run won its stage. Filter on it to see every winner in the project. |
| `best-sft` / `best-rm` / `best-perl` | Which stage it won, since one project holds the sweeps of every stage and task. |

The run's **notes** field is also set to the metric that won it, e.g.
*"Selected by Auto-PERL as the best SFT trial (eval/loss=0.31042)."*

In the W&B runs table, filter with `tags contains best-sft` to jump straight
to it. W&B has no literal "pin", so tags are the closest equivalent - and
unlike renaming the run, they are reversible and lose no information.

If a stage is re-run and a *different* trial wins, the tags are stripped from
the previous winner first, so exactly one run per sweep is ever tagged. Any
other tags you added yourself are left untouched.

> [!NOTE]
> Tagging is cosmetic and deliberately best-effort. It happens between the
> sweep and the retraining of the winner, so a W&B hiccup there prints a
> warning and carries on rather than discarding hours of GPU time. Model
> selection never depends on the tag.

#### Where the "Best" column comes from

The `Best` column - and the metric column of the live leaderboard - is filled
from **two different sources at two different times**:

| When | Source | Accuracy |
|---|---|---|
| While the sweep runs | The metric key scraped out of the agent's stdout | Best effort |
| Once the stage ends | The W&B API (`fetch_best_run`) | Authoritative |

The live number therefore depends on the stage's `metric:` **literally
appearing in the output**. The scraper accepts the spellings a training
script actually uses - `eval/loss`, `eval_loss`, and for a nested key like
`train/rewards/reward_fn/mean` also the suffix `rewards/reward_fn/mean`. It
deliberately does **not** fall back to the last segment alone: matching a
bare `loss` would pick up the HF `Trainer`'s *training* loss and rank the
whole leaderboard on the wrong quantity.

`-` in that column means "no value seen yet", which is legitimate early in a
stage. If trials are finishing and it is still empty, the campaign says so
once:

```
NOTICE  SFT: 2 trials have finished but no 'eval/loss' value appeared in the
        agent output, so the leaderboard and the Best column stay empty. The
        final ranking still uses the W&B API; only the live view is affected.
        Check that the metric name matches what the training script logs.
```

The usual cause is a mismatch between `metric.name` in `scripts/sweep_*.yaml`
and what the training script logs. **The sweep itself is unaffected** - W&B
optimises on its own copy of the metric, and the stage's final best run is
queried from the API - so this is a display problem, not a lost experiment.

#### How a trial is scored: `selection_strategy`

A sweep trial produces a *curve*, not a number, so "the best trial" needs a
rule. Each stage picks one:

| Stage | `selection_strategy` | Scored on |
|---|---|---|
| SFT | `best` | Lowest `eval/loss` at **any** eval step |
| RM | `best` | Highest `eval/roc_auc` at **any** eval step |
| PE-RL | `final` | `train/rewards/reward_fn/mean` at the **last** step |

**Why `best` for SFT and RM.** The materialization publishes the best-scoring
checkpoint (`--load_best_model_at_end`), so a trial that bottoms out at step 60
and then overfits is shipped as its step-60 checkpoint regardless. Ranking such
a trial on its *last* eval would judge it by the tail we throw away. Before this
was fixed, selection and deployment used two different criteria.

**Why `final` for PE-RL.** `rewards/reward_fn/mean` is a *training* signal
logged every optimizer step over 8 sampled generations. Its maximum reliably
lands early, before the policy stabilises, so ranking on it would select the
luckiest batch of completions rather than the best configuration. The converged
level is the honest signal.

**The caveat.** Maximising over (trials x eval steps) is a best-of-N over noisy
estimates, so the reported number is optimistically biased - the model is still
at least as good as a final-step pick in expectation, but the *score* is not an
unbiased estimate of its quality. This matters most for the RM, which becomes
PE-RL's reward: a reward model cherry-picked at a lucky ROC-AUC spike is a
noisier signal for the policy to exploit. Two things keep this visible:

* The campaign log prints how much early stopping recovered, e.g.
  *"Early stopping recovered 0.04120 of eval/roc_auc versus the end of the run;
  the published checkpoint is the step-150 one."* A large gap on a noisy metric
  is a warning sign, not a result.
* The markdown report records the selection step and the final-step value next
  to the headline number.

The trustworthy number remains the **eval stage's autorater score**, which is
computed on a held-out test set that took no part in any of this selection.

#### What ends up on the Hub: `checkpoint_policy`

Nothing is discarded. Every materialization run publishes **both** the best and
the final checkpoint; `checkpoint_policy` only decides which of the two is the
repository's *default*:

| Stage | Repo root (`from_pretrained(repo_id)`) | Subfolder |
|---|---|---|
| SFT | best `eval/loss` checkpoint | `last/` |
| RM | best `eval/roc_auc` checkpoint | `last/` |
| PE-RL | **final-step** policy | `best/` |

PE-RL defaults to the final checkpoint on purpose: a peak-reward adapter is one
lucky batch away from being a collapsed policy, so the stable model is what you
get unless you ask for the other one.

```python
# The default: the checkpoint the campaign chose for this stage.
model = AutoPeftModelForCausalLM.from_pretrained("leobianco/npov_PERL")

# The companion, for comparison.
peak = AutoPeftModelForCausalLM.from_pretrained(
    "leobianco/npov_PERL", subfolder="best"
)
```

Anywhere this codebase accepts a model reference you can also write
`leobianco/npov_PERL:best` - `parse_hf_repo_reference` understands the
`repo:subfolder` and `repo/tree/<rev>/<subfolder>` forms.

A `checkpoints.json` manifest at the repository root records which checkpoint
is which, at what step, and under which metric:

```json
{
  "default": "last",
  "default_step": 200,
  "metric_for_best_model": "rewards/reward_fn/mean",
  "greater_is_better": true,
  "best_metric": 1.75,
  "checkpoints": {
    "best": {"step": 120, "subfolder": "best", "available": true},
    "last": {"step": 200, "subfolder": null,   "available": true}
  }
}
```

`subfolder: null` means "at the root". When the best checkpoint *is* the final
one, both entries point at the root and nothing is uploaded twice. If a
companion upload fails the manifest says `available: false` rather than
advertising a subfolder that 404s - the root checkpoint is pushed first and is
never put at risk by the companion.

Uploads exclude `optimizer.pt`, `scheduler.pt` and the RNG dumps: they are
useless for evaluation and would dwarf the adapter itself.

**Overriding it.** Per stage, in the campaign YAML:

```yaml
rm_stage:
  selection_strategy: final   # rank trials on their last eval instead
  checkpoint_policy: final    # ... and make the last checkpoint the default
```

`materialization_eval_steps` controls how finely the winner's retraining can
locate that best checkpoint; it defaults to each sweep YAML's `--eval_steps`
(SFT 10, RM 50, PE-RL 50). Setting it coarser than the sweep means the peak
the trial was ranked on may not be reachable in the run that produces the
artifact.




### 9. Headless Mode (for `nohup` or logging to file)
```bash
python3 scripts/run_campaign.py run --task npov --no-tui > campaign.log 2>&1 &
```

### Appearance flags
`--no-color` (also honors `NO_COLOR`), `--ascii` (also `PERL_ASCII=1`), `--plain`
(no live block at all), `--debug` (full tracebacks). They work on either side of
the subcommand.

---

## 🎛️ Interactive Hotkey Controls (Inside `tmux`)

While the live dashboard is attached, single keystrokes (no Enter needed) control the run.
Hotkeys are automatically disabled when stdin is not a TTY, and the terminal is
always restored on exit.

| Hotkey | Action | Description |
| :---: | :--- | :--- |
| **`[a]`** | **Advance with Best** | Seals the active sweep, selects the current top run, and promotes it to the next stage. |
| **`[p]`** | **Pause / Resume** | Parks the campaign before the next stage. The campaign stays alive — it does **not** stop. |
| **`[s]`** | **Stop** | Finishes the current stage, then ends the campaign gracefully and writes the report. |
| **`[x]`** | **Abort** | Same as stop, but skips report generation. |
| **`[l]`** | **Log verbosity** | Cycles all → milestones → off. |
| **`[+]` / `[-]`** | **Resize** | Grows/shrinks the live leaderboard. |
| **`[?]`** | **Help** | Toggles the in-place hotkey cheat-sheet. |
| **`[q]`** | **Detach** | Detaches the dashboard; the campaign keeps running headless. |

`Ctrl-C` once requests a graceful stop; twice aborts.

---

## 🖥️ Running in `tmux` on Cloud / GCP VMs

When executing long, multi-stage campaigns on a remote Google Cloud Compute Engine VM:

### 1. Launch Inside `tmux` with UTF-8 Support
To ensure proper rendering of box-drawing glyphs and borders:
```bash
tmux -u new -s auto-perl
```
Ensure your shell locale supports UTF-8:
```bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

### 2. SIGHUP Immunity & Disconnect Protection
The orchestrator explicitly ignores `SIGHUP` (`_survive_terminal_hangup`), so your campaign will continue running even if an SSH connection drops. You can safely detach anytime (`Ctrl-B, d`) and reattach (`tmux attach -t auto-perl`).

### 3. Recommended 2-Pane Workflow
Split your tmux window (`Ctrl-B, %`):
* **Pane 1 (Runner)**:
  ```bash
  python3 scripts/run_campaign.py run --task npov --preset standard
  ```
* **Pane 2 (Monitor)**:
  ```bash
  python3 scripts/run_campaign.py status --task npov --watch
  ```
  `--watch` polls `state.json` every 5 seconds without touching GPU resources or interfering with the runner.

### 4. Low-Bandwidth & Terminal Fallbacks
* If your terminal font garbles box glyphs, force ASCII rendering:
  ```bash
  python3 scripts/run_campaign.py run --task npov --preset standard --ascii
  ```
* For slow SSH connections or plain streaming logs without full-screen redraws:
  ```bash
  python3 scripts/run_campaign.py run --task npov --preset standard --plain
  # or --no-tui
  ```

---

## 📝 Declarative Configuration Example (`configs/campaign.yaml`)

```yaml
campaign:
  # Pinning a name pins the state file to
  # ./checkpoints/npov/npov_production_v1_state.json, so re-running this
  # config requires --resume or --fresh. See "Resuming vs. Restarting"
  # above. Omit `name` to get a timestamped, always-new campaign.
  name: "npov_production_v1"
  task: "npov"
  seed: 130104
  user: "leobianco"
  project: "new_perl"
  base_model: "google/gemma-4-E4B-it"

stages: [autorater, sft, rm, perl, eval]

# Which dataset(s) the reward model is trained on. Naming two of them runs
# an RM *and* a PE-RL sweep for each, then scores both policies in one
# evaluation. See "Choosing the reward model's training dataset" above.
rm_dataset_flavors: [organic]

sft_stage:
  enabled: true
  sweep_config: "scripts/sweep_sft.yaml"
  max_runs: 30
  metric: "eval/loss"
  goal: "minimize"
  # Rank trials on the best eval step, not the last one; publish that same
  # checkpoint. See "How a trial is scored" above.
  selection_strategy: "best"
  checkpoint_policy: "best"
  materialization_eval_steps: 10

rm_stage:
  enabled: true
  sweep_config: "scripts/sweep_rm.yaml"
  max_runs: 30
  metric: "eval/roc_auc"
  goal: "maximize"
  selection_strategy: "best"
  checkpoint_policy: "best"
  materialization_eval_steps: 50

perl_stage:
  enabled: true
  sweep_config: "scripts/sweep_perl.yaml"
  max_runs: 10
  metric: "rewards/reward_fn/mean"
  goal: "maximize"
  # Never "best": the training reward is far too noisy per step for its peak
  # to mean anything. "final_window" averages the end of training, which is
  # the level your eye reads off the W&B curve.
  selection_strategy: "final_window"
  selection_window: 10
  checkpoint_policy: "final"   # repo root = last policy; best/ holds the peak
  materialization_eval_steps: 50
  sft_model_path: "auto"
  reward_model_path: "auto"

eval_stage:
  enabled: true
  evaluator_model: "gemini-2.5-flash"
  max_eval_samples: 1000
  # Few-shot examples given to the judge. The `autorater` stage is calibrated
  # with this same value, so the threshold it fits applies to the scoring.
  evaluator_num_fewshot: 2
  # Fallback decision threshold, used only when `autorater` is not in stages.
  # When it is, the campaign scores at the threshold that stage fitted.
  threshold: 0.1025
  # ROC-AUC below which the judge is flagged as too weak to trust. Advisory:
  # it warns in the dashboard and the report, it does not stop the campaign.
  min_autorater_auc: 0.85

reporting:
  generate_markdown: true
  publish_wandb_report: true

# Power the VM off when there is nothing left to do - the campaign-level
# equivalent of SHUTDOWN=true in scripts/perl.sh. Off by default.
#
# Fires on COMPLETED and on FAILED (a campaign that dies at hour two is the
# most expensive one to leave running). Never fires on a stop you asked for
# with the hotkey or Ctrl-C, and never in a --dry-run.
shutdown_when_done: false
# Cancellable countdown before the machine goes down; Ctrl-C during it
# aborts the shutdown. 0 powers off immediately.
shutdown_grace_seconds: 60
```

Arm it for a single overnight launch without touching the YAML:

```bash
python3 scripts/run_campaign.py run --task ragtruth --shutdown
python3 scripts/run_campaign.py resume --task ragtruth --shutdown

# ...or disarm a config that has it on:
python3 scripts/run_campaign.py run --task ragtruth --config my.yaml --no-shutdown
```

An armed shutdown is echoed in the pre-launch review block as
`On finish  POWER OFF the VM after a 60s cancellable countdown`, so it is
never a surprise after the fact.


---

## 🗑️ Safe Removal (Zero-Lockin Guarantee)

If you ever decide to remove Auto-PERL and return to running manual bash scripts, simply delete the orchestrator files:
```bash
rm -rf src/orchestrator scripts/run_campaign.py
```
Your repository will be in its exact prior state, with all original models, pipelines, and scripts functioning without issue.
