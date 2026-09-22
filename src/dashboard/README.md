# Auto-PERL Dashboard

A **static web dashboard** for Auto-PERL campaigns, published to a private
Hugging Face Space so you can read it from anywhere — including when the
research VM is powered off.

```bash
# Preview locally
python3 scripts/dashboard.py serve

# Publish a snapshot now
python3 scripts/dashboard.py publish --repo-id leobianco/auto-perl

# Keep publishing as campaigns progress (the normal mode)
python3 scripts/dashboard.py watch --repo-id leobianco/auto-perl
```

Live at **https://huggingface.co/spaces/leobianco/auto-perl** (private: visible
only when logged in as `leobianco`).

---

## What it shows

**Campaign list** — every campaign found under `checkpoints/`, newest first:
campaign id, task, base model, status and RM dataset flavor. Filter by task,
status, base model or RM dataset; search across campaign ids, sweep names and
model repo ids; toggle *Completed only*, *Hide dry runs* and *Show archived*.
Filtering happens in the browser and your selection is remembered between
visits. The list deliberately carries no metrics: it is for *finding* a
campaign, and the numbers only mean something next to the run that produced
them.

**Campaign page** — title, status and dry-run chips, a one-line identity strip,
the headline cards, the run metadata, a per-stage table (status, trials, best
metric with its selection provenance, published model), the evaluation table,
the links panel, the rendered markdown report, and the campaign configuration
as launched.

**Headline cards** — one pair per trained policy, so a campaign that fanned out
over two reward-model dataset flavors gets a pair for each rather than one
number that hides the comparison the fan-out exists to make. The first card is
the policy's hallucination rate; the second is how that rate *changed* against
the SFT baseline sampled at the same temperature, as a percentage. Fewer
hallucinations reads as a negative number and is green.

> The eval stage stores deltas signed so that positive always means
> improvement, whichever direction the underlying metric runs in. The card
> negates that, because "the hallucination rate went up by 2%" is how the
> number is spoken, and a green `+` is a trap.

**Evaluation table** — one column per target (`sft`, `sft@t0.7`, `perl`,
`delta`) and one row per *curated* metric: decoding temperature, hallucination
and faithfulness rates, the reward-hacking rubric, repetition rate, ROUGE-1/2/L
F1, BERTScore mean and std, perplexity. The eval stage records upwards of fifty
keys per target — every generation statistic in mean/std/median form, every
ROUGE variant in F1/precision/recall form — and tabulating all of them produced
a table too wide to read. The rest sit behind an *N more metrics* disclosure
directly below, and the full report further down has everything.

Every `delta` cell carries a second, smaller number: the same improvement as a
**share of the baseline it was measured against**, so a hallucination rate of
0.20 dropping to 0.10 reads `+0.1` with `+50.0%` beneath it rather than only
`+0.1`. The sign convention is the delta's, not the metric's — positive is
always better.

> The baseline is not stored anywhere: each policy is compared against the SFT
> row sampled at *its own* decoding temperature, which differs per branch. So
> the share is computed by inverting the delta against the policy column beside
> it (`baseline = policy - sign * delta`), which inherits whatever pairing the
> eval stage chose instead of re-deriving it. The sign comes from
> `eval_metrics.delta_sign`, the same function that produced the delta, so the
> two cannot drift apart. A baseline that is zero or negative gets no share at
> all rather than an invented one.

`provenance_*` keys never reach either table: their *values* are adapter repo
ids and checkpoint directories, which is what made the table wide in the first
place. The dashboard imports `src.orchestrator.eval_metrics` to decide what
counts as an audit key, so its definition cannot drift from the report's.

**Dry runs are labelled.** A rehearsal (`dry_run: true`, or — for states
written before that flag was persisted — sweep ids of the form `mock_sweep_*`)
gets a chip in the list and a banner on its page. Its W&B links are built from
ids the orchestrator invented locally and will not resolve; its model and
dataset links follow the real naming convention but nothing was ever pushed
under those names. Everything a *real* campaign records is real.

**Links panel** — the point of the whole thing:

| Group | Contents |
| :--- | :--- |
| Weights & Biases | One sweep per stage, the winning run of each, the project and the evaluation project |
| Hugging Face models | Every published adapter, plus its file tree (companion checkpoint + `checkpoints.json`) |
| Hugging Face generations | The completions dataset of each evaluation target, labelled with its decoding temperature |
| Hugging Face datasets | The autorater calibration set, the SFT/PE-RL sets, and one RM set per dataset flavor |

The dashboard makes **no API calls**. Every one of those is a plain hyperlink
that your browser follows with credentials this code never sees.

---

## Architecture

```
state files  ->  index  ->  build (static HTML)  ->  publish (HF Space)
```

| Module | Responsibility |
| :--- | :--- |
| `index.py` | Glob `checkpoints/**/*_state.json`, parse into view models, redact secrets |
| `links.py` | The only place a W&B or Hub URL is constructed |
| `render.py` | Markdown → HTML (tables, GitHub alerts, heading anchors) |
| `build.py` | Emit `site/`: list page, campaign pages, `data/index.json` |
| `publish.py` | Upload to a static Space, or copy to a directory |
| `watcher.py` | Poll, debounce, publish on change |

It is **read-only**: it never writes into `checkpoints/` or `reports/`, never
imports the campaign engine, and never talks to a running campaign. Deleting
`src/dashboard/` and `scripts/dashboard.py` restores the repository exactly.

The CSS and JS live in `static/` in the source tree but are **inlined into
every generated page**, so `site/` has no `static/` directory. On a private
Space the document is fetched with credentials the browser already holds,
while a subresource on the same host is not guaranteed to inherit them — and a
stylesheet that 401s gives you unstyled HTML with no visible error. Inlining
costs ~15 KB a page and removes the failure mode entirely.

### Zero new dependencies

`markdown-it-py`, `huggingface-hub` and `pyyaml` are already pinned in
`requirements.txt`. `markdown-it-py` is imported optionally: without it the
reports render verbatim in a `<pre>` block instead of failing the build.

---

## When it publishes

| Trigger | Behaviour |
| :--- | :--- |
| A campaign reaches `COMPLETED` / `FAILED` / `STOPPED` / `ABORTED` | **Immediately**, bypassing the debounce |
| Trial counters, best metric, a new report | After `--debounce` seconds (default 60) |
| Nothing changed | Never — the content fingerprint is compared first |

> [!IMPORTANT]
> The terminal-transition rule is what makes `shutdown_when_done` safe. The VM
> powers off after `shutdown_grace_seconds` (default 60), and the campaign's
> *result* is the single most valuable snapshot of its life. On campaigns armed
> with `--shutdown`, consider `shutdown_grace_seconds: 120` to leave the upload
> more room.

The snapshot timestamp in the page header is **when the content last changed**,
not when the watcher last looked. An idle repository produces no commits, so a
long-unchanged timestamp on an idle VM is expected; an in-progress campaign
whose timestamp has stopped moving is flagged as *stale* instead, with a banner
saying the VM is most likely off.

---

## Running it permanently

In a tmux window next to your campaigns:

```bash
tmux new-window -t auto-perl -n dashboard
python3 scripts/dashboard.py watch --repo-id leobianco/auto-perl
```

Or as a user service that survives reboots:

```ini
# ~/.config/systemd/user/auto-perl-dashboard.service
[Unit]
Description=Auto-PERL dashboard publisher
After=network-online.target

[Service]
WorkingDirectory=%h/new_perl
ExecStart=/usr/bin/python3 scripts/dashboard.py watch --repo-id leobianco/auto-perl
Restart=always
RestartSec=30
Environment=HF_HOME=%h/.cache/huggingface

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now auto-perl-dashboard
journalctl --user -u auto-perl-dashboard -f
```

`AUTO_PERL_SPACE=leobianco/auto-perl` can replace `--repo-id` everywhere.

---

## Commands

| Command | What it does |
| :--- | :--- |
| `build` | Emit the static site into `./site` |
| `serve` | Build, then preview at `http://127.0.0.1:8000` (the exact bytes that get published) |
| `publish` | Build, then upload if the content changed (`--force` to override, `--dry-run` to rehearse) |
| `watch` | Build and publish continuously (`--once` for cron-style use) |

Shared flags: `--root`, `--out`, `--no-archived`, `--repo-id`, `--backend
{hf_space,dir}`, `--destination`, `--public`, `--force`, `--dry-run`.

> [!CAUTION]
> `--public` only applies when the Space is *created*. An existing Space never
> has its visibility changed by this tool: silently making a private page
> public is not something a build script should be able to do.

---

## Security and privacy

* The Space is **private**. Only your Hugging Face account can read it.
* `config_dict` is **redacted** before publishing: any string under a key
  matching `key|token|secret|password|credential` is replaced.
* Raw HTML in a report is **not** rendered, and every dynamic string (campaign
  ids, sweep names, repo ids) is HTML-escaped.
* Nothing is uploaded except the built site: no state files, no logs, no
  checkpoints.

---

## Known coupling

The completions dataset ids are **recomputed at build time** with
`src.utils.build_eval_dataset_repo_id`, the same function the eval stage used,
rather than read from the state file. This avoids persisting anything new, at
the cost that changing the naming convention retroactively breaks the
generations links of older campaigns. Those links carry a `derived` badge, and
`tests/test_links.py` pins the current convention with golden values so the
change is at least noisy.

---

## Tests

```bash
python3 -m unittest discover -s src/dashboard/tests -t .
```

182 tests, hermetic, no network. They pass on a bare `/usr/bin/python3` (13
markdown-specific tests skip) and in the project environment (all run).
