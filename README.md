# The Language of Thought Is a Design Choice — Unifying Demonstration

Code for the empirical demonstration in **"Position: The Language of Thought Is a
Design Choice, Not a Compliance Problem"** (LP4FM @ NeurIPS 2026, position track).

The experiment is **inference-only**. It reproduces the field's scattered findings
about thinking-language policies inside a *single controlled setup*, so the joint
**accuracy × faithfulness × cost** landscape can be drawn without cross-paper
protocol confounds and so the operating point implied by language-consistency
training can be located on it.

---

## Contents

1. [Quick start](#1-quick-start)
2. [What gets run](#2-what-gets-run)
3. [Repo-local by construction](#3-repo-local-by-construction)
4. [Command reference](#4-command-reference)
5. [Experiment recipes](#5-experiment-recipes)
6. [Resuming and splitting runs](#6-resuming-and-splitting-runs)
7. [Predictions are kept, losslessly](#7-predictions-are-kept-losslessly)
8. [Outputs](#8-outputs)
9. [Hypotheses → where each is computed](#9-hypotheses-where-each-is-computed)
10. [Data and models: what is actually downloaded](#10-data-and-models-what-is-actually-downloaded)
11. [Two deviations from the spec](#11-two-deviations-from-the-spec)
12. [Logging](#12-logging)
13. [Only one model is ever in GPU memory](#13-only-one-model-is-ever-in-gpu-memory)
14. [Tuning for your GPU](#14-tuning-for-your-gpu)
15. [Troubleshooting](#15-troubleshooting)
16. [Repository layout](#16-repository-layout)

---

## 1. Quick start

On the 48 GB GPU box:

```bash
git clone <this repo> && cd Position-The-Language-of-Thought-Is-a-Design-Choice

./run.sh setup          # uv venv + deps, all inside the repo (~10 min, vLLM on Linux+NVIDIA)
./run.sh doctor         # environment / GPU / data / model report — read this before running
./run.sh data           # download + build MGSM and Global-MMLU item sets (~2 min)
./run.sh models         # download the two 8B models into assets/models (~32 GB)
./run.sh generate       # the experiment (resumable; ~8-14 h)
./run.sh report         # score + fertility + analysis + figures + policy cards (~10 min)
```

Or in one shot (each stage in order, same as above):

```bash
./run.sh all 2>&1 | tee logs/all.log
```

Before committing a long run, prove the plumbing works end to end on a 0.6 B model
(a few minutes, CPU is fine):

```bash
./run.sh smoke
```

`smoke` numbers are **not results** — it exists only to verify the pipeline.

### Local (macOS / no GPU)

`./run.sh setup` installs the `[dev]` extra and the code falls back to the
`transformers` backend automatically. Everything except full-scale generation
works. Use `--limit` to keep it small:

```bash
./run.sh generate --models smoke-qwen-0_6b --limit 2 --max-new-tokens 160
```

---

## 2. What gets run

**Conditions** — the question is always in the target language and the
answer-format instruction never changes; only the thinking-language instruction
varies (`src/ltd/prompts.py`):

| Key | Policy | Instruction |
| --- | --- | --- |
| `target` | **Target-CoT** | think *and* answer in L — the operating point consistency training implies |
| `en_cot` | **En-CoT** | think in English, answer in L |
| `free` | **Free-CoT** | no language constraint — reveals the model's preferred medium |
| `mixed` | **Mixed-CoT** | think mainly in English, preserve names/quantities/quotes in L, answer in L |

**Languages** — `hi` Hindi, `bn` Bengali, `te` Telugu (Indic / high fertility),
`sw` Swahili (Latin script, low resource), `de` German, `es` Spanish (close,
high resource), plus `en` as the anchor.

**Tasks** — MGSM (250 items/language, numeric) and Global-MMLU
culturally-agnostic split (200 items/language, 4-way MCQ).

**Models** — `r1-llama-8b` (DeepSeek-R1-Distill-Llama-8B) and `qwen3-8b`
(Qwen3-8B in thinking mode). `r1-qwen-7b` and `qwen3-4b` are configured for the
scale contrast; `smoke-qwen-0_6b` is for plumbing checks only.

Total: 2 models × (250 + 200) items × 7 languages × 4 conditions = **25,200
generations**.

---

## 3. Repo-local by construction

Nothing is written to `$HOME`. Every cache-bearing environment variable is
redirected into the checkout in **two** places, so it holds whether you enter
through the shell or through Python:

* [`scripts/env.sh`](scripts/env.sh) — sourced by `run.sh` before Python starts
* [`src/ltd/env.py`](src/ltd/env.py) — runs on `import ltd`, before `torch`/`vllm`

Redirected: `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`,
`XDG_CACHE_HOME`, `XDG_CONFIG_HOME`, `TORCH_HOME`, `TORCHINDUCTOR_CACHE_DIR`,
`TRITON_CACHE_DIR`, `MPLCONFIGDIR`, `NLTK_DATA`, `TIKTOKEN_CACHE_DIR`,
`NUMBA_CACHE_DIR`, `VLLM_CACHE_ROOT`, `VLLM_ASSETS_CACHE`, `OUTLINES_CACHE_DIR`,
`FLASHINFER_WORKSPACE_BASE`, `UV_CACHE_DIR`, `UV_PROJECT_ENVIRONMENT`.

Verify at any time — this prints every variable that still resolves outside the
repo, and says so explicitly when none do:

```bash
./run.sh doctor
```

Disk budget: weights ≈ 32 GB (two 8B bf16 models), datasets ≈ 1 GB, predictions
≈ 1–2 GB per full run. `./run.sh clean-caches` reclaims `assets/cache` while
keeping weights and data.

---

## 4. Command reference

`./run.sh <stage> [args...]` forwards `args` to the CLI. The CLI is also directly
usable: `.venv/bin/python -m ltd <command>` (or `ltd <command>` with the venv
activated).

| Stage | CLI | What it does |
| --- | --- | --- |
| `setup` | — | creates `.venv` via uv, installs `[gpu]` on Linux+NVIDIA else `[dev]` |
| `doctor` | `ltd doctor` | environment, GPU, backend choice, cache containment, data/model/run inventory |
| `data` | `ltd fetch-data --verify` | builds `data/processed/<task>/<lang>.jsonl`, verifies cross-language alignment |
| `models` | `ltd fetch-models` | snapshots weights into `assets/models/snapshots/` |
| `prompts` | `ltd prompts --show` | exports/prints every prompt block (release artefact + review) |
| `generate` | `ltd generate` | inference; resumable; the long stage |
| `status` | `ltd status` | per-cell progress, throughput, truncation, errors |
| `score` | `ltd score` | extraction + metrics → `results/results.parquet` |
| `fertility` | `ltd fertility` | per-language tokenizer fertility |
| `analyze` | `ltd analyze` | H1–H5 tests → `results/tables/`, `results/analysis.json` |
| `figures` | `ltd figures` | the four figures (PDF + PNG) |
| `policy-card` | `ltd policy-card` | blank template + filled card per model |
| `verify` | `ltd verify` | check prediction shards for corruption, duplication, ledger disagreement |
| `export` | `ltd export` | archive predictions to `exports/` (self-contained parquet + checksummed manifest) |
| `report` | `ltd report` | score + fertility + analyze + figures + policy-card + export |
| `clean-caches` | — | deletes `assets/cache`, keeps weights and data |

Selection flags accepted by `generate`, `score`, `fetch-data`, `fetch-models`,
`fertility`, `report`:

```
--models r1-llama-8b qwen3-8b | all
--tasks  mgsm gmmlu           | all
--langs  en hi bn te sw de es | all
--conditions target en_cot free mixed | all
```

`generate` also takes `--run-id`, `--resume`, `--backend {auto,vllm,hf}`,
`--limit N`, `--n-samples N`, `--chunk-size N`, `--max-new-tokens N`,
`--instruction-language {en,native}`.

---

## 5. Experiment recipes

Every command below is safe to interrupt and rerun.

**E0 — plumbing check (minutes, CPU ok).**

```bash
./run.sh smoke
```

**E1 — the main experiment (the paper's table and figures).**

```bash
./run.sh generate --run-id main-full
./run.sh report   --run-id main-full
```

**E2 — one model at a time** (the practical way to run on one GPU: each command
loads one model, so you can stop between them).

```bash
./run.sh generate --run-id main-r1llama --models r1-llama-8b
./run.sh generate --run-id main-qwen3   --models qwen3-8b
./run.sh report   --run-id main-r1llama main-qwen3     # analyse both together
```

**E3 — MGSM first** (H1/H2/H3/H4 only need the numeric task; get the landscape
figure early).

```bash
./run.sh generate --run-id mgsm-only --tasks mgsm
./run.sh report   --run-id mgsm-only
```

**E4 — the Indic core** (the languages the position paper's argument turns on).

```bash
./run.sh generate --run-id indic --langs hi bn te en --tasks mgsm
./run.sh report   --run-id indic
```

**E5 — split by policy** (useful when GPU time comes in short blocks).

```bash
./run.sh generate --run-id pol-target --conditions target
./run.sh generate --run-id pol-encot  --conditions en_cot
./run.sh generate --run-id pol-free   --conditions free
./run.sh generate --run-id pol-mixed  --conditions mixed
./run.sh report   --run-id all
```

**E6 — scale contrast** (Qwen3-4B vs Qwen3-8B, same tokenizer).

```bash
./run.sh models   --models qwen3-4b
./run.sh generate --run-id scale-4b --models qwen3-4b --tasks mgsm
./run.sh report   --run-id scale-4b main-qwen3
```

**E7 — native-instruction robustness** (appendix: does writing the instruction in
L instead of English change the picture?).

```bash
./run.sh generate --run-id native-instr --instruction-language native \
                  --tasks mgsm --langs hi bn te de es
./run.sh report   --run-id native-instr
```

**E8 — tokenizer fertility alone** (no GPU needed; the mechanism behind H3).

```bash
./run.sh fertility --models all
```

**E8b — validate the analysis with no GPU at all.** `scripts/make_synthetic_results.py`
writes a `results_synth.parquet` of exactly the real experiment's shape (25,200
rows, 112 cells) with *known* effects planted for H1–H5. Use it to confirm the
analysis recovers them and every figure renders before you spend GPU hours:

```bash
.venv/bin/python scripts/make_synthetic_results.py
.venv/bin/python -m ltd analyze --results results/results_synth.parquet --skip-fertility
.venv/bin/python -m ltd figures --results results/results_synth.parquet

# then clear the synthetic outputs before a real run
rm -f results/results_synth.parquet results/tables/fertility.csv
```

Expect: H1 gap larger for distant languages with positive Spearman ρ against
`typ_distance`; H3 Target-CoT dominated in the Indic cells; H5 argmax `mixed`.
These are **not results** — the numbers are synthetic by construction.

**E9 — longer budget for the truncation confound.** Target-script traces are
longer, so a fixed token budget truncates them more often. Rerun the Indic cells
with a bigger budget and compare `truncation_rate`:

```bash
./run.sh generate --run-id budget6k --tasks mgsm --langs hi bn te \
                  --max-new-tokens 6144
./run.sh report   --run-id budget6k
```

---

## 6. Resuming and splitting runs

**Runs are separated by time or by algorithm.** `--run-id` defaults to a
timestamp (`run-20260730-141500`); pass an explicit id to group a run by model,
policy, or language set (as in E2/E5 above). `score`, `analyze` and `report`
accept **several** run ids, plus `latest` and `all`, so runs launched separately
are merged for analysis.

**Resuming.** The unit of work is a *cell* = (model, task, language, condition).
Predictions are appended to
`runs/<run_id>/predictions/<model>/<task>/<lang>/<condition>.jsonl` and flushed
with `fsync` after every `--chunk-size` generations (default 64). To resume, run
the same command with the same `--run-id`:

```bash
./run.sh generate --run-id main-full        # picks up exactly where it stopped
```

On resume the code:

* reads the item ids already present in each shard and generates only the rest,
* repairs a truncated trailing line (the kill-during-write artefact) instead of
  crashing,
* skips loading a model entirely when all of its cells are already complete,
* reuses per-cell score caches, rescoring only shards that changed.

Check progress at any time, including from another shell while a run is going:

```bash
./run.sh status --run-id main-full
./run.sh status --list
```

Seeds are derived deterministically from `(config.seed, cell, chunk offset)` with
SHA-256, so a resumed run samples exactly as an uninterrupted one would.

---

## 7. Predictions are kept, losslessly

Every generation is written to disk and nothing is thrown away. This matters
because re-running the experiment costs GPU-days, and because a year from now the
only defensible answer to "what exactly did the model see?" is a stored record.

**What each prediction row carries** (`runs/<run_id>/predictions/**.jsonl`, one
JSON object per generation):

| Group | Fields |
| --- | --- |
| Identity | `item_id`, `sample_index`, `model`, `task`, `language`, `condition`, `run_id` |
| Raw output | `output` (the **complete** generated text, trace included, truncated or not) |
| Exact input | `prompt` (fully chat-templated, byte-for-byte what the model was fed), `user_message`, `system_prompt` |
| Provenance | `model_hf_id`, `model_revision` (Hub commit SHA), `decoding`, `seed`, `max_new_tokens`, `backend`, `instruction_language`, `generated_at` |
| Outcome | `gold`, `n_prompt_tokens`, `n_completion_tokens`, `finish_reason`, `truncated` |
| Item context | `question`, `question_en`, `options`, `subject`, `subject_category`, `sample_id` |

The rendered `prompt` is stored rather than recomputed on demand: a later
`transformers` or chat-template version would render it differently, and there
would be no way to detect that after the fact.

**Archive before you delete anything.** `runs/` is resumable working state and is
gitignored; `exports/` is the durable artefact:

```bash
./run.sh verify --run-id all      # shard integrity: corruption, duplicates, ledger drift
./run.sh export --run-id all      # -> exports/<run_id>/{predictions.parquet, manifest.json}
./run.sh export --run-id all --archive   # also tar.gz the raw run dir for cold storage
```

`exports/<run_id>/predictions.parquet` is **self-contained**: raw generations plus
every score, in one zstd-compressed file readable with nothing but pandas.

```python
import pandas as pd
df = pd.read_parquet("exports/main-full/predictions.parquet")   # ~56 columns
```

`manifest.json` is the provenance record — per-shard SHA-256 checksums, the
parquet's own checksum, model revisions, the frozen run config, the git commit,
and the integrity report. It is small and **is committed to git**, so the
provenance of a result survives even when the data lives elsewhere.

`ltd report` exports automatically (`--no-export` to skip). Export is read-only
over `runs/` and never mutates a run.

Two safeguards worth knowing about:

* Export **refuses to be silent** about gaps. If shards predate a schema field, the
  missing columns are logged as a warning and recorded in
  `manifest.missing_provenance_columns`.
* Scoring a *subset* will not quietly shrink a fuller `results/results.parquet` —
  the previous file is preserved as `results.superseded.parquet` with a warning.
  (Raw predictions are never at risk from this; only the derived view.)

---

## 8. Outputs

```
results/
  results.parquet              one row per generation, all metrics
  analysis.json                H1-H5 verdicts, caveats, literature map
  tables/
    cells.csv                  per-cell accuracy/cost/compliance/retention + 95% CIs
    accuracy_matrix.csv        language × condition accuracy, with per-cell prior-work citation
    h1_ordering.csv            paired contrasts, bootstrap CIs, exact McNemar p
    h3_pareto.csv              Pareto status of each policy per language
    h4_drift.json              logistic regressions of correctness on English drift
    h5_retention.csv           retention by policy with CIs
    fertility.csv              tokens/word per language, ratio vs English
  figures/
    fig1_landscape_<model>_<task>.pdf   accuracy × cost, Pareto front, consistency corner
    fig2_accuracy_matrix.pdf            language × condition heatmap with CIs
    fig3_drift_and_retention.pdf        H4 drift curve + H5 retention bars
    fig4_gap_vs_distance.pdf            gap vs typological distance and vs fertility
artifacts/
  policy_card_template.md      the blank disclosure artefact the paper proposes
  policy_card_<model>.md       filled from this run, as the worked example
  prompts.json                 every prompt block, for release
exports/<run_id>/
  predictions.parquet          LOSSLESS archive: raw outputs + exact prompts + scores
  manifest.json                checksums, model revisions, config, integrity report
  <run_id>_raw.tar.gz          the whole run directory (with --archive)
```

Per-generation columns in `results.parquet` include `correct`, `prediction`,
`n_completion_tokens`, `truncated`, `trace_target_share`, `trace_english_share`,
`trace_compliant`, `answer_compliant`, `number_retention`, `entity_retention`,
`anchor_retention`, `extraction_failed`, `well_formed_trace`, `seed`,
`degenerate_cell`.

---

## 9. Hypotheses → where each is computed

| Hypothesis | Computed in | Output |
| --- | --- | --- |
| **H1** En-CoT ≥ Mixed > Target for hi/bn/te/sw; gap shrinks for de/es | `analyze.h1_ordering`, `analyze.h1_verdict` | `tables/h1_ordering.csv`, `analysis.json:verdicts.H1`, fig 2 & 4 |
| **H2** Mixed-CoT recovers most of En-CoT accuracy while keeping target anchors | `analyze.h2_verdict` | `analysis.json:verdicts.H2` (`recovery_fraction`) |
| **H3** Target-CoT is Pareto-dominated on accuracy × cost for high-fertility languages | `analyze.h3_pareto`, `analyze.h3_verdict`, `fertility` | `tables/h3_pareto.csv`, **fig 1**, fig 4c |
| **H4** Within Target-CoT, English drift predicts correctness | `analyze.h4_drift` (logit, controls for length) | `tables/h4_drift.json`, fig 3a |
| **H5** Anchor retention is highest under Mixed-CoT | `analyze.h5_retention`, `faithfulness.retention` | `tables/h5_retention.csv`, fig 3b |

Each cell in `accuracy_matrix.csv` carries a `reproduces` column naming the prior
finding it reproduces (XReasoning, Language Matters, Cross-lingual Collapse,
Impact of Language Mixing, KO-REAson), which is what §5 of the paper needs.

`analysis.json` also carries a `caveats` list that the paper should quote:
prompted ≠ trained, retention is a lower bound, `typ_distance` is an
author-assigned ordinal proxy, truncation rates differ by language, and the
English-anchor cells are degenerate by construction.

---

## 10. Data and models: what is actually downloaded

| What | Source | Notes |
| --- | --- | --- |
| MGSM (`bn de en es sw te`) | `juletxara/mgsm`, `test` | 250 items, item-aligned across languages by row index (verified) |
| MGSM Hindi | `nvidia/GSM8K-Hi` + `openai/gsm8k` | see below |
| Global-MMLU | `CohereLabs/Global-MMLU`, `test` | filtered to `cultural_sensitivity_label == "CA"`, then 200 items stratified by `subject_category` on the **intersection** of `sample_id`s across all languages |
| Models | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, `Qwen/Qwen3-8B` | ≈ 32 GB total |

**Hindi is not in MGSM** (its configs are `bn de en es fr ja ru sw te th zh`). The
build creates an item-aligned MGSM-Hi: it locates each MGSM-en question in the
GSM8K `main/test` split, takes the Hindi translation at the same index from
`nvidia/GSM8K-Hi`, and keeps the item only if the Hindi gold answer parses to the
same number as MGSM's `answer_number`. On the current data this yields all
**250/250 items with 0 gold mismatches**, and the build logs confirm MGSM's test
split is the first 250 GSM8K test items. Items that failed the check would be
dropped and logged rather than silently kept.

`ltd fetch-data --verify` (what `./run.sh data` runs) asserts that item ids and
gold labels agree across every language before you spend GPU time.

---


## 12. Logging

Four sinks, always:

* console — INFO, terse, plays nicely with progress bars
* `logs/<run_id>/<command>.log` — DEBUG, everything for that command
* `logs/ltd.log` — INFO, append-only across all runs, the audit trail
* `logs/run_sh.log` — the shell driver's own stage log

Progress bars (tqdm) cover data loading, model download, generation (inner
per-cell and outer whole-run), scoring, fertility, bootstrap aggregation,
hypothesis fitting and figure building. Because a bar's redraws never reach a log
file, `LoggedProgress` also writes a progress line to the log every 10 % with
elapsed time, rate, ETA, live tokens/s and truncation rate — so a crashed run is
diagnosable from `logs/` alone.

```bash
tail -f logs/main-full/generate.log
grep -E "^H[1-5]|Pareto|gap" logs/adhoc/analyze.log
```

---

## 13. Only one model is ever in GPU memory

With `gpu_memory_utilization: 0.90`, vLLM reserves ~43 GB of a 48 GB card for the
KV cache. **Two models cannot co-reside.** The pipeline therefore loads models
strictly one at a time — the loop is `model → task → language → condition`, and
the previous model is released before the next is loaded. Two mechanisms enforce
it:

| `--model-isolation` | What happens | When |
| --- | --- | --- |
| `subprocess` | Each model runs in its own child process. When it exits the **OS** reclaims every byte, regardless of what the inference library left behind. | default when >1 model on vLLM |
| `inprocess` | The engine is torn down in place: engine core shut down, model executor dropped, vLLM's distributed state destroyed, then `gc.collect()` + `empty_cache()`. | single model, or the `hf` backend |
| `auto` | `subprocess` when more than one model runs on vLLM, else `inprocess`. | the default |

Subprocess isolation is the default for multi-model runs because in-process
teardown depends on vLLM internals that move between releases — a child process
is the only guarantee the OS itself enforces. It costs one model load (~1 min)
per model and nothing else: all state lives in `runs/<run_id>/`, so the workers
coordinate through the filesystem and the run stays fully resumable. A worker that
dies is retried by rerunning the same command.

```bash
./run.sh generate --run-id main-full                              # auto -> subprocess
./run.sh generate --run-id main-full --model-isolation inprocess  # force in-place
```

**Memory is verified, not assumed.** GPU usage is logged before loading, after
loading and after releasing each model. If more than 2 GiB fails to come back and
another model is still queued, you get an error naming the leak and telling you to
switch to `--model-isolation subprocess` — instead of an OOM ten minutes later:

```
ERROR | 41.2 GiB of GPU memory was NOT released after r1-llama-8b
        (baseline 0.4 GiB, now 41.6 GiB) — the next model will probably OOM.
        Rerun with `--model-isolation subprocess`.
```

If you would rather run models in separate invocations entirely (recipe E2), that
also works and always has — `report` merges runs.

---

## 14. Tuning for your GPU

Defaults in `configs/default.yaml` are set for **one 48 GB card** with an 8B bf16
model (16 GB weights, the rest KV cache):

```yaml
generation:
  gpu_memory_utilization: 0.90
  max_num_seqs: 128
  chunk_size: 64          # generations per checkpoint flush
  swap_space_gb: 4
models:
  qwen3-8b:
    max_model_len: 4608   # prompts are < 600 tokens; 3072 new tokens for MGSM
```

* **OOM at load:** lower `gpu_memory_utilization` to `0.85`, then `max_num_seqs`
  to 64. If it happens on the *second* model, the first was not released — see
  §13 and use `--model-isolation subprocess`.
* **OOM mid-run:** lower `max_num_seqs`; leave `gpu_memory_utilization` alone.
* **Throughput:** raise `chunk_size` (fewer flushes, more work lost on a crash) and
  `max_num_seqs`. 64/128 is a deliberate compromise favouring resumability.
* **Truncation too high** (watch the `trunc` column in `status`/`score`): raise
  `max_new_tokens` and `max_model_len` together. Truncation is a real confound
  here — Indic traces are longer, so a fixed budget penalises them; always report
  `truncation_rate` next to accuracy (see E9).

---
## 16. Repository layout

```
run.sh                       single entrypoint (setup / data / models / generate / report / ...)
scripts/env.sh               repo-local environment, sourced before Python starts
pyproject.toml               uv project; [gpu] = vLLM (Linux), [dev] = transformers
configs/
  default.yaml               languages, tasks, models, conditions, decoding, scoring
  native_instructions.yaml   translated instruction blocks (appendix)
src/ltd/
  env.py                     pins every cache into the repo (imported first)
  config.py                  typed config
  logging_utils.py           4-sink logging + LoggedProgress
  data_build.py              MGSM (+ Hindi bridge) and Global-MMLU builders
  prompts.py                 the four conditions, chat-template rendering
  models_fetch.py            snapshot weights into assets/models
  backends.py                vLLM and transformers backends
  runstore.py                run dirs, append-only shards, resume ledger
  generate.py                the inference driver (chunked, checkpointed)
  extract.py                 trace/answer split, Unicode-digit-safe extraction
  trace_language.py          script ranges + langid, compliance, English drift
  faithfulness.py            number/entity anchor retention (H5)
  fertility.py               per-language tokenizer fertility (H3 mechanism)
  score.py                   shards -> results.parquet (cached per cell)
  stats_utils.py             bootstrap CIs, McNemar, logit, Pareto
  analyze.py                 H1-H5 and the tables
  figures.py                 the four figures
  policy_card.py             the disclosure artefact
  cli.py                     argparse CLI
data/       raw/ (HF cache) and processed/<task>/<lang>.jsonl
assets/     models/ (weights) and cache/ (every library cache)
runs/       <run_id>/{config.json,state.json,predictions/,scored/}
logs/       ltd.log, run_sh.log, <run_id>/<command>.log
results/    results.parquet, analysis.json, tables/, figures/
artifacts/  policy cards, prompts.json
paper/      the submission; builds with XeLaTeX (the qualitative appendix sets Indic scripts)
  fonts/                     vendored Noto Devanagari/Bengali/Telugu, SIL OFL 1.1
```

### Regenerating the paper's numbers

```bash
.venv/bin/python scripts/paper_stats.py               # every aggregate table and statistic
.venv/bin/python scripts/make_qualitative_examples.py # transcripts, error taxonomy, trace overlap
```

The second reads `runs/main-full-v2/{predictions,scored}` rather than
`results/`, since the qualitative appendix quotes generated text. It writes
`results/tables/{error_taxonomy,trace_overlap}.csv`,
`results/tables/qualitative_examples.json`, and prints the LaTeX transcript blocks
to stdout for pasting into `paper/neurips_2026.tex`, which holds them inline.
The transcripts it selects are pinned by item id, and each pin asserts the
property it was chosen to illustrate, so the script fails rather than silently
substituting an unrepresentative generation.

### Citation

If you use this code, cite the position paper and the five prior works it
reproduces: XReasoning (arXiv 2505.22888), Language Matters (2505.17407),
Cross-lingual Collapse (2506.05850), Impact of Language Mixing (2507.15849), and
Long CoT Across Languages (2508.14828).
