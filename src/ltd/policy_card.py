"""The thinking-language policy card — the paper's disclosure artefact (spec §10).

Two products:

* a blank **template** (``artifacts/policy_card_template.md``) that the paper asks
  reasoning-model releases to fill in, and
* a **filled** card per model, generated from this run's measurements, as the
  worked example: per language, the recommended policy, the accuracy delta vs.
  free thinking, the cost delta, and the faithfulness delta.

The recommendation follows the decision framework of §7: on verifiable-reasoning
tasks prefer the policy that is Pareto-optimal on accuracy × cost, breaking ties
towards the one that keeps target-language anchors (Mixed-CoT) — and always
disclose rather than silently defaulting to consistency.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from .config import CONDITION_LABELS, Config
from .env import ARTIFACTS_DIR
from .logging_utils import get_logger

log = get_logger("policy_card")

TEMPLATE = """# Thinking-Language Policy Card

*Template accompanying "Position: The Language of Thought Is a Design Choice, Not a
Compliance Problem" (LP4FM @ NeurIPS 2026). Copy this file into your model release
and fill every row. The ask is disclosure, not any particular policy.*

## 1. Model

| Field | Value |
| --- | --- |
| Model | |
| Release date | |
| Reasoning mode | |
| Tokenizer | |

## 2. Declared thinking-language policy

> One sentence: what language does this model reason in, as a function of the input
> language, and was that an explicit design choice?

- **Policy:** `consistency` | `english-hub` | `free` | `mixed-anchored` | `adaptive` | `undisclosed`
- **Enforced by:** prompt / SFT / RL reward / constrained decoding / nothing
- **User-visible trace:** yes / no / configurable

## 3. Per-language disclosure

For each supported language, report the measured cost of the shipped policy against
a *free-thinking* baseline on a verifiable-reasoning benchmark. State the benchmark,
the number of items, and the decoding config.

| Language | Script | Shipped policy | Δ accuracy vs free (pp) | Δ completion tokens vs free (%) | Δ faithfulness vs free | Tokenizer fertility vs English (×) | Trace-language compliance (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | |

## 4. Task-conditioning

Does the policy vary by task type? (§7 of the position paper argues it should.)

| Task type | Policy | Justification |
| --- | --- | --- |
| Verifiable reasoning (math, code, logic) | | |
| Cultural / contextual | | |
| Safety-sensitive | | |
| Open generation | | |

## 5. Justification and known costs

- Why this policy, for these languages?
- Which languages pay the largest accuracy tax under it, and how large is it?
- If the policy is `consistency`: what accuracy budget was allocated to it, and was
  the budget met?
- What would make you change the policy?

## 6. Reproducibility

| Field | Value |
| --- | --- |
| Benchmark(s) and split | |
| Items per language | |
| Decoding config | |
| Seeds | |
| Trace-language measurement method | |
| Faithfulness measurement method | |
"""


def write_template(path: Path | None = None) -> Path:
    target = path or ARTIFACTS_DIR / "policy_card_template.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(TEMPLATE, encoding="utf-8")
    log.info("policy card template -> %s", target)
    return target


#: Accuracy at or below this counts as "native capability is absent" for the
#: purposes of the framework's last row, which asks for no policy at all rather
#: than for a winner among indistinguishable near-zero cells. For the numeric task
#: the floor is zero; for the four-option MCQ it is the 25% chance rate.
FLOOR = {"mgsm": 0.05, "gmmlu": 0.30}


def _recommend(part: Any) -> tuple[str, str]:
    """Pick a policy for one (model, task, language) cell and say why."""
    from .stats_utils import pareto_front

    rows = part.reset_index(drop=True)
    points = list(zip(rows["accuracy"], rows["mean_completion_tokens"], strict=True))
    front = set(pareto_front(points))
    candidates = rows.loc[sorted(front)]

    # Framework row 6: where the capability is absent, no thinking-language policy
    # is better than another, and recommending one would be the compliance-as-
    # camouflage failure the paper argues against.
    task_name = str(rows["task"].iloc[0]) if "task" in rows else ""
    floor = FLOOR.get(task_name)
    if floor is not None and rows["accuracy"].max() <= floor:
        return (
            "none — invest in capability",
            f"every policy is at or below the floor for this task "
            f"(best accuracy {rows['accuracy'].max():.3f}); the framework asks for "
            f"disclosure of the gap, not a policy over it",
        )

    # Among Pareto-optimal policies, treat as tied every policy whose accuracy is
    # inside the bootstrap interval of the most accurate one -- this study's own
    # finding is that the accuracy axis is flat at this sample size, so ranking on
    # a point estimate would recommend noise. Ties then go to the framework's
    # default: the policy that keeps the question's anchors and answers natively.
    best_row = candidates.loc[candidates["accuracy"].idxmax()]
    band = best_row["accuracy_lo"] if "accuracy_lo" in candidates else best_row["accuracy"] - 0.01
    if band != band:  # NaN
        band = best_row["accuracy"] - 0.01
    near_best = candidates[candidates["accuracy"] >= band]
    preference = ["mixed", "en_cot", "free", "target"]
    chosen = min(
        near_best["condition"].astype(str),
        key=lambda condition: preference.index(condition) if condition in preference else 99,
    )
    chosen_row = rows[rows["condition"].astype(str) == chosen].iloc[0]
    target_row = rows[rows["condition"].astype(str) == "target"]
    reason = f"Pareto-optimal on accuracy × cost (acc={chosen_row['accuracy']:.3f}, " \
             f"{chosen_row['mean_completion_tokens']:.0f} tok)"
    if str(best_row["condition"]) != chosen:
        reason += (
            f"; indistinguishable from the most accurate policy "
            f"({best_row['condition_label']}, acc={best_row['accuracy']:.3f}), so the "
            f"framework default breaks the tie"
        )
    if not target_row.empty and chosen != "target":
        delta = 100 * (chosen_row["accuracy"] - target_row.iloc[0]["accuracy"])
        cost = target_row.iloc[0]["mean_completion_tokens"] - chosen_row["mean_completion_tokens"]
        reason += (
            f"; consistency (Target-CoT) would cost {delta:.1f} pp accuracy and "
            f"{cost:+.0f} tokens"
        )
    return CONDITION_LABELS.get(chosen, chosen), reason


def build_filled(
    cells: Any,
    config: Config,
    fertility: Any | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """One filled card per model, from this run's measurements."""
    target_dir = output_dir or ARTIFACTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for model in sorted(cells["model"].unique()):
        spec = config.model(model)
        lines = [
            f"# Thinking-Language Policy Card — {spec.hf_id}",
            "",
            "*Filled automatically by `ltd policy-card` from this repository's own "
            "measurements, as the worked example accompanying "
            '"Position: The Language of Thought Is a Design Choice, Not a Compliance Problem".*',
            "",
            f"Generated: {date.today().isoformat()}",
            "",
            "## 1. Model",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Model | `{spec.hf_id}` |",
            f"| Reasoning mode | {spec.thinking} |",
            f"| Decoding | {spec.decoding} |",
            f"| Max model len | {spec.max_model_len} |",
            "",
            "## 2. Measured policy landscape",
            "",
            "Deltas are against **Free-CoT** (no thinking-language constraint), which is the "
            "baseline the position paper asks releases to disclose against, because it is the "
            "model's own preferred medium.",
            "",
        ]

        for task_name in sorted(cells[cells["model"] == model]["task"].unique()):
            part_task = cells[(cells["model"] == model) & (cells["task"] == task_name)]
            lines += [
                f"### Task: `{task_name}`",
                "",
                "| Language | Script | Policy | Accuracy | Δacc vs Free (pp) | Mean tokens | "
                "Δtok vs Free (%) | Anchor retention | Δretention vs Free | Compliance | Trunc. |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for lang_code in sorted(
                part_task["language"].unique(),
                key=lambda code: config.language(code).typ_distance,
            ):
                part = part_task[part_task["language"] == lang_code]
                free = part[part["condition"].astype(str) == "free"]
                base_accuracy = float(free["accuracy"].iloc[0]) if not free.empty else None
                base_tokens = float(free["mean_completion_tokens"].iloc[0]) if not free.empty else None
                base_retention = float(free["anchor_retention"].iloc[0]) if not free.empty else None
                lang_spec = config.language(lang_code)
                for _, row in part.iterrows():
                    d_acc = (
                        f"{100 * (row['accuracy'] - base_accuracy):+.1f}"
                        if base_accuracy is not None else "n/a"
                    )
                    d_tok = (
                        f"{100 * (row['mean_completion_tokens'] / base_tokens - 1):+.1f}"
                        if base_tokens else "n/a"
                    )
                    retention = row["anchor_retention"]
                    d_ret = (
                        f"{retention - base_retention:+.3f}"
                        if base_retention is not None and retention == retention
                        else "n/a"
                    )
                    lines.append(
                        f"| {lang_spec.name} ({lang_code}) | {lang_spec.script} | "
                        f"{row['condition_label']} | {row['accuracy']:.3f} | {d_acc} | "
                        f"{row['mean_completion_tokens']:.0f} | {d_tok} | "
                        f"{retention:.3f} | {d_ret} | "
                        f"{100 * row['compliance_rate']:.0f}% | "
                        f"{100 * row['truncation_rate']:.0f}% |"
                    )
            lines.append("")

            lines += [
                "#### Recommended policy under the §7 decision framework",
                "",
                "| Language | Recommended | Fertility vs En (×) | Rationale |",
                "| --- | --- | --- | --- |",
            ]
            for lang_code in sorted(
                part_task["language"].unique(),
                key=lambda code: config.language(code).typ_distance,
            ):
                part = part_task[part_task["language"] == lang_code]
                recommended, reason = _recommend(part)
                ratio = "n/a"
                if fertility is not None and not getattr(fertility, "empty", True):
                    match = fertility[
                        (fertility["model"] == model)
                        & (fertility["task"] == task_name)
                        & (fertility["language"] == lang_code)
                    ]
                    if not match.empty and match["fertility_ratio"].notna().any():
                        value = float(match["fertility_ratio"].mean())
                        # A non-English ratio below 1 is impossible for a byte-level
                        # BPE with full byte coverage, so it is an instrumentation
                        # artifact and must not be shown as a measurement.
                        ratio = (
                            "n/a (instrumentation artifact)"
                            if lang_code != "en" and value < 1.0
                            else f"{value:.2f}"
                        )
                lines.append(
                    f"| {config.language(lang_code).name} ({lang_code}) | {recommended} | "
                    f"{ratio} | {reason} |"
                )
            lines.append("")

        lines += [
            "## 3. Caveats this card must carry",
            "",
            "- These are **prompted** policies measured at inference time, not trained ones. "
            "The training-side price of consistency is reported independently at 5–10 pp "
            "(Cross-lingual Collapse, arXiv 2506.05850) and agrees in sign with this map.",
            "- Retention is verbatim anchor retention, a *lower bound* on faithfulness.",
            "- Rows for `en` where Target-CoT / En-CoT / Mixed-CoT coincide are degenerate by "
            "construction and measure prompt-wording noise, not a language effect.",
            "- Truncation rates differ across languages because target-script traces are longer; "
            "read accuracy together with the truncation column.",
            "",
            "## 4. What the position paper asks of releases",
            "",
            "Disclose the shipped policy and its measured per-language cost, rather than "
            "defaulting to language consistency and leaving the tax unmeasured. See "
            "`artifacts/policy_card_template.md` for the blank form.",
            "",
        ]

        path = target_dir / f"policy_card_{model}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        log.info("filled policy card -> %s", path)
        written.append(path)
    return written
