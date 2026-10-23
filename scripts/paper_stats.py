#!/usr/bin/env python3
"""Recompute every number quoted in the paper from ``results/``.

The paper's tables are pasted into ``paper/neurips_2026.tex`` as literal LaTeX so
that it compiles without a Python toolchain. This script is the audit trail: it
regenerates the same table bodies and the derived statistics quoted in the text,
straight from ``results/tables/*.csv`` and ``results/analysis.json``.

    python scripts/paper_stats.py            # print everything
    python scripts/paper_stats.py --tables   # only the LaTeX table bodies
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
ANALYSIS = ROOT / "results" / "analysis.json"

LANGS = ["de", "es", "sw", "hi", "bn", "te"]
NAMES = {
    "en": "English", "de": "German", "es": "Spanish",
    "sw": "Swahili", "hi": "Hindi", "bn": "Bengali", "te": "Telugu",
}
DISTANT = ["sw", "hi", "bn", "te"]
CLOSE = ["de", "es"]
CONDS = ["target", "en_cot", "free", "mixed"]
LABELS = {"target": "Target-CoT", "en_cot": "En-CoT", "free": "Free-CoT", "mixed": "Mixed-CoT"}


def load(name: str) -> list[dict[str, str]]:
    with (TABLES / f"{name}.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


CELLS = load("cells")
H1 = load("h1_ordering")
FERT = load("fertility")
H5 = load("h5_retention")
ANALYSIS_JSON = json.loads(ANALYSIS.read_text(encoding="utf-8"))


def cell(model: str, task: str, lang: str, cond: str) -> dict[str, str]:
    for row in CELLS:
        if (row["model"], row["task"], row["language"], row["condition"]) == (model, task, lang, cond):
            return row
    raise KeyError((model, task, lang, cond))


def delta(model: str, task: str, lang: str, contrast: str) -> dict[str, str]:
    for row in H1:
        if (row["model"], row["task"], row["language"], row["contrast"]) == (model, task, lang, contrast):
            return row
    raise KeyError((model, task, lang, contrast))


def fertility(model: str, task: str, lang: str) -> float:
    for row in FERT:
        if (row["model"], row["task"], row["language"]) == (model, task, lang):
            return float(row["fertility_ratio"])
    raise KeyError((model, task, lang))


def spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    rx, ry = rank(xs), rank(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    rho = num / den
    n = len(xs)
    t = rho * math.sqrt((n - 2) / max(1 - rho ** 2, 1e-12))
    return rho, _t_sf(abs(t), n - 2) * 2


def _t_sf(t: float, df: int) -> float:
    """One-sided survival function of Student's t (regularised incomplete beta)."""
    x = df / (df + t * t)

    def betacf(a: float, b: float, x: float) -> float:
        tiny, eps = 1e-300, 3e-16
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) > tiny else tiny)
        h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            step = d * c
            h *= step
            if abs(step - 1.0) < eps:
                break
        return h

    def betai(a: float, b: float, x: float) -> float:
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        front = math.exp(
            math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
            + a * math.log(x) + b * math.log(1 - x)
        )
        if x < (a + 1) / (a + b + 2):
            return front * betacf(a, b, x) / a
        return 1.0 - front * betacf(b, a, 1 - x) / b

    return betai(df / 2.0, 0.5, x) / 2.0


# ---------------------------------------------------------------------------
# Table 1 — accuracy and cost of each policy, Qwen3-8B
# ---------------------------------------------------------------------------
def table_main(model: str = "qwen3-8b") -> str:
    lines: list[str] = []
    for task, label in (("mgsm", "MGSM (250 items/language)"), ("gmmlu", "Global-MMLU CA (200 items/language)")):
        lines.append(rf"    \multicolumn{{7}}{{l}}{{\emph{{{label}}}}} \\")
        for lang in LANGS:
            base = cell(model, task, lang, "target")
            acc = 100 * float(base["accuracy"])
            row = [f"{NAMES[lang]} ({lang})", f"{acc:.1f}"]
            for contrast in ("En-CoT - Target-CoT", "Mixed-CoT - Target-CoT", "Free-CoT - Target-CoT"):
                d = delta(model, task, lang, contrast)
                row.append(
                    f"${float(d['delta_pp']):+.1f}$ "
                    f"[{float(d['delta_lo_pp']):+.1f}, {float(d['delta_hi_pp']):+.1f}]"
                )
            tokens = float(base["mean_completion_tokens"])
            ratio_en = tokens / float(cell(model, task, lang, "en_cot")["mean_completion_tokens"])
            row.append(f"{tokens:.0f} ({ratio_en:.2f}$\\times$)")
            lines.append("    " + " & ".join(row) + r" \\")
        lines.append(r"    \addlinespace")
    return "\n".join(lines).rstrip("\n").removesuffix(r"\addlinespace").rstrip()


# ---------------------------------------------------------------------------
# Table 2 — what each policy actually delivers, Qwen3-8B
# ---------------------------------------------------------------------------
def table_delivery(model: str = "qwen3-8b") -> str:
    panels = [
        ("trace_target_share", "Share of the reasoning trace in the target language"),
        ("target_anchor_rate", "Traces retaining at least one verbatim target-language anchor"),
        ("answer_compliance_rate", "Final answers written in the target language"),
    ]
    lines: list[str] = []
    for key, caption in panels:
        lines.append(rf"    \multicolumn{{7}}{{l}}{{\emph{{{caption}}}}} \\")
        for cond in CONDS:
            values = []
            for lang in LANGS:
                pair = [float(cell(model, task, lang, cond)[key]) for task in ("mgsm", "gmmlu")]
                values.append(f"{100 * mean(pair):.0f}")
            lines.append("    " + " & ".join([LABELS[cond]] + values) + r" \\")
        lines.append(r"    \addlinespace")
    return "\n".join(lines).rstrip("\n").removesuffix(r"\addlinespace").rstrip()


# ---------------------------------------------------------------------------
# Appendix Table — full accuracy matrix
# ---------------------------------------------------------------------------
def table_accuracy_matrix() -> str:
    lines: list[str] = []
    for model in ("qwen3-8b", "r1-llama-8b"):
        for task in ("mgsm", "gmmlu"):
            lines.append(rf"    \multicolumn{{5}}{{l}}{{\emph{{{model}, {task}}}}} \\")
            for lang in ["en"] + LANGS:
                row = [f"{NAMES[lang]} ({lang})"]
                for cond in CONDS:
                    c = cell(model, task, lang, cond)
                    row.append(
                        f"{100 * float(c['accuracy']):.1f} "
                        f"[{100 * float(c['accuracy_lo']):.1f}, {100 * float(c['accuracy_hi']):.1f}]"
                    )
                lines.append("    " + " & ".join(row) + r" \\")
            lines.append(r"    \addlinespace")
    return "\n".join(lines).rstrip("\n").removesuffix(r"\addlinespace").rstrip()


def table_tokens_truncation() -> str:
    lines: list[str] = []
    for model in ("qwen3-8b", "r1-llama-8b"):
        for task in ("mgsm", "gmmlu"):
            lines.append(rf"    \multicolumn{{5}}{{l}}{{\emph{{{model}, {task}}}}} \\")
            for lang in ["en"] + LANGS:
                row = [f"{NAMES[lang]} ({lang})"]
                for cond in CONDS:
                    c = cell(model, task, lang, cond)
                    row.append(
                        f"{float(c['mean_completion_tokens']):.0f} / "
                        f"{100 * float(c['truncation_rate']):.1f}"
                    )
                lines.append("    " + " & ".join(row) + r" \\")
            lines.append(r"    \addlinespace")
    return "\n".join(lines).rstrip("\n").removesuffix(r"\addlinespace").rstrip()


def table_retention(model: str = "qwen3-8b") -> str:
    lines: list[str] = []
    for metric, caption in (
        ("entity_retention", "Entity retention"),
        ("number_retention", "Number retention"),
    ):
        lines.append(rf"    \multicolumn{{7}}{{l}}{{\emph{{{caption}}}}} \\")
        for cond in CONDS:
            values = []
            for lang in LANGS:
                picked = [
                    float(row["mean"]) for row in H5
                    if row["model"] == model and row["language"] == lang
                    and row["condition"] == cond and row["metric"] == metric
                ]
                values.append(f"{100 * mean(picked):.0f}" if picked else "--")
            lines.append("    " + " & ".join([LABELS[cond]] + values) + r" \\")
        lines.append(r"    \addlinespace")
    return "\n".join(lines).rstrip("\n").removesuffix(r"\addlinespace").rstrip()


def table_fertility(model: str = "qwen3-8b") -> str:
    lines = []
    for lang in ["en"] + LANGS:
        row = [f"{NAMES[lang]} ({lang})"]
        for task in ("mgsm", "gmmlu"):
            for key in ("tokens_per_word", "fertility_ratio"):
                value = [r for r in FERT if (r["model"], r["task"], r["language"]) == (model, task, lang)][0][key]
                row.append(f"{float(value):.2f}")
        lines.append("    " + " & ".join(row) + r" \\")
    return "\n".join(lines)


def _truncation_sensitivity() -> list[str]:
    """Truncated generations are scored on whatever answer they emitted.

    Appendix E quotes both the count that still scores correct and the largest
    per-cell move that forcing them to count as incorrect would produce. Both
    need the per-generation records rather than the cell aggregates.
    """
    parquet = ROOT / "results" / "results.parquet"
    if not parquet.exists():
        return ["truncation sensitivity: results/results.parquet not present, skipped"]
    try:
        import pandas as pd
    except ImportError:
        return ["truncation sensitivity: pandas not installed, skipped"]

    frame = pd.read_parquet(parquet, columns=["model", "task", "language", "condition",
                                              "correct", "truncated", "extraction_failed"])
    truncated = frame[frame["truncated"]]
    lines = [
        f"truncated generations: {len(truncated)}, of which scored correct: "
        f"{int(truncated['correct'].sum())} (they emitted a parseable answer before the cap)",
        f"extraction failures: {int(frame['extraction_failed'].sum())} "
        f"({100 * frame['extraction_failed'].mean():.2f}% of generations), all scored incorrect",
    ]
    keys = ["model", "task", "language", "condition"]
    as_scored = frame.groupby(keys, observed=True)["correct"].mean()
    forced = (
        frame.assign(strict=frame["correct"] & ~frame["truncated"])
        .groupby(keys, observed=True)["strict"].mean()
    )
    move = (100 * (as_scored - forced)).sort_values(ascending=False)
    lines.append("largest accuracy drops if truncated generations were forced incorrect:")
    for key, value in move.head(3).items():
        lines.append(f"    {'/'.join(map(str, key))}: -{value:.1f} pp")
    qwen = move[[key for key in move.index if key[0] == "qwen3-8b"]]
    lines.append(f"    max over all Qwen3-8B cells: -{qwen.max():.1f} pp")
    return lines


#: Byte-level BPE markers that should have been decoded back to whitespace.
BPE_MARKERS = ("Ġ", "Ċ", "ĉ")


def _decode_defect() -> list[str]:
    """Audit the R1-Distill text-decoding defect documented in Appendix M.

    Needs the exported per-generation records, which carry the raw output text;
    ``results/results.parquet`` keeps only the derived measurements.
    """
    export = ROOT / "exports" / "main-full-v2" / "predictions.parquet"
    if not export.exists():
        return ["decode defect: exports/main-full-v2/predictions.parquet not present, skipped"]
    try:
        import pandas as pd
    except ImportError:
        return ["decode defect: pandas not installed, skipped"]

    frame = pd.read_parquet(export, columns=["model", "task", "language", "condition",
                                             "item_id", "output", "question", "gold"])
    affected = frame["output"].map(lambda t: any(m in t for m in BPE_MARKERS))
    lines = ["records whose stored output carries undecoded byte-level markers:"]
    for model, share in affected.groupby(frame["model"], observed=True).mean().items():
        total = int(affected[frame["model"] == model].sum())
        lines.append(f"    {model}: {100 * share:.1f}% ({total} records)")
    lines.append(
        f"    question text affected: "
        f"{100 * frame['question'].map(lambda t: any(m in t for m in BPE_MARKERS)).mean():.1f}%"
    )

    # Rescore the affected model after re-decoding, to bound the accuracy impact.
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from ltd.extract import extract_prediction, is_correct, split_trace_answer
    except ImportError:
        lines.append("    (rescoring skipped: ltd package not importable)")
        return lines

    scored = pd.read_parquet(ROOT / "results" / "results.parquet",
                             columns=["model", "task", "language", "condition", "item_id", "correct"])
    merged = frame.merge(scored, on=["model", "task", "language", "condition", "item_id"])
    merged = merged[merged["model"] == "r1-llama-8b"]

    def decode(text: str) -> str:
        return text.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")

    flipped, worst = 0, 0.0
    for (task, lang, cond), part in merged.groupby(["task", "language", "condition"], observed=True):
        kind = "numeric" if task == "mgsm" else "mcq"
        redone = []
        for _, row in part.iterrows():
            text = decode(row["output"])
            _, answer, _ = split_trace_answer(text)
            redone.append(is_correct(kind, extract_prediction(kind, answer, text), row["gold"]))
        stored = part["correct"].tolist()
        flipped += sum(a != b for a, b in zip(redone, stored))
        worst = max(worst, abs(100 * (sum(redone) - sum(stored)) / len(stored)))
    lines.append(
        f"    re-decoding and rescoring flips {flipped} of {len(merged)} records; "
        f"largest per-cell accuracy move {worst:.1f} pp"
    )
    return lines


# ---------------------------------------------------------------------------
# Derived statistics quoted in prose
# ---------------------------------------------------------------------------
def prose_stats() -> list[str]:
    out: list[str] = []
    out.append(f"n_generations = {ANALYSIS_JSON['n_generations']}")

    # (1) paired accuracy deltas: how many of the 120 non-English contrasts move.
    non_en = [r for r in H1 if r["language"] != "en"]
    sig = [r for r in non_en if r["significant"] == "True"]
    out.append(f"non-English paired contrasts: {len(non_en)}, CI excludes 0: {len(sig)}")
    for r in sig:
        out.append(
            f"    {r['model']}/{r['task']}/{r['language']} {r['contrast']}: "
            f"{float(r['delta_pp']):+.1f} pp, McNemar p={float(r['mcnemar_p']):.4f}"
        )

    # (1b) McNemar agreement with the bootstrap criterion at alpha = 0.05.
    agree = [r for r in sig if float(r["mcnemar_p"]) <= 0.05]
    out.append(f"detected contrasts where McNemar also has p<=0.05: {len(agree)}/{len(sig)}")
    for r in sig:
        if float(r["mcnemar_p"]) > 0.05:
            out.append(
                f"    disagrees: {r['model']}/{r['task']}/{r['language']} "
                f"{r['contrast']} p={float(r['mcnemar_p']):.4f}"
            )

    # (1b-ii) multiplicity. The paper reports the eight detected contrasts
    # without correction, so Appendix "Statistical significance" states what a
    # correction over the whole family of 120 would do to them.
    family = sorted(float(r["mcnemar_p"]) for r in non_en)
    bonferroni = 0.05 / len(family)
    out.append(
        f"multiplicity: family={len(family)} contrasts, Bonferroni threshold="
        f"{bonferroni:.2e}, smallest McNemar p={family[0]:.4f}, "
        f"surviving Bonferroni={sum(1 for p in family if p <= bonferroni)}"
    )
    discoveries = 0
    for rank, p_value in enumerate(family, start=1):
        if p_value <= rank / len(family) * 0.05:
            discoveries = rank
    out.append(f"multiplicity: Benjamini-Hochberg at FDR 0.05 yields {discoveries} discoveries")
    largest = max(
        (r for r in non_en if r["model"] == "qwen3-8b"),
        key=lambda r: abs(float(r["delta_pp"])),
    )
    out.append(
        f"largest |delta| on qwen3-8b: {float(largest['delta_pp']):+.1f} pp "
        f"({largest['task']}/{largest['language']} {largest['contrast']}), "
        f"CI [{largest['delta_lo_pp']}, {largest['delta_hi_pp']}]"
    )

    # (1c) width of the paired bootstrap intervals, which is what the power
    # statements in the text and the limitations section quote.
    for model in ("qwen3-8b", "r1-llama-8b"):
        for task in ("mgsm", "gmmlu", None):
            widths = sorted(
                (float(r["delta_hi_pp"]) - float(r["delta_lo_pp"])) / 2
                for r in non_en
                if r["model"] == model and (task is None or r["task"] == task)
            )
            mid = widths[len(widths) // 2] if len(widths) % 2 else (
                widths[len(widths) // 2 - 1] + widths[len(widths) // 2]) / 2
            out.append(
                f"{model}/{task or 'both'}: CI half-width min={widths[0]:.2f} "
                f"median={mid:.2f} max={widths[-1]:.2f} pp (n={len(widths)})"
            )

    # (2) mean deltas by language group, per model.
    for model in ("qwen3-8b", "r1-llama-8b"):
        for contrast in ("En-CoT - Target-CoT", "Mixed-CoT - Target-CoT", "Free-CoT - Target-CoT"):
            for name, group in (("distant", DISTANT), ("close", CLOSE), ("all", LANGS)):
                values = [
                    float(r["delta_pp"]) for r in H1
                    if r["model"] == model and r["contrast"] == contrast and r["language"] in group
                ]
                out.append(
                    f"{model} {contrast} {name}: n={len(values)} "
                    f"mean={mean(values):+.2f} pp range=[{min(values):+.1f}, {max(values):+.1f}]"
                )

    # (3) Pareto dominance of the Target-CoT corner.
    dominated = ANALYSIS_JSON["verdicts"]["H3"]["target_cot_dominated"]
    for model in ("qwen3-8b", "r1-llama-8b"):
        for name, group in (("distant", DISTANT), ("close", CLOSE)):
            keys = [k for k in dominated if k.startswith(model) and k.rsplit("/", 1)[1] in group]
            hits = [k for k in keys if dominated[k]["dominated"]]
            out.append(f"{model} {name}: Target-CoT dominated in {len(hits)}/{len(keys)} cells")

    # (3b) which policy does the dominating, and how much cheaper the mixed
    # policy is than Target-CoT in the cells where the norm is dominated.
    dominators: dict[str, int] = {}
    savings: list[float] = []
    for model in ("qwen3-8b",):
        for task in ("mgsm", "gmmlu"):
            for lang in DISTANT:
                key = f"{model}/{task}/{lang}"
                for name in (dominated[key]["dominated_by"] or "").split(","):
                    name = name.strip()
                    if name:
                        dominators[name] = dominators.get(name, 0) + 1
                target_tokens = float(cell(model, task, lang, "target")["mean_completion_tokens"])
                mixed_tokens = float(cell(model, task, lang, "mixed")["mean_completion_tokens"])
                savings.append(100 * (1 - mixed_tokens / target_tokens))
    out.append(f"qwen3-8b distant cells: Target-CoT dominated by {dominators}")
    out.append(
        f"qwen3-8b distant cells: Mixed-CoT saves {min(savings):.1f}-{max(savings):.1f}% "
        f"of Target-CoT tokens (n={len(savings)})"
    )

    # (3c) Target/En-CoT token ratio by language group, both tasks.
    for name, group in (("distant", DISTANT), ("close", CLOSE)):
        ratios = [
            float(cell("qwen3-8b", task, lang, "target")["mean_completion_tokens"])
            / float(cell("qwen3-8b", task, lang, "en_cot")["mean_completion_tokens"])
            for task in ("mgsm", "gmmlu") for lang in group
        ]
        above = [r for r in ratios if r > 1]
        out.append(
            f"qwen3-8b {name}: Target/En-CoT token ratio {min(ratios):.2f}-{max(ratios):.2f} "
            f"({len(above)}/{len(ratios)} above 1)"
        )

    # (4) token cost ratio vs fertility, Qwen3 tokenizer only. English is dropped
    # for the same reason it is dropped from every other contrast: Target-CoT and
    # En-CoT are the same instruction there, so the ratio is prompt-wording noise.
    rows = [
        r for r in ANALYSIS_JSON["verdicts"]["H3"]["token_cost_ratio_target_over_en_cot"]
        if r["cell"].startswith("qwen3-8b") and not r["cell"].endswith("/en")
    ]
    rho, p = spearman([r["fertility_ratio"] for r in rows], [r["target_over_en_cot"] for r in rows])
    out.append(f"Qwen3 tokenizer: Spearman(fertility, Target/En-CoT tokens) rho={rho:.2f} p={p:.3f} n={len(rows)}")

    # (5) trace-level compliance under Target-CoT.
    for task in ("mgsm", "gmmlu"):
        parts = [
            f"{lang}={100 * float(cell('qwen3-8b', task, lang, 'target')['compliance_rate']):.0f}%"
            for lang in LANGS
        ]
        out.append(f"qwen3-8b/{task} Target-CoT trace compliance: " + ", ".join(parts))

    # (6) answer-level compliance, all policies.
    for cond in CONDS:
        parts = [
            f"{lang}={100 * mean([float(cell('qwen3-8b', t, lang, cond)['answer_compliance_rate']) for t in ('mgsm', 'gmmlu')]):.0f}%"
            for lang in LANGS
        ]
        out.append(f"qwen3-8b {LABELS[cond]} answer compliance: " + ", ".join(parts))

    # (7) H4 within-cell drift regression.
    h4 = ANALYSIS_JSON["verdicts"]["H4"]
    out.append(
        f"H4: {h4['n_cells_positive_significant']}/{h4['n_cells_fitted']} fitted cells with a "
        f"significantly positive English-share coefficient ({h4['cells_positive_significant']})"
    )
    not_fitted = [key for key, value in h4["per_cell"].items() if value.get("status") != "ok"]
    out.append(f"H4: {len(not_fitted)} cells not estimable ({not_fitted})")
    for key in h4["cells_positive_significant"]:
        fit = h4["per_cell"][key]
        out.append(
            f"    {key}: odds ratio per 10pp drift = {fit['odds_ratio_per_10pp_drift']:.1f}, "
            f"p = {fit['english_share_p']:.4f}"
        )
    out.append(
        f"H4 pooled fit: n={h4['pooled']['n']}, english_share coef="
        f"{h4['pooled']['params']['english_share']:+.2f} (confounded; not reported as evidence)"
    )

    # (8) H5 argmax counts. The released counts in analysis.json cover all 28
    # cells. The paper reports the twelve non-English Qwen3-8B ones: for `en`
    # three of the four instructions coincide, so the argmax is meaningless, and
    # every R1-Distill generation carries the text-decoding defect that makes
    # word-level retention unmeasurable on that model.
    out.append(f"H5 argmax counts, all 28 cells (released): {ANALYSIS_JSON['verdicts']['H5']['argmax_condition_counts']}")
    for metric in ("entity_retention", "anchor_retention", "number_retention"):
        for label, models in (("qwen3-8b only, 12 cells", ("qwen3-8b",)),
                              ("both models, 24 cells", ("qwen3-8b", "r1-llama-8b"))):
            wins: dict[str, int] = {}
            for model in models:
                for task in ("mgsm", "gmmlu"):
                    for lang in LANGS:
                        scores = {
                            row["condition"]: float(row["mean"]) for row in H5
                            if (row["model"], row["task"], row["language"], row["metric"])
                            == (model, task, lang, metric)
                        }
                        if scores:
                            best = max(scores, key=lambda cond: scores[cond])
                            wins[best] = wins.get(best, 0) + 1
            out.append(f"H5 argmax, non-English, {metric} [{label}]: {wins}")

    # (9) truncation under Target-CoT, against En-CoT in the same cell.
    for model in ("qwen3-8b", "r1-llama-8b"):
        worst = max(
            (
                (float(cell(model, task, lang, "target")["truncation_rate"]), task, lang)
                for task in ("mgsm", "gmmlu") for lang in LANGS
            ),
        )
        same_cell_en = float(cell(model, worst[1], worst[2], "en_cot")["truncation_rate"])
        out.append(
            f"{model}: worst Target-CoT truncation {100 * worst[0]:.1f}% ({worst[1]}/{worst[2]}), "
            f"En-CoT in the same cell {100 * same_cell_en:.1f}%"
        )

    # (9b) how truncated generations are scored, and what forcing them to count
    # as incorrect would do. Needs the per-generation parquet, so it is optional:
    # the rest of this script is stdlib-only and runs off the CSV tables alone.
    out.extend(_truncation_sensitivity())

    # (9c) the R1-Distill text-decoding defect and its bounded consequences.
    out.extend(_decode_defect())

    # (10) tokenizer fertility relative to English, Qwen3 tokenizer.
    for lang in LANGS:
        ratios = [fertility("qwen3-8b", task, lang) for task in ("mgsm", "gmmlu")]
        out.append(f"qwen3 fertility vs English, {lang}: {min(ratios):.2f}-{max(ratios):.2f}x")

    # (11) R1-Distill fertility, the excluded instrumentation artifact.
    bad = [
        (row["task"], row["language"], float(row["tokens_per_word"]), float(row["fertility_ratio"]))
        for row in FERT
        if row["model"] == "r1-llama-8b" and float(row["fertility_ratio"]) < 1.0
    ]
    out.append(f"r1-llama-8b cells with fertility ratio < 1 (excluded as an artifact): {len(bad)}")
    for task, lang, tpw, ratio in bad:
        out.append(f"    {task}/{lang}: {tpw:.2f} tokens/word, {ratio:.2f}x English")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", action="store_true", help="print only the LaTeX table bodies")
    args = parser.parse_args()

    bodies = {
        "Table 1 (main: accuracy and cost, qwen3-8b)": table_main(),
        "Table 2 (main: what each policy delivers, qwen3-8b)": table_delivery(),
        "Table A (appendix: full accuracy matrix)": table_accuracy_matrix(),
        "Table B (appendix: tokens / truncation)": table_tokens_truncation(),
        "Table C (appendix: retention, qwen3-8b)": table_retention(),
        "Table D (appendix: fertility, Qwen3 tokenizer)": table_fertility(),
    }
    for title, body in bodies.items():
        print(f"\n%% ==== {title} ====")
        print(body)

    if not args.tables:
        print("\n%% ==== derived statistics quoted in the text ====")
        for line in prose_stats():
            print(line)


if __name__ == "__main__":
    main()
