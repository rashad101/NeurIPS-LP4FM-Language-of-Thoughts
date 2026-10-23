"""Analysis: the cell table, the landscape, and the five hypotheses.

Outputs (all under ``results/``):

``tables/cells.csv``        one row per (model, task, language, condition) with
                            bootstrap CIs on accuracy, cost, compliance, retention
``tables/accuracy_matrix.csv``   language × condition accuracy, per model/task
``tables/h1_ordering.csv``  H1 pairwise paired contrasts with McNemar p-values
``tables/h3_pareto.csv``    per-language Pareto status of each policy
``tables/h4_drift.json``    logistic regressions of correctness on English drift
``tables/h5_retention.csv`` retention by condition with CIs
``analysis.json``           everything above, machine-readable, plus verdicts
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import CONDITION_LABELS, CONDITION_ORDER, Config
from .env import RESULTS_DIR, TABLES_DIR
from .logging_utils import LoggedProgress, get_logger, log_table
from .stats_utils import (
    bootstrap_mean,
    bootstrap_paired_diff,
    dominates,
    logistic_drift_regression,
    mcnemar_exact,
    pareto_front,
    spearman,
    stable_seed,
)

log = get_logger("analyze")

ANALYSIS_JSON = RESULTS_DIR / "analysis.json"

#: Per-cell mapping to the prior finding each cell reproduces (spec §7, figure 2).
LITERATURE_MAP = {
    "target": "XReasoning (2505.22888): controlling the thinking language costs accuracy; "
              "Long CoT Across Languages (2508.14828) Target-CoT condition",
    "en_cot": "Long CoT Across Languages (2508.14828) En-CoT condition; the operating point "
              "consistency training moves *away* from",
    "free": "Language Matters (2505.17407): models default to hub languages when unconstrained",
    "mixed": "KO-REAson / Language-Mixed CoT: trained mixed CoT beats both monolingual extremes; "
             "Impact of Language Mixing (2507.15849): mixing is strategic",
}


@dataclass
class AnalysisResult:
    cells: Any
    accuracy_matrix: Any
    h1: Any
    h2: dict[str, Any]
    h3: Any
    h4: dict[str, Any]
    h5: Any
    verdicts: dict[str, Any]
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# Cell-level aggregation
# ---------------------------------------------------------------------------


def cell_table(frame: Any, config: Config) -> Any:
    """One row per (model, task, language, condition) with bootstrap CIs."""
    import pandas as pd

    scoring = config.scoring
    groups = list(frame.groupby(["model", "task", "language", "condition"], dropna=False))
    rows: list[dict[str, Any]] = []

    with LoggedProgress(len(groups), "cell aggregates", unit="cell", logger_name="analyze") as bar:
        for (model, task, language, condition), part in groups:
            seed = stable_seed(model, task, language, condition)
            accuracy = bootstrap_mean(part["correct"].astype(float).tolist(),
                                      scoring.bootstrap_iters, scoring.bootstrap_alpha, seed)
            tokens = bootstrap_mean(part["n_completion_tokens"].astype(float).tolist(),
                                    min(scoring.bootstrap_iters, 2000),
                                    scoring.bootstrap_alpha, seed + 1)
            retention_part = part[part["in_faithfulness_subsample"].fillna(False).astype(bool)]
            anchor = bootstrap_mean(
                retention_part["anchor_retention"].dropna().astype(float).tolist(),
                min(scoring.bootstrap_iters, 2000), scoring.bootstrap_alpha, seed + 2,
            )
            entity = bootstrap_mean(
                retention_part["entity_retention"].dropna().astype(float).tolist(),
                min(scoring.bootstrap_iters, 2000), scoring.bootstrap_alpha, seed + 3,
            )
            number = bootstrap_mean(
                retention_part["number_retention"].dropna().astype(float).tolist(),
                min(scoring.bootstrap_iters, 2000), scoring.bootstrap_alpha, seed + 4,
            )
            lang_spec = config.language(language)
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "language": language,
                    "language_name": lang_spec.name,
                    "script": lang_spec.script,
                    "family": lang_spec.family,
                    "typ_distance": lang_spec.typ_distance,
                    "resource_proxy_gb": lang_spec.resource_proxy_gb,
                    "resource_tier": lang_spec.resource_tier,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS.get(condition, condition),
                    "degenerate_cell": bool(part["degenerate_cell"].iloc[0]),
                    "n": int(len(part)),
                    "accuracy": accuracy.mean,
                    "accuracy_lo": accuracy.lo,
                    "accuracy_hi": accuracy.hi,
                    "mean_completion_tokens": tokens.mean,
                    "tokens_lo": tokens.lo,
                    "tokens_hi": tokens.hi,
                    "median_completion_tokens": float(part["n_completion_tokens"].median()),
                    "truncation_rate": float(part["truncated"].mean()),
                    "extraction_failure_rate": float(part["extraction_failed"].mean()),
                    "well_formed_rate": float(part["well_formed_trace"].mean()),
                    "trace_target_share": float(part["trace_target_share"].mean()),
                    "trace_english_share": float(part["trace_english_share"].mean()),
                    "compliance_rate": float(part["trace_compliant"].mean()),
                    "target_anchor_rate": float(part["trace_has_target_anchor"].mean()),
                    "answer_compliance_rate": float(part["answer_compliant"].mean()),
                    "anchor_retention": anchor.mean,
                    "anchor_retention_lo": anchor.lo,
                    "anchor_retention_hi": anchor.hi,
                    "entity_retention": entity.mean,
                    "entity_retention_lo": entity.lo,
                    "entity_retention_hi": entity.hi,
                    "number_retention": number.mean,
                    "number_retention_lo": number.lo,
                    "number_retention_hi": number.hi,
                    "n_retention_items": int(len(retention_part)),
                    "reproduces": LITERATURE_MAP.get(condition, ""),
                }
            )
            bar.update(1)

    table = pd.DataFrame(rows)
    table["condition"] = pd.Categorical(table["condition"], CONDITION_ORDER, ordered=True)
    return table.sort_values(["model", "task", "typ_distance", "condition"]).reset_index(drop=True)


def accuracy_matrix(cells: Any) -> Any:
    """Language × condition accuracy with CI strings, per model and task."""
    import pandas as pd

    records = []
    for (model, task), part in cells.groupby(["model", "task"], observed=True):
        for _, row in part.iterrows():
            records.append(
                {
                    "model": model,
                    "task": task,
                    "language": row["language"],
                    "condition": row["condition_label"],
                    "accuracy": row["accuracy"],
                    "ci": f"[{row['accuracy_lo']:.3f}, {row['accuracy_hi']:.3f}]",
                    "display": f"{row['accuracy']:.3f} [{row['accuracy_lo']:.3f}, {row['accuracy_hi']:.3f}]",
                    "n": row["n"],
                    "degenerate": row["degenerate_cell"],
                    "reproduces": row["reproduces"],
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# H1 / H2 — condition ordering
# ---------------------------------------------------------------------------


def _paired(frame: Any, left: str, right: str) -> tuple[list[float], list[float], int]:
    """Align two conditions on item_id (+ sample_index) for paired tests."""
    keys = ["item_id", "sample_index"]
    a = frame[frame["condition"] == left].set_index(keys)["correct"].astype(float)
    b = frame[frame["condition"] == right].set_index(keys)["correct"].astype(float)
    shared = a.index.intersection(b.index)
    return a.loc[shared].tolist(), b.loc[shared].tolist(), len(shared)


def h1_ordering(frame: Any, config: Config) -> Any:
    """Paired contrasts between conditions, per model/task/language."""
    import pandas as pd

    contrasts = [("en_cot", "target"), ("mixed", "target"), ("en_cot", "mixed"),
                 ("free", "target"), ("free", "en_cot")]
    rows: list[dict[str, Any]] = []
    groups = list(frame.groupby(["model", "task", "language"], dropna=False))

    with LoggedProgress(len(groups) * len(contrasts), "H1 contrasts", unit="test",
                        logger_name="analyze") as bar:
        for (model, task, language), part in groups:
            lang_spec = config.language(language)
            for left, right in contrasts:
                a, b, n_pairs = _paired(part, left, right)
                if n_pairs == 0:
                    bar.update(1)
                    continue
                diff = bootstrap_paired_diff(
                    a, b, config.scoring.bootstrap_iters, config.scoring.bootstrap_alpha,
                    seed=stable_seed(model, task, language, left, right),
                )
                b01, b10, p = mcnemar_exact([bool(x) for x in a], [bool(x) for x in b])
                rows.append(
                    {
                        "model": model,
                        "task": task,
                        "language": language,
                        "typ_distance": lang_spec.typ_distance,
                        "contrast": f"{CONDITION_LABELS[left]} - {CONDITION_LABELS[right]}",
                        "left": left,
                        "right": right,
                        "n_pairs": n_pairs,
                        "delta_pp": 100 * diff.mean,
                        "delta_lo_pp": 100 * diff.lo,
                        "delta_hi_pp": 100 * diff.hi,
                        "significant": bool(diff.lo > 0 or diff.hi < 0),
                        "mcnemar_p": p,
                        "b_left_only": b10,
                        "b_right_only": b01,
                        "degenerate": bool(
                            part[part["condition"] == left]["degenerate_cell"].any()
                            or part[part["condition"] == right]["degenerate_cell"].any()
                        ),
                    }
                )
                bar.update(1)
    return pd.DataFrame(rows)


def h1_verdict(h1: Any, config: Config) -> dict[str, Any]:
    """H1: En-CoT >= Mixed > Target for distant languages; gap shrinks for de/es."""
    if h1.empty:
        return {"status": "no_data"}
    distant = [c for c in config.default_languages
               if config.language(c).typ_distance >= 0.5 and c != "en"]
    close = [c for c in config.default_languages
             if 0 < config.language(c).typ_distance < 0.5]

    subset = h1[(h1["left"] == "en_cot") & (h1["right"] == "target") & (~h1["degenerate"])]
    per_language = (
        subset.groupby("language")["delta_pp"].mean().to_dict() if not subset.empty else {}
    )
    distant_gap = [v for k, v in per_language.items() if k in distant]
    close_gap = [v for k, v in per_language.items() if k in close]
    mean_distant = sum(distant_gap) / len(distant_gap) if distant_gap else float("nan")
    mean_close = sum(close_gap) / len(close_gap) if close_gap else float("nan")

    rho, p = spearman(
        [config.language(k).typ_distance for k in per_language],
        list(per_language.values()),
    )
    return {
        "status": "ok",
        "en_cot_minus_target_pp": {k: round(v, 2) for k, v in per_language.items()},
        "mean_gap_distant_pp": mean_distant,
        "mean_gap_close_pp": mean_close,
        "gap_larger_for_distant": bool(mean_distant > mean_close),
        "spearman_gap_vs_typ_distance": {"rho": rho, "p": p},
        "n_significant_positive": int(
            ((subset["delta_pp"] > 0) & subset["significant"]).sum()
        ),
        "n_significant_negative": int(
            ((subset["delta_pp"] < 0) & subset["significant"]).sum()
        ),
        "interpretation": (
            "H1 predicts En-CoT >= Mixed > Target for hi/bn/te/sw with a smaller gap for "
            "de/es. Positive en_cot-target deltas and a positive Spearman rho against "
            "typological distance support it."
        ),
    }


def h2_verdict(cells: Any) -> dict[str, Any]:
    """H2: Mixed-CoT recovers most of En-CoT's accuracy while keeping target anchors."""
    if cells.empty:
        return {"status": "no_data"}
    rows = []
    for (model, task, language), part in cells.groupby(["model", "task", "language"], observed=True):
        by_condition = part.set_index("condition")["accuracy"].to_dict()
        anchors = part.set_index("condition")["target_anchor_rate"].to_dict()
        target = by_condition.get("target")
        en_cot = by_condition.get("en_cot")
        mixed = by_condition.get("mixed")
        if None in (target, en_cot, mixed):
            continue
        span = en_cot - target
        recovery = (mixed - target) / span if abs(span) > 1e-9 else float("nan")
        rows.append(
            {
                "model": model,
                "task": task,
                "language": language,
                "target_acc": target,
                "mixed_acc": mixed,
                "en_cot_acc": en_cot,
                "recovery_fraction": recovery,
                "mixed_target_anchor_rate": anchors.get("mixed"),
                "en_cot_target_anchor_rate": anchors.get("en_cot"),
            }
        )
    if not rows:
        return {"status": "insufficient_conditions"}
    recoveries = [r["recovery_fraction"] for r in rows
                  if r["recovery_fraction"] == r["recovery_fraction"]]
    return {
        "status": "ok",
        "per_cell": rows,
        "mean_recovery_fraction": sum(recoveries) / len(recoveries) if recoveries else float("nan"),
        "interpretation": (
            "recovery_fraction = (Mixed - Target) / (En-CoT - Target). Values near 1 mean "
            "Mixed-CoT recovers En-CoT accuracy; compare mixed_target_anchor_rate against "
            "en_cot_target_anchor_rate to show the anchors are retained (H2)."
        ),
    }


# ---------------------------------------------------------------------------
# H3 — the Pareto landscape
# ---------------------------------------------------------------------------


def h3_pareto(cells: Any, fertility: Any | None = None) -> Any:
    """Per (model, task, language): which policies are Pareto-optimal on acc × cost."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for (model, task, language), part in cells.groupby(["model", "task", "language"], observed=True):
        part = part.reset_index(drop=True)
        points = list(zip(part["accuracy"].tolist(),
                          part["mean_completion_tokens"].tolist(), strict=True))
        front = set(pareto_front(points))
        for index, row in part.iterrows():
            point = (row["accuracy"], row["mean_completion_tokens"])
            dominators = [
                part.loc[j, "condition_label"]
                for j in range(len(part))
                if j != index and dominates(points[j], point)
            ]
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "language": language,
                    "typ_distance": row["typ_distance"],
                    "condition": row["condition"],
                    "condition_label": row["condition_label"],
                    "accuracy": row["accuracy"],
                    "mean_completion_tokens": row["mean_completion_tokens"],
                    "on_pareto_front": index in front,
                    "dominated_by": ", ".join(dominators),
                    "n_dominators": len(dominators),
                }
            )
    table = pd.DataFrame(rows)
    if fertility is not None and not getattr(fertility, "empty", True):
        ratios = (
            fertility.groupby(["model", "task", "language"])["fertility_ratio"]
            .mean()
            .reset_index()
        )
        table = table.merge(ratios, on=["model", "task", "language"], how="left")
    return table


def h3_verdict(pareto: Any, config: Config) -> dict[str, Any]:
    """H3: Target-CoT is Pareto-dominated for high-fertility (Indic) languages."""
    if pareto.empty:
        return {"status": "no_data"}
    target = pareto[pareto["condition"] == "target"]
    dominated: dict[str, Any] = {}
    for (model, task, language), part in target.groupby(["model", "task", "language"], observed=True):
        row = part.iloc[0]
        dominated[f"{model}/{task}/{language}"] = {
            "dominated": bool(row["n_dominators"] > 0),
            "dominated_by": row["dominated_by"],
            "accuracy": float(row["accuracy"]),
            "mean_completion_tokens": float(row["mean_completion_tokens"]),
        }
    indic = {"hi", "bn", "te"}
    indic_cells = {k: v for k, v in dominated.items() if k.rsplit("/", 1)[-1] in indic}
    european_cells = {k: v for k, v in dominated.items() if k.rsplit("/", 1)[-1] in {"de", "es"}}

    cost_rows = []
    for (model, task, language), part in pareto.groupby(["model", "task", "language"], observed=True):
        by_condition = part.set_index("condition")["mean_completion_tokens"].to_dict()
        if "target" in by_condition and "en_cot" in by_condition and by_condition["en_cot"]:
            cost_rows.append(
                {
                    "cell": f"{model}/{task}/{language}",
                    "language": language,
                    "target_over_en_cot": by_condition["target"] / by_condition["en_cot"],
                    "fertility_ratio": float(part["fertility_ratio"].mean())
                    if "fertility_ratio" in part else None,
                }
            )
    rho, p = spearman(
        [r["fertility_ratio"] for r in cost_rows if r["fertility_ratio"] is not None],
        [r["target_over_en_cot"] for r in cost_rows if r["fertility_ratio"] is not None],
    )
    return {
        "status": "ok",
        "target_cot_dominated": dominated,
        "n_indic_cells_dominated": sum(1 for v in indic_cells.values() if v["dominated"]),
        "n_indic_cells": len(indic_cells),
        "n_european_cells_dominated": sum(1 for v in european_cells.values() if v["dominated"]),
        "n_european_cells": len(european_cells),
        "token_cost_ratio_target_over_en_cot": cost_rows,
        "spearman_cost_ratio_vs_fertility": {"rho": rho, "p": p},
        "interpretation": (
            "H3 holds when Target-CoT is Pareto-dominated (accuracy <= and tokens >= some "
            "other policy) in the Indic cells but not, or less, in the European ones."
        ),
    }


# ---------------------------------------------------------------------------
# H4 — drift predicts accuracy inside Target-CoT
# ---------------------------------------------------------------------------


def h4_drift(frame: Any, config: Config) -> dict[str, Any]:
    target = frame[frame["condition"] == "target"]
    out: dict[str, Any] = {"per_cell": {}, "pooled": {}}
    if target.empty:
        return {"status": "no_target_cot_data"}

    groups = list(target.groupby(["model", "task", "language"], dropna=False))
    with LoggedProgress(len(groups) + 1, "H4 regressions", unit="fit", logger_name="analyze") as bar:
        for (model, task, language), part in groups:
            if config.language(language).is_anchor:
                bar.update(1)
                continue  # drift is undefined when the target language *is* English
            result = logistic_drift_regression(
                part["trace_english_share"].tolist(),
                part["correct"].astype(bool).tolist(),
                extra={"n_completion_tokens": part["n_completion_tokens"].astype(float).tolist()},
            )
            result["mean_english_share"] = float(part["trace_english_share"].mean())
            result["accuracy"] = float(part["correct"].mean())
            # Descriptive split, robust to a failed fit.
            median = part["trace_english_share"].median()
            high = part[part["trace_english_share"] > median]["correct"].astype(float)
            low = part[part["trace_english_share"] <= median]["correct"].astype(float)
            result["accuracy_high_drift"] = float(high.mean()) if len(high) else None
            result["accuracy_low_drift"] = float(low.mean()) if len(low) else None
            result["drift_split_delta_pp"] = (
                100 * (high.mean() - low.mean()) if len(high) and len(low) else None
            )
            out["per_cell"][f"{model}/{task}/{language}"] = result
            bar.update(1)

        pooled = target[~target["language"].isin(
            [code for code in config.languages if config.language(code).is_anchor]
        )]
        if not pooled.empty:
            out["pooled"] = logistic_drift_regression(
                pooled["trace_english_share"].tolist(),
                pooled["correct"].astype(bool).tolist(),
                extra={"n_completion_tokens": pooled["n_completion_tokens"].astype(float).tolist()},
            )
        bar.update(1)

    positive = [
        key for key, value in out["per_cell"].items()
        if value.get("status") == "ok" and value["params"].get("english_share", 0) > 0
        and value["p_values"].get("english_share", 1) < 0.05
    ]
    out["status"] = "ok"
    out["n_cells_positive_significant"] = len(positive)
    out["cells_positive_significant"] = positive
    out["n_cells_fitted"] = sum(
        1 for value in out["per_cell"].values() if value.get("status") == "ok"
    )
    out["interpretation"] = (
        "H4 predicts a positive coefficient on english_share: within Target-CoT, traces "
        "that drift to English are more often correct. This replicates 'mixing is "
        "strategic' (2507.15849) at the instruction level rather than by constrained decoding."
    )
    return out


# ---------------------------------------------------------------------------
# H5 — retention
# ---------------------------------------------------------------------------


def h5_retention(frame: Any, config: Config) -> Any:
    import pandas as pd

    subset = frame[frame["in_faithfulness_subsample"].fillna(False).astype(bool)]
    if subset.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (model, task, language, condition), part in subset.groupby(
        ["model", "task", "language", "condition"], dropna=False
    ):
        seed = stable_seed(model, task, language, condition, "h5")
        for metric in ("anchor_retention", "entity_retention", "number_retention"):
            values = part[metric].dropna().astype(float).tolist()
            estimate = bootstrap_mean(values, min(config.scoring.bootstrap_iters, 4000),
                                      config.scoring.bootstrap_alpha, seed)
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "language": language,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS.get(condition, condition),
                    "metric": metric,
                    "mean": estimate.mean,
                    "lo": estimate.lo,
                    "hi": estimate.hi,
                    "n": estimate.n,
                }
            )
    return pd.DataFrame(rows)


def h5_verdict(h5: Any) -> dict[str, Any]:
    if h5 is None or h5.empty:
        return {"status": "no_data"}
    wins: dict[str, Any] = {}
    for metric in h5["metric"].unique():
        part = h5[h5["metric"] == metric]
        counts: dict[str, int] = {}
        for _key, group in part.groupby(["model", "task", "language"]):
            group = group.dropna(subset=["mean"])
            if group.empty:
                continue
            best = group.loc[group["mean"].idxmax(), "condition"]
            counts[str(best)] = counts.get(str(best), 0) + 1
        wins[str(metric)] = counts
    return {
        "status": "ok",
        "argmax_condition_counts": wins,
        "interpretation": (
            "H5 predicts Mixed-CoT has the highest retention. Counts show, per metric, how "
            "many (model, task, language) cells each policy wins."
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    frame: Any,
    config: Config,
    fertility_table: Any | None = None,
    output_dir: Path | None = None,
) -> AnalysisResult:
    tables_dir = output_dir or TABLES_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)

    log.info("analysing %d generations over %d cells", len(frame), frame.groupby(
        ["model", "task", "language", "condition"], dropna=False).ngroups)

    cells = cell_table(frame, config)
    matrix = accuracy_matrix(cells)
    h1 = h1_ordering(frame, config)
    pareto = h3_pareto(cells, fertility_table)
    h4 = h4_drift(frame, config)
    h5 = h5_retention(frame, config)

    verdicts = {
        "H1": h1_verdict(h1, config),
        "H2": h2_verdict(cells),
        "H3": h3_verdict(pareto, config),
        "H4": {k: v for k, v in h4.items() if k != "per_cell"} | {
            "per_cell": {
                key: {
                    k: v for k, v in value.items()
                    if k in {"status", "n", "accuracy", "mean_english_share",
                             "drift_split_delta_pp", "pseudo_r2",
                             "odds_ratio_per_10pp_drift"}
                }
                | {"english_share_coef": value.get("params", {}).get("english_share")}
                | {"english_share_p": value.get("p_values", {}).get("english_share")}
                for key, value in h4.get("per_cell", {}).items()
            }
        },
        "H5": h5_verdict(h5),
    }

    cells.to_csv(tables_dir / "cells.csv", index=False)
    matrix.to_csv(tables_dir / "accuracy_matrix.csv", index=False)
    h1.to_csv(tables_dir / "h1_ordering.csv", index=False)
    pareto.to_csv(tables_dir / "h3_pareto.csv", index=False)
    (tables_dir / "h4_drift.json").write_text(
        json.dumps(h4, indent=2, default=str), encoding="utf-8"
    )
    if not h5.empty:
        h5.to_csv(tables_dir / "h5_retention.csv", index=False)

    payload = {
        "n_generations": int(len(frame)),
        "runs": sorted(frame["run_id"].dropna().unique().tolist()),
        "models": sorted(frame["model"].unique().tolist()),
        "tasks": sorted(frame["task"].unique().tolist()),
        "languages": sorted(frame["language"].unique().tolist()),
        "conditions": sorted(frame["condition"].unique().tolist()),
        "verdicts": verdicts,
        "literature_map": LITERATURE_MAP,
        "caveats": [
            "Prompted conditions, not trained policies. The training-side price of "
            "consistency (5-10 pp, Cross-lingual Collapse 2506.05850) is cited, not measured here.",
            "Retention is verbatim-anchor retention: a lower bound on faithfulness.",
            "typ_distance is an author-assigned ordinal proxy, not a measured lang2vec distance.",
            "Truncation rates differ by language because target-script traces are longer; "
            "always report truncation_rate alongside accuracy.",
            "For language=en, Target-CoT/En-CoT/Mixed-CoT are definitionally the same "
            "instruction; those cells are flagged degenerate and measure prompt-wording noise.",
        ],
    }
    ANALYSIS_JSON.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("analysis -> %s", ANALYSIS_JSON)

    _log_headlines(cells, verdicts)
    return AnalysisResult(cells, matrix, h1, verdicts["H2"], pareto, h4, h5, verdicts, payload)


def _log_headlines(cells: Any, verdicts: dict[str, Any]) -> None:
    for (model, task), part in cells.groupby(["model", "task"], observed=True):
        log.info("=== %s / %s ===", model, task)
        log_table(
            [
                [
                    row["language"], row["condition_label"],
                    f"{row['accuracy']:.3f}",
                    f"[{row['accuracy_lo']:.3f},{row['accuracy_hi']:.3f}]",
                    f"{row['mean_completion_tokens']:.0f}",
                    f"{row['compliance_rate']:.2f}",
                    f"{row['trace_english_share']:.2f}",
                    f"{row['anchor_retention']:.2f}" if row["anchor_retention"] == row["anchor_retention"] else "-",
                    f"{row['truncation_rate']:.1%}",
                    "deg" if row["degenerate_cell"] else "",
                ]
                for _, row in part.iterrows()
            ],
            ["lang", "policy", "acc", "95% CI", "tok", "compl", "en%", "retain", "trunc", ""],
            logger_name="analyze",
        )
    for name in ("H1", "H2", "H3", "H4", "H5"):
        verdict = verdicts.get(name, {})
        log.info("%s: status=%s", name, verdict.get("status"))
    h1 = verdicts.get("H1", {})
    if h1.get("status") == "ok":
        log.info(
            "H1: mean En-CoT - Target gap = %.2f pp (distant) vs %.2f pp (close); rho vs typ_distance = %.2f",
            h1.get("mean_gap_distant_pp", float("nan")),
            h1.get("mean_gap_close_pp", float("nan")),
            h1.get("spearman_gap_vs_typ_distance", {}).get("rho", float("nan")),
        )
    h3 = verdicts.get("H3", {})
    if h3.get("status") == "ok":
        log.info(
            "H3: Target-CoT Pareto-dominated in %d/%d Indic cells vs %d/%d European cells",
            h3.get("n_indic_cells_dominated", 0), h3.get("n_indic_cells", 0),
            h3.get("n_european_cells_dominated", 0), h3.get("n_european_cells", 0),
        )
    h4 = verdicts.get("H4", {})
    if h4.get("status") == "ok":
        log.info(
            "H4: %d/%d fitted cells show a significantly positive English-drift coefficient",
            h4.get("n_cells_positive_significant", 0), h4.get("n_cells_fitted", 0),
        )


def load_analysis() -> dict[str, Any]:
    if not ANALYSIS_JSON.exists():
        raise FileNotFoundError(f"{ANALYSIS_JSON} not found — run `ltd analyze` first")
    return json.loads(ANALYSIS_JSON.read_text(encoding="utf-8"))
