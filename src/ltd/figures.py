"""The four figures of spec §7.

1. ``fig1_landscape``   accuracy × token-cost per language, four policies as
   operating points, Pareto front drawn, Target-CoT (the corner that
   consistency training implies) called out when it is dominated. *The* figure.
2. ``fig2_accuracy_matrix``  language × condition accuracy heatmap with CIs,
   presented as "our setup reproduces the literature".
3. ``fig3_drift_and_retention``  H4 drift–accuracy relation and H5 retention bars.
4. ``fig4_gap_vs_distance``  the En-CoT − Target-CoT gap against typological
   distance and against measured tokenizer fertility.

Figures are written as both PDF (for the paper) and PNG (for quick looks).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .config import CONDITION_LABELS, CONDITION_ORDER, Config
from .env import FIGURES_DIR
from .logging_utils import LoggedProgress, get_logger
from .stats_utils import pareto_front, spearman

log = get_logger("figures")

# Colour-blind-safe (Okabe-Ito), one colour per policy, used everywhere.
POLICY_COLORS = {
    "target": "#D55E00",   # vermillion — the consistency corner
    "en_cot": "#0072B2",   # blue
    "free": "#009E73",     # green
    "mixed": "#CC79A7",    # purple
}
POLICY_MARKERS = {"target": "s", "en_cot": "o", "free": "^", "mixed": "D"}

# Display names for figure text. The run keys are lowercase slugs, but the paper
# calls these things Qwen3-8B, MGSM and Global MMLU, so the figures must too.
MODEL_DISPLAY = {"qwen3-8b": "Qwen3-8B", "r1-llama-8b": "R1-Distill-Llama-8B"}
TASK_DISPLAY = {"mgsm": "MGSM", "gmmlu": "Global MMLU"}


def display_model(key: str) -> str:
    return MODEL_DISPLAY.get(key, key)


def display_task(key: str) -> str:
    return TASK_DISPLAY.get(key, key)

#: Same Okabe-Ito family, used where the series are tasks rather than policies
#: (fig4). Colour and marker shape both carry the task, so the series stay
#: separable in greyscale print as well as in colour.
SERIES_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
SERIES_MARKERS = ["o", "^", "s", "D", "v"]

#: The model the paper's per-model tables are built on. Figures 3 and 4 are drawn
#: on it alone so that a reader comparing a figure with a table sees the same
#: numbers, and because the second model's tokenizer fertility measurement is an
#: instrumentation artifact excluded from every fertility analysis.
PRIMARY_MODEL = "qwen3-8b"


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save(fig, name: str, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"{name}.{suffix}"
        fig.savefig(path)
        paths.append(path)
    log.info("figure -> %s", output_dir / f"{name}.{{pdf,png}}")
    return paths


def _grid(plt, n: int, per_row: int = 4, size: tuple[float, float] = (3.0, 2.7)):
    rows = max(1, (n + per_row - 1) // per_row)
    cols = min(per_row, max(n, 1))
    fig, axes = plt.subplots(rows, cols, figsize=(size[0] * cols, size[1] * rows), squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat


# ---------------------------------------------------------------------------
# Figure 1 — the landscape
# ---------------------------------------------------------------------------


def _annotate_corner(ax: Any, point: tuple[float, float],
                     others: list[tuple[float, float]]) -> None:
    """Label ``point`` from whichever axes corner sits furthest from the data.

    A fixed offset from the marker lands on top of another policy's marker or
    error bar in most panels, so we place the text in the emptiest corner and
    run a leader line back to the point it describes.
    """
    from math import hypot

    ax.autoscale_view()
    to_axes = ax.transLimits.transform
    seen = [to_axes(p) for p in others]
    corners = {
        (0.03, 0.04): ("left", "bottom"),
        (0.97, 0.04): ("right", "bottom"),
        (0.03, 0.96): ("left", "top"),
        (0.97, 0.96): ("right", "top"),
    }
    spot = max(corners, key=lambda c: min(hypot(c[0] - x, c[1] - y) for x, y in seen))
    ha, va = corners[spot]
    ax.annotate(
        "consistency corner\n(dominated)",
        xy=point, xycoords="data",
        xytext=spot, textcoords="axes fraction",
        ha=ha, va=va, fontsize=6.5, color=POLICY_COLORS["target"],
        arrowprops=dict(arrowstyle="->", color=POLICY_COLORS["target"],
                        lw=0.7, shrinkB=7, shrinkA=2),
    )


def fig1_landscape(cells: Any, config: Config, output_dir: Path, task: str | None = None) -> list[Path]:
    plt = _setup_matplotlib()
    paths: list[Path] = []

    tasks = [task] if task else sorted(cells["task"].unique())
    for task_name in tasks:
        for model in sorted(cells["model"].unique()):
            part = cells[(cells["task"] == task_name) & (cells["model"] == model)]
            if part.empty:
                continue
            languages = sorted(part["language"].unique(),
                               key=lambda code: config.language(code).typ_distance)
            fig, axes = _grid(plt, len(languages), per_row=4, size=(3.1, 2.9))

            for ax, lang_code in zip(axes, languages):
                cell = part[part["language"] == lang_code].reset_index(drop=True)
                points = list(zip(cell["accuracy"], cell["mean_completion_tokens"]))
                front = set(pareto_front(points))
                costs = [p[1] for p in points]
                cost_mid = (min(costs) + max(costs)) / 2 if costs else 0.0
                is_anchor = config.language(lang_code).is_anchor

                # Pareto staircase.
                ordered = sorted(front, key=lambda i: points[i][1])
                if len(ordered) > 1:
                    ax.plot(
                        [points[i][1] for i in ordered],
                        [points[i][0] for i in ordered],
                        color="0.55", lw=1.0, ls="--", zorder=1,
                        label="Pareto front",
                    )

                target_xy = None
                drawn: list[tuple[float, float]] = []
                for index, row in cell.iterrows():
                    condition = str(row["condition"])
                    dominated = index not in front
                    drawn.append((row["mean_completion_tokens"], row["accuracy"]))
                    ax.errorbar(
                        row["mean_completion_tokens"], row["accuracy"],
                        yerr=[[row["accuracy"] - row["accuracy_lo"]],
                              [row["accuracy_hi"] - row["accuracy"]]],
                        xerr=[[max(row["mean_completion_tokens"] - row["tokens_lo"], 0)],
                              [max(row["tokens_hi"] - row["mean_completion_tokens"], 0)]],
                        fmt=POLICY_MARKERS.get(condition, "o"),
                        color=POLICY_COLORS.get(condition, "0.3"),
                        markersize=7,
                        markerfacecolor=POLICY_COLORS.get(condition, "0.3") if not dominated else "white",
                        markeredgewidth=1.4,
                        elinewidth=0.9, capsize=2, zorder=3,
                        label=CONDITION_LABELS.get(condition, condition),
                    )
                    # Call out the consistency corner only where it means something:
                    # for English, Target-CoT *is* English thinking, so labelling it
                    # "the consistency corner" would be misleading.
                    if condition == "target" and dominated and not is_anchor:
                        target_xy = (row["mean_completion_tokens"], row["accuracy"])

                spec = config.language(lang_code)
                degenerate = bool(cell["degenerate_cell"].any())
                ax.set_title(
                    f"{spec.name} ({lang_code})"
                    + ("  [anchor]" if degenerate else "")
                    + f"\ntypological distance {spec.typ_distance:.2f}, "
                      f"{spec.resource_tier}-resource",
                    fontsize=8.5,
                )
                ax.set_xlabel("mean completion tokens")
                ax.set_ylabel("accuracy")
                # Headroom so the annotation and error bars stay inside the axes.
                ax.margins(x=0.20, y=0.16)
                if target_xy is not None:
                    _annotate_corner(ax, target_xy, drawn)

            unique: dict[str, Any] = {}
            for ax in axes:
                handles, labels = ax.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    unique.setdefault(label, handle)
            # Fixed order, so companion figures carry identical legends whatever
            # each panel happened to draw.
            order = [CONDITION_LABELS[c] for c in CONDITION_ORDER] + ["Pareto front"]
            unique = {k: unique[k] for k in order if k in unique} | \
                     {k: v for k, v in unique.items() if k not in order}
            fig.legend(unique.values(), unique.keys(), loc="lower center",
                       ncol=min(5, len(unique)), frameon=False, bbox_to_anchor=(0.5, -0.02))
            fig.suptitle(
                "Accuracy and token cost by thinking-language policy, "
                f"{display_model(model)}, {display_task(task_name)}",
                fontsize=11,
            )
            fig.tight_layout(rect=(0, 0.04, 1, 0.96))
            paths += _save(fig, f"fig1_landscape_{model}_{task_name}", output_dir)
            plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Figure 2 — accuracy matrix
# ---------------------------------------------------------------------------


def fig2_accuracy_matrix(cells: Any, config: Config, output_dir: Path) -> list[Path]:
    plt = _setup_matplotlib()
    import numpy as np

    paths: list[Path] = []
    combos = sorted({(row["model"], row["task"]) for _, row in cells.iterrows()})
    if not combos:
        return paths

    fig, axes = _grid(plt, len(combos), per_row=2, size=(4.6, 3.4))
    for ax, (model, task_name) in zip(axes, combos):
        part = cells[(cells["model"] == model) & (cells["task"] == task_name)]
        languages = sorted(part["language"].unique(),
                           key=lambda code: config.language(code).typ_distance)
        conditions = [c for c in CONDITION_ORDER if c in set(part["condition"].astype(str))]
        matrix = np.full((len(languages), len(conditions)), np.nan)
        for i, lang_code in enumerate(languages):
            for j, condition in enumerate(conditions):
                sel = part[(part["language"] == lang_code) & (part["condition"] == condition)]
                if not sel.empty:
                    matrix[i, j] = float(sel["accuracy"].iloc[0])

        image = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], rotation=20, ha="right")
        ax.set_yticks(range(len(languages)))
        ax.set_yticklabels(
            [f"{config.language(c).name} ({c})" for c in languages]
        )
        ax.grid(False)
        for i in range(len(languages)):
            for j in range(len(conditions)):
                if np.isnan(matrix[i, j]):
                    continue
                sel = part[(part["language"] == languages[i]) & (part["condition"] == conditions[j])]
                lo, hi = float(sel["accuracy_lo"].iloc[0]), float(sel["accuracy_hi"].iloc[0])
                ax.text(
                    j, i, f"{matrix[i, j]:.2f}\n[{lo:.2f},{hi:.2f}]",
                    ha="center", va="center", fontsize=6,
                    color="white" if matrix[i, j] < 0.6 else "black",
                )
        ax.set_title(f"{display_model(model)}, {display_task(task_name)}", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, label="accuracy")

    fig.suptitle("Accuracy by language and thinking-language policy "
                 "(95% bootstrap confidence interval)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    paths += _save(fig, "fig2_accuracy_matrix", output_dir)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Figure 3 — H4 drift and H5 retention
# ---------------------------------------------------------------------------


def fig3_drift_and_retention(
    frame: Any, h5: Any, config: Config, output_dir: Path, n_bins: int = 5,
    model: str = PRIMARY_MODEL,
) -> list[Path]:
    plt = _setup_matplotlib()
    import numpy as np

    # Drawn on one model so panel (b) matches the per-model retention table; the
    # second model's cells are at the accuracy floor and carry no policy signal.
    if model is not None and "model" in getattr(frame, "columns", []):
        frame = frame[frame["model"] == model]
    if model is not None and h5 is not None and "model" in getattr(h5, "columns", []):
        h5 = h5[h5["model"] == model]

    target = frame[(frame["condition"] == "target") & (frame["language"] != "en")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

    # -- (a) H4: accuracy vs binned English drift, one line per language ------
    ax = axes[0]
    if target.empty:
        ax.text(0.5, 0.5, "no Target-CoT data", ha="center", va="center", transform=ax.transAxes)
    else:
        for lang_code in sorted(target["language"].unique(),
                                key=lambda code: config.language(code).typ_distance):
            part = target[target["language"] == lang_code]
            shares = part["trace_english_share"].to_numpy(dtype=float)
            correct = part["correct"].to_numpy(dtype=float)
            if len(shares) < n_bins * 4 or np.nanstd(shares) < 1e-9:
                continue
            quantiles = np.nanquantile(shares, np.linspace(0, 1, n_bins + 1))
            quantiles = np.unique(quantiles)
            if len(quantiles) < 3:
                continue
            centres, rates, errors = [], [], []
            for lo, hi in zip(quantiles[:-1], quantiles[1:]):
                mask = (shares >= lo) & (shares <= hi)
                if mask.sum() < 5:
                    continue
                rate = float(correct[mask].mean())
                centres.append(float(shares[mask].mean()))
                rates.append(rate)
                errors.append(float(np.sqrt(max(rate * (1 - rate), 1e-9) / mask.sum())))
            if len(centres) >= 2:
                ax.errorbar(centres, rates, yerr=errors, marker="o", ms=4, lw=1.2,
                            capsize=2, label=f"{lang_code}")
        rho, p = spearman(
            target["trace_english_share"].tolist(), target["correct"].astype(float).tolist()
        )
        ax.set_title(
            f"(a) H4: English drift vs accuracy within Target-CoT\npooled Spearman ρ={rho:.2f} (p={p:.1e})",
            fontsize=9,
        )
        ax.set_xlabel("English share of trace")
        ax.set_ylabel("accuracy")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(ncol=2, frameon=False, fontsize=7)

    # -- (b) H5: retention by policy ------------------------------------------
    ax = axes[1]
    if h5 is None or getattr(h5, "empty", True):
        ax.text(0.5, 0.5, "no retention data", ha="center", va="center", transform=ax.transAxes)
    else:
        part = h5[(h5["metric"] == "entity_retention") & (h5["language"] != "en")]
        languages = sorted(part["language"].unique(),
                           key=lambda code: config.language(code).typ_distance)
        conditions = [c for c in CONDITION_ORDER if c in set(part["condition"].astype(str))]
        width = 0.8 / max(len(conditions), 1)
        positions = np.arange(len(languages))
        for index, condition in enumerate(conditions):
            means, errors = [], []
            for lang_code in languages:
                sel = part[(part["language"] == lang_code) & (part["condition"] == condition)]
                if sel.empty:
                    means.append(np.nan)
                    errors.append(0.0)
                    continue
                mean = float(sel["mean"].mean())
                means.append(mean)
                errors.append(float(max(sel["hi"].mean() - mean, 0)))
            ax.bar(
                positions + index * width - 0.4 + width / 2, means, width * 0.92,
                yerr=errors, capsize=2, label=CONDITION_LABELS[condition],
                color=POLICY_COLORS.get(condition, "0.5"), edgecolor="white", linewidth=0.4,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(languages)
        ax.set_ylabel("entity retention in trace")
        ax.set_xlabel("language")
        ax.set_title("(b) H5: verbatim entity retention by policy", fontsize=9)
        ax.legend(ncol=2, frameon=False, fontsize=7)

    fig.tight_layout()
    paths = _save(fig, "fig3_drift_and_retention", output_dir)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Figure 4 — gap vs typological distance and fertility
# ---------------------------------------------------------------------------


def fig4_gap_vs_distance(
    cells: Any, config: Config, fertility: Any | None, output_dir: Path,
    model: str = PRIMARY_MODEL,
) -> list[Path]:
    plt = _setup_matplotlib()
    import numpy as np

    # One model only: the other tokenizer's fertility measurement is the
    # instrumentation artifact documented in the fertility appendix, so plotting
    # or fitting it here would contradict its exclusion from the analysis.
    if model is not None and "model" in getattr(cells, "columns", []):
        cells = cells[cells["model"] == model]

    rows = []
    for (model, task_name, language), part in cells.groupby(
        ["model", "task", "language"], observed=True
    ):
        by_condition = part.set_index("condition")["accuracy"].to_dict()
        tokens = part.set_index("condition")["mean_completion_tokens"].to_dict()
        if "target" not in by_condition or "en_cot" not in by_condition:
            continue
        if config.language(language).is_anchor:
            continue
        spec = config.language(language)
        ratio = None
        if fertility is not None and not getattr(fertility, "empty", True):
            match = fertility[
                (fertility["model"] == model)
                & (fertility["task"] == task_name)
                & (fertility["language"] == language)
            ]
            if not match.empty:
                ratio = float(match["fertility_ratio"].mean())
        rows.append(
            {
                "model": model,
                "task": task_name,
                "language": language,
                "typ_distance": spec.typ_distance,
                "resource_proxy_gb": spec.resource_proxy_gb,
                "fertility_ratio": ratio,
                "gap_pp": 100 * (by_condition["en_cot"] - by_condition["target"]),
                "token_ratio": (
                    tokens["target"] / tokens["en_cot"] if tokens.get("en_cot") else np.nan
                ),
            }
        )

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    if not rows:
        for ax in axes:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        paths = _save(fig, "fig4_gap_vs_distance", output_dir)
        plt.close(fig)
        return paths

    import pandas as pd

    table = pd.DataFrame(rows)
    # A single model, so the series are the two tasks. Colour and marker both
    # carry the task, which keeps them separable in greyscale print.
    task_colors = {
        task_name: SERIES_COLORS[index % len(SERIES_COLORS)]
        for index, task_name in enumerate(sorted(table["task"].unique()))
    }
    task_markers = {
        task_name: SERIES_MARKERS[index % len(SERIES_MARKERS)]
        for index, task_name in enumerate(sorted(table["task"].unique()))
    }

    panels = [
        ("typ_distance", "gap_pp", "typological distance from English",
         "En-CoT − Target-CoT (pp)", "(a) accuracy tax vs typological distance"),
        ("fertility_ratio", "gap_pp", "tokenizer fertility vs English (×)",
         "En-CoT − Target-CoT (pp)", "(b) accuracy tax vs tokenizer fertility"),
        ("fertility_ratio", "token_ratio", "tokenizer fertility vs English (×)",
         "Target-CoT / En-CoT tokens (×)", "(c) token tax vs tokenizer fertility"),
    ]
    for ax, (xcol, ycol, xlabel, ylabel, title) in zip(axes, panels):
        subset = table.dropna(subset=[xcol, ycol])
        for task_name in sorted(subset["task"].unique()):
            part = subset[subset["task"] == task_name]
            color = task_colors[task_name]
            ax.scatter(
                part[xcol], part[ycol], s=46, marker=task_markers[task_name],
                facecolor=color, edgecolor="white", linewidth=0.7,
                alpha=0.95, zorder=3, label=display_task(task_name),
            )
            for _, row in part.iterrows():
                ax.annotate(row["language"], (row[xcol], row[ycol]),
                            textcoords="offset points", xytext=(4, 3), fontsize=7,
                            color=color)
        if len(subset) >= 3:
            rho, p = spearman(subset[xcol].tolist(), subset[ycol].tolist())
            fit = np.polyfit(subset[xcol], subset[ycol], 1)
            xs = np.linspace(subset[xcol].min(), subset[xcol].max(), 50)
            ax.plot(xs, np.polyval(fit, xs), color="0.45", lw=1.1, ls="--", zorder=1)
            title = f"{title}\nρ={rho:.2f} (p={p:.3f})"
        if ycol == "gap_pp":
            ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=min(4, len(unique)),
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    paths = _save(fig, "fig4_gap_vs_distance", output_dir)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_all(
    frame: Any,
    cells: Any,
    h5: Any,
    config: Config,
    fertility: Any | None = None,
    output_dir: Path | None = None,
    which: Sequence[str] | None = None,
) -> list[Path]:
    target_dir = output_dir or FIGURES_DIR
    wanted = set(which) if which else {"fig1", "fig2", "fig3", "fig4"}
    paths: list[Path] = []

    steps = [
        ("fig1", lambda: fig1_landscape(cells, config, target_dir)),
        ("fig2", lambda: fig2_accuracy_matrix(cells, config, target_dir)),
        ("fig3", lambda: fig3_drift_and_retention(frame, h5, config, target_dir)),
        ("fig4", lambda: fig4_gap_vs_distance(cells, config, fertility, target_dir)),
    ]
    active = [(name, fn) for name, fn in steps if name in wanted]
    with LoggedProgress(len(active), "figures", unit="fig", logger_name="figures") as bar:
        for name, fn in active:
            try:
                paths += fn()
            except Exception:
                log.exception("%s failed; continuing with the remaining figures", name)
            bar.update(1, {"figure": name})
    return paths
