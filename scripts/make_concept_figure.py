#!/usr/bin/env python3
"""Draw the paper's conceptual figure (``paper/figures/concept.png``).

The argument in one picture: a reasoning trace serves two objectives with
different optimality criteria; the consistency norm fuses them into a single
constraint, at a price that scales with how badly the tokenizer serves the
script; the paper's position is to separate them and disclose the choice.

Every quantity annotated on the figure is read from ``results/`` so the picture
cannot drift from the numbers in the tables.

    .venv/bin/python scripts/make_concept_figure.py
"""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUT_PNG = ROOT / "paper" / "figures" / "concept.png"
OUT_PDF = ROOT / "paper" / "figures" / "concept.pdf"

DISTANT = ["sw", "hi", "bn", "te"]
NEAR = ["de", "es"]
NAMES = {"de": "German", "es": "Spanish", "sw": "Swahili",
         "hi": "Hindi", "bn": "Bengali", "te": "Telugu"}

# --- numbers, read from the released tables --------------------------------
CELLS = list(csv.DictReader((TABLES / "cells.csv").open(encoding="utf-8")))
FERT = list(csv.DictReader((TABLES / "fertility.csv").open(encoding="utf-8")))


def cell(task: str, lang: str, cond: str) -> dict[str, str]:
    for row in CELLS:
        if (row["model"], row["task"], row["language"], row["condition"]) == (
                "qwen3-8b", task, lang, cond):
            return row
    raise KeyError((task, lang, cond))


FERTILITY = {
    lang: mean(float(r["fertility_ratio"]) for r in FERT
               if r["model"] == "qwen3-8b" and r["language"] == lang)
    for lang in NEAR + DISTANT
}
RATIO = {
    lang: mean(float(cell(task, lang, "target")["mean_completion_tokens"])
               / float(cell(task, lang, "en_cot")["mean_completion_tokens"])
               for task in ("mgsm", "gmmlu"))
    for lang in NEAR + DISTANT
}
DISTANT_RATIOS = [
    float(cell(task, lang, "target")["mean_completion_tokens"])
    / float(cell(task, lang, "en_cot")["mean_completion_tokens"])
    for task in ("mgsm", "gmmlu") for lang in DISTANT
]
ABOVE = [value for value in DISTANT_RATIOS if value > 1]
DE_COMPLIANCE = [100 * float(cell(task, "de", "target")["compliance_rate"])
                 for task in ("mgsm", "gmmlu")]

# --- palette ----------------------------------------------------------------
INK = "#20242B"
MUTED = "#5A616D"
NEUTRAL_E, NEUTRAL_F = "#3E5C76", "#EEF2F6"
NORM_E, NORM_F = "#A4343A", "#FBEDEE"
POS_E, POS_F = "#1E7B4D", "#EAF5EF"

FIG_W, FIG_H = 13.6, 5.9


def rounded(ax, x, y, w, h, *, fc, ec, lw=1.1, r=1.1, z=2, ls="solid"):
    ax.add_patch(FancyBboxPatch(
        (x + r, y + r), w - 2 * r, h - 2 * r,
        boxstyle=f"round,pad={r},rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
        zorder=z, mutation_aspect=FIG_W / FIG_H,
    ))


def wrap(text: str, width_units: float, fontsize: float) -> list[str]:
    """Wrap ``text`` to a column ``width_units`` wide in figure units.

    DejaVu Sans averages about 0.52 em per character, which is close enough to
    keep every line inside its box at the sizes used here.
    """
    inches = width_units / 100 * FIG_W
    chars = max(8, int(inches / (0.52 * fontsize / 72)))
    return textwrap.wrap(text, chars) or [""]


def arrow(ax, start, end, *, color=MUTED, lw=1.6, style="-|>", scale=15, z=6, conn=None):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=scale,
        linewidth=lw, color=color, zorder=z,
        connectionstyle=conn or "arc3,rad=0",
        shrinkA=0, shrinkB=0,
    ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---------------------------------------------------------------- panel A
    ax_x, ax_w = 1.0, 33.0
    rounded(ax, ax_x, 6, ax_w, 88, fc="white", ec="#C9CFD8", lw=1.2, r=1.2, z=1)
    ax.text(ax_x + ax_w / 2, 89.5, "A reasoning trace does two jobs",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color=INK, zorder=5)
    ax.text(ax_x + ax_w / 2, 84.4, "and they have different optimality criteria",
            ha="center", va="center", fontsize=9.3, color=MUTED, style="italic", zorder=5)

    # computation
    rounded(ax, ax_x + 2.4, 62.5, ax_w - 4.8, 17.5, fc=NEUTRAL_F, ec=NEUTRAL_E, lw=1.1, r=0.9, z=3)
    ax.text(ax_x + 4.6, 75.6, "trace as COMPUTATION", ha="left", va="center",
            fontsize=9.8, fontweight="bold", color=NEUTRAL_E, zorder=5)
    for offset, line in enumerate([
        "the medium the model computes in",
        "choose it for whatever makes the",
        "answer most likely to be correct",
    ]):
        ax.text(ax_x + 4.6, 71.6 - offset * 3.3, line, ha="left", va="center",
                fontsize=8.8, color=INK, zorder=5)

    # the trace itself
    rounded(ax, ax_x + 2.4, 45.4, ax_w - 4.8, 9.2, fc="white", ec=INK, lw=1.6, r=0.85, z=3)
    ax.text(ax_x + ax_w / 2, 50.0, "one reasoning trace", ha="center", va="center",
            fontsize=11.0, fontweight="bold", color=INK, zorder=5)
    arrow(ax, (ax_x + ax_w / 2, 62.2), (ax_x + ax_w / 2, 55.2), lw=1.4)
    arrow(ax, (ax_x + ax_w / 2, 37.8), (ax_x + ax_w / 2, 44.8), lw=1.4)

    # communication
    rounded(ax, ax_x + 2.4, 20.0, ax_w - 4.8, 17.5, fc=NEUTRAL_F, ec=NEUTRAL_E, lw=1.1, r=0.9, z=3)
    ax.text(ax_x + 4.6, 33.1, "trace as COMMUNICATION", ha="left", va="center",
            fontsize=9.8, fontweight="bold", color=NEUTRAL_E, zorder=5)
    for offset, line in enumerate([
        "an artifact a person reads and checks",
        "choose it for the audience, usually",
        "the language the question was asked in",
    ]):
        ax.text(ax_x + 4.6, 29.1 - offset * 3.3, line, ha="left", va="center",
                fontsize=8.8, color=INK, zorder=5)

    ax.text(ax_x + ax_w / 2, 12.5,
            "One mechanism is being asked\nto satisfy both at once.",
            ha="center", va="center", fontsize=9.2, color=INK, style="italic", zorder=5)

    # ------------------------------------------------------------ fork arrows
    arrow(ax, (ax_x + ax_w + 0.8, 62), (39.2, 74), lw=2.0, scale=17,
          conn="arc3,rad=-0.18", color=NORM_E)
    arrow(ax, (ax_x + ax_w + 0.8, 38), (39.2, 26), lw=2.0, scale=17,
          conn="arc3,rad=0.18", color=POS_E)

    # ---------------------------------------------------------------- panel B
    bx, bw = 40.0, 59.0
    rounded(ax, bx, 52.0, bw, 42.0, fc=NORM_F, ec=NORM_E, lw=1.3, r=1.1, z=1)
    ax.text(bx + 2.6, 89.4, "The consistency norm fuses them",
            ha="left", va="center", fontsize=12.0, fontweight="bold", color=NORM_E, zorder=5)
    ax.text(bx + 2.6, 84.6,
            "one constraint, that the trace must be written in the user's language, whatever it costs",
            ha="left", va="center", fontsize=8.9, color=MUTED, style="italic", zorder=5)

    # Left column: what the norm costs. Right column: why it is regressive.
    col_x, col_w = bx + 3.2, 27.0
    y = 79.2
    for head, body in [
        ("A prompt cannot buy it.",
         f"0% of traces clear the threshold for Swahili, Hindi, Bengali and "
         f"Telugu. Only German complies, at {DE_COMPLIANCE[0]:.0f} to {DE_COMPLIANCE[1]:.0f}%."),
        ("Enforcing it costs accuracy.",
         "priced by the training-side literature, not by us."),
        ("Asking for it already costs tokens.",
         f"{min(ABOVE):.2f} to {max(ABOVE):.2f}× the tokens of an English trace, "
         f"in {len(ABOVE)} of the 8 distant cells."),
    ]:
        for line in wrap(head, col_w, 9.0):
            ax.text(col_x, y, line, ha="left", va="center",
                    fontsize=9.0, fontweight="bold", color=INK, zorder=5)
            y -= 2.7
        for line in wrap(body, col_w, 8.5):
            ax.text(col_x, y, line, ha="left", va="center",
                    fontsize=8.5, color=INK, zorder=5)
            y -= 2.5
        y -= 1.5

    # the regressive mechanism, drawn from measured fertility
    chart_x = bx + 32.5
    y = 79.2
    for line in wrap("and the cost is regressive: it scales with how badly the "
                     "tokenizer serves the script", 24.0, 9.0):
        ax.text(chart_x, y, line, ha="left", va="center",
                fontsize=9.0, fontweight="bold", color=NORM_E, zorder=5)
        y -= 2.7

    order = ["es", "de", "sw", "hi", "bn", "te"]
    label_x, bar_x, bar_max = chart_x + 6.2, chart_x + 6.8, 12.5
    top = FERTILITY[order[-1]]
    y -= 1.4
    for lang in order:
        width = bar_max * FERTILITY[lang] / top
        ax.add_patch(FancyBboxPatch(
            (bar_x, y - 0.62), width, 1.24,
            boxstyle="round,pad=0,rounding_size=0.2",
            facecolor=NORM_E if lang in DISTANT else "#CDA9AC",
            edgecolor="none", zorder=4, mutation_aspect=FIG_W / FIG_H,
        ))
        ax.text(label_x, y, NAMES[lang], ha="right", va="center",
                fontsize=7.8, color=INK, zorder=5)
        ax.text(bar_x + width + 0.8, y, f"{FERTILITY[lang]:.1f}×",
                ha="left", va="center", fontsize=7.8, color=MUTED, zorder=5)
        y -= 2.3
    ax.text(chart_x, y - 0.4, "tokens per word against English, Qwen3 tokenizer",
            ha="left", va="center", fontsize=7.6, color=MUTED, style="italic", zorder=5)

    # ---------------------------------------------------------------- panel C
    rounded(ax, bx, 6.0, bw, 38.0, fc=POS_F, ec=POS_E, lw=1.3, r=1.1, z=1)
    ax.text(bx + 2.6, 39.4, "Our position: separate them, then disclose the choice",
            ha="left", va="center", fontsize=12.0, fontweight="bold", color=POS_E, zorder=5)

    steps = [
        ("1", "Compute in the best medium",
         "reason in the hub language, keeping the question's entities verbatim"),
        ("2", "Render natively, and label it",
         "present the account in the reader's language, as a rendering"),
        ("3", "Disclose the policy and its price",
         "a thinking-language policy card, per language, per release"),
    ]
    step_w = (bw - 5.2 - 2 * 1.2) / 3
    for offset, (num, head, body) in enumerate(steps):
        x = bx + 2.6 + offset * (step_w + 1.2)
        rounded(ax, x, 12.6, step_w, 20.4, fc="white", ec="#B9D3C4", lw=1.0, r=0.8, z=3)
        ax.text(x + 2.5, 30.0, num, ha="center", va="center", fontsize=8.8,
                fontweight="bold", color="white", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.3", facecolor=POS_E, edgecolor="none"))
        y = 30.0
        for line in wrap(head, step_w - 6.6, 8.8):
            ax.text(x + 4.6, y, line, ha="left", va="center", fontsize=8.8,
                    fontweight="bold", color=INK, zorder=5)
            y -= 2.6
        y = min(y, 26.4) - 0.6
        for line in wrap(body, step_w - 3.4, 8.2):
            ax.text(x + 1.9, y, line, ha="left", va="center", fontsize=8.2,
                    color=INK, zorder=5)
            y -= 2.4

    ax.text(bx + 2.6, 9.4,
            "choose per task type × typological distance × audience, and never score compliance as a virtue in itself",
            ha="left", va="center", fontsize=8.4, color=POS_E, style="italic", zorder=5)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    pad = 1.4
    box = Bbox.from_extents(
        (0 - pad) / 100 * FIG_W, (6 - pad - 1.2) / 100 * FIG_H,
        (100 + pad) / 100 * FIG_W, (94 + pad + 1.2) / 100 * FIG_H,
    )
    for path in (OUT_PNG, OUT_PDF):
        fig.savefig(path, dpi=400, facecolor="white", bbox_inches=box)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
