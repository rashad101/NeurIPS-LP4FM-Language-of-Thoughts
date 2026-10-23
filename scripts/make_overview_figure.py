#!/usr/bin/env python3
"""Draw a generated pipeline schematic (``paper/figures/overview_generated.png``).

SUPERSEDED. The paper now ships a hand-drawn methodology overview at
``paper/figures/overview.png``; this script writes to a separate filename so
that it cannot overwrite it. Kept because it is still a useful cross-check:
the language set, item counts, decoding configuration, seed and completion
budgets are read from ``configs/default.yaml``, and the generation total is
recomputed from those numbers rather than typed in, so running it verifies
that the shipped diagram agrees with the configuration.

    .venv/bin/python scripts/make_overview_figure.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox

OUT_PNG = ROOT / "paper" / "figures" / "overview_generated.png"
OUT_PDF = ROOT / "paper" / "figures" / "overview_generated.pdf"

# --- facts, read from the experiment configuration -------------------------
CFG = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
LANGS = CFG["default_languages"]
MODELS = CFG["default_models"]
CONDITIONS = CFG["conditions"]
MGSM_N = CFG["tasks"]["mgsm"]["n_items"]
GMMLU_N = CFG["tasks"]["gmmlu"]["n_items"]
MGSM_CAP = CFG["tasks"]["mgsm"]["max_new_tokens"]
GMMLU_CAP = CFG["tasks"]["gmmlu"]["max_new_tokens"]
SEED = CFG["seed"]
QWEN_DEC = CFG["models"]["qwen3-8b"]["decoding"]
THRESHOLD = CFG["scoring"]["compliance_threshold"]
SUBSAMPLE = CFG["scoring"]["faithfulness_subsample"]
BOOTSTRAP = CFG["scoring"]["bootstrap_iters"]

N_CELLS = len(MODELS) * len(CONDITIONS) * len(LANGS) * 2
MGSM_GENS = MGSM_N * len(LANGS) * len(CONDITIONS) * len(MODELS)
GMMLU_GENS = GMMLU_N * len(LANGS) * len(CONDITIONS) * len(MODELS)
TOTAL = MGSM_GENS + GMMLU_GENS

# --- palette, one hue per stage --------------------------------------------
STAGES = [
    dict(num="1", title="Experimental Factors", edge="#1F5FA9", fill="#EAF2FB", tint="#F5F9FE"),
    dict(num="2", title="Inference Generation", edge="#1E7B4D", fill="#E8F5EE", tint="#F4FBF7"),
    dict(num="3", title="Per-Generation Records", edge="#6B3FA0", fill="#F1EBF9", tint="#F8F5FC"),
    dict(num="4", title="Evaluation Metrics", edge="#C86A10", fill="#FDF0E1", tint="#FEF8F1"),
    dict(num="5", title="Analysis & Outputs", edge="#B3282D", fill="#FBEBEC", tint="#FDF5F6"),
]

#: Body entries are bullets; each is wrapped to the card width at draw time.
CARDS: list[list[tuple[str, list[str]]]] = [
    [
        ("Models (2)", [
            "Qwen3-8B, thinking mode",
            "R1-Distill-Llama-8B",
        ]),
        ("Thinking-Language Policies (4)", [
            "Target-CoT: reason and answer in L",
            "En-CoT: reason in English, answer in L",
            "Mixed-CoT: English, L entities verbatim",
            "Free-CoT: no constraint",
        ]),
        ("Languages (7)", [
            "EN anchor, DE, ES, SW",
            "HI, BN, TE (Brahmic scripts)",
        ]),
        ("Tasks (2)", [
            f"MGSM: {MGSM_N} items per language",
            f"Global MMLU CA: {GMMLU_N} items",
        ]),
    ],
    [
        ("Input Construction", [
            "Question in L, byte-identical across policies",
            "Answer format held constant, so only the "
            "thinking instruction varies",
        ]),
        ("Generation Grid", [
            f"{len(MODELS)} models × {len(CONDITIONS)} policies × "
            f"{len(LANGS)} languages × 2 tasks",
            f"= {N_CELLS} cells, every item run once",
        ]),
        ("Decoding", [
            f"temp {QWEN_DEC['temperature']}, top-p {QWEN_DEC['top_p']}, "
            f"top-k {QWEN_DEC['top_k']} (Qwen) / off (R1)",
            f"max new tokens {MGSM_CAP:,} / {GMMLU_CAP:,}",
            f"1 sample per item, seed {SEED}",
        ]),
        ("__total__", [
            f"{MGSM_GENS:,} MGSM + {GMMLU_GENS:,} Global MMLU",
            f"= {TOTAL:,} generations",
        ]),
    ],
    [
        ("Recorded Artifacts", [
            "Full output text, stored losslessly",
            "Prompt and completion token counts",
            "Finish reason and truncation flag",
            "Model revision, seed, decoding config",
        ]),
        ("Trace Parsing", [
            "Split at the thinking delimiter into trace and "
            "answer, and flag outputs that omit it",
        ]),
        ("Normalization", [
            "Unicode digit normalization before extraction",
            "Number and content-word anchors taken from "
            "the question",
        ]),
        ("Degenerate Cells", [
            "For English, three of the four instructions "
            "coincide, so those cells are flagged and "
            "excluded from every contrast",
        ]),
    ],
    [
        ("Accuracy", [
            "Digit-normalized exact match (MGSM) or "
            "option letter (Global MMLU)",
        ]),
        ("Token Cost", [
            "Mean completion tokens, with truncation rate",
        ]),
        ("Trace-Language Share", [
            "Script ranges for Brahmic targets",
            "Per-segment langid for Latin targets",
            f"Compliant when share ≥ {THRESHOLD:g}",
        ]),
        ("Answer-Language Fidelity", [
            "Whether the final answer is written in L",
        ]),
        ("Verbatim Retention", [
            "Question numbers and content words that "
            f"reappear in the trace ({SUBSAMPLE}-item subsample)",
        ]),
    ],
    [
        ("Landscape Mapping", [
            "Pareto fronts in (accuracy, token cost) per "
            "model × task × language cell",
        ]),
        ("Statistics", [
            f"{BOOTSTRAP:,}-iteration percentile bootstrap",
            "Item-paired contrasts, exact McNemar",
        ]),
        ("Preregistered Tests", [
            "H1–H5 verdicts, reported including the three "
            "the data did not support",
        ]),
        ("Tokenizer Fertility", [
            "Tokens per word against English, Qwen3 tokenizer",
        ]),
        ("Deliverables", [
            "Tables and figures, the decision framework, "
            "the thinking-language policy card",
        ]),
    ],
]

# --- geometry ---------------------------------------------------------------
FIG_W, FIG_H = 16.4, 7.0
MARGIN_X, GAP = 0.5, 1.25
PANEL_W = (100 - 2 * MARGIN_X - 4 * GAP) / 5
PANEL_TOP, PANEL_BOTTOM = 96.5, 3.0
HEADER_H = 7.0
CARD_PAD, CARD_GAP = 1.0, 1.4
TITLE_H, LINE_H = 2.8, 2.35
BODY_FS, TITLE_FS, HEAD_FS = 8.2, 9.3, 11.2
BULLET_GAP = 0.5          # extra space between bullets within a card

#: Characters that fit on one body line, from the panel's inner width. DejaVu
#: Sans averages ~0.52 em per character at this size.
INNER_W_IN = (PANEL_W - 5.0) / 100 * FIG_W
WRAP = max(20, int(INNER_W_IN / (0.52 * BODY_FS / 72)))


def rounded(ax, x, y, w, h, *, fc, ec, lw=1.0, r=0.9, z=1):
    ax.add_patch(FancyBboxPatch(
        (x + r, y + r), w - 2 * r, h - 2 * r,
        boxstyle=f"round,pad={r},rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
        mutation_aspect=FIG_W / FIG_H,
    ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Pass one: wrap every card and measure it, so the panels can be made just
    # tall enough for the fullest column instead of a guessed height.
    layout = []
    for cards in CARDS:
        wrapped = [
            [textwrap.wrap(bullet, WRAP) or [""] for bullet in body]
            for _, body in cards
        ]
        heights = []
        for (title, _), bullets in zip(cards, wrapped):
            lines = sum(len(part) for part in bullets)
            if title == "__total__":
                heights.append(2 * CARD_PAD + LINE_H * lines + 1.4)
            else:
                heights.append(2 * CARD_PAD + TITLE_H + LINE_H * lines
                               + BULLET_GAP * (len(bullets) - 1))
        layout.append((wrapped, heights))

    tallest = max(sum(h) + CARD_GAP * (len(h) - 1) for _, h in layout)
    panel_h = HEADER_H + tallest + 2 * CARD_GAP
    panel_bottom = PANEL_TOP - panel_h

    for index, (stage, cards) in enumerate(zip(STAGES, CARDS)):
        x0 = MARGIN_X + index * (PANEL_W + GAP)
        wrapped, heights = layout[index]

        rounded(ax, x0, panel_bottom, PANEL_W, panel_h,
                fc="white", ec=stage["edge"], lw=1.5, r=1.0, z=1)
        rounded(ax, x0, PANEL_TOP - HEADER_H, PANEL_W, HEADER_H,
                fc=stage["fill"], ec=stage["edge"], lw=1.5, r=1.0, z=2)

        badge_x = x0 + 2.3
        badge_y = PANEL_TOP - HEADER_H / 2
        ax.add_patch(Circle((badge_x, badge_y), 1.35, transform=ax.transData,
                            facecolor=stage["edge"], edgecolor="none", zorder=4))
        ax.text(badge_x, badge_y, stage["num"], ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=5)
        ax.text(badge_x + 2.6, badge_y, stage["title"], ha="left", va="center",
                fontsize=11.5, fontweight="bold", color=stage["edge"], zorder=5)

        # Cards keep their content height; the column's slack is spread between
        # them, and the stack is centred so the five columns share a baseline.
        available = panel_h - HEADER_H
        slack = available - sum(heights)
        gap = min(max(CARD_GAP, slack / (len(cards) + 1)), 3.2) if slack > 0 else CARD_GAP
        stack = sum(heights) + gap * (len(cards) - 1)
        y = PANEL_TOP - HEADER_H - max(CARD_GAP, (available - stack) / 2)
        for (title, _), bullets, height in zip(cards, wrapped, heights):
            y -= height
            total_card = title == "__total__"
            rounded(ax, x0 + 1.3, y, PANEL_W - 2.6, height,
                    fc=stage["fill"] if total_card else stage["tint"],
                    ec=stage["edge"] if total_card else "#D8DBE2",
                    lw=1.3 if total_card else 0.8, r=0.65, z=3)

            if total_card:
                flat = [line for part in bullets for line in part]
                for offset, line in enumerate(flat):
                    ax.text(x0 + PANEL_W / 2,
                            y + height - CARD_PAD - LINE_H * (offset + 0.55),
                            line, ha="center", va="center", zorder=6,
                            fontsize=9.6 if offset else 8.4,
                            fontweight="bold" if offset else "normal",
                            color=stage["edge"] if offset else "#333A45")
            else:
                ax.text(x0 + 2.6, y + height - CARD_PAD - TITLE_H / 2, title,
                        ha="left", va="center", fontsize=TITLE_FS,
                        fontweight="bold", color=stage["edge"], zorder=6)
                cursor = y + height - CARD_PAD - TITLE_H
                for part in bullets:
                    for line in part:
                        ax.text(x0 + 2.6, cursor - LINE_H * 0.55, line,
                                ha="left", va="center", fontsize=BODY_FS,
                                color="#333A45", zorder=6)
                        cursor -= LINE_H
                    cursor -= BULLET_GAP
            y -= gap

        if index < 4:
            arrow_x = x0 + PANEL_W
            ax.add_patch(FancyArrowPatch(
                (arrow_x + 0.15, 50), (arrow_x + GAP - 0.1, 50),
                arrowstyle="-|>", mutation_scale=18,
                linewidth=2.0, color="#7A828F", zorder=7,
            ))

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # The axes fills the figure and spans 0-100 in both directions, so data
    # units map linearly to inches. Crop to the drawn band rather than to the
    # axes, which would leave the unused lower half of the canvas in the file.
    # The rounded panels extend one corner-radius beyond their nominal box, so
    # the crop has to clear that as well as the stroke width.
    pad = 2.0
    right = MARGIN_X + 5 * PANEL_W + 4 * GAP
    box = Bbox.from_extents(
        (MARGIN_X - pad) / 100 * FIG_W,
        (panel_bottom - pad) / 100 * FIG_H,
        (right + pad) / 100 * FIG_W,
        (PANEL_TOP + pad) / 100 * FIG_H,
    )
    for path, dpi in ((OUT_PNG, 400), (OUT_PDF, 400)):
        fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches=box)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
