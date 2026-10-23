"""Synthesise a plausible results.parquet so the analysis can be validated with no GPU.

This does NOT produce results. It plants known effects — a Target-CoT accuracy
penalty that scales with typological distance (H1), a Mixed-CoT recovery (H2),
fertility-scaled token cost (H3), a real within-Target-CoT drift benefit (H4), and
highest anchor retention under Mixed-CoT (H5) — then lets you check that
``ltd analyze`` recovers them and that every figure renders at full scale
(25,200 rows, 112 cells, exactly the real experiment's shape).

Use it to verify the analysis end of the pipeline before spending GPU hours::

    .venv/bin/python scripts/make_synthetic_results.py
    .venv/bin/python -m ltd analyze --results results/results_synth.parquet --skip-fertility
    .venv/bin/python -m ltd figures --results results/results_synth.parquet

Then delete the synthetic outputs before a real run::

    rm -f results/results_synth.parquet results/tables/fertility.csv

Expected: H1 gap larger for distant languages with positive Spearman rho against
typ_distance; H3 Target-CoT dominated in the Indic cells; H5 argmax = mixed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from ltd.config import load_config

cfg = load_config()
rng = np.random.default_rng(7)

# Ground-truth effects we plant, so we can check the analysis recovers them.
BASE = {"mgsm": 0.72, "gmmlu": 0.58}
COND_PENALTY = {"target": -0.09, "en_cot": 0.0, "free": -0.01, "mixed": -0.02}
TOK = {"mgsm": 620.0, "gmmlu": 480.0}

rows = []
for model, model_scale in [("r1-llama-8b", 0.0), ("qwen3-8b", 0.04)]:
    for task in ["mgsm", "gmmlu"]:
        n_items = 250 if task == "mgsm" else 200
        item_ids = [f"{task}-{i:04d}" for i in range(n_items)]
        # Shared item difficulty so paired tests have something to cancel.
        difficulty = rng.normal(0, 0.9, n_items)
        faith_ids = set(item_ids[:50])
        for code in cfg.default_languages:
            spec = cfg.language(code)
            d = spec.typ_distance
            fert = 1.0 + 2.2 * d  # stand-in for measured fertility
            for cond in cfg.conditions:
                # Target-CoT penalty scales with typological distance (H1).
                pen = COND_PENALTY[cond] * (1.0 if cond != "target" else (0.3 + 1.6 * d))
                p = BASE[task] + model_scale + pen - 0.10 * d
                # English drift, by condition and script (H4 regressor).
                if code == "en":
                    drift = rng.uniform(0.9, 1.0, n_items)
                elif cond == "target":
                    drift = np.clip(rng.beta(1.4, 6.0, n_items) + 0.25 * d, 0, 1)
                elif cond == "en_cot":
                    drift = rng.uniform(0.9, 1.0, n_items)
                elif cond == "free":
                    drift = rng.uniform(0.75, 1.0, n_items)
                else:
                    drift = rng.uniform(0.7, 0.95, n_items)
                # H4: within Target-CoT, drift genuinely helps.
                logit = np.log(p / (1 - p)) - difficulty + (1.6 * (drift - drift.mean())
                                                            if cond == "target" else 0.0)
                correct = rng.random(n_items) < 1 / (1 + np.exp(-logit))
                # Cost: target-script traces cost fertility-scaled tokens (H3).
                scale = fert if cond == "target" else (1.0 + 0.12 * (fert - 1))
                tokens = np.clip(rng.normal(TOK[task] * scale, 90, n_items), 40, 3072)
                trunc = tokens >= 3000
                target_share = 1 - drift
                for i, item_id in enumerate(item_ids):
                    in_sub = item_id in faith_ids
                    # H5: retention highest under Mixed-CoT.
                    ret_base = {"target": 0.55, "en_cot": 0.18, "free": 0.25, "mixed": 0.78}[cond]
                    rows.append({
                        "run_id": "synth", "model": model, "task": task, "language": code,
                        "condition": cond,
                        "condition_label": {"target": "Target-CoT", "en_cot": "En-CoT",
                                            "free": "Free-CoT", "mixed": "Mixed-CoT"}[cond],
                        "item_id": item_id, "sample_index": 0,
                        "gold": "18", "prediction": "18" if correct[i] else "19",
                        "correct": bool(correct[i]),
                        "extraction_failed": bool(rng.random() < 0.01),
                        "well_formed_trace": True,
                        "n_prompt_tokens": 300, "n_completion_tokens": float(tokens[i]),
                        "finish_reason": "length" if trunc[i] else "stop",
                        "truncated": bool(trunc[i]), "max_new_tokens": 3072,
                        "seed": 1, "backend": "synth", "instruction_language": "en",
                        "degenerate_cell": bool(code == "en" and cond in {"en_cot", "mixed"}),
                        "trace_chars": int(tokens[i] * 3), "answer_chars": 40,
                        "trace_target_share": float(target_share[i]),
                        "trace_english_share": float(drift[i]),
                        "trace_target_script_share": float(target_share[i]),
                        "trace_latin_script_share": float(drift[i]),
                        "trace_langid_top": code if target_share[i] > 0.5 else "en",
                        "trace_langid_target_share": float(target_share[i]),
                        "trace_langid_en_share": float(drift[i]),
                        "trace_alpha_chars": int(tokens[i] * 2.5),
                        "trace_words": int(tokens[i] / 1.4),
                        "trace_target_script_tokens": int(50 * target_share[i]),
                        "trace_compliant": bool(target_share[i] >= 0.5),
                        "trace_has_target_anchor": bool(target_share[i] > 0.02),
                        "answer_target_share": float(target_share[i]),
                        "answer_compliant": bool(cond != "free"),
                        "subject": "algebra" if task == "gmmlu" else None,
                        "subject_category": "STEM" if task == "gmmlu" else None,
                        "in_faithfulness_subsample": in_sub,
                        "number_retention": float(np.clip(rng.normal(ret_base + 0.1, 0.15), 0, 1)) if in_sub else None,
                        "entity_retention": float(np.clip(rng.normal(ret_base, 0.15), 0, 1)) if in_sub else None,
                        "anchor_retention": float(np.clip(rng.normal(ret_base + 0.05, 0.15), 0, 1)) if in_sub else None,
                        "n_number_anchors": 3 if in_sub else None,
                        "n_entity_anchors": 8 if in_sub else None,
                    })

frame = pd.DataFrame(rows)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "results_synth.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
frame.to_parquet(out, index=False)
print(f"wrote {out} — {len(frame)} rows, {frame.groupby(['model','task','language','condition']).ngroups} cells")

# A synthetic fertility table too.
frows = []
for model in ["r1-llama-8b", "qwen3-8b"]:
    bump = 1.0 if model == "r1-llama-8b" else 0.85  # different tokenizers
    for task in ["mgsm", "gmmlu"]:
        for code in cfg.default_languages:
            d = cfg.language(code).typ_distance
            tpw = 1.25 * (1 + 2.2 * d * bump)
            frows.append({"model": model, "tokenizer": model, "task": task, "language": code,
                          "n_items": 250, "n_tokens": 1000, "n_words": 800,
                          "n_chars": 4000, "tokens_per_word": tpw,
                          "tokens_per_char": tpw / 5})
fert = pd.DataFrame(frows)
base = {(m, t): fert[(fert.model == m) & (fert.task == t) & (fert.language == "en")]
        ["tokens_per_word"].iloc[0] for m in fert.model.unique() for t in fert.task.unique()}
fert["fertility_ratio"] = [r.tokens_per_word / base[(r.model, r.task)] for r in fert.itertuples()]
fert["fertility_ratio_chars"] = fert["fertility_ratio"]
(ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
fert.to_csv(ROOT / "results" / "tables" / "fertility.csv", index=False)
print("wrote synthetic fertility.csv")
