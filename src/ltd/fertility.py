"""Tokenizer fertility per language — the mechanism behind the cost axis (H3).

Fertility is tokens per whitespace-delimited word, measured on the *questions* of
each item set (identical content across languages by construction, so the ratio is
a clean per-language property of the tokenizer).

``fertility_ratio`` normalises by English, which is the number the paper wants:
"a Telugu trace of the same semantic length costs N times as many tokens".

This links workshop topic vi (reasoning media) to topic iii (multilinguality):
the token tax of thinking in L is largely a tokenizer property, not a reasoning
property.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from .config import Config
from .data_build import read_items
from .env import TABLES_DIR
from .logging_utils import LoggedProgress, get_logger, log_table
from .models_fetch import load_tokenizer
from .text_utils import word_count

log = get_logger("fertility")


def measure(
    config: Config,
    models: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    model_keys = config.resolve_models(models)
    task_names = config.resolve_tasks(tasks)
    lang_codes = config.resolve_languages(languages)

    rows: list[dict[str, Any]] = []
    total = len(model_keys) * len(task_names) * len(lang_codes)
    with LoggedProgress(total, "fertility", unit="cell") as bar:
        for model_key in model_keys:
            spec = config.model(model_key)
            tokenizer = load_tokenizer(spec)
            for task_name in task_names:
                for lang_code in lang_codes:
                    try:
                        items = read_items(task_name, lang_code)
                    except FileNotFoundError:
                        log.warning("no item set for %s/%s — skipping", task_name, lang_code)
                        bar.update(1)
                        continue
                    n_tokens = 0
                    n_words = 0
                    n_chars = 0
                    for item in items:
                        text = item["question"]
                        n_tokens += len(tokenizer.encode(text, add_special_tokens=False))
                        n_words += word_count(text)
                        n_chars += len(text)
                    rows.append(
                        {
                            "model": model_key,
                            "tokenizer": spec.hf_id,
                            "task": task_name,
                            "language": lang_code,
                            "n_items": len(items),
                            "n_tokens": n_tokens,
                            "n_words": n_words,
                            "n_chars": n_chars,
                            "tokens_per_word": n_tokens / n_words if n_words else None,
                            "tokens_per_char": n_tokens / n_chars if n_chars else None,
                        }
                    )
                    bar.update(1, {"model": model_key, "lang": lang_code})

    _add_ratios(rows)
    return rows


def _add_ratios(rows: list[dict[str, Any]]) -> None:
    """Attach English-normalised fertility ratios in place."""
    baseline = {
        (row["model"], row["task"]): row["tokens_per_word"]
        for row in rows
        if row["language"] == "en" and row["tokens_per_word"]
    }
    baseline_chars = {
        (row["model"], row["task"]): row["tokens_per_char"]
        for row in rows
        if row["language"] == "en" and row["tokens_per_char"]
    }
    for row in rows:
        key = (row["model"], row["task"])
        base = baseline.get(key)
        row["fertility_ratio"] = (
            row["tokens_per_word"] / base if base and row["tokens_per_word"] else None
        )
        base_c = baseline_chars.get(key)
        row["fertility_ratio_chars"] = (
            row["tokens_per_char"] / base_c if base_c and row["tokens_per_char"] else None
        )
        if base is None:
            log.warning(
                "no English baseline for model=%s task=%s; fertility_ratio unavailable "
                "(include `en` in --langs when building data)",
                row["model"], row["task"],
            )


def save(rows: list[dict[str, Any]], name: str = "fertility") -> tuple[Any, Any]:
    """Write fertility to ``results/tables/`` as CSV + JSON. Returns paths."""
    import pandas as pd

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    csv_path = TABLES_DIR / f"{name}.csv"
    json_path = TABLES_DIR / f"{name}.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    if not frame.empty:
        display = frame[frame["task"] == frame["task"].iloc[0]]
        log_table(
            [
                [
                    row["model"], row["language"],
                    f"{row['tokens_per_word']:.3f}" if row["tokens_per_word"] else "-",
                    f"{row['fertility_ratio']:.2f}x" if row["fertility_ratio"] else "-",
                ]
                for _, row in display.iterrows()
            ],
            ["model", "lang", "tok/word", "vs en"],
            logger_name="fertility",
        )
    log.info("fertility -> %s", csv_path)
    return csv_path, json_path


def load(name: str = "fertility") -> Any:
    import pandas as pd

    path = TABLES_DIR / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
