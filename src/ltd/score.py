"""Turn prediction shards into ``results.parquet`` — one row per generation.

Scoring is incremental and resumable in the same way generation is: per-cell
scored shards are cached under ``runs/<run_id>/scored/`` keyed by the source
shard's size and mtime, so re-running ``ltd score`` after adding more generations
only rescores what changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import CONDITION_LABELS, Config
from .env import RESULTS_DIR
from .extract import extract_prediction, is_correct, split_trace_answer
from .faithfulness import retention, subsample_item_ids
from .logging_utils import LoggedProgress, get_logger, log_table
from .prompts import degenerate_condition
from .runstore import Cell, RunStore
from .trace_language import analyse_trace

log = get_logger("score")

RESULTS_PARQUET = RESULTS_DIR / "results.parquet"


def _cache_dir(store: RunStore) -> Path:
    path = store.run_dir / "scored"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_paths(store: RunStore, cell: Cell) -> tuple[Path, Path]:
    stem = cell.key.replace("/", "__")
    base = _cache_dir(store)
    return base / f"{stem}.jsonl", base / f"{stem}.meta.json"


def _shard_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def score_cell(
    config: Config,
    store: RunStore,
    cell: Cell,
    *,
    faithfulness_ids: set[str] | None,
    force: bool = False,
) -> list[dict[str, Any]]:
    shard = cell.shard_path(store.run_dir)
    if not shard.exists():
        return []

    scored_path, meta_path = _cache_paths(store, cell)
    fingerprint = _shard_fingerprint(shard)
    if scored_path.exists() and meta_path.exists() and not force:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        if meta.get("source") == fingerprint and meta.get("scorer_version") == SCORER_VERSION:
            with scored_path.open("r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            log.info("%s: reusing cached scores (%d rows)", cell.key, len(rows))
            return rows

    task = config.task(cell.task)
    lang = config.language(cell.language)
    scoring = config.scoring

    rows: list[dict[str, Any]] = []
    records = list(store.read_shard(cell))

    # The retention subsample is a fixed set of item ids so retention is comparable
    # across cells. A partial cell (`--limit N`, or a run stopped early) may contain
    # none of them, which would silently leave H5 with no data — score every present
    # item instead, and say so.
    if faithfulness_ids is not None:
        present = {record.get("item_id") for record in records}
        if not (faithfulness_ids & present):
            log.warning(
                "%s: none of the %d retention-subsample items are present (%d generations); "
                "scoring retention on all of them instead — not comparable with a full run",
                cell.key, len(faithfulness_ids), len(records),
            )
            faithfulness_ids = None

    with LoggedProgress(len(records), f"score {cell.key}", unit="gen", logger_name="score") as bar:
        for record in records:
            rows.append(
                _score_record(
                    record=record,
                    config=config,
                    cell=cell,
                    task_kind=task.kind,
                    lang_code=lang.code,
                    compliance_threshold=scoring.compliance_threshold,
                    anchor_threshold=scoring.anchor_threshold,
                    min_anchor_tokens=scoring.min_anchor_tokens,
                    min_anchor_token_len=scoring.min_anchor_token_len,
                    in_faithfulness_subsample=(
                        faithfulness_ids is None or record.get("item_id") in faithfulness_ids
                    ),
                )
            )
            bar.update(1)

    with scored_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta_path.write_text(
        json.dumps({"source": fingerprint, "scorer_version": SCORER_VERSION, "n_rows": len(rows)},
                   indent=2),
        encoding="utf-8",
    )
    return rows


#: Bump when scoring semantics change, so cached scores are invalidated.
SCORER_VERSION = 5


def _score_record(
    *,
    record: dict[str, Any],
    config: Config,
    cell: Cell,
    task_kind: str,
    lang_code: str,
    compliance_threshold: float,
    anchor_threshold: float,
    min_anchor_tokens: int,
    min_anchor_token_len: int,
    in_faithfulness_subsample: bool,
) -> dict[str, Any]:
    output = record.get("output") or ""
    trace, answer, well_formed = split_trace_answer(output)
    prediction = extract_prediction(task_kind, answer, output)
    gold = record.get("gold")
    correct = is_correct(task_kind, prediction, gold)

    trace_stats = analyse_trace(
        trace,
        lang_code,
        compliance_threshold=compliance_threshold,
        anchor_threshold=anchor_threshold,
        min_anchor_tokens=min_anchor_tokens,
    )
    answer_stats = analyse_trace(
        answer,
        lang_code,
        compliance_threshold=compliance_threshold,
        anchor_threshold=anchor_threshold,
        min_anchor_tokens=min_anchor_tokens,
    )

    row: dict[str, Any] = {
        "run_id": record.get("run_id"),
        "model": record.get("model", cell.model),
        "task": record.get("task", cell.task),
        "language": record.get("language", cell.language),
        "condition": record.get("condition", cell.condition),
        "condition_label": CONDITION_LABELS.get(cell.condition, cell.condition),
        "item_id": record.get("item_id"),
        "sample_index": record.get("sample_index", 0),
        "gold": gold,
        "prediction": prediction,
        "correct": bool(correct),
        "extraction_failed": prediction is None,
        "well_formed_trace": well_formed,
        "n_prompt_tokens": record.get("n_prompt_tokens"),
        "n_completion_tokens": record.get("n_completion_tokens"),
        "finish_reason": record.get("finish_reason"),
        "truncated": bool(record.get("truncated", False)),
        "max_new_tokens": record.get("max_new_tokens"),
        "seed": record.get("seed"),
        "backend": record.get("backend"),
        "instruction_language": record.get("instruction_language"),
        "degenerate_cell": degenerate_condition(cell.language, cell.condition),
        "trace_chars": len(trace),
        "answer_chars": len(answer),
        # -- trace language (H4 regressor lives here) --
        "trace_target_share": trace_stats.target_share,
        "trace_english_share": trace_stats.english_share,
        "trace_target_script_share": trace_stats.target_script_share,
        "trace_latin_script_share": trace_stats.latin_script_share,
        "trace_langid_top": trace_stats.langid_top,
        "trace_langid_target_share": trace_stats.langid_target_share,
        "trace_langid_en_share": trace_stats.langid_en_share,
        "trace_alpha_chars": trace_stats.alpha_chars,
        "trace_words": trace_stats.n_words,
        "trace_target_script_tokens": trace_stats.n_target_script_tokens,
        "trace_compliant": trace_stats.compliant,
        "trace_has_target_anchor": trace_stats.has_target_anchor,
        # -- answer language (trace-as-communication side) --
        "answer_target_share": answer_stats.target_share,
        "answer_compliant": answer_stats.compliant,
        "subject": record.get("subject"),
        "subject_category": record.get("subject_category"),
        "in_faithfulness_subsample": bool(in_faithfulness_subsample),
    }

    if in_faithfulness_subsample:
        stats = retention(
            record.get("question") or "",
            trace,
            lang_code,
            min_token_len=min_anchor_token_len,
        )
        row.update(
            {
                "number_retention": stats.number_retention,
                "entity_retention": stats.entity_retention,
                "anchor_retention": stats.anchor_retention,
                "n_number_anchors": stats.n_number_anchors,
                "n_entity_anchors": stats.n_entity_anchors,
            }
        )
    else:
        row.update(
            {
                "number_retention": None,
                "entity_retention": None,
                "anchor_retention": None,
                "n_number_anchors": None,
                "n_entity_anchors": None,
            }
        )
    return row


def score_runs(
    config: Config,
    run_ids: Sequence[str],
    *,
    models: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
    force: bool = False,
    output: Path | None = None,
) -> Any:
    """Score one or more runs into a single parquet. Returns the DataFrame."""
    import pandas as pd

    filters = {
        "model": set(config.resolve_models(models)) if models else None,
        "task": set(config.resolve_tasks(tasks)) if tasks else None,
        "language": set(config.resolve_languages(languages)) if languages else None,
        "condition": set(config.resolve_conditions(conditions)) if conditions else None,
    }

    all_rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        store = RunStore(run_id)
        cells = [
            cell for cell in store.discovered_cells()
            if _keep(cell, filters)
        ]
        if not cells:
            log.warning("run %s: no matching prediction shards", run_id)
            continue
        log.info("run %s: scoring %d cells", run_id, len(cells))

        faith_ids_by_task = _faithfulness_ids(config, store, cells)

        with LoggedProgress(len(cells), f"score run {run_id}", unit="cell",
                            logger_name="score") as bar:
            for cell in cells:
                rows = score_cell(
                    config, store, cell,
                    faithfulness_ids=faith_ids_by_task.get(cell.task),
                    force=force,
                )
                for row in rows:
                    row["run_id"] = run_id
                all_rows.extend(rows)
                bar.update(1, {"cell": cell.key, "rows": len(rows)})

    if not all_rows:
        raise RuntimeError(
            "nothing scored — check --run-id and that `ltd generate` produced shards"
        )

    frame = pd.DataFrame(all_rows)
    frame = _drop_duplicate_generations(frame)
    path = output or RESULTS_PARQUET
    path.parent.mkdir(parents=True, exist_ok=True)

    # Scoring a subset must not silently shrink a fuller results file. The raw
    # predictions are safe either way (this only rewrites a derived view), but a
    # analysis run against a truncated parquet would be quietly wrong.
    if path.exists():
        try:
            previous = pd.read_parquet(path, columns=["run_id", "model", "task",
                                                      "language", "condition"])
        except Exception:
            previous = None
        if previous is not None and len(previous) > len(frame):
            backup = path.with_name(f"{path.stem}.superseded{path.suffix}")
            path.replace(backup)
            log.warning(
                "the existing %s held %d rows and this scoring pass produced %d "
                "(a narrower selection); the previous file was kept as %s — rerun "
                "`ltd score --run-id all` to rebuild the full view",
                path.name, len(previous), len(frame), backup.name,
            )

    frame.to_parquet(path, index=False)
    log.info("results -> %s (%d rows, %d cells)", path, len(frame), frame.groupby(
        ["run_id", "model", "task", "language", "condition"], dropna=False).ngroups)
    _log_overview(frame)
    return frame


def _keep(cell: Cell, filters: dict[str, set[str] | None]) -> bool:
    for field, allowed in filters.items():
        if allowed is None:
            continue
        if getattr(cell, field) not in allowed:
            return False
    return True


def _faithfulness_ids(
    config: Config, store: RunStore, cells: Sequence[Cell]
) -> dict[str, set[str] | None]:
    """The fixed retention subsample per task (same items for every cell)."""
    n = config.scoring.faithfulness_subsample
    out: dict[str, set[str] | None] = {}
    for task_name in {cell.task for cell in cells}:
        if n <= 0:
            out[task_name] = None  # all items
            continue
        try:
            from .data_build import read_items

            # Use the English item set's ids: item_ids are shared across languages.
            reference_lang = "en" if "en" in config.languages else next(iter(config.languages))
            item_ids = [item["item_id"] for item in read_items(task_name, reference_lang)]
        except FileNotFoundError:
            log.warning("cannot read item ids for %s; scoring retention on all items", task_name)
            out[task_name] = None
            continue
        out[task_name] = subsample_item_ids(item_ids, n, config.seed)
        log.info(
            "%s: retention subsample = %d items (seed=%d)",
            task_name, len(out[task_name] or []), config.seed,
        )
    return out


def _drop_duplicate_generations(frame: Any) -> Any:
    keys = ["run_id", "model", "task", "language", "condition", "item_id", "sample_index"]
    before = len(frame)
    frame = frame.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    if len(frame) != before:
        log.warning(
            "dropped %d duplicate generations (same cell+item+sample appended twice)",
            before - len(frame),
        )
    return frame


def _log_overview(frame: Any) -> None:
    grouped = frame.groupby(["model", "task", "language", "condition"], dropna=False).agg(
        n=("correct", "size"),
        acc=("correct", "mean"),
        tok=("n_completion_tokens", "mean"),
        trunc=("truncated", "mean"),
        target=("trace_target_share", "mean"),
        xfail=("extraction_failed", "mean"),
    ).reset_index()
    log_table(
        [
            [
                row["model"], row["task"], row["language"],
                CONDITION_LABELS.get(row["condition"], row["condition"]),
                int(row["n"]), f"{row['acc']:.3f}", f"{row['tok']:.0f}",
                f"{row['trunc']:.1%}", f"{row['target']:.2f}", f"{row['xfail']:.1%}",
            ]
            for _, row in grouped.iterrows()
        ],
        ["model", "task", "lang", "condition", "n", "acc", "tok", "trunc", "target%", "xfail"],
        logger_name="score",
    )


def load_results(path: Path | None = None) -> Any:
    import pandas as pd

    target = path or RESULTS_PARQUET
    if not target.exists():
        raise FileNotFoundError(f"{target} not found — run `ltd score` first")
    return pd.read_parquet(target)
