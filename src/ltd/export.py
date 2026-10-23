"""Durable archival of predictions — the artefact you keep after the GPU is gone.

``runs/<run_id>/predictions/**.jsonl`` is the working format: append-only, cheap to
resume, but scattered across hundreds of files and excluded from git. This module
turns a run into something that survives:

``exports/<run_id>/predictions.parquet``
    One row per generation, **lossless** — the full generated text, the exact
    prompt shown to the model, the decoding config, the seed, the Hub revision of
    the weights, and (when available) every score. Analysable on its own with
    nothing but pandas.
``exports/<run_id>/manifest.json``
    Counts, per-shard SHA-256 checksums, the frozen run config, git commit, model
    revisions, and the schema. This is what lets you prove, later, that the
    parquet is the same data the paper was written from.
``exports/<run_id>.tar.gz`` (with ``--archive``)
    The entire run directory, checksummed, for cold storage.

Nothing here mutates a run: export is read-only over ``runs/``.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .config import Config
from .env import REPO_ROOT
from .logging_utils import LoggedProgress, get_logger, git_sha, log_table
from .runstore import Cell, RunStore, _pred_key

log = get_logger("export")

EXPORTS_DIR = REPO_ROOT / "exports"

#: Columns that must survive archival. If one of these is missing from a shard the
#: export still succeeds but the manifest records the gap, so it is never silent.
REQUIRED_COLUMNS = [
    "item_id", "sample_index", "model", "task", "language", "condition",
    "gold", "output", "n_completion_tokens", "finish_reason", "truncated", "seed",
]
PROVENANCE_COLUMNS = [
    "model_hf_id", "model_revision", "prompt", "user_message", "system_prompt",
    "decoding", "max_new_tokens", "backend", "instruction_language", "generated_at",
]


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------


def verify_run(config: Config, run_id: str, deep: bool = False) -> dict[str, Any]:
    """Check a run's shards for corruption, duplication and ledger disagreement.

    Returns a report; ``problems`` empty means the run is safe to export.
    """
    store = RunStore(run_id)
    states = store.all_states()
    cells = store.discovered_cells()
    problems: list[str] = []
    per_cell: list[dict[str, Any]] = []
    total = 0

    with LoggedProgress(len(cells), f"verify {run_id}", unit="cell", logger_name="export") as bar:
        for cell in cells:
            path = cell.shard_path(store.run_dir)
            n_rows = 0
            seen: set[str] = set()
            duplicates = 0
            empty_outputs = 0
            bad_lines = 0

            with path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        bad_lines += 1
                        problems.append(f"{cell.key}: unparseable JSON at line {line_no}")
                        continue
                    n_rows += 1
                    key = _pred_key(str(record.get("item_id")), int(record.get("sample_index", 0)))
                    if key in seen:
                        duplicates += 1
                    seen.add(key)
                    if not (record.get("output") or "").strip():
                        empty_outputs += 1
                    if deep:
                        for column in REQUIRED_COLUMNS:
                            if column not in record:
                                problems.append(
                                    f"{cell.key}: line {line_no} missing required column {column!r}"
                                )
                                break

            if duplicates:
                problems.append(f"{cell.key}: {duplicates} duplicate (item_id, sample_index) rows")
            ledger = states.get(cell.key)
            if ledger and ledger.n_done != n_rows:
                problems.append(
                    f"{cell.key}: ledger says {ledger.n_done} done but the shard holds {n_rows}"
                )
            if ledger and ledger.status == "error":
                problems.append(f"{cell.key}: cell ended in error — {ledger.error}")

            total += n_rows
            per_cell.append({
                "cell": cell.key, "rows": n_rows, "duplicates": duplicates,
                "empty_outputs": empty_outputs, "bad_lines": bad_lines,
                "status": ledger.status if ledger else "unknown",
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            })
            bar.update(1)

    # Cells in the ledger that have no shard at all.
    for key, state in states.items():
        if key not in {cell.key for cell in cells} and state.n_done:
            problems.append(f"{key}: ledger reports {state.n_done} generations but no shard exists")

    report = {
        "run_id": run_id,
        "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_cells": len(cells),
        "n_generations": total,
        "cells": per_cell,
        "problems": problems,
        "ok": not problems,
    }
    if problems:
        log.warning("run %s: %d integrity problem(s)", run_id, len(problems))
        for problem in problems[:20]:
            log.warning("  %s", problem)
    else:
        log.info("run %s: %d generations across %d cells, no integrity problems ✓",
                 run_id, total, len(cells))
    return report


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _read_scored(store: RunStore, cell: Cell) -> dict[str, dict[str, Any]]:
    """Cached scores for a cell keyed by (item_id, sample_index), if scored."""
    stem = cell.key.replace("/", "__")
    path = store.run_dir / "scored" / f"{stem}.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[_pred_key(str(row.get("item_id")), int(row.get("sample_index", 0)))] = row
    return out


def export_run(
    config: Config,
    run_id: str,
    *,
    output_dir: Path | None = None,
    archive: bool = False,
    verify: bool = True,
) -> dict[str, Any]:
    """Write a self-contained parquet + manifest for one run."""
    import pandas as pd

    store = RunStore(run_id)
    cells = store.discovered_cells()
    if not cells:
        raise FileNotFoundError(f"run {run_id!r} has no prediction shards to export")

    target = (output_dir or EXPORTS_DIR) / run_id
    target.mkdir(parents=True, exist_ok=True)

    report = verify_run(config, run_id) if verify else {"problems": [], "cells": []}

    rows: list[dict[str, Any]] = []
    score_columns: set[str] = set()
    with LoggedProgress(len(cells), f"export {run_id}", unit="cell", logger_name="export") as bar:
        for cell in cells:
            scored = _read_scored(store, cell)
            for record in store.read_shard(cell):
                key = _pred_key(str(record.get("item_id")), int(record.get("sample_index", 0)))
                merged = dict(record)
                merged["run_id"] = run_id
                score = scored.get(key)
                if score:
                    for column, value in score.items():
                        # Never let a score overwrite the raw generation record.
                        if column not in merged:
                            merged[column] = value
                            score_columns.add(column)
                rows.append(merged)
            bar.update(1, {"cell": cell.key, "rows": len(rows)})

    frame = pd.DataFrame(rows)
    # `decoding` is a dict; parquet is happier with a JSON string.
    if "decoding" in frame.columns:
        frame["decoding"] = frame["decoding"].apply(
            lambda v: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
        )
    if "options" in frame.columns:
        frame["options"] = frame["options"].apply(
            lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
        )

    parquet_path = target / "predictions.parquet"
    frame.to_parquet(parquet_path, index=False, compression="zstd")

    missing_required = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    missing_provenance = [c for c in PROVENANCE_COLUMNS if c not in frame.columns]
    if missing_required:
        log.warning("export %s is missing required columns: %s", run_id, missing_required)
    if missing_provenance:
        log.warning(
            "export %s lacks provenance columns %s — these shards predate the "
            "prompt/revision recording, so the exact model input cannot be recovered",
            run_id, missing_provenance,
        )

    manifest = {
        "run_id": run_id,
        "exported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "n_generations": int(len(frame)),
        "n_cells": len(cells),
        "scored": bool(score_columns),
        "columns": sorted(frame.columns.tolist()),
        "missing_required_columns": missing_required,
        "missing_provenance_columns": missing_provenance,
        "models": sorted(frame["model"].dropna().unique().tolist()) if "model" in frame else [],
        "model_revisions": (
            frame.dropna(subset=["model_hf_id"])
            .groupby("model_hf_id")["model_revision"].first().to_dict()
            if "model_hf_id" in frame.columns and "model_revision" in frame.columns else {}
        ),
        "languages": sorted(frame["language"].dropna().unique().tolist()) if "language" in frame else [],
        "tasks": sorted(frame["task"].dropna().unique().tolist()) if "task" in frame else [],
        "conditions": sorted(frame["condition"].dropna().unique().tolist()) if "condition" in frame else [],
        "run_config": store.read_config(),
        "integrity": report,
        "predictions_parquet": {
            "path": parquet_path.name,
            "bytes": parquet_path.stat().st_size,
            "sha256": _sha256(parquet_path),
        },
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    log.info(
        "exported %s: %d generations, %d columns -> %s (%.1f MB)",
        run_id, len(frame), len(frame.columns), parquet_path,
        parquet_path.stat().st_size / 1e6,
    )

    if archive:
        archive_path = _archive_run(store, target)
        manifest["archive"] = {
            "path": archive_path.name,
            "bytes": archive_path.stat().st_size,
            "sha256": _sha256(archive_path),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    return manifest


def _archive_run(store: RunStore, target: Path) -> Path:
    """tar.gz the raw run directory for cold storage."""
    archive_path = target / f"{store.run_id}_raw.tar.gz"
    members = sorted(p for p in store.run_dir.rglob("*") if p.is_file())
    with LoggedProgress(len(members), f"archive {store.run_id}", unit="file",
                        logger_name="export") as bar:
        with tarfile.open(archive_path, "w:gz") as tar:
            for member in members:
                tar.add(member, arcname=str(member.relative_to(store.run_dir.parent)))
                bar.update(1)
    log.info("archived raw run -> %s (%.1f MB)", archive_path, archive_path.stat().st_size / 1e6)
    return archive_path


def export_runs(
    config: Config,
    run_ids: Sequence[str],
    *,
    output_dir: Path | None = None,
    archive: bool = False,
    verify: bool = True,
) -> list[dict[str, Any]]:
    manifests = []
    for run_id in run_ids:
        manifests.append(
            export_run(config, run_id, output_dir=output_dir, archive=archive, verify=verify)
        )
    log_table(
        [
            [
                m["run_id"], m["n_generations"], m["n_cells"],
                "yes" if m["scored"] else "no",
                len(m["integrity"].get("problems", [])),
                f"{m['predictions_parquet']['bytes'] / 1e6:.1f} MB",
            ]
            for m in manifests
        ],
        ["run", "generations", "cells", "scored", "problems", "parquet"],
        logger_name="export",
    )
    return manifests


def load_export(run_id: str, output_dir: Path | None = None) -> Any:
    """Read an exported run back — the entry point for any future analysis."""
    import pandas as pd

    path = (output_dir or EXPORTS_DIR) / run_id / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `ltd export --run-id {run_id}` first")
    return pd.read_parquet(path)


def list_exports(output_dir: Path | None = None) -> list[str]:
    base = output_dir or EXPORTS_DIR
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "predictions.parquet").exists())
