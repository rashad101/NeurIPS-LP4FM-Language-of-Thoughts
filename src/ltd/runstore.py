"""Run directories, append-only prediction shards, and the resume ledger.

Layout::

    runs/<run_id>/
        config.json                  frozen config + CLI selection + git sha
        state.json                   per-cell status ledger (atomic writes)
        predictions/<model>/<task>/<lang>/<condition>.jsonl

A **cell** is one (model, task, language, condition) combination — the unit of
work, of resumption, and of run separation. Predictions are appended one JSON
object per line and flushed (``flush`` + ``fsync``) after every chunk, so a run
killed mid-flight loses at most the in-flight chunk.

Resume works by reading the item ids already present in a shard. A trailing
partial line (the classic kill-during-write artefact) is detected and truncated
rather than crashing the next run.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from .env import RUNS_DIR
from .logging_utils import get_logger, git_sha

log = get_logger("runstore")


def new_run_id(prefix: str = "run") -> str:
    """Timestamped run id — the 'separable by time' half of run separation."""
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


@dataclass(frozen=True)
class Cell:
    model: str
    task: str
    language: str
    condition: str

    @property
    def key(self) -> str:
        return f"{self.model}/{self.task}/{self.language}/{self.condition}"

    def shard_path(self, run_dir: Path) -> Path:
        return (
            run_dir / "predictions" / self.model / self.task / self.language
            / f"{self.condition}.jsonl"
        )


@dataclass
class CellState:
    status: str = "pending"  # pending | running | done | error
    n_done: int = 0
    n_total: int = 0
    updated: str = ""
    elapsed_s: float = 0.0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


class RunStore:
    """Everything on disk for one run id."""

    def __init__(self, run_id: str, root: Path | None = None) -> None:
        self.run_id = run_id
        self.run_dir = (root or RUNS_DIR) / run_id
        self.predictions_dir = self.run_dir / "predictions"
        self.state_path = self.run_dir / "state.json"
        self.config_path = self.run_dir / "config.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, CellState] = self._load_state()

    # -- config -----------------------------------------------------------
    def write_config(self, payload: dict[str, Any]) -> None:
        enriched = dict(payload)
        enriched.setdefault("run_id", self.run_id)
        enriched["git_sha"] = git_sha()
        enriched["written_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        if self.config_path.exists():
            try:
                previous = json.loads(self.config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                previous = {}
            history = previous.pop("history", [])
            history.append({k: previous.get(k) for k in ("written_at", "git_sha", "selection")})
            enriched["history"] = history
        _atomic_write_json(self.config_path, enriched)
        log.info("run config -> %s", self.config_path)

    def read_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    # -- state ledger -----------------------------------------------------
    def _load_state(self) -> dict[str, CellState]:
        if not self.state_path.exists():
            return {}
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("state.json unreadable; starting a fresh ledger (shards are intact)")
            return {}
        return {key: CellState(**value) for key, value in raw.get("cells", {}).items()}

    def flush_state(self) -> None:
        _atomic_write_json(
            self.state_path,
            {
                "run_id": self.run_id,
                "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
                "cells": {key: asdict(value) for key, value in self._state.items()},
            },
        )

    def cell_state(self, cell: Cell) -> CellState:
        return self._state.setdefault(cell.key, CellState())

    def update_cell(self, cell: Cell, **fields: Any) -> CellState:
        state = self.cell_state(cell)
        for key, value in fields.items():
            if key == "extra" and isinstance(value, dict):
                state.extra.update(value)
            else:
                setattr(state, key, value)
        state.updated = datetime.now().astimezone().isoformat(timespec="seconds")
        self.flush_state()
        return state

    def all_states(self) -> dict[str, CellState]:
        return dict(self._state)

    def reload_state(self) -> dict[str, CellState]:
        """Re-read the ledger from disk.

        Needed when another process (an isolated per-model worker) has been
        updating the same run directory.
        """
        self._state = self._load_state()
        return dict(self._state)

    # -- prediction shards ------------------------------------------------
    def completed_item_ids(self, cell: Cell) -> set[str]:
        """Item ids already written, repairing a truncated final line if needed."""
        path = cell.shard_path(self.run_dir)
        if not path.exists():
            return set()

        done: set[str] = set()
        good_bytes = 0
        with path.open("rb") as fh:
            for raw in fh:
                if not raw.endswith(b"\n"):
                    log.warning(
                        "%s: last line is incomplete (%d bytes) — truncating and resuming",
                        path.name, len(raw),
                    )
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("%s: dropping unparseable line at byte %d", path.name, good_bytes)
                        break
                    item_id = record.get("item_id")
                    sample = record.get("sample_index", 0)
                    if item_id is not None:
                        done.add(_pred_key(str(item_id), int(sample)))
                good_bytes += len(raw)

        actual = path.stat().st_size
        if good_bytes != actual:
            with path.open("r+b") as fh:
                fh.truncate(good_bytes)
            log.info("%s: truncated %d trailing bytes", path.name, actual - good_bytes)
        return done

    def open_shard(self, cell: Cell) -> "ShardWriter":
        return ShardWriter(cell.shard_path(self.run_dir))

    def read_shard(self, cell: Cell) -> Iterator[dict[str, Any]]:
        path = cell.shard_path(self.run_dir)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    log.warning("skipping unparseable line in %s", path)

    def discovered_cells(self) -> list[Cell]:
        """Every cell with a shard on disk (used by score/analyze)."""
        cells: list[Cell] = []
        if not self.predictions_dir.exists():
            return cells
        for shard in sorted(self.predictions_dir.rglob("*.jsonl")):
            relative = shard.relative_to(self.predictions_dir)
            if len(relative.parts) != 4:
                log.warning("ignoring unexpected shard path %s", shard)
                continue
            model, task, language, filename = relative.parts
            cells.append(Cell(model=model, task=task, language=language,
                              condition=Path(filename).stem))
        return cells


def _pred_key(item_id: str, sample_index: int) -> str:
    return f"{item_id}#{sample_index}"


class ShardWriter:
    """Append-only JSONL writer that fsyncs on every flush."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("a", encoding="utf-8")

    def write_many(self, records: Iterable[dict[str, Any]]) -> int:
        count = 0
        for record in records:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
        self.flush()
        return count

    def flush(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._fh.close()

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def list_runs(root: Path | None = None) -> list[str]:
    base = root or RUNS_DIR
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith("."))


def latest_run(root: Path | None = None) -> str | None:
    runs = list_runs(root)
    return runs[-1] if runs else None


def resolve_run_ids(requested: Iterable[str] | None, root: Path | None = None) -> list[str]:
    """Resolve run selectors: explicit ids, ``all``, or ``latest`` (the default)."""
    available = list_runs(root)
    if not requested:
        newest = latest_run(root)
        if newest is None:
            raise FileNotFoundError(
                f"no runs found under {root or RUNS_DIR}; run `ltd generate` first"
            )
        return [newest]
    out: list[str] = []
    for item in requested:
        if item == "all":
            out.extend(available)
        elif item == "latest":
            newest = latest_run(root)
            if newest:
                out.append(newest)
        else:
            if item not in available:
                raise FileNotFoundError(f"run {item!r} not found; available: {available}")
            out.append(item)
    seen: set[str] = set()
    unique = []
    for item in out:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique
