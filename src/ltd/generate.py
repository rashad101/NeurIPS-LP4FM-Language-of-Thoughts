"""The inference driver: iterate cells, generate in chunks, checkpoint constantly.

Ordering matters for cost. The model is the expensive thing to load, so the loop
is ``model -> task -> language -> condition``: each model is loaded exactly once
per invocation, and every cell it owns is finished before the next model starts.

Every chunk of ``chunk_size`` prompts is written to disk and the ledger updated
before the next chunk starts, so ``ltd generate --run-id <same>`` resumes with at
most one chunk of lost work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Sequence

from . import prompts as prompt_lib
from .backends import BaseBackend, Generation, build_backend, choose_backend
from .config import CONDITION_LABELS, Config
from .data_build import read_items
from .env import REPO_ROOT
from .logging_utils import LoggedProgress, get_logger, log_gpu_memory, log_table
from .models_fetch import model_revision
from .runstore import Cell, RunStore, _pred_key

log = get_logger("generate")


@dataclass
class GenerateSelection:
    models: list[str]
    tasks: list[str]
    languages: list[str]
    conditions: list[str]
    limit: int | None = None
    n_samples: int = 1
    backend: str = "auto"
    chunk_size: int = 64
    max_new_tokens: dict[str, int] | None = None
    #: "auto" | "subprocess" | "inprocess" — see :func:`resolve_isolation`.
    model_isolation: str = "auto"
    config_path: str | None = None
    instruction_language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": self.models,
            "tasks": self.tasks,
            "languages": self.languages,
            "conditions": self.conditions,
            "limit": self.limit,
            "n_samples": self.n_samples,
            "backend": self.backend,
            "chunk_size": self.chunk_size,
            "max_new_tokens": self.max_new_tokens,
            "model_isolation": self.model_isolation,
            "config_path": self.config_path,
            "instruction_language": self.instruction_language,
        }

    def child_argv(self, model_key: str, run_id: str) -> list[str]:
        """CLI arguments to generate exactly this selection for one model."""
        argv = [
            "generate",
            "--run-id", run_id,
            "--models", model_key,
            "--tasks", *self.tasks,
            "--langs", *self.languages,
            "--conditions", *self.conditions,
            "--backend", self.backend,
            "--chunk-size", str(self.chunk_size),
            "--n-samples", str(self.n_samples),
            # The child must never recurse into another layer of subprocesses.
            "--model-isolation", "inprocess",
        ]
        if self.limit is not None:
            argv += ["--limit", str(self.limit)]
        if self.max_new_tokens:
            # A single override was applied uniformly across tasks by the CLI.
            argv += ["--max-new-tokens", str(next(iter(self.max_new_tokens.values())))]
        if self.config_path:
            argv += ["--config", self.config_path]
        if self.instruction_language:
            argv += ["--instruction-language", self.instruction_language]
        return argv


def resolve_isolation(selection: GenerateSelection, backend_kind: str) -> str:
    """Decide whether each model gets its own process.

    ``auto`` isolates whenever more than one model is scheduled on a real GPU
    backend. Rationale: vLLM reserves ``gpu_memory_utilization`` of the card for
    the KV cache, and while in-process teardown (``VLLMBackend.shutdown``) is
    thorough, it depends on vLLM internals that shift between releases. A child
    process is the only guarantee the OS itself enforces — when it exits, every
    byte is returned, regardless of what vLLM did or did not release.
    """
    if selection.model_isolation != "auto":
        return selection.model_isolation
    if len(selection.models) > 1 and backend_kind == "vllm":
        return "subprocess"
    return "inprocess"


def plan_cells(selection: GenerateSelection) -> list[Cell]:
    """Cells in execution order (model outermost — loaded once each)."""
    cells: list[Cell] = []
    for model in selection.models:
        for task in selection.tasks:
            for language in selection.languages:
                for condition in selection.conditions:
                    cells.append(Cell(model, task, language, condition))
    return cells


def _stable_hash(text: str) -> int:
    """Deterministic seed material, stable across interpreters.

    ``hash()`` is salted for ``str``, so it cannot be used: resuming a run in a
    new process must reproduce the same seeds it would have used originally.
    """
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def build_requests(
    config: Config,
    cell: Cell,
    items: Sequence[dict[str, Any]],
    tokenizer: Any,
    n_samples: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Rendered prompts for a cell, one per (item, sample_index)."""
    task = config.task(cell.task)
    lang = config.language(cell.language)
    model = config.model(cell.model)
    selected = items[:limit] if limit else items

    requests: list[dict[str, Any]] = []
    for item in selected:
        messages = prompt_lib.build_messages(
            task=task,
            lang=lang,
            condition=cell.condition,
            item=item,
            instruction_language=config.instruction_language,
        )
        prompt = prompt_lib.render_prompt(tokenizer, model, messages)
        for sample_index in range(n_samples):
            requests.append(
                {
                    "item_id": item["item_id"],
                    "sample_index": sample_index,
                    "prompt": prompt,
                    "system_message": messages[0]["content"],
                    "user_message": messages[-1]["content"],
                    "gold": item["gold"],
                    "question": item["question"],
                    "item": item,
                }
            )
    return requests


def run_cell(
    config: Config,
    store: RunStore,
    backend: BaseBackend,
    cell: Cell,
    selection: GenerateSelection,
    outer_bar: LoggedProgress | None = None,
) -> dict[str, Any]:
    task = config.task(cell.task)
    max_new_tokens = (selection.max_new_tokens or {}).get(cell.task, task.max_new_tokens)

    items = read_items(cell.task, cell.language)
    requests = build_requests(
        config, cell, items, backend.tokenizer, selection.n_samples, selection.limit
    )
    done_keys = store.completed_item_ids(cell)
    pending = [
        request for request in requests
        if _pred_key(request["item_id"], request["sample_index"]) not in done_keys
    ]

    state = store.update_cell(
        cell,
        status="running" if pending else "done",
        n_total=len(requests),
        n_done=len(requests) - len(pending),
        extra={
            "backend": backend.name,
            "max_new_tokens": max_new_tokens,
            "degenerate": prompt_lib.degenerate_condition(cell.language, cell.condition),
        },
    )

    if not pending:
        # The outer bar's total excludes already-finished work, so nothing to add.
        log.info("%s: already complete (%d/%d) — skipping", cell.key, state.n_done, len(requests))
        return {"cell": cell.key, "generated": 0, "resumed": state.n_done, "total": len(requests)}

    log.info(
        "%s [%s]: %d to generate (%d already done), max_new_tokens=%d",
        cell.key, CONDITION_LABELS.get(cell.condition, cell.condition),
        len(pending), len(requests) - len(pending), max_new_tokens,
    )

    chunk_size = max(1, selection.chunk_size)
    started = time.time()
    generated = 0
    truncated = 0
    completion_tokens = 0
    revision = model_revision(config.model(cell.model))

    try:
        with store.open_shard(cell) as shard, LoggedProgress(
            len(pending), f"gen {cell.key}", unit="gen", logger_name="generate"
        ) as bar:
            for start in range(0, len(pending), chunk_size):
                chunk = pending[start : start + chunk_size]
                seed = _stable_hash(f"{config.seed}|{cell.key}|{start}")
                chunk_started = time.time()
                outputs = backend.generate(
                    [request["prompt"] for request in chunk], max_new_tokens, seed
                )
                records = [
                    _record(config, cell, request, output, seed, max_new_tokens,
                            backend.name, revision)
                    # strict: a backend returning fewer outputs than prompts must
                    # fail loudly, not silently drop predictions.
                    for request, output in zip(chunk, outputs, strict=True)
                ]
                shard.write_many(records)

                generated += len(records)
                truncated += sum(1 for output in outputs if output.truncated)
                completion_tokens += sum(output.n_completion_tokens for output in outputs)
                chunk_seconds = time.time() - chunk_started

                store.update_cell(
                    cell,
                    n_done=len(requests) - len(pending) + generated,
                    elapsed_s=time.time() - started,
                    extra={
                        "mean_completion_tokens": round(completion_tokens / max(generated, 1), 1),
                        "truncation_rate": round(truncated / max(generated, 1), 4),
                        "tokens_per_s": round(
                            sum(o.n_completion_tokens for o in outputs) / max(chunk_seconds, 1e-6), 1
                        ),
                    },
                )
                bar.update(
                    len(records),
                    {
                        "tok/s": f"{sum(o.n_completion_tokens for o in outputs) / max(chunk_seconds, 1e-6):.0f}",
                        "trunc": f"{truncated / max(generated, 1):.1%}",
                    },
                )
                if outer_bar:
                    outer_bar.update(len(records))
    except Exception as exc:
        store.update_cell(cell, status="error", error=f"{type(exc).__name__}: {exc}",
                          elapsed_s=time.time() - started)
        log.exception("%s failed after %d generations", cell.key, generated)
        raise

    elapsed = time.time() - started
    store.update_cell(cell, status="done", n_done=len(requests), elapsed_s=elapsed,
                      error=None)
    log.info(
        "%s: done — %d generated in %.1fs (%.1f gen/s, mean %.0f completion tokens, %.1f%% truncated)",
        cell.key, generated, elapsed, generated / max(elapsed, 1e-6),
        completion_tokens / max(generated, 1), 100 * truncated / max(generated, 1),
    )
    return {
        "cell": cell.key,
        "generated": generated,
        "resumed": len(requests) - len(pending),
        "total": len(requests),
        "elapsed_s": elapsed,
        "truncation_rate": truncated / max(generated, 1),
        "mean_completion_tokens": completion_tokens / max(generated, 1),
    }


def _record(
    config: Config,
    cell: Cell,
    request: dict[str, Any],
    output: Generation,
    seed: int,
    max_new_tokens: int,
    backend_name: str,
    model_revision: str | None = None,
) -> dict[str, Any]:
    """One prediction row.

    Deliberately self-describing and *lossless*: a shard must remain analysable on
    its own years from now, without the config, the item sets, or the tokenizer
    that produced it. That is why the fully rendered ``prompt`` is stored verbatim
    rather than recomputed on demand — a later transformers/chat-template version
    would silently render it differently, and there would be no way to tell.
    """
    record = {
        "item_id": request["item_id"],
        "sample_index": request["sample_index"],
        "model": cell.model,
        "model_hf_id": config.model(cell.model).hf_id,
        "model_revision": model_revision,
        "task": cell.task,
        "language": cell.language,
        "condition": cell.condition,
        "gold": request["gold"],
        "output": output.text,
        "n_prompt_tokens": output.n_prompt_tokens,
        "n_completion_tokens": output.n_completion_tokens,
        "finish_reason": output.finish_reason,
        "truncated": output.truncated,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "decoding": config.model(cell.model).decoding,
        "backend": backend_name,
        "instruction_language": config.instruction_language,
        "question": request["question"],
        # Exactly what the model was shown, at both levels of assembly.
        "system_prompt": request["system_message"],
        "user_message": request["user_message"],
        "prompt": request["prompt"],
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    item = request["item"]
    for key in ("subject", "subject_category", "sample_id", "question_en", "options"):
        if key in item:
            record[key] = item[key]
    return record


def run(
    config: Config,
    store: RunStore,
    selection: GenerateSelection,
) -> list[dict[str, Any]]:
    cells = plan_cells(selection)
    backend_kind = choose_backend(selection.backend)

    # Pre-count total work so the outer bar is meaningful.
    totals: dict[str, int] = {}
    for task in selection.tasks:
        for language in selection.languages:
            n = len(read_items(task, language))
            if selection.limit:
                n = min(n, selection.limit)
            totals[f"{task}/{language}"] = n * selection.n_samples
    total_work = sum(
        totals[f"{cell.task}/{cell.language}"] for cell in cells
    )

    already = 0
    for cell in cells:
        already += len(store.completed_item_ids(cell))
    remaining = max(total_work - already, 0)

    log.info(
        "plan: %d cells, %d generations total, %d already on disk, %d remaining (backend=%s)",
        len(cells), total_work, already, remaining, backend_kind,
    )
    log_table(
        [[cell.model, cell.task, cell.language, CONDITION_LABELS[cell.condition],
          totals[f"{cell.task}/{cell.language}"]] for cell in cells],
        ["model", "task", "lang", "condition", "n"],
        logger_name="generate",
    )

    store.write_config(
        {
            "selection": selection.to_dict(),
            "config": config.to_dict(),
            "backend_resolved": backend_kind,
            "planned_generations": total_work,
        }
    )

    isolation = resolve_isolation(selection, backend_kind)
    log.info(
        "model isolation: %s (%d model(s) will be loaded strictly one at a time)",
        isolation, len(selection.models),
    )

    if isolation == "subprocess":
        return _run_isolated(config, store, selection, cells, totals)

    results: list[dict[str, Any]] = []
    outer = LoggedProgress(remaining, "all cells", unit="gen", logger_name="generate")
    baseline_mb = log_gpu_memory("before any model")
    try:
        for position, model_key in enumerate(selection.models):
            model_cells = [cell for cell in cells if cell.model == model_key]
            if all(
                len(store.completed_item_ids(cell)) >= totals[f"{cell.task}/{cell.language}"]
                for cell in model_cells
            ):
                log.info("model %s: all cells complete — not loading weights", model_key)
                for cell in model_cells:
                    store.update_cell(cell, status="done")
                continue

            model = config.model(model_key)
            log.info(
                "loading model %d/%d: %s (%s)",
                position + 1, len(selection.models), model_key, model.hf_id,
            )
            backend = build_backend(model, config.generation, selection.backend)
            log_gpu_memory(f"after loading {model_key}")
            try:
                for cell in model_cells:
                    results.append(
                        run_cell(config, store, backend, cell, selection, outer_bar=outer)
                    )
            finally:
                log.info("releasing model %s", model_key)
                backend.shutdown()
                del backend
                _reclaim(baseline_mb, model_key, more_models=position + 1 < len(selection.models))
    finally:
        outer.close()

    _log_summary(results)
    return results


def _reclaim(baseline_mb: int | None, model_key: str, more_models: bool) -> None:
    """Force a collection after a model is released and check the memory came back.

    A silent teardown failure would surface as an OOM several minutes later while
    loading the next model, which is a miserable thing to debug at 3am. Check it
    here, immediately, and say so.
    """
    import gc

    gc.collect()
    after_mb = log_gpu_memory(f"after releasing {model_key}")
    if baseline_mb is None or after_mb is None:
        return

    leaked_mb = after_mb - baseline_mb
    if leaked_mb > _LEAK_TOLERANCE_MB:
        message = (
            f"{leaked_mb / 1024:.1f} GiB of GPU memory was NOT released after {model_key} "
            f"(baseline {baseline_mb / 1024:.1f} GiB, now {after_mb / 1024:.1f} GiB)"
        )
        if more_models:
            log.error(
                "%s — the next model will probably OOM. Rerun with "
                "`--model-isolation subprocess` to give each model its own process.",
                message,
            )
        else:
            log.warning("%s (no further models to load, so this run is unaffected)", message)
    else:
        log.info("GPU memory reclaimed after %s ✓", model_key)


#: Slack for allocator fragmentation and the CUDA context itself.
_LEAK_TOLERANCE_MB = 2048


def _run_isolated(
    config: Config,
    store: RunStore,
    selection: GenerateSelection,
    cells: Sequence[Cell],
    totals: dict[str, int],
) -> list[dict[str, Any]]:
    """Run each model in its own child process.

    The child does the real work; when it exits the OS reclaims every byte of GPU
    memory it held, no matter what the inference library left behind. Because all
    state lives in ``runs/<run_id>/`` (append-only shards plus the ledger), the
    children coordinate through the filesystem and the whole thing stays resumable:
    a child that dies is retried by rerunning the same command.
    """
    import subprocess
    import sys

    failures: list[tuple[str, int]] = []
    log_gpu_memory("before any model")

    for position, model_key in enumerate(selection.models):
        model_cells = [cell for cell in cells if cell.model == model_key]
        if all(
            len(store.completed_item_ids(cell)) >= totals[f"{cell.task}/{cell.language}"]
            for cell in model_cells
        ):
            log.info("model %s: all cells complete — skipping", model_key)
            for cell in model_cells:
                store.update_cell(cell, status="done")
            continue

        argv = [sys.executable, "-m", "ltd", *selection.child_argv(model_key, store.run_id)]
        log.info(
            "=== model %d/%d: %s — launching isolated worker ===",
            position + 1, len(selection.models), model_key,
        )
        log.debug("child argv: %s", " ".join(argv))

        completed = subprocess.run(argv, cwd=str(REPO_ROOT), check=False)
        if completed.returncode != 0:
            failures.append((model_key, completed.returncode))
            log.error(
                "model %s: worker exited %d — its finished generations are safe on disk; "
                "rerun the same command to resume it",
                model_key, completed.returncode,
            )
        else:
            log.info("model %s: worker finished cleanly", model_key)
        log_gpu_memory(f"after {model_key} worker exit")

    if failures:
        raise RuntimeError(
            "isolated worker(s) failed: "
            + ", ".join(f"{key} (exit {code})" for key, code in failures)
            + " — rerun the same command with the same --run-id to resume"
        )

    # The children wrote the shards; report from the ledger they updated. Restrict
    # to the cells this invocation asked for — the run directory may also hold
    # cells from earlier invocations with a different selection.
    store.reload_state()
    states = store.all_states()
    wanted = {cell.key for cell in cells}
    results = [
        {
            "cell": key,
            "generated": state.n_done,
            "resumed": 0,
            "total": state.n_total,
            "elapsed_s": state.elapsed_s,
            "truncation_rate": float(state.extra.get("truncation_rate", 0.0) or 0.0),
            "mean_completion_tokens": float(state.extra.get("mean_completion_tokens", 0.0) or 0.0),
        }
        for key, state in sorted(states.items())
        if key in wanted
    ]
    _log_summary(results)
    return results


def _log_summary(results: Iterable[dict[str, Any]]) -> None:
    rows = [
        [
            result["cell"],
            result["generated"],
            result["resumed"],
            result["total"],
            f"{result.get('elapsed_s', 0):.0f}s",
            f"{result.get('truncation_rate', 0):.1%}",
            f"{result.get('mean_completion_tokens', 0):.0f}",
        ]
        for result in results
    ]
    if rows:
        log_table(
            rows,
            ["cell", "new", "resumed", "total", "elapsed", "trunc", "mean_tok"],
            logger_name="generate",
        )
