"""``ltd`` command line interface.

Import order matters: :mod:`ltd.env` runs first (via the package ``__init__``) so
that every library cache is inside the repo before torch/transformers/vllm load.
Heavy imports therefore live *inside* the command functions, not at module top.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import env  # noqa: F401  (must be first: pins caches into the repo)
from .config import CONDITION_LABELS, Config, load_config
from .env import ARTIFACTS_DIR, FIGURES_DIR, REPO_ROOT, RESULTS_DIR, TABLES_DIR
from .logging_utils import get_logger, log_environment, setup_logging
from .runstore import RunStore, list_runs, new_run_id, resolve_run_ids

log = get_logger("cli")


# ---------------------------------------------------------------------------
# Argument plumbing
# ---------------------------------------------------------------------------


def _add_selection_args(parser: argparse.ArgumentParser, *, with_conditions: bool = True) -> None:
    parser.add_argument("--models", nargs="+", metavar="KEY",
                        help="model keys from the config, or 'all' (default: config default_models)")
    parser.add_argument("--tasks", nargs="+", metavar="TASK",
                        help="mgsm and/or gmmlu, or 'all'")
    parser.add_argument("--langs", nargs="+", metavar="LANG",
                        help="language codes (en hi bn te sw de es), or 'all'")
    if with_conditions:
        parser.add_argument("--conditions", nargs="+", metavar="COND",
                            choices=list(CONDITION_LABELS) + ["all"],
                            help="thinking-language policies: target en_cot free mixed, or 'all'")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None,
                        help="path to a YAML config (default: configs/default.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug-level console output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ltd",
        description=(
            "Unifying demonstration for 'The Language of Thought Is a Design Choice'. "
            "All models, datasets and caches stay inside the repository."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- doctor -----------------------------------------------------------
    doctor = subparsers.add_parser("doctor", help="report environment, GPU, cache locations")
    _add_common_args(doctor)

    # -- fetch-data -------------------------------------------------------
    fetch_data = subparsers.add_parser("fetch-data", help="download and build the item sets")
    _add_common_args(fetch_data)
    _add_selection_args(fetch_data, with_conditions=False)
    fetch_data.add_argument("--force", action="store_true", help="rebuild even if files exist")
    fetch_data.add_argument("--verify", action="store_true",
                            help="check cross-language item/gold alignment after building")

    # -- fetch-models -----------------------------------------------------
    fetch_models = subparsers.add_parser("fetch-models", help="download weights into assets/models")
    _add_common_args(fetch_models)
    _add_selection_args(fetch_models, with_conditions=False)
    fetch_models.add_argument("--force", action="store_true", help="re-download")

    # -- prompts ----------------------------------------------------------
    prompts_cmd = subparsers.add_parser("prompts", help="print/export the prompt inventory")
    _add_common_args(prompts_cmd)
    prompts_cmd.add_argument("--instruction-language", choices=["en", "native"], default=None)
    prompts_cmd.add_argument("--show", action="store_true", help="also print a rendered example")
    prompts_cmd.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "prompts.json")

    # -- generate ---------------------------------------------------------
    generate = subparsers.add_parser(
        "generate",
        help="run inference (resumable: pass the same --run-id to continue)",
    )
    _add_common_args(generate)
    _add_selection_args(generate)
    generate.add_argument("--run-id", default=None,
                          help="run directory name; default is a fresh timestamped id")
    generate.add_argument("--resume", metavar="RUN_ID", default=None,
                          help="alias for --run-id on an existing run; 'latest' picks the newest")
    generate.add_argument("--backend", choices=["auto", "vllm", "hf"], default=None)
    generate.add_argument("--limit", type=int, default=None,
                          help="cap items per cell (dry runs)")
    generate.add_argument("--n-samples", type=int, default=None)
    generate.add_argument("--chunk-size", type=int, default=None,
                          help="generations per checkpoint flush")
    generate.add_argument("--max-new-tokens", type=int, default=None,
                          help="override the per-task generation budget")
    generate.add_argument("--instruction-language", choices=["en", "native"], default=None)
    generate.add_argument(
        "--model-isolation", choices=["auto", "subprocess", "inprocess"], default=None,
        help="how to keep models from co-residing in GPU memory: 'subprocess' gives "
             "each model its own process (the OS guarantees a full release on exit), "
             "'inprocess' tears the engine down in place, 'auto' (default) uses "
             "subprocess when more than one model runs on vLLM",
    )

    # -- status -----------------------------------------------------------
    status = subparsers.add_parser("status", help="show per-cell progress of a run")
    _add_common_args(status)
    status.add_argument("--run-id", nargs="+", default=None,
                        help="run ids, 'latest' (default) or 'all'")
    status.add_argument("--list", action="store_true", help="just list run ids")

    # -- score ------------------------------------------------------------
    score = subparsers.add_parser("score", help="extract answers and metrics -> results.parquet")
    _add_common_args(score)
    _add_selection_args(score)
    score.add_argument("--run-id", nargs="+", default=None,
                       help="run ids to score, 'latest' (default) or 'all'")
    score.add_argument("--force", action="store_true", help="ignore the per-cell score cache")
    score.add_argument("--out", type=Path, default=None,
                       help="parquet path (default: results/results.parquet)")

    # -- fertility --------------------------------------------------------
    fertility = subparsers.add_parser("fertility", help="measure per-language tokenizer fertility")
    _add_common_args(fertility)
    _add_selection_args(fertility, with_conditions=False)

    # -- analyze ----------------------------------------------------------
    analyze = subparsers.add_parser("analyze", help="hypothesis tests and tables")
    _add_common_args(analyze)
    analyze.add_argument("--results", type=Path, default=None,
                         help="parquet to analyse (default: results/results.parquet)")
    analyze.add_argument("--skip-fertility", action="store_true",
                         help="do not measure fertility (needs tokenizers on disk)")

    # -- figures ----------------------------------------------------------
    figures = subparsers.add_parser("figures", help="build the four figures")
    _add_common_args(figures)
    figures.add_argument("--results", type=Path, default=None)
    figures.add_argument("--which", nargs="+", choices=["fig1", "fig2", "fig3", "fig4"],
                         default=None)
    figures.add_argument("--out", type=Path, default=None)

    # -- policy-card ------------------------------------------------------
    card = subparsers.add_parser("policy-card", help="emit the disclosure artefacts")
    _add_common_args(card)
    card.add_argument("--results", type=Path, default=None)
    card.add_argument("--template-only", action="store_true")
    card.add_argument("--out", type=Path, default=None)

    # -- export -----------------------------------------------------------
    export = subparsers.add_parser(
        "export",
        help="archive predictions to a self-contained parquet + manifest (do this before deleting runs/)",
    )
    _add_common_args(export)
    export.add_argument("--run-id", nargs="+", default=None,
                        help="run ids to export, 'latest' (default) or 'all'")
    export.add_argument("--archive", action="store_true",
                        help="also tar.gz the raw run directory for cold storage")
    export.add_argument("--no-verify", action="store_true",
                        help="skip the integrity check (not recommended)")
    export.add_argument("--out", type=Path, default=None,
                        help="destination directory (default: exports/)")
    export.add_argument("--list", action="store_true", help="list existing exports")

    # -- verify -----------------------------------------------------------
    verify = subparsers.add_parser("verify", help="check prediction shards for corruption")
    _add_common_args(verify)
    verify.add_argument("--run-id", nargs="+", default=None)
    verify.add_argument("--deep", action="store_true",
                        help="also check every row for the required columns")

    # -- report -----------------------------------------------------------
    report = subparsers.add_parser("report", help="score + fertility + analyze + figures + card")
    _add_common_args(report)
    _add_selection_args(report)
    report.add_argument("--run-id", nargs="+", default=None)
    report.add_argument("--force", action="store_true")
    report.add_argument("--no-export", action="store_true",
                        help="skip archiving predictions (export runs by default)")

    return parser


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace, config: Config) -> int:
    import platform

    from .backends import choose_backend, vllm_available
    from .data_build import processed_path
    from .env import escaped_home_report
    from .models_fetch import is_downloaded, local_model_path

    log.info("repo root: %s", REPO_ROOT)
    log.info("config: %s", config.source_path)
    log.info("python: %s (%s)", sys.version.split()[0], platform.platform())
    log.info("vLLM available: %s -> backend '%s'", vllm_available(), choose_backend("auto"))

    escaped = escaped_home_report()
    if escaped:
        log.warning("these cache vars resolve OUTSIDE the repo: %s", escaped)
    else:
        log.info("all cache environment variables resolve inside the repo ✓")

    for key in ("torch", "transformers", "datasets", "vllm", "py3langid", "statsmodels"):
        try:
            module = __import__(key)
            log.info("  %-13s %s", key, getattr(module, "__version__", "?"))
        except ImportError:
            log.warning("  %-13s NOT INSTALLED", key)

    log.info("--- data ---")
    for task_name in config.tasks:
        for lang_code in config.default_languages:
            path = processed_path(task_name, lang_code)
            if path.exists():
                n = sum(1 for line in path.open(encoding="utf-8") if line.strip())
                log.info("  %-6s %-3s %4d items", task_name, lang_code, n)
            else:
                log.warning("  %-6s %-3s MISSING (run `ltd fetch-data`)", task_name, lang_code)

    log.info("--- models ---")
    for key in config.default_models:
        spec = config.model(key)
        log.info(
            "  %-14s %-46s %s", key, spec.hf_id,
            "present" if is_downloaded(spec) else f"absent ({local_model_path(spec)})",
        )

    log.info("--- runs ---")
    runs = list_runs()
    log.info("  %d run(s): %s", len(runs), ", ".join(runs[-5:]) if runs else "none")
    return 0


def cmd_fetch_data(args: argparse.Namespace, config: Config) -> int:
    from .data_build import build_all, verify_alignment

    summary = build_all(config, args.tasks, args.langs, force=args.force)
    for task_name, per_lang in summary.items():
        log.info("%s: %s", task_name, per_lang)

    if args.verify:
        problems = verify_alignment(config, args.tasks, args.langs)
        if problems:
            for problem in problems[:40]:
                log.error("alignment: %s", problem)
            log.error("%d alignment problem(s) found", len(problems))
            return 1
        log.info("alignment verified: item ids and golds agree across languages ✓")
    return 0


def cmd_fetch_models(args: argparse.Namespace, config: Config) -> int:
    from .models_fetch import download_all

    paths = download_all(config, args.models, force=args.force)
    for key, path in paths.items():
        log.info("%s -> %s", key, path)
    return 0


def cmd_prompts(args: argparse.Namespace, config: Config) -> int:
    from .prompts import build_user_message, prompt_inventory

    instruction_language = args.instruction_language or config.instruction_language
    rows = prompt_inventory(config, instruction_language)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %d prompt blocks -> %s", len(rows), args.out)

    if args.show:
        from .data_build import read_items

        for task_name in config.default_tasks:
            task = config.task(task_name)
            for lang_code in config.default_languages:
                try:
                    items = read_items(task_name, lang_code)
                except FileNotFoundError:
                    log.warning("no items for %s/%s; run `ltd fetch-data`", task_name, lang_code)
                    continue
                for condition in config.conditions:
                    message = build_user_message(
                        task=task, lang=config.language(lang_code), condition=condition,
                        item=items[0], instruction_language=instruction_language,
                    )
                    print(f"\n{'=' * 78}\n{task_name} / {lang_code} / {condition}\n{'=' * 78}")
                    print(message)
    return 0


def cmd_generate(args: argparse.Namespace, config: Config) -> int:
    from .generate import GenerateSelection, run

    if args.instruction_language:
        config.instruction_language = args.instruction_language

    run_id = args.run_id or args.resume
    if run_id in {"latest", None} and args.resume:
        resolved = resolve_run_ids([args.resume])
        run_id = resolved[0]
    if not run_id:
        run_id = new_run_id()
    setup_logging("generate", run_id, args.verbose)
    log_environment("generate", {"run_id": run_id})

    generation = config.generation
    selection = GenerateSelection(
        models=config.resolve_models(args.models),
        tasks=config.resolve_tasks(args.tasks),
        languages=config.resolve_languages(args.langs),
        conditions=config.resolve_conditions(args.conditions),
        limit=args.limit,
        n_samples=args.n_samples or generation.n_samples,
        backend=args.backend or generation.backend,
        chunk_size=args.chunk_size or generation.chunk_size,
        max_new_tokens=(
            {task: args.max_new_tokens for task in config.tasks} if args.max_new_tokens else None
        ),
        model_isolation=args.model_isolation or generation.model_isolation,
        config_path=str(args.config) if args.config else None,
        instruction_language=args.instruction_language,
    )
    store = RunStore(run_id)
    log.info("run_id=%s  dir=%s", run_id, store.run_dir)
    run(config, store, selection)
    log.info("run %s complete. Next: ltd score --run-id %s", run_id, run_id)
    return 0


def cmd_status(args: argparse.Namespace, config: Config) -> int:
    from .logging_utils import log_table

    runs = list_runs()
    if args.list or not runs:
        log.info("%d run(s):", len(runs))
        for run_id in runs:
            log.info("  %s", run_id)
        return 0

    for run_id in resolve_run_ids(args.run_id):
        store = RunStore(run_id)
        states = store.all_states()
        if not states:
            log.warning("run %s: no ledger yet", run_id)
            continue
        rows = []
        totals = {"done": 0, "total": 0}
        for key in sorted(states):
            state = states[key]
            totals["done"] += state.n_done
            totals["total"] += state.n_total
            rows.append([
                key, state.status, f"{state.n_done}/{state.n_total}",
                f"{state.n_done / state.n_total:.0%}" if state.n_total else "-",
                f"{state.elapsed_s:.0f}s",
                state.extra.get("mean_completion_tokens", "-"),
                state.extra.get("truncation_rate", "-"),
                (state.error or "")[:40],
            ])
        log.info("=== run %s (%s) ===", run_id, store.run_dir)
        log_table(rows, ["cell", "status", "done", "%", "elapsed", "mean_tok", "trunc", "error"],
                  logger_name="cli")
        pct = totals["done"] / totals["total"] if totals["total"] else 0
        log.info("overall: %d/%d generations (%.1f%%)", totals["done"], totals["total"], 100 * pct)
        errored = [key for key, state in states.items() if state.status == "error"]
        if errored:
            log.warning("%d cell(s) in error: %s", len(errored), ", ".join(errored))
            log.warning("re-run `ltd generate --run-id %s ...` to resume them", run_id)
    return 0


def cmd_score(args: argparse.Namespace, config: Config) -> int:
    from .score import score_runs

    run_ids = resolve_run_ids(args.run_id)
    log.info("scoring runs: %s", ", ".join(run_ids))
    frame = score_runs(
        config, run_ids,
        models=args.models, tasks=args.tasks, languages=args.langs,
        conditions=args.conditions, force=args.force, output=args.out,
    )
    log.info("scored %d generations", len(frame))
    return 0


def cmd_fertility(args: argparse.Namespace, config: Config) -> int:
    from .fertility import measure, save

    rows = measure(config, args.models, args.tasks, args.langs)
    save(rows)
    return 0


def _load_fertility_or_measure(config: Config, skip: bool) -> Any:
    from . import fertility as fertility_mod

    table = fertility_mod.load()
    if table.empty and not skip:
        log.info("no fertility table yet — measuring now")
        try:
            rows = fertility_mod.measure(config)
            fertility_mod.save(rows)
            table = fertility_mod.load()
        except Exception:
            log.exception("fertility measurement failed; continuing without the cost mechanism axis")
    return table


def cmd_analyze(args: argparse.Namespace, config: Config) -> int:
    from . import analyze as analyze_mod
    from .score import load_results

    frame = load_results(args.results)
    table = _load_fertility_or_measure(config, args.skip_fertility)
    analyze_mod.run(frame, config, table)
    log.info("tables -> %s", TABLES_DIR)
    return 0


def cmd_figures(args: argparse.Namespace, config: Config) -> int:
    from . import analyze as analyze_mod
    from . import figures as figures_mod
    from .score import load_results

    frame = load_results(args.results)
    table = _load_fertility_or_measure(config, skip=True)
    cells = analyze_mod.cell_table(frame, config)
    h5 = analyze_mod.h5_retention(frame, config)
    paths = figures_mod.build_all(
        frame, cells, h5, config, table, output_dir=args.out or FIGURES_DIR, which=args.which
    )
    log.info("%d figure file(s) written to %s", len(paths), args.out or FIGURES_DIR)
    return 0


def cmd_policy_card(args: argparse.Namespace, config: Config) -> int:
    from . import analyze as analyze_mod
    from . import policy_card as card_mod
    from .score import load_results

    card_mod.write_template()
    if args.template_only:
        return 0
    frame = load_results(args.results)
    table = _load_fertility_or_measure(config, skip=True)
    cells = analyze_mod.cell_table(frame, config)
    card_mod.build_filled(cells, config, table, output_dir=args.out)
    return 0


def cmd_export(args: argparse.Namespace, config: Config) -> int:
    from .export import EXPORTS_DIR, export_runs, list_exports

    if args.list:
        exports = list_exports(args.out)
        log.info("%d export(s) under %s:", len(exports), args.out or EXPORTS_DIR)
        for name in exports:
            log.info("  %s", name)
        return 0

    run_ids = resolve_run_ids(args.run_id)
    manifests = export_runs(
        config, run_ids,
        output_dir=args.out, archive=args.archive, verify=not args.no_verify,
    )
    problems = sum(len(m["integrity"].get("problems", [])) for m in manifests)
    total = sum(m["n_generations"] for m in manifests)
    log.info("exported %d generations from %d run(s) -> %s",
             total, len(manifests), args.out or EXPORTS_DIR)
    if problems:
        log.warning("%d integrity problem(s) recorded in the manifests — review before publishing",
                    problems)
        return 1
    return 0


def cmd_verify(args: argparse.Namespace, config: Config) -> int:
    from .export import verify_run

    run_ids = resolve_run_ids(args.run_id)
    failed = 0
    for run_id in run_ids:
        report = verify_run(config, run_id, deep=args.deep)
        if not report["ok"]:
            failed += 1
    if failed:
        log.error("%d of %d run(s) have integrity problems", failed, len(run_ids))
        return 1
    log.info("all %d run(s) verified ✓", len(run_ids))
    return 0


def cmd_report(args: argparse.Namespace, config: Config) -> int:
    from . import analyze as analyze_mod
    from . import figures as figures_mod
    from . import policy_card as card_mod
    from .fertility import load as load_fertility
    from .fertility import measure, save
    from .score import score_runs

    run_ids = resolve_run_ids(args.run_id)
    frame = score_runs(
        config, run_ids,
        models=args.models, tasks=args.tasks, languages=args.langs,
        conditions=args.conditions, force=args.force,
    )
    try:
        save(measure(config, args.models, args.tasks, args.langs))
    except Exception:
        log.exception("fertility measurement failed; the cost-mechanism panels will be empty")
    table = load_fertility()

    result = analyze_mod.run(frame, config, table)
    figures_mod.build_all(frame, result.cells, result.h5, config, table)
    card_mod.write_template()
    card_mod.build_filled(result.cells, config, table)

    # Archive the raw predictions by default: `runs/` is gitignored working state,
    # `exports/` is what survives.
    if not args.no_export:
        from .export import EXPORTS_DIR, export_runs

        try:
            export_runs(config, run_ids)
            log.info("  exports   %s", EXPORTS_DIR)
        except Exception:
            log.exception("export failed — predictions are still intact under runs/")

    log.info("report complete:")
    log.info("  results   %s", RESULTS_DIR / "results.parquet")
    log.info("  analysis  %s", RESULTS_DIR / "analysis.json")
    log.info("  tables    %s", TABLES_DIR)
    log.info("  figures   %s", FIGURES_DIR)
    log.info("  artifacts %s", ARTIFACTS_DIR)
    return 0


COMMANDS = {
    "doctor": cmd_doctor,
    "fetch-data": cmd_fetch_data,
    "fetch-models": cmd_fetch_models,
    "prompts": cmd_prompts,
    "generate": cmd_generate,
    "status": cmd_status,
    "score": cmd_score,
    "fertility": cmd_fertility,
    "analyze": cmd_analyze,
    "figures": cmd_figures,
    "policy-card": cmd_policy_card,
    "export": cmd_export,
    "verify": cmd_verify,
    "report": cmd_report,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # `generate` sets up logging itself (it knows the run id).
    if args.command != "generate":
        run_hint = None
        raw = getattr(args, "run_id", None)
        if isinstance(raw, list) and raw:
            run_hint = raw[0]
        elif isinstance(raw, str):
            run_hint = raw
        setup_logging(args.command, run_hint, getattr(args, "verbose", False))
        log_environment(args.command)

    config = load_config(args.config)
    handler = COMMANDS[args.command]
    try:
        return handler(args, config)
    except KeyboardInterrupt:
        log.warning("interrupted — progress is checkpointed; rerun with the same --run-id to resume")
        return 130
    except Exception as exc:
        log.exception("%s failed: %s", args.command, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
