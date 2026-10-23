"""Build the item sets into ``data/processed/`` (repo-local, never ``$HOME``).

Two datasets, per spec §4:

**MGSM** — 250 items per language. MGSM's test split is item-aligned across
languages (row *i* is the same GSM8K problem in every config) and equals the
first 250 problems of ``openai/gsm8k`` ``main/test``. We verify both facts rather
than assuming them.

*Hindi is not in MGSM* (configs: bn de en es fr ja ru sw te th zh). We build an
item-aligned MGSM-Hi: locate each MGSM-en question in the GSM8K test split, take
the Hindi translation at the same index from ``nvidia/GSM8K-Hi`` (which preserves
GSM8K test order), and keep the item only if the Hindi gold answer parses to the
same number as MGSM's ``answer_number``. Items failing that check are dropped and
logged, so a silent misalignment cannot enter the results.

**Global-MMLU** — the culturally-agnostic (``CA``) subset, 200 items per language
stratified by ``subject_category``. Items are selected once on the *intersection*
of ``sample_id`` sets across all requested languages, so every language gets the
same 200 questions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import Config, LanguageSpec, TaskSpec
from .env import PROCESSED_DATA_DIR, RAW_DATA_DIR
from .logging_utils import LoggedProgress, get_logger, progress
from .text_utils import canonical_number, extract_numbers, normalize_text

log = get_logger("data")

MCQ_LETTERS = ["A", "B", "C", "D"]


@dataclass
class ItemSet:
    task: str
    language: str
    items: list[dict[str, Any]]
    path: Path

    def __len__(self) -> int:
        return len(self.items)


def processed_path(task: str, language: str) -> Path:
    return PROCESSED_DATA_DIR / task / f"{language}.jsonl"


def manifest_path() -> Path:
    return PROCESSED_DATA_DIR / "manifest.json"


# ---------------------------------------------------------------------------
# HF loading helpers
# ---------------------------------------------------------------------------


def _load_hf(dataset: str, config: str | None, split: str) -> Any:
    """Load a HF dataset into the repo-local cache."""
    from datasets import load_dataset  # imported late: env must be pinned first
    from datasets.utils.logging import disable_progress_bar

    # We drive our own bars; the library's would interleave with them.
    disable_progress_bar()

    log.info("loading HF dataset %s config=%s split=%s", dataset, config, split)
    return load_dataset(
        dataset,
        config,
        split=split,
        cache_dir=str(RAW_DATA_DIR / "hf"),
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_items(task: str, language: str) -> list[dict[str, Any]]:
    path = processed_path(task, language)
    if not path.exists():
        raise FileNotFoundError(
            f"item set missing: {path}. Run `ltd fetch-data` first."
        )
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# MGSM
# ---------------------------------------------------------------------------


def _mgsm_gold(row: dict[str, Any]) -> str | None:
    value = row.get("answer_number")
    if value is not None and str(value).strip() not in {"", "None"}:
        return canonical_number(str(value))
    answer = row.get("answer")
    if answer:
        text = str(answer)
        if "####" in text:
            return canonical_number(text.rsplit("####", 1)[1])
        numbers = extract_numbers(text)
        if numbers:
            return numbers[-1]
    return None


def _gsm8k_gold(answer: str) -> str | None:
    if "####" in answer:
        return canonical_number(answer.rsplit("####", 1)[1])
    numbers = extract_numbers(answer)
    return numbers[-1] if numbers else None


def build_mgsm_reference(task: TaskSpec) -> list[dict[str, Any]]:
    """The English MGSM items with their GSM8K test index — the alignment key."""
    mgsm_en = _load_hf(task.hf_dataset, "en", task.hf_split)
    bridge = task.hindi_bridge or {}
    gsm8k = _load_hf(bridge["en_dataset"], bridge["en_config"], bridge["en_split"])

    index_by_question: dict[str, int] = {}
    for idx, row in enumerate(progress(gsm8k, desc="index gsm8k-en", unit="item")):
        index_by_question.setdefault(normalize_text(row["question"]), idx)

    reference: list[dict[str, Any]] = []
    missing = 0
    for position, row in enumerate(progress(mgsm_en, desc="align mgsm-en", unit="item")):
        key = normalize_text(row["question"])
        gsm_index = index_by_question.get(key)
        if gsm_index is None:
            missing += 1
            log.warning("MGSM-en item %d has no GSM8K match; excluded from Hindi bridge", position)
        reference.append(
            {
                "position": position,
                "gsm8k_index": gsm_index,
                "question_en": row["question"].strip(),
                "gold": _mgsm_gold(row),
            }
        )
    aligned = sum(1 for r in reference if r["gsm8k_index"] is not None)
    log.info(
        "MGSM-en: %d items, %d aligned to GSM8K test (%d unmatched)",
        len(reference), aligned, missing,
    )
    if aligned and all(
        r["gsm8k_index"] == r["position"] for r in reference if r["gsm8k_index"] is not None
    ):
        log.info("MGSM test == first %d GSM8K test items (index-identity confirmed)", len(reference))
    return reference


def build_mgsm_language(
    task: TaskSpec,
    lang: LanguageSpec,
    reference: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    n_items = task.n_items

    if lang.mgsm_source == "mgsm":
        rows = _load_hf(task.hf_dataset, lang.code, task.hf_split)
        items = []
        for position, row in enumerate(rows):
            if position >= n_items:
                break
            gold = _mgsm_gold(row)
            ref = reference[position] if position < len(reference) else {}
            if gold is None:
                log.warning("mgsm/%s item %d: unparseable gold, skipped", lang.code, position)
                continue
            if ref.get("gold") is not None and gold != ref["gold"]:
                log.warning(
                    "mgsm/%s item %d: gold %s != English gold %s (kept, flagged)",
                    lang.code, position, gold, ref["gold"],
                )
            items.append(
                {
                    "item_id": f"mgsm-{position:04d}",
                    "task": "mgsm",
                    "language": lang.code,
                    "source": f"{task.hf_dataset}:{lang.code}",
                    "question": row["question"].strip(),
                    "question_en": ref.get("question_en"),
                    "gold": gold,
                    "gold_en": ref.get("gold"),
                    "aligned": True,
                }
            )
        return items

    if lang.mgsm_source == "gsm8k_hi_aligned":
        return _build_mgsm_via_bridge(task, lang, reference)

    raise ValueError(f"unknown mgsm_source {lang.mgsm_source!r} for language {lang.code}")


def _build_mgsm_via_bridge(
    task: TaskSpec,
    lang: LanguageSpec,
    reference: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Item-aligned MGSM for a language absent from MGSM (currently Hindi)."""
    bridge = task.hindi_bridge
    if not bridge:
        raise ValueError("task has no hindi_bridge configured")
    translated = _load_hf(bridge["hi_dataset"], bridge["hi_config"], bridge["hi_split"])
    log.info(
        "bridge source %s has %d rows (GSM8K test has 1319)",
        bridge["hi_dataset"], len(translated),
    )

    items: list[dict[str, Any]] = []
    dropped_unaligned = 0
    dropped_gold = 0
    for ref in reference[: task.n_items]:
        gsm_index = ref["gsm8k_index"]
        if gsm_index is None or gsm_index >= len(translated):
            dropped_unaligned += 1
            continue
        row = translated[int(gsm_index)]
        gold_translated = _gsm8k_gold(str(row.get("answer", "")))
        gold_en = ref["gold"]
        if gold_en is None:
            dropped_gold += 1
            continue
        if gold_translated is not None and gold_translated != gold_en:
            # The translation's own gold disagrees with MGSM's -> the pair is not
            # trustworthy as the same problem. Drop it rather than guess.
            dropped_gold += 1
            log.warning(
                "bridge item gsm8k[%s]: translated gold %s != MGSM gold %s, dropped",
                gsm_index, gold_translated, gold_en,
            )
            continue
        items.append(
            {
                "item_id": f"mgsm-{ref['position']:04d}",
                "task": "mgsm",
                "language": lang.code,
                "source": f"{bridge['hi_dataset']}@gsm8k_index={gsm_index}",
                "question": str(row["question"]).strip(),
                "question_en": ref["question_en"],
                "gold": gold_en,
                "gold_en": gold_en,
                "aligned": True,
            }
        )
    log.info(
        "mgsm/%s via bridge: %d items kept, %d unaligned, %d gold-mismatch/unparseable",
        lang.code, len(items), dropped_unaligned, dropped_gold,
    )
    if len(items) < 0.9 * task.n_items:
        log.warning(
            "mgsm/%s retained only %d/%d items — check the bridge dataset before "
            "reporting per-language numbers",
            lang.code, len(items), task.n_items,
        )
    return items


# ---------------------------------------------------------------------------
# Global-MMLU
# ---------------------------------------------------------------------------


def _gmmlu_rows(task: TaskSpec, lang: LanguageSpec) -> dict[str, dict[str, Any]]:
    """CA-labelled rows for one language keyed by ``sample_id``."""
    dataset = _load_hf(task.hf_dataset, lang.gmmlu_config, task.hf_split)
    wanted = task.cultural_sensitivity_label
    out: dict[str, dict[str, Any]] = {}
    for row in dataset:
        if wanted and str(row.get("cultural_sensitivity_label", "")).strip() != wanted:
            continue
        answer = str(row.get("answer", "")).strip().upper()
        if answer not in MCQ_LETTERS:
            continue
        options = [str(row.get(f"option_{c.lower()}", "")).strip() for c in MCQ_LETTERS]
        if any(not opt for opt in options):
            continue
        out[str(row["sample_id"])] = {
            "sample_id": str(row["sample_id"]),
            "subject": str(row.get("subject", "")),
            "subject_category": str(row.get("subject_category", "unknown")),
            "question": str(row["question"]).strip(),
            "options": options,
            "gold": answer,
        }
    log.info("gmmlu/%s: %d usable %s rows", lang.code, len(out), wanted)
    return out


def select_gmmlu_sample_ids(
    task: TaskSpec,
    per_language: dict[str, dict[str, dict[str, Any]]],
    seed: int,
) -> list[str]:
    """Pick ``n_items`` sample_ids present in *every* language, stratified by category."""
    import random

    lang_codes = list(per_language)
    shared = set(per_language[lang_codes[0]])
    for code in lang_codes[1:]:
        shared &= set(per_language[code])
    log.info("gmmlu: %d sample_ids shared across %s", len(shared), lang_codes)
    if not shared:
        raise RuntimeError("no Global-MMLU items shared across the requested languages")

    # Gold labels must agree across languages (they do by construction, but check).
    reference = per_language[lang_codes[0]]
    consistent = []
    for sid in sorted(shared):
        golds = {per_language[code][sid]["gold"] for code in lang_codes}
        if len(golds) == 1:
            consistent.append(sid)
        else:
            log.warning("gmmlu %s: gold disagrees across languages %s, dropped", sid, golds)

    by_category: dict[str, list[str]] = {}
    for sid in consistent:
        by_category.setdefault(reference[sid]["subject_category"], []).append(sid)

    rng = random.Random(seed)
    for sids in by_category.values():
        rng.shuffle(sids)

    target = min(task.n_items, len(consistent))
    categories = sorted(by_category)
    # Proportional allocation, then round-robin top-up so the total lands exactly.
    total_pool = sum(len(by_category[c]) for c in categories)
    quota = {c: int(target * len(by_category[c]) / total_pool) for c in categories}
    selected: list[str] = []
    for category in categories:
        selected.extend(by_category[category][: quota[category]])
    cursor = {c: quota[c] for c in categories}
    while len(selected) < target:
        progressed = False
        for category in categories:
            if len(selected) >= target:
                break
            pool = by_category[category]
            if cursor[category] < len(pool):
                selected.append(pool[cursor[category]])
                cursor[category] += 1
                progressed = True
        if not progressed:
            break

    selected = sorted(set(selected))
    counts = {}
    for sid in selected:
        cat = reference[sid]["subject_category"]
        counts[cat] = counts.get(cat, 0) + 1
    log.info("gmmlu: selected %d items; per-category %s", len(selected), counts)
    return selected


def build_gmmlu(
    task: TaskSpec,
    languages: Sequence[LanguageSpec],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    per_language: dict[str, dict[str, dict[str, Any]]] = {}
    with LoggedProgress(len(languages), "gmmlu load", unit="lang") as bar:
        for lang in languages:
            per_language[lang.code] = _gmmlu_rows(task, lang)
            bar.update(1, {"lang": lang.code})

    sample_ids = select_gmmlu_sample_ids(task, per_language, seed)

    out: dict[str, list[dict[str, Any]]] = {}
    for lang in languages:
        rows = per_language[lang.code]
        items = []
        for position, sid in enumerate(sample_ids):
            row = rows[sid]
            items.append(
                {
                    "item_id": f"gmmlu-{sid}",
                    "task": "gmmlu",
                    "language": lang.code,
                    "source": f"{task.hf_dataset}:{lang.gmmlu_config}",
                    "sample_id": sid,
                    "subject": row["subject"],
                    "subject_category": row["subject_category"],
                    "question": row["question"],
                    "options": row["options"],
                    "gold": row["gold"],
                    "aligned": True,
                    "position": position,
                }
            )
        out[lang.code] = items
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_all(
    config: Config,
    tasks: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    force: bool = False,
) -> dict[str, dict[str, int]]:
    task_names = config.resolve_tasks(tasks)
    lang_codes = config.resolve_languages(languages)
    lang_specs = [config.language(code) for code in lang_codes]
    summary: dict[str, dict[str, int]] = {}

    for task_name in task_names:
        task = config.task(task_name)
        summary[task_name] = {}

        pending = [
            lang for lang in lang_specs
            if force or not processed_path(task_name, lang.code).exists()
        ]
        for lang in lang_specs:
            if lang not in pending:
                existing = read_items(task_name, lang.code)
                summary[task_name][lang.code] = len(existing)
                log.info(
                    "%s/%s already built (%d items) — use --force to rebuild",
                    task_name, lang.code, len(existing),
                )

        if not pending:
            continue

        if task_name == "mgsm":
            reference = build_mgsm_reference(task)
            with LoggedProgress(len(pending), "build mgsm", unit="lang") as bar:
                for lang in pending:
                    items = build_mgsm_language(task, lang, reference)
                    path = processed_path(task_name, lang.code)
                    _write_jsonl(path, items)
                    summary[task_name][lang.code] = len(items)
                    log.info("wrote %s (%d items)", path.relative_to(path.parents[3]), len(items))
                    bar.update(1, {"lang": lang.code, "n": len(items)})
        elif task_name == "gmmlu":
            # Selection must consider *all* requested languages so the item set is
            # shared; rebuild every language when any is missing.
            built = build_gmmlu(task, lang_specs, config.seed)
            with LoggedProgress(len(lang_specs), "write gmmlu", unit="lang") as bar:
                for lang in lang_specs:
                    path = processed_path(task_name, lang.code)
                    _write_jsonl(path, built[lang.code])
                    summary[task_name][lang.code] = len(built[lang.code])
                    bar.update(1, {"lang": lang.code, "n": len(built[lang.code])})
        else:
            raise ValueError(f"no builder for task {task_name!r}")

    _write_manifest(config, summary)
    return summary


def _write_manifest(config: Config, summary: dict[str, dict[str, int]]) -> None:
    path = manifest_path()
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    counts = existing.get("counts", {})
    for task_name, per_lang in summary.items():
        counts.setdefault(task_name, {}).update(per_lang)
    payload = {
        "seed": config.seed,
        "counts": counts,
        "sources": {
            name: {
                "hf_dataset": spec.hf_dataset,
                "split": spec.hf_split,
                "n_items": spec.n_items,
                "cultural_sensitivity_label": spec.cultural_sensitivity_label,
                "hindi_bridge": spec.hindi_bridge,
            }
            for name, spec in config.tasks.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("manifest -> %s", path)


def verify_alignment(config: Config, tasks: Iterable[str] | None = None,
                     languages: Iterable[str] | None = None) -> list[str]:
    """Check every language's item set shares item_ids and golds. Returns problems."""
    problems: list[str] = []
    for task_name in config.resolve_tasks(tasks):
        codes = config.resolve_languages(languages)
        loaded: dict[str, list[dict[str, Any]]] = {}
        for code in codes:
            try:
                loaded[code] = read_items(task_name, code)
            except FileNotFoundError as exc:
                problems.append(str(exc))
        if len(loaded) < 2:
            continue
        id_sets = {code: {item["item_id"] for item in items} for code, items in loaded.items()}
        shared = set.intersection(*id_sets.values())
        for code, ids in id_sets.items():
            extra = len(ids - shared)
            if extra:
                problems.append(
                    f"{task_name}/{code}: {extra} item_ids not shared with all languages"
                )
        golds = {
            code: {item["item_id"]: item["gold"] for item in items}
            for code, items in loaded.items()
        }
        reference_code = next(iter(golds))
        for item_id in sorted(shared):
            values = {golds[code][item_id] for code in golds}
            if len(values) > 1:
                problems.append(
                    f"{task_name}/{item_id}: golds disagree across languages {values} "
                    f"(reference={golds[reference_code][item_id]})"
                )
        log.info("%s: %d shared items across %d languages", task_name, len(shared), len(loaded))
    return problems
