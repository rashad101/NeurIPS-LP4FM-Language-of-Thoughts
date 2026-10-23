#!/usr/bin/env python3
"""Build the qualitative transcripts and the error taxonomy of the appendix.

This is the audit trail for Appendix "Qualitative and error analysis" in the same
sense that ``scripts/paper_stats.py`` is the audit trail for the aggregate tables.
Everything it emits is derived from ``runs/<run>/predictions`` and
``runs/<run>/scored``, so nothing in that appendix is hand-transcribed.

    python scripts/make_qualitative_examples.py

Writes:
  results/tables/error_taxonomy.csv        per-cell error composition
  results/tables/error_taxonomy.json       the aggregates quoted in the prose
  results/tables/qualitative_examples.json the selected transcripts, in full
It also prints the LaTeX transcript blocks to stdout. The paper keeps them inline
in ``paper/neurips_2026.tex`` rather than inputting a generated file, so paste the
printed blocks over the existing ones after a re-run.

The transcripts are pinned by item id rather than re-selected on every run, so
the paper does not change under an unrelated edit. Each pin carries the criteria
it was chosen to satisfy, and the script asserts them, so a pin that stops being
representative fails loudly instead of silently misleading.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import regex

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ltd.extract import extract_prediction, is_correct, split_trace_answer  # noqa: E402

RUN = ROOT / "runs" / "main-full-v2"
TABLES = ROOT / "results" / "tables"

MODELS = ["qwen3-8b", "r1-llama-8b"]
TASKS = {"mgsm": "numeric", "gmmlu": "mcq"}
LANGS = ["de", "es", "sw", "hi", "bn", "te"]  # English is excluded, as everywhere else
CONDS = ["target", "en_cot", "free", "mixed"]
LABELS = {"target": "Target-CoT", "en_cot": "En-CoT", "free": "Free-CoT", "mixed": "Mixed-CoT"}
NAMES = {"de": "German", "es": "Spanish", "sw": "Swahili",
         "hi": "Hindi", "bn": "Bengali", "te": "Telugu"}
TASK_NAMES = {"mgsm": "MGSM", "gmmlu": "Global MMLU"}
MODEL_NAMES = {"qwen3-8b": "Qwen3-8B", "r1-llama-8b": "R1-Distill-Llama-8B"}

# Scripts that need their own font in the LaTeX output. Latin-script targets
# (de, es, sw) ride on the document font.
SCRIPT_CMD = {"hi": "hitext", "bn": "bntext", "te": "tetext"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _byte_decoder() -> dict[str, int]:
    """Inverse of the GPT-2 byte-to-unicode map used by byte-level BPE.

    Needed because every R1-Distill record is stored with its byte-level markers
    undecoded (Appendix "A text-decoding defect in the R1-Distill records"), so a
    space reads as U+0120 and a newline as U+010A. We re-decode only for display.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) \
        + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER = _byte_decoder()


def redecode(text: str) -> str:
    """Undo the byte-level BPE storage defect, leaving already-clean text alone."""
    if "Ġ" not in text and "Ċ" not in text:
        return text
    try:
        raw = bytes(_BYTE_DECODER[ch] for ch in text)
    except KeyError:
        return text
    return raw.decode("utf-8", errors="replace")


def load_cell(model: str, task: str, lang: str, cond: str) -> list[dict]:
    """Join the raw generations with their scored records on item id."""
    pred_path = RUN / "predictions" / model / task / lang / f"{cond}.jsonl"
    scored_path = RUN / "scored" / f"{model}__{task}__{lang}__{cond}.jsonl"
    preds: dict[str, dict] = {}
    with pred_path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            preds[rec["item_id"]] = rec
    rows = []
    with scored_path.open(encoding="utf-8") as fh:
        for line in fh:
            scored = json.loads(line)
            pred = preds.get(scored["item_id"])
            if pred is None:
                continue
            merged = dict(pred)
            merged.update({k: v for k, v in scored.items() if v is not None or k not in merged})
            rows.append(merged)
    return rows


# --------------------------------------------------------------------------- #
# Error taxonomy
# --------------------------------------------------------------------------- #

ERROR_TYPES = ["wrong_parsed", "no_answer", "truncated_reasoning", "truncated_rendering"]


def classify(rec: dict) -> str:
    """Label a generation with exactly one outcome.

    ``correct`` is the complement of the four error types, so the five labels
    partition every generation in a cell.

    * ``wrong_parsed``        finished, an answer was parsed, the answer is wrong
    * ``no_answer``           finished, nothing parseable to score
    * ``truncated_reasoning`` hit the completion cap with the trace still open
    * ``truncated_rendering`` hit the cap after ``</think>``, i.e. the budget went
      on writing the answer rather than on reasoning
    """
    if rec["correct"]:
        return "correct"
    if rec["truncated"]:
        return "truncated_rendering" if "</think>" in rec["output"] else "truncated_reasoning"
    if rec["extraction_failed"]:
        return "no_answer"
    return "wrong_parsed"


def trace_only_correct(rec: dict, kind: str) -> bool:
    """Would the trace alone have scored, if the answer segment were discarded?

    This separates a computation that failed from a computation that succeeded
    and lost its score in the rendering.
    """
    trace, _, _ = split_trace_answer(rec["output"])
    return is_correct(kind, extract_prediction(kind, trace, trace), rec["gold"])


def build_taxonomy() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    recoverable: list[dict] = []
    totals: Counter = Counter()

    for model in MODELS:
        for task, kind in TASKS.items():
            for lang in LANGS:
                for cond in CONDS:
                    recs = load_cell(model, task, lang, cond)
                    counts: Counter = Counter(classify(r) for r in recs)
                    n_recover = 0
                    for rec in recs:
                        if classify(rec) == "truncated_rendering" and trace_only_correct(rec, kind):
                            n_recover += 1
                            recoverable.append({
                                "model": model, "task": task, "language": lang,
                                "condition": cond, "item_id": rec["item_id"],
                                "gold": rec["gold"], "scored_prediction": rec["prediction"],
                            })
                    row = {
                        "model": model, "task": task, "language": lang, "condition": cond,
                        "n": len(recs),
                        **{k: counts.get(k, 0) for k in ["correct", *ERROR_TYPES]},
                        "truncated_rendering_recoverable": n_recover,
                    }
                    rows.append(row)
                    for key in ["correct", *ERROR_TYPES]:
                        totals[(model, task, cond, key)] += counts.get(key, 0)
                        totals[(model, key)] += counts.get(key, 0)
                    totals[(model, "n")] += len(recs)
                    totals[(model, task, cond, "n")] += len(recs)

    def pct(model, task, cond, key):
        n = totals[(model, task, cond, "n")]
        return 100.0 * totals[(model, task, cond, key)] / n if n else 0.0

    summary = {
        "note": "All figures over the six non-English languages; English cells are "
                "excluded here as they are from every contrast in the paper.",
        "by_model_task_condition": {
            f"{m}|{t}|{c}": {k: round(pct(m, t, c, k), 2) for k in ["correct", *ERROR_TYPES]}
            for m in MODELS for t in TASKS for c in CONDS
        },
        "by_model": {
            m: {k: round(100.0 * totals[(m, k)] / totals[(m, "n")], 2)
                for k in ["correct", *ERROR_TYPES]} | {"n": totals[(m, "n")]}
            for m in MODELS
        },
        "recoverable_rendering_truncations": recoverable,
        "n_recoverable": len(recoverable),
    }
    return rows, summary


# --------------------------------------------------------------------------- #
# Trace overlap
# --------------------------------------------------------------------------- #
# An instrument independent of the language shares of Table 3. If asking for a
# native trace barely changes the language of the trace, does it change the
# trace at all? We compare the Target-CoT and En-CoT traces for the same item,
# and calibrate that against two conditions that both ask for English reasoning,
# whose only difference is prompt wording and sampling.

_WORD_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+")

OVERLAP_PAIRS = [("target", "en_cot"), ("en_cot", "free"), ("target", "free"),
                 ("target", "mixed")]


def _trace_bags(model: str, task: str, lang: str, cond: str) -> dict[str, set[str]]:
    bags = {}
    for rec in load_cell(model, task, lang, cond):
        trace, _, _ = split_trace_answer(rec["output"])
        bags[rec["item_id"]] = {w.lower() for w in _WORD_RE.findall(trace)}
    return bags


def build_trace_overlap() -> list[dict]:
    """Mean per-item Jaccard overlap of trace vocabularies, between policies.

    Qwen3-8B only: this is a word-level measurement, and the R1-Distill records
    carry the text-decoding defect that makes every word-level measurement on
    them unusable.
    """
    rows = []
    for task in TASKS:
        for lang in LANGS:
            bags = {c: _trace_bags("qwen3-8b", task, lang, c) for c in CONDS}
            ids = set.intersection(*(set(b) for b in bags.values()))
            row = {"model": "qwen3-8b", "task": task, "language": lang, "n_items": len(ids)}
            for a, b in OVERLAP_PAIRS:
                scores = []
                for item in ids:
                    x, y = bags[a][item], bags[b][item]
                    if x and y:
                        scores.append(len(x & y) / len(x | y))
                row[f"{a}__{b}"] = round(sum(scores) / len(scores), 4) if scores else None
            rows.append(row)
    return rows


def emit_overlap_table(rows: list[dict]) -> str:
    out = []
    for task in TASKS:
        out.append(f"    \\multicolumn{{5}}{{l}}{{\\emph{{{TASK_NAMES[task]}}}}} \\\\")
        for lang in LANGS:
            row = next(r for r in rows if r["task"] == task and r["language"] == lang)
            vals = " & ".join(f"{row[f'{a}__{b}']:.2f}" for a, b in OVERLAP_PAIRS)
            out.append(f"    {NAMES[lang]} ({lang}) & {vals} \\\\")
        out.append("    \\addlinespace")
    return "\n".join(out[:-1])


# --------------------------------------------------------------------------- #
# Transcript selection
# --------------------------------------------------------------------------- #
# Each entry pins one generation and records what it has to satisfy to stand for
# the claim it illustrates. ``check`` is asserted at build time.

EXAMPLES: list[dict] = [
    {
        "key": "hi-four-policies",
        "group": "One item, four policies",
        "model": "qwen3-8b", "task": "mgsm", "lang": "hi",
        "items": [("target", "mgsm-0194"), ("en_cot", "mgsm-0194"),
                  ("free", "mgsm-0194"), ("mixed", "mgsm-0194")],
        "check": lambda r: r["correct"],
        "head": 300, "tail": 170, "answer": 260,
        "note": "The four thinking-language instructions produce one trace. Target-CoT "
                "read an instruction demanding that every step of its reasoning appear in "
                "Hindi, and it wrote English indistinguishable from En-CoT's own, "
                "down to the self-check at the end. The instruction moves the answer "
                "instead, and even that only sometimes. Target-CoT and Mixed-CoT answer in "
                "Hindi, Free-CoT answers in Hindi without being asked, and En-CoT, which "
                "requested a Hindi answer in the same words Target-CoT used, answers in "
                "English. All four traces also anglicize the given name, so the Hindi "
                "\\hitext{वायलेटा} enters the computation as ``Violet'', which the entity "
                "retention of Table~\\ref{tab:retention} counts as a loss. Mixed-CoT spends "
                "958 tokens against the 450 of Target-CoT, and it spends the difference "
                "deliberating in English over which Hindi word to answer in.",
    },
    {
        "key": "bn-target-entity-loss",
        "promote": True,   # shown in the main text, not the appendix
        "group": "What an English trace does to the entities of the question",
        "model": "qwen3-8b", "task": "mgsm", "lang": "bn",
        "items": [("target", "mgsm-0000")],
        "check": lambda r: r["correct"] and r["trace_target_share"] == 0.0
                           and r["answer_compliant"],
        "head": 330, "tail": 165, "answer": 240,
        "note": "Asked for a Bengali chain of thought, the model writes English and "
                "silently recasts the question as it translates. "
                "\\bntext{জেনেট}, Janet, becomes ``Gene'', and the protagonist stays ``he'' "
                "for the length of the computation. The arithmetic never touches the name, "
                "so the item scores correct and the Bengali answer restores "
                "\\bntext{জেনেট}. The generation is nonetheless a faithfulness failure that "
                "accuracy cannot see and that anchor retention can, which argues for "
                "keeping the two on separate axes. It also shows the concrete form of the "
                "regression our position describes, because reasoning in a hub language "
                "costs a question its particulars first, and the particulars of a Bengali "
                "question are Bengali.",
    },
    {
        "key": "de-target-complies",
        "group": "Compliance where it is nearly free",
        "model": "qwen3-8b", "task": "mgsm", "lang": "de",
        "items": [("target", "mgsm-0113")],
        "check": lambda r: r["trace_compliant"] and r["correct"],
        "head": 340, "tail": 0, "answer": 240,
        "note": "German is the one language in which the instruction takes hold, and this "
                "is what taking hold looks like. The trace runs German after the model's "
                "habitual English opener, the answer runs German, and the whole "
                "generation costs 209 tokens. Compliance sacrifices nothing here, which is "
                "exactly why Table~\\ref{tab:main} reports no German entry in which another "
                "policy dominates Target-CoT.",
    },
    {
        "key": "te-target-quotation",
        "group": "What the residual target-language share is made of",
        "model": "qwen3-8b", "task": "mgsm", "lang": "te",
        "items": [("target", "mgsm-0225")],
        "check": lambda r: 0.10 < r["trace_target_share"] < 0.5 and r["correct"],
        "head": 430, "tail": 150, "answer": 200,
        "note": "This trace carries more Telugu than any other Telugu trace under "
                "Target-CoT, at a 15\\% target share, and none of that share is reasoning. "
                "The Telugu in it quotes the question, pasted into an English sentence so "
                "that the model can translate it and then reason about the translation. "
                "The 1 to 4\\% shares of Table~\\ref{tab:delivery} consist of exactly this, "
                "and it explains why no Brahmic-script trace meets a threshold of 0.5. The "
                "target language appears in these traces as an object of reference rather "
                "than as the medium.",
    },
    {
        "key": "hi-mixed-anchor",
        "group": "A verbatim anchor that does not carry its meaning",
        "model": "qwen3-8b", "task": "mgsm", "lang": "hi",
        "items": [("mixed", "mgsm-0217")],
        "check": lambda r: r["trace_has_target_anchor"] and r["correct"],
        "head": 430, "tail": 0, "answer": 200,
        "note": "Mixed-CoT does what the instruction asks. It carries the Hindi clause "
                "into the trace verbatim, and this generation therefore counts as "
                "retaining an anchor in Table~\\ref{tab:delivery}. It then glosses "
                "\\hitext{डहलिया}, dahlias, as ``marigolds'' and reasons on for eleven "
                "marigolds. The arithmetic escapes unaffected, the item scores correct, and "
                "the Hindi answer restores the right flower. The example bounds what "
                "verbatim retention measures. Retention records that a surface form reached "
                "the trace, not that its meaning did, so a policy can score well on it "
                "while the computation quietly proceeds on a mistranslation.",
    },
    {
        "key": "sw-render-runaway",
        "group": "The computation succeeds and the rendering loses the score",
        "model": "qwen3-8b", "task": "gmmlu", "lang": "sw",
        "items": [("target", "gmmlu-abstract_algebra/test/33")],
        "check": lambda r: r["truncated"] and not r["correct"] and "</think>" in r["output"],
        "head": 260, "tail": 190, "answer": 300,
        "note": "The English reasoning finishes, and it finishes correctly, closing on "
                "``the correct answer should be C'', which is the gold option. The model "
                "then starts the Swahili answer it was asked for, falls into a repetition "
                "loop, and spends the remainder of an 8{,}192-token budget inside that "
                "loop. The extractor finds no \\#\\#\\#\\# marker, falls back on the letter "
                "it can reach within the loop, and the item scores wrong. Nothing failed "
                "here except the rendering.",
    },
    {
        "key": "r1-te-no-capability",
        "group": "Consistency asked of a model that does not have the language",
        "model": "r1-llama-8b", "task": "mgsm", "lang": "te",
        "items": [("target", "mgsm-0000")],
        "check": lambda r: not r["correct"] and r["truncated"],
        "head": 430, "tail": 150, "answer": 0,
        "note": "The instruction asks for a Telugu chain of thought. The model never reads "
                "the Telugu at all, because the question reaches it as ``16..\\$2''. It "
                "guesses at a cryptarithmetic puzzle and fills 12{,}288 tokens without ever "
                "closing the trace. This is the 0.4\\% MGSM Telugu entry of "
                "Table~\\ref{tab:acc-full} seen from the inside, and it is what "
                "\\S\\ref{sec:landscape} calls compliance as camouflage. A "
                "language-consistency reward applied to this model would optimize the "
                "surface language of that text, and no computation lies underneath that "
                "surface for the language to stay consistent with.",
    },
]

def elide(text: str, head: int, tail: int) -> str:
    """Collapse whitespace, then keep a head and optionally a tail with an ellipsis.

    Both cuts snap outward to a space, so an excerpt never begins or ends in the
    middle of a word.
    """
    flat = re.sub(r"\s+", " ", text).strip()
    if head <= 0:
        return ""
    if len(flat) <= head + tail + 20:
        return flat
    cut = flat.find(" ", head)
    front = (flat[:cut] if cut != -1 else flat).rstrip()
    if tail <= 0:
        return front + " [...]"
    start = flat.rfind(" ", 0, len(flat) - tail)
    back = flat[start + 1:] if start != -1 else flat[-tail:]
    return front + " [...] " + back


def collect_examples() -> list[dict]:
    out = []
    for spec in EXAMPLES:
        kind = TASKS[spec["task"]]
        parts = []
        for cond, item_id in spec["items"]:
            recs = {r["item_id"]: r for r in load_cell(spec["model"], spec["task"],
                                                       spec["lang"], cond)}
            if item_id not in recs:
                raise SystemExit(f"{spec['key']}: item {item_id} missing from "
                                 f"{spec['model']}/{spec['task']}/{spec['lang']}/{cond}")
            rec = recs[item_id]
            if not spec["check"](rec):
                raise SystemExit(f"{spec['key']}: pinned item {item_id} ({cond}) no longer "
                                 f"satisfies the selection criteria")
            trace, answer, _ = split_trace_answer(rec["output"])
            parts.append({
                "condition": cond,
                "condition_label": LABELS[cond],
                "item_id": item_id,
                "correct": rec["correct"],
                "truncated": rec["truncated"],
                "n_completion_tokens": rec["n_completion_tokens"],
                "trace_target_share": rec.get("trace_target_share"),
                "answer_compliant": rec.get("answer_compliant"),
                "gold": rec["gold"],
                "prediction": rec["prediction"],
                "trace_full": redecode(trace),
                "answer_full": redecode(answer),
                "trace": elide(redecode(trace), spec["head"], spec["tail"]),
                "answer": elide(redecode(answer), spec["answer"], 0),
                "trace_only_correct": trace_only_correct(rec, kind),
            })
        first = recs[spec["items"][0][1]]
        out.append({
            "key": spec["key"], "group": spec["group"], "model": spec["model"],
            "note": spec["note"], "promote": spec.get("promote", False),
            "task": spec["task"], "language": spec["lang"],
            "question": first.get("question", ""),
            "question_en": first.get("question_en") or "",
            "options": first.get("options") or [],
            "instruction": _instruction_of(first),
            "parts": parts,
        })
    return out


def _instruction_of(rec: dict) -> str:
    """The thinking-language instruction, recovered from the stored user turn.

    The user turn is question, then (for Global MMLU) the options, then a final
    block holding the thinking-language instruction followed by the one-line
    answer-format instruction. Only the first part of that block varies by
    condition, so we drop the last line.
    """
    blocks = [b for b in rec.get("user_message", "").split("\n\n") if b.strip()]
    if not blocks:
        return ""
    lines = [ln.strip() for ln in blocks[-1].splitlines() if ln.strip()]
    return " ".join(lines[:-1]) if len(lines) > 1 else " ".join(lines)


# --------------------------------------------------------------------------- #
# LaTeX emission
# --------------------------------------------------------------------------- #

_TEX_ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# Characters the model emits that the document font either lacks or renders
# better through a command.
_GLYPHS = {
    "×": r"\ensuremath{\times}", "÷": r"\ensuremath{\div}",
    "–": "--", "—": "---", "→": r"\ensuremath{\rightarrow}",
    "’": "'", "‘": "`", "“": "``", "”": "''",
    "…": r"\dots{}", "​": "", " ": "~",
}


def normalize_quotes(text: str) -> str:
    """Turn straight double quotes into TeX's paired open/close forms.

    Done as a pass over the whole excerpt rather than inside ``tex_escape``,
    because ``wrap_script`` escapes one run at a time and the pairing has to be
    tracked across run boundaries: the model quotes the question in its own
    script inside an otherwise English sentence.
    """
    out, open_quote = [], True
    for ch in text:
        if ch == '"':
            out.append("“" if open_quote else "”")
            open_quote = not open_quote
        else:
            out.append(ch)
    return "".join(out)


def tex_escape(text: str) -> str:
    out = [_GLYPHS.get(ch) or _TEX_ESCAPES.get(ch, ch) for ch in text]
    return "".join(out).replace("...", r"\dots{}")


_SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F), "bn": (0x0980, 0x09FF), "te": (0x0C00, 0x0C7F),
}

# Zero-width joiner and non-joiner, and the danda and double danda, all sit
# outside the per-script blocks but belong to the run they appear in. The
# joiners in particular are instructions to the shaper, so splitting a run at
# one of them would change how the text renders.
_STICKY = {0x200C, 0x200D, 0x0964, 0x0965}


def wrap_script(text: str, lang: str) -> str:
    """Wrap runs of target-script characters in the font command for that script.

    Runs are wrapped rather than whole strings because the interesting excerpts
    are mixed: an English trace that quotes the Telugu question, a Hindi answer
    with Latin numerals and a Latin given name inside it.
    """
    text = normalize_quotes(text)
    cmd = SCRIPT_CMD.get(lang)
    if cmd is None:
        return tex_escape(text)
    lo, hi = _SCRIPT_RANGES[lang]

    def belongs(ch: str, mode: bool) -> bool:
        if lo <= ord(ch) <= hi:
            return True
        # Sticky characters, combining marks and whitespace continue whichever
        # run they are already in rather than starting a new one.
        if mode and (ord(ch) in _STICKY or unicodedata.category(ch) in ("Mn", "Mc")
                     or ch.isspace()):
            return True
        return False

    out: list[str] = []
    buf: list[str] = []
    mode = False

    def flush() -> None:
        if not buf:
            return
        chunk = "".join(buf)
        out.append(f"\\{cmd}{{{tex_escape(chunk)}}}" if mode else tex_escape(chunk))
        buf.clear()

    for ch in text:
        this = belongs(ch, mode)
        if this != mode:
            flush()
            mode = this
        buf.append(ch)
    flush()
    return "".join(out)


def emit_latex(examples: list[dict]) -> str:
    lines = [
        "% Generated by scripts/make_qualitative_examples.py -- do not edit by hand.",
        "% Inputted by neurips_2026.tex. Needs XeLaTeX or LuaLaTeX for the Indic scripts.",
        "",
    ]
    for ex in examples:
        lang = ex["language"]
        model = MODEL_NAMES[ex["model"]]
        task = TASK_NAMES[ex["task"]]
        header = (f"{ex['group']}. {model}, {task} {NAMES[lang]}, "
                  f"item \\texttt{{{tex_escape(ex['parts'][0]['item_id'])}}}")
        lines.append(f"\\begin{{transcript}}{{{header}}}")
        lines.append(f"\\trfield{{Question}}{{{wrap_script(re.sub(r'\\s+', ' ', ex['question']).strip(), lang)}}}")
        if ex["options"]:
            opts = r"\quad ".join(
                f"\\textbf{{{chr(65 + i)}.}}~" + wrap_script(re.sub(r"\s+", " ", o).strip(), lang)
                for i, o in enumerate(ex["options"]))
            lines.append(f"\\trfield{{Options}}{{{opts}}}")
        if ex["question_en"]:
            gloss = re.sub(r"\s+", " ", ex["question_en"]).strip()
            lines.append(f"\\trfield{{Gloss}}{{\\emph{{{wrap_script(gloss, None)}}}}}")
        if len(ex["parts"]) == 1:
            lines.append(f"\\trfield{{Instruction}}{{{wrap_script(ex['instruction'], None)}}}")
        for part in ex["parts"]:
            tag = part["condition_label"]
            meta = f"{part['n_completion_tokens']} tokens"
            if part["truncated"]:
                meta += ", truncated at the cap"
            meta += ", correct" if part["correct"] else f", scored incorrect (gold {tex_escape(str(part['gold']))})"
            if len(ex["parts"]) > 1:
                lines.append(f"\\trpolicy{{{tag}}}{{{meta}}}")
            else:
                lines.append(f"\\trfield{{Outcome}}{{{meta}}}")
            if part["trace"]:
                lines.append(f"\\trfield{{Trace}}{{{wrap_script(part['trace'], lang)}}}")
            if part["answer"]:
                lines.append(f"\\trfield{{Answer}}{{{wrap_script(part['answer'], lang)}}}")
        lines.append(f"\\trnote{{{ex['note']}}}")
        lines.append("\\end{transcript}")
        lines.append("")
    return "\n".join(lines)


def emit_taxonomy_table(rows: list[dict]) -> str:
    """The LaTeX body of the error-composition table, aggregated over languages."""
    agg: dict[tuple[str, str, str], Counter] = {}
    for row in rows:
        key = (row["model"], row["task"], row["condition"])
        acc = agg.setdefault(key, Counter())
        for field in ["n", "correct", *ERROR_TYPES]:
            acc[field] += row[field]
    out = []
    for model in MODELS:
        for task in TASKS:
            out.append(f"    \\multicolumn{{6}}{{l}}{{\\emph{{{MODEL_NAMES[model]}, "
                       f"{TASK_NAMES[task]}}}}} \\\\")
            for cond in CONDS:
                acc = agg[(model, task, cond)]
                n = acc["n"]
                vals = " & ".join(f"{100.0 * acc[k] / n:.1f}" for k in ["correct", *ERROR_TYPES])
                out.append(f"    {LABELS[cond]} & {vals} \\\\")
            out.append("    \\addlinespace")
    return "\n".join(out[:-1])


# --------------------------------------------------------------------------- #

def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    rows, summary = build_taxonomy()

    with (TABLES / "error_taxonomy.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (TABLES / "error_taxonomy.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    overlap = build_trace_overlap()
    with (TABLES / "trace_overlap.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(overlap[0]))
        writer.writeheader()
        writer.writerows(overlap)

    examples = collect_examples()
    (TABLES / "qualitative_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    promoted = [e for e in examples if e["promote"]]
    rest = [e for e in examples if not e["promote"]]

    print("=" * 72)
    print("Error composition (%), six non-English languages, per model/task/condition")
    print("=" * 72)
    print(emit_taxonomy_table(rows))
    print()
    print("=" * 72)
    print("Mean per-item trace-vocabulary overlap between policies (Qwen3-8B)")
    print("=" * 72)
    print(emit_overlap_table(overlap))
    print()
    print("Per model, over both tasks:")
    for model, vals in summary["by_model"].items():
        print(f"  {MODEL_NAMES[model]}: " + ", ".join(f"{k}={v}" for k, v in vals.items()))
    print(f"\nRendering truncations whose trace alone would have scored: "
          f"{summary['n_recoverable']}")
    by_cell = Counter((r["language"], r["condition"])
                      for r in summary["recoverable_rendering_truncations"])
    for (lang, cond), count in by_cell.most_common():
        print(f"  {NAMES[lang]} {LABELS[cond]}: {count}")
    print(f"\nWrote {TABLES/'trace_overlap.csv'}, {TABLES/'error_taxonomy.csv'},")
    print(f"      {TABLES/'error_taxonomy.json'}, {TABLES/'qualitative_examples.json'}")

    # The paper holds the transcripts inline, so emit them for pasting rather
    # than writing a generated .tex that neurips_2026.tex would have to input.
    print("\n" + "=" * 72)
    print("LaTeX for the transcript in the main text (paste into the landscape section)")
    print("=" * 72)
    print(emit_latex(promoted))
    print("=" * 72)
    print("LaTeX for the six transcripts in the appendix")
    print("=" * 72)
    print(emit_latex(rest))


if __name__ == "__main__":
    main()
