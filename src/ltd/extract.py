"""Split a generation into trace + answer, and extract the predicted answer.

Extraction is *identical across conditions* by construction — the answer-format
instruction never varies (see :mod:`ltd.prompts`) — so accuracy differences
cannot be an artefact of easier parsing in one condition.

Numeric extraction is Unicode-digit-normalised (spec §9), so a Devanagari
``१८``, a Bengali ``১৮`` and an ASCII ``18`` all score identically.
"""

from __future__ import annotations

import re

from .text_utils import canonical_number, extract_numbers, normalize_digits

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_HASH_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
# "the answer is B", "answer: (C)", "Antwort: D", "उत्तर: A" — letter fallbacks.
_LETTER_RE = re.compile(r"(?<![A-Za-z])\(?([A-D])\)?(?![A-Za-z])")
MCQ_LETTERS = ("A", "B", "C", "D")


def split_trace_answer(text: str) -> tuple[str, str, bool]:
    """Return ``(trace, answer, well_formed)``.

    * Both tags present, or only ``</think>`` (the R1 template pre-opens it):
      trace is everything before ``</think>``.
    * Only ``<think>`` and no close (truncation): everything after ``<think>`` is
      trace, answer is empty, ``well_formed=False``.
    * Neither tag: the model ignored the thinking format. We treat the whole
      output as *both* trace and answer, and flag it, because a non-thinking
      output still has a language and still contains an answer.
    """
    if THINK_CLOSE in text:
        head, _, tail = text.partition(THINK_CLOSE)
        if THINK_OPEN in head:
            head = head.split(THINK_OPEN, 1)[1]
        return head.strip(), tail.strip(), True
    if THINK_OPEN in text:
        return text.split(THINK_OPEN, 1)[1].strip(), "", False
    return text.strip(), text.strip(), False


def extract_numeric(answer_segment: str, full_text: str) -> str | None:
    """Predicted number, preferring the requested ``#### <n>`` form.

    Search order: ``####`` in the answer segment, ``\\boxed{}`` in the answer
    segment, last number in the answer segment, then the same three in the full
    text (which recovers truncated-but-answered generations).
    """
    for segment in (answer_segment, full_text):
        if not segment:
            continue
        hits = _HASH_RE.findall(segment)
        if hits:
            value = _first_number(hits[-1])
            if value is not None:
                return value
        boxed = _BOXED_RE.findall(segment)
        if boxed:
            value = _first_number(boxed[-1])
            if value is not None:
                return value
        numbers = extract_numbers(segment)
        if numbers:
            return numbers[-1]
    return None


def _first_number(fragment: str) -> str | None:
    numbers = extract_numbers(fragment)
    return numbers[0] if numbers else None


def extract_mcq(answer_segment: str, full_text: str) -> str | None:
    """Predicted option letter A-D."""
    for segment in (answer_segment, full_text):
        if not segment:
            continue
        normalized = normalize_digits(segment)
        hits = _HASH_RE.findall(normalized)
        if hits:
            letter = _first_letter(hits[-1])
            if letter:
                return letter
        boxed = _BOXED_RE.findall(normalized)
        if boxed:
            letter = _first_letter(boxed[-1])
            if letter:
                return letter
        # Last standalone A-D anywhere in the segment.
        letters = _LETTER_RE.findall(normalized)
        if letters:
            return letters[-1].upper()
    return None


def _first_letter(fragment: str) -> str | None:
    match = _LETTER_RE.search(fragment.upper())
    return match.group(1) if match else None


def extract_prediction(task_kind: str, answer_segment: str, full_text: str) -> str | None:
    if task_kind == "numeric":
        return extract_numeric(answer_segment, full_text)
    if task_kind == "mcq":
        extracted = extract_mcq(answer_segment, full_text)
        return extracted if extracted in MCQ_LETTERS else None
    raise ValueError(f"unknown task kind {task_kind!r}")


def is_correct(task_kind: str, prediction: str | None, gold: str | None) -> bool:
    if prediction is None or gold is None:
        return False
    if task_kind == "numeric":
        left = canonical_number(prediction)
        right = canonical_number(gold)
        return left is not None and left == right
    return prediction.strip().upper() == gold.strip().upper()
