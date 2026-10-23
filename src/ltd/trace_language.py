"""Trace-language composition, compliance, and English drift (spec §6).

Two signals, used where each is reliable:

* **Script ranges** are near-perfect for Devanagari / Bengali / Telugu, which
  cannot be confused with Latin. For these languages the target-language share is
  simply the share of alphabetic characters in the target script.
* **langid** (``py3langid``, offline) is needed for Latin-script targets — German,
  Spanish, Swahili and English are all Latin, so script tells us nothing. We vote
  per *line* and aggregate, which keeps the measure sensitive to code-switching
  within a trace instead of collapsing it to one label.

``english_share`` is the H4 regressor. It is defined so that it means the same
thing in both regimes: the fraction of the trace that is English.
"""

from __future__ import annotations

import functools
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .logging_utils import get_logger
from .text_utils import script_fractions, tokens_in_script, word_count

log = get_logger("trace_lang")

LANG_TO_SCRIPT = {
    "en": "Latin",
    "de": "Latin",
    "es": "Latin",
    "sw": "Latin",
    "hi": "Devanagari",
    "bn": "Bengali",
    "te": "Telugu",
}
LATIN_SCRIPT_LANGS = {code for code, script in LANG_TO_SCRIPT.items() if script == "Latin"}

_SEGMENT_RE = re.compile(r"[^\n.!?۔।॥]+")
_MIN_SEGMENT_CHARS = 12


@dataclass
class TraceLanguageStats:
    target_script_share: float = 0.0
    latin_script_share: float = 0.0
    other_script_share: float = 0.0
    alpha_chars: int = 0
    script_fractions: dict[str, float] = field(default_factory=dict)
    langid_top: str | None = None
    langid_target_share: float = 0.0
    langid_en_share: float = 0.0
    n_segments: int = 0
    n_words: int = 0
    n_target_script_tokens: int = 0
    target_share: float = 0.0     # the headline "how much of the trace is in L"
    english_share: float = 0.0    # the H4 drift regressor
    compliant: bool = False
    has_target_anchor: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@functools.lru_cache(maxsize=1)
def _langid_identifier():
    """A py3langid identifier restricted to our languages, or ``None``."""
    try:
        import py3langid
    except ImportError:
        log.warning(
            "py3langid not installed — Latin-script language attribution falls back "
            "to script ranges only, which cannot separate en/de/es/sw. Install the "
            "package before reporting compliance for Latin-script languages."
        )
        return None
    try:
        from py3langid.langid import LanguageIdentifier, MODEL_FILE

        identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
        identifier.set_languages(sorted(LANG_TO_SCRIPT))
        return identifier
    except Exception as exc:  # pragma: no cover - depends on package internals
        log.warning("could not restrict py3langid to our language set (%s); using defaults", exc)
        return py3langid


def _segments(text: str) -> list[str]:
    return [
        segment.strip()
        for segment in _SEGMENT_RE.findall(text)
        if len(segment.strip()) >= _MIN_SEGMENT_CHARS
    ]


def _langid_shares(text: str, target_lang: str) -> tuple[str | None, float, float, int]:
    identifier = _langid_identifier()
    if identifier is None:
        return None, 0.0, 0.0, 0

    segments = _segments(text)
    if not segments:
        segments = [text.strip()] if text.strip() else []
    if not segments:
        return None, 0.0, 0.0, 0

    votes: dict[str, int] = {}
    for segment in segments:
        try:
            label = identifier.classify(segment)[0]
        except Exception:  # a pathological segment should not kill scoring
            continue
        votes[label] = votes.get(label, 0) + 1
    total = sum(votes.values())
    if total == 0:
        return None, 0.0, 0.0, len(segments)
    top = max(votes, key=lambda key: votes[key])
    return (
        top,
        votes.get(target_lang, 0) / total,
        votes.get("en", 0) / total,
        len(segments),
    )


def analyse_trace(
    text: str,
    target_lang: str,
    *,
    compliance_threshold: float = 0.5,
    anchor_threshold: float = 0.05,
    min_anchor_tokens: int = 1,
) -> TraceLanguageStats:
    stats = TraceLanguageStats()
    if not text or not text.strip():
        return stats

    fractions, alpha = script_fractions(text)
    stats.script_fractions = {k: round(v, 4) for k, v in sorted(fractions.items())}
    stats.alpha_chars = alpha
    target_script = LANG_TO_SCRIPT.get(target_lang, "Latin")
    stats.target_script_share = fractions.get(target_script, 0.0)
    stats.latin_script_share = fractions.get("Latin", 0.0)
    stats.other_script_share = max(
        0.0,
        1.0 - stats.latin_script_share - (
            0.0 if target_script == "Latin" else stats.target_script_share
        ),
    )

    top, target_votes, en_votes, n_segments = _langid_shares(text, target_lang)
    stats.langid_top = top
    stats.langid_target_share = target_votes
    stats.langid_en_share = en_votes
    stats.n_segments = n_segments

    stats.n_words = word_count(text)

    if target_lang in LATIN_SCRIPT_LANGS:
        # Script is uninformative; trust the per-line langid vote.
        stats.target_share = target_votes
        stats.english_share = en_votes if target_lang != "en" else target_votes
        # No script signal for anchors either, so fall back to the share measure.
        stats.has_target_anchor = stats.target_share >= anchor_threshold
    else:
        # Non-Latin target: script share is the ground truth, and anything Latin in
        # the trace is (for these models) overwhelmingly English.
        stats.target_share = stats.target_script_share
        stats.english_share = stats.latin_script_share
        # Anchors are counted in *tokens*, not characters: a handful of preserved
        # target-language words in a long English trace is the Mixed-CoT signature
        # (H2), and it is invisible to any character-share threshold.
        stats.n_target_script_tokens = len(tokens_in_script(text, target_script))
        stats.has_target_anchor = stats.n_target_script_tokens >= min_anchor_tokens

    stats.compliant = stats.target_share >= compliance_threshold
    return stats
