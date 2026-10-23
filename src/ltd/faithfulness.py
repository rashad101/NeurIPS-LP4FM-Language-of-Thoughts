"""Input-faithfulness proxies — the third leg of the landscape (H5).

Prior work discusses faithfulness of multilingual traces qualitatively. We need a
number, and it has to be cheap, language-neutral and auditable. We use *anchor
retention*: how much of the question's own material survives into the trace.

Two anchor families, reported separately because they behave differently:

``number_retention``
    Share of the question's distinct numbers (Unicode-normalised) that appear in
    the trace. A trace that drops or garbles quantities is unfaithful in a way
    that directly threatens correctness.

``entity_retention``
    Share of the question's distinct content tokens (length >= ``min_token_len``,
    not in the stoplist) that appear verbatim in the trace. Under Target-CoT these
    are target-language words; under En-CoT the model must translate them, so they
    vanish; Mixed-CoT explicitly asks for them to be preserved. This is exactly
    the axis H5 predicts Mixed-CoT wins.

Both are *proxies*, and the paper must present them as such: verbatim retention
is a lower bound on faithfulness (a correct translation of an entity is faithful
but scores zero here). The comparison across conditions is still meaningful
because the measure is applied identically to every condition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .text_utils import extract_numbers, normalize_digits, words

#: Tiny, hand-written stoplists. Deliberately short: these exist to stop function
#: words dominating the anchor set, not to be exhaustive lexicons.
STOPWORDS: dict[str, set[str]] = {
    "en": {
        "the", "and", "that", "with", "for", "from", "this", "have", "has", "had",
        "she", "he", "they", "them", "their", "there", "then", "than", "each",
        "many", "much", "does", "did", "how", "what", "when", "which", "who",
        "will", "would", "were", "was", "are", "his", "her", "its", "into",
        "also", "been", "being", "other", "some", "more", "most", "such", "only",
        "very", "about", "after", "before", "every", "these", "those", "because",
    },
    "de": {
        "der", "die", "das", "und", "eine", "einen", "einem", "einer", "eines",
        "nicht", "sich", "auch", "aber", "wenn", "dann", "also", "wird", "wurde",
        "haben", "hatte", "sind", "waren", "wie", "was", "welche", "jeder",
        "viele", "mehr", "sehr", "über", "nach", "vor", "bei", "mit", "für",
        "von", "dem", "den", "des", "ihre", "seine", "diese", "dieser",
    },
    "es": {
        "que", "los", "las", "una", "unos", "unas", "del", "con", "por", "para",
        "como", "más", "pero", "sus", "este", "esta", "estos", "estas", "cada",
        "todos", "todas", "muy", "sobre", "entre", "cuando", "donde", "cuál",
        "cuánto", "cuántos", "tiene", "tienen", "hace", "hacen", "son", "era",
        "fue", "ser", "está", "están", "hay", "también", "porque",
    },
    "sw": {
        "kwa", "wa", "ya", "na", "ni", "katika", "kila", "hii", "hiyo", "hizo",
        "yake", "yao", "zake", "ana", "wana", "kuna", "lakini", "pia", "kama",
        "ambayo", "ambao", "baada", "kabla", "zaidi", "sana", "ngapi", "wote",
        "kwenye", "juu", "chini", "hadi", "alikuwa", "walikuwa",
    },
    # Indic stoplists: common postpositions/auxiliaries. Short words are already
    # excluded by min_token_len, so only the longer function words matter here.
    "hi": {
        "करता", "करती", "करते", "किया", "किये", "कितने", "कितना", "कितनी",
        "उसके", "उसकी", "उनके", "उनकी", "इसके", "इसकी", "प्रत्येक", "लेकिन",
        "इसलिए", "क्योंकि", "जिससे", "साथ", "लिए", "अपने", "अपनी", "होता",
        "होती", "होते", "रहा", "रही", "रहे", "और", "एक",
    },
    "bn": {
        "করেন", "করেছেন", "করে", "কতটি", "কতগুলি", "তার", "তাদের", "তাঁর",
        "এবং", "কিন্তু", "প্রতিটি", "সাথে", "জন্য", "কারণে", "থেকে", "হয়েছে",
        "হচ্ছে", "একটি", "সেই", "যেটি", "যারা", "মোট",
    },
    "te": {
        "చేస్తాడు", "చేస్తుంది", "చేసారు", "ఎన్ని", "ఎంత", "అతని", "ఆమె",
        "వారి", "మరియు", "కానీ", "ప్రతి", "కోసం", "నుండి", "వల్ల", "ఉంది",
        "ఉన్నాయి", "అయితే", "మొత్తం", "ఒక",
    },
}


@dataclass
class RetentionStats:
    n_number_anchors: int = 0
    n_numbers_retained: int = 0
    number_retention: float | None = None
    n_entity_anchors: int = 0
    n_entities_retained: int = 0
    entity_retention: float | None = None
    anchor_retention: float | None = None  # numbers + entities pooled

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def number_anchors(question: str) -> list[str]:
    """Distinct canonical numbers in the question, order-preserving."""
    seen: set[str] = set()
    out: list[str] = []
    for number in extract_numbers(question):
        if number not in seen:
            seen.add(number)
            out.append(number)
    return out


def entity_anchors(question: str, lang_code: str, min_token_len: int = 4) -> list[str]:
    """Distinct content tokens of the question, lowercased, order-preserving."""
    stop = STOPWORDS.get(lang_code, set())
    seen: set[str] = set()
    out: list[str] = []
    for token in words(question):
        lowered = token.lower()
        if len(lowered) < min_token_len:
            continue
        if lowered.isdigit() or any(char.isdigit() for char in lowered):
            continue  # numbers are handled by number_anchors
        if lowered in stop:
            continue
        if lowered not in seen:
            seen.add(lowered)
            out.append(lowered)
    return out


def _contains_number(haystack_numbers: Iterable[str], anchor: str) -> bool:
    return anchor in set(haystack_numbers)


def retention(
    question: str,
    trace: str,
    lang_code: str,
    *,
    min_token_len: int = 4,
) -> RetentionStats:
    stats = RetentionStats()
    if not question:
        return stats

    numbers = number_anchors(question)
    stats.n_number_anchors = len(numbers)
    if numbers:
        trace_numbers = set(extract_numbers(trace))
        stats.n_numbers_retained = sum(1 for anchor in numbers if anchor in trace_numbers)
        stats.number_retention = stats.n_numbers_retained / len(numbers)

    entities = entity_anchors(question, lang_code, min_token_len)
    stats.n_entity_anchors = len(entities)
    if entities:
        trace_tokens = {token.lower() for token in words(normalize_digits(trace))}
        stats.n_entities_retained = sum(1 for anchor in entities if anchor in trace_tokens)
        stats.entity_retention = stats.n_entities_retained / len(entities)

    pooled_total = stats.n_number_anchors + stats.n_entity_anchors
    if pooled_total:
        stats.anchor_retention = (
            stats.n_numbers_retained + stats.n_entities_retained
        ) / pooled_total
    return stats


def subsample_item_ids(item_ids: list[str], n: int, seed: int) -> set[str]:
    """Deterministic subsample of item ids (spec §6: retention on 50 items).

    The same items are used for every model/language/condition, which is what
    makes the retention numbers comparable across cells.
    """
    import random

    if n <= 0 or n >= len(item_ids):
        return set(item_ids)
    rng = random.Random(seed)
    return set(rng.sample(sorted(item_ids), n))
