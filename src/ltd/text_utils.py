"""Unicode-aware text normalisation shared by data building and scoring.

The Indic-numeral handling here is the mitigation named in spec §9 ("extraction
noise for Indic numerals -> Unicode digit normalization").
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation, localcontext

import regex

# Unicode blocks we care about, as (name, start, end) inclusive codepoint ranges.
SCRIPT_RANGES: list[tuple[str, int, int]] = [
    ("Latin", 0x0041, 0x005A),
    ("Latin", 0x0061, 0x007A),
    ("Latin", 0x00C0, 0x024F),
    ("Greek", 0x0370, 0x03FF),
    ("Cyrillic", 0x0400, 0x04FF),
    ("Hebrew", 0x0590, 0x05FF),
    ("Arabic", 0x0600, 0x06FF),
    ("Devanagari", 0x0900, 0x097F),
    ("DevanagariExt", 0xA8E0, 0xA8FF),
    ("Bengali", 0x0980, 0x09FF),
    ("Gurmukhi", 0x0A00, 0x0A7F),
    ("Gujarati", 0x0A80, 0x0AFF),
    ("Oriya", 0x0B00, 0x0B7F),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F),
    ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F),
    ("Sinhala", 0x0D80, 0x0DFF),
    ("Thai", 0x0E00, 0x0E7F),
    ("Hangul", 0xAC00, 0xD7AF),
    ("Hangul", 0x1100, 0x11FF),
    ("Hiragana", 0x3040, 0x309F),
    ("Katakana", 0x30A0, 0x30FF),
    ("Han", 0x4E00, 0x9FFF),
    ("Ethiopic", 0x1200, 0x137F),
]

# Aliases so "DevanagariExt" folds into "Devanagari".
_SCRIPT_ALIAS = {"DevanagariExt": "Devanagari"}

# Word tokenisation MUST use the `regex` module, not `re`. Python's built-in `re`
# defines \w via str.isalnum(), which is False for Unicode combining marks
# (category Mn/Mc). Devanagari, Bengali and Telugu write vowels as such marks, so
# `re.findall(r"\w+", "जेनेट")` returns ['ज','न','ट'] — every Indic word shatters
# at each matra. That silently corrupts word counts (hence tokenizer fertility,
# H3) and anchor extraction (hence entity retention, H5) for exactly the
# languages this paper is about. `regex`'s \w includes \p{M}.
_WORD_RE = regex.compile(r"[\p{L}\p{M}\p{N}_]+")
_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")

# Longest digit run treated as a number; see `canonical_number`.
_MAX_NUMBER_DIGITS = 4300


def script_of(char: str) -> str | None:
    """Return the script name of an alphabetic character, else ``None``."""
    if not char.isalpha():
        return None
    code = ord(char)
    for name, start, end in SCRIPT_RANGES:
        if start <= code <= end:
            return _SCRIPT_ALIAS.get(name, name)
    try:
        # Fall back to the Unicode name's first token ("TELUGU LETTER KA" -> TELUGU).
        return unicodedata.name(char).split()[0].title()
    except ValueError:
        return None


def script_histogram(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for char in text:
        name = script_of(char)
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts


def script_fractions(text: str) -> tuple[dict[str, float], int]:
    """Per-script share of alphabetic characters, plus the alphabetic char count."""
    counts = script_histogram(text)
    total = sum(counts.values())
    if total == 0:
        return {}, 0
    return {name: count / total for name, count in counts.items()}, total


def normalize_digits(text: str) -> str:
    """Map every Unicode decimal digit to its ASCII equivalent.

    Handles Devanagari (१२३), Bengali (১২৩), Telugu (౧౨౩), Arabic-Indic, fullwidth
    digits, and anything else with a Unicode decimal value.
    """
    out = []
    for char in text:
        if char.isdigit() and not char.isascii():
            try:
                out.append(str(unicodedata.decimal(char)))
                continue
            except (TypeError, ValueError):
                try:
                    out.append(str(unicodedata.digit(char)))
                    continue
                except (TypeError, ValueError):
                    pass
        out.append(char)
    return "".join(out)


def normalize_text(text: str) -> str:
    """NFKC + digit normalisation + whitespace collapse."""
    text = unicodedata.normalize("NFKC", text)
    text = normalize_digits(text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_number(token: str) -> str | None:
    """Canonicalise a numeric string for exact match.

    Strips thousands separators and trailing zeros so that ``18``, ``18.0``,
    ``18.00`` and ``1,8 00`` -> ``18``. Returns ``None`` if not numeric.
    """
    token = normalize_digits(token).strip()
    token = token.replace(" ", "").replace(" ", "")
    token = token.rstrip(".")
    negative = token.startswith("-")
    token = token.lstrip("+-")
    if not token:
        return None

    # Decide whether ',' and '.' are separators or decimal points.
    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):  # 1.234,56 (European)
            token = token.replace(".", "").replace(",", ".")
        else:  # 1,234.56 (Anglo)
            token = token.replace(",", "")
    elif "," in token:
        parts = token.split(",")
        if len(parts) == 2 and len(parts[1]) != 3:
            token = token.replace(",", ".")  # decimal comma
        else:
            token = token.replace(",", "")  # thousands
    if token.count(".") > 1:
        head, _, tail = token.rpartition(".")
        token = head.replace(".", "") + "." + tail

    if not re.fullmatch(r"\d*\.?\d+|\d+\.?", token):
        return None
    token = token.rstrip(".")

    # Degenerate generations (a model looping on a digit) yield runs of hundreds
    # of digits. `float()` overflows those to `inf` and `int(inf)` raises
    # OverflowError; CPython also caps int<->str conversion at 4300 digits. Such
    # runs are never a real answer, so drop them rather than crash the scorer.
    if len(token) - ("." in token) > _MAX_NUMBER_DIGITS:
        return None

    # Decimal, not float: exact past 2**53 and immune to the inf overflow above.
    # The local context must hold every digit of the token, else rounding a long
    # value raises InvalidOperation instead of returning it.
    with localcontext() as ctx:
        ctx.prec = len(token) + 7
        try:
            value = Decimal(token)
        except InvalidOperation:
            return None
        if negative:
            value = -value
        value = round(value, 6)
        if value == value.to_integral_value():
            return str(int(value))
        return format(value.normalize(), "f")


def extract_numbers(text: str) -> list[str]:
    """All canonical numbers appearing in ``text`` (digits normalised first)."""
    normalized = normalize_digits(text)
    out = []
    for match in _NUMBER_RE.finditer(normalized):
        canon = canonical_number(match.group(0))
        if canon is not None:
            out.append(canon)
    return out


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def word_count(text: str) -> int:
    return len(words(text))


def tokens_in_script(text: str, script_name: str) -> list[str]:
    """Word tokens whose alphabetic characters are mostly in ``script_name``.

    Used to count *anchors* — a few preserved target-language words inside an
    otherwise English trace. A character-share measure cannot see those (one
    Devanagari word in a 500-character English trace is <1% of characters), which
    is exactly the Mixed-CoT signature H2 is about, so anchors are counted in
    tokens instead.
    """
    out = []
    for token in words(text):
        counts = script_histogram(token)
        total = sum(counts.values())
        if total and counts.get(script_name, 0) / total > 0.5:
            out.append(token)
    return out
