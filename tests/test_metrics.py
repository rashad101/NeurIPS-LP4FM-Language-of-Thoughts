"""Checks for the load-bearing scoring logic.

Every accuracy number in the paper flows through answer extraction and every
compliance/faithfulness number through the script and anchor code, so these are
the pieces that must not be silently wrong. Run with::

    .venv/bin/python -m pytest tests -q
    .venv/bin/python tests/test_metrics.py      # no pytest needed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ltd.extract import (  # noqa: E402
    extract_prediction,
    is_correct,
    split_trace_answer,
)
from ltd.generate import GenerateSelection, resolve_isolation  # noqa: E402
from ltd.faithfulness import entity_anchors, number_anchors, retention  # noqa: E402
from ltd.stats_utils import dominates, pareto_front  # noqa: E402
from ltd.text_utils import canonical_number, extract_numbers, normalize_digits  # noqa: E402
from ltd.trace_language import analyse_trace  # noqa: E402


# ---------------------------------------------------------------------------
# Unicode digits — the spec §9 mitigation
# ---------------------------------------------------------------------------


def test_normalize_digits_across_scripts():
    assert normalize_digits("१८") == "18"          # Devanagari
    assert normalize_digits("১৮") == "18"          # Bengali
    assert normalize_digits("౧౮") == "18"          # Telugu
    assert normalize_digits("١٨") == "18"          # Arabic-Indic
    assert normalize_digits("１８") == "18"          # fullwidth
    assert normalize_digits("18") == "18"           # ASCII untouched
    # Non-digit characters must survive unchanged.
    assert normalize_digits("उत्तर: १८ डॉलर") == "उत्तर: 18 डॉलर"


def test_canonical_number_forms():
    assert canonical_number("18") == "18"
    assert canonical_number("18.0") == "18"
    assert canonical_number("18.00") == "18"
    assert canonical_number("1,234") == "1234"      # Anglo thousands
    assert canonical_number("1.234,56") == "1234.56"  # European
    assert canonical_number("1,234.56") == "1234.56"  # Anglo decimal
    assert canonical_number("-7") == "-7"
    assert canonical_number("$18") is None
    assert canonical_number("abc") is None
    assert canonical_number("१८") == "18"
    # A decimal comma with a non-3-digit tail is a decimal point, not a separator.
    assert canonical_number("3,5") == "3.5"


def test_extract_numbers_order():
    assert extract_numbers("she sells 9 eggs at $2 each for 18 dollars") == ["9", "2", "18"]
    assert extract_numbers("१६ - ३ - ४ = ९") == ["16", "3", "4", "9"]


# ---------------------------------------------------------------------------
# Trace / answer split
# ---------------------------------------------------------------------------


def test_split_both_tags():
    trace, answer, ok = split_trace_answer("<think>\nreasoning here\n</think>\nThe answer.\n#### 18")
    assert trace == "reasoning here"
    assert "#### 18" in answer
    assert ok is True


def test_split_close_tag_only_r1_style():
    """The R1 template pre-opens <think>, so only the close tag is in the output."""
    trace, answer, ok = split_trace_answer("reasoning only\n</think>\nAnswer: 18")
    assert trace == "reasoning only"
    assert answer == "Answer: 18"
    assert ok is True


def test_split_truncated_mid_thought():
    trace, answer, ok = split_trace_answer("<think>\nstill reasoning when the budget ran out")
    assert trace == "still reasoning when the budget ran out"
    assert answer == ""
    assert ok is False


def test_split_no_tags_flags_malformed():
    trace, answer, ok = split_trace_answer("Just an answer. #### 18")
    assert ok is False
    # Both halves are the whole text: a non-thinking output still has a language
    # and still carries an answer.
    assert trace == answer == "Just an answer. #### 18"


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def test_numeric_prefers_hash_form():
    assert extract_prediction("numeric", "blah 99 blah\n#### 18", "") == "18"


def test_numeric_native_digits_and_boxed():
    assert extract_prediction("numeric", "उत्तर\n#### १८", "") == "18"
    assert extract_prediction("numeric", r"so \boxed{18}", "") == "18"
    assert extract_prediction("numeric", "the total is 18 dollars", "") == "18"


def test_numeric_falls_back_to_full_text_when_truncated():
    """A generation truncated after the answer must still score."""
    full = "<think>16-3-4=9, 9*2=18, so #### 18 and then it kept going"
    assert extract_prediction("numeric", "", full) == "18"


def test_numeric_returns_none_without_numbers():
    assert extract_prediction("numeric", "no digits at all", "") is None


def test_mcq_extraction():
    assert extract_prediction("mcq", "#### B", "") == "B"
    assert extract_prediction("mcq", "the answer is (C)", "") == "C"
    assert extract_prediction("mcq", r"\boxed{D}", "") == "D"
    # A bare letter inside a word must not be picked up.
    assert extract_prediction("mcq", "Antwort: A", "") == "A"
    assert extract_prediction("mcq", "no option here", "") is None


def test_is_correct_normalises_both_sides():
    assert is_correct("numeric", "18.0", "18")
    assert is_correct("numeric", "१८", "18")
    assert not is_correct("numeric", "19", "18")
    assert not is_correct("numeric", None, "18")
    assert is_correct("mcq", "b", "B")
    assert not is_correct("mcq", "A", "B")


# ---------------------------------------------------------------------------
# Trace language / compliance / drift
# ---------------------------------------------------------------------------


def test_devanagari_trace_is_compliant_and_low_drift():
    stats = analyse_trace("जेनेट के पास सोलह अंडे हैं और वह तीन खाती है", "hi")
    assert stats.target_script_share > 0.95
    assert stats.compliant is True
    assert stats.english_share < 0.05


def test_english_trace_under_hindi_target_is_full_drift():
    stats = analyse_trace("Janet has sixteen eggs and she eats three of them", "hi")
    assert stats.target_share < 0.05
    assert stats.english_share > 0.95
    assert stats.compliant is False
    assert stats.has_target_anchor is False


def test_mixed_trace_keeps_anchor_without_compliance():
    text = (
        "Janet has 16 eggs. She eats 3 for breakfast. "
        "The question calls them अंडे so I keep that word."
    )
    stats = analyse_trace(text, "hi")
    assert stats.has_target_anchor is True      # anchors retained
    assert stats.compliant is False             # but predominantly English
    assert 0.0 < stats.target_share < 0.5


def test_latin_script_target_uses_langid_not_script():
    """de/es/sw are Latin, so script ranges cannot separate them from English."""
    german = analyse_trace(
        "Janet hat sechzehn Eier und sie isst jeden Morgen drei davon zum Fruehstueck. "
        "Danach backt sie Muffins fuer ihre Freunde mit vier weiteren Eiern.",
        "de",
    )
    english = analyse_trace(
        "Janet has sixteen eggs and she eats three of them every morning for breakfast. "
        "After that she bakes muffins for her friends using four more eggs.",
        "de",
    )
    # Both are ~100% Latin script; only langid can tell them apart.
    assert german.latin_script_share > 0.95 and english.latin_script_share > 0.95
    assert german.target_share > english.target_share
    assert english.english_share > german.english_share


def test_empty_trace_is_safe():
    stats = analyse_trace("", "bn")
    assert stats.alpha_chars == 0
    assert stats.compliant is False
    assert stats.target_share == 0.0


# ---------------------------------------------------------------------------
# Faithfulness anchors (H5)
# ---------------------------------------------------------------------------


def test_number_anchors_deduplicate_and_normalise():
    assert number_anchors("16 eggs, 3 eaten, 4 baked, 16 again") == ["16", "3", "4"]
    assert number_anchors("१६ अंडे और ३ खाए") == ["16", "3"]


def test_entity_anchors_skip_stopwords_short_tokens_and_numbers():
    anchors = entity_anchors("Janet has 16 eggs and the farmers market pays her", "en")
    assert "janet" in anchors
    assert "farmers" in anchors
    assert "the" not in anchors      # stopword
    assert "has" not in anchors      # shorter than min length
    assert "16" not in anchors       # numbers are a separate anchor family


def test_retention_rewards_verbatim_preservation():
    question = "जेनेट के पास 16 अंडे हैं और वह 3 खाती है"
    mixed_trace = "Janet has 16 eggs, eats 3, the question says अंडे"
    en_trace = "Janet has sixteen eggs and eats three of them"

    mixed = retention(question, mixed_trace, "hi")
    english = retention(question, en_trace, "hi")

    # Numbers survive in the mixed trace but are spelled out in the English one.
    assert mixed.number_retention == 1.0
    assert english.number_retention == 0.0
    # The target-language anchor survives only in the mixed trace.
    assert (mixed.entity_retention or 0) > (english.entity_retention or 0)
    assert 0.0 <= (mixed.anchor_retention or 0) <= 1.0


def test_retention_is_none_without_anchors():
    stats = retention("", "some trace", "hi")
    assert stats.number_retention is None
    assert stats.anchor_retention is None


# ---------------------------------------------------------------------------
# Pareto logic (H3)
# ---------------------------------------------------------------------------


def test_pareto_front_maximises_accuracy_minimises_cost():
    # (accuracy, tokens): B is better than A on both axes -> A is dominated.
    points = [(0.50, 900), (0.60, 700), (0.55, 500), (0.40, 400)]
    front = set(pareto_front(points))
    assert 0 not in front            # (0.50, 900) dominated by (0.60, 700)
    assert 1 in front                # best accuracy
    assert 2 in front                # cheaper, slightly worse
    assert 3 in front                # cheapest


def test_dominates_is_strict_in_at_least_one_axis():
    assert dominates((0.6, 700), (0.5, 900))
    assert not dominates((0.6, 700), (0.6, 700))    # identical: no domination
    assert not dominates((0.6, 900), (0.5, 700))    # better acc but worse cost


# ---------------------------------------------------------------------------
# Model isolation — models must never co-reside in GPU memory
# ---------------------------------------------------------------------------


def _selection(**overrides):
    base = dict(
        models=["r1-llama-8b", "qwen3-8b"], tasks=["mgsm"], languages=["hi", "bn"],
        conditions=["target", "en_cot"],
    )
    base.update(overrides)
    return GenerateSelection(**base)


def test_auto_isolation_uses_subprocess_for_multiple_vllm_models():
    """Two 8B models cannot co-reside at gpu_memory_utilization=0.9 on 48 GB."""
    assert resolve_isolation(_selection(), "vllm") == "subprocess"


def test_auto_isolation_stays_inprocess_for_a_single_model():
    assert resolve_isolation(_selection(models=["qwen3-8b"]), "vllm") == "inprocess"


def test_auto_isolation_stays_inprocess_on_the_hf_backend():
    # The transformers backend does not pre-reserve a KV cache, and dry runs
    # should not pay for process churn.
    assert resolve_isolation(_selection(), "hf") == "inprocess"


def test_explicit_isolation_overrides_auto():
    assert resolve_isolation(_selection(model_isolation="subprocess"), "hf") == "subprocess"
    assert resolve_isolation(_selection(model_isolation="inprocess"), "vllm") == "inprocess"


def test_child_argv_preserves_the_selection_and_cannot_recurse():
    """A wrong argv here would silently run a different experiment."""
    selection = _selection(
        limit=25, n_samples=1, chunk_size=32, backend="vllm",
        max_new_tokens={"mgsm": 4096}, instruction_language="native",
        config_path="configs/default.yaml",
    )
    argv = selection.child_argv("qwen3-8b", "main-full")
    joined = " ".join(argv)

    assert argv[0] == "generate"
    # Exactly one model, and it is the one requested.
    assert argv[argv.index("--models") + 1] == "qwen3-8b"
    assert "r1-llama-8b" not in joined
    # The child must not spawn grandchildren.
    assert "--model-isolation inprocess" in joined
    # Every selection dimension survives the round trip.
    assert "--run-id main-full" in joined
    assert "--tasks mgsm" in joined
    assert "--langs hi bn" in joined
    assert "--conditions target en_cot" in joined
    assert "--limit 25" in joined
    assert "--chunk-size 32" in joined
    assert "--backend vllm" in joined
    assert "--max-new-tokens 4096" in joined
    assert "--instruction-language native" in joined
    assert "--config configs/default.yaml" in joined


def test_child_argv_omits_absent_optional_flags():
    argv = " ".join(_selection().child_argv("qwen3-8b", "r1"))
    assert "--limit" not in argv
    assert "--max-new-tokens" not in argv
    assert "--instruction-language" not in argv
    assert "--config" not in argv


# ---------------------------------------------------------------------------
# Runner for use without pytest
# ---------------------------------------------------------------------------


def _main() -> int:
    tests = {
        name: obj
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    }
    failures = []
    for name, fn in tests.items():
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failures.append((name, f"AssertionError: {exc}"))
            print(f"  FAIL  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    for name, message in failures:
        print(f"  {name}: {message}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
