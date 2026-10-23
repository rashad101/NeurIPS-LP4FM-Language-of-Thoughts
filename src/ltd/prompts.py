"""The four thinking-language conditions and their prompt rendering.

Design invariants (spec §5) — the whole experiment rests on these:

* The **question** is always in the target language L, byte-identical across
  conditions and models.
* The **answer-format** instruction is identical across conditions, so answer
  extraction cannot differ by condition.
* The only thing that varies across conditions is the **thinking-language
  instruction**.
* With ``instruction_language="en"`` the instruction *text* is also held constant
  across languages (only the language *name* changes), which is what removes the
  instruction-quality confound from the accuracy comparison. The ``native``
  variant exists for the robustness appendix.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

from .config import CONDITION_LABELS, Config, LanguageSpec, ModelSpec, TaskSpec
from .env import CONFIGS_DIR

SYSTEM_PROMPT = "You are a careful problem solver."

# English instruction blocks. `{L}` is the target language's English name.
EN_THINKING_INSTRUCTIONS: dict[str, str] = {
    "target": (
        "Write your entire chain of thought in {L}. Every step of your reasoning "
        "must be written in {L}, and give the final answer in {L}."
    ),
    "en_cot": (
        "Write your entire chain of thought in English. Every step of your "
        "reasoning must be written in English, but give the final answer in {L}."
    ),
    "free": (
        "Think step by step. Think in whichever language you can reason best in."
    ),
    "mixed": (
        "Reason mainly in English, but keep the names, quantities, quotations and "
        "culture-specific terms from the question in {L}, exactly as they are "
        "written. Give the final answer in {L}."
    ),
}

EN_ANSWER_INSTRUCTIONS: dict[str, str] = {
    "numeric": (
        "At the end of your response, write the final numeric answer on its own "
        "line in this form: #### <number>"
    ),
    "mcq": (
        "At the end of your response, write the final option on its own line in "
        "this form: #### <letter>, where <letter> is A, B, C or D."
    ),
}

# For Free-CoT the answer instruction must not name a language, otherwise the
# condition would smuggle in a constraint. `free` therefore only gets the format
# rule, and the target-language answer requirement is dropped by design — this is
# what makes Free-CoT reveal the model's *preferred* medium (spec §5, condition 3).
FREE_ANSWER_SUFFIX = ""


@functools.lru_cache(maxsize=4)
def load_native_instructions(path: str | None = None) -> dict[str, dict[str, Any]]:
    cfg_path = Path(path) if path else CONFIGS_DIR / "native_instructions.yaml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def thinking_instruction(condition: str, lang: LanguageSpec, instruction_language: str) -> str:
    if condition not in CONDITION_LABELS:
        raise KeyError(f"unknown condition {condition!r}")
    if instruction_language == "native":
        native = load_native_instructions()
        block = native.get(lang.code)
        if block is None or condition not in block:
            raise KeyError(
                f"no native instruction for language={lang.code} condition={condition}; "
                "add it to configs/native_instructions.yaml"
            )
        return str(block[condition]).strip()
    return EN_THINKING_INSTRUCTIONS[condition].format(L=lang.name).strip()


def answer_instruction(task: TaskSpec, lang: LanguageSpec, instruction_language: str) -> str:
    key = "answer_numeric" if task.kind == "numeric" else "answer_mcq"
    if instruction_language == "native":
        native = load_native_instructions()
        block = native.get(lang.code, {})
        if key in block:
            return str(block[key]).strip()
    return EN_ANSWER_INSTRUCTIONS[task.kind].strip()


def format_question(task: TaskSpec, item: dict[str, Any]) -> str:
    """Render the item body in the target language."""
    if task.kind == "numeric":
        return item["question"].strip()
    options = item["options"]  # list of 4 strings, already in the target language
    letters = ["A", "B", "C", "D"]
    lines = [item["question"].strip(), ""]
    lines += [f"{letter}. {text.strip()}" for letter, text in zip(letters, options, strict=True)]
    return "\n".join(lines)


def build_user_message(
    *,
    task: TaskSpec,
    lang: LanguageSpec,
    condition: str,
    item: dict[str, Any],
    instruction_language: str = "en",
) -> str:
    """Assemble the user turn: question, then thinking instruction, then format."""
    parts = [format_question(task, item), ""]
    think = thinking_instruction(condition, lang, instruction_language)
    if think:
        parts.append(think)
    parts.append(answer_instruction(task, lang, instruction_language))
    return "\n".join(parts).strip()


def build_messages(
    *,
    task: TaskSpec,
    lang: LanguageSpec,
    condition: str,
    item: dict[str, Any],
    instruction_language: str = "en",
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_message(
                task=task,
                lang=lang,
                condition=condition,
                item=item,
                instruction_language=instruction_language,
            ),
        },
    ]


def render_prompt(tokenizer: Any, model: ModelSpec, messages: list[dict[str, str]]) -> str:
    """Apply the model's chat template, enabling thinking mode where switchable.

    * ``qwen3``: pass ``enable_thinking=True``; the template then leaves the
      ``<think>`` block for the model to open.
    * ``r1_distill``: the official template already appends ``<think>\\n``; we do
      not pass extra kwargs. Some community mirrors of the template omit it, so
      we append it ourselves when it is missing — the trace parser tolerates both.
    """
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if model.thinking == "switchable":
        kwargs["enable_thinking"] = True
    try:
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Template does not accept enable_thinking (older transformers / mirror).
        kwargs.pop("enable_thinking", None)
        prompt = tokenizer.apply_chat_template(messages, **kwargs)

    if model.family == "r1_distill" and "<think>" not in prompt[-64:]:
        prompt = prompt + "<think>\n"
    return prompt


def degenerate_condition(lang_code: str, condition: str) -> bool:
    """True when a condition is definitionally identical to Target-CoT.

    For the English anchor, ``target`` and ``en_cot`` (and ``mixed``) request the
    same thing. We still run them — the accuracy spread is a useful measure of
    prompt-wording noise, and the paper reports it as such — but analysis marks
    these cells so they are never read as a language effect.
    """
    return lang_code == "en" and condition in {"en_cot", "mixed"}


def prompt_inventory(config: Config, instruction_language: str | None = None) -> list[dict[str, str]]:
    """Every (task, language, condition) instruction block, for release/review."""
    instr_lang = instruction_language or config.instruction_language
    rows = []
    for task_name in config.tasks:
        task = config.task(task_name)
        for lang_code in config.languages:
            lang = config.language(lang_code)
            for condition in config.conditions:
                try:
                    think = thinking_instruction(condition, lang, instr_lang)
                except KeyError as exc:
                    think = f"<MISSING: {exc}>"
                rows.append(
                    {
                        "task": task_name,
                        "language": lang_code,
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "instruction_language": instr_lang,
                        "system": SYSTEM_PROMPT,
                        "thinking_instruction": think,
                        "answer_instruction": answer_instruction(task, lang, instr_lang),
                        "degenerate": str(degenerate_condition(lang_code, condition)),
                    }
                )
    return rows
