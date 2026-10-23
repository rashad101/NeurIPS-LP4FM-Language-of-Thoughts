# Thinking-Language Policy Card

*Template accompanying "Position: The Language of Thought Is a Design Choice, Not a
Compliance Problem" (LP4FM @ NeurIPS 2026). Copy this file into your model release
and fill every row. The ask is disclosure, not any particular policy.*

## 1. Model

| Field | Value |
| --- | --- |
| Model | |
| Release date | |
| Reasoning mode | |
| Tokenizer | |

## 2. Declared thinking-language policy

> One sentence: what language does this model reason in, as a function of the input
> language, and was that an explicit design choice?

- **Policy:** `consistency` | `english-hub` | `free` | `mixed-anchored` | `adaptive` | `undisclosed`
- **Enforced by:** prompt / SFT / RL reward / constrained decoding / nothing
- **User-visible trace:** yes / no / configurable

## 3. Per-language disclosure

For each supported language, report the measured cost of the shipped policy against
a *free-thinking* baseline on a verifiable-reasoning benchmark. State the benchmark,
the number of items, and the decoding config.

| Language | Script | Shipped policy | Δ accuracy vs free (pp) | Δ completion tokens vs free (%) | Δ faithfulness vs free | Tokenizer fertility vs English (×) | Trace-language compliance (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | |

## 4. Task-conditioning

Does the policy vary by task type? (§7 of the position paper argues it should.)

| Task type | Policy | Justification |
| --- | --- | --- |
| Verifiable reasoning (math, code, logic) | | |
| Cultural / contextual | | |
| Safety-sensitive | | |
| Open generation | | |

## 5. Justification and known costs

- Why this policy, for these languages?
- Which languages pay the largest accuracy tax under it, and how large is it?
- If the policy is `consistency`: what accuracy budget was allocated to it, and was
  the budget met?
- What would make you change the policy?

## 6. Reproducibility

| Field | Value |
| --- | --- |
| Benchmark(s) and split | |
| Items per language | |
| Decoding config | |
| Seeds | |
| Trace-language measurement method | |
| Faithfulness measurement method | |
