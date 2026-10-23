# Thinking-Language Policy Card — Qwen/Qwen3-8B

*Filled automatically by `ltd policy-card` from this repository's own measurements, as the worked example accompanying "Position: The Language of Thought Is a Design Choice, Not a Compliance Problem".*

Generated: 2026-08-12

## 1. Model

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen3-8B` |
| Reasoning mode | switchable |
| Decoding | {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0} |
| Max model len | 16384 |

## 2. Measured policy landscape

Deltas are against **Free-CoT** (no thinking-language constraint), which is the baseline the position paper asks releases to disclose against, because it is the model's own preferred medium.

### Task: `gmmlu`

| Language | Script | Policy | Accuracy | Δacc vs Free (pp) | Mean tokens | Δtok vs Free (%) | Anchor retention | Δretention vs Free | Compliance | Trunc. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| English (en) | Latin | Target-CoT | 0.880 | -0.5 | 1605 | -10.9 | 0.863 | -0.003 | 100% | 2% |
| English (en) | Latin | En-CoT | 0.870 | -1.5 | 1536 | -14.8 | 0.845 | -0.021 | 100% | 0% |
| English (en) | Latin | Free-CoT | 0.885 | +0.0 | 1802 | +0.0 | 0.866 | +0.000 | 100% | 4% |
| English (en) | Latin | Mixed-CoT | 0.865 | -2.0 | 1416 | -21.4 | 0.856 | -0.010 | 100% | 2% |
| German (de) | Latin | Target-CoT | 0.805 | -3.5 | 1357 | -12.5 | 0.591 | +0.255 | 55% | 2% |
| German (de) | Latin | En-CoT | 0.820 | -2.0 | 1474 | -4.9 | 0.318 | -0.018 | 0% | 2% |
| German (de) | Latin | Free-CoT | 0.840 | +0.0 | 1550 | +0.0 | 0.336 | +0.000 | 0% | 2% |
| German (de) | Latin | Mixed-CoT | 0.835 | -0.5 | 1370 | -11.6 | 0.364 | +0.028 | 0% | 2% |
| Spanish (es) | Latin | Target-CoT | 0.825 | -3.0 | 1332 | -10.2 | 0.355 | +0.061 | 0% | 2% |
| Spanish (es) | Latin | En-CoT | 0.830 | -2.5 | 1357 | -8.5 | 0.272 | -0.021 | 0% | 1% |
| Spanish (es) | Latin | Free-CoT | 0.855 | +0.0 | 1483 | +0.0 | 0.293 | +0.000 | 0% | 2% |
| Spanish (es) | Latin | Mixed-CoT | 0.835 | -2.0 | 1334 | -10.0 | 0.351 | +0.058 | 0% | 1% |
| Swahili (sw) | Latin | Target-CoT | 0.475 | -2.5 | 2818 | +33.9 | 0.557 | -0.096 | 0% | 18% |
| Swahili (sw) | Latin | En-CoT | 0.485 | -1.5 | 1923 | -8.6 | 0.569 | -0.084 | 0% | 2% |
| Swahili (sw) | Latin | Free-CoT | 0.500 | +0.0 | 2105 | +0.0 | 0.653 | +0.000 | 0% | 2% |
| Swahili (sw) | Latin | Mixed-CoT | 0.500 | +0.0 | 2035 | -3.3 | 0.646 | -0.007 | 0% | 6% |
| Hindi (hi) | Devanagari | Target-CoT | 0.765 | -0.5 | 1920 | +11.2 | 0.233 | +0.008 | 0% | 2% |
| Hindi (hi) | Devanagari | En-CoT | 0.780 | +1.0 | 1577 | -8.7 | 0.190 | -0.034 | 0% | 2% |
| Hindi (hi) | Devanagari | Free-CoT | 0.770 | +0.0 | 1727 | +0.0 | 0.224 | +0.000 | 0% | 2% |
| Hindi (hi) | Devanagari | Mixed-CoT | 0.750 | -2.0 | 1614 | -6.5 | 0.268 | +0.043 | 0% | 1% |
| Bengali (bn) | Bengali | Target-CoT | 0.745 | -1.0 | 2007 | +12.1 | 0.225 | -0.011 | 0% | 2% |
| Bengali (bn) | Bengali | En-CoT | 0.720 | -3.5 | 1568 | -12.4 | 0.267 | +0.030 | 0% | 2% |
| Bengali (bn) | Bengali | Free-CoT | 0.755 | +0.0 | 1790 | +0.0 | 0.236 | +0.000 | 0% | 2% |
| Bengali (bn) | Bengali | Mixed-CoT | 0.720 | -3.5 | 1775 | -0.8 | 0.256 | +0.019 | 0% | 1% |
| Telugu (te) | Telugu | Target-CoT | 0.705 | +1.0 | 2230 | +17.8 | 0.252 | -0.021 | 0% | 3% |
| Telugu (te) | Telugu | En-CoT | 0.710 | +1.5 | 1594 | -15.8 | 0.280 | +0.007 | 0% | 2% |
| Telugu (te) | Telugu | Free-CoT | 0.695 | +0.0 | 1893 | +0.0 | 0.273 | +0.000 | 0% | 2% |
| Telugu (te) | Telugu | Mixed-CoT | 0.730 | +3.5 | 1823 | -3.7 | 0.274 | +0.000 | 0% | 0% |

#### Recommended policy under the §7 decision framework

| Language | Recommended | Fertility vs En (×) | Rationale |
| --- | --- | --- | --- |
| English (en) | Mixed-CoT | 1.00 | Pareto-optimal on accuracy × cost (acc=0.865, 1416 tok); indistinguishable from the most accurate policy (Free-CoT, acc=0.885), so the framework default breaks the tie; consistency (Target-CoT) would cost -1.5 pp accuracy and +190 tokens |
| German (de) | Mixed-CoT | 1.55 | Pareto-optimal on accuracy × cost (acc=0.835, 1370 tok); indistinguishable from the most accurate policy (Free-CoT, acc=0.840), so the framework default breaks the tie; consistency (Target-CoT) would cost 3.0 pp accuracy and -13 tokens |
| Spanish (es) | Mixed-CoT | 1.27 | Pareto-optimal on accuracy × cost (acc=0.835, 1334 tok); indistinguishable from the most accurate policy (Free-CoT, acc=0.855), so the framework default breaks the tie; consistency (Target-CoT) would cost 1.0 pp accuracy and -3 tokens |
| Swahili (sw) | Mixed-CoT | 1.91 | Pareto-optimal on accuracy × cost (acc=0.500, 2035 tok); consistency (Target-CoT) would cost 2.5 pp accuracy and +784 tokens |
| Hindi (hi) | En-CoT | 3.64 | Pareto-optimal on accuracy × cost (acc=0.780, 1577 tok); consistency (Target-CoT) would cost 1.5 pp accuracy and +344 tokens |
| Bengali (bn) | En-CoT | 5.27 | Pareto-optimal on accuracy × cost (acc=0.720, 1568 tok); indistinguishable from the most accurate policy (Free-CoT, acc=0.755), so the framework default breaks the tie; consistency (Target-CoT) would cost -2.5 pp accuracy and +439 tokens |
| Telugu (te) | Mixed-CoT | 8.36 | Pareto-optimal on accuracy × cost (acc=0.730, 1823 tok); consistency (Target-CoT) would cost 2.5 pp accuracy and +407 tokens |

### Task: `mgsm`

| Language | Script | Policy | Accuracy | Δacc vs Free (pp) | Mean tokens | Δtok vs Free (%) | Anchor retention | Δretention vs Free | Compliance | Trunc. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| English (en) | Latin | Target-CoT | 0.968 | +0.8 | 1000 | -8.8 | 0.881 | +0.002 | 100% | 0% |
| English (en) | Latin | En-CoT | 0.964 | +0.4 | 1036 | -5.6 | 0.868 | -0.011 | 100% | 0% |
| English (en) | Latin | Free-CoT | 0.960 | +0.0 | 1097 | +0.0 | 0.879 | +0.000 | 100% | 0% |
| English (en) | Latin | Mixed-CoT | 0.972 | +1.2 | 895 | -18.4 | 0.875 | -0.004 | 100% | 0% |
| German (de) | Latin | Target-CoT | 0.908 | -0.4 | 900 | -19.7 | 0.523 | +0.063 | 35% | 0% |
| German (de) | Latin | En-CoT | 0.900 | -1.2 | 966 | -13.8 | 0.399 | -0.061 | 0% | 0% |
| German (de) | Latin | Free-CoT | 0.912 | +0.0 | 1120 | +0.0 | 0.460 | +0.000 | 0% | 0% |
| German (de) | Latin | Mixed-CoT | 0.908 | -0.4 | 968 | -13.6 | 0.436 | -0.024 | 0% | 0% |
| Spanish (es) | Latin | Target-CoT | 0.940 | +0.4 | 956 | -16.2 | 0.378 | -0.103 | 2% | 0% |
| Spanish (es) | Latin | En-CoT | 0.944 | +0.8 | 1004 | -12.0 | 0.403 | -0.078 | 0% | 0% |
| Spanish (es) | Latin | Free-CoT | 0.936 | +0.0 | 1141 | +0.0 | 0.481 | +0.000 | 0% | 0% |
| Spanish (es) | Latin | Mixed-CoT | 0.936 | +0.0 | 947 | -17.0 | 0.424 | -0.057 | 0% | 0% |
| Swahili (sw) | Latin | Target-CoT | 0.576 | -1.6 | 1987 | -18.7 | 0.684 | -0.099 | 0% | 4% |
| Swahili (sw) | Latin | En-CoT | 0.556 | -3.6 | 2221 | -9.1 | 0.765 | -0.018 | 0% | 3% |
| Swahili (sw) | Latin | Free-CoT | 0.592 | +0.0 | 2443 | +0.0 | 0.782 | +0.000 | 0% | 1% |
| Swahili (sw) | Latin | Mixed-CoT | 0.600 | +0.8 | 1840 | -24.7 | 0.733 | -0.049 | 0% | 0% |
| Hindi (hi) | Devanagari | Target-CoT | 0.856 | -2.0 | 1367 | +4.0 | 0.361 | +0.020 | 0% | 0% |
| Hindi (hi) | Devanagari | En-CoT | 0.872 | -0.4 | 1285 | -2.3 | 0.327 | -0.014 | 0% | 0% |
| Hindi (hi) | Devanagari | Free-CoT | 0.876 | +0.0 | 1314 | +0.0 | 0.341 | +0.000 | 0% | 0% |
| Hindi (hi) | Devanagari | Mixed-CoT | 0.880 | +0.4 | 1213 | -7.7 | 0.406 | +0.065 | 0% | 0% |
| Bengali (bn) | Bengali | Target-CoT | 0.884 | +0.0 | 1402 | -8.2 | 0.360 | -0.022 | 0% | 0% |
| Bengali (bn) | Bengali | En-CoT | 0.868 | -1.6 | 1306 | -14.5 | 0.338 | -0.045 | 0% | 0% |
| Bengali (bn) | Bengali | Free-CoT | 0.884 | +0.0 | 1527 | +0.0 | 0.382 | +0.000 | 0% | 0% |
| Bengali (bn) | Bengali | Mixed-CoT | 0.884 | +0.0 | 1363 | -10.7 | 0.350 | -0.033 | 0% | 0% |
| Telugu (te) | Telugu | Target-CoT | 0.808 | -0.4 | 1995 | +6.1 | 0.317 | -0.048 | 0% | 2% |
| Telugu (te) | Telugu | En-CoT | 0.800 | -1.2 | 1653 | -12.1 | 0.340 | -0.025 | 0% | 0% |
| Telugu (te) | Telugu | Free-CoT | 0.812 | +0.0 | 1880 | +0.0 | 0.365 | +0.000 | 0% | 1% |
| Telugu (te) | Telugu | Mixed-CoT | 0.816 | +0.4 | 1870 | -0.5 | 0.410 | +0.045 | 0% | 0% |

#### Recommended policy under the §7 decision framework

| Language | Recommended | Fertility vs En (×) | Rationale |
| --- | --- | --- | --- |
| English (en) | Mixed-CoT | 1.00 | Pareto-optimal on accuracy × cost (acc=0.972, 895 tok); consistency (Target-CoT) would cost 0.4 pp accuracy and +105 tokens |
| German (de) | Free-CoT | 1.40 | Pareto-optimal on accuracy × cost (acc=0.912, 1120 tok); consistency (Target-CoT) would cost 0.4 pp accuracy and -221 tokens |
| Spanish (es) | Mixed-CoT | 1.28 | Pareto-optimal on accuracy × cost (acc=0.936, 947 tok); indistinguishable from the most accurate policy (En-CoT, acc=0.944), so the framework default breaks the tie; consistency (Target-CoT) would cost -0.4 pp accuracy and +9 tokens |
| Swahili (sw) | Mixed-CoT | 1.82 | Pareto-optimal on accuracy × cost (acc=0.600, 1840 tok); consistency (Target-CoT) would cost 2.4 pp accuracy and +147 tokens |
| Hindi (hi) | Mixed-CoT | 3.36 | Pareto-optimal on accuracy × cost (acc=0.880, 1213 tok); consistency (Target-CoT) would cost 2.4 pp accuracy and +154 tokens |
| Bengali (bn) | Mixed-CoT | 4.72 | Pareto-optimal on accuracy × cost (acc=0.884, 1363 tok); consistency (Target-CoT) would cost 0.0 pp accuracy and +39 tokens |
| Telugu (te) | Mixed-CoT | 7.65 | Pareto-optimal on accuracy × cost (acc=0.816, 1870 tok); consistency (Target-CoT) would cost 0.8 pp accuracy and +125 tokens |

## 3. Caveats this card must carry

- These are **prompted** policies measured at inference time, not trained ones. The training-side price of consistency is reported independently at 5–10 pp (Cross-lingual Collapse, arXiv 2506.05850) and agrees in sign with this map.
- Retention is verbatim anchor retention, a *lower bound* on faithfulness.
- Rows for `en` where Target-CoT / En-CoT / Mixed-CoT coincide are degenerate by construction and measure prompt-wording noise, not a language effect.
- Truncation rates differ across languages because target-script traces are longer; read accuracy together with the truncation column.

## 4. What the position paper asks of releases

Disclose the shipped policy and its measured per-language cost, rather than defaulting to language consistency and leaving the tax unmeasured. See `artifacts/policy_card_template.md` for the blank form.
