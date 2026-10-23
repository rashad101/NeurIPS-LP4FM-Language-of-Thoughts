# Thinking-Language Policy Card — deepseek-ai/DeepSeek-R1-Distill-Llama-8B

*Filled automatically by `ltd policy-card` from this repository's own measurements, as the worked example accompanying "Position: The Language of Thought Is a Design Choice, Not a Compliance Problem".*

Generated: 2026-08-12

## 1. Model

| Field | Value |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| Reasoning mode | always |
| Decoding | {'temperature': 0.6, 'top_p': 0.95, 'top_k': -1, 'min_p': 0.0} |
| Max model len | 16384 |

## 2. Measured policy landscape

Deltas are against **Free-CoT** (no thinking-language constraint), which is the baseline the position paper asks releases to disclose against, because it is the model's own preferred medium.

### Task: `gmmlu`

| Language | Script | Policy | Accuracy | Δacc vs Free (pp) | Mean tokens | Δtok vs Free (%) | Anchor retention | Δretention vs Free | Compliance | Trunc. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| English (en) | Latin | Target-CoT | 0.575 | -4.0 | 1431 | +15.1 | 0.116 | -0.006 | 0% | 6% |
| English (en) | Latin | En-CoT | 0.600 | -1.5 | 1343 | +8.0 | 0.112 | -0.010 | 0% | 4% |
| English (en) | Latin | Free-CoT | 0.615 | +0.0 | 1243 | +0.0 | 0.122 | +0.000 | 0% | 4% |
| English (en) | Latin | Mixed-CoT | 0.570 | -4.5 | 1338 | +7.6 | 0.117 | -0.006 | 0% | 6% |
| German (de) | Latin | Target-CoT | 0.355 | -6.5 | 1337 | +15.5 | 0.101 | +0.007 | 0% | 4% |
| German (de) | Latin | En-CoT | 0.385 | -3.5 | 1097 | -5.2 | 0.073 | -0.021 | 0% | 2% |
| German (de) | Latin | Free-CoT | 0.420 | +0.0 | 1157 | +0.0 | 0.094 | +0.000 | 0% | 3% |
| German (de) | Latin | Mixed-CoT | 0.465 | +4.5 | 1457 | +25.9 | 0.077 | -0.017 | 0% | 5% |
| Spanish (es) | Latin | Target-CoT | 0.325 | -7.0 | 1341 | +9.6 | 0.080 | +0.008 | 0% | 4% |
| Spanish (es) | Latin | En-CoT | 0.350 | -4.5 | 1661 | +35.8 | 0.077 | +0.005 | 0% | 4% |
| Spanish (es) | Latin | Free-CoT | 0.395 | +0.0 | 1223 | +0.0 | 0.072 | +0.000 | 0% | 2% |
| Spanish (es) | Latin | Mixed-CoT | 0.385 | -1.0 | 1512 | +23.6 | 0.085 | +0.013 | 0% | 4% |
| Swahili (sw) | Latin | Target-CoT | 0.220 | +1.5 | 2624 | +42.8 | 0.090 | -0.013 | 100% | 18% |
| Swahili (sw) | Latin | En-CoT | 0.285 | +8.0 | 1959 | +6.6 | 0.135 | +0.033 | 100% | 4% |
| Swahili (sw) | Latin | Free-CoT | 0.205 | +0.0 | 1838 | +0.0 | 0.103 | +0.000 | 100% | 6% |
| Swahili (sw) | Latin | Mixed-CoT | 0.340 | +13.5 | 1901 | +3.4 | 0.105 | +0.002 | 100% | 8% |
| Hindi (hi) | Devanagari | Target-CoT | 0.240 | -1.5 | 2754 | +12.9 | 0.069 | -0.004 | 0% | 13% |
| Hindi (hi) | Devanagari | En-CoT | 0.240 | -1.5 | 2466 | +1.1 | 0.067 | -0.006 | 0% | 11% |
| Hindi (hi) | Devanagari | Free-CoT | 0.255 | +0.0 | 2439 | +0.0 | 0.073 | +0.000 | 0% | 14% |
| Hindi (hi) | Devanagari | Mixed-CoT | 0.255 | +0.0 | 1837 | -24.7 | 0.072 | -0.001 | 0% | 4% |
| Bengali (bn) | Bengali | Target-CoT | 0.270 | +4.5 | 2401 | -19.3 | 0.055 | -0.004 | 0% | 10% |
| Bengali (bn) | Bengali | En-CoT | 0.155 | -7.0 | 2341 | -21.3 | 0.064 | +0.005 | 0% | 9% |
| Bengali (bn) | Bengali | Free-CoT | 0.225 | +0.0 | 2975 | +0.0 | 0.058 | +0.000 | 0% | 20% |
| Bengali (bn) | Bengali | Mixed-CoT | 0.210 | -1.5 | 1907 | -35.9 | 0.053 | -0.006 | 0% | 8% |
| Telugu (te) | Telugu | Target-CoT | 0.225 | -8.0 | 3215 | +17.0 | 0.071 | +0.006 | 0% | 18% |
| Telugu (te) | Telugu | En-CoT | 0.250 | -5.5 | 3190 | +16.1 | 0.069 | +0.003 | 0% | 20% |
| Telugu (te) | Telugu | Free-CoT | 0.305 | +0.0 | 2748 | +0.0 | 0.066 | +0.000 | 0% | 19% |
| Telugu (te) | Telugu | Mixed-CoT | 0.215 | -9.0 | 2594 | -5.6 | 0.052 | -0.014 | 0% | 14% |

#### Recommended policy under the §7 decision framework

| Language | Recommended | Fertility vs En (×) | Rationale |
| --- | --- | --- | --- |
| English (en) | Free-CoT | 1.00 | Pareto-optimal on accuracy × cost (acc=0.615, 1243 tok); consistency (Target-CoT) would cost 4.0 pp accuracy and +188 tokens |
| German (de) | Mixed-CoT | 1.58 | Pareto-optimal on accuracy × cost (acc=0.465, 1457 tok); consistency (Target-CoT) would cost 11.0 pp accuracy and -120 tokens |
| Spanish (es) | Free-CoT | 1.34 | Pareto-optimal on accuracy × cost (acc=0.395, 1223 tok); consistency (Target-CoT) would cost 7.0 pp accuracy and +117 tokens |
| Swahili (sw) | Mixed-CoT | 1.56 | Pareto-optimal on accuracy × cost (acc=0.340, 1901 tok); consistency (Target-CoT) would cost 12.0 pp accuracy and +723 tokens |
| Hindi (hi) | none — invest in capability | n/a (instrumentation artifact) | every policy is at or below the floor for this task (best accuracy 0.255); the framework asks for disclosure of the gap, not a policy over it |
| Bengali (bn) | none — invest in capability | n/a (instrumentation artifact) | every policy is at or below the floor for this task (best accuracy 0.270); the framework asks for disclosure of the gap, not a policy over it |
| Telugu (te) | Free-CoT | n/a (instrumentation artifact) | Pareto-optimal on accuracy × cost (acc=0.305, 2748 tok); consistency (Target-CoT) would cost 8.0 pp accuracy and +466 tokens |

### Task: `mgsm`

| Language | Script | Policy | Accuracy | Δacc vs Free (pp) | Mean tokens | Δtok vs Free (%) | Anchor retention | Δretention vs Free | Compliance | Trunc. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| English (en) | Latin | Target-CoT | 0.716 | +4.8 | 527 | +15.5 | 0.161 | -0.010 | 0% | 0% |
| English (en) | Latin | En-CoT | 0.700 | +3.2 | 494 | +8.3 | 0.171 | +0.000 | 0% | 0% |
| English (en) | Latin | Free-CoT | 0.668 | +0.0 | 457 | +0.0 | 0.171 | +0.000 | 0% | 0% |
| English (en) | Latin | Mixed-CoT | 0.712 | +4.4 | 548 | +20.1 | 0.178 | +0.007 | 0% | 0% |
| German (de) | Latin | Target-CoT | 0.172 | +2.8 | 680 | +7.2 | 0.134 | -0.012 | 0% | 1% |
| German (de) | Latin | En-CoT | 0.208 | +6.4 | 855 | +34.8 | 0.136 | -0.010 | 0% | 2% |
| German (de) | Latin | Free-CoT | 0.144 | +0.0 | 635 | +0.0 | 0.146 | +0.000 | 0% | 1% |
| German (de) | Latin | Mixed-CoT | 0.216 | +7.2 | 1083 | +70.7 | 0.140 | -0.007 | 0% | 3% |
| Spanish (es) | Latin | Target-CoT | 0.124 | +4.0 | 1098 | +7.5 | 0.146 | +0.011 | 0% | 2% |
| Spanish (es) | Latin | En-CoT | 0.132 | +4.8 | 2018 | +97.5 | 0.162 | +0.027 | 0% | 7% |
| Spanish (es) | Latin | Free-CoT | 0.084 | +0.0 | 1022 | +0.0 | 0.135 | +0.000 | 0% | 3% |
| Spanish (es) | Latin | Mixed-CoT | 0.096 | +1.2 | 1767 | +72.9 | 0.158 | +0.023 | 0% | 5% |
| Swahili (sw) | Latin | Target-CoT | 0.036 | +2.0 | 4949 | +65.0 | 0.160 | -0.021 | 100% | 28% |
| Swahili (sw) | Latin | En-CoT | 0.000 | -1.6 | 2632 | -12.3 | 0.163 | -0.018 | 100% | 6% |
| Swahili (sw) | Latin | Free-CoT | 0.016 | +0.0 | 3000 | +0.0 | 0.181 | +0.000 | 100% | 9% |
| Swahili (sw) | Latin | Mixed-CoT | 0.020 | +0.4 | 3392 | +13.1 | 0.167 | -0.014 | 100% | 14% |
| Hindi (hi) | Devanagari | Target-CoT | 0.004 | -0.8 | 2632 | +25.1 | 0.133 | +0.013 | 0% | 12% |
| Hindi (hi) | Devanagari | En-CoT | 0.012 | +0.0 | 2832 | +34.6 | 0.144 | +0.023 | 0% | 14% |
| Hindi (hi) | Devanagari | Free-CoT | 0.012 | +0.0 | 2105 | +0.0 | 0.121 | +0.000 | 0% | 10% |
| Hindi (hi) | Devanagari | Mixed-CoT | 0.004 | -0.8 | 2946 | +40.0 | 0.126 | +0.005 | 0% | 14% |
| Bengali (bn) | Bengali | Target-CoT | 0.024 | +2.0 | 3968 | +72.8 | 0.114 | +0.017 | 0% | 21% |
| Bengali (bn) | Bengali | En-CoT | 0.012 | +0.8 | 3542 | +54.2 | 0.124 | +0.027 | 0% | 18% |
| Bengali (bn) | Bengali | Free-CoT | 0.004 | +0.0 | 2297 | +0.0 | 0.097 | +0.000 | 0% | 13% |
| Bengali (bn) | Bengali | Mixed-CoT | 0.016 | +1.2 | 3017 | +31.3 | 0.111 | +0.014 | 0% | 14% |
| Telugu (te) | Telugu | Target-CoT | 0.004 | -0.4 | 3674 | +180.5 | 0.103 | +0.018 | 0% | 18% |
| Telugu (te) | Telugu | En-CoT | 0.020 | +1.2 | 2280 | +74.1 | 0.097 | +0.012 | 0% | 7% |
| Telugu (te) | Telugu | Free-CoT | 0.008 | +0.0 | 1310 | +0.0 | 0.085 | +0.000 | 0% | 6% |
| Telugu (te) | Telugu | Mixed-CoT | 0.016 | +0.8 | 2944 | +124.8 | 0.095 | +0.010 | 0% | 11% |

#### Recommended policy under the §7 decision framework

| Language | Recommended | Fertility vs En (×) | Rationale |
| --- | --- | --- | --- |
| English (en) | En-CoT | 1.00 | Pareto-optimal on accuracy × cost (acc=0.700, 494 tok); indistinguishable from the most accurate policy (Target-CoT, acc=0.716), so the framework default breaks the tie; consistency (Target-CoT) would cost -1.6 pp accuracy and +33 tokens |
| German (de) | Mixed-CoT | 1.50 | Pareto-optimal on accuracy × cost (acc=0.216, 1083 tok); consistency (Target-CoT) would cost 4.4 pp accuracy and -403 tokens |
| Spanish (es) | En-CoT | 1.35 | Pareto-optimal on accuracy × cost (acc=0.132, 2018 tok); consistency (Target-CoT) would cost 0.8 pp accuracy and -920 tokens |
| Swahili (sw) | none — invest in capability | 1.56 | every policy is at or below the floor for this task (best accuracy 0.036); the framework asks for disclosure of the gap, not a policy over it |
| Hindi (hi) | none — invest in capability | n/a (instrumentation artifact) | every policy is at or below the floor for this task (best accuracy 0.012); the framework asks for disclosure of the gap, not a policy over it |
| Bengali (bn) | none — invest in capability | n/a (instrumentation artifact) | every policy is at or below the floor for this task (best accuracy 0.024); the framework asks for disclosure of the gap, not a policy over it |
| Telugu (te) | none — invest in capability | n/a (instrumentation artifact) | every policy is at or below the floor for this task (best accuracy 0.020); the framework asks for disclosure of the gap, not a policy over it |

## 3. Caveats this card must carry

- These are **prompted** policies measured at inference time, not trained ones. The training-side price of consistency is reported independently at 5–10 pp (Cross-lingual Collapse, arXiv 2506.05850) and agrees in sign with this map.
- Retention is verbatim anchor retention, a *lower bound* on faithfulness.
- Rows for `en` where Target-CoT / En-CoT / Mixed-CoT coincide are degenerate by construction and measure prompt-wording noise, not a language effect.
- Truncation rates differ across languages because target-script traces are longer; read accuracy together with the truncation column.

## 4. What the position paper asks of releases

Disclose the shipped policy and its measured per-language cost, rather than defaulting to language consistency and leaving the tax unmeasured. See `artifacts/policy_card_template.md` for the blank form.
