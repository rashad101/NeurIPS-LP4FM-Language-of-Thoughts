"""Bootstrap CIs, paired tests and Pareto logic used by the analysis."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def stable_seed(*parts: Any) -> int:
    """Reproducible seed from arbitrary labels.

    Python salts ``hash()`` for ``str``, so deriving bootstrap seeds from it would
    give different confidence intervals on every invocation. Every CI in the paper
    must be reproducible from the same parquet, so seeds go through SHA-256.
    """
    material = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


@dataclass
class Estimate:
    mean: float
    lo: float
    hi: float
    n: int

    def as_tuple(self) -> tuple[float, float, float]:
        return self.mean, self.lo, self.hi

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


def bootstrap_mean(
    values: Sequence[float],
    iters: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap CI for a mean (accuracy, tokens, retention...)."""
    array = np.asarray([v for v in values if v is not None and not _isnan(v)], dtype=float)
    if array.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    if array.size == 1:
        return Estimate(float(array[0]), float(array[0]), float(array[0]), 1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(iters, array.size))
    means = array[draws].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Estimate(float(array.mean()), float(lo), float(hi), int(array.size))


def bootstrap_paired_diff(
    a: Sequence[float],
    b: Sequence[float],
    iters: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """CI for mean(a) - mean(b) on *paired* items (same item ids, resampled together).

    Pairing is what makes the accuracy differences between conditions tight: the
    item difficulty variance cancels.
    """
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.size != right.size:
        raise ValueError(f"paired bootstrap needs equal lengths, got {left.size} vs {right.size}")
    mask = ~(np.isnan(left) | np.isnan(right))
    left, right = left[mask], right[mask]
    if left.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    diff = left - right
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, diff.size, size=(iters, diff.size))
    means = diff[draws].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Estimate(float(diff.mean()), float(lo), float(hi), int(diff.size))


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> tuple[int, int, float]:
    """Exact McNemar test on paired binary outcomes. Returns (b01, b10, p)."""
    from scipy import stats

    left = np.asarray(a_correct, dtype=bool)
    right = np.asarray(b_correct, dtype=bool)
    b01 = int(np.sum(~left & right))
    b10 = int(np.sum(left & ~right))
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    p = float(stats.binomtest(b10, n, 0.5, alternative="two-sided").pvalue)
    return b01, b10, p


def logistic_drift_regression(
    english_share: Sequence[float],
    correct: Sequence[bool],
    extra: dict[str, Sequence[float]] | None = None,
) -> dict[str, object]:
    """H4: does English drift inside Target-CoT predict being correct?

    ``correct ~ 1 + english_share (+ controls)``, logit link. Returns coefficients,
    standard errors, p-values and pseudo-R², or a diagnostic when the fit is
    impossible (no variation in the regressor or the outcome).
    """
    import pandas as pd
    import statsmodels.api as sm

    data = {"english_share": np.asarray(english_share, dtype=float)}
    if extra:
        for name, values in extra.items():
            data[name] = np.asarray(values, dtype=float)
    frame = pd.DataFrame(data)
    outcome = pd.Series(np.asarray(correct, dtype=float), name="correct")

    mask = frame.notna().all(axis=1) & outcome.notna()
    frame, outcome = frame[mask], outcome[mask]

    if len(frame) < 20:
        return {"status": "insufficient_data", "n": int(len(frame))}
    if outcome.nunique() < 2:
        return {"status": "no_outcome_variation", "n": int(len(frame)),
                "accuracy": float(outcome.mean())}
    if frame["english_share"].std() < 1e-9:
        return {"status": "no_regressor_variation", "n": int(len(frame)),
                "english_share": float(frame["english_share"].mean())}

    design = sm.add_constant(frame, has_constant="add")
    try:
        result = sm.Logit(outcome, design).fit(disp=0, maxiter=200)
    except Exception as exc:  # separation, singular design, ...
        return {"status": f"fit_failed: {type(exc).__name__}: {exc}", "n": int(len(frame))}

    return {
        "status": "ok",
        "n": int(result.nobs),
        "params": {k: float(v) for k, v in result.params.items()},
        "std_err": {k: float(v) for k, v in result.bse.items()},
        "p_values": {k: float(v) for k, v in result.pvalues.items()},
        "conf_int": {
            k: [float(row[0]), float(row[1])]
            for k, row in result.conf_int().iterrows()
        },
        "pseudo_r2": float(result.prsquared),
        "odds_ratio_per_10pp_drift": float(np.exp(0.10 * result.params["english_share"])),
    }


def pareto_front(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of points not dominated on (maximise x, minimise y).

    Here x = accuracy, y = token cost: a point is dominated when another has
    accuracy >= and cost <=, with at least one strict.
    """
    front: list[int] = []
    for i, (xi, yi) in enumerate(points):
        dominated = False
        for j, (xj, yj) in enumerate(points):
            if i == j:
                continue
            if xj >= xi and yj <= yi and (xj > xi or yj < yi):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Does ``a`` Pareto-dominate ``b`` on (max accuracy, min cost)?"""
    return a[0] >= b[0] and a[1] <= b[1] and (a[0] > b[0] or a[1] < b[1])


def spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Spearman correlation, returning NaN rather than warning on degenerate input."""
    from scipy import stats

    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    mask = ~(np.isnan(left) | np.isnan(right))
    if mask.sum() < 3:
        return float("nan"), float("nan")
    left, right = left[mask], right[mask]
    # A constant input makes rho undefined; scipy warns and returns NaN. Dry runs
    # and single-language selections hit this routinely, so handle it quietly.
    if left.std() < 1e-12 or right.std() < 1e-12:
        return float("nan"), float("nan")
    result = stats.spearmanr(left, right)
    return float(result.statistic), float(result.pvalue)


def _isnan(value: object) -> bool:
    try:
        return bool(np.isnan(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
