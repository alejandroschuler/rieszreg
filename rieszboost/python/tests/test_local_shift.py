"""LocalShift partial-parameter estimand."""

import numpy as np

from rieszboost.engine import build_augmented, fit
from rieszboost.estimands import LocalShift
from rieszboost.tracer import trace


def test_local_shift_traces_below_threshold():
    m = LocalShift(delta=1.0, threshold=0.0)
    pairs = trace(m, {"a": -0.5, "x": 0.3})
    coefs = sorted(c for c, _ in pairs)
    assert coefs == [-1.0, 1.0]
    sample_a = sorted(p["a"] for _, p in pairs)
    assert sample_a == [-0.5, 0.5]


def test_local_shift_above_threshold_contributes_nothing():
    m = LocalShift(delta=1.0, threshold=0.0)
    pairs = trace(m, {"a": 0.5, "x": 0.3})
    assert pairs == []


def test_local_shift_at_threshold_excluded():
    """A == threshold is NOT < threshold — boundary excluded."""
    m = LocalShift(delta=1.0, threshold=0.0)
    pairs = trace(m, {"a": 0.0, "x": 0.0})
    assert pairs == []


def test_local_shift_recovers_truth_on_continuous_dgp():
    """Lee-Schuler Section 4.2 LASE setup (partial parameter).
    DGP: X ~ U(0,2), A|X ~ N(X^2 - 1, 2). Shift delta=1, threshold t=0.
    True partial-LASE representer:
        alpha_0(A, X) = 1(A < t+δ) * p(A-δ|X)/p(A|X) - 1(A < t)
    Under A|X ~ N(X^2-1, 2), the density ratio is exp((2(A-X^2)+1)/4).
    """
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(0, 2, n)
    a = rng.normal(x**2 - 1.0, np.sqrt(2.0))
    rows = [{"a": float(ai), "x": float(xi)} for ai, xi in zip(a, x)]

    delta, t = 1.0, 0.0
    n_tr = int(0.8 * n)
    booster = fit(
        rows[:n_tr],
        LocalShift(delta=delta, threshold=t),
        feature_keys=("a", "x"),
        valid_rows=rows[n_tr:],
        num_boost_round=2000,
        early_stopping_rounds=20,
        learning_rate=0.05,
        max_depth=3,
        reg_lambda=10.0,
        seed=0,
    )
    alpha_hat = booster.predict(rows)

    density_ratio = np.exp((2 * (a - x**2) + 1) / 4)
    alpha_true = (a < t + delta).astype(float) * density_ratio - (a < t).astype(float)
    rmse = float(np.sqrt(np.mean((alpha_hat - alpha_true) ** 2)))
    # Lee-Schuler Table 4 reports RieszBoost LASE α-RMSE at n=500: 0.252.
    # Loose threshold here: smoke test.
    assert rmse < 0.6, f"RMSE {rmse:.3f} too high"


def test_local_shift_augmentation_skips_above_threshold():
    rows = [{"a": -0.5, "x": 0.0}, {"a": 0.5, "x": 0.0}]
    aug = build_augmented(
        rows, LocalShift(delta=1.0, threshold=0.0), feature_keys=("a", "x")
    )
    # Below-threshold row: 2 rows (post-merge of original (-0.5,0) with linear term).
    # Above-threshold row: just the original alpha^2 row at (0.5, 0).
    assert aug.features.shape == (3, 2)
    above_idx = np.where(aug.origin_index == 1)[0]
    assert aug.a[above_idx].sum() == 1.0
    assert aug.b[above_idx].sum() == 0.0
