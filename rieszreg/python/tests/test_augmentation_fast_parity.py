"""Parity tests for the vectorised ``build_augmented_fast`` path.

For each built-in estimand factory, verify that the fast vectorised
augmentation produces an :class:`AugmentedDataset` equivalent (modulo
row order) to what the slow symbolic ``build_augmented`` produces.

"Equivalent" means:
  - Same set of (rounded) augmented rows.
  - For each unique row, the same (sum_is_original, sum_C, count) when
    aggregated across the slow and fast paths.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from rieszreg.augmentation import build_augmented, build_augmented_fast
from rieszreg.estimands import ATE, ATT, AdditiveShift, LocalShift, TSM


def _make_features(n=100, p=2, seed=0, treatment_levels=(0.0, 1.0)):
    """Generate a small (n, p) feature ndarray with discrete treatment +
    continuous covariates."""
    rng = np.random.default_rng(seed)
    a = rng.choice(treatment_levels, size=n).astype(float)
    x = rng.normal(0.0, 1.0, size=(n, p - 1))
    return np.column_stack([a, x])


def _features_to_rows(features, feature_keys):
    return [
        {k: features[i, j] for j, k in enumerate(feature_keys)}
        for i in range(features.shape[0])
    ]


def _compare_aug(slow, fast):
    """Compare two AugmentedDatasets modulo row ordering."""
    assert slow.n_rows == fast.n_rows, (slow.n_rows, fast.n_rows)
    assert slow.features.shape == fast.features.shape

    def _signature(aug):
        # (origin_index, rounded feature row, is_orig, pdc) tuple per row.
        out = []
        for i in range(aug.features.shape[0]):
            row = tuple(round(x, 9) for x in aug.features[i])
            out.append((
                int(aug.origin_index[i]),
                row,
                round(float(aug.is_original[i]), 9),
                round(float(aug.potential_deriv_coef[i]), 9),
            ))
        return sorted(out)

    sig_slow = _signature(slow)
    sig_fast = _signature(fast)
    assert sig_slow == sig_fast, (
        f"Augmented datasets differ:\n  slow={sig_slow[:3]}...\n"
        f"  fast={sig_fast[:3]}..."
    )


@pytest.mark.parametrize(
    "estimand_factory, treatment_levels",
    [
        (lambda: ATE(treatment="a", covariates=("x0", "x1")), (0.0, 1.0)),
        (lambda: ATT(treatment="a", covariates=("x0", "x1")), (0.0, 1.0)),
        (lambda: TSM(treatment="a", covariates=("x0", "x1"), level=1.0), (0.0, 1.0)),
        (lambda: TSM(treatment="a", covariates=("x0", "x1"), level=0.0), (0.0, 1.0)),
        (
            lambda: AdditiveShift(delta=0.5, treatment="a", covariates=("x0", "x1")),
            (0.0, 1.0, 2.0),
        ),
        (
            lambda: LocalShift(
                delta=0.3, threshold=1.5, treatment="a", covariates=("x0", "x1")
            ),
            (0.0, 1.0, 2.0),
        ),
    ],
)
def test_fast_path_matches_slow(estimand_factory, treatment_levels):
    """For every built-in factory, the fast path produces the same
    augmented dataset (modulo row ordering) as the slow symbolic path."""
    estimand = estimand_factory()
    features = _make_features(n=50, p=3, treatment_levels=treatment_levels)
    rows = _features_to_rows(features, estimand.feature_keys)
    slow = build_augmented(rows, estimand)
    fast = build_augmented_fast(features, estimand)
    assert fast is not None, "fast path returned None unexpectedly"
    _compare_aug(slow, fast)


def test_fast_path_returns_none_for_custom_estimand():
    """Custom estimands (no factory_spec) must return None so the
    caller falls back to the symbolic tracer."""
    from rieszreg import FiniteEvalEstimand

    def m(alpha):
        def inner(z, y=None):
            return alpha(a=z["a"], x=z["x"])
        return inner

    custom = FiniteEvalEstimand(feature_keys=("a", "x"), m=m, name="custom")
    features = _make_features(n=20, p=2)
    assert build_augmented_fast(features, custom) is None


def test_fast_path_returns_none_when_y_supplied():
    """Built-in estimands ignore y; if the caller supplies y, fall back
    to the slow path so behaviour stays identical (the slow path passes
    y through to the user's m)."""
    estimand = ATE(treatment="a", covariates=("x0",))
    features = _make_features(n=20, p=2)
    y = np.zeros(20)
    assert build_augmented_fast(features, estimand, y) is None


def test_fast_path_is_meaningfully_faster_than_slow():
    """Fast path should be ≥ 20× faster than the symbolic tracer on a
    moderate dataset. The headline ratio is much higher (~100×); 20×
    is a generous floor that survives noise across machines."""
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(10)))
    features = _make_features(n=2000, p=11)
    rows = _features_to_rows(features, estimand.feature_keys)

    n_iters = 5
    t0 = time.perf_counter()
    for _ in range(n_iters):
        build_augmented_fast(features, estimand)
    fast_s = (time.perf_counter() - t0) / n_iters

    t0 = time.perf_counter()
    for _ in range(n_iters):
        build_augmented(rows, estimand)
    slow_s = (time.perf_counter() - t0) / n_iters

    assert fast_s * 20 < slow_s, (
        f"build_augmented_fast should be ≥ 20× faster; "
        f"got fast={fast_s*1e3:.2f} ms vs slow={slow_s*1e3:.2f} ms"
    )
