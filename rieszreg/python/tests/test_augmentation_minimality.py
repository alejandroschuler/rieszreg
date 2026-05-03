"""Pin the per-subject augmented-row counts for built-in estimands.

The tracer + augmentation engine should generate exactly the rows the loss
needs and no more. Specifically, for ATT and LocalShift, subjects whose
multiplicative factor (`a` or `1{a < threshold}`) is zero should contribute
only one augmented row (the original observation), not phantom counterfactual
rows with zero coefficients. This matches Lee & Schuler (2025)'s ATT
augmentation layout — counterfactuals only for treated subjects — and means
our generic `Estimand.m → trace → build_augmented` pipeline produces the
minimal augmentation automatically (no estimand-specific code path required).

If the tracer ever stops short-circuiting on multiplicative zeros, this test
fails and surfaces wasted work for ATT-style partial-parameter estimands.
"""

from __future__ import annotations

import collections

import numpy as np

from rieszreg import ATE, ATT, LocalShift, build_augmented


def _make_rows(a: np.ndarray, x: np.ndarray) -> list[dict]:
    return [{"a": float(a[i]), "x": float(x[i])} for i in range(len(a))]


def test_ate_two_rows_per_subject():
    """ATE always touches both treatment levels → 2 augmented rows per subject."""
    rng = np.random.default_rng(0)
    n = 50
    a = rng.binomial(1, 0.5, n).astype(float)
    x = rng.uniform(0, 1, n)
    aug = build_augmented(_make_rows(a, x), ATE())

    assert aug.features.shape[0] == 2 * n
    counts = collections.Counter(aug.origin_index.tolist())
    assert set(counts.values()) == {2}, counts


def test_att_treated_two_rows_untreated_one_row():
    """ATT (partial) skips counterfactuals for untreated subjects.

    Untreated rows have `a_i = 0`, so the multiplicative factor in
    `m(α)(z) = a · (α(1, x) − α(0, x))` evaluates to 0 before the tracer
    sees any α(...) call. The tracer returns [] for those subjects, and
    only the original observation row gets emitted by build_augmented.

    Total expected augmented rows = n + n_treated.
    """
    rng = np.random.default_rng(0)
    n = 200
    a = rng.binomial(1, 0.5, n).astype(float)
    x = rng.uniform(0, 1, n)
    n_treated = int(a.sum())

    aug = build_augmented(_make_rows(a, x), ATT())
    assert aug.features.shape[0] == n + n_treated, (
        f"expected {n + n_treated} augmented rows (1 per untreated, 2 per "
        f"treated); got {aug.features.shape[0]}"
    )

    counts = collections.Counter(aug.origin_index.tolist())
    treated_counts = {counts[i] for i in range(n) if a[i] == 1}
    untreated_counts = {counts[i] for i in range(n) if a[i] == 0}
    assert treated_counts == {2}, f"treated row counts: {treated_counts}"
    assert untreated_counts == {1}, f"untreated row counts: {untreated_counts}"

    # For untreated subjects the single row should be the original
    # observation: is_original = 1, potential_deriv_coef = 0.
    for i in range(n):
        if a[i] == 0:
            mask = aug.origin_index == i
            j = int(np.where(mask)[0][0])
            assert aug.is_original[j] == 1.0
            assert aug.potential_deriv_coef[j] == 0.0


def test_local_shift_skips_above_threshold_subjects():
    """LocalShift multiplies by 1{a < threshold}, so subjects above the
    threshold contribute only their original row."""
    rng = np.random.default_rng(0)
    n = 200
    a = rng.uniform(-1, 1, n)
    x = rng.uniform(0, 1, n)
    threshold = 0.0
    n_below = int((a < threshold).sum())

    aug = build_augmented(_make_rows(a, x), LocalShift(delta=0.1, threshold=threshold))
    # 1 row per subject above threshold + 2 rows per subject below.
    expected = n + n_below
    assert aug.features.shape[0] == expected, (
        f"expected {expected} augmented rows; got {aug.features.shape[0]}"
    )

    counts = collections.Counter(aug.origin_index.tolist())
    above = {counts[i] for i in range(n) if a[i] >= threshold}
    below = {counts[i] for i in range(n) if a[i] < threshold}
    assert above == {1}, above
    assert below == {2}, below
