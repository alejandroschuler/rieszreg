"""Tests for rieszreg.testing.dgps + parity helpers."""

from __future__ import annotations

import numpy as np
import pytest

from rieszreg.testing import dgps, parity


def test_linear_gaussian_ate_sample_shape_and_columns():
    dgp = dgps.linear_gaussian_ate()
    rng = np.random.default_rng(0)
    df = dgp.sample(200, rng)
    assert {"a", "x", "y"}.issubset(df.columns)
    assert len(df) == 200
    assert dgp.feature_keys == ("a", "x")
    assert dgp.estimand_factory == "ATE"


def test_linear_gaussian_ate_true_alpha_matches_closed_form():
    dgp = dgps.linear_gaussian_ate()
    rng = np.random.default_rng(0)
    df = dgp.sample(50, rng)
    alpha = dgp.true_alpha(df)
    # Sign flips with treatment.
    assert np.all(alpha[df["a"] == 1.0] > 0)
    assert np.all(alpha[df["a"] == 0.0] < 0)


def test_logistic_tsm_alpha_non_negative():
    dgp = dgps.logistic_tsm(level=1.0)
    rng = np.random.default_rng(1)
    df = dgp.sample(100, rng)
    alpha = dgp.true_alpha(df)
    assert np.all(alpha >= 0)
    assert dgp.estimand_factory == "TSM"


def test_parity_compare_identical_arrays():
    a = np.array([1.0, 2.0, 3.0])
    rep = parity.compare(a, a)
    assert rep.rmse == 0.0
    assert rep.max_abs_diff == 0.0


def test_parity_compare_constant_offset():
    a = np.linspace(0, 1, 10)
    b = a + 0.1
    rep = parity.compare(a, b)
    assert rep.rmse == pytest.approx(0.1)
    assert rep.max_abs_diff == pytest.approx(0.1)
    # Pearson is 1 (perfect linear relationship).
    assert rep.pearson == pytest.approx(1.0, abs=1e-9)


def test_parity_compare_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        parity.compare(np.zeros(3), np.zeros(4))


def test_assert_consistency_passes_for_oracle():
    dgp = dgps.linear_gaussian_ate()

    def oracle(train, test):
        return dgp.true_alpha(test)

    rmses = dgps.assert_consistency(
        oracle, dgp=dgp, n_grid=(50, 100), tol_at_max_n=0.01,
    )
    assert all(r == pytest.approx(0.0, abs=1e-9) for r in rmses)


def test_assert_consistency_rejects_non_decreasing():
    dgp = dgps.linear_gaussian_ate()

    def bad(train, test):
        # Returns increasingly noisy predictions.
        n = len(test)
        return dgp.true_alpha(test) + 5.0 * np.arange(1, n + 1) / n

    with pytest.raises(AssertionError, match="diverged|above tolerance"):
        dgps.assert_consistency(bad, dgp=dgp, n_grid=(50, 60), tol_at_max_n=0.01)
