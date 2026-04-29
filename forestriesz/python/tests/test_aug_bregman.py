"""AugForestRieszRegressor with non-quadratic Bregman losses.

The augmentation-style backend handles all four built-in losses by post-hoc
replacing each leaf's stored θ with the per-leaf Newton optimum. Tree
structure is still chosen by the squared-loss MSE criterion — splits that
maximize variance reduction in -B/(2A) also separate the monotonically-
related Bregman optima well in practice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rieszreg import (
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
    SquaredLoss,
)
from rieszreg.testing import dgps

from forestriesz import (
    AugForestRieszRegressor,
    ATE,
    TSM,
)


# ---- closed-form / Newton solver unit tests -------------------------------


def test_solver_closed_form_squared_matches_minus_b_over_2a():
    """For squared loss the per-leaf optimum is exactly -B/(2A); one Newton step."""
    from forestriesz._leaf_solver import solve_leaf_bregman

    a = np.array([1.0, 1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, -2.0, +2.0])
    phi = np.ones((4, 1))
    theta = solve_leaf_bregman(SquaredLoss(), a, b, phi)
    expected = -b.sum() / (2.0 * a.sum())
    np.testing.assert_allclose(theta[0], expected, atol=1e-8)


def test_solver_kl_recovers_log_minus_b_over_2a():
    """For KLLoss the per-leaf optimum satisfies α = -B/(2A); η = log(α)."""
    from forestriesz._leaf_solver import solve_leaf_bregman

    a = np.array([2.0, 3.0, 0.0])
    b = np.array([0.0, 0.0, -8.0])     # b ≤ 0 (KLLoss requirement)
    phi = np.ones((3, 1))
    theta = solve_leaf_bregman(KLLoss(), a, b, phi)
    expected_alpha = -b.sum() / (2.0 * a.sum())     # 8 / 10 = 0.8
    expected_eta = np.log(expected_alpha)
    np.testing.assert_allclose(theta[0], expected_eta, atol=1e-6)


def test_solver_bernoulli_recovers_logit_minus_b_over_2a():
    """For BernoulliLoss the per-leaf optimum satisfies α = -B/(2A); η = logit(α)."""
    from forestriesz._leaf_solver import solve_leaf_bregman

    a = np.array([5.0, 5.0, 0.0])
    b = np.array([0.0, 0.0, -4.0])     # α* = 4/10 = 0.4 ∈ (0, 1)
    phi = np.ones((3, 1))
    theta = solve_leaf_bregman(BernoulliLoss(), a, b, phi)
    p = -b.sum() / (2.0 * a.sum())
    expected_eta = np.log(p / (1.0 - p))
    np.testing.assert_allclose(theta[0], expected_eta, atol=1e-6)


def test_solver_handles_empty_leaf():
    """Empty leaf returns the init θ (= 0 by default)."""
    from forestriesz._leaf_solver import solve_leaf_bregman

    a = np.array([])
    b = np.array([])
    phi = np.zeros((0, 1))
    theta = solve_leaf_bregman(SquaredLoss(), a, b, phi)
    assert theta.shape == (1,)
    assert theta[0] == 0.0


# ---- end-to-end: predictions stay in the loss's natural domain ------------


@pytest.fixture
def df_binary():
    rng = np.random.default_rng(0)
    n = 800
    x = rng.normal(size=n)
    pi = 1.0 / (1.0 + np.exp(-0.5 * x))
    a = (rng.uniform(size=n) < pi).astype(float)
    return pd.DataFrame({"a": a, "x": x}), x, pi, a


def test_kl_loss_predictions_strictly_positive(df_binary):
    df, _, _, _ = df_binary
    est = AugForestRieszRegressor(
        estimand=TSM(level=1),
        loss=KLLoss(),
        n_estimators=50,
        min_samples_leaf=10,
        random_state=0,
    )
    est.fit(df)
    pred = est.predict(df)
    assert pred.shape == (len(df),)
    assert np.all(pred > 0), "KLLoss must produce strictly positive α̂"
    assert np.all(np.isfinite(pred))


def test_bernoulli_loss_predictions_in_unit_interval(df_binary):
    df, _, _, _ = df_binary
    est = AugForestRieszRegressor(
        estimand=TSM(level=1),
        loss=BernoulliLoss(),
        n_estimators=50,
        min_samples_leaf=10,
        random_state=0,
    )
    est.fit(df)
    pred = est.predict(df)
    assert np.all((pred > 0) & (pred < 1)), "BernoulliLoss must produce α̂ ∈ (0, 1)"


def test_bounded_squared_predictions_in_bounds(df_binary):
    df, _, _, _ = df_binary
    est = AugForestRieszRegressor(
        estimand=ATE(),
        loss=BoundedSquaredLoss(lo=-15.0, hi=15.0),
        n_estimators=50,
        min_samples_leaf=10,
        random_state=0,
    )
    est.fit(df)
    pred = est.predict(df)
    assert np.all((pred > -15.0) & (pred < 15.0)), "BoundedSquaredLoss must clip to its bounds"


# ---- consistency: KL converges to the true IPW representer ----------------


def test_kl_converges_to_truth_on_tsm():
    """KLLoss with TSM should recover α₀ = 1{T=1}/π(X) on the logistic_tsm DGP."""
    def fit_predict(train, test):
        est = AugForestRieszRegressor(
            estimand=TSM(level=1.0),
            loss=KLLoss(),
            n_estimators=200,
            min_samples_leaf=10,
            random_state=0,
        )
        est.fit(train)
        return est.predict(test)

    rmses = dgps.assert_consistency(
        fit_predict,
        dgp=dgps.logistic_tsm(level=1.0),
        n_grid=(500, 2000),
        rng_seed=0,
        tol_at_max_n=1.0,
        monotonicity_slack=0.5,
    )
    assert rmses[-1] < rmses[0] * 1.5


# ---- save/load round-trips the leaf_eta_table ----------------------------


def test_save_load_round_trip_preserves_kl_predictions(tmp_path, df_binary):
    df, _, _, _ = df_binary
    est = AugForestRieszRegressor(
        estimand=TSM(level=1),
        loss=KLLoss(),
        n_estimators=30,
        min_samples_leaf=15,
        random_state=0,
    )
    est.fit(df)
    pred_before = est.predict(df)

    save_dir = tmp_path / "kl_fit"
    est.save(save_dir)

    loaded = AugForestRieszRegressor.load(save_dir)
    pred_after = loaded.predict(df)
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-12)


# ---- coefficient validation: KL/Bernoulli reject ATE-style data ----------


def test_kl_rejects_signed_b_coefficients(df_binary):
    """ATE produces both +b and -b (additive in the trace), so KLLoss must raise."""
    df, _, _, _ = df_binary
    est = AugForestRieszRegressor(
        estimand=ATE(),
        loss=KLLoss(),
        n_estimators=10,
        random_state=0,
    )
    with pytest.raises(ValueError, match="non-negative"):
        est.fit(df)


def test_bernoulli_rejects_signed_b_coefficients(df_binary):
    df, _, _, _ = df_binary
    est = AugForestRieszRegressor(
        estimand=ATE(),
        loss=BernoulliLoss(),
        n_estimators=10,
        random_state=0,
    )
    with pytest.raises(ValueError, match="non-negative"):
        est.fit(df)
