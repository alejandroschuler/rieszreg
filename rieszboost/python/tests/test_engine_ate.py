"""End-to-end test: synthetic ATE DGP from Lee-Schuler Section 4.1, fit a Riesz
representer with the fast engine, compare to the closed-form A/pi(x) - (1-A)/(1-pi(x))."""

import numpy as np
import pytest

import rieszboost
from rieszboost.engine import build_augmented, fit
from rieszboost.estimands import ATE


def _logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _simulate(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    pi = _logit(-0.02 * x - x**2 + 4.0 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    return x, a, pi


def _to_rows(x, a):
    return [{"a": int(ai), "x": float(xi)} for ai, xi in zip(a, x)]


def test_augmentation_shape_for_ate():
    x, a, _ = _simulate(50, seed=1)
    rows = _to_rows(x, a)
    aug = build_augmented(rows, ATE(), feature_keys=("a", "x"))
    # ATE: per row contributes z_i with (a=1, b=0), then merges -2*alpha(1,x) and
    # +2*alpha(0,x). When the observed a_i is in {0,1}, one of the counterfactual
    # points coincides with z_i, so we get exactly 2 unique points per row.
    assert aug.features.shape == (2 * len(rows), 2)
    # Sum of b coefficients across both rows for a single i: -2 + 2 = 0.
    for i in range(len(rows)):
        idx = np.where(aug.origin_index == i)[0]
        assert pytest.approx(aug.b[idx].sum()) == 0.0
        # Quadratic coef sum is 1 (only the observed point).
        assert pytest.approx(aug.a[idx].sum()) == 1.0


def test_ate_recovers_inverse_propensity():
    n = 4000
    x, a, pi = _simulate(n, seed=42)
    rows = _to_rows(x, a)

    booster = fit(
        rows,
        ATE(),
        feature_keys=("a", "x"),
        num_boost_round=300,
        learning_rate=0.05,
        max_depth=4,
        seed=0,
    )

    # True Riesz representer for ATE: A/pi(X) - (1-A)/(1-pi(X)).
    alpha_true = a / pi - (1 - a) / (1 - pi)
    alpha_hat = booster.predict(rows)

    rmse = float(np.sqrt(np.mean((alpha_hat - alpha_true) ** 2)))
    # Lee-Schuler Table 1: their RieszBoost gets RMSE ~0.92 on n=500. With n=4000
    # we expect noticeably better; loosely cap at < 1.0 for a smoke test.
    assert rmse < 1.0, f"RMSE {rmse:.3f} too high — RieszBoost not recovering alpha_0"

    # Sanity: predicted alpha at observed treated rows should be positive on
    # average (it's 1/pi > 0), at controls negative (it's -1/(1-pi) < 0).
    assert alpha_hat[a == 1].mean() > 0
    assert alpha_hat[a == 0].mean() < 0


def test_init_m1_for_ate_gives_zero_baseline():
    """ATE: m(z, 1) = 1 - 1 = 0 for every row, so init='m1' should give base_score=0."""
    x, a, _ = _simulate(100, seed=3)
    rows = _to_rows(x, a)
    booster = fit(
        rows,
        ATE(),
        feature_keys=("a", "x"),
        num_boost_round=1,
        init="m1",
    )
    assert booster.base_score == 0.0


def test_public_api_exports():
    assert hasattr(rieszboost, "fit")
    assert hasattr(rieszboost, "ATE")
    assert hasattr(rieszboost, "TSM")
    assert hasattr(rieszboost, "AdditiveShift")


def test_early_stopping_halts_before_max_rounds():
    n = 1000
    x_tr, a_tr, _ = _simulate(n, seed=10)
    x_va, a_va, _ = _simulate(n, seed=11)
    rows_tr = _to_rows(x_tr, a_tr)
    rows_va = _to_rows(x_va, a_va)

    booster = fit(
        rows_tr,
        ATE(),
        feature_keys=("a", "x"),
        valid_rows=rows_va,
        num_boost_round=2000,
        early_stopping_rounds=20,
        learning_rate=0.05,
        max_depth=4,
        seed=0,
    )
    # Should stop well before 2000 rounds on this small problem.
    assert booster.best_iteration is not None
    assert booster.best_iteration < 1500
    assert booster.best_score is not None


def test_riesz_loss_matches_metric_at_best_iteration():
    n = 800
    x_tr, a_tr, _ = _simulate(n, seed=20)
    x_va, a_va, _ = _simulate(n, seed=21)
    rows_tr = _to_rows(x_tr, a_tr)
    rows_va = _to_rows(x_va, a_va)

    booster = fit(
        rows_tr,
        ATE(),
        feature_keys=("a", "x"),
        valid_rows=rows_va,
        num_boost_round=200,
        early_stopping_rounds=20,
        learning_rate=0.05,
        max_depth=4,
        seed=0,
    )
    held_out = booster.riesz_loss(rows_va, ATE())
    assert pytest.approx(held_out, rel=1e-6) == booster.best_score


def test_early_stopping_requires_valid_rows():
    rows = _to_rows(*(_simulate(50, seed=0)[:2]))
    with pytest.raises(ValueError):
        fit(
            rows,
            ATE(),
            feature_keys=("a", "x"),
            num_boost_round=10,
            early_stopping_rounds=5,
        )


def test_gradient_only_recovers_inverse_propensity():
    """Friedman-style gradient boosting (no Hessian) should also work for ATE."""
    n = 4000
    x, a, pi = _simulate(n, seed=42)
    rows = _to_rows(x, a)

    booster = fit(
        rows,
        ATE(),
        feature_keys=("a", "x"),
        gradient_only=True,
        num_boost_round=300,
        learning_rate=0.05,
        max_depth=4,
        seed=0,
    )
    alpha_hat = booster.predict(rows)
    alpha_true = a / pi - (1 - a) / (1 - pi)
    rmse = float(np.sqrt(np.mean((alpha_hat - alpha_true) ** 2)))
    assert rmse < 1.5, f"gradient_only RMSE {rmse:.3f} too high"
