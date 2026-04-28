"""Bregman-Riesz loss: SquaredLoss equivalence + KLLoss density-ratio recovery."""

import numpy as np
import pytest

from rieszboost.engine import build_augmented, fit
from rieszboost.estimands import ATE, TSM
from rieszboost.losses import KLLoss, SquaredLoss


def _logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _simulate(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    pi = _logit(-0.02 * x - x**2 + 4.0 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    return x, a, pi


def test_squared_loss_explicit_matches_default():
    """Passing loss_spec=SquaredLoss() must give bitwise-identical results to
    the default behavior (which is also SquaredLoss())."""
    x, a, _ = _simulate(500, seed=0)
    rows = [{"a": int(ai), "x": float(xi)} for ai, xi in zip(a, x)]
    b1 = fit(rows, ATE(), feature_keys=("a", "x"),
             num_boost_round=20, learning_rate=0.1, max_depth=3, seed=0)
    b2 = fit(rows, ATE(), feature_keys=("a", "x"),
             loss_spec=SquaredLoss(),
             num_boost_round=20, learning_rate=0.1, max_depth=3, seed=0)
    np.testing.assert_array_equal(b1.predict(rows), b2.predict(rows))


def test_kl_loss_rejects_signed_coefficients():
    """ATE has m-coefficients (+1, -1) — KLLoss must refuse it."""
    x, a, _ = _simulate(200, seed=1)
    rows = [{"a": int(ai), "x": float(xi)} for ai, xi in zip(a, x)]
    with pytest.raises(ValueError, match="non-negative"):
        fit(rows, ATE(), feature_keys=("a", "x"),
            loss_spec=KLLoss(),
            num_boost_round=5, learning_rate=0.1, max_depth=3)


def test_kl_loss_predicts_positive_alpha():
    """Exp link → all predictions strictly positive by construction.
    Sanity-only test; KL's accuracy on treatment-style augmented data is more
    delicate than squared loss (see test_kl_correlates_with_truth_on_treated)."""
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.uniform(0, 1, n)
    pi = _logit(-0.02 * x - x**2 + 4 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    rows = [{"a": int(ai), "x": float(xi)} for ai, xi in zip(a, x)]

    booster = fit(
        rows,
        TSM(level=1, treatment="a", covariates=("x",)),
        feature_keys=("a", "x"),
        loss_spec=KLLoss(),
        num_boost_round=50,
        learning_rate=0.05,
        max_depth=3,
    )
    alpha_hat = booster.predict(rows)
    assert alpha_hat.min() > 0
    # Init was alpha=1.0 (η=0), so reasonable trees should not collapse to zero.
    assert alpha_hat.max() > 0.5


def test_kl_correlates_with_truth_on_treated():
    """KL fit on treated rows should track 1/pi(X) at least weakly.

    KL's stability is limited by the augmented-data formulation: pure-
    counterfactual rows (a=0, b=-2 in our coefficients) push α upward
    unboundedly with no offsetting quadratic. Standard squared loss is more
    robust here. We test only a low correlation bar (≥ 0.3); this is a sanity
    check that KL's optimization is moving in the right direction, not a
    claim that KL is a recommended fitter for this problem.
    """
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(0, 1, n)
    pi = _logit(-0.02 * x - x**2 + 4 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    rows = [{"a": int(ai), "x": float(xi)} for ai, xi in zip(a, x)]
    n_tr = int(0.8 * n)

    booster_kl = fit(
        rows[:n_tr],
        TSM(level=1, treatment="a", covariates=("x",)),
        feature_keys=("a", "x"),
        loss_spec=KLLoss(),
        valid_rows=rows[n_tr:],
        num_boost_round=2000,
        early_stopping_rounds=20,
        learning_rate=0.05,
        max_depth=3,
        reg_lambda=10.0,
        seed=0,
    )
    alpha_hat = booster_kl.predict(rows)
    treated = a == 1
    corr = float(np.corrcoef(alpha_hat[treated], 1.0 / pi[treated])[0, 1])
    assert corr > 0.3, f"KL fit not correlated with truth: corr={corr:.3f}"


def test_kl_riesz_loss_attribute_uses_alpha_space():
    """booster.riesz_loss(...) should compute loss in α space (post-link)."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0, 1, n)
    pi = _logit(-0.02 * x - x**2 + 4 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    rows = [{"a": int(ai), "x": float(xi)} for ai, xi in zip(a, x)]

    booster = fit(
        rows,
        TSM(level=1, treatment="a", covariates=("x",)),
        feature_keys=("a", "x"),
        loss_spec=KLLoss(),
        num_boost_round=20,
        learning_rate=0.1,
        max_depth=3,
    )
    held = booster.riesz_loss(rows, TSM(level=1, treatment="a", covariates=("x",)))
    # KL loss values can be any real number (not necessarily nonneg); just
    # check it's finite.
    assert np.isfinite(held)
