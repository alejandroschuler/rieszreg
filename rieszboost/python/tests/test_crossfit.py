"""K-fold cross-fitting smoke test on the Lee-Schuler binary-treatment DGP."""

import numpy as np

from rieszboost.crossfit import crossfit
from rieszboost.estimands import ATE


def _logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _simulate(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    pi = _logit(-0.02 * x - x**2 + 4.0 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    return x, a, pi


def test_crossfit_covers_every_row():
    x, a, _ = _simulate(500, seed=0)
    rows = [{"a": int(ai), "x": float(xi)} for ai, xi in zip(a, x)]
    result = crossfit(
        rows,
        ATE(),
        feature_keys=("a", "x"),
        n_folds=5,
        seed=1,
        num_boost_round=50,
        learning_rate=0.1,
        max_depth=3,
    )
    assert result.alpha_hat.shape == (500,)
    assert set(np.unique(result.fold_assignment)).issubset(set(range(5)))
    assert len(result.boosters) == 5
    assert np.all(np.isfinite(result.alpha_hat))


def test_crossfit_oof_predictions_track_truth():
    """OOF predictions should correlate strongly with the true Riesz representer.
    Use shallow trees + heavier ridge to keep extrapolation under control."""
    n = 2000
    x, a, pi = _simulate(n, seed=7)
    rows = [{"a": int(ai), "x": float(xi)} for ai, xi in zip(a, x)]
    result = crossfit(
        rows,
        ATE(),
        feature_keys=("a", "x"),
        n_folds=5,
        seed=2,
        num_boost_round=500,
        early_stopping_rounds=10,
        early_stopping_inner_split=0.2,
        learning_rate=0.05,
        max_depth=3,
        reg_lambda=10.0,
    )
    alpha_true = a / pi - (1 - a) / (1 - pi)
    corr = float(np.corrcoef(result.alpha_hat, alpha_true)[0, 1])
    assert corr > 0.9, f"OOF predictions weakly correlated with truth (corr={corr:.3f})"


def test_crossfit_with_inner_early_stopping():
    n = 600
    x, a, _ = _simulate(n, seed=3)
    rows = [{"a": int(ai), "x": float(xi)} for ai, xi in zip(a, x)]
    result = crossfit(
        rows,
        ATE(),
        feature_keys=("a", "x"),
        n_folds=4,
        seed=0,
        num_boost_round=500,
        early_stopping_rounds=20,
        early_stopping_inner_split=0.25,
        learning_rate=0.05,
        max_depth=3,
    )
    # All folds should have triggered early stopping (well below 500 rounds).
    for b in result.boosters:
        assert b.best_iteration is not None
        assert b.best_iteration < 500
