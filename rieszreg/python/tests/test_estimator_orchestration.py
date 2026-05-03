"""Smoke tests for the RieszEstimator orchestrator with a stub backend.

This exercises the orchestration path (row conversion, augmentation, fit/predict
plumbing, score) without depending on a heavyweight backend like xgboost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rieszreg import (
    ATE,
    AugmentedDataset,
    FitResult,
    KLLoss,
    RieszEstimator,
    SquaredLoss,
    TSM,
    aug_loss_alpha,
    build_augmented,
)


class _StubPredictor:
    """Tiny predictor that returns η = 0 + base_score for every row."""

    kind = "stub"

    def __init__(self, base_score: float, loss):
        self.base_score = base_score
        self.loss = loss

    def predict_eta(self, features):
        return np.full(np.asarray(features).shape[0], self.base_score)

    def predict_alpha(self, features):
        return self.loss.link_to_alpha(self.predict_eta(features))


class _StubBackend:
    """Backend that returns a stub predictor; ignores augmented data entirely."""

    def fit_augmented(self, aug_train, aug_valid, loss, **kw):
        del aug_valid, kw
        assert isinstance(aug_train, AugmentedDataset)
        return FitResult(
            predictor=_StubPredictor(base_score=kw.get("base_score", 0.0)
                                     if False else 0.0, loss=loss),
            best_iteration=None,
            best_score=None,
        )


def test_fit_predict_with_dataframe():
    df = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0], "x": [0.1, 0.2, 0.3, 0.4]})
    est = RieszEstimator(
        estimand=ATE(), backend=_StubBackend(), loss=SquaredLoss(),
    ).fit(df)
    pred = est.predict(df)
    assert pred.shape == (4,)
    # Stub predictor returns 0 + base_score; with default init, base_score == 0.
    assert np.allclose(pred, 0.0)


def test_fit_predict_with_ndarray():
    X = np.array([[0.0, 0.1], [1.0, 0.2], [0.0, 0.3]])
    est = RieszEstimator(
        estimand=ATE(), backend=_StubBackend(), loss=SquaredLoss(),
    ).fit(X)
    pred = est.predict(X)
    assert pred.shape == (3,)


def test_score_uses_squared_yardstick():
    """`score()` evaluates the canonical squared Riesz loss regardless of the
    training loss (sklearn convention: scoring is detached from training)."""
    df = pd.DataFrame({"a": [0.0, 1.0], "x": [0.1, 0.5]})

    est_sq = RieszEstimator(
        estimand=ATE(), backend=_StubBackend(), loss=SquaredLoss(),
    ).fit(df)
    est_kl = RieszEstimator(
        estimand=TSM(level=1), backend=_StubBackend(), loss=KLLoss(),
    ).fit(df)

    def _expected_neg_squared(est):
        rows = [
            {k: df[k].iloc[i] for k in est.estimand.feature_keys}
            for i in range(len(df))
        ]
        aug = build_augmented(rows, est.estimand)
        eta = est.predictor_.predict_eta(aug.features)
        alpha = est.loss_.link_to_alpha(eta)
        sq = SquaredLoss()
        return -float(
            np.sum(aug_loss_alpha(sq, aug.is_original, aug.potential_deriv_coef, alpha))
            / aug.n_rows
        )

    assert est_sq.score(df) == pytest.approx(_expected_neg_squared(est_sq))
    assert est_kl.score(df) == pytest.approx(_expected_neg_squared(est_kl))
    # Squared training: score == -riesz_loss (yardstick coincides with training loss).
    assert est_sq.score(df) == pytest.approx(-est_sq.riesz_loss(df))


def test_predict_unfitted_raises():
    est = RieszEstimator(estimand=ATE(), backend=_StubBackend())
    with pytest.raises(RuntimeError, match="not fitted"):
        est.predict(np.zeros((1, 2)))


def test_no_backend_raises_at_fit():
    est = RieszEstimator(estimand=ATE())
    with pytest.raises(ValueError, match="requires a `backend"):
        est.fit(np.zeros((1, 2)))


def test_dataframe_missing_columns_raises():
    df = pd.DataFrame({"a": [0.0, 1.0]})
    est = RieszEstimator(estimand=ATE(), backend=_StubBackend())
    with pytest.raises(ValueError, match="missing columns"):
        est.fit(df)


def test_ndarray_with_extra_keys_raises():
    from rieszreg import StochasticIntervention
    est = RieszEstimator(
        estimand=StochasticIntervention(samples_key="shift_samples"),
        backend=_StubBackend(),
    )
    with pytest.raises(ValueError, match="per-row payload"):
        est.fit(np.zeros((2, 2)))


def test_sklearn_clone_round_trip():
    from sklearn.base import clone
    est = RieszEstimator(estimand=ATE(), backend=_StubBackend(), random_state=42)
    cloned = clone(est)
    assert cloned.random_state == 42
    # Cloned estimator is unfit.
    assert not hasattr(cloned, "predictor_")
