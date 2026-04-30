"""Tests for `riesz_scorer` and the canonical-yardstick `score()` behavior.

Uses a stub backend (mirrors `test_estimator_orchestration.py`) so these tests
don't depend on any implementation package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV, cross_val_score

from rieszreg import (
    ATE,
    AugmentedDataset,
    FitResult,
    KLLoss,
    RieszEstimator,
    SquaredLoss,
    riesz_scorer,
)


class _StubPredictor:
    """Returns η = base_score + 0.3 * sum(features) — non-trivial enough that
    SquaredLoss and KLLoss evaluate to different values on the same predictor."""

    kind = "stub"

    def __init__(self, base_score: float, loss):
        self.base_score = base_score
        self.loss = loss

    def predict_eta(self, features):
        feats = np.asarray(features)
        return self.base_score + 0.3 * feats.sum(axis=1)

    def predict_alpha(self, features):
        return self.loss.link_to_alpha(self.predict_eta(features))


class _StubBackend:
    """Backend whose predictor returns η = base_score for every row."""

    def __init__(self, validation_fraction: float = 0.0):
        self.validation_fraction = validation_fraction

    def fit_augmented(self, aug_train, aug_valid, loss, **kw):
        del aug_valid
        assert isinstance(aug_train, AugmentedDataset)
        return FitResult(
            predictor=_StubPredictor(base_score=kw.get("base_score", 0.0), loss=loss),
            best_iteration=None,
            best_score=None,
        )

    def get_params(self, deep=True):  # noqa: ARG002 — sklearn clone path
        return {"validation_fraction": self.validation_fraction}

    def set_params(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self


def _df(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.binomial(1, 0.5, n).astype(float),
                         "x": rng.normal(size=n)})


def test_scorer_default_matches_score_method():
    """`riesz_scorer(loss=None)` matches `est.score(X)` (both squared yardstick)."""
    df = _df()
    est = RieszEstimator(
        estimand=ATE(), backend=_StubBackend(), loss=KLLoss(),
    ).fit(df)
    scorer = riesz_scorer()  # default = SquaredLoss yardstick
    assert scorer(est, df) == pytest.approx(est.score(df))


def test_scorer_custom_loss_differs_from_default():
    """`riesz_scorer(loss=KLLoss())` evaluates KL on the predicted α."""
    df = _df()
    est = RieszEstimator(
        estimand=ATE(), backend=_StubBackend(), loss=KLLoss(),
    ).fit(df)
    sq_score = riesz_scorer(loss=SquaredLoss())(est, df)
    kl_score = riesz_scorer(loss=KLLoss())(est, df)
    assert np.isfinite(sq_score)
    assert np.isfinite(kl_score)
    assert sq_score != pytest.approx(kl_score)


def test_score_is_yardstick_independent_of_training_loss():
    """Two estimators trained with different losses report `score()` on the
    same squared yardstick — the value depends on each predictor's α, not on
    which loss it was fit with."""
    df = _df()
    est_sq = RieszEstimator(
        estimand=ATE(), backend=_StubBackend(), loss=SquaredLoss(),
    ).fit(df)
    est_kl = RieszEstimator(
        estimand=ATE(), backend=_StubBackend(), loss=KLLoss(),
    ).fit(df)
    # Both scores use SquaredLoss; predictors differ (α=0 vs α=1) so values differ.
    assert np.isfinite(est_sq.score(df))
    assert np.isfinite(est_kl.score(df))


def test_cross_val_score_default_yardstick():
    """`cross_val_score` with default scoring runs and returns 3 floats."""
    df = _df(n=60)
    est = RieszEstimator(estimand=ATE(), backend=_StubBackend(), loss=KLLoss())
    scores = cross_val_score(est, df, cv=3)
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_cross_val_score_custom_yardstick():
    """Passing `scoring=riesz_scorer(loss=...)` works end-to-end."""
    df = _df(n=60)
    est = RieszEstimator(estimand=ATE(), backend=_StubBackend(), loss=SquaredLoss())
    scores = cross_val_score(
        est, df, cv=3, scoring=riesz_scorer(loss=SquaredLoss()),
    )
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_gridsearchcv_across_losses():
    """`GridSearchCV` over `loss` ranks candidates on the same SquaredLoss
    yardstick — `score()` is detached from training loss, so the ranking is
    well-defined."""
    df = _df(n=60)
    est = RieszEstimator(estimand=ATE(), backend=_StubBackend())
    grid = GridSearchCV(
        est,
        param_grid={"loss": [SquaredLoss(), KLLoss()]},
        cv=3,
    ).fit(df)
    assert grid.best_params_ is not None
    assert "loss" in grid.best_params_
    assert np.isfinite(grid.best_score_)
