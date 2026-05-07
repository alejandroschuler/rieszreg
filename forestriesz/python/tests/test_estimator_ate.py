"""End-to-end ATE recovery on the linear-Gaussian DGP.

ATE requires the locally linear sieve (the locally constant moment is
identically zero — that case has its own test in test_loss_unsupported.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from rieszreg.testing import dgps

from forestriesz import ATE, ForestRieszRegressor, default_riesz_features


def _fit_predict(train, test):
    est = ForestRieszRegressor(
        estimand=ATE(),
        riesz_feature_fns=default_riesz_features(ATE()),
        n_estimators=200,
        min_samples_leaf=10,
        random_state=0,
    )
    est.fit(train)
    return est.predict(test)


def test_ate_consistency_grid():
    rmses = dgps.assert_consistency(
        _fit_predict,
        dgp=dgps.linear_gaussian_ate(),
        n_grid=(500, 2000),
        rng_seed=0,
        tol_at_max_n=1.0,
        monotonicity_slack=0.5,
    )
    # RMSE should drop with sample size (lax check; single-seed noise is real).
    assert rmses[-1] < rmses[0] * 1.5


def test_ate_predict_shape_and_finite(linear_gaussian_ate_df):
    est = ForestRieszRegressor(
        estimand=ATE(),
        riesz_feature_fns=default_riesz_features(ATE()),
        n_estimators=50,
        random_state=0,
    )
    est.fit(linear_gaussian_ate_df)
    pred = est.predict(linear_gaussian_ate_df)
    assert pred.shape == (len(linear_gaussian_ate_df),)
    assert np.all(np.isfinite(pred))


def test_ate_score_is_negative_riesz_loss(linear_gaussian_ate_df):
    est = ForestRieszRegressor(
        estimand=ATE(),
        riesz_feature_fns=default_riesz_features(ATE()),
        n_estimators=30,
        random_state=0,
    )
    est.fit(linear_gaussian_ate_df)
    score = est.score(linear_gaussian_ate_df)
    loss = est.riesz_loss(linear_gaussian_ate_df)
    assert score == pytest.approx(-loss)
