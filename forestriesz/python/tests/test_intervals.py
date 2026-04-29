"""predict_interval requires honest=True + inference=True + locally constant."""

from __future__ import annotations

import numpy as np
import pytest

from forestriesz import ATE, ForestRieszRegressor, TSM, default_riesz_features


def test_predict_interval_returns_lb_le_ub(logistic_tsm_df):
    """TSM uses a single-basis sieve (p=1), which v1 supports for intervals."""
    est = ForestRieszRegressor(
        estimand=TSM(level=1),    # auto sieve = [1{T=1}]
        n_estimators=100,
        min_samples_leaf=15,
        honest=True,
        inference=True,
        random_state=0,
    )
    est.fit(logistic_tsm_df)
    pred = est.predict(logistic_tsm_df)
    lb, ub = est.predict_interval(logistic_tsm_df, alpha=0.1)
    assert lb.shape == pred.shape == ub.shape
    assert np.all(lb <= ub + 1e-9)
    # Point prediction should usually fall between lb and ub.
    inside = ((lb <= pred + 1e-9) & (pred <= ub + 1e-9)).mean()
    assert inside > 0.9


def test_predict_interval_raises_without_honest(logistic_tsm_df):
    est = ForestRieszRegressor(
        estimand=TSM(level=1),
        n_estimators=20,
        honest=False,
        inference=False,
        random_state=0,
    )
    est.fit(logistic_tsm_df)
    with pytest.raises(RuntimeError, match="honest=True"):
        est.predict_interval(logistic_tsm_df)


def test_predict_interval_raises_for_multibasis_sieve(linear_gaussian_ate_df):
    """ATE with [1{T=0}, 1{T=1}] is p=2; v1 raises with delta-method hint."""
    est = ForestRieszRegressor(
        estimand=ATE(),
        riesz_feature_fns=default_riesz_features(ATE()),
        n_estimators=20,        # multiple of subforest_size (4) required when inference=True
        honest=True,
        inference=True,
        random_state=0,
    )
    est.fit(linear_gaussian_ate_df)
    with pytest.raises(NotImplementedError, match="single-basis sieves"):
        est.predict_interval(linear_gaussian_ate_df)
