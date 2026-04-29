"""sklearn-conformance: clone, GridSearchCV, cross_val_predict."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_predict, KFold

from forestriesz import ATE, ForestRieszRegressor, default_riesz_features, TSM


def test_clone_preserves_constructor_args():
    fns = default_riesz_features(ATE())
    est = ForestRieszRegressor(
        estimand=ATE(),
        riesz_feature_fns=fns,
        n_estimators=42,
        max_depth=4,
        min_samples_leaf=7,
        honest=True,
        random_state=123,
    )
    clone_est = clone(est)
    # sklearn clone copies constructor params; estimator must be re-instantiable.
    assert clone_est.n_estimators == est.n_estimators
    assert clone_est.max_depth == est.max_depth
    assert clone_est.min_samples_leaf == est.min_samples_leaf
    assert clone_est.honest == est.honest
    assert clone_est.random_state == est.random_state
    # sklearn deep-copies non-estimator params; lists become new list objects
    # carrying the same callable identities.
    assert clone_est.riesz_feature_fns == est.riesz_feature_fns


def test_grid_search_runs(logistic_tsm_df):
    est = ForestRieszRegressor(
        estimand=TSM(level=1),
        n_estimators=20,
        random_state=0,
    )
    grid = {
        "max_depth": [3, 5],
        "min_samples_leaf": [5, 15],
    }
    gs = GridSearchCV(
        est,
        grid,
        cv=2,
        n_jobs=1,
        refit=True,
    )
    gs.fit(logistic_tsm_df)
    assert hasattr(gs, "best_params_")
    pred = gs.predict(logistic_tsm_df)
    assert pred.shape == (len(logistic_tsm_df),)


def test_cross_val_predict_runs(logistic_tsm_df):
    est = ForestRieszRegressor(
        estimand=TSM(level=1),
        n_estimators=30,
        min_samples_leaf=10,
        random_state=0,
    )
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    pred = cross_val_predict(est, logistic_tsm_df, cv=cv, n_jobs=1)
    assert pred.shape == (len(logistic_tsm_df),)
    assert np.all(np.isfinite(pred))
