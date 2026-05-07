"""ForestDiagnostics extends Diagnostics with forest-specific extras."""

from __future__ import annotations

import numpy as np

from forestriesz import (
    ATE,
    ForestDiagnostics,
    ForestRieszRegressor,
    TSM,
    default_riesz_features,
    diagnose_forest,
)


def test_diagnose_forest_returns_extras(logistic_tsm_df):
    est = ForestRieszRegressor(
        estimand=TSM(level=1),
        n_estimators=20,
        random_state=0,
    )
    est.fit(logistic_tsm_df)

    d = diagnose_forest(est, logistic_tsm_df)
    assert isinstance(d, ForestDiagnostics)
    assert d.n == len(logistic_tsm_df)
    assert d.rms >= 0
    assert d.feature_importances.size > 0
    assert np.isfinite(d.mean_leaf_size) or np.isnan(d.mean_leaf_size)
    assert np.isfinite(d.n_leaves_mean)


def test_diagnose_forest_summary_works(linear_gaussian_ate_df):
    est = ForestRieszRegressor(
        estimand=ATE(),
        riesz_feature_fns=default_riesz_features(ATE()),
        n_estimators=20,
        random_state=0,
    )
    est.fit(linear_gaussian_ate_df)
    d = diagnose_forest(est, linear_gaussian_ate_df)
    summary = d.summary()
    assert "RMS magnitude" in summary
