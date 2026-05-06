"""sklearn-conformance: clone, get_params, GridSearchCV, cross_val_predict."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_predict

from riesztree import ATE, RieszTreeRegressor


def _est(covariate_keys):
    return RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=covariate_keys),
        max_depth=4,
    )


def test_clone_roundtrip(covariate_keys):
    est = _est(covariate_keys)
    est2 = clone(est)
    assert est.get_params() == est2.get_params()


def test_get_set_params(covariate_keys):
    est = _est(covariate_keys)
    p = est.get_params()
    expected = {
        "estimand", "loss", "max_depth", "min_samples_split", "min_samples_leaf",
        "max_leaves", "growth_policy", "pruning_alpha", "early_stopping_rounds",
        "validation_fraction", "categorical_features", "init", "random_state",
    }
    assert expected.issubset(set(p.keys()))
    est.set_params(max_depth=7, growth_policy="leafwise", max_leaves=20)
    assert est.max_depth == 7 and est.growth_policy == "leafwise" and est.max_leaves == 20


def test_grid_search_cv(linear_gaussian_ate, covariate_keys):
    make, _ = linear_gaussian_ate
    df = make(800, seed=0)
    gs = GridSearchCV(
        _est(covariate_keys),
        {"max_depth": [2, 4, 6]},
        cv=3, n_jobs=1,
    )
    gs.fit(df)
    assert gs.best_params_["max_depth"] in (2, 4, 6)


def test_cross_val_predict(linear_gaussian_ate, covariate_keys):
    make, _ = linear_gaussian_ate
    df = make(600, seed=0)
    a_hat = cross_val_predict(_est(covariate_keys), df, cv=3, n_jobs=1)
    assert a_hat.shape == (600,)
    # Sanity: mean ≈ 0 (ATE Riesz averages to 0 in expectation).
    assert abs(float(np.mean(a_hat))) < 1.0
