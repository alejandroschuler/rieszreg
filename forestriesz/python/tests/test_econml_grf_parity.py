"""Parity vs vanilla EconML BaseGRF (via _RieszGRF).

forestriesz wraps EconML's `BaseGRF` through the `_RieszGRF` subclass: the
backend computes per-row moments + Jacobians, packs them into the EconML T
slot, and delegates the split search to BaseGRF's MSE criterion. This test
replicates that packing by hand, instantiates `_RieszGRF` directly, and
verifies that `ForestRieszRegressor.predict` agrees on a synthetic ATE
dataset.

The goal is to catch wrapper-level bugs (T-packing, base_score handling,
split-feature resolution, predictor reconstruction) without depending on a
separate EconML public Riesz API — `BaseGRF` is the stable interface
forestriesz commits to. If an EconML release ever changes BaseGRF's
contract, this test fails first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rieszreg import ATE

from forestriesz import ForestRieszRegressor
from forestriesz._grf import _RieszGRF
from forestriesz.feature_fns import default_riesz_features


def _binary_dgp(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    pi = 1.0 / (1.0 + np.exp(-(-0.02 * x - x ** 2 + 4 * np.log(x + 0.3) + 1.5)))
    a = rng.binomial(1, pi).astype(float)
    return pd.DataFrame({"a": a, "x": x})


def _hand_pack_T(features: np.ndarray, A: np.ndarray, phi_fns) -> np.ndarray:
    """Replicate ForestRieszBackend's T = [vec(JJ), A] packing."""
    n, p = len(features), len(phi_fns)
    phi_W = np.column_stack(
        [np.asarray(fn(features), dtype=float) for fn in phi_fns]
    )                                                # (n, p)
    JJ = np.einsum("ij,ik->ijk", phi_W, phi_W).reshape(n, p * p)
    return np.ascontiguousarray(np.column_stack([JJ, A])), phi_W


def test_forest_riesz_matches_handbuilt_grf():
    """ForestRieszRegressor.predict ≈ vanilla _RieszGRF call with same packing.

    Uses the auto-resolved [1{T=0}, 1{T=1}] sieve for ATE and matched forest
    hyperparams. Because both paths go through the identical RNG seed and
    the same packed (X, T, y), predictions should be bit-identical.
    """
    df = _binary_dgp(n=400, seed=0)
    estimand = ATE(treatment="a", covariates=("x",))

    common = dict(
        n_estimators=20,
        max_depth=4,
        min_samples_leaf=20,
        max_samples=0.45,
        min_balancedness_tol=0.45,
        random_state=0,
        honest=False,
        inference=False,
    )

    # --- Wrapper path ---
    wrap = ForestRieszRegressor(
        estimand=estimand,
        riesz_feature_fns="auto",
        **common,
    ).fit(df)
    alpha_wrap = wrap.predict(df)

    # --- Hand-built path: same sieve, same packing, direct _RieszGRF call ---
    phi_fns = default_riesz_features(estimand)
    p = len(phi_fns)
    feature_keys = estimand.feature_keys                     # ("a", "x")
    features = df[list(feature_keys)].to_numpy(dtype=float)  # (n, 2)

    # Per-row A[i, j] = m(W_i; φ_j) for ATE = φ_j(1, x_i) - φ_j(0, x_i).
    A_mat = np.zeros((len(df), p))
    for i, x_i in enumerate(features[:, 1]):
        point_one  = np.array([[1.0, x_i]])
        point_zero = np.array([[0.0, x_i]])
        for j, fn in enumerate(phi_fns):
            A_mat[i, j] = float(fn(point_one)[0]) - float(fn(point_zero)[0])

    T, _ = _hand_pack_T(features, A_mat, phi_fns)
    y_dummy = np.zeros((len(df), 1), dtype=float)

    # Split features: ATE auto-resolves to covariate-only (drop the treatment
    # col since the sieve already encodes it).
    from forestriesz.feature_fns import default_split_feature_indices
    split_idx = list(default_split_feature_indices(estimand, "auto"))
    X_split = features[:, split_idx]

    direct = _RieszGRF(n_outputs_riesz=p, criterion="mse", **common, fit_intercept=True,
                       subforest_size=4, n_jobs=-1, verbose=0, warm_start=False,
                       min_samples_split=10, min_weight_fraction_leaf=0.0,
                       min_var_fraction_leaf=None, max_features="auto",
                       min_impurity_decrease=0.0)
    direct.fit(X_split, T, y_dummy)

    # Reconstruct α̂(z) = θ(z_split) · φ(z) following ForestPredictor.predict_eta.
    theta_hand = np.asarray(direct.predict(X_split))         # (n, p)
    if theta_hand.ndim == 1:
        theta_hand = theta_hand.reshape(-1, 1)
    phi_eval = np.column_stack([fn(features) for fn in phi_fns])
    alpha_direct = (theta_hand * phi_eval).sum(axis=1)

    # Bit-identity (same RNG seed, same packed inputs).
    np.testing.assert_allclose(
        alpha_wrap, alpha_direct, rtol=0, atol=1e-12,
        err_msg=("ForestRieszRegressor predictions diverged from a hand-built "
                 "_RieszGRF call with identical packing. The wrapper has "
                 "drifted from the EconML BaseGRF contract."),
    )
