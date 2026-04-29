"""ForestRieszBackend — implements `rieszreg.MomentBackend`.

Consumes raw rows + the estimand directly (the moment-style entry point),
computes per-row moments via `rieszreg.trace`, packs them as a linear-moment
problem for EconML's `BaseGRF`, and returns a `FitResult` whose predictor is a
`ForestPredictor`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from rieszreg import (
    AugmentedDataset,
    Estimand,
    FitResult,
    LossSpec,
    SquaredLoss,
    build_augmented,
    trace,
)

from ._grf import _RieszGRF
from .feature_fns import default_split_feature_indices
from .predictor import ForestPredictor


def _materialize_features(
    rows: list[dict[str, Any]], feature_keys: Sequence[str]
) -> np.ndarray:
    return np.array([[r[k] for k in feature_keys] for r in rows], dtype=float)


def _eval_phi(
    features: np.ndarray, phi_fns: Sequence[Callable]
) -> np.ndarray:
    """Stack vectorized basis evaluations into an (n, p) matrix."""
    return np.column_stack([np.asarray(fn(features), dtype=float) for fn in phi_fns])


def _compute_per_row_moments(
    rows: list[dict[str, Any]],
    estimand: Estimand,
    phi_fns: Sequence[Callable],
    feature_keys: Sequence[str],
) -> np.ndarray:
    """Compute A[i, j] = m(W_i; phi_j) = sum over (coef, point) in trace(W_i)
    of coef * phi_j(point), for each original row i and basis j."""
    n = len(rows)
    p = len(phi_fns)
    if n == 0:
        return np.zeros((0, p))
    A = np.zeros((n, p))
    for i, row in enumerate(rows):
        for coef, point in trace(estimand, row):
            point_arr = np.array([[point[k] for k in feature_keys]], dtype=float)
            phi_at_point = np.array([float(fn(point_arr)[0]) for fn in phi_fns])
            A[i] += coef * phi_at_point
    return A


def _holdout_riesz_loss(
    rows_valid: list[dict[str, Any]],
    estimand: Estimand,
    predictor: ForestPredictor,
    loss: LossSpec,
) -> float:
    """Mean per-original-row Riesz loss on the validation rows.

    Uses build_augmented + loss_row to share the formula with the rest of the
    framework, so val scores are comparable across backends.
    """
    if not rows_valid:
        return float("nan")
    aug = build_augmented(rows_valid, estimand)
    eta = predictor.predict_eta(aug.features)
    alpha = loss.link_to_alpha(eta)
    return float(np.sum(loss.loss_row(aug.a, aug.b, alpha)) / aug.n_rows)


@dataclass
class ForestRieszBackend:
    """Random-forest Riesz regression backend.

    Wraps EconML's ``BaseGRF`` with the linear-moment criterion. Implements
    ``MomentBackend.fit_rows`` so it consumes raw rows and uses
    ``rieszreg.trace`` to evaluate per-row moments directly — no augmented
    dataset blow-up.

    Parameters
    ----------
    riesz_feature_fns
        Basis ``φ_1, …, φ_p`` for the locally linear sieve (each callable
        takes a feature matrix ``(n, n_features)`` and returns ``(n,)``). When
        ``None``, a constant basis is used (locally constant α per leaf).
    split_feature_indices
        Which feature columns the forest splits on. When ``None``, a default
        is chosen from the estimand and sieve (covariates only when a
        treatment-indexed sieve is supplied; otherwise all features).
    n_estimators, max_depth, min_samples_split, min_samples_leaf,
    min_weight_fraction_leaf, min_var_fraction_leaf, max_features,
    min_impurity_decrease, max_samples, min_balancedness_tol, honest,
    inference, fit_intercept, subforest_size, n_jobs, random_state, verbose
        Forwarded to ``econml.grf._base_grf.BaseGRF``. ``honest`` defaults to
        False because cross-fitting (``cross_val_predict``) does not require
        honesty; flip to True when you want ``predict_interval``.
    l2
        Ridge added to the per-leaf Jacobian for numerical stability.
    """

    riesz_feature_fns: list[Callable] | None = None
    split_feature_indices: Sequence[int] | None = None
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    min_weight_fraction_leaf: float = 0.0
    min_var_fraction_leaf: float | None = None
    max_features: object = "auto"
    min_impurity_decrease: float = 0.0
    max_samples: float = 0.45
    min_balancedness_tol: float = 0.45
    honest: bool = False
    inference: bool = False
    fit_intercept: bool = True
    subforest_size: int = 4
    l2: float = 0.01
    n_jobs: int = -1
    random_state: int = 0
    verbose: int = 0

    def fit_rows(
        self,
        rows_train: list[dict[str, Any]],
        rows_valid: list[dict[str, Any]] | None,
        estimand: Estimand,
        loss: LossSpec,
        *,
        n_estimators: int,
        learning_rate: float,
        base_score: float,
        early_stopping_rounds: int | None,
        random_state: int,
        hyperparams: dict[str, Any],
    ) -> FitResult:
        if not isinstance(loss, SquaredLoss):
            raise NotImplementedError(
                f"ForestRieszBackend currently supports SquaredLoss only "
                f"(got {type(loss).__name__}). Other Bregman losses require "
                "a per-leaf Newton iteration on the augmented loss; planned "
                "for a future release."
            )
        # Forests are non-iterative; orchestrator-level boosting knobs are ignored.
        del n_estimators, learning_rate, early_stopping_rounds, hyperparams

        seed = random_state if random_state is not None else self.random_state
        feature_keys = estimand.feature_keys

        # 1. Materialize feature matrix.
        features = _materialize_features(rows_train, feature_keys)

        # 2. Resolve sieve. None => locally constant (single basis function = 1).
        phi_fns = self.riesz_feature_fns or [lambda f: np.ones(len(f))]
        p = len(phi_fns)

        # 3. Per-row basis values φ(W_i).
        phi_W = _eval_phi(features, phi_fns)             # (n, p)

        # 4. Per-row moment A[i, j] = m(W_i; φ_j).
        A = _compute_per_row_moments(rows_train, estimand, phi_fns, feature_keys)

        # 5. Fold base_score into A so the predictor returns base_score + leaf θ·φ.
        if base_score != 0.0:
            A = A - base_score * phi_W

        # 6. Pack T = [vec(φφ') | φ] per row, y = A. The Jacobian J = φφ' is
        # symmetric so flattening order is immaterial; we use row-major.
        JJ = np.einsum("ij,ik->ijk", phi_W, phi_W).reshape(len(rows_train), p * p)
        T_pack = np.column_stack([JJ, phi_W])
        y_pack = A

        # 7. Choose split features.
        split_idx = self.split_feature_indices
        if split_idx is None:
            split_idx = default_split_feature_indices(estimand, self.riesz_feature_fns)
        split_idx = tuple(int(i) for i in split_idx)
        X_split = features[:, list(split_idx)]

        # 8. Fit forest.
        forest = _RieszGRF(
            n_estimators=self.n_estimators,
            criterion="mse",
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_var_fraction_leaf=self.min_var_fraction_leaf,
            max_features=self.max_features,
            min_impurity_decrease=self.min_impurity_decrease,
            max_samples=self.max_samples,
            min_balancedness_tol=self.min_balancedness_tol,
            honest=self.honest,
            inference=self.inference,
            fit_intercept=self.fit_intercept,
            subforest_size=self.subforest_size,
            n_jobs=self.n_jobs,
            random_state=seed,
            verbose=self.verbose,
            warm_start=False,
        )
        forest.fit(X_split, T_pack, y_pack)

        predictor = ForestPredictor(
            forest=forest,
            loss=loss,
            base_score=base_score,
            riesz_feature_fns=self.riesz_feature_fns,
            feature_keys=tuple(feature_keys),
            split_feature_indices=split_idx,
        )

        val_score = None
        if rows_valid:
            val_score = _holdout_riesz_loss(rows_valid, estimand, predictor, loss)

        return FitResult(
            predictor=predictor,
            best_iteration=None,
            best_score=val_score,
            history=None,
        )
