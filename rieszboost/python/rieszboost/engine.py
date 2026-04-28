"""Fast engine: data augmentation + xgboost custom objective.

For each training row i, the user's m extracts a finite list of (coefficient,
point) pairs. We assemble an augmented dataset where every row j contributes a
loss term

    a_j * alpha(z_j)^2 + b_j * alpha(z_j)

with gradient 2*a_j*F_j + b_j and Hessian 2*a_j. The original row i contributes
(a=1, b=0) at point z_i (the alpha^2 term in the Riesz loss); each pair (c, p)
from m(z_i) contributes (a=0, b=-2c) at point p. Duplicate points within a row
are merged by summing (a, b).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import xgboost as xgb

from .losses import LossSpec, SquaredLoss
from .tracer import trace


@dataclass
class AugmentedDataset:
    features: np.ndarray  # shape (n_aug, n_features)
    a: np.ndarray         # shape (n_aug,) — quadratic coefficient
    b: np.ndarray         # shape (n_aug,) — linear coefficient
    origin_index: np.ndarray  # shape (n_aug,) — index into original rows


def _row_to_features(point: dict[str, Any], feature_keys: Sequence[str]) -> np.ndarray:
    return np.asarray([point[k] for k in feature_keys], dtype=float)


def build_augmented(
    rows: Sequence[dict[str, Any]],
    m: Callable,
    feature_keys: Sequence[str],
) -> AugmentedDataset:
    """Trace m on each row and assemble the augmented (features, a, b) arrays."""
    feats: list[np.ndarray] = []
    a_list: list[float] = []
    b_list: list[float] = []
    origin: list[int] = []

    for i, z in enumerate(rows):
        # Per-row accumulator: point_key -> (a, b)
        acc: dict[tuple, tuple[float, float]] = {}
        # Original row contributes the alpha^2 term at z itself.
        z_pt = {k: z[k] for k in feature_keys}
        z_key = tuple(z_pt[k] for k in feature_keys)
        acc[z_key] = (1.0, 0.0)

        # Linear functional contributes -2c * alpha(p) for each (c, p).
        for coef, point in trace(m, z):
            missing = [k for k in feature_keys if k not in point]
            if missing:
                raise ValueError(
                    f"m evaluated alpha at a point missing keys {missing}; "
                    f"all feature_keys {list(feature_keys)} must be specified."
                )
            key = tuple(point[k] for k in feature_keys)
            cur_a, cur_b = acc.get(key, (0.0, 0.0))
            acc[key] = (cur_a, cur_b - 2.0 * coef)

        for key, (a, b) in acc.items():
            feats.append(np.asarray(key, dtype=float))
            a_list.append(a)
            b_list.append(b)
            origin.append(i)

    return AugmentedDataset(
        features=np.vstack(feats) if feats else np.zeros((0, len(feature_keys))),
        a=np.asarray(a_list, dtype=float),
        b=np.asarray(b_list, dtype=float),
        origin_index=np.asarray(origin, dtype=np.int64),
    )


def _make_objective(
    a: np.ndarray,
    b: np.ndarray,
    loss_spec: LossSpec,
    hessian_floor: float = 2.0,
):
    """xgboost custom objective. Delegates to `loss_spec.gradient` / `.hessian`.

    For SquaredLoss the floor is critical: counterfactual rows have true
    Hessian 0 (only the b·F linear term in the loss). xgboost's second-order
    leaf weight is ``-G/(H+reg_lambda)``; if H ≈ 0 for a leaf full of
    counterfactuals, the weight becomes ``-G/reg_lambda`` and blows up unless
    reg_lambda is huge. Flooring at 2.0 (the natural Hessian of original a=1
    rows) keeps every row contributing meaningfully to H, mimicking the
    uniform weighting that first-order gradient boosting (Friedman 2001) uses
    by construction.
    """

    def obj(preds: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        del dtrain
        grad = loss_spec.gradient(a, b, preds)
        hess = loss_spec.hessian(a, b, preds, hessian_floor)
        return grad, hess

    return obj


def _make_riesz_metric(
    a_val: np.ndarray,
    b_val: np.ndarray,
    n_val_rows: int,
    loss_spec: LossSpec,
):
    """xgboost custom_metric returning the per-row validation Riesz loss.
    xgboost predicts η; we apply link_to_alpha and evaluate the loss in α-space."""
    def metric(predt: np.ndarray, dval) -> tuple[str, float]:
        del dval
        alpha = loss_spec.link_to_alpha(predt)
        per_row = loss_spec.loss_row(a_val, b_val, alpha)
        return "riesz_loss", float(np.sum(per_row) / n_val_rows)
    return metric


def fit(
    rows: Sequence[dict[str, Any]],
    m: Callable,
    feature_keys: Sequence[str],
    *,
    loss_spec: LossSpec | None = None,
    valid_rows: Sequence[dict[str, Any]] | None = None,
    num_boost_round: int = 100,
    early_stopping_rounds: int | None = None,
    learning_rate: float = 0.1,
    max_depth: int = 5,
    reg_lambda: float = 1.0,
    subsample: float = 1.0,
    base_score: float | None = None,
    seed: int = 0,
    init: str | float | None = None,
    hessian_floor: float = 2.0,
    verbose_eval: bool | int = False,
) -> "RieszBooster":
    """Fit a Riesz representer to the user's m via the fast augmented-data path.

    `loss_spec` defaults to `SquaredLoss()`; pass `KLLoss()` for density-ratio
    targets where α₀ is positive (TSM, IPSI). If `valid_rows` is given, a
    per-row validation Riesz loss is computed each round; pair with
    `early_stopping_rounds` to halt when it stops improving.
    """
    if loss_spec is None:
        loss_spec = SquaredLoss()

    aug = build_augmented(rows, m, feature_keys)
    loss_spec.validate_coefficients(aug.b)

    if init is None:
        init = loss_spec.default_init_alpha()
    if init == "m1":
        per_row = [sum(c for c, _ in trace(m, z)) for z in rows]
        init_alpha = float(np.mean(per_row))
    elif isinstance(init, (int, float)):
        init_alpha = float(init)
    else:
        raise ValueError(f"init must be a float or 'm1'; got {init!r}")
    # base_score lives in η space (xgboost adds it to additive tree predictions).
    base_score = float(loss_spec.alpha_to_eta(init_alpha))

    dtrain = xgb.DMatrix(aug.features)
    params = {
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "reg_lambda": reg_lambda,
        "subsample": subsample,
        "base_score": base_score,
        "seed": seed,
        "disable_default_eval_metric": 1,
    }

    evals: list[tuple] = []
    custom_metric = None
    if valid_rows is not None:
        aug_val = build_augmented(valid_rows, m, feature_keys)
        loss_spec.validate_coefficients(aug_val.b)
        dvalid = xgb.DMatrix(aug_val.features)
        evals = [(dvalid, "valid")]
        custom_metric = _make_riesz_metric(
            aug_val.a, aug_val.b, len(valid_rows), loss_spec
        )
    elif early_stopping_rounds is not None:
        raise ValueError("early_stopping_rounds requires valid_rows to be provided")

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        obj=_make_objective(aug.a, aug.b, loss_spec, hessian_floor=hessian_floor),
        evals=evals,
        custom_metric=custom_metric,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )
    best_iteration = getattr(booster, "best_iteration", None)
    best_score = getattr(booster, "best_score", None)
    return RieszBooster(
        booster=booster,
        feature_keys=tuple(feature_keys),
        base_score=base_score,
        best_iteration=best_iteration,
        best_score=float(best_score) if best_score is not None else None,
        loss_spec=loss_spec,
    )


@dataclass
class RieszBooster:
    booster: xgb.Booster
    feature_keys: tuple[str, ...]
    base_score: float
    best_iteration: int | None = None
    best_score: float | None = None
    loss_spec: LossSpec | None = None

    def _predict_eta(self, dmatrix) -> np.ndarray:
        """Raw additive xgboost output (η)."""
        if self.best_iteration is not None:
            return self.booster.predict(
                dmatrix, iteration_range=(0, self.best_iteration + 1)
            )
        return self.booster.predict(dmatrix)

    def _predict_dmatrix(self, dmatrix) -> np.ndarray:
        eta = self._predict_eta(dmatrix)
        ls = self.loss_spec if self.loss_spec is not None else SquaredLoss()
        return np.asarray(ls.link_to_alpha(eta))

    def predict(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        X = np.asarray(
            [[row[k] for k in self.feature_keys] for row in rows], dtype=float
        )
        return self._predict_dmatrix(xgb.DMatrix(X))

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        return self._predict_dmatrix(xgb.DMatrix(np.asarray(X, dtype=float)))

    def riesz_loss(
        self,
        rows: Sequence[dict[str, Any]],
        m: Callable,
    ) -> float:
        """Per-row empirical Riesz loss on rows, using this booster's loss_spec
        (defaults to SquaredLoss if missing)."""
        aug = build_augmented(rows, m, self.feature_keys)
        F = self.predict_array(aug.features)
        ls = self.loss_spec if self.loss_spec is not None else SquaredLoss()
        return float(np.sum(ls.loss_row(aug.a, aug.b, F)) / len(rows))
