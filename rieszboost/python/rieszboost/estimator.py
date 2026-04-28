"""Main user-facing class: a sklearn-compatible Riesz representer estimator.

Designed to compose with `sklearn.model_selection.GridSearchCV`,
`cross_val_predict`, `clone`, etc. Configuration objects (estimand, loss,
backend) are baked in at construction; standard `.fit / .predict / .score`
at use time. Mirrors ngboost's API style.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from .augmentation import AugmentedDataset, build_augmented
from .backends import Backend, XGBoostBackend
from .estimand import Estimand
from .losses import LossSpec, SquaredLoss
from .tracer import trace


def _is_dataframe(X) -> bool:
    return hasattr(X, "columns") and hasattr(X, "iloc")


def _rows_from_X(X, estimand: Estimand) -> list[dict]:
    """Convert ndarray or DataFrame X into a list of row-dicts keyed by
    `estimand.feature_keys` and `estimand.extra_keys`. Ndarray input is
    interpreted column-by-column in `feature_keys` order; DataFrame columns
    are matched by name."""
    if _is_dataframe(X):
        cols_needed = list(estimand.feature_keys) + list(estimand.extra_keys)
        missing = [c for c in cols_needed if c not in X.columns]
        if missing:
            raise ValueError(
                f"DataFrame is missing columns required by estimand "
                f"{estimand.name!r}: {missing}"
            )
        rows = []
        for i in range(len(X)):
            row = {}
            for k in cols_needed:
                v = X[k].iloc[i]
                row[k] = v
            rows.append(row)
        return rows

    arr = np.asarray(X)
    if estimand.extra_keys:
        raise ValueError(
            f"Estimand {estimand.name!r} requires per-row payload "
            f"({estimand.extra_keys}), which an ndarray cannot carry. "
            "Pass a pandas DataFrame instead."
        )
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] != len(estimand.feature_keys):
        raise ValueError(
            f"Estimand {estimand.name!r} expects {len(estimand.feature_keys)} "
            f"feature columns ({estimand.feature_keys}), got X.shape[1]="
            f"{arr.shape[1]}."
        )
    return [
        {k: arr[i, j] for j, k in enumerate(estimand.feature_keys)}
        for i in range(arr.shape[0])
    ]


def _features_from_rows(rows: Sequence[dict], estimand: Estimand) -> np.ndarray:
    return np.asarray(
        [[row[k] for k in estimand.feature_keys] for row in rows], dtype=float
    )


def _split_X(X, validation_fraction: float, random_state: int):
    n = len(X) if _is_dataframe(X) else len(np.asarray(X))
    if validation_fraction <= 0:
        return X, None
    idx = np.arange(n)
    tr_idx, va_idx = train_test_split(
        idx, test_size=validation_fraction, random_state=random_state
    )
    if _is_dataframe(X):
        return X.iloc[tr_idx], X.iloc[va_idx]
    arr = np.asarray(X)
    return arr[tr_idx], arr[va_idx]


class RieszBooster(BaseEstimator):
    """Gradient-boosted estimator for the Riesz representer α₀ of a linear
    functional. ngboost / sklearn-style object-oriented API.

    Parameters
    ----------
    estimand : Estimand
        Carries `feature_keys`, `extra_keys`, and the `m(z, alpha)` callable.
        Required.
    backend : Backend, default=XGBoostBackend()
        Where the actual tree fitting happens. Swap to `SklearnBackend(...)`
        to use a non-tree base learner (KernelRidge, MLPs, etc.).
    loss : LossSpec, default=SquaredLoss()
        The Bregman-Riesz loss to minimize. `KLLoss()` is the alternative.
    n_estimators : int, default=200
    learning_rate : float, default=0.05
    max_depth : int, default=4
    reg_lambda : float, default=1.0
    subsample : float, default=1.0
    early_stopping_rounds : int or None
        If set, requires either `validation_fraction>0` or `eval_set=...` at
        fit time.
    validation_fraction : float, default=0.0
        Fraction of training data held out internally for early stopping.
    init : float, "m1", or None
        α-space initialization. None defers to `loss.default_init_alpha()`.
    random_state : int, default=0
    """

    def __init__(
        self,
        estimand: Estimand,
        backend: Backend | None = None,
        loss: LossSpec | None = None,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        reg_lambda: float = 1.0,
        subsample: float = 1.0,
        early_stopping_rounds: int | None = None,
        validation_fraction: float = 0.0,
        init: float | str | None = None,
        random_state: int = 0,
    ):
        # Store args verbatim — required by sklearn for clone / get_params.
        self.estimand = estimand
        self.backend = backend
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.init = init
        self.random_state = random_state

    # ---- internal accessors that resolve defaults ----

    def _resolved_backend(self) -> Backend:
        return self.backend if self.backend is not None else XGBoostBackend()

    def _resolved_loss(self) -> LossSpec:
        return self.loss if self.loss is not None else SquaredLoss()

    def _xgb_hyperparams(self) -> dict:
        # Only meaningful for XGBoostBackend; SklearnBackend ignores them.
        return {
            "max_depth": self.max_depth,
            "reg_lambda": self.reg_lambda,
            "subsample": self.subsample,
        }

    # ---- sklearn API ----

    def fit(self, X, y=None, eval_set=None) -> "RieszBooster":
        loss = self._resolved_loss()
        backend = self._resolved_backend()

        # Resolve init in α-space, then convert to η.
        init_arg = self.init
        if init_arg is None:
            init_alpha = loss.default_init_alpha()
        elif init_arg == "m1":
            rows_for_init = _rows_from_X(X, self.estimand)
            per_row = [sum(c for c, _ in trace(self.estimand, z)) for z in rows_for_init]
            init_alpha = float(np.mean(per_row))
        elif isinstance(init_arg, (int, float)):
            init_alpha = float(init_arg)
        else:
            raise ValueError(f"init must be float, 'm1', or None; got {init_arg!r}")
        base_score = float(loss.alpha_to_eta(init_alpha))

        # Resolve validation slice.
        if eval_set is not None:
            X_train, X_valid = X, eval_set
        elif self.validation_fraction > 0 or self.early_stopping_rounds is not None:
            X_train, X_valid = _split_X(X, self.validation_fraction or 0.2, self.random_state)
        else:
            X_train, X_valid = X, None

        rows_train = _rows_from_X(X_train, self.estimand)
        aug_train = build_augmented(rows_train, self.estimand)

        aug_valid = None
        if X_valid is not None and len(X_valid) > 0:
            rows_valid = _rows_from_X(X_valid, self.estimand)
            aug_valid = build_augmented(rows_valid, self.estimand)

        result = backend.fit_augmented(
            aug_train, aug_valid, loss,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            base_score=base_score,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            hyperparams=self._xgb_hyperparams(),
        )

        self.predictor_ = result.predictor
        self.best_iteration_ = result.best_iteration
        self.best_score_ = result.best_score
        self.base_score_ = base_score
        self.loss_ = loss
        self.feature_keys_ = self.estimand.feature_keys
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "predictor_"):
            raise RuntimeError("RieszBooster is not fitted yet. Call .fit() first.")
        rows = _rows_from_X(X, self.estimand)
        feats = _features_from_rows(rows, self.estimand)
        return self.predictor_.predict_alpha(feats)

    def riesz_loss(self, X) -> float:
        """Per-row empirical Riesz loss on X under this booster's loss."""
        if not hasattr(self, "predictor_"):
            raise RuntimeError("RieszBooster is not fitted yet.")
        rows = _rows_from_X(X, self.estimand)
        aug = build_augmented(rows, self.estimand)
        eta = self.predictor_.predict_eta(aug.features)
        alpha = self.loss_.link_to_alpha(eta)
        return float(np.sum(self.loss_.loss_row(aug.a, aug.b, alpha)) / aug.n_rows)

    def score(self, X, y=None) -> float:
        """Return negative held-out Riesz loss — sklearn convention is
        higher-is-better for `score`, so we flip the sign of the loss."""
        return -self.riesz_loss(X)

    def diagnose(self, X, **kwargs):
        from .diagnostics import diagnose
        return diagnose(booster=self, X=X, **kwargs)

    # ---- serialization ----

    def save(self, path) -> None:
        """Save a fitted booster to a directory.

        Writes:
          - `booster.ubj` (XGBoostBackend) or `predictor.joblib` (SklearnBackend)
          - `metadata.json` with the loss spec, estimand factory_spec (if
            built-in), feature_keys, base_score, best_iteration_, etc.

        Custom (non-built-in) estimands cannot be auto-reconstructed; the file
        will save fine, but `RieszBooster.load(path)` will require the user to
        pass `estimand=...` explicitly.
        """
        import json
        from pathlib import Path

        if not hasattr(self, "predictor_"):
            raise RuntimeError("Cannot save unfitted RieszBooster. Call .fit() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.predictor_.save(path)

        metadata = {
            "rieszboost_format_version": 1,
            "predictor_kind": self.predictor_.kind,
            "loss": self.loss_.to_spec(),
            "estimand_factory_spec": self.estimand.factory_spec,  # None if custom
            "feature_keys": list(self.feature_keys_),
            "extra_keys": list(self.estimand.extra_keys),
            "base_score": self.base_score_,
            "best_iteration": self.best_iteration_,
            "best_score": self.best_score_,
            # Constructor hyperparameters (informational; not used on load).
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "reg_lambda": self.reg_lambda,
                "subsample": self.subsample,
                "early_stopping_rounds": self.early_stopping_rounds,
                "validation_fraction": self.validation_fraction,
                "init": self.init,
                "random_state": self.random_state,
            },
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path, *, estimand: "Estimand | None" = None) -> "RieszBooster":
        """Load a booster from a directory written by `save(...)`.

        For custom (non-built-in) estimands, pass `estimand=` to inject the
        original Estimand instance. For built-ins, reconstruction is
        automatic.
        """
        import json
        from pathlib import Path

        from .backends.sklearn import SklearnPredictor
        from .backends.xgboost import XGBoostPredictor
        from .estimand import estimand_from_spec
        from .losses import loss_from_spec

        path = Path(path)
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        loss = loss_from_spec(metadata["loss"])

        if estimand is None:
            spec = metadata.get("estimand_factory_spec")
            if spec is None:
                raise ValueError(
                    f"Saved booster at {path} has a custom (non-built-in) "
                    "estimand. Pass `estimand=...` explicitly to "
                    "RieszBooster.load(path, estimand=my_estimand)."
                )
            estimand = estimand_from_spec(spec)

        kind = metadata["predictor_kind"]
        if kind == "xgboost":
            predictor = XGBoostPredictor.load(
                path,
                base_score=metadata["base_score"],
                loss=loss,
                best_iteration=metadata.get("best_iteration"),
            )
        elif kind == "sklearn":
            predictor = SklearnPredictor.load(
                path,
                base_score=metadata["base_score"],
                loss=loss,
                best_iteration=metadata.get("best_iteration"),
            )
        else:
            raise ValueError(f"Unknown predictor kind: {kind!r}")

        # Reconstruct the booster shell with the original hyperparameters
        # (so get_params reflects what fit was called with).
        hp = metadata.get("hyperparameters", {})
        booster = cls(
            estimand=estimand,
            loss=loss,
            n_estimators=hp.get("n_estimators", 200),
            learning_rate=hp.get("learning_rate", 0.05),
            max_depth=hp.get("max_depth", 4),
            reg_lambda=hp.get("reg_lambda", 1.0),
            subsample=hp.get("subsample", 1.0),
            early_stopping_rounds=hp.get("early_stopping_rounds"),
            validation_fraction=hp.get("validation_fraction", 0.0),
            init=hp.get("init"),
            random_state=hp.get("random_state", 0),
        )
        booster.predictor_ = predictor
        booster.best_iteration_ = metadata.get("best_iteration")
        booster.best_score_ = metadata.get("best_score")
        booster.base_score_ = metadata["base_score"]
        booster.loss_ = loss
        booster.feature_keys_ = tuple(metadata["feature_keys"])
        return booster
