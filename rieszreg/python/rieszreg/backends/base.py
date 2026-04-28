"""Backend protocol: the swappable component that consumes augmented data
and a LossSpec and produces a fitted Predictor.

Concrete backends live in implementation packages (rieszboost, krrr, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..augmentation import AugmentedDataset
from ..losses import LossSpec


class Predictor(Protocol):
    """Output of `Backend.fit_augmented`. RieszEstimator delegates to this for
    prediction. Implementations should apply the loss spec's link in
    `predict_alpha` so callers see α̂, not raw η.

    Attributes
    ----------
    kind : str
        Short identifier (e.g. "xgboost", "sklearn", "kernel-ridge") used by
        the registry-based load path.
    """

    kind: str

    def predict_eta(self, features: np.ndarray) -> np.ndarray: ...
    def predict_alpha(self, features: np.ndarray) -> np.ndarray: ...

    def save(self, dir_path) -> None:
        """Write the binary payload (booster.ubj, predictor.joblib, ...) into
        `dir_path`. Metadata (loss, estimand spec, hyperparameters) is written
        by the orchestrator estimator separately."""
        ...


@dataclass
class FitResult:
    predictor: Predictor
    best_iteration: int | None = None
    best_score: float | None = None
    history: list[float] | None = None


class Backend(Protocol):
    def fit_augmented(
        self,
        aug_train: AugmentedDataset,
        aug_valid: AugmentedDataset | None,
        loss: LossSpec,
        *,
        n_estimators: int,
        learning_rate: float,
        base_score: float,
        early_stopping_rounds: int | None,
        random_state: int,
        hyperparams: dict[str, Any],
    ) -> FitResult:
        ...


# ----- Predictor loader registry (used by RieszEstimator.load) -----

_PREDICTOR_LOADERS: dict[str, Any] = {}


def register_predictor_loader(kind: str, loader) -> None:
    """Register a loader callable for a predictor kind.

    Implementation packages call this at import time:

        register_predictor_loader("xgboost", XGBoostPredictor.load)

    The loader signature is `(dir_path, base_score, loss, best_iteration) -> Predictor`.
    """
    _PREDICTOR_LOADERS[kind] = loader


def load_predictor(kind: str, dir_path, *, base_score, loss, best_iteration):
    """Look up a registered loader and instantiate the predictor."""
    if kind not in _PREDICTOR_LOADERS:
        raise ValueError(
            f"No loader registered for predictor kind {kind!r}. "
            f"Import the implementation package (e.g. `import rieszboost`) "
            f"before calling .load(...). Registered kinds: {sorted(_PREDICTOR_LOADERS)}."
        )
    return _PREDICTOR_LOADERS[kind](
        dir_path, base_score=base_score, loss=loss, best_iteration=best_iteration
    )
