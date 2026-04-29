"""Predictor wrapping the fitted GRF + sieve evaluation.

Stores the forest plus the metadata needed to map a query feature vector back
through the basis: ``α(z) = θ(z_split) · φ(z) + base_score``. Registers itself
with the rieszreg loader registry on import.

User-supplied basis callables are not pickled. Save persists the forest and
small scalar metadata; load requires the user to repass ``riesz_feature_fns=``
to ``ForestRieszRegressor.load`` for sieve-fit estimators.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar

import numpy as np

from rieszreg import LossSpec, register_predictor_loader
from rieszreg.losses import loss_from_spec


def _const_phi(features: np.ndarray) -> np.ndarray:
    return np.ones(len(features), dtype=float)


@dataclass
class ForestPredictor:
    forest: object  # _RieszGRF; declared opaquely to avoid an import cycle
    loss: LossSpec
    base_score: float
    riesz_feature_fns: list[Callable] | None
    feature_keys: tuple[str, ...]
    split_feature_indices: tuple[int, ...]

    kind: ClassVar[str] = "forestriesz"

    def _phi(self, features: np.ndarray) -> np.ndarray:
        if self.riesz_feature_fns is None:
            return _const_phi(features).reshape(-1, 1)
        return np.column_stack(
            [np.asarray(fn(features), dtype=float) for fn in self.riesz_feature_fns]
        )

    def _x_split(self, features: np.ndarray) -> np.ndarray:
        return features[:, list(self.split_feature_indices)]

    def predict_eta(self, features: np.ndarray) -> np.ndarray:
        theta = np.asarray(self.forest.predict(self._x_split(features)))
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)
        phi = self._phi(features)
        return (theta * phi).sum(axis=1) + self.base_score

    def predict_alpha(self, features: np.ndarray) -> np.ndarray:
        return self.loss.link_to_alpha(self.predict_eta(features))

    def predict_interval(
        self, features: np.ndarray, *, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        if not getattr(self.forest, "inference", False):
            raise RuntimeError(
                "predict_interval requires honest=True and inference=True at "
                "fit time. Re-fit ForestRieszRegressor with both flags enabled."
            )
        if self.riesz_feature_fns is not None:
            raise NotImplementedError(
                "predict_interval is supported for locally constant only in v1. "
                "The sieve case requires a delta-method on θ' φ(x) that is "
                "planned for v2. Re-fit without riesz_feature_fns to use "
                "intervals."
            )
        lb, ub = self.forest.predict_interval(self._x_split(features), alpha=alpha)
        return (
            np.asarray(lb).flatten() + self.base_score,
            np.asarray(ub).flatten() + self.base_score,
        )

    # ---- serialization ----

    def save(self, dir_path) -> None:
        import joblib

        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.forest, path / "forest.joblib")
        extras = {
            "kind": self.kind,
            "loss": self.loss.to_spec(),
            "base_score": float(self.base_score),
            "feature_keys": list(self.feature_keys),
            "split_feature_indices": list(self.split_feature_indices),
            "has_sieve": self.riesz_feature_fns is not None,
        }
        with open(path / "predictor_extras.json", "w") as f:
            json.dump(extras, f, indent=2)

    @classmethod
    def load(cls, dir_path, *, base_score, loss, best_iteration):
        import joblib

        path = Path(dir_path)
        with open(path / "predictor_extras.json") as f:
            extras = json.load(f)
        forest = joblib.load(path / "forest.joblib")
        return cls(
            forest=forest,
            loss=loss if loss is not None else loss_from_spec(extras["loss"]),
            base_score=float(extras["base_score"]) if base_score is None else base_score,
            riesz_feature_fns=None,  # caller patches in via ForestRieszRegressor.load
            feature_keys=tuple(extras["feature_keys"]),
            split_feature_indices=tuple(int(i) for i in extras["split_feature_indices"]),
        )


register_predictor_loader("forestriesz", ForestPredictor.load)
