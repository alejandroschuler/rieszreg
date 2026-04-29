"""Predictor for the augmentation-style forest backend.

Mirrors ``ForestPredictor`` but trained on augmented evaluation points, so
prediction takes the full feature vector — there is no ``split_feature_indices``
because the splitter saw every feature dimension at fit time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar

import numpy as np

from rieszreg import LossSpec, register_predictor_loader
from rieszreg.losses import loss_from_spec


def _const_phi(features: np.ndarray) -> np.ndarray:
    return np.ones(len(features), dtype=float)


@dataclass
class AugForestPredictor:
    forest: object  # _RieszGRF
    loss: LossSpec
    base_score: float
    riesz_feature_fns: list[Callable] | None

    kind: ClassVar[str] = "aug-forestriesz"

    def _phi(self, features: np.ndarray) -> np.ndarray:
        if self.riesz_feature_fns is None:
            return _const_phi(features).reshape(-1, 1)
        return np.column_stack(
            [np.asarray(fn(features), dtype=float) for fn in self.riesz_feature_fns]
        )

    def predict_eta(self, features: np.ndarray) -> np.ndarray:
        theta = np.asarray(self.forest.predict(features))
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)
        phi = self._phi(features)
        return (theta * phi).sum(axis=1) + self.base_score

    def predict_alpha(self, features: np.ndarray) -> np.ndarray:
        return self.loss.link_to_alpha(self.predict_eta(features))

    def save(self, dir_path) -> None:
        import joblib

        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.forest, path / "forest.joblib")
        extras = {
            "kind": self.kind,
            "loss": self.loss.to_spec(),
            "base_score": float(self.base_score),
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
            riesz_feature_fns=None,
        )


register_predictor_loader("aug-forestriesz", AugForestPredictor.load)
