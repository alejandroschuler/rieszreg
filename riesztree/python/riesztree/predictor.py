"""Predictor wrapping the fitted tree.

Holds the tree datastructure plus the loss for link mapping. Squared loss
uses the identity link so ``predict_eta`` and ``predict_alpha`` agree; for
KL / Bernoulli the leaf stores α directly (post-projected to the link
domain), and ``predict_eta`` returns ``loss.alpha_to_eta(α)``.

Registers itself with rieszreg's loader registry on import.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Sequence

import numpy as np

from rieszreg import LossSpec, register_predictor_loader
from rieszreg.losses import loss_from_spec

from .tree import Node, from_dict, predict_array, to_dict


@dataclass
class RieszTreePredictor:
    tree: Node
    loss: LossSpec
    base_score: float
    feature_keys: tuple[str, ...]
    categorical_features: tuple[int, ...] = ()

    kind: ClassVar[str] = "riesztree"

    def predict_alpha(self, features: np.ndarray) -> np.ndarray:
        """Per-leaf α* sits in α-space already; return as-is."""
        return predict_array(self.tree, features)

    def predict_eta(self, features: np.ndarray) -> np.ndarray:
        alpha = self.predict_alpha(features)
        return self.loss.alpha_to_eta(alpha)

    # ---- serialization ----

    def save(self, dir_path) -> None:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": self.kind,
            "loss": self.loss.to_spec(),
            "base_score": float(self.base_score),
            "feature_keys": list(self.feature_keys),
            "categorical_features": list(int(i) for i in self.categorical_features),
            "tree": to_dict(self.tree),
        }
        with open(path / "predictor.json", "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, dir_path, *, base_score, loss, best_iteration):
        del best_iteration
        path = Path(dir_path)
        with open(path / "predictor.json") as f:
            payload = json.load(f)
        return cls(
            tree=from_dict(payload["tree"]),
            loss=loss if loss is not None else loss_from_spec(payload["loss"]),
            base_score=float(payload["base_score"]) if base_score is None else float(base_score),
            feature_keys=tuple(payload["feature_keys"]),
            categorical_features=tuple(int(i) for i in payload.get("categorical_features", [])),
        )


register_predictor_loader("riesztree", RieszTreePredictor.load)
