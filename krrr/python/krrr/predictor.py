"""Predictor that satisfies `rieszboost.backends.base.Predictor`.

Wraps a `SolveResult` (from one of the solvers) plus the kernel and loss spec
into something that exposes `predict_eta(X)` / `predict_alpha(X)` for
`RieszBooster.predict` to call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rieszreg.backends.base import register_predictor_loader
from rieszreg.losses import LossSpec, loss_from_spec

from .kernels import Kernel, kernel_from_spec
from .solvers import SolveResult


@dataclass
class KernelPredictor:
    """Wraps a `SolveResult` for prediction. Implements the rieszboost
    `Predictor` protocol (`predict_eta`, `predict_alpha`)."""

    kernel: Kernel
    loss: LossSpec
    result: SolveResult
    base_score: float = 0.0
    feature_keys: tuple[str, ...] = ()

    kind = "krrr"

    def predict_eta(self, features: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(features, dtype=float))
        if self.result.kind == "dual":
            K_new = self.kernel(X, self.result.support)
            eta = K_new @ self.result.gamma
        elif self.result.kind == "primal":
            phi = self.result.feature_map(X)
            eta = phi @ self.result.weights
        else:
            raise ValueError(f"Unknown SolveResult.kind: {self.result.kind!r}")
        return eta + self.base_score

    def predict_alpha(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.loss.link_to_alpha(self.predict_eta(features)))

    # ---- Serialization ---------------------------------------------------

    def save(self, dir_path) -> None:
        import json

        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        payload = {
            "kernel": self.kernel.to_spec(),
            "loss": self.loss.to_spec(),
            "base_score": float(self.base_score),
            "feature_keys": list(self.feature_keys),
            "result_kind": self.result.kind,
            "result_extra": self.result.extra or {},
        }
        with open(dir_path / "predictor.json", "w") as f:
            json.dump(payload, f, indent=2)

        if self.result.kind == "dual":
            np.savez(
                dir_path / "predictor.npz",
                support=self.result.support,
                gamma=self.result.gamma,
            )
        elif self.result.kind == "primal":
            fm = self.result.feature_map
            np.savez(
                dir_path / "predictor.npz",
                weights=self.result.weights,
                rff_W=fm.W,
                rff_b=fm.b,
                rff_scale=np.asarray([fm.scale]),
            )

    @classmethod
    def load(cls, dir_path, base_score=None, loss=None, best_iteration=None):
        import json

        from .solvers.rff import RFFFeatureMap

        dir_path = Path(dir_path)
        with open(dir_path / "predictor.json") as f:
            payload = json.load(f)
        npz = np.load(dir_path / "predictor.npz")

        kernel = kernel_from_spec(payload["kernel"])
        loss_loaded = loss if loss is not None else loss_from_spec(payload["loss"])
        result_kind = payload["result_kind"]

        if result_kind == "dual":
            result = SolveResult(
                kind="dual",
                support=npz["support"],
                gamma=npz["gamma"],
                extra=payload.get("result_extra", {}),
            )
        elif result_kind == "primal":
            fm = RFFFeatureMap(
                W=npz["rff_W"], b=npz["rff_b"], scale=float(npz["rff_scale"][0])
            )
            result = SolveResult(
                kind="primal",
                weights=npz["weights"],
                feature_map=fm,
                extra=payload.get("result_extra", {}),
            )
        else:
            raise ValueError(f"Unknown result_kind: {result_kind!r}")

        bs = payload["base_score"] if base_score is None else float(base_score)
        return cls(
            kernel=kernel,
            loss=loss_loaded,
            result=result,
            base_score=bs,
            feature_keys=tuple(payload.get("feature_keys", ())),
        )


register_predictor_loader("krrr", KernelPredictor.load)
