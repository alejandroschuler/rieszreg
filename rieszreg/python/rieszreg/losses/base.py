"""LossSpec Protocol for Bregman-Riesz losses.

The Bregman-Riesz loss generalizes the squared Riesz loss via a strictly convex
potential φ:

    L_φ(α) = const + E[ψ(α(z))] - E[m(z, φ'(α))]

with ψ(t) = t·φ'(t) - φ(t). For finite-point m, the augmented-dataset
reformulation gives a per-row loss term

    a_j · ψ(α(z̃_j)) + (b_j / 2) · φ'(α(z̃_j))

where (a_j, b_j) are augmented coefficients (a_j = 1 for original rows, a_j = 0
and b_j = -2·c_k for the k-th counterfactual point of m).

A boosting backend's additive booster outputs `η = sum of trees + base_score`.
Each LossSpec defines a **link** mapping η → α. SquaredLoss uses the identity
link (η = α). KLLoss uses the exp link (α = exp(η)) so that α stays positive
under all leaf updates. Gradient and Hessian are computed in η space.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class LossSpec(Protocol):
    name: str

    def link_to_alpha(self, eta: np.ndarray) -> np.ndarray:
        """Inverse link: convert boosted output η to α."""
        ...

    def alpha_to_eta(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """Forward link: convert α to η (used for `init=` translation)."""
        ...

    def loss_row(self, a: np.ndarray, b: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Per-row loss in α space: a·ψ(α) + (b/2)·φ'(α)."""
        ...

    def gradient(self, a: np.ndarray, b: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """∂loss_row/∂η, using α = link_to_alpha(η)."""
        ...

    def hessian(
        self, a: np.ndarray, b: np.ndarray, eta: np.ndarray, hessian_floor: float
    ) -> np.ndarray:
        """∂²loss_row/∂η² (floored)."""
        ...

    def default_init_alpha(self) -> float:
        """Sensible α-space default for `init=` if user doesn't override."""
        ...

    def validate_coefficients(self, b: np.ndarray) -> None:
        """Raise if (a, b) violate this loss's domain."""
        ...

    def to_spec(self) -> dict:
        """Return a JSON-serializable {"type": str, "args": dict} round-trip spec."""
        ...
