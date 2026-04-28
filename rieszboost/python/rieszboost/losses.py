"""LossSpec abstraction for Bregman-Riesz losses.

The Bregman-Riesz loss generalizes the squared Riesz loss via a strictly convex
potential φ:

    L_φ(α) = const + E[ψ(α(z))] - E[m(z, φ'(α))]

with ψ(t) = t·φ'(t) - φ(t). For finite-point m, the augmented-dataset
reformulation gives a per-row loss term

    a_j · ψ(α(z̃_j)) + (b_j / 2) · φ'(α(z̃_j))

where (a_j, b_j) are augmented coefficients (a_j = 1 for original rows, a_j = 0
and b_j = -2·c_k for the k-th counterfactual point of m).

xgboost's additive booster outputs `η = sum of trees + base_score`. Each
LossSpec defines a **link** mapping η → α. SquaredLoss uses the identity link
(η = α). KLLoss uses the exp link (α = exp(η)) so that α stays positive
under all leaf updates. xgboost's gradient and Hessian are computed in η space.
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


class SquaredLoss:
    """φ(t) = t². ψ(t) = t². Identity link η = α.
    Per-row loss = a·α² + b·α. Gradient (in η) = 2a·η + b. Hessian = 2a.
    """

    name = "squared"

    def link_to_alpha(self, eta):
        return eta

    def alpha_to_eta(self, alpha):
        return alpha

    def loss_row(self, a, b, alpha):
        return a * alpha**2 + b * alpha

    def gradient(self, a, b, eta):
        return 2.0 * a * eta + b

    def hessian(self, a, b, eta, hessian_floor):
        del b, eta
        return np.maximum(2.0 * a, hessian_floor)

    def default_init_alpha(self):
        return 0.0

    def validate_coefficients(self, b):
        return  # any signed b ok


class KLLoss:
    """φ(t) = t·log(t). ψ(t) = t. Exp link: α = exp(η) so α > 0 always.
    Per-row loss in α: a·α + (b/2)·log(α).
    With η = log(α), loss in η: a·exp(η) + (b/2)·η.
    Gradient (in η) = a·exp(η) + b/2.
    Hessian (in η) = a·exp(η)  (positive whenever a ≥ 0; floored).

    Requires all m-coefficients to be non-negative (b ≤ 0 in augmented data),
    which restricts to density-ratio-style estimands (TSM, IPSI, etc.).
    """

    name = "kl"

    def __init__(self, max_eta: float = 50.0):
        # Clip η before exp() to avoid overflow when xgboost makes a big step.
        self.max_eta = float(max_eta)

    def _clip(self, eta):
        return np.clip(eta, -self.max_eta, self.max_eta)

    def link_to_alpha(self, eta):
        return np.exp(self._clip(eta))

    def alpha_to_eta(self, alpha):
        if isinstance(alpha, np.ndarray):
            if np.any(alpha <= 0):
                raise ValueError("KLLoss requires positive alpha for init.")
            return np.log(alpha)
        if alpha <= 0:
            raise ValueError("KLLoss requires positive alpha for init.")
        return float(np.log(alpha))

    def loss_row(self, a, b, alpha):
        # alpha is already α (post-link); guard log against zero.
        alpha = np.maximum(alpha, np.exp(-self.max_eta))
        return a * alpha + 0.5 * b * np.log(alpha)

    def gradient(self, a, b, eta):
        alpha = np.exp(self._clip(eta))
        return a * alpha + 0.5 * b

    def hessian(self, a, b, eta, hessian_floor):
        del b
        alpha = np.exp(self._clip(eta))
        return np.maximum(a * alpha, hessian_floor)

    def default_init_alpha(self):
        return 1.0

    def validate_coefficients(self, b):
        if np.any(b > 0):
            raise ValueError(
                "KLLoss requires all m-coefficients to be non-negative "
                "(equivalently: all augmented `b` values <= 0). Your m has at "
                "least one row with a positive linear coefficient — try "
                "SquaredLoss instead, or restrict to density-ratio estimands "
                "(TSM, IPSI, etc.)."
            )
