"""LossSpec Protocol for Bregman-Riesz losses.

The Bregman-Riesz loss generalizes the squared Riesz loss via a strictly convex
potential φ:

    L_φ(α) = const + E[ψ(α(z))] - E[m(z, φ'(α))]

with ψ(t) = t·φ'(t) - φ(t). For finite-point m, the augmented-dataset
reformulation gives a per-row loss term

    a_j · ψ(α(z̃_j)) + (b_j / 2) · φ'(α(z̃_j))

where (a_j, b_j) are augmented coefficients (a_j = 1 for original rows, a_j = 0
and b_j = -2·c_k for the k-th counterfactual point of m).

Backends produce a real-valued score `η` (additive trees + base_score for
boosting; closed-form linear combinations for kernel ridge; the network
output for neural backends; per-leaf θ·φ for forests). Each LossSpec
defines a **link** mapping η → α. SquaredLoss uses the identity link
(η = α). KLLoss uses the exp link (α = exp(η)) so that α stays positive.
Gradient and Hessian are computed in η space; backends that need them call
`gradient(...)` and `hessian(...)` (others apply autograd directly to
`loss_row(...)`).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class LossSpec(Protocol):
    name: str

    def link_to_alpha(self, eta: np.ndarray) -> np.ndarray:
        """Inverse link: convert backend output η to α."""
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

    def best_constant_init(self, m_bar: float) -> float:
        """Loss-minimizing constant α* given m̄ = E[m(Z, 1)].

        For any Bregman loss with strictly convex φ, the constant minimizer
        of the per-row Riesz loss `ψ(a) - φ'(a)·m̄` satisfies the FOC
        `ψ'(a) = φ''(a)·m̄`, and `ψ(t) = t·φ'(t) - φ(t)` gives
        `ψ'(t) = t·φ''(t)`, so `a* = m̄`. Implementations project `m̄`
        into the loss's α-domain (KL needs α > 0; Bernoulli needs
        α ∈ (0, 1); BoundedSquared needs α ∈ (lo, hi)).
        """
        ...

    def validate_coefficients(self, b: np.ndarray) -> None:
        """Raise if (a, b) violate this loss's domain."""
        ...

    def to_spec(self) -> dict:
        """Return a JSON-serializable {"type": str, "args": dict} round-trip spec."""
        ...
