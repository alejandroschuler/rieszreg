"""KL Bregman-Riesz loss: φ(t) = t·log(t), exp link, density-ratio targets."""

from __future__ import annotations

import numpy as np


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
        # Clip η before exp() to avoid overflow when the booster makes a big step.
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

    def to_spec(self) -> dict:
        return {"type": "KLLoss", "args": {"max_eta": self.max_eta}}
