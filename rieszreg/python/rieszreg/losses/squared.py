"""Squared Riesz loss: φ(t) = t², identity link."""

from __future__ import annotations

import numpy as np


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

    def to_spec(self) -> dict:
        return {"type": "SquaredLoss", "args": {}}
