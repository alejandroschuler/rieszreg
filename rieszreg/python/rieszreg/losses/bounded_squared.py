"""Squared loss with sigmoid-scaled link forcing α ∈ (lo, hi)."""

from __future__ import annotations

import numpy as np


class BoundedSquaredLoss:
    """Squared loss in α-space, but with a sigmoid-scaled link forcing
    α ∈ (lo, hi). Useful when you have a hard prior bound on the representer
    (e.g. trimmed propensity 1/π̂ ∈ [1, 1/ε]).

    Loss in α: a·α² + b·α  (same shape as `SquaredLoss`).
    Link: α = lo + R · σ(η),  where R = hi − lo.
    Gradient (η) = (2a·α + b) · R · σ(η)·(1 − σ(η)).
    Hessian  (η) = 2a·(R·σ(1−σ))² + (2a·α + b)·R·σ(1−σ)(1 − 2σ).

    Strictly speaking this is squared loss with a saturating reparameterization,
    not a *new* Bregman divergence. Predictions stay in (lo, hi) by
    construction; if true α₀ is outside that range the fit saturates.
    """

    name = "bounded_squared"

    def __init__(self, lo: float, hi: float, max_abs_eta: float = 30.0):
        if not (lo < hi):
            raise ValueError(f"lo ({lo}) must be < hi ({hi}).")
        self.lo = float(lo)
        self.hi = float(hi)
        self.max_abs_eta = float(max_abs_eta)

    def _R(self):
        return self.hi - self.lo

    def _sigma(self, eta):
        eta_c = np.clip(eta, -self.max_abs_eta, self.max_abs_eta)
        return 1.0 / (1.0 + np.exp(-eta_c))

    def link_to_alpha(self, eta):
        return self.lo + self._R() * self._sigma(eta)

    def alpha_to_eta(self, alpha):
        if isinstance(alpha, np.ndarray):
            u = (alpha - self.lo) / self._R()
            if np.any((u <= 0) | (u >= 1)):
                raise ValueError(
                    f"BoundedSquaredLoss requires alpha in ({self.lo}, {self.hi})."
                )
            return np.log(u / (1.0 - u))
        u = (alpha - self.lo) / self._R()
        if not (0 < u < 1):
            raise ValueError(
                f"BoundedSquaredLoss requires alpha in ({self.lo}, {self.hi})."
            )
        return float(np.log(u / (1.0 - u)))

    def loss_row(self, a, b, alpha):
        return a * alpha**2 + b * alpha

    def gradient(self, a, b, eta):
        sigma = self._sigma(eta)
        alpha = self.lo + self._R() * sigma
        # d/dη loss = (2aα + b) · dα/dη ;  dα/dη = R σ (1−σ).
        return (2.0 * a * alpha + b) * self._R() * sigma * (1.0 - sigma)

    def hessian(self, a, b, eta, hessian_floor):
        sigma = self._sigma(eta)
        alpha = self.lo + self._R() * sigma
        R = self._R()
        d_alpha = R * sigma * (1.0 - sigma)
        d2_alpha = R * sigma * (1.0 - sigma) * (1.0 - 2.0 * sigma)
        # d²/dη² loss = d²L/dα² · (dα/dη)² + dL/dα · d²α/dη²
        h = 2.0 * a * d_alpha**2 + (2.0 * a * alpha + b) * d2_alpha
        return np.maximum(h, hessian_floor)

    def default_init_alpha(self):
        return 0.5 * (self.lo + self.hi)

    def validate_coefficients(self, b):
        return  # any signed b ok (it's still squared loss in α)

    def to_spec(self) -> dict:
        return {
            "type": "BoundedSquaredLoss",
            "args": {"lo": self.lo, "hi": self.hi, "max_abs_eta": self.max_abs_eta},
        }
