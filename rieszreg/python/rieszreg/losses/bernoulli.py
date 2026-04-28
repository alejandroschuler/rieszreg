"""Bernoulli Bregman-Riesz loss: binary entropy φ, sigmoid link."""

from __future__ import annotations

import numpy as np


class BernoulliLoss:
    """φ(t) = t·log(t) + (1−t)·log(1−t)  (binary entropy on (0, 1)).
    ψ(t) = -log(1 − t).  Sigmoid link: α = σ(η) ∈ (0, 1).

    Per-row loss in η:
        l(η) = a · softplus(η) + (b/2) · η      (since softplus(η) = -log(1-σ(η)))
    Gradient (η) = a · α + b/2,  Hessian (η) = a · α(1 − α).

    Use when α₀ is known to lie in (0, 1) by problem structure (e.g. trimmed
    propensity-score-style representers). If true α₀ exceeds 1, the sigmoid
    saturates and the fit plateaus near 1 — there is no warning, just a poor
    fit. Validate that your representer is bounded before reaching for this.
    """

    name = "bernoulli"

    def __init__(self, max_abs_eta: float = 30.0):
        # Clip η before σ to avoid 0/1 saturation that ruins the gradient.
        self.max_abs_eta = float(max_abs_eta)

    def _clip(self, eta):
        return np.clip(eta, -self.max_abs_eta, self.max_abs_eta)

    def link_to_alpha(self, eta):
        eta = self._clip(eta)
        return 1.0 / (1.0 + np.exp(-eta))

    def alpha_to_eta(self, alpha):
        if isinstance(alpha, np.ndarray):
            if np.any((alpha <= 0) | (alpha >= 1)):
                raise ValueError("BernoulliLoss requires alpha in (0, 1) for init.")
            return np.log(alpha / (1.0 - alpha))
        if not (0 < alpha < 1):
            raise ValueError("BernoulliLoss requires alpha in (0, 1) for init.")
        return float(np.log(alpha / (1.0 - alpha)))

    def loss_row(self, a, b, alpha):
        # In α-space the per-row loss is a · ψ(α) + (b/2) · φ'(α).
        # ψ(α) = -log(1−α);  φ'(α) = log(α/(1−α)) = logit(α).
        eps = np.exp(-self.max_abs_eta)
        a_clip = np.clip(alpha, eps, 1.0 - eps)
        return -a * np.log(1.0 - a_clip) + 0.5 * b * np.log(a_clip / (1.0 - a_clip))

    def gradient(self, a, b, eta):
        alpha = self.link_to_alpha(eta)
        return a * alpha + 0.5 * b

    def hessian(self, a, b, eta, hessian_floor):
        del b
        alpha = self.link_to_alpha(eta)
        return np.maximum(a * alpha * (1.0 - alpha), hessian_floor)

    def default_init_alpha(self):
        return 0.5

    def validate_coefficients(self, b):
        if np.any(b > 0):
            raise ValueError(
                "BernoulliLoss requires all m-coefficients to be non-negative "
                "(equivalently: all augmented `b` values <= 0). Same constraint "
                "as KLLoss — it's specific to density-ratio-style targets."
            )

    def to_spec(self) -> dict:
        return {"type": "BernoulliLoss", "args": {"max_abs_eta": self.max_abs_eta}}
