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

    def to_spec(self) -> dict:
        """Return a JSON-serializable {"type": str, "args": dict} round-trip spec."""
        ...


def loss_from_spec(spec: dict) -> "LossSpec":
    """Reconstruct a LossSpec from its `to_spec()` dict."""
    cls_name = spec["type"]
    args = spec.get("args", {})
    if cls_name == "SquaredLoss":
        return SquaredLoss(**args)
    if cls_name == "KLLoss":
        return KLLoss(**args)
    if cls_name == "BernoulliLoss":
        return BernoulliLoss(**args)
    if cls_name == "BoundedSquaredLoss":
        return BoundedSquaredLoss(**args)
    raise ValueError(f"Unknown loss spec type: {cls_name!r}")


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

    def to_spec(self) -> dict:
        return {"type": "KLLoss", "args": {"max_eta": self.max_eta}}


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
