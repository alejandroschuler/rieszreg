"""Trace `m` on each row and build the augmented dataset for the
Bregman-Riesz loss. Each augmented row r contributes a per-row loss term

    D_r · h̃(α(z_r)) + C_r · h'(α(z_r))

(the squared-loss case h̃ = t², h' = 2t simplifies to `D·α² + 2·C·α`).
The original observation Z_i seeds row i with (is_original=1,
potential_deriv_coef=0); each (coef, point) pair from m(z_i) contributes
(is_original=0, potential_deriv_coef=-coef) at the point. Duplicate
points within a row are merged by summing the two coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .estimands.base import FiniteEvalEstimand
from .estimands.tracer import trace


@dataclass
class AugmentedDataset:
    features: np.ndarray              # (n_aug, n_features)
    is_original: np.ndarray           # (n_aug,) — 1 if z_r == Z_{i_r}, else 0 (D_r)
    potential_deriv_coef: np.ndarray  # (n_aug,) — coefficient on h'(α) (C_r)
    origin_index: np.ndarray          # (n_aug,) — index into original rows
    n_rows: int                       # number of original rows


def build_augmented(
    rows: Sequence[dict[str, Any]],
    estimand: FiniteEvalEstimand,
) -> AugmentedDataset:
    if not isinstance(estimand, FiniteEvalEstimand):
        raise TypeError(
            f"build_augmented() requires a FiniteEvalEstimand; got "
            f"{type(estimand).__name__}."
        )
    feature_keys = estimand.feature_keys

    feats: list[np.ndarray] = []
    is_orig_list: list[float] = []
    pdc_list: list[float] = []
    origin: list[int] = []

    for i, z in enumerate(rows):
        acc: dict[tuple, tuple[float, float]] = {}
        z_key = tuple(z[k] for k in feature_keys)
        acc[z_key] = (1.0, 0.0)

        for coef, point in trace(estimand, z):
            missing = [k for k in feature_keys if k not in point]
            if missing:
                raise ValueError(
                    f"m evaluated alpha at a point missing keys {missing}; "
                    f"all feature_keys {list(feature_keys)} must be specified."
                )
            key = tuple(point[k] for k in feature_keys)
            cur_d, cur_c = acc.get(key, (0.0, 0.0))
            acc[key] = (cur_d, cur_c - coef)

        for key, (d, c) in acc.items():
            feats.append(np.asarray(key, dtype=float))
            is_orig_list.append(d)
            pdc_list.append(c)
            origin.append(i)

    return AugmentedDataset(
        features=np.vstack(feats) if feats else np.zeros((0, len(feature_keys))),
        is_original=np.asarray(is_orig_list, dtype=float),
        potential_deriv_coef=np.asarray(pdc_list, dtype=float),
        origin_index=np.asarray(origin, dtype=np.int64),
        n_rows=len(rows),
    )


# ---------------------------------------------------------------------------
# Augmented-loss helpers. The (is_original, potential_deriv_coef) pair lives
# with the augmentation engine; these helpers combine them with the loss's
# α-space functions and η-space link to give per-row loss / gradient / hessian
# for backends that fit in η-space.

def aug_loss_alpha(loss, is_original, potential_deriv_coef, alpha):
    """Per-row augmented loss in α-space: D · h_tilde(α) + C · h'(α)."""
    return (
        is_original * loss.tilde_potential(alpha)
        + potential_deriv_coef * loss.potential_deriv(alpha)
    )


def aug_loss_eta(loss, is_original, potential_deriv_coef, eta):
    """Per-row augmented loss in η-space."""
    return aug_loss_alpha(loss, is_original, potential_deriv_coef, loss.link_to_alpha(eta))


def aug_grad_eta(loss, is_original, potential_deriv_coef, eta):
    """∂[D·h_tilde(α) + C·h'(α)]/∂η. Routes to the loss's analytic helper."""
    return loss.aug_grad_eta(is_original, potential_deriv_coef, eta)


def aug_hess_eta(loss, is_original, potential_deriv_coef, eta, hessian_floor):
    """∂²[D·h_tilde(α) + C·h'(α)]/∂η² (floored). Routes to the loss's analytic helper."""
    return loss.aug_hess_eta(is_original, potential_deriv_coef, eta, hessian_floor)
