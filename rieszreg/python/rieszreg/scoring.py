"""sklearn scorer factory for Bregman-Riesz losses.

`RieszEstimator.score()` uses the canonical squared yardstick for cross-
estimator comparability (analog of R² for regressors). When a different
yardstick is wanted — e.g. KL on a density-ratio problem — pass
`scoring=riesz_scorer(loss=KLLoss())` to any sklearn CV utility.
"""

from __future__ import annotations

import numpy as np

from .augmentation import aug_loss_alpha, build_augmented
from .estimator import _rows_from_X
from .losses import LossSpec, SquaredLoss


def riesz_scorer(loss: LossSpec | None = None):
    """Return an sklearn-compatible scorer (`(estimator, X, y=None) -> float`).

    Parameters
    ----------
    loss : LossSpec or None, default=None
        Yardstick loss to evaluate on the held-out fold. If `None`, defaults
        to `SquaredLoss()` (matches `RieszEstimator.score`).

    Notes
    -----
    The fitted estimator's own link maps backend output η to α; the yardstick
    `loss` is then evaluated on that α with the held-out augmented (a, b)
    coefficients. The yardstick must accept the estimator's α: `SquaredLoss`
    has unrestricted α-domain, while `KLLoss` requires α > 0,
    `BernoulliLoss` requires α ∈ (0, 1), and `BoundedSquaredLoss(lo, hi)`
    requires α ∈ (lo, hi).
    """
    yardstick = loss if loss is not None else SquaredLoss()

    def _scorer(estimator, X, y=None) -> float:
        if not hasattr(estimator, "predictor_"):
            raise RuntimeError(
                f"{type(estimator).__name__} is not fitted yet."
            )
        rows = _rows_from_X(X, estimator.estimand)
        aug = build_augmented(rows, estimator.estimand)
        eta = estimator.predictor_.predict_eta(aug.features)
        alpha_hat = estimator.loss_.link_to_alpha(eta)
        return -float(
            np.sum(aug_loss_alpha(yardstick, aug.is_original, aug.potential_deriv_coef, alpha_hat))
            / aug.n_rows
        )

    return _scorer
