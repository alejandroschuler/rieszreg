"""K-fold cross-fitting: fit α̂_(-k) on folds excluding k, predict on fold k.

Returns out-of-fold predictions for every row, suitable for plug-in to TMLE /
one-step / DML downstream estimators. Optionally uses an inner held-out split
for early stopping inside each fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .engine import RieszBooster, fit


@dataclass
class CrossFitResult:
    alpha_hat: np.ndarray  # shape (n,) — out-of-fold predictions
    fold_assignment: np.ndarray  # shape (n,) — which fold each row was held out in
    boosters: list[RieszBooster]


def crossfit(
    rows: Sequence[dict[str, Any]],
    m: Callable,
    feature_keys: Sequence[str],
    *,
    n_folds: int = 5,
    seed: int = 0,
    early_stopping_inner_split: float | None = None,
    **fit_kwargs: Any,
) -> CrossFitResult:
    """Fit a Riesz representer with K-fold cross-fitting.

    `early_stopping_inner_split`, if set (e.g. 0.2), carves a held-out slice
    out of each training fold to drive early stopping. `fit_kwargs` are
    forwarded to `fit` (e.g. learning_rate, max_depth, num_boost_round,
    early_stopping_rounds, init).
    """
    n = len(rows)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, size=n)

    alpha_hat = np.empty(n)
    boosters: list[RieszBooster] = []

    for k in range(n_folds):
        train_mask = fold != k
        test_mask = fold == k
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)

        train_rows = [rows[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]

        if early_stopping_inner_split is not None:
            inner_rng = np.random.default_rng(seed + k + 1)
            shuffled = inner_rng.permutation(len(train_rows))
            n_val = max(1, int(round(early_stopping_inner_split * len(train_rows))))
            val_idx = shuffled[:n_val]
            tr_idx = shuffled[n_val:]
            inner_train = [train_rows[i] for i in tr_idx]
            inner_valid = [train_rows[i] for i in val_idx]
            booster = fit(
                inner_train,
                m,
                feature_keys,
                valid_rows=inner_valid,
                **fit_kwargs,
            )
        else:
            booster = fit(train_rows, m, feature_keys, **fit_kwargs)

        alpha_hat[test_idx] = booster.predict(test_rows)
        boosters.append(booster)

    return CrossFitResult(
        alpha_hat=alpha_hat,
        fold_assignment=fold,
        boosters=boosters,
    )
