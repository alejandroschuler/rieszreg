"""Canonical DGPs for estimator-consistency tests.

Each DGP exposes:
  - `sample(n, rng)` -> pandas DataFrame
  - `true_alpha(z)` -> closed-form α₀ at a row z
  - `feature_keys` matching the corresponding rieszreg estimand factory

Implementation packages run their backend against these DGPs at growing n and
assert the learned α̂ approaches the true α₀.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class DGP:
    name: str
    feature_keys: tuple[str, ...]
    sample: Callable[..., object]            # (n, rng) -> DataFrame
    true_alpha: Callable[..., np.ndarray]    # (df) -> array of α₀ per row
    estimand_factory: str                    # "ATE", "ATT", ...


def linear_gaussian_ate(
    *,
    p_treated: float = 0.5,
    treatment_effect: float = 1.0,
    sigma_x: float = 1.0,
) -> DGP:
    """Linear-Gaussian ATE DGP.

    A ~ Bernoulli(π(x)) with logit π(x) = β · x; X ~ N(0, σ²).
    Closed-form Riesz representer for ATE: α₀(a, x) = (2a − 1) / [a · π(x) + (1−a)·(1−π(x))].
    """
    feature_keys = ("a", "x")

    def sample(n: int, rng: np.random.Generator):
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("rieszreg.testing.dgps requires pandas") from e
        x = rng.normal(0.0, sigma_x, size=n)
        # Use a fixed propensity model logit(π) = 0.5 x for nontrivial overlap.
        logit = 0.5 * x
        pi = 1.0 / (1.0 + np.exp(-logit))
        a = (rng.uniform(0, 1, size=n) < pi).astype(float)
        # Outcome (not used by Riesz fitter; included for downstream EIF tests).
        y = treatment_effect * a + 0.5 * x + rng.normal(0.0, 1.0, size=n)
        return pd.DataFrame({"a": a, "x": x, "y": y})

    def true_alpha(df) -> np.ndarray:
        x = np.asarray(df["x"])
        a = np.asarray(df["a"])
        pi = 1.0 / (1.0 + np.exp(-0.5 * x))
        prob_a = a * pi + (1.0 - a) * (1.0 - pi)
        sign = 2.0 * a - 1.0
        return sign / prob_a

    return DGP(
        name="linear_gaussian_ate",
        feature_keys=feature_keys,
        sample=sample,
        true_alpha=true_alpha,
        estimand_factory="ATE",
    )


def logistic_tsm(level: float = 1.0, sigma_x: float = 1.0) -> DGP:
    """Treatment-specific mean DGP. Riesz representer α₀(a, x) = 1[a=level] / π(x|level).

    Useful as a density-ratio target: α₀ ≥ 0 everywhere, so KLLoss applies.
    """
    feature_keys = ("a", "x")

    def sample(n: int, rng: np.random.Generator):
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("rieszreg.testing.dgps requires pandas") from e
        x = rng.normal(0.0, sigma_x, size=n)
        logit = 0.5 * x
        pi = 1.0 / (1.0 + np.exp(-logit))
        a = (rng.uniform(0, 1, size=n) < pi).astype(float)
        y = (a == level).astype(float) + 0.5 * x + rng.normal(0.0, 1.0, size=n)
        return pd.DataFrame({"a": a, "x": x, "y": y})

    def true_alpha(df) -> np.ndarray:
        x = np.asarray(df["x"])
        a = np.asarray(df["a"])
        pi = 1.0 / (1.0 + np.exp(-0.5 * x))
        prob_a = a * pi + (1.0 - a) * (1.0 - pi)
        return (a == level).astype(float) / prob_a

    return DGP(
        name="logistic_tsm",
        feature_keys=feature_keys,
        sample=sample,
        true_alpha=true_alpha,
        estimand_factory="TSM",
    )


def assert_consistency(
    fit_predict,
    *,
    dgp: DGP,
    n_grid: tuple[int, ...] = (500, 2000),
    rng_seed: int = 0,
    tol_at_max_n: float = 0.5,
):
    """Assert RMSE of α̂ vs α₀ shrinks across n and lands below `tol_at_max_n`.

    Parameters
    ----------
    fit_predict : callable
        Takes (train_df, test_df) → (alpha_hat_array, ...). Must return alpha_hat
        as the first element.
    """
    rng = np.random.default_rng(rng_seed)
    rmses = []
    for n in n_grid:
        train = dgp.sample(n, rng)
        test = dgp.sample(2 * n, rng)
        alpha_hat = np.asarray(fit_predict(train, test))[: len(test)]
        alpha_true = dgp.true_alpha(test)
        rmses.append(float(np.sqrt(np.mean((alpha_hat - alpha_true) ** 2))))
    # Monotone (non-strict) decrease + final-n tolerance.
    if not (rmses[-1] <= rmses[0] + 1e-6):
        raise AssertionError(
            f"{dgp.name}: RMSE did not decrease across n_grid={n_grid}; got {rmses}"
        )
    if rmses[-1] > tol_at_max_n:
        raise AssertionError(
            f"{dgp.name}: RMSE at n={n_grid[-1]} is {rmses[-1]:.3f}, "
            f"above tolerance {tol_at_max_n}"
        )
    return rmses
