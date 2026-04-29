"""Quickstart: TSM (treatment-specific mean) on a synthetic DGP.

Run from the repo root: ``.venv/bin/python forestriesz/examples/tsm_quickstart.py``

Demonstrates the locally constant fit. TSM has a single evaluation point
m(z, α) = α(level, x), so the constant basis works without a sieve.
This is the only flavor the R wrapper exposes in v1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forestriesz import (
    ForestRieszRegressor,
    TSM,
    diagnose_forest,
)


def main() -> None:
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.uniform(0, 1, n)
    pi = 1 / (1 + np.exp(-(0.5 * x - 0.3)))
    a = rng.binomial(1, pi).astype(float)
    df = pd.DataFrame({"a": a, "x": x})

    fr = ForestRieszRegressor(
        estimand=TSM(level=1, treatment="a", covariates=("x",)),
        n_estimators=500,
        min_samples_leaf=10,
        max_samples=0.5,
        honest=True,
        inference=True,
        random_state=0,
    )
    fr.fit(df)
    alpha_hat = fr.predict(df)
    truth = (a == 1).astype(float) / pi

    print(f"alpha_hat range      : [{alpha_hat.min():.3f}, {alpha_hat.max():.3f}]")
    print(f"truth     range      : [{truth.min():.3f}, {truth.max():.3f}]")
    print(f"correlation w/ truth : {np.corrcoef(alpha_hat, truth)[0, 1]:.3f}")
    print(f"RMSE vs truth        : {float(np.sqrt(np.mean((alpha_hat - truth) ** 2))):.3f}")

    # Honest-split confidence intervals (locally constant only)
    lb, ub = fr.predict_interval(df, alpha=0.1)
    width = float(np.mean(ub - lb))
    print(f"mean 90% CI width    : {width:.3f}")
    print()
    print(diagnose_forest(fr, df).summary())


if __name__ == "__main__":
    main()
