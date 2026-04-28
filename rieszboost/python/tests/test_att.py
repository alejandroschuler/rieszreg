"""ATT partial-parameter estimand: m(z, alpha) = a * (alpha(1,x) - alpha(0,x)).

The full ATT theta_ATT = E[mu(1,X) - mu(0,X) | A=1] is not a Riesz functional
(it depends on the marginal P(A=1)). What we fit here is the Riesz representer
of the partial parameter theta_partial = E[A * (mu(1,X) - mu(0,X))].
True representer: alpha_partial(A, X) = A - (1-A) * pi(X) / (1-pi(X)).
"""

import numpy as np

from rieszboost.engine import build_augmented, fit
from rieszboost.estimands import ATT
from rieszboost.tracer import trace


def test_att_traces_to_zero_for_controls():
    m = ATT()
    # treated row: full ATE-style contrast scaled by a=1
    treated_pairs = trace(m, {"a": 1, "x": 0.5})
    coefs = sorted(c for c, _ in treated_pairs)
    assert coefs == [-1.0, 1.0]
    # control row (a=0): every term carries factor 0, contributes nothing
    assert trace(m, {"a": 0, "x": 0.5}) == []


def test_att_augmentation_skips_controls():
    rows = [{"a": 1, "x": 0.5}, {"a": 0, "x": 0.7}]
    aug = build_augmented(rows, ATT(), feature_keys=("a", "x"))
    # Treated row: 2 unique points (the (1,x) one merges with the original).
    # Control row: just the original alpha^2 row at (0, 0.7).
    assert aug.features.shape == (3, 2)
    # Control row's contributions are pure quadratic (a=1, b=0).
    ctrl_idx = np.where(aug.origin_index == 1)[0]
    assert aug.a[ctrl_idx].sum() == 1.0
    assert aug.b[ctrl_idx].sum() == 0.0


def _logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def test_att_partial_recovers_truth_on_lee_schuler_dgp():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(0, 1, n)
    pi = _logit(-0.02 * x - x**2 + 4 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    rows = [{"a": int(ai), "x": float(xi)} for ai, xi in zip(a, x)]

    n_tr = int(0.8 * n)
    booster = fit(
        rows[:n_tr],
        ATT(),
        feature_keys=("a", "x"),
        valid_rows=rows[n_tr:],
        num_boost_round=2000,
        early_stopping_rounds=20,
        learning_rate=0.05,
        max_depth=3,
        reg_lambda=10.0,
        seed=0,
    )
    alpha_hat = booster.predict(rows)
    # Partial-parameter representer: A - (1-A) pi(X) / (1 - pi(X))
    alpha_true = a - (1 - a) * pi / (1 - pi)
    rmse = float(np.sqrt(np.mean((alpha_hat - alpha_true) ** 2)))
    # Lee-Schuler Table 1 reports RieszBoost RMSE on ATT (partial) at n=500: 0.435.
    # We're at n=4000 with similar hyperparams; expect comparable or better.
    assert rmse < 0.5, f"RMSE {rmse:.3f} too high"
