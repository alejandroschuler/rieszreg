# rieszboost

Gradient boosting for Riesz representers — directly estimate the Riesz representer α₀ of a linear functional θ(P) = E[m(Z, g₀)] without ever deriving or inverting a propensity-style ratio. Implements [Lee & Schuler, *RieszBoost* (arXiv:2501.04871)](https://arxiv.org/abs/2501.04871).

## Status

v0.0.1 — Python fast path (xgboost) is functional. R wrapper, slow general path, Bregman extension, and longitudinal estimands are planned.

## Why

In semiparametric estimation (ATE, treatment-specific means, shift interventions, longitudinal interventions) one-step / TMLE / DML estimators require α̂ — the Riesz representer of the target functional. The classical approach derives α₀'s analytical form (e.g. inverse propensity score for ATE), estimates its components, and substitutes them in. That breaks down under positivity violations and gets unwieldy for non-standard estimands. Riesz regression directly minimizes a loss whose minimum is α₀ regardless of analytical form. RieszBoost does this with gradient boosting — fast, tabular-data-friendly, and easy to tune.

## Install

```sh
git clone https://github.com/alejandroschuler/rieszboost.git
cd rieszboost
python3 -m venv .venv
.venv/bin/pip install -e python/
```

On macOS, `xgboost` requires `libomp`:

```sh
brew install libomp
```

## Quickstart — ATE

```python
import numpy as np
import rieszboost

# Synthetic binary-treatment data
rng = np.random.default_rng(0)
n = 4000
x = rng.uniform(0, 1, n)
pi = 1 / (1 + np.exp(-(-0.02*x - x**2 + 4*np.log(x + 0.3) + 1.5)))
a = rng.binomial(1, pi)
rows = [{"a": int(ai), "x": float(xi)} for ai, xi in zip(a, x)]

# 80/20 train/valid split for early stopping
n_tr = int(0.8 * n)
train_rows, valid_rows = rows[:n_tr], rows[n_tr:]

booster = rieszboost.fit(
    train_rows,
    rieszboost.ATE(),               # m(z, alpha) = alpha(1, x) - alpha(0, x)
    feature_keys=("a", "x"),
    valid_rows=valid_rows,
    num_boost_round=2000,
    early_stopping_rounds=20,       # halt when held-out Riesz loss stops improving
    learning_rate=0.05,
    max_depth=4,
)

alpha_hat = booster.predict(rows)
# alpha_hat ≈ A/π(X) - (1-A)/(1-π(X)) — without ever estimating π(X)

# Diagnostics — magnitude, tail extremes, near-positivity warnings
print(rieszboost.diagnose(booster=booster, rows=valid_rows, m=rieszboost.ATE()).summary())
```

## Cross-fitting for downstream inference

When you'll plug α̂ into a TMLE / one-step / DML estimator, use cross-fitting so the predictions you use are out-of-fold:

```python
result = rieszboost.crossfit(
    rows,
    rieszboost.ATE(),
    feature_keys=("a", "x"),
    n_folds=5,
    early_stopping_inner_split=0.2,  # carve a held-out slice inside each fold
    num_boost_round=2000,
    early_stopping_rounds=20,
    learning_rate=0.05,
    max_depth=3,
    reg_lambda=10.0,                 # keep extrapolation tame
)
alpha_hat_oof = result.alpha_hat   # shape (n,) — out-of-fold predictions
```

> **Note on hyperparameters.** Boosted Riesz representers can extrapolate aggressively at low-overlap boundaries. Shallower trees (`max_depth=3`) and a heavier ridge (`reg_lambda=10`) plus early stopping keep the tails under control. Always run `rieszboost.diagnose(...)` on the fit and inspect the warnings.

## Custom estimands

The natural API is to write `m(z, alpha)` opaquely. The library traces it to extract the linear-form structure:

```python
def m_att(z, alpha):
    """ATT representer: averages over the treated marginal."""
    p_treated = 0.4   # marginal P(A=1), estimated externally
    return (alpha(a=1, x=z["x"]) - alpha(a=0, x=z["x"])) * (z["a"] / p_treated)
```

`alpha(...)` calls record evaluation points; `+`, `-`, and scalar `*` compose them into a `LinearForm`. Anything outside that (e.g. `alpha(...) ** 2`, `alpha(...) + 1.0`) raises — by construction the fast path supports exactly the class of finite linear combinations of point evaluations of α.

## Built-in estimands

| Factory | m(z, α) | Notes |
|---|---|---|
| `rieszboost.ATE(treatment, covariates)` | α(1, x) − α(0, x) | Binary treatment ATE |
| `rieszboost.TSM(level, treatment, covariates)` | α(level, x) | Treatment-specific mean |
| `rieszboost.AdditiveShift(delta, treatment, covariates)` | α(a + δ, x) − α(a, x) | Continuous-treatment shift effect |

More planned: `ATT`, `Longitudinal` (LMTP-style), stochastic-shift variants.

## What works today

- Opaque `m(z, alpha)` API with linearity enforced by construction.
- Fast path: data augmentation + xgboost custom objective (gradient `2aF + b`, Hessian `2a`).
- ATE / TSM / AdditiveShift estimand factories.
- `init={0, float, "m1"}` initialization.
- Early stopping on held-out Riesz loss (`valid_rows=` + `early_stopping_rounds=`).
- K-fold cross-fitting (`rieszboost.crossfit(...)`) with optional inner-split early stopping.
- Diagnostics (`rieszboost.diagnose(...)`): RMS, extremes, |α| quantiles, near-positivity warnings, held-out Riesz loss.

## On the roadmap

- lightgbm engine adapter.
- Slow general path with sklearn / JAX base learners (for derivatives, integrals against known densities).
- Longitudinal/LMTP estimand factory and ATT factory.
- R wrapper via reticulate.
- Bregman extension (Hines & Miles / Kato 2026).

See `CLAUDE.md` and `~/.claude/plans/i-d-like-to-write-crystalline-raven.md` for the full plan.

## Related work

- [Chernozhukov et al., RieszNet & ForestRiesz (2110.03031)](https://arxiv.org/abs/2110.03031) — neural-net and random-forest Riesz regression.
- [Chernozhukov et al., Auto-DML via Riesz Regression (2104.14737)](https://arxiv.org/abs/2104.14737) — origin of the squared Riesz loss.
- [Singh, Kernel Ridge Riesz Representers (2102.11076)](https://arxiv.org/abs/2102.11076) — closed-form RKHS estimator.
- [Hines & Miles (2510.16127)](https://arxiv.org/abs/2510.16127) and [Kato (2601.07752)](https://arxiv.org/abs/2601.07752) — Bregman-divergence generalization.
- [van der Laan et al. (2501.11868)](https://arxiv.org/abs/2501.11868) — auto-DML for smooth functionals beyond linear.

## Tests

```sh
.venv/bin/python -m pytest python/tests -v
```

## License

TBD.
