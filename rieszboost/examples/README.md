# Examples

End-to-end scripts demonstrating `rieszboost` on the simulation studies of [Lee & Schuler (2025), arXiv:2501.04871](https://arxiv.org/abs/2501.04871).

## What's here

```
lee_schuler/
  binary_dgp.py            # Section 4.1: ATE + ATT, binary treatment
  continuous_dgp.py        # Section 4.2: ASE + LASE, continuous treatment
  _compare_with_reference.py   # Head-to-head against Lee-Schuler's reference code
  COMPARISON.md            # Math walkthrough + numbers from the cross-check
```

Run any of them with `.venv/bin/python examples/lee_schuler/<script>.py --n_reps 50` (paper uses `--n_reps 500`; defaults are smaller for short wall time).

## Estimands and Riesz functionals

Two of the four estimands in Lee-Schuler are not themselves Riesz functionals — they involve the marginal `P(A=...)`, which isn't a regression nuisance. The standard pipeline (Hubbard 2011 for ATT; Susmann 2024 for LASE) fits the Riesz representer of a *partial parameter* and applies a delta-method correction downstream.

| Parameter | Riesz functional? | Riesz representer fit | Delta-method downstream |
|---|---|---|---|
| **ATE** = E[μ(1,X) − μ(0,X)] | yes — `m(O,μ) = μ(1,X) − μ(0,X)` | `α₀ = A/π − (1−A)/(1−π)` | none |
| **ATT** = E[μ(1,X) − μ(0,X) \| A=1] | **no** — equals `θ_partial / P(A=1)` | partial: `m(O,μ) = A·(μ(1,X) − μ(0,X))`. `α_partial = A − (1−A)π/(1−π)` | `ψ_ATT = ψ_partial / p̂` with EIF correction |
| **ASE** = E[μ(A+δ,X) − μ(A,X)] | yes — `m(O,μ) = μ(A+δ,X) − μ(A,X)` | `α₀ = p(A−δ\|X)/p(A\|X) − 1` | none |
| **LASE** = E[μ(A+δ,X) − μ(A,X) \| A < t] | **no** — equals `θ_partial / P(A<t)` | partial: `m(O,μ) = 1(A<t)·(μ(A+δ,X) − μ(A,X))`. `α_partial = 1(A<t+δ)·p(A−δ\|X)/p(A\|X) − 1(A<t)` | `ψ_LASE = ψ_partial / p̂_t` with EIF correction |

`rieszboost`'s built-in `ATT()` and `LocalShift(delta, threshold)` factories return the *partial-parameter* m for ATT and LASE respectively. The example scripts then build the EIF and EEE estimator inline.

## Reproducing the paper

The scripts use a single fixed hyperparameter setting plus held-out early stopping; the paper CV-tunes over a grid (`learning_rate ∈ {0.001, 0.01, 0.1, 0.25}`, `max_depth ∈ {3, 5, 7}`, `n_estimators ∈ {10..200}`). Without CV, our final-parameter EEE estimates land within roughly 1 SE of the paper for ATE/ATT/ASE; α-RMSE numbers are somewhat worse (factor of 1–2). Wrap a CV loop around `rieszboost.fit(...)` to close the remaining gap.

| | Paper α-RMSE | Ours α-RMSE | Paper final-param RMSE | Ours final-param RMSE |
|---|---|---|---|---|
| ATE  | 0.92 | ~1.15 | 0.187 (94% cov) | ~0.20 (90% cov) |
| ATT  | 0.44 | ~0.77 | 0.177 (95% cov) | ~0.19 (98% cov) |
| ASE  | 0.37 | ~0.46 | 2.80 (93% cov) | ~3.85 (90% cov) |
| LASE | 0.25 | ~0.37 | 1.86 (95% cov) | ~4.4 (32% cov) |

LASE is the worst case — its representer has step discontinuities at `a = t` and `a = t + δ`, and tree boosting smooths them into ramps. CV-tuned hyperparameters help substantially; `max_depth ≥ 5` plus more boosting rounds is roughly the right move.

## Cross-check vs the reference implementation

`_compare_with_reference.py` runs both `rieszboost` and Kaitlyn Lee's reference fitter ([`kaitlynjlee/boosting_for_rr`](https://github.com/kaitlynjlee/boosting_for_rr)) on identical data and reports per-row α̂ disagreement. With `gradient_only=True` and `learning_rate=lr_ref/2`, our engine reproduces theirs to:

| | Pearson(ref, ours) | RMSE(ref vs ours) | RMSE vs truth (either) |
|---|---|---|---|
| ATE | 0.998 | 0.13 | ~1.0 |
| ATT | 0.986 | 0.19 | ~0.75 |

Disagreement is an order of magnitude smaller than disagreement of either implementation with the truth, so what's left is split-finding differences between sklearn `DecisionTreeRegressor` (exhaustive scan) and xgboost (histogram). The augmentation, gradient, and loss are mathematically equivalent. See [COMPARISON.md](lee_schuler/COMPARISON.md) for the math walkthrough behind the `lr_ref/2` rescaling and a separate bug we caught in the reference code.

```sh
git clone https://github.com/kaitlynjlee/boosting_for_rr /tmp/lee_ref
PYTHONPATH=/tmp/lee_ref .venv/bin/python examples/lee_schuler/_compare_with_reference.py \
    --n 500 --n_seeds 10 --lr 0.1 --n_estimators 100 --max_depth 3
```
