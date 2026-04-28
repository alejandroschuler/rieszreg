"""Built-in estimand factories. Each returns an opaque m(z, alpha) callable
suitable for the fast engine."""

from __future__ import annotations

from typing import Callable, Sequence


def ATE(treatment: str = "a", covariates: Sequence[str] = ("x",)) -> Callable:
    """Average treatment effect: m(z, alpha) = alpha(a=1, x) - alpha(a=0, x)."""
    cov = tuple(covariates)

    def m(z, alpha):
        x_kwargs = {k: z[k] for k in cov}
        return alpha(**{treatment: 1, **x_kwargs}) - alpha(**{treatment: 0, **x_kwargs})

    return m


def TSM(level, treatment: str = "a", covariates: Sequence[str] = ("x",)) -> Callable:
    """Treatment-specific mean: m(z, alpha) = alpha(a=level, x)."""
    cov = tuple(covariates)

    def m(z, alpha):
        x_kwargs = {k: z[k] for k in cov}
        return alpha(**{treatment: level, **x_kwargs})

    return m


def AdditiveShift(
    delta: float, treatment: str = "a", covariates: Sequence[str] = ("x",)
) -> Callable:
    """Additive shift effect: m(z, alpha) = alpha(a + delta, x) - alpha(a, x)."""
    cov = tuple(covariates)

    def m(z, alpha):
        a = z[treatment]
        x_kwargs = {k: z[k] for k in cov}
        return alpha(**{treatment: a + delta, **x_kwargs}) - alpha(
            **{treatment: a, **x_kwargs}
        )

    return m


def LocalShift(
    delta: float,
    threshold: float,
    treatment: str = "a",
    covariates: Sequence[str] = ("x",),
) -> Callable:
    """Riesz representer of the local-additive-shift **partial parameter**
    `θ_partial = E[1(A < threshold) · (μ(A+δ, X) − μ(A, X))]`.

    `m(z, α) = 1(a < threshold) · (α(a+δ, x) − α(a, x))`.

    Like ATT, the *full* local-shift effect (LASE)
    `E[μ(A+δ,X) − μ(A,X) | A < threshold] = θ_partial / P(A < threshold)`
    is **not** a Riesz functional (it depends on the marginal P(A < threshold)).
    Use this factory to fit α̂_partial; for inference on LASE, build the
    delta-method EIF (Susmann 2024 / Lee-Schuler appendix):

        φ(O) = (1/p_t) [1(A<t)(μ(A+δ,X) − μ(A,X) − ψ_LASE) + α̂_partial(Y − μ(O))]

    with p̂_t = mean(A < threshold).
    """
    cov = tuple(covariates)

    def m(z, alpha):
        a = z[treatment]
        if a >= threshold:
            return 0
        x_kwargs = {k: z[k] for k in cov}
        return alpha(**{treatment: a + delta, **x_kwargs}) - alpha(
            **{treatment: a, **x_kwargs}
        )

    return m


def StochasticIntervention(
    samples_key: str = "shift_samples",
    treatment: str = "a",
    covariates: Sequence[str] = ("x",),
) -> Callable:
    """Stochastic intervention via pre-computed Monte Carlo samples.

    The functional is θ = E[∫ μ(a', X) g(a' | A, X) da'] for some intervention
    density g. We approximate the integral by Monte Carlo: each row `z` must
    contain a sequence `z[samples_key]` of treatment values drawn from
    g(· | a, x). The empirical m is

        m(z, alpha) = (1/K) Σ_k alpha(a' = z[samples_key][k], x)

    Pre-sample once before calling `fit(...)`, e.g.:

        rng = np.random.default_rng(0)
        for row in rows:
            row["shift_samples"] = rng.normal(
                row["a"] + delta, sigma, size=n_mc_samples
            )

    Increasing `n_mc_samples` reduces Monte Carlo noise; common choice is
    10–50. Note `feature_keys` should NOT include `samples_key` (it's not
    a tree feature, just a per-row payload).
    """
    cov = tuple(covariates)

    def m(z, alpha):
        x_kwargs = {k: z[k] for k in cov}
        samples = z[samples_key]
        K = len(samples)
        if K == 0:
            return 0
        return sum(
            alpha(**{treatment: float(s), **x_kwargs}) for s in samples
        ) / K

    return m


def ATT(treatment: str = "a", covariates: Sequence[str] = ("x",)) -> Callable:
    """Riesz representer of the **partial parameter** for ATT,
    θ_partial = E[A·(μ(1,X) − μ(0,X))]; m(z, α) = a · (α(1, x) − α(0, x)).
    For control rows (a=0) the contribution is zero — those rows contribute
    only the α² term in the loss.

    The full ATT, θ_ATT = θ_partial / P(A=1), is **not** itself a Riesz
    functional (it depends on the marginal P(A=1), not on μ). The standard
    pipeline (Hubbard 2011) is:

        1. Fit α̂_partial via this factory.
        2. Estimate p̂ = mean(A).
        3. Use the delta-method EIF
              φ(O) = (1/p̂)·[A·(μ̂(1,X) − μ̂(0,X) − ψ̂_ATT) + α̂_partial·(Y − μ̂)]
           and solve for ψ̂_ATT = (1/p̂)·mean(A(μ̂(1)−μ̂(0)) + α̂_partial(Y−μ̂)).

    See examples/lee_schuler/binary_dgp.py for a worked example.
    """
    cov = tuple(covariates)

    def m(z, alpha):
        a = z[treatment]
        x_kwargs = {k: z[k] for k in cov}
        return a * (
            alpha(**{treatment: 1, **x_kwargs}) - alpha(**{treatment: 0, **x_kwargs})
        )

    return m
