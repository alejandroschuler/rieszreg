"""Estimand: a self-contained description of the linear functional to fit.

An `Estimand` carries (1) the column names alpha is indexed by (`feature_keys`),
(2) per-row payload columns that aren't tree features but are referenced by m
(`extra_keys`, e.g. "shift_samples" for stochastic interventions), and (3) the
opaque m(z, alpha) callable itself.

The orchestrator estimator reads `feature_keys` and `extra_keys` off the
estimand at fit time — no need for the user to pass these as separate arguments.

This module also holds the six built-in factories (`ATE`, `ATT`, `TSM`,
`AdditiveShift`, `LocalShift`, `StochasticIntervention`) and `estimand_from_spec`
for round-trip serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass
class Estimand:
    feature_keys: tuple[str, ...]
    m: Callable[..., Any]
    extra_keys: tuple[str, ...] = ()
    name: str = "custom"
    # If set, identifies a built-in factory + ctor args so the estimand can be
    # reconstructed from JSON or pickle. Custom user-supplied m()s leave this
    # None and rely on the user's m being picklable on its own.
    factory_spec: dict | None = None

    def __call__(self, z, alpha):
        return self.m(z, alpha)

    def __reduce__(self):
        """Round-trip via the factory_spec for built-in estimands.

        Stock pickle / joblib can't serialize the closure `m` returned by a
        factory function, so we redirect to `estimand_from_spec(...)` on
        unpickle. Custom estimands without a factory_spec fall back to the
        default dataclass reduce — that requires the user's `m` to be
        importable / picklable.
        """
        if self.factory_spec is not None:
            return (estimand_from_spec, (self.factory_spec,))
        return (
            _rebuild_custom_estimand,
            (self.feature_keys, self.m, self.extra_keys, self.name),
        )


def _rebuild_custom_estimand(feature_keys, m, extra_keys, name):
    return Estimand(
        feature_keys=feature_keys,
        m=m,
        extra_keys=extra_keys,
        name=name,
        factory_spec=None,
    )


def ATE(treatment: str = "a", covariates: Sequence[str] = ("x",)) -> Estimand:
    """Average treatment effect: m(z, α) = α(1, x) − α(0, x)."""
    cov = tuple(covariates)

    def m(z, alpha):
        x_kwargs = {k: z[k] for k in cov}
        return alpha(**{treatment: 1, **x_kwargs}) - alpha(**{treatment: 0, **x_kwargs})

    return Estimand(
        feature_keys=(treatment, *cov), m=m, name="ATE",
        factory_spec={"factory": "ATE", "args": {"treatment": treatment, "covariates": list(cov)}},
    )


def ATT(treatment: str = "a", covariates: Sequence[str] = ("x",)) -> Estimand:
    """ATT *partial parameter* m(z, α) = a · (α(1, x) − α(0, x)).

    Full ATT divides by P(A=1) and is not a Riesz functional — combine
    α̂_partial with a delta-method EIF (Hubbard 2011) downstream.
    """
    cov = tuple(covariates)

    def m(z, alpha):
        a = z[treatment]
        x_kwargs = {k: z[k] for k in cov}
        return a * (
            alpha(**{treatment: 1, **x_kwargs}) - alpha(**{treatment: 0, **x_kwargs})
        )

    return Estimand(
        feature_keys=(treatment, *cov), m=m, name="ATT",
        factory_spec={"factory": "ATT", "args": {"treatment": treatment, "covariates": list(cov)}},
    )


def TSM(level, treatment: str = "a", covariates: Sequence[str] = ("x",)) -> Estimand:
    """Treatment-specific mean: m(z, α) = α(level, x)."""
    cov = tuple(covariates)

    def m(z, alpha):
        x_kwargs = {k: z[k] for k in cov}
        return alpha(**{treatment: level, **x_kwargs})

    return Estimand(
        feature_keys=(treatment, *cov), m=m, name=f"TSM(level={level!r})",
        factory_spec={"factory": "TSM", "args": {"level": level, "treatment": treatment, "covariates": list(cov)}},
    )


def AdditiveShift(
    delta: float, treatment: str = "a", covariates: Sequence[str] = ("x",)
) -> Estimand:
    """Additive shift effect: m(z, α) = α(a + δ, x) − α(a, x)."""
    cov = tuple(covariates)

    def m(z, alpha):
        a = z[treatment]
        x_kwargs = {k: z[k] for k in cov}
        return alpha(**{treatment: a + delta, **x_kwargs}) - alpha(
            **{treatment: a, **x_kwargs}
        )

    return Estimand(
        feature_keys=(treatment, *cov), m=m, name=f"AdditiveShift(delta={delta})",
        factory_spec={"factory": "AdditiveShift", "args": {"delta": delta, "treatment": treatment, "covariates": list(cov)}},
    )


def LocalShift(
    delta: float,
    threshold: float,
    treatment: str = "a",
    covariates: Sequence[str] = ("x",),
) -> Estimand:
    """LASE *partial parameter* m(z, α) = 1(a < threshold) · (α(a+δ, x) − α(a, x)).

    Full LASE divides by P(A < threshold) and is not a Riesz functional.
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

    return Estimand(
        feature_keys=(treatment, *cov),
        m=m,
        name=f"LocalShift(delta={delta}, threshold={threshold})",
        factory_spec={"factory": "LocalShift", "args": {"delta": delta, "threshold": threshold, "treatment": treatment, "covariates": list(cov)}},
    )


def StochasticIntervention(
    samples_key: str = "shift_samples",
    treatment: str = "a",
    covariates: Sequence[str] = ("x",),
) -> Estimand:
    """Stochastic intervention via Monte Carlo samples per row.

    Each row carries `z[samples_key]` = sequence of treatment values drawn
    from the intervention density. `m(z, α) = (1/K) Σ_k α(a' = sample_k, x)`.

    Pre-sample once before fit:

        rng = np.random.default_rng(0)
        df["shift_samples"] = [rng.normal(a + delta, sigma, K) for a in df["a"]]
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

    return Estimand(
        feature_keys=(treatment, *cov),
        m=m,
        extra_keys=(samples_key,),
        name=f"StochasticIntervention(samples_key={samples_key!r})",
        factory_spec={"factory": "StochasticIntervention", "args": {"samples_key": samples_key, "treatment": treatment, "covariates": list(cov)}},
    )


# Registry for round-tripping. Updated when new built-in factories are added.
_FACTORY_REGISTRY = {
    "ATE": ATE,
    "ATT": ATT,
    "TSM": TSM,
    "AdditiveShift": AdditiveShift,
    "LocalShift": LocalShift,
    "StochasticIntervention": StochasticIntervention,
}


def estimand_from_spec(spec: dict) -> Estimand:
    """Reconstruct an Estimand from its `factory_spec` dict. Only built-in
    factories round-trip; custom estimands must be re-passed at load time."""
    factory_name = spec["factory"]
    if factory_name not in _FACTORY_REGISTRY:
        raise ValueError(
            f"Unknown estimand factory {factory_name!r}; only built-ins "
            f"({sorted(_FACTORY_REGISTRY)}) are round-trippable. For custom "
            f"estimands, pass `estimand=...` explicitly to .load(...)."
        )
    return _FACTORY_REGISTRY[factory_name](**spec.get("args", {}))
