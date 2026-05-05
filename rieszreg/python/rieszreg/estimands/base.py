"""Estimand: a self-contained description of the linear functional to fit.

`Estimand` is the abstract base. Concrete usage goes through `FiniteEvalEstimand`,
the subclass for estimands whose `m` reduces to a finite linear combination of
point evaluations of `alpha` (ATE, ATT, TSM, additive shifts, ...). Every
built-in factory returns a `FiniteEvalEstimand`; the tracer, augmentation
engine, and orchestrator only accept `FiniteEvalEstimand`.

The class carries (1) the column names alpha is indexed by (`feature_keys`),
and (2) the opaque `m(alpha)(z, y)` operator itself.

`m` is an operator: it takes a candidate function `alpha` and returns a function
of the row `z` and the per-row outcome `y`. The orchestrator calls
`m(alpha)(z, y)` row-by-row, passing a `Tracer` for `alpha` to extract the
linear-form structure. `Y` flows in sklearn-style: separate from `Z` at every
layer (no outcome column inside the row dict). When the user's `m` doesn't read
`y` (the case for every built-in), the inner closure ignores its second arg.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


class Estimand:
    """Abstract base class for estimands.

    Do not construct directly — use `FiniteEvalEstimand` for the finite-evaluation
    case (every estimand currently supported by `rieszreg`). Future subclasses
    may handle estimands outside the finite-evaluation algebra (integrals,
    derivatives without a finite-difference reduction, etc.).
    """

    pass


@dataclass(eq=False)
class FiniteEvalEstimand(Estimand):
    """Estimand whose `m(alpha)(z, y)` is a finite linear combination of point
    evaluations of `alpha`. The tracer extracts the (coefficient, point) pairs;
    the augmentation engine uses them to build the augmented dataset.
    """

    feature_keys: tuple[str, ...]
    m: Callable[..., Any]
    name: str = "custom"
    # If set, identifies a built-in factory + ctor args so the estimand can be
    # reconstructed from JSON or pickle. Custom user-supplied m()s leave this
    # None and rely on the user's m being picklable on its own.
    factory_spec: dict | None = None

    def __call__(self, alpha):
        return self.m(alpha)

    def __eq__(self, other) -> bool:
        if not isinstance(other, FiniteEvalEstimand):
            return NotImplemented
        # Built-in estimands compare by factory_spec — two `ATE()` calls
        # produce different `m` closures but represent the same functional.
        if self.factory_spec is not None or other.factory_spec is not None:
            return self.factory_spec == other.factory_spec
        # Custom estimands fall back to identity-on-`m` plus structural fields.
        return (
            self.feature_keys == other.feature_keys
            and self.name == other.name
            and self.m is other.m
        )

    def __hash__(self) -> int:
        if self.factory_spec is not None:
            # factory_spec is JSON-serializable by design; use that as the key.
            import json
            return hash(json.dumps(self.factory_spec, sort_keys=True, default=str))
        return hash((self.feature_keys, self.name, id(self.m)))

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
            (self.feature_keys, self.m, self.name),
        )


def _rebuild_custom_estimand(feature_keys, m, name):
    return FiniteEvalEstimand(
        feature_keys=feature_keys,
        m=m,
        name=name,
        factory_spec=None,
    )


def ATE(treatment: str = "a", covariates: Sequence[str] = ("x",)) -> FiniteEvalEstimand:
    """Average treatment effect: m(α)(z, y) = α(1, x) − α(0, x)."""
    cov = tuple(covariates)

    def m(alpha):
        def inner(z, y=None):
            x_kwargs = {k: z[k] for k in cov}
            return alpha(**{treatment: 1, **x_kwargs}) - alpha(**{treatment: 0, **x_kwargs})
        return inner

    return FiniteEvalEstimand(
        feature_keys=(treatment, *cov), m=m, name="ATE",
        factory_spec={"factory": "ATE", "args": {"treatment": treatment, "covariates": list(cov)}},
    )


def ATT(treatment: str = "a", covariates: Sequence[str] = ("x",)) -> FiniteEvalEstimand:
    """ATT *partial-estimand* surface: m(α)(z, y) = a · (α(1, x) − α(0, x)).

    Full ATT divides by P(A=1) and is not a Riesz functional — combine
    α̂_partial with a delta-method EIF (Hubbard 2011) downstream.
    """
    cov = tuple(covariates)

    def m(alpha):
        def inner(z, y=None):
            a = z[treatment]
            x_kwargs = {k: z[k] for k in cov}
            return a * (
                alpha(**{treatment: 1, **x_kwargs}) - alpha(**{treatment: 0, **x_kwargs})
            )
        return inner

    return FiniteEvalEstimand(
        feature_keys=(treatment, *cov), m=m, name="ATT",
        factory_spec={"factory": "ATT", "args": {"treatment": treatment, "covariates": list(cov)}},
    )


def TSM(level, treatment: str = "a", covariates: Sequence[str] = ("x",)) -> FiniteEvalEstimand:
    """Treatment-specific mean: m(α)(z, y) = α(level, x)."""
    cov = tuple(covariates)

    def m(alpha):
        def inner(z, y=None):
            x_kwargs = {k: z[k] for k in cov}
            return alpha(**{treatment: level, **x_kwargs})
        return inner

    return FiniteEvalEstimand(
        feature_keys=(treatment, *cov), m=m, name=f"TSM(level={level!r})",
        factory_spec={"factory": "TSM", "args": {"level": level, "treatment": treatment, "covariates": list(cov)}},
    )


def AdditiveShift(
    delta: float, treatment: str = "a", covariates: Sequence[str] = ("x",)
) -> FiniteEvalEstimand:
    """Additive shift effect: m(α)(z, y) = α(a + δ, x) − α(a, x)."""
    cov = tuple(covariates)

    def m(alpha):
        def inner(z, y=None):
            a = z[treatment]
            x_kwargs = {k: z[k] for k in cov}
            return alpha(**{treatment: a + delta, **x_kwargs}) - alpha(
                **{treatment: a, **x_kwargs}
            )
        return inner

    return FiniteEvalEstimand(
        feature_keys=(treatment, *cov), m=m, name=f"AdditiveShift(delta={delta})",
        factory_spec={"factory": "AdditiveShift", "args": {"delta": delta, "treatment": treatment, "covariates": list(cov)}},
    )


def LocalShift(
    delta: float,
    threshold: float,
    treatment: str = "a",
    covariates: Sequence[str] = ("x",),
) -> FiniteEvalEstimand:
    """LASE *partial-estimand* surface: m(α)(z, y) = 1(a < threshold) · (α(a+δ, x) − α(a, x)).

    Full LASE divides by P(A < threshold) and is not a Riesz functional.
    """
    cov = tuple(covariates)

    def m(alpha):
        def inner(z, y=None):
            a = z[treatment]
            if a >= threshold:
                return 0
            x_kwargs = {k: z[k] for k in cov}
            return alpha(**{treatment: a + delta, **x_kwargs}) - alpha(
                **{treatment: a, **x_kwargs}
            )
        return inner

    return FiniteEvalEstimand(
        feature_keys=(treatment, *cov),
        m=m,
        name=f"LocalShift(delta={delta}, threshold={threshold})",
        factory_spec={"factory": "LocalShift", "args": {"delta": delta, "threshold": threshold, "treatment": treatment, "covariates": list(cov)}},
    )


def StochasticIntervention(
    samples_key: str = "shift_samples",
    treatment: str = "a",
    covariates: Sequence[str] = ("x",),
) -> FiniteEvalEstimand:
    """Stochastic intervention via Monte Carlo samples per row.

    Currently being rewritten — the previous implementation relied on an
    `extra_keys` payload mechanism that has been removed. A reintroduction
    will land in a follow-up that establishes how per-row samples flow into
    `m(alpha)(z, y)` without the payload-column shortcut.
    """
    raise NotImplementedError(
        "StochasticIntervention is being rewritten; will be re-added in a future PR."
    )


# Registry for round-tripping. Updated when new built-in factories are added.
_FACTORY_REGISTRY = {
    "ATE": ATE,
    "ATT": ATT,
    "TSM": TSM,
    "AdditiveShift": AdditiveShift,
    "LocalShift": LocalShift,
}


def estimand_from_spec(spec: dict) -> FiniteEvalEstimand:
    """Reconstruct a FiniteEvalEstimand from its `factory_spec` dict. Only
    built-in factories round-trip; custom estimands must be re-passed at load
    time."""
    factory_name = spec["factory"]
    if factory_name not in _FACTORY_REGISTRY:
        raise ValueError(
            f"Unknown estimand factory {factory_name!r}; only built-ins "
            f"({sorted(_FACTORY_REGISTRY)}) are round-trippable. For custom "
            f"estimands, pass `estimand=...` explicitly to .load(...)."
        )
    return _FACTORY_REGISTRY[factory_name](**spec.get("args", {}))
