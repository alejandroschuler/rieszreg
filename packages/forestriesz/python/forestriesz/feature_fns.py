"""Default sieve features and split-feature selection for ForestRiesz.

A "sieve feature" is a vectorized callable that maps a feature matrix
``(n, n_features)`` to a column ``(n,)``. The columns of the input matrix are
ordered by the estimand's ``feature_keys``. The forest fits a per-leaf
coefficient θ over these basis functions; predictions are ``α(z) = θ(z) · φ(z)``.

For ATE-style estimands the natural basis is treatment indicators, so the
forest can split on covariates only and let the sieve resolve treatment.
"""

from __future__ import annotations

from typing import Callable, Sequence

from rieszreg import Estimand


def default_riesz_features(estimand: Estimand) -> list[Callable] | None:
    """Sensible sieve features for built-in estimands.

    Returns a list of callables for estimands with an obvious basis (ATE / ATT
    use treatment indicators, TSM uses a single level indicator). Returns
    ``None`` for estimands without one — pass ``riesz_feature_fns=None`` to use
    the locally constant default.
    """
    spec = estimand.factory_spec
    if not spec:
        return None
    factory = spec.get("factory")
    if factory in ("ATE", "ATT"):
        return [
            lambda f: 1.0 - f[:, 0],
            lambda f: f[:, 0],
        ]
    if factory == "TSM":
        level = float(spec["args"]["level"])
        return [lambda f, v=level: (f[:, 0] == v).astype(float)]
    return None


def default_split_feature_indices(
    estimand: Estimand,
    riesz_feature_fns: Sequence[Callable] | None,
) -> tuple[int, ...]:
    """Pick which feature columns the forest splits on.

    With a sieve, splits avoid features the sieve already resolves: for
    treatment-indexed sieves (ATE/ATT/TSM) the treatment column is dropped from
    the splitter so the forest only partitions covariate space. Without a
    sieve, every feature is fair game for splits.
    """
    n_features = len(estimand.feature_keys)
    if riesz_feature_fns is None:
        return tuple(range(n_features))

    spec = estimand.factory_spec
    if spec and spec.get("factory") in ("ATE", "ATT", "TSM"):
        # Treatment is feature_keys[0]; the sieve handles its variation.
        return tuple(range(1, n_features))

    return tuple(range(n_features))
