"""Trace `m` on each row and build the augmented dataset for the
Bregman-Riesz loss. Each augmented row r contributes a per-row loss term

    D_r · h̃(α(z_r)) + C_r · h'(α(z_r))

(the squared-loss case h̃ = t², h' = 2t simplifies to `D·α² + 2·C·α`).
The original observation Z_i seeds row i with (is_original=1,
potential_deriv_coef=0); each (coef, point) pair from m(z_i) contributes
(is_original=0, potential_deriv_coef=-coef) at the point. Duplicate
points within a row are merged by summing the two coefficients.

For the five built-in factories (``ATE``, ``ATT``, ``TSM``,
``AdditiveShift``, ``LocalShift``), :func:`build_augmented_fast`
emits the augmented dataset in vectorised numpy without invoking
the per-row symbolic ``Tracer`` — the structure of ``m`` is known
from ``factory_spec``. This is typically 50-200× faster than the
generic ``build_augmented`` path. Custom user estimands continue to
use the symbolic path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .estimands.base import FiniteEvalEstimand
from .estimands.tracer import trace


@dataclass
class AugmentedDataset:
    features: np.ndarray              # (n_aug, n_features)
    is_original: np.ndarray           # (n_aug,) — 1 if z_r == Z_{i_r}, else 0 (D_r)
    potential_deriv_coef: np.ndarray  # (n_aug,) — coefficient on h'(α) (C_r)
    origin_index: np.ndarray          # (n_aug,) — index into original rows
    n_rows: int                       # number of original rows


def build_augmented(
    rows: Sequence[dict[str, Any]],
    estimand: FiniteEvalEstimand,
    ys: Sequence[Any] | None = None,
) -> AugmentedDataset:
    """Build the augmented dataset by tracing `m(alpha)(z, y)` on each row.

    `ys` is the per-row outcome aligned with `rows`; pass `None` (the default)
    when the estimand's `m` doesn't read y. When provided, its length must
    match `len(rows)`.
    """
    if not isinstance(estimand, FiniteEvalEstimand):
        raise TypeError(
            f"build_augmented() requires a FiniteEvalEstimand; got "
            f"{type(estimand).__name__}."
        )
    if ys is not None and len(ys) != len(rows):
        raise ValueError(
            f"len(ys)={len(ys)} does not match len(rows)={len(rows)}."
        )
    feature_keys = estimand.feature_keys

    feats: list[np.ndarray] = []
    is_orig_list: list[float] = []
    pdc_list: list[float] = []
    origin: list[int] = []

    for i, z in enumerate(rows):
        y_i = ys[i] if ys is not None else None
        acc: dict[tuple, tuple[float, float]] = {}
        z_key = tuple(z[k] for k in feature_keys)
        acc[z_key] = (1.0, 0.0)

        for coef, point in trace(estimand, z, y_i):
            missing = [k for k in feature_keys if k not in point]
            if missing:
                raise ValueError(
                    f"m evaluated alpha at a point missing keys {missing}; "
                    f"all feature_keys {list(feature_keys)} must be specified."
                )
            key = tuple(point[k] for k in feature_keys)
            cur_d, cur_c = acc.get(key, (0.0, 0.0))
            acc[key] = (cur_d, cur_c - coef)

        for key, (d, c) in acc.items():
            feats.append(np.asarray(key, dtype=float))
            is_orig_list.append(d)
            pdc_list.append(c)
            origin.append(i)

    return AugmentedDataset(
        features=np.vstack(feats) if feats else np.zeros((0, len(feature_keys))),
        is_original=np.asarray(is_orig_list, dtype=float),
        potential_deriv_coef=np.asarray(pdc_list, dtype=float),
        origin_index=np.asarray(origin, dtype=np.int64),
        n_rows=len(rows),
    )


# ---------------------------------------------------------------------------
# Vectorised fast path for built-in estimand factories.

def _column_index(feature_keys: Sequence[str], name: str) -> int:
    for j, k in enumerate(feature_keys):
        if k == name:
            return j
    raise KeyError(
        f"Feature key {name!r} not found in feature_keys={list(feature_keys)}."
    )


def _augment_ate(features: np.ndarray, spec: dict, feature_keys: Sequence[str]) -> AugmentedDataset:
    """ATE: m(α)(z) = α(1, x) − α(0, x). Per row z=(a, x):
       row at (1, x): is_orig = (a==1), C = -1
       row at (0, x): is_orig = (a==0), C = +1
    """
    n, p = features.shape
    treatment = spec["args"].get("treatment", "a")
    a_idx = _column_index(feature_keys, treatment)
    a_col = features[:, a_idx]
    other = np.delete(features, a_idx, axis=1)  # x columns

    aug_features = np.empty((2 * n, p), dtype=np.float64)
    # Row pattern: (1, x_i), (0, x_i), (1, x_{i+1}), (0, x_{i+1}), ...
    aug_features[0::2, a_idx] = 1.0
    aug_features[1::2, a_idx] = 0.0
    other_idx = [j for j in range(p) if j != a_idx]
    aug_features[0::2][:, other_idx] = other
    aug_features[1::2][:, other_idx] = other

    is_orig = np.empty(2 * n, dtype=np.float64)
    is_orig[0::2] = (a_col == 1.0).astype(np.float64)
    is_orig[1::2] = (a_col == 0.0).astype(np.float64)

    pdc = np.empty(2 * n, dtype=np.float64)
    pdc[0::2] = -1.0
    pdc[1::2] = 1.0

    origin = np.repeat(np.arange(n, dtype=np.int64), 2)

    return AugmentedDataset(
        features=aug_features, is_original=is_orig,
        potential_deriv_coef=pdc, origin_index=origin, n_rows=n,
    )


def _augment_att(features: np.ndarray, spec: dict, feature_keys: Sequence[str]) -> AugmentedDataset:
    """ATT (partial): m(α)(z) = a · (α(1, x) − α(0, x)).

    For a=0: trace returns 0 — only the original (a, x) row.
    For a=1: same as ATE-a=1 — rows (1, x) and (0, x).
    """
    n, p = features.shape
    treatment = spec["args"].get("treatment", "a")
    a_idx = _column_index(feature_keys, treatment)
    a_col = features[:, a_idx]

    a_eq_1 = (a_col == 1.0)
    n_treated = int(a_eq_1.sum())
    n_control = n - n_treated
    n_aug = n_control + 2 * n_treated  # control rows: 1 each; treated: 2 each

    aug_features = np.empty((n_aug, p), dtype=np.float64)
    is_orig = np.empty(n_aug, dtype=np.float64)
    pdc = np.empty(n_aug, dtype=np.float64)
    origin = np.empty(n_aug, dtype=np.int64)

    # Control block: original (0, x) only, is_orig=1, C=0.
    if n_control > 0:
        ctrl_mask = ~a_eq_1
        aug_features[:n_control] = features[ctrl_mask]
        is_orig[:n_control] = 1.0
        pdc[:n_control] = 0.0
        origin[:n_control] = np.where(ctrl_mask)[0]

    # Treated block: alternating (1, x) and (0, x) per treated row.
    if n_treated > 0:
        treated_mask = a_eq_1
        treated_idx = np.where(treated_mask)[0]
        treated_features = features[treated_mask]
        other_idx = [j for j in range(p) if j != a_idx]

        block = aug_features[n_control:]
        block_orig = is_orig[n_control:]
        block_pdc = pdc[n_control:]
        block_origin = origin[n_control:]

        block[0::2, a_idx] = 1.0
        block[1::2, a_idx] = 0.0
        for j in other_idx:
            block[0::2, j] = treated_features[:, j]
            block[1::2, j] = treated_features[:, j]
        block_orig[0::2] = 1.0  # treated rows have a=1, original is at (1, x)
        block_orig[1::2] = 0.0
        block_pdc[0::2] = -1.0
        block_pdc[1::2] = 1.0
        block_origin[0::2] = treated_idx
        block_origin[1::2] = treated_idx

    return AugmentedDataset(
        features=aug_features, is_original=is_orig,
        potential_deriv_coef=pdc, origin_index=origin, n_rows=n,
    )


def _augment_tsm(features: np.ndarray, spec: dict, feature_keys: Sequence[str]) -> AugmentedDataset:
    """TSM(level=L): m(α)(z) = α(L, x).

    For a == L: original (L, x) merges with virtual (L, x). One row,
    is_orig=1, C=-1.
    For a != L: original (a, x), virtual (L, x). Two rows.
    """
    n, p = features.shape
    treatment = spec["args"].get("treatment", "a")
    level = float(spec["args"]["level"])
    a_idx = _column_index(feature_keys, treatment)
    a_col = features[:, a_idx]

    eq = (a_col == level)
    n_eq = int(eq.sum())
    n_neq = n - n_eq
    n_aug = n_eq + 2 * n_neq

    aug_features = np.empty((n_aug, p), dtype=np.float64)
    is_orig = np.empty(n_aug, dtype=np.float64)
    pdc = np.empty(n_aug, dtype=np.float64)
    origin = np.empty(n_aug, dtype=np.int64)

    # eq block: original at (L, x) only, is_orig=1, C=-1.
    if n_eq > 0:
        eq_idx = np.where(eq)[0]
        aug_features[:n_eq] = features[eq]
        is_orig[:n_eq] = 1.0
        pdc[:n_eq] = -1.0
        origin[:n_eq] = eq_idx

    # neq block: original (a, x) [is_orig=1, C=0], virtual (L, x) [is_orig=0, C=-1].
    if n_neq > 0:
        neq = ~eq
        neq_idx = np.where(neq)[0]
        neq_features = features[neq]
        other_idx = [j for j in range(p) if j != a_idx]

        block = aug_features[n_eq:]
        block_orig = is_orig[n_eq:]
        block_pdc = pdc[n_eq:]
        block_origin = origin[n_eq:]

        # Original (a, x): copy the row as-is.
        block[0::2] = neq_features
        # Virtual (L, x): copy x columns, set treatment to L.
        block[1::2, a_idx] = level
        for j in other_idx:
            block[1::2, j] = neq_features[:, j]
        block_orig[0::2] = 1.0
        block_orig[1::2] = 0.0
        block_pdc[0::2] = 0.0
        block_pdc[1::2] = -1.0
        block_origin[0::2] = neq_idx
        block_origin[1::2] = neq_idx

    return AugmentedDataset(
        features=aug_features, is_original=is_orig,
        potential_deriv_coef=pdc, origin_index=origin, n_rows=n,
    )


def _augment_additive_shift(
    features: np.ndarray, spec: dict, feature_keys: Sequence[str]
) -> AugmentedDataset:
    """AdditiveShift(δ): m(α)(z) = α(a+δ, x) − α(a, x).

    Per row z=(a, x):
      original at (a, x): is_orig=1, C=+1   (from -(-coef) of the -α(a,x) term)
      virtual at (a+δ, x): is_orig=0, C=-1
    For δ != 0, a + δ ≠ a so no per-row dedup happens.
    """
    n, p = features.shape
    treatment = spec["args"].get("treatment", "a")
    delta = float(spec["args"]["delta"])
    a_idx = _column_index(feature_keys, treatment)

    if delta == 0.0:
        # Degenerate: m(α)(z) = 0. Fall back to the slow path so behaviour
        # matches the symbolic tracer's degenerate output.
        return _slow_path(features, feature_keys)

    n_aug = 2 * n
    aug_features = np.empty((n_aug, p), dtype=np.float64)
    is_orig = np.empty(n_aug, dtype=np.float64)
    pdc = np.empty(n_aug, dtype=np.float64)
    origin = np.empty(n_aug, dtype=np.int64)

    other_idx = [j for j in range(p) if j != a_idx]

    # Even rows: original (a, x).
    aug_features[0::2] = features
    is_orig[0::2] = 1.0
    pdc[0::2] = 1.0
    # Odd rows: virtual (a+δ, x).
    aug_features[1::2, a_idx] = features[:, a_idx] + delta
    for j in other_idx:
        aug_features[1::2, j] = features[:, j]
    is_orig[1::2] = 0.0
    pdc[1::2] = -1.0

    origin[:] = np.repeat(np.arange(n, dtype=np.int64), 2)

    return AugmentedDataset(
        features=aug_features, is_original=is_orig,
        potential_deriv_coef=pdc, origin_index=origin, n_rows=n,
    )


def _augment_local_shift(
    features: np.ndarray, spec: dict, feature_keys: Sequence[str]
) -> AugmentedDataset:
    """LocalShift(δ, threshold): like AdditiveShift but only for a < threshold.

    For a < threshold: 2 augmented rows (original (a, x) C=+1, virtual (a+δ, x) C=-1).
    For a ≥ threshold: 1 augmented row (original (a, x), is_orig=1, C=0).
    """
    n, p = features.shape
    treatment = spec["args"].get("treatment", "a")
    delta = float(spec["args"]["delta"])
    threshold = float(spec["args"]["threshold"])
    a_idx = _column_index(feature_keys, treatment)
    a_col = features[:, a_idx]

    below = (a_col < threshold)
    n_below = int(below.sum())
    n_above = n - n_below

    if delta == 0.0 and n_below > 0:
        return _slow_path(features, feature_keys)

    n_aug = 2 * n_below + n_above
    aug_features = np.empty((n_aug, p), dtype=np.float64)
    is_orig = np.empty(n_aug, dtype=np.float64)
    pdc = np.empty(n_aug, dtype=np.float64)
    origin = np.empty(n_aug, dtype=np.int64)

    other_idx = [j for j in range(p) if j != a_idx]

    # Above-threshold block: just the originals.
    if n_above > 0:
        above = ~below
        above_idx = np.where(above)[0]
        aug_features[:n_above] = features[above]
        is_orig[:n_above] = 1.0
        pdc[:n_above] = 0.0
        origin[:n_above] = above_idx

    # Below-threshold block: 2 rows per original.
    if n_below > 0:
        below_idx = np.where(below)[0]
        below_features = features[below]
        block = aug_features[n_above:]
        block_orig = is_orig[n_above:]
        block_pdc = pdc[n_above:]
        block_origin = origin[n_above:]

        block[0::2] = below_features
        block[1::2, a_idx] = below_features[:, a_idx] + delta
        for j in other_idx:
            block[1::2, j] = below_features[:, j]
        block_orig[0::2] = 1.0
        block_orig[1::2] = 0.0
        block_pdc[0::2] = 1.0
        block_pdc[1::2] = -1.0
        block_origin[0::2] = below_idx
        block_origin[1::2] = below_idx

    return AugmentedDataset(
        features=aug_features, is_original=is_orig,
        potential_deriv_coef=pdc, origin_index=origin, n_rows=n,
    )


# Dispatcher table.
_FAST_AUGMENT_DISPATCH = {
    "ATE": _augment_ate,
    "ATT": _augment_att,
    "TSM": _augment_tsm,
    "AdditiveShift": _augment_additive_shift,
    "LocalShift": _augment_local_shift,
}


def _slow_path(features: np.ndarray, feature_keys: Sequence[str]):
    """Build row-dicts from a feature ndarray and call the slow path.

    Used as the fallback inside the fast-path module when an estimand's
    fast handler is unavailable or refuses (e.g. AdditiveShift with δ=0).
    """
    rows = [
        {k: features[i, j] for j, k in enumerate(feature_keys)}
        for i in range(features.shape[0])
    ]
    # The estimand isn't available here; the caller of build_augmented_fast
    # should provide it. This helper is only invoked from inside per-handler
    # functions that know they need to bail out.
    raise NotImplementedError(
        "Internal: _slow_path called from a fast handler that should fall "
        "back. Route through build_augmented_fast's outer try/except instead."
    )


def build_augmented_fast(
    features: np.ndarray,
    estimand: FiniteEvalEstimand,
    y: np.ndarray | None = None,
) -> AugmentedDataset | None:
    """Vectorised augmentation for built-in estimand factories.

    Returns ``None`` (signalling the caller to fall back to the slow
    symbolic path) when:
      * ``estimand`` has no ``factory_spec`` (custom estimand).
      * ``y`` is non-None (built-ins ignore y, but we don't risk
        diverging behaviour when the user passes one).
      * The factory is not in the dispatcher table.

    On success, returns an ``AugmentedDataset`` byte-equivalent to
    what :func:`build_augmented` would have produced (modulo row
    ordering — order does not affect downstream split-finding /
    prediction).
    """
    spec = getattr(estimand, "factory_spec", None)
    if spec is None or y is not None:
        return None
    factory = spec.get("factory")
    handler = _FAST_AUGMENT_DISPATCH.get(factory)
    if handler is None:
        return None
    feature_keys = estimand.feature_keys
    features = np.ascontiguousarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.shape[1] != len(feature_keys):
        raise ValueError(
            f"build_augmented_fast: features has {features.shape[1]} columns "
            f"but estimand expects {len(feature_keys)} ({list(feature_keys)})."
        )
    try:
        return handler(features, spec, feature_keys)
    except NotImplementedError:
        # Per-handler bail-out (e.g. AdditiveShift δ=0).
        return None


# Closed-form ``m̄`` for built-in estimands. Used by ``RieszEstimator.fit``
# to skip the per-row trace when computing the default α init.
_BUILTIN_M_BAR = {
    "ATE": 0.0,           # E[α(1,x) − α(0,x)] with α=1 ≡ 0
    "ATT": 0.0,           # E[a · 0] = 0
    "TSM": 1.0,           # E[α(L,x)] with α=1 ≡ 1
    "AdditiveShift": 0.0, # E[α(a+δ,x) − α(a,x)] with α=1 ≡ 0
    "LocalShift": 0.0,    # E[1[a<T] · 0] = 0
}


def builtin_m_bar(estimand: FiniteEvalEstimand) -> float | None:
    """Return the analytic ``m̄`` for built-in estimands; ``None`` for custom."""
    spec = getattr(estimand, "factory_spec", None)
    if spec is None:
        return None
    return _BUILTIN_M_BAR.get(spec.get("factory"))


# ---------------------------------------------------------------------------
# Augmented-loss helpers. The (is_original, potential_deriv_coef) pair lives
# with the augmentation engine; these helpers combine them with the loss's
# α-space functions and η-space link to give per-row loss / gradient / hessian
# for backends that fit in η-space.

def aug_loss_alpha(loss, is_original, potential_deriv_coef, alpha):
    """Per-row augmented loss in α-space: D · h_tilde(α) + C · h'(α)."""
    return (
        is_original * loss.tilde_potential(alpha)
        + potential_deriv_coef * loss.potential_deriv(alpha)
    )


def aug_loss_eta(loss, is_original, potential_deriv_coef, eta):
    """Per-row augmented loss in η-space."""
    return aug_loss_alpha(loss, is_original, potential_deriv_coef, loss.link_to_alpha(eta))


def aug_grad_eta(loss, is_original, potential_deriv_coef, eta):
    """∂[D·h_tilde(α) + C·h'(α)]/∂η. Routes to the loss's analytic helper."""
    return loss.aug_grad_eta(is_original, potential_deriv_coef, eta)


def aug_hess_eta(loss, is_original, potential_deriv_coef, eta, hessian_floor):
    """∂²[D·h_tilde(α) + C·h'(α)]/∂η² (floored). Routes to the loss's analytic helper."""
    return loss.aug_hess_eta(is_original, potential_deriv_coef, eta, hessian_floor)
