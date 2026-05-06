"""Python-side facade for the Cython splitter.

Translates a ``LossSpec`` instance to the integer ``loss_kind`` used by
the compiled kernels, plus the ``(lo, hi)`` parameters needed by
``BoundedSquaredLoss``. Custom user losses (anything not in the four
built-ins) return ``None`` from :func:`loss_kind_for`; the dispatcher
in :mod:`riesztree.grow` then falls back to the pure-Python splitter
with a one-time ``UserWarning``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from rieszreg import (
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
    LossSpec,
    SquaredLoss,
)

# Loss-kind constants exported from the compiled extension. Re-exported
# here for tests and for the grow.py dispatcher to use.
try:  # extension imports lazily so a fresh source checkout still loads
    from ._loss_kernels import (
        LOSS_BERNOULLI_PY as LOSS_BERNOULLI,
        LOSS_BOUNDED_SQUARED_PY as LOSS_BOUNDED_SQUARED,
        LOSS_KL_PY as LOSS_KL,
        LOSS_SQUARED_PY as LOSS_SQUARED,
    )
except ImportError:
    LOSS_SQUARED = 0
    LOSS_KL = 1
    LOSS_BERNOULLI = 2
    LOSS_BOUNDED_SQUARED = 3


def loss_kind_for(loss: LossSpec) -> tuple[int, float, float] | None:
    """Map a LossSpec to ``(loss_kind, bounded_lo, bounded_hi)``.

    Returns ``None`` for losses outside the four built-ins. The
    ``bounded_lo`` / ``bounded_hi`` slots are unused for non-bounded
    losses; we fill them with NaN so any accidental dereference is
    loud.
    """
    if isinstance(loss, SquaredLoss):
        return LOSS_SQUARED, np.nan, np.nan
    if isinstance(loss, KLLoss):
        return LOSS_KL, np.nan, np.nan
    if isinstance(loss, BernoulliLoss):
        return LOSS_BERNOULLI, np.nan, np.nan
    if isinstance(loss, BoundedSquaredLoss):
        return LOSS_BOUNDED_SQUARED, float(loss.lo), float(loss.hi)
    return None


def best_split_continuous_fast(
    feature_col: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    idx: np.ndarray,
    *,
    loss_kind: int,
    bounded_lo: float,
    bounded_hi: float,
    min_orig_leaf: int,
):
    """Fast Cython best-split sweep on a continuous feature.

    Falls back to the pure-Python ``best_split_continuous`` if the
    compiled extension hasn't been built. The dtypes are coerced once
    here so the inner loop sees the layout it expects.
    """
    try:
        from . import _splitter_c  # type: ignore[attr-defined]
    except ImportError:
        from ..splitter import best_split_continuous

        # Build a Python leaf_loss closure compatible with the Python splitter.
        from ._loss_kernels import py_dispatch_leaf_loss

        def _leaf_loss(D_, C_):
            return py_dispatch_leaf_loss(loss_kind, D_, C_, bounded_lo, bounded_hi)

        return best_split_continuous(
            feature_col, D, C, idx, _leaf_loss, min_orig_leaf=min_orig_leaf
        )

    feature_col = np.ascontiguousarray(feature_col, dtype=np.float64)
    D = np.ascontiguousarray(D, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    idx = np.ascontiguousarray(idx, dtype=np.int64)
    return _splitter_c.best_split_continuous_c(
        feature_col, D, C, idx,
        int(loss_kind), float(bounded_lo), float(bounded_hi),
        int(min_orig_leaf),
    )


_WARNED_FALLBACK: set[type] = set()


def warn_python_fallback(loss: LossSpec) -> None:
    """Emit a one-time UserWarning when the user's loss isn't in the four
    built-ins and growth has to use the Python splitter."""
    cls = type(loss)
    if cls in _WARNED_FALLBACK:
        return
    _WARNED_FALLBACK.add(cls)
    warnings.warn(
        f"Loss {cls.__name__!r} is not among the riesztree fast-splitter "
        f"built-ins (Squared/KL/Bernoulli/BoundedSquared); falling back "
        f"to the pure-Python splitter. Phase 5 will add a registration "
        f"hook for Numba @cfunc kernels.",
        UserWarning,
        stacklevel=3,
    )
