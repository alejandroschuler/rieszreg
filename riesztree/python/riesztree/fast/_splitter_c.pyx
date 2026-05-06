# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""Cython continuous-split sweep.

Mirrors ``riesztree.splitter.best_split_continuous`` exactly, but the
inner loop runs at C speed (no per-threshold Python attribute lookups,
no per-call ``leaf_loss(...)`` Python function dispatch). The argsort
+ cumulative sums up front are still numpy calls (already fast).

The function-pointer dispatch into the four built-in loss kernels lives
in :mod:`_loss_kernels`. Custom user losses fall back to the Python
splitter — see :mod:`riesztree.grow.best_split_at`.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport INFINITY, isfinite

from ._loss_kernels cimport dispatch_leaf_loss

ctypedef cnp.float64_t f64
ctypedef cnp.int64_t i64
ctypedef cnp.intp_t isize


# Function-pointer signature for a user-supplied leaf-loss kernel. Numba's
# ``@cfunc("float64(float64, float64)")`` produces exactly this shape;
# users register the cfunc's ``.address`` via
# :func:`riesztree.fast.register_fast_leaf_solver`.
ctypedef double (*leaf_loss_fn)(double, double) noexcept nogil


def best_split_continuous_c(
    cnp.ndarray[f64, ndim=1] feature_col,
    cnp.ndarray[f64, ndim=1] D,
    cnp.ndarray[f64, ndim=1] C,
    cnp.ndarray[i64, ndim=1] idx,
    int loss_kind,
    double bounded_lo,
    double bounded_hi,
    int min_orig_leaf,
):
    """Best gain split on a continuous feature for the rows in ``idx``.

    Returns ``(gain, threshold, left_idx, right_idx)`` or ``None`` —
    same shape as the Python ``best_split_continuous``.
    """
    cdef Py_ssize_t n = idx.shape[0]
    if n < 2:
        return None

    # Extract feature values for the leaf, then sort (ascending).
    cdef cnp.ndarray[f64, ndim=1] vals = feature_col.take(idx)
    cdef cnp.ndarray[isize, ndim=1] order = np.argsort(vals, kind="mergesort")
    cdef cnp.ndarray[i64, ndim=1] sidx = idx.take(order)
    cdef cnp.ndarray[f64, ndim=1] svals = vals.take(order)
    cdef cnp.ndarray[f64, ndim=1] sD = D.take(sidx)
    cdef cnp.ndarray[f64, ndim=1] sC = C.take(sidx)
    cdef cnp.ndarray[i64, ndim=1] sorig = (sD > 0).astype(np.int64)

    cdef cnp.ndarray[f64, ndim=1] cum_D = np.cumsum(sD)
    cdef cnp.ndarray[f64, ndim=1] cum_C = np.cumsum(sC)
    cdef cnp.ndarray[i64, ndim=1] cum_orig = np.cumsum(sorig)

    cdef f64[::1] svals_v = svals
    cdef f64[::1] cum_D_v = cum_D
    cdef f64[::1] cum_C_v = cum_C
    cdef i64[::1] cum_orig_v = cum_orig

    cdef double total_D = cum_D_v[n - 1]
    cdef double total_C = cum_C_v[n - 1]
    cdef i64 total_orig = cum_orig_v[n - 1]
    cdef double parent_loss = dispatch_leaf_loss(
        loss_kind, total_D, total_C, bounded_lo, bounded_hi
    )

    cdef double best_gain = -INFINITY
    cdef Py_ssize_t best_k = -1
    cdef double best_threshold = 0.0
    cdef double D_l, C_l, D_r, C_r, L_l, L_r, gain
    cdef i64 n_l, n_r
    cdef Py_ssize_t k

    for k in range(n - 1):
        if svals_v[k] == svals_v[k + 1]:
            continue
        D_l = cum_D_v[k]
        C_l = cum_C_v[k]
        n_l = cum_orig_v[k]
        D_r = total_D - D_l
        C_r = total_C - C_l
        n_r = total_orig - n_l
        if n_l < min_orig_leaf or n_r < min_orig_leaf:
            continue
        L_l = dispatch_leaf_loss(loss_kind, D_l, C_l, bounded_lo, bounded_hi)
        L_r = dispatch_leaf_loss(loss_kind, D_r, C_r, bounded_lo, bounded_hi)
        if not isfinite(L_l) or not isfinite(L_r):
            continue
        gain = parent_loss - L_l - L_r
        if gain > best_gain:
            best_gain = gain
            best_k = k
            best_threshold = 0.5 * (svals_v[k] + svals_v[k + 1])

    if best_k < 0:
        return None

    # Slices and copies match the Python splitter's return contract.
    cdef cnp.ndarray[i64, ndim=1] left_idx = sidx[: best_k + 1].copy()
    cdef cnp.ndarray[i64, ndim=1] right_idx = sidx[best_k + 1:].copy()
    return (best_gain, best_threshold, left_idx, right_idx)


def best_split_continuous_user_c(
    cnp.ndarray[f64, ndim=1] feature_col,
    cnp.ndarray[f64, ndim=1] D,
    cnp.ndarray[f64, ndim=1] C,
    cnp.ndarray[i64, ndim=1] idx,
    Py_ssize_t leaf_loss_addr,
    int min_orig_leaf,
):
    """Cython best-split sweep using a user-supplied leaf-loss function.

    ``leaf_loss_addr`` is the C-callable address of a ``double(double, double)``
    function — typically the ``.address`` attribute of a Numba ``@cfunc``
    instance with signature ``"float64(float64, float64)"``. The user closes
    over any extra constants their loss needs (analogous to how
    ``BoundedSquaredLoss`` closes over ``lo``/``hi``).

    Same return shape as :func:`best_split_continuous_c` and the Python
    ``best_split_continuous``.
    """
    cdef leaf_loss_fn leaf_loss = <leaf_loss_fn><void*>leaf_loss_addr

    cdef Py_ssize_t n = idx.shape[0]
    if n < 2:
        return None

    cdef cnp.ndarray[f64, ndim=1] vals = feature_col.take(idx)
    cdef cnp.ndarray[isize, ndim=1] order = np.argsort(vals, kind="mergesort")
    cdef cnp.ndarray[i64, ndim=1] sidx = idx.take(order)
    cdef cnp.ndarray[f64, ndim=1] svals = vals.take(order)
    cdef cnp.ndarray[f64, ndim=1] sD = D.take(sidx)
    cdef cnp.ndarray[f64, ndim=1] sC = C.take(sidx)
    cdef cnp.ndarray[i64, ndim=1] sorig = (sD > 0).astype(np.int64)

    cdef cnp.ndarray[f64, ndim=1] cum_D = np.cumsum(sD)
    cdef cnp.ndarray[f64, ndim=1] cum_C = np.cumsum(sC)
    cdef cnp.ndarray[i64, ndim=1] cum_orig = np.cumsum(sorig)

    cdef f64[::1] svals_v = svals
    cdef f64[::1] cum_D_v = cum_D
    cdef f64[::1] cum_C_v = cum_C
    cdef i64[::1] cum_orig_v = cum_orig

    cdef double total_D = cum_D_v[n - 1]
    cdef double total_C = cum_C_v[n - 1]
    cdef i64 total_orig = cum_orig_v[n - 1]
    cdef double parent_loss = leaf_loss(total_D, total_C)

    cdef double best_gain = -INFINITY
    cdef Py_ssize_t best_k = -1
    cdef double best_threshold = 0.0
    cdef double D_l, C_l, D_r, C_r, L_l, L_r, gain
    cdef i64 n_l, n_r
    cdef Py_ssize_t k

    for k in range(n - 1):
        if svals_v[k] == svals_v[k + 1]:
            continue
        D_l = cum_D_v[k]
        C_l = cum_C_v[k]
        n_l = cum_orig_v[k]
        D_r = total_D - D_l
        C_r = total_C - C_l
        n_r = total_orig - n_l
        if n_l < min_orig_leaf or n_r < min_orig_leaf:
            continue
        L_l = leaf_loss(D_l, C_l)
        L_r = leaf_loss(D_r, C_r)
        if not isfinite(L_l) or not isfinite(L_r):
            continue
        gain = parent_loss - L_l - L_r
        if gain > best_gain:
            best_gain = gain
            best_k = k
            best_threshold = 0.5 * (svals_v[k] + svals_v[k + 1])

    if best_k < 0:
        return None
    cdef cnp.ndarray[i64, ndim=1] left_idx = sidx[: best_k + 1].copy()
    cdef cnp.ndarray[i64, ndim=1] right_idx = sidx[best_k + 1:].copy()
    return (best_gain, best_threshold, left_idx, right_idx)
