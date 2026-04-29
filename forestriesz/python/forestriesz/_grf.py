"""Internal BaseGRF subclass that consumes prepacked moment data.

The backend computes the per-row Jacobian J = phi(W_i) phi(W_i)' and moment
A = m(W_i; phi) before fit, packs them into the EconML T/y arrays, and
delegates split search and leaf solve to BaseGRF's MSE criterion.

Packing convention used by `forestriesz.backend.ForestRieszBackend`:

    X       : (n, n_split)            split features
    T       : (n, p² + p)              first p² cols = J flattened (Fortran),
                                       next p cols  = phi(W_i)
    y       : (n, p)                   per-row moment vector A = m(W_i; phi)
"""

from __future__ import annotations

import numpy as np
from econml.grf._base_grf import BaseGRF


class _RieszGRF(BaseGRF):
    """BaseGRF specialized for the Riesz linear-moment equation.

    `_get_alpha_and_pointJ` returns the per-row (A, J) the criterion needs;
    `_get_n_outputs_decomposition` declares all p outputs are relevant
    (no nuisance dimensions).
    """

    def _get_alpha_and_pointJ(self, X, T, y, **kwargs):
        p = y.shape[1]
        # First p² columns of T hold J flattened in Fortran order per row.
        pointJ = np.ascontiguousarray(T[:, : p * p])
        alpha = np.ascontiguousarray(y)
        return alpha, pointJ

    def _get_n_outputs_decomposition(self, X, T, y, **kwargs):
        p = y.shape[1]
        return p, p
