"""Reference-parity utilities: compare a candidate predictor against a saved
reference α̂ array on identical data, reporting Pearson correlation and RMSE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParityReport:
    n: int
    pearson: float
    rmse: float
    max_abs_diff: float

    def summary(self) -> str:
        return (
            f"parity (n={self.n}): Pearson={self.pearson:.4f}, "
            f"RMSE={self.rmse:.4f}, max|Δ|={self.max_abs_diff:.4f}"
        )


def compare(alpha_a: np.ndarray, alpha_b: np.ndarray) -> ParityReport:
    a = np.asarray(alpha_a, dtype=float)
    b = np.asarray(alpha_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    if a.std() == 0 or b.std() == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(a, b)[0, 1])
    return ParityReport(n=a.size, pearson=pearson, rmse=rmse, max_abs_diff=max_abs)
