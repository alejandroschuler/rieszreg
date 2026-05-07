"""Tests for the base Diagnostics dataclass and diagnose() helper."""

from __future__ import annotations

import numpy as np

from rieszreg import Diagnostics, diagnose


def test_diagnose_from_alpha_hat_basic():
    alpha = np.array([0.0, 1.0, -1.0, 2.0, -3.0])
    d = diagnose(alpha_hat=alpha)
    assert isinstance(d, Diagnostics)
    assert d.n == 5
    assert d.rms > 0
    assert d.min == -3.0
    assert d.max == 2.0
    # No extreme rows under default threshold of 30.
    assert d.n_extreme == 0


def test_extreme_threshold_warning():
    alpha = np.array([0.0, 1.0, 100.0, -200.0])  # 50% extreme at threshold 30.
    d = diagnose(alpha_hat=alpha)
    assert d.n_extreme == 2
    assert d.extreme_fraction == 0.5
    # First warning fires.
    assert any("near-positivity" in w for w in d.warnings)


def test_outlier_warning_fires_for_single_extrapolation():
    # 99 modest values + 1 huge outlier.
    alpha = np.concatenate([np.full(99, 1.0), [200.0]])
    d = diagnose(alpha_hat=alpha)
    assert any(">10x" in w for w in d.warnings)


def test_diagnostics_summary_renders():
    alpha = np.array([0.5, -1.5, 2.0])
    d = diagnose(alpha_hat=alpha)
    text = d.summary()
    assert "Riesz representer diagnostics" in text
    assert "RMS magnitude" in text
