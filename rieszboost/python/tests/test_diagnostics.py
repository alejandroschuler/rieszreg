import numpy as np
import pytest

from rieszboost.diagnostics import diagnose


def test_diagnose_from_array():
    rng = np.random.default_rng(0)
    alpha = rng.normal(0, 1, size=500)
    d = diagnose(alpha)
    assert d.n == 500
    assert d.min < d.max
    assert pytest.approx(d.rms, rel=1e-3) == float(np.sqrt(np.mean(alpha**2)))
    assert d.warnings == []


def test_diagnose_warns_on_extreme_outlier():
    rng = np.random.default_rng(1)
    alpha = np.concatenate([rng.normal(0, 1, 999), [500.0]])
    d = diagnose(alpha, extreme_threshold=30.0)
    assert d.n_extreme == 1
    assert any("max |alpha_hat|" in w for w in d.warnings)


def test_diagnose_warns_on_many_extremes():
    alpha = np.concatenate([np.zeros(900), np.full(100, 50.0)])
    d = diagnose(alpha, extreme_threshold=30.0, extreme_fraction_warn=0.01)
    assert d.extreme_fraction == 0.1
    assert any("near-positivity" in w for w in d.warnings)


def test_diagnose_requires_alpha_or_booster():
    with pytest.raises(ValueError):
        diagnose()


def test_diagnose_summary_renders_without_error():
    d = diagnose(np.linspace(-2, 2, 100))
    s = d.summary()
    assert "RMS magnitude" in s
    assert "extreme rows" in s


def test_diagnose_with_booster_includes_riesz_loss():
    from rieszboost.engine import fit
    from rieszboost.estimands import ATE

    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(0, 1, n)
    pi = 1 / (1 + np.exp(-(-0.02 * x - x**2 + 4 * np.log(x + 0.3) + 1.5)))
    a = rng.binomial(1, pi)
    rows = [{"a": int(ai), "x": float(xi)} for ai, xi in zip(a, x)]
    booster = fit(
        rows,
        ATE(),
        feature_keys=("a", "x"),
        num_boost_round=20,
        learning_rate=0.1,
        max_depth=3,
    )
    d = diagnose(booster=booster, rows=rows, m=ATE())
    assert d.riesz_loss is not None
    assert d.n == n
