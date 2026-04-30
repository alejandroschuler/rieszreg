"""Tracer linear-form algebra tests."""

from __future__ import annotations

import pytest

from rieszreg import Estimand, LinearForm, Tracer, trace


def _coef_at(pairs, **point):
    for c, p in pairs:
        if p == point:
            return c
    return 0.0


def test_single_call_yields_single_term():
    alpha = Tracer()
    form = alpha(a=1, x=0.5)
    pairs = form.as_pairs()
    assert pairs == [(1.0, {"a": 1, "x": 0.5})]


def test_addition_and_subtraction():
    alpha = Tracer()
    form = alpha(a=1, x=1.0) - alpha(a=0, x=1.0) + 2 * alpha(a=2, x=1.0)
    pairs = form.as_pairs()
    assert _coef_at(pairs, a=1, x=1.0) == 1.0
    assert _coef_at(pairs, a=0, x=1.0) == -1.0
    assert _coef_at(pairs, a=2, x=1.0) == 2.0


def test_dedup_merges_same_point():
    alpha = Tracer()
    form = alpha(a=1, x=1.0) + 0.5 * alpha(a=1, x=1.0)
    pairs = form.as_pairs()
    assert pairs == [(1.5, {"a": 1, "x": 1.0})]


def test_zero_coef_dropped():
    alpha = Tracer()
    form = alpha(a=1, x=1.0) - alpha(a=1, x=1.0)
    assert form.as_pairs() == []


def test_scalar_multiplication():
    alpha = Tracer()
    form = 3.0 * alpha(a=1, x=1.0)
    assert form.as_pairs() == [(3.0, {"a": 1, "x": 1.0})]


def test_division_by_scalar():
    alpha = Tracer()
    form = alpha(a=1, x=1.0) / 4.0
    assert form.as_pairs() == [(0.25, {"a": 1, "x": 1.0})]


def test_constant_addition_raises():
    alpha = Tracer()
    with pytest.raises(TypeError, match="constant offsets"):
        alpha(a=1, x=1.0) + 1.0


def test_squaring_raises():
    def m(alpha):
        def inner(z):
            return alpha(a=1, x=z["x"]) ** 2
        return inner
    est = Estimand(feature_keys=("a", "x"), m=m)
    with pytest.raises(TypeError):
        trace(est, {"a": 0, "x": 0.5})


def test_non_linearform_return_raises():
    def m(alpha):
        def inner(z):
            return "not a linear form"
        return inner
    est = Estimand(feature_keys=("a", "x"), m=m)
    with pytest.raises(TypeError, match="LinearForm"):
        trace(est, {"a": 0, "x": 0.5})


def test_zero_scalar_return_yields_empty():
    def m(alpha):
        def inner(z):
            return 0
        return inner
    est = Estimand(feature_keys=("a", "x"), m=m)
    assert trace(est, {"a": 0, "x": 0.5}) == []
