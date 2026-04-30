"""Smoke tests for Bregman-Riesz LossSpec implementations."""

from __future__ import annotations

import numpy as np
import pytest

from rieszreg import (
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
    LossSpec,
    SquaredLoss,
    loss_from_spec,
)


@pytest.mark.parametrize(
    "loss",
    [SquaredLoss(), KLLoss(), BernoulliLoss(), BoundedSquaredLoss(lo=0.0, hi=1.0)],
)
def test_to_from_spec_round_trip(loss: LossSpec):
    rebuilt = loss_from_spec(loss.to_spec())
    assert type(rebuilt) is type(loss)
    assert rebuilt.name == loss.name


def test_squared_link_is_identity():
    loss = SquaredLoss()
    eta = np.array([-1.0, 0.0, 2.0])
    assert np.allclose(loss.link_to_alpha(eta), eta)
    assert np.allclose(loss.alpha_to_eta(eta), eta)


def test_squared_gradient_and_hessian_match_finite_diff():
    loss = SquaredLoss()
    a = np.array([1.0, 0.0, 1.0])
    b = np.array([0.0, -2.0, -1.5])
    eta = np.array([0.5, 0.5, 0.5])
    h = 1e-6
    grad_analytic = loss.gradient(a, b, eta)
    loss_plus = loss.loss_row(a, b, loss.link_to_alpha(eta + h))
    loss_minus = loss.loss_row(a, b, loss.link_to_alpha(eta - h))
    grad_numeric = (loss_plus - loss_minus) / (2 * h)
    np.testing.assert_allclose(grad_analytic, grad_numeric, rtol=1e-4)
    # Hessian under squared loss is 2a (floored).
    hess = loss.hessian(a, b, eta, hessian_floor=0.0)
    np.testing.assert_allclose(hess, 2.0 * a)


def test_kl_link_keeps_alpha_positive():
    loss = KLLoss(max_eta=10.0)
    eta = np.array([-5.0, 0.0, 8.0])
    alpha = loss.link_to_alpha(eta)
    assert np.all(alpha > 0)


def test_kl_rejects_positive_b():
    loss = KLLoss()
    with pytest.raises(ValueError, match="non-negative"):
        loss.validate_coefficients(np.array([0.0, 0.5]))


def test_bernoulli_link_in_unit_interval():
    loss = BernoulliLoss(max_abs_eta=20.0)
    eta = np.linspace(-30, 30, 21)
    alpha = loss.link_to_alpha(eta)
    assert np.all((alpha > 0) & (alpha < 1))


def test_bounded_squared_clipping():
    loss = BoundedSquaredLoss(lo=2.0, hi=8.0)
    eta = np.array([-50.0, 0.0, 50.0])
    alpha = loss.link_to_alpha(eta)
    assert alpha.min() > 2.0
    assert alpha.max() < 8.0
    # m̄ outside the interval is clipped to the interior.
    assert loss.best_constant_init(0.0) > 2.0    # below lo → floored
    assert loss.best_constant_init(100.0) < 8.0  # above hi → ceilinged
    # m̄ inside the interval round-trips.
    assert loss.best_constant_init(5.0) == pytest.approx(5.0)


def test_bounded_squared_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="must be"):
        BoundedSquaredLoss(lo=5.0, hi=3.0)


def test_loss_from_spec_unknown_raises():
    with pytest.raises(ValueError, match="Unknown loss spec type"):
        loss_from_spec({"type": "Nope"})
