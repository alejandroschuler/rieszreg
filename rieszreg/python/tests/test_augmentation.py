"""Tests for build_augmented + AugmentedDataset."""

from __future__ import annotations

import numpy as np
import pytest

from rieszreg import (
    ATE,
    AdditiveShift,
    AugmentedDataset,
    FiniteEvalEstimand,
    build_augmented,
)


def test_ate_augmentation_shape():
    rows = [{"a": 0.0, "x": 1.0}, {"a": 1.0, "x": 2.0}]
    aug = build_augmented(rows, ATE())
    assert aug.n_rows == 2
    # Each row contributes itself (D=1, C=0) plus 2 counterfactual points
    # (D=0, C=±1) at (1, x) and (0, x). For row 0 (a=0), the original row
    # coincides with the (D=0, x=1.0) counterfactual point so they merge.
    # That merge means n_aug = 2 + 2 = 4 (not 6).
    assert aug.features.shape == (4, 2)
    assert aug.is_original.shape == (4,)
    assert aug.potential_deriv_coef.shape == (4,)
    # Origin index points back at the source row.
    assert aug.origin_index.shape == (4,)
    assert set(aug.origin_index.tolist()) == {0, 1}


def test_original_rows_have_D_one_C_zero():
    rows = [{"a": 0.5, "x": 1.5}]
    # Use AdditiveShift so the counterfactual point is distinct from the original.
    aug = build_augmented(rows, AdditiveShift(delta=0.7))
    # Pick out the row where features match (0.5, 1.5).
    mask = (aug.features[:, 0] == 0.5) & (aug.features[:, 1] == 1.5)
    assert mask.sum() == 1  # original row distinct from counterfactual
    # On the original row, D=1, C=-coef where coef is from m at this point.
    # m(alpha)(z) = alpha(a + delta, x) - alpha(a, x). At a=0.5: coef on
    # alpha(a=0.5, x=1.5) is -1, so C = +1. Plus the natural D=1 contribution.
    assert aug.is_original[mask].item() == 1.0
    assert aug.potential_deriv_coef[mask].item() == 1.0


def test_counterfactual_rows_have_D_zero():
    rows = [{"a": 0.5, "x": 1.5}]
    aug = build_augmented(rows, AdditiveShift(delta=0.7))
    # The shifted point (a=1.2, x=1.5) is a pure counterfactual: D=0, C=-1.
    mask = np.isclose(aug.features[:, 0], 1.2) & (aug.features[:, 1] == 1.5)
    assert mask.sum() == 1
    assert aug.is_original[mask].item() == 0.0
    assert aug.potential_deriv_coef[mask].item() == -1.0


def test_empty_rows():
    aug = build_augmented([], ATE())
    assert isinstance(aug, AugmentedDataset)
    assert aug.n_rows == 0
    assert aug.features.shape == (0, 2)


def test_origin_index_groups_rows():
    rows = [
        {"a": 0.0, "x": 1.0},
        {"a": 1.0, "x": 2.0},
        {"a": 0.0, "x": 3.0},
    ]
    aug = build_augmented(rows, ATE())
    # Each original row gets an origin_index equal to its position;
    # counterfactual rows for that row carry the same origin.
    counts = np.bincount(aug.origin_index)
    assert (counts > 0).all()
    assert counts.sum() == aug.features.shape[0]


def test_y_dependent_m_is_plumbed_through():
    """`build_augmented(rows, estimand, ys)` passes y into m(alpha)(z, y)."""
    tau = 0.0

    def m(alpha):
        def inner(z, y):
            indicator = 1.0 if y > tau else 0.0
            return indicator * (alpha(a=1, x=z["x"]) - alpha(a=0, x=z["x"]))
        return inner

    estimand = FiniteEvalEstimand(feature_keys=("a", "x"), m=m, name="upper-half-ate")
    rows = [{"a": 0.0, "x": 0.5}, {"a": 1.0, "x": 0.5}]

    # y > tau on row 0 → row 0 generates ATE-style augmentation.
    # y ≤ tau on row 1 → row 1 collapses to just the original observation.
    ys = [1.0, -1.0]
    aug = build_augmented(rows, estimand, ys)

    counts_per_row = {}
    for i in aug.origin_index.tolist():
        counts_per_row[i] = counts_per_row.get(i, 0) + 1
    assert counts_per_row[0] == 2  # two augmented rows for the active subject
    assert counts_per_row[1] == 1  # one (original-only) for the inactive subject


def test_y_length_mismatch_raises():
    rows = [{"a": 0.0, "x": 0.5}, {"a": 1.0, "x": 0.5}]
    with pytest.raises(ValueError, match="does not match"):
        build_augmented(rows, ATE(), ys=[1.0])
