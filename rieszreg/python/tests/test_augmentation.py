"""Tests for build_augmented + AugmentedDataset."""

from __future__ import annotations

import numpy as np

from rieszreg import ATE, AdditiveShift, AugmentedDataset, build_augmented


def test_ate_augmentation_shape():
    rows = [{"a": 0.0, "x": 1.0}, {"a": 1.0, "x": 2.0}]
    aug = build_augmented(rows, ATE())
    assert aug.n_rows == 2
    # Each row contributes itself (a=1, b=0) plus 2 counterfactual points
    # (a=0, b=±2) at (1, x) and (0, x). For row 0 (a=0), the original row
    # coincides with the (a=0, x=1.0) counterfactual point so they merge.
    # That merge means n_aug = 2 + 2 = 4 (not 6).
    assert aug.features.shape == (4, 2)
    assert aug.a.shape == (4,)
    assert aug.b.shape == (4,)
    # Origin index points back at the source row.
    assert aug.origin_index.shape == (4,)
    assert set(aug.origin_index.tolist()) == {0, 1}


def test_original_rows_have_a_one_b_zero():
    rows = [{"a": 0.5, "x": 1.5}]
    # Use AdditiveShift so the counterfactual point is distinct from the original.
    aug = build_augmented(rows, AdditiveShift(delta=0.7))
    # Pick out the row where features match (0.5, 1.5).
    mask = (aug.features[:, 0] == 0.5) & (aug.features[:, 1] == 1.5)
    assert mask.sum() == 1  # original row distinct from counterfactual
    # On the original row, a=1, b=-2*coef where coef is from m at this point.
    # m(alpha)(z) = alpha(a + delta, x) - alpha(a, x). At a=0.5: coef on
    # alpha(a=0.5, x=1.5) is -1, so b = +2. Plus the natural a=1 contribution.
    assert aug.a[mask].item() == 1.0
    assert aug.b[mask].item() == 2.0


def test_counterfactual_rows_have_a_zero():
    rows = [{"a": 0.5, "x": 1.5}]
    aug = build_augmented(rows, AdditiveShift(delta=0.7))
    # The shifted point (a=1.2, x=1.5) is a pure counterfactual: a=0, b=-2.
    mask = np.isclose(aug.features[:, 0], 1.2) & (aug.features[:, 1] == 1.5)
    assert mask.sum() == 1
    assert aug.a[mask].item() == 0.0
    assert aug.b[mask].item() == -2.0


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
