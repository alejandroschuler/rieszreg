"""Speed comparison: AugForestRieszRegressor vs fast random-forest libraries.

The augmented-Riesz problem doesn't have an exact equivalent in standard
random-forest libraries (their objective is `(y - α)²`, not the
augmented `D·α² + 2C·α`), so this bench compares **infrastructure
speed** rather than end-to-end task speed:

  - AugForestRieszRegressor: fit a forest on the augmented (n_aug × p)
    data, with SquaredLoss. n_aug = k·n where k is the per-row
    augmentation factor (2 for ATE).
  - sklearn RandomForestRegressor: fit a regression forest on a fresh
    random regression with the same n_aug × p shape and the same
    n_estimators / max_depth / max_samples.
  - sklearn ExtraTreesRegressor: random-threshold RF variant (typically
    faster than the exact-split baseline).
  - LightGBM (boosting='rf'): histogram-based bagged trees.
  - XGBoost (num_parallel_tree=n_estimators, n_estimators=1): the
    XGBoost analog of an RF.

The web survey of fast CPU random forest implementations (May 2026)
also lists scikit-learn-intelex (Intel-optimized sklearn), cuML
(NVIDIA GPU), and ThunderGBM (GPU). Those aren't installed in this
workspace; if you have a GPU, cuML's RandomForestRegressor would be
the clear winner (20-45x over sklearn per NVIDIA benchmarks).

Run from the workspace root::

    uv run python packages/forestriesz/python/benchmarks/bench_aug_forest_compare.py
"""
from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rieszreg import ATE, SquaredLoss
from forestriesz import AugForestRieszRegressor


@dataclass
class Result:
    library: str
    n_aug: int
    p: int
    n_estimators: int
    max_depth: int | None
    n_jobs: int
    fit_seconds: float
    predict_seconds: float = 0.0


def _make_ate_data(n: int, p: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, p))
    pi = 1.0 / (1.0 + np.exp(-0.5 * X[:, 0]))
    a = (rng.uniform(0, 1, size=n) < pi).astype(float)
    return pd.DataFrame({**{f"x{j}": X[:, j] for j in range(p)}, "a": a})


def _make_regression_data(n_aug: int, p: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_aug, p))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0.0, 0.5, size=n_aug)
    return X, y


def _fit_aug_forest(
    n: int, p: int, n_estimators: int, depth: int, n_jobs: int, seed: int = 0,
    splitter: str = "exact",
) -> Result:
    df = _make_ate_data(n, p, seed)
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(p)))
    est = AugForestRieszRegressor(
        estimand=estimand,
        loss=SquaredLoss(),
        n_estimators=n_estimators,
        max_depth=depth,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=True,
        max_samples=None,
        n_jobs=n_jobs,
        random_state=seed,
        splitter=splitter,
    )
    gc.collect()
    t0 = time.perf_counter()
    est.fit(df)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = est.predict(df)
    pred_s = time.perf_counter() - t0
    n_aug = 2 * n  # ATE: original + 1 counterfactual eval per row
    return Result(
        f"AugForestRieszRegressor({splitter})", n_aug, p, n_estimators, depth, n_jobs, fit_s, pred_s
    )


def _fit_sklearn_rf(
    n_aug: int, p: int, n_estimators: int, depth: int, n_jobs: int, seed: int = 0
) -> Result:
    from sklearn.ensemble import RandomForestRegressor

    X, y = _make_regression_data(n_aug, p, seed)
    est = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=depth,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=True,
        n_jobs=n_jobs,
        random_state=seed,
    )
    gc.collect()
    t0 = time.perf_counter()
    est.fit(X, y)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = est.predict(X)
    pred_s = time.perf_counter() - t0
    return Result(
        "sklearn-RandomForestRegressor",
        n_aug, p, n_estimators, depth, n_jobs, fit_s, pred_s,
    )


def _fit_sklearn_extra_trees(
    n_aug: int, p: int, n_estimators: int, depth: int, n_jobs: int, seed: int = 0
) -> Result:
    from sklearn.ensemble import ExtraTreesRegressor

    X, y = _make_regression_data(n_aug, p, seed)
    est = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=depth,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=True,
        n_jobs=n_jobs,
        random_state=seed,
    )
    gc.collect()
    t0 = time.perf_counter()
    est.fit(X, y)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = est.predict(X)
    pred_s = time.perf_counter() - t0
    return Result(
        "sklearn-ExtraTreesRegressor",
        n_aug, p, n_estimators, depth, n_jobs, fit_s, pred_s,
    )


def _fit_lightgbm_rf(
    n_aug: int, p: int, n_estimators: int, depth: int | None, n_jobs: int, seed: int = 0
) -> Result | None:
    try:
        import lightgbm as lgb
    except ImportError:
        return None

    X, y = _make_regression_data(n_aug, p, seed)
    # LightGBM uses max_depth=-1 for "no limit" and num_leaves to cap the
    # leaf count. With unlimited depth, set num_leaves large enough that
    # min_child_samples=1 binds first.
    lgb_max_depth = -1 if depth is None else depth
    # When max_depth is unbounded, cap leaves at min(n_aug, 2**13) so
    # the splitter doesn't waste time looking for leaves it can't make
    # under min_child_samples=1.
    lgb_num_leaves = min(n_aug, 8192) if depth is None else 2 ** min(depth, 13)
    est = lgb.LGBMRegressor(
        boosting_type="rf",
        n_estimators=n_estimators,
        max_depth=lgb_max_depth,
        num_leaves=lgb_num_leaves,
        min_child_samples=1,
        bagging_fraction=0.8,
        bagging_freq=1,
        feature_fraction=1.0,
        n_jobs=n_jobs,
        random_state=seed,
        verbose=-1,
    )
    gc.collect()
    t0 = time.perf_counter()
    est.fit(X, y)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = est.predict(X)
    pred_s = time.perf_counter() - t0
    return Result(
        "lightgbm(rf)", n_aug, p, n_estimators, depth, n_jobs, fit_s, pred_s
    )


def _fit_xgboost_rf(
    n_aug: int, p: int, n_estimators: int, depth: int | None, n_jobs: int, seed: int = 0
) -> Result | None:
    try:
        import xgboost as xgb
    except ImportError:
        return None

    X, y = _make_regression_data(n_aug, p, seed)
    # XGBoost uses max_depth=0 with grow_policy='lossguide' for "no limit",
    # but 'hist' tree method requires a finite max_depth. Use a large value
    # so min_child_weight=1 binds first.
    xgb_max_depth = 32 if depth is None else depth
    est = xgb.XGBRegressor(
        n_estimators=1,
        num_parallel_tree=n_estimators,
        max_depth=xgb_max_depth,
        tree_method="hist",
        subsample=0.8,
        colsample_bynode=1.0,
        learning_rate=1.0,
        n_jobs=n_jobs,
        random_state=seed,
        verbosity=0,
    )
    gc.collect()
    t0 = time.perf_counter()
    est.fit(X, y)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = est.predict(X)
    pred_s = time.perf_counter() - t0
    return Result(
        "xgboost(rf)", n_aug, p, n_estimators, depth, n_jobs, fit_s, pred_s
    )


def run(grid, n_jobs: int) -> pd.DataFrame:
    rows: list[Result] = []
    for n, p, n_estimators, depth in grid:
        n_aug = 2 * n
        for splitter in ("exact", "hist"):
            r = _fit_aug_forest(n, p, n_estimators, depth, n_jobs, splitter=splitter)
            rows.append(r)
            print(
                f"  n={n} p={p} n_est={n_estimators} d={depth if depth is not None else 'None':>4} "
                f"{r.library:38s}: fit={r.fit_seconds:7.3f}s pred={r.predict_seconds:6.3f}s"
            )
        for fn in (
            _fit_sklearn_rf,
            _fit_sklearn_extra_trees,
            _fit_lightgbm_rf,
            _fit_xgboost_rf,
        ):
            r = fn(n_aug, p, n_estimators, depth, n_jobs)
            if r is None:
                continue
            rows.append(r)
            print(
                f"  n={n} p={p} n_est={n_estimators} d={depth if depth is not None else 'None':>4} "
                f"{r.library:38s}: fit={r.fit_seconds:7.3f}s pred={r.predict_seconds:6.3f}s"
            )
    return pd.DataFrame([r.__dict__ for r in rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=-1, help="Cores per fit.")
    parser.add_argument("--quick", action="store_true", help="Run a fast subset.")
    args = parser.parse_args()

    if args.quick:
        grid = [
            (1000, 5, 100, None),
            (5000, 5, 100, None),
        ]
    else:
        grid = [
            # max_depth=None matches sklearn RandomForestRegressor's default —
            # grow each tree until leaves saturate min_samples_leaf=1.
            (1000, 5, 100, None),
            (5000, 5, 100, None),
            (5000, 20, 100, None),
            (5000, 5, 500, None),
            # Shallow-tree comparison kept for the depth-bias check.
            (5000, 5, 100, 8),
        ]

    # Warm-up: prime joblib worker pool, lightgbm/xgboost JIT, and
    # the riesztree Cython kernels so the first timed row isn't
    # contaminated by import / fork costs.
    print("warming up...")
    warm_grid = [(500, 5, 50, 4)]
    _ = run(warm_grid, n_jobs=args.n_jobs)

    print(f"\n--- AugForest vs fast random forests (n_jobs={args.n_jobs}) ---")
    df = run(grid, n_jobs=args.n_jobs)

    print("\n--- summary ---")
    pd.set_option("display.float_format", lambda v: f"{v:8.3f}")
    print(
        df[["library", "n_aug", "p", "n_estimators", "max_depth", "fit_seconds", "predict_seconds"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
