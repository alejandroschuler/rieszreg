# BENCH_BASELINE.md

Locked baseline for the **pure-Python splitter path** (riesztree v0.0.1).
Numbers below are the reference every later optimisation phase must beat.
Re-run with:

```sh
cd /tmp   # cwd must NOT contain a `rieszreg` directory
/Users/aschuler/Desktop/RieszReg/.venv/bin/python \
  /Users/aschuler/Desktop/RieszReg/riesztree/python/benchmarks/bench_fit.py \
  --grid small \
  --label v0.0.1-pure-python \
  --max-fit-seconds 120
```

Hardware / OS for the recorded numbers:

- macOS Darwin 24.6.0, Apple Silicon
- Python 3.13.5
- Single fit per cell, no replication.

## Small grid (`--grid small`, 32 configs)

`{loss} × {n} × {p} × {max_depth} × {growth_policy}` =
`{squared, kl} × {1000, 10000} × {5, 20} × {8, 16} × {depthwise, leafwise}`.

Predict is timed on a fixed 10 000-row test set (`Config.n_predict`).
RMSE is α̂ vs. the closed-form Riesz representer on the test set.

| loss | n | p | max_depth | growth | fit (s) | predict 10k (s) | leaves | depth | rmse |
|---|---|---|---|---|---|---|---|---|---|
| squared | 1000 | 5 | 8 | depthwise | 0.15 | 0.49 | 58 | 8 | 0.90 |
| squared | 1000 | 5 | 8 | leafwise | 0.15 | 0.49 | 58 | 8 | 0.90 |
| squared | 1000 | 5 | 16 | depthwise | 0.16 | 0.49 | 78 | 12 | 0.93 |
| squared | 1000 | 5 | 16 | leafwise | 0.16 | 0.49 | 78 | 12 | 0.93 |
| squared | 1000 | 20 | 8 | depthwise | 0.58 | 1.75 | 49 | 8 | 1.11 |
| squared | 1000 | 20 | 8 | leafwise | 0.55 | 1.69 | 49 | 8 | 1.11 |
| squared | 1000 | 20 | 16 | depthwise | 0.60 | 1.70 | 74 | 13 | 1.14 |
| squared | 1000 | 20 | 16 | leafwise | 0.60 | 1.75 | 74 | 13 | 1.14 |
| squared | 10000 | 5 | 8 | depthwise | 1.75 | 0.49 | 121 | 8 | 0.57 |
| squared | 10000 | 5 | 8 | leafwise | 1.79 | 0.53 | 121 | 8 | 0.57 |
| squared | 10000 | 5 | 16 | depthwise | 2.44 | 0.50 | 615 | 16 | 0.97 |
| squared | 10000 | 5 | 16 | leafwise | 2.53 | 0.51 | 615 | 16 | 0.97 |
| squared | 10000 | 20 | 8 | depthwise | 6.40 | 1.75 | 104 | 8 | 0.80 |
| squared | 10000 | 20 | 8 | leafwise | 6.31 | 1.73 | 104 | 8 | 0.80 |
| squared | 10000 | 20 | 16 | depthwise | 9.07 | 1.77 | 547 | 16 | 1.25 |
| squared | 10000 | 20 | 16 | leafwise | 9.19 | 1.71 | 547 | 16 | 1.25 |
| kl | 1000 | 5 | 8 | depthwise | 0.12 | 0.49 | 31 | 8 | 0.63 |
| kl | 1000 | 5 | 8 | leafwise | 0.13 | 0.49 | 31 | 8 | 0.63 |
| kl | 1000 | 5 | 16 | depthwise | 0.13 | 0.51 | 41 | 12 | 0.64 |
| kl | 1000 | 5 | 16 | leafwise | 0.13 | 0.51 | 41 | 12 | 0.64 |
| kl | 1000 | 20 | 8 | depthwise | 0.45 | 1.77 | 25 | 8 | 0.78 |
| kl | 1000 | 20 | 8 | leafwise | 0.44 | 1.69 | 25 | 8 | 0.78 |
| kl | 1000 | 20 | 16 | depthwise | 0.47 | 1.71 | 37 | 12 | 0.79 |
| kl | 1000 | 20 | 16 | leafwise | 0.47 | 1.70 | 37 | 12 | 0.79 |
| kl | 10000 | 5 | 8 | depthwise | 1.39 | 0.53 | 64 | 8 | 0.41 |
| kl | 10000 | 5 | 8 | leafwise | 1.43 | 0.50 | 64 | 8 | 0.41 |
| kl | 10000 | 5 | 16 | depthwise | 1.86 | 0.50 | 333 | 16 | 0.72 |
| kl | 10000 | 5 | 16 | leafwise | 1.84 | 0.50 | 333 | 16 | 0.72 |
| kl | 10000 | 20 | 8 | depthwise | 5.23 | 1.76 | 62 | 8 | 0.55 |
| kl | 10000 | 20 | 8 | leafwise | 5.19 | 1.78 | 62 | 8 | 0.55 |
| kl | 10000 | 20 | 16 | depthwise | 6.72 | 1.72 | 334 | 16 | 0.94 |
| kl | 10000 | 20 | 16 | leafwise | 6.72 | 1.73 | 334 | 16 | 0.94 |

### What jumps out

- `predict` on 10 k rows dominates `fit` for `n=1000`: a Python tree-walk
  cost of ~50 µs/row (p=5) to ~170 µs/row (p=20). That's the predict
  bottleneck the plan removes in Phase 3.
- `fit` scales roughly `O(n × p)` with a steep Python-loop constant:
  ≈ 10× from `(n=1k, p=5)` to `(n=10k, p=20)` at fixed depth.
- Going from `max_depth=8` to `16` adds ~30–50% to fit time — leaves
  multiply but each leaf scan shrinks; Python overhead dominates.
- `depthwise` and `leafwise` are essentially interchangeable in wall time
  on these grids, since both are Python loops over the same per-leaf work.
- KL is faster than Squared at matched (n, p, depth) because the TSM
  augmentation produces ~`n` augmented rows (one per original) while the
  ATE augmentation produces ~`2n` (one per treatment level).

## Headline target grid (`--grid full`)

The "full" grid (`{n ∈ 10⁴, 10⁵, 10⁶} × {p ∈ 10, 50} × {max_depth ∈ 8, 16, ∞} × {depthwise, leafwise}`, 72 configs) is what the optimised paths aim at. On the pure-Python path most cells of this grid don't finish in
reasonable wall time; baseline numbers there are not recorded. Reaching
that regime is the *point* of the optimisation phases.

| Phase | What it ships | Headline target ratio (vs. v0.0.1 pure-Python) |
|---|---|---|
| Phase 3 | flat-array tree + Cython predict | `predict 10k` ↓ ~10× (≤ 50 ms target on the slowest baseline cell) |
| Phase 4 | `splitter='exact'` (Cython) | `fit` ↓ ≥ 10× on `(n=10⁵, p=20, depth=10)` |
| Phase 6 | `splitter='hist'` + `max_bins` | additional ~5× on Phase 4 at `n=10⁶` |
| End-state | Phases 4 + 6 + 8 + 9 | `fit` ↓ ≥ 30× exact / ≥ 100× hist on `(n=10⁵, p=20, depth=10)`; deep trees (`max_depth=None`) at `n=10⁵` finish in seconds |

## Memory ceiling

Peak resident set during `fit` on `(n=10⁶, p=50)` must stay under
`4 ×` the augmented-data size. Not measured at v0.0.1 baseline (the config
does not finish on the pure-Python path); becomes a gating check from
Phase 4 onward.

## Method notes

- Each cell is a single fit. Wall-time variance across repeated runs is
  small relative to the 10×–100× ratios we target. Phases reporting sub-2×
  improvements should add replication.
- `rmse` is recorded as a sanity check that timing comparisons are made on
  equally-fit trees. Cross-phase RMSE drift larger than ~5% is treated
  as a regression: it means the optimisation also changed the algorithm.
- `predict` time is nearly independent of `n_train`; it scales with
  `n_test × p × depth` and is dominated by the Python tree-walk
  (`predict_array` in [python/riesztree/tree.py](python/riesztree/tree.py)).
- Bench raw CSVs under `python/benchmarks/results/` are gitignored; only
  this file (and the headline numbers it locks in) is tracked.
