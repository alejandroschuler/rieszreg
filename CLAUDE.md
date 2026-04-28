# RieszReg (meta-project)

A family of packages for Riesz regression. Top-level coordinator for:

- [`rieszreg/`](rieszreg/) — meta-package: shared abstractions (`Estimand`, `LossSpec`, `RieszEstimator`, augmentation, diagnostics, `Backend` Protocol, testing utilities, R6 base class).
- [`rieszboost/`](rieszboost/) — gradient-boosting backend (Lee & Schuler 2025).
- [`krrr/`](krrr/) — kernel-ridge backend (Singh 2021).

Implementation packages depend on `rieszreg` and provide concrete backends. `RIESZREG_DESIGN.md` is the authoritative design + learner-package contract.

## Dependency graph

```
rieszreg (no deps on impl packages)
   ↑
   ├── rieszboost      (XGBoostBackend, SklearnBackend, RieszBooster)
   ├── krrr            (KernelRidgeBackend, kernels, solvers, KernelRieszRegressor)
   └── <future-pkg>    (its backend(s) + thin convenience class)
```

## Where things live

| Concern | Home |
|---|---|
| `Estimand` factories (ATE, ATT, …), `LinearForm`, `Tracer` | `rieszreg/python/rieszreg/estimands/` |
| `LossSpec` Protocol + 4 built-in losses | `rieszreg/python/rieszreg/losses/` |
| Augmentation engine | `rieszreg/python/rieszreg/augmentation.py` |
| `Backend` Protocol + predictor-loader registry | `rieszreg/python/rieszreg/backends/base.py` |
| `Diagnostics` base class + `diagnose()` | `rieszreg/python/rieszreg/diagnostics.py` |
| `RieszEstimator` (sklearn orchestrator) | `rieszreg/python/rieszreg/estimator.py` |
| Canonical DGPs, sklearn-conformance, parity helpers | `rieszreg/python/rieszreg/testing/` |
| Shared base R6 class | `rieszreg/r/rieszreg/R/rieszreg.R` |
| User guide (single Quarto site) | `docs/` |
| Reference papers | `reference/` |
| Pre-commit hook template | `.githooks/pre-commit` (each package keeps a copy) |
| CI workflow templates | `.github/workflows/` |
| Concrete backends + convenience subclasses | each implementation package's `python/<pkg>/backends/` and `python/<pkg>/<estimator>.py` |

## Adding a new estimand or loss

Goes in `rieszreg`, not the implementation packages.

1. Add the factory / class in `rieszreg/python/rieszreg/estimands/base.py` (or `losses/<name>.py`).
2. Wire into the relevant `__init__.py` and the `_FACTORY_REGISTRY` / `loss_from_spec` registry.
3. Update the docs page (`docs/estimands.qmd` or `docs/losses.qmd`) and the README.
4. Add a test under `rieszreg/python/tests/`.
5. Implementation packages whose backend supports the new feature should add a smoke test confirming round-trip works.

## Adding a new backend

Implement the `Backend` Protocol from `rieszreg/python/rieszreg/backends/base.py` in your package's `python/<pkg>/backends/`. Register the predictor loader on import:

```python
from rieszreg.backends import register_predictor_loader
register_predictor_loader("my-kind", MyPredictor.load)
```

Provide a convenience class subclassing `rieszreg.RieszEstimator`. Subclass `rieszreg::RieszEstimatorR6` for the R wrapper. See `RIESZREG_DESIGN.md` Part B for the full contract.

## Run tests

```sh
.venv/bin/python -m pytest rieszreg/python/tests -q
.venv/bin/python -m pytest rieszboost/python/tests -q
.venv/bin/python -m pytest krrr/python/tests -q

Rscript -e '
  Sys.setenv(RETICULATE_PYTHON = file.path(getwd(), ".venv/bin/python"))
  pkgload::load_all("rieszreg/r/rieszreg")
  pkgload::load_all("rieszboost/r/rieszboost")
  testthat::test_dir("rieszboost/r/rieszboost/tests/testthat")
  pkgload::load_all("krrr/r/krrr")
  testthat::test_dir("krrr/r/krrr/tests/testthat")
'
```

## Doc-tone rules (enforced by .githooks/pre-commit)

User-facing docs describe what's currently in the package, in plain instructive prose matching the [ngboost user guide](https://stanfordmlgroup.github.io/ngboost/intro.html). Two failure modes the hook checks for:

1. **No design-decision metacommentary.** Don't explain the API's negative space — what we removed, intentionally didn't build, or chose between. Just describe what the function does and how to use it.
2. **No AI-flavored hedge or editorial framing.** Avoid phrases like "the workhorse", "the right choice for almost every", "almost never needs tuning", "the natural way/API", "rather than reinvent". Avoid em-dashes peppered through prose. Sentences should be short (8–15 words on average), active voice.

## sklearn-first rule

Before writing any procedural code with loops, splits, grids, or folds, ask *"is there an sklearn way?"*. If yes, use it. Bespoke is reserved for things sklearn genuinely doesn't cover (the `LinearForm` tracer, the custom xgboost objective, the Bregman `LossSpec`).

## Status

- rieszreg: 64 Python tests passing (unit tests for tracer, losses, estimands, augmentation, diagnostics, orchestrator with stub backend, testing utilities).
- rieszboost: 109 Python tests + 11 R parity tests passing.
- krrr: 31 Python tests + 1 R parity test passing.
- Unified Quarto docs site renders all 15 pages.
- Pre-commit hook + CI workflow templates wired but not yet activated by `git config core.hooksPath` in any clone.
