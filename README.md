# RieszReg

A family of packages for **Riesz regression** — direct estimation of the Riesz representer α₀ of a linear functional θ(P) = E[m(Z, g₀)], the building block of one-step, TMLE, and DML estimators in semiparametric inference.

## Layout

```
RieszReg/
├── rieszreg/         # meta-package: shared abstractions
│   ├── python/       # Estimand, LossSpec, augmentation, RieszEstimator, testing utilities
│   └── r/rieszreg/   # shared R6 base class, estimand + loss factories
├── rieszboost/       # gradient-boosting backends (Lee & Schuler 2025)
│   ├── python/       # XGBoostBackend, SklearnBackend, RieszBooster
│   └── r/rieszboost/ # R6 wrapper subclassing rieszreg::RieszEstimatorR6
├── krrr/             # kernel ridge backend (Singh 2021)
│   ├── python/       # KernelRidgeBackend, kernels, solvers, KernelRieszRegressor
│   └── r/krrr/       # R6 wrapper subclassing rieszreg::RieszEstimatorR6
├── forestriesz/      # random-forest backend (Chernozhukov et al. ICML 2022)
│   ├── python/       # ForestRieszBackend, ForestRieszRegressor, default_riesz_features
│   └── r/forestriesz/ # R6 wrapper subclassing rieszreg::RieszEstimatorR6
├── docs/             # unified Quarto user guide (sklearn-style sectioning)
├── reference/        # arXiv paper index, shared across packages
├── .githooks/        # canonical pre-commit hook (living-doc rule + tone lint)
├── .github/workflows # canonical CI (pytest + R parity, docs deploy)
└── rieszreg/DESIGN.md  # meta-package design + learner-package contract
```

The user guide is a single Quarto site at [`docs/`](docs/) — sklearn-style, not per-package — covering Concepts, Get started, Usage, Backends (one sub-page per backend package), R interface, Developing, and References.

## Install

The four packages live in sibling GitHub repos:
[rieszreg](https://github.com/rieszreg/rieszreg) (this repo, the meta-package + unified docs),
[rieszboost](https://github.com/rieszreg/rieszboost),
[krrr](https://github.com/rieszreg/krrr),
[forestriesz](https://github.com/rieszreg/forestriesz).
Clone them as siblings into a parent directory; the docs builds and CI assume
that layout.

```sh
mkdir RieszReg && cd RieszReg
git clone https://github.com/rieszreg/rieszreg.git
git clone https://github.com/rieszreg/rieszboost.git
git clone https://github.com/rieszreg/krrr.git
git clone https://github.com/rieszreg/forestriesz.git
python3 -m venv .venv
.venv/bin/pip install -e rieszreg/python
.venv/bin/pip install -e rieszboost/python      # gradient-boosting backend
.venv/bin/pip install -e krrr/python            # kernel-ridge backend
.venv/bin/pip install -e forestriesz/python     # random-forest backend
```

`rieszboost`'s `XGBoostBackend` requires OpenMP; on macOS, `brew install libomp` once.

## Quickstart

```python
from rieszboost import RieszBooster
from rieszreg import ATE

booster = RieszBooster(estimand=ATE(), n_estimators=200, early_stopping_rounds=20,
                       validation_fraction=0.2)
booster.fit(df)
alpha_hat = booster.predict(df)
```

Or compose explicitly:

```python
from rieszreg import RieszEstimator, ATE, SquaredLoss
from krrr import KernelRidgeBackend, Gaussian

est = RieszEstimator(
    estimand=ATE(), loss=SquaredLoss(),
    backend=KernelRidgeBackend(kernel=Gaussian(length_scale="median")),
)
est.fit(df)
```

## Status

- **rieszreg** v0.0.1 — feature-complete: estimand machinery, four Bregman losses, augmentation engine, both `Backend` (augmentation-style) and `MomentBackend` (moment-style) Protocols with orchestrator dispatch, RieszEstimator, base R6 class, testing utilities. 68 Python tests passing.
- **rieszboost** v0.0.1 — sklearn-compatible `RieszBooster` with `XGBoostBackend` (default) and `SklearnBackend`; 110 Python tests + 11 R parity tests passing.
- **krrr** v0.0.1 — sklearn-compatible `KernelRieszRegressor`; four solvers (direct, Nyström-CG, RFF, optional Falkon); 36 Python tests + 1 R parity test passing.
- **forestriesz** v0.0.1 — sklearn-compatible `ForestRieszRegressor` on EconML's `BaseGRF`; locally constant + locally linear sieve fits; honest-split `predict_interval`; 34 Python tests + 1 R parity test passing.

## Tests

```sh
.venv/bin/python -m pytest rieszreg/python/tests -q
.venv/bin/python -m pytest rieszboost/python/tests -q
.venv/bin/python -m pytest krrr/python/tests -q
.venv/bin/python -m pytest forestriesz/python/tests -q

Rscript -e '
  RETICULATE_PYTHON <- file.path(getwd(), ".venv/bin/python")
  Sys.setenv(RETICULATE_PYTHON = RETICULATE_PYTHON)
  pkgload::load_all("rieszreg/r/rieszreg")
  pkgload::load_all("rieszboost/r/rieszboost")
  testthat::test_dir("rieszboost/r/rieszboost/tests/testthat")
  pkgload::load_all("krrr/r/krrr")
  testthat::test_dir("krrr/r/krrr/tests/testthat")
  pkgload::load_all("forestriesz/r/forestriesz")
  testthat::test_dir("forestriesz/r/forestriesz/tests/testthat")
'
```

## Contributing a new learner package

`RIESZREG_DESIGN.md` (Part B) is the contract: depend on `rieszreg`, implement either the `Backend` Protocol (augmentation-style — for kernel ridge, gradient boosting) or the `MomentBackend` Protocol (moment-style — for random forests, neural nets), satisfy the sklearn-conformance subset, contribute docs pages to `docs/`, follow the doc-tone and living-doc rules. The pre-commit hook at `.githooks/pre-commit` enforces the doc-tone and API-changes-update-docs rules.

## References

The meta-project's [`reference/`](reference/) directory indexes the foundational papers (Lee & Schuler 2025, Chernozhukov et al., Singh, Hines & Miles, Kato, van der Laan et al.) with arXiv IDs and a refetch script.
