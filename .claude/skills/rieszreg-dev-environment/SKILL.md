---
name: rieszreg-dev-environment
description: Operational rules for running tests, rendering docs, or executing code in the rieszreg meta-project. Triggers when running `pytest` across multiple packages, `quarto render` on the docs site, R parity tests, or anything that crosses package boundaries (rieszreg + rieszboost / krrr / forestriesz / riesznet / riesztree). Especially important inside `.claude/worktrees/*` — worktree clones do not contain the shared `.venv`.
---

# rieszreg dev-environment rules

## The repo is a uv-workspace monorepo

`/Users/aschuler/Desktop/RieszReg/` is a single git repo (`github.com/rieszreg/rieszreg`). All six packages live under `packages/`:

```
RieszReg/
├── packages/
│   ├── rieszreg/                 ← meta-package (Python + R)
│   ├── rieszboost/               ← XGBoost backend
│   ├── krrr/                     ← kernel ridge backend
│   ├── forestriesz/              ← random forest backend
│   ├── riesznet/                 ← neural-net backend
│   └── riesztree/                ← single-tree backend
├── docs/                         ← Quarto user guide
├── tools/r/install.R             ← R workspace installer (pak)
├── pyproject.toml                ← uv workspace root
└── .venv/                        ← editable installs of all 6 Python packages
```

The Python side is a uv workspace: `uv sync --all-packages` produces a single `.venv` with editable installs of every package. The R side uses pak to install the same 6 packages from their workspace-local paths.

## Running things from a worktree

A `git worktree` checkout (e.g. `.claude/worktrees/<name>/`) contains the full source tree but **not** the shared `.venv`. The venv at `/Users/aschuler/Desktop/RieszReg/.venv/` belongs to the main checkout.

| Operation | Works in worktree? | Notes |
|---|---|---|
| `pytest packages/<any>/python/tests` | ✓ if you set up a venv | `uv sync --all-packages` creates one inside the worktree |
| `quarto render docs/` | ✓ if you set up a venv | docs `_setup.qmd` reads from the active venv via reticulate |
| Running R parity tests | ✓ if R workspace is installed | `Rscript tools/r/install.R` installs all 6 R packages |
| Editing notation across all packages | ✓ | The full source tree is present |

**Two ways to run:**

1. **Use the main checkout's `.venv` directly** (fastest, no install). Reference it by absolute path:
   ```sh
   /Users/aschuler/Desktop/RieszReg/.venv/bin/python -m pytest packages/rieszreg/python/tests -q
   ```
   The editable installs there resolve `import rieszreg` etc., regardless of which worktree you're in. Files paths inside `.venv` resolve to the main checkout, so this verifies code from the main checkout, not the worktree.

2. **Build a fresh venv in the worktree** (verifies worktree code). From the worktree root:
   ```sh
   uv sync --all-packages --all-extras
   uv run pytest packages/rieszreg/python/tests -q
   ```

For **R parity tests in the worktree**, use option 2 (the R packages need to point at the worktree's editable Python installs):

```sh
uv sync --all-packages --all-extras
Rscript tools/r/install.R              # installs all 6 R packages via pak
RETICULATE_PYTHON=$(uv run python -c 'import sys; print(sys.executable)') \
  Rscript -e '
    library(rieszreg); library(rieszboost)
    testthat::test_dir("packages/rieszboost/r/rieszboost/tests/testthat")
  '
```

## R workspace install (one-time setup)

`tools/r/install.R` installs all six R packages plus testthat via pak. It uses `local::./packages/<pkg>/r/<pkg>` refs, so pak resolves cross-package deps against the workspace-local DESCRIPTIONs (rather than failing on a CRAN lookup of the not-yet-published `rieszreg` package).

```sh
Rscript tools/r/install.R                # all six packages
Rscript tools/r/install.R rieszboost     # rieszreg + just rieszboost
```

After installation, R sessions can load packages directly:

```r
library(rieszreg)
library(rieszboost)
testthat::test_dir("packages/rieszboost/r/rieszboost/tests/testthat")
```

No `pkgload::load_all` dance needed. The CI rtests job uses the same pak-based workspace install through `r-lib/actions/setup-r-dependencies` with explicit `local::` refs.

When a sibling's DESCRIPTION changes (new Imports, version bump), re-run `Rscript tools/r/install.R <sibling>` to pick up the changes.

## Sanity-check before claiming a local verification worked

Before reporting "I verified locally", confirm the operation didn't silently fall back to a no-op:

- `quarto render`: did chunks actually execute, or did `_freeze` absorb them? Check with `ls docs/_freeze/<page>/` timestamps.
- pytest: did it collect tests, or skip the directory? Check the test count in the summary.
- R parity test: did `library(<pkg>)` succeed, or fall back to a stale install? `find.package("<pkg>")` shows where R found it.

If a verification can't be run inside the worktree (e.g. you don't want to install the R workspace just to check a typo fix), **say so explicitly** — don't claim verification you didn't do. Push to a branch and let CI run, or run from the main checkout.
