---
name: rieszreg-dev-environment
description: Operational rules for running tests, rendering docs, or executing code in the rieszreg meta-project. Triggers when running `pytest` across multiple packages, `quarto render` on the docs site, R parity tests, or anything that crosses package boundaries (rieszreg + rieszboost / krrr / forestriesz / riesznet). Especially important inside `.claude/worktrees/*` — worktrees of the rieszreg repo are NOT self-contained.
---

# rieszreg dev-environment rules

## Worktrees miss the sibling impl packages

The `rieszreg/` repo is one of **five sibling git repos** living under `/Users/aschuler/Desktop/RieszReg/`:

```
RieszReg/                  ← container, not a git repo
├── rieszreg/              ← the repo this worktree is of
├── rieszboost/            ← sibling repo
├── krrr/                  ← sibling repo
├── forestriesz/           ← sibling repo
├── riesznet/              ← sibling repo
├── docs/                  ← Quarto site (depends on ALL packages above)
└── .venv/                 ← shared editable installs of all five
```

A `git worktree` of `rieszreg` only contains the `rieszreg/` files. The siblings (`rieszboost/`, `krrr/`, `forestriesz/`, `riesznet/`) are **not** in the worktree — they're separate git repos that happen to live next to the main `rieszreg/` checkout.

CI works because `.github/workflows/docs.yml` and `.github/workflows/test.yml` use `actions/checkout` to clone each sibling into the workspace. Locally, the worktree does not.

## What this breaks inside a worktree

| Operation | Works in worktree? | Notes |
|---|---|---|
| `pytest rieszreg/python/tests` | ✓ | Only needs rieszreg itself |
| `pytest rieszboost/python/tests` (or krrr / forestriesz / riesznet) | ✗ | Sibling dir doesn't exist |
| `quarto render docs/` | ✗ | `docs/_setup.qmd` calls `pkgload::load_all("../rieszboost/r/rieszboost")` — fails on `pkgload_no_desc` |
| Editing `docs/*.qmd` content | ✓ | Editing is fine; rendering is not |
| Editing notation across all packages | ✗ | Worktree only sees rieszreg |
| Running R parity tests for any impl package | ✗ | `pkgload::load_all` of sibling fails |

## How to handle each case

**Need to render docs locally** (e.g. to verify a `_freeze` cache regeneration or test a new doc page):

→ Render from the main checkout at `/Users/aschuler/Desktop/RieszReg/`, not the worktree. Copy your edited `.qmd` over, render, then revert. Or: ask the user to `git pull` the main checkout to your branch's commit and render there.

**Need to test a sibling package** (rieszboost, krrr, forestriesz, riesznet):

→ Run from `/Users/aschuler/Desktop/RieszReg/<package>/` directly, not from the worktree. The shared `.venv` at `/Users/aschuler/Desktop/RieszReg/.venv/` has editable installs of all five packages.

**Need to make a cross-cutting change** (e.g. renaming `LinearFormEstimand` → `FiniteEvalEstimand` requires updates in rieszreg + every impl package):

→ Worktree handles only the rieszreg side. The impl-package changes need synchronized PRs in their own repos. Plan them as parallel PRs from the start. CI in `rieszreg` runs against `main` of the sibling repos by default — coordinate the merge order so siblings land first.

## Sanity-check before claiming a local verification worked

Before reporting "I verified locally", confirm the operation didn't silently fall back to a no-op:

- `quarto render`: did chunks actually execute, or did `_freeze` absorb them? Check with `ls docs/_freeze/<page>/` timestamps.
- Sibling-package test: did pytest collect tests, or did the directory not exist? Check the test count.
- R parity test: did `pkgload::load_all` succeed, or did it abort early?

If the worktree environment can't run the verification, **say so explicitly** — don't pretend the fix is verified. Push to a branch and let CI verify, or render from the main checkout.

## The `.venv`

The single Python venv at `/Users/aschuler/Desktop/RieszReg/.venv/` has editable installs of all five packages plus dev/doc deps (matplotlib, pandas, causaldata, xgboost, jax, etc.). When checking whether a Python module is available locally, use:

```sh
/Users/aschuler/Desktop/RieszReg/.venv/bin/python -c "import <module>"
```

The shared venv is **not** present inside `.claude/worktrees/<name>/`. Always reference it by absolute path.
