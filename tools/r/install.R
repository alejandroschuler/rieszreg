#!/usr/bin/env Rscript
# Install the rieszreg R workspace via pak.
#
# Run from the repo root:
#
#   Rscript tools/r/install.R                 # install all 6 local R packages
#   Rscript tools/r/install.R rieszboost      # install rieszreg + just one sibling
#
# pak resolves the workspace-local DESCRIPTIONs together, so a sibling's
# `Imports: rieszreg (>= 0.0.1)` is satisfied by the local rieszreg directory
# rather than failing on a CRAN lookup.

WORKSPACE_PACKAGES <- c(
  "rieszreg",
  "rieszboost",
  "krrr",
  "forestriesz",
  "riesznet",
  "riesztree"
)

local_ref <- function(pkg) sprintf("local::./packages/%s/r/%s", pkg, pkg)

args <- commandArgs(trailingOnly = TRUE)
target <- if (length(args) == 0L) {
  WORKSPACE_PACKAGES
} else {
  unknown <- setdiff(args, WORKSPACE_PACKAGES)
  if (length(unknown)) {
    stop("Unknown package(s): ", paste(unknown, collapse = ", "),
         ". Known: ", paste(WORKSPACE_PACKAGES, collapse = ", "))
  }
  unique(c("rieszreg", args))
}

if (!requireNamespace("pak", quietly = TRUE)) {
  install.packages("pak", repos = "https://cran.r-project.org")
}

refs <- c(vapply(target, local_ref, character(1)), "any::testthat")

cat("Installing R workspace via pak:\n")
cat(paste0("  - ", refs, "\n"), sep = "")

pak::pkg_install(refs, ask = FALSE, upgrade = FALSE)

cat("\nDone. Test the workspace with:\n")
cat("  uv run --no-sync Rscript -e 'library(rieszreg); library(rieszboost); cat(\"ok\\n\")'\n")
