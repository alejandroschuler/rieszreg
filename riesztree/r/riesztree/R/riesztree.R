#' riesztree: R wrapper for the Python riesztree library
#'
#' Mirrors the Python sklearn-style API. Configure once with
#' `use_python_riesztree()`, then construct a [RieszTreeRegressor] and call
#' `$fit(df)` / `$predict(df)`.
#'
#' Estimand and loss factories live in the shared `rieszreg` R package and are
#' re-exported from here for convenience.
#'
#' @keywords internal
"_PACKAGE"


.rt <- new.env(parent = emptyenv())


#' Configure the Python interpreter that holds the riesztree module.
#'
#' Call this once per session before any other riesztree function. Forwards
#' to `reticulate::use_python` / `reticulate::use_virtualenv` as appropriate.
#'
#' @param python Path to the Python interpreter or virtualenv directory.
#' @param required Whether reticulate should fail if the Python is unavailable.
#' @export
use_python_riesztree <- function(python = NULL, required = TRUE) {
  if (!is.null(python)) {
    if (dir.exists(python)) {
      reticulate::use_virtualenv(python, required = required)
    } else {
      reticulate::use_python(python, required = required)
    }
  }
  .rt$mod <- reticulate::import("riesztree", convert = FALSE)
  invisible(.rt$mod)
}


.module <- function() {
  if (is.null(.rt$mod)) {
    .rt$mod <- reticulate::import("riesztree", convert = FALSE)
  }
  .rt$mod
}


# ---- Main estimator (R6 subclass) -----------------------------------------

#' RieszTreeRegressor — single-tree Riesz regression.
#'
#' Subclass of [rieszreg::RieszEstimatorR6] that defaults the backend to a
#' single tree fit on the augmented Bregman-Riesz loss. Surfaces tree
#' hyperparameters (`max_depth`, `min_samples_split`, `min_samples_leaf`,
#' `max_leaves`, `growth_policy`, `pruning_alpha`, `early_stopping_rounds`,
#' `categorical_features`, ...) on the constructor.
#'
#' @export
RieszTreeRegressor <- R6::R6Class(
  "RieszTreeRegressor",
  inherit = rieszreg::RieszEstimatorR6,
  public = list(
    initialize = function(estimand,
                          loss = NULL,
                          max_depth = 8L,
                          min_samples_split = 20L,
                          min_samples_leaf = 10L,
                          max_leaves = 31L,
                          growth_policy = "depthwise",
                          pruning_alpha = 0.0,
                          early_stopping_rounds = NULL,
                          validation_fraction = 0.1,
                          categorical_features = NULL,
                          init = NULL,
                          random_state = 0L) {
      args <- list(
        estimand = estimand,
        max_depth = as.integer(max_depth),
        min_samples_split = as.integer(min_samples_split),
        min_samples_leaf = as.integer(min_samples_leaf),
        max_leaves = as.integer(max_leaves),
        growth_policy = growth_policy,
        pruning_alpha = pruning_alpha,
        validation_fraction = validation_fraction,
        random_state = as.integer(random_state)
      )
      if (!is.null(loss)) args$loss <- loss
      if (!is.null(init)) args$init <- init
      if (!is.null(early_stopping_rounds)) {
        args$early_stopping_rounds <- as.integer(early_stopping_rounds)
      }
      if (!is.null(categorical_features)) {
        args$categorical_features <- as.integer(categorical_features)
      }
      py_object <- do.call(.module()$RieszTreeRegressor, args)
      super$initialize(py_object = py_object, estimand = estimand)
    }
  )
)


#' Load a fitted RieszTreeRegressor from a directory written by `$save()`.
#'
#' For built-in estimands, fully reconstructs the estimand from the metadata.
#' For custom estimands (Python-only), pass `estimand=` explicitly.
#' @param path Directory path.
#' @param estimand Optional user-supplied `Estimand` (required for custom m).
#' @export
load_riesz_tree_regressor <- function(path, estimand = NULL) {
  args <- list(path = path)
  if (!is.null(estimand)) args$estimand <- estimand
  py_obj <- do.call(.module()$RieszTreeRegressor$load, args)
  rt <- RieszTreeRegressor$new(estimand = py_obj$estimand)
  rt$py <- py_obj
  rt$estimand <- py_obj$estimand
  rt
}


# Estimand and loss factories are re-exported from rieszreg via NAMESPACE.
