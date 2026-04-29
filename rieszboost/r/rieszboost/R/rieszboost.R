#' rieszboost: R wrapper for the Python rieszboost library
#'
#' Mirrors the Python sklearn-style API. Configure once with
#' `use_python_rieszboost()`, then construct a [RieszBooster] and call
#' `$fit(df)` / `$predict(df)`.
#'
#' Estimand and loss factories live in the shared `rieszreg` R package and
#' are re-exported from here for convenience.
#'
#' @keywords internal
"_PACKAGE"


.rb <- new.env(parent = emptyenv())


#' Configure the Python interpreter that holds the rieszboost module.
#'
#' Call this once per session before any other rieszboost function. Forwards
#' to `reticulate::use_python` / `reticulate::use_virtualenv` as appropriate.
#'
#' @param python Path to the Python interpreter or virtualenv directory.
#' @param required Whether reticulate should fail if the Python is unavailable.
#' @export
use_python_rieszboost <- function(python = NULL, required = TRUE) {
  if (!is.null(python)) {
    if (dir.exists(python)) {
      reticulate::use_virtualenv(python, required = required)
    } else {
      reticulate::use_python(python, required = required)
    }
  }
  .rb$mod <- reticulate::import("rieszboost", convert = FALSE)
  invisible(.rb$mod)
}


.module <- function() {
  if (is.null(.rb$mod)) {
    .rb$mod <- reticulate::import("rieszboost", convert = FALSE)
  }
  .rb$mod
}


# ---- Backends (rieszboost-specific) ----

#' Default backend: data augmentation + xgboost custom objective.
#' @param hessian_floor Lower bound on per-row Hessian (default 2.0).
#' @param gradient_only If TRUE, disable second-order Newton step (Friedman 2001 mode).
#' @export
XGBoostBackend <- function(hessian_floor = 2.0, gradient_only = FALSE) {
  .module()$XGBoostBackend(hessian_floor = hessian_floor,
                           gradient_only = gradient_only)
}

#' Slow general backend: Friedman gradient boosting with arbitrary
#' sklearn-compatible base learner.
#' @param base_learner_factory A zero-arg R function (or Python callable)
#'   returning a fresh sklearn estimator each round.
#' @export
SklearnBackend <- function(base_learner_factory) {
  .module()$SklearnBackend(base_learner_factory = base_learner_factory)
}


# ---- Main estimator (R6 subclass) ----

#' RieszBooster — gradient-boosted estimator for the Riesz representer.
#'
#' Subclass of [rieszreg::RieszEstimatorR6] that defaults the backend to
#' `XGBoostBackend()` and surfaces xgboost-specific hyperparameters
#' (`max_depth`, `reg_lambda`, `subsample`) on the constructor.
#'
#' @export
RieszBooster <- R6::R6Class(
  "RieszBooster",
  inherit = rieszreg::RieszEstimatorR6,
  public = list(
    #' @param estimand An `Estimand` returned by [rieszreg::ATE()] etc.
    #' @param backend Backend object; default `XGBoostBackend()`.
    #' @param loss Loss spec; default `SquaredLoss()`.
    #' @param n_estimators,learning_rate,max_depth,reg_lambda,subsample
    #'   Hyperparameters.
    #' @param early_stopping_rounds,validation_fraction Early-stopping config.
    #' @param init Initial alpha (NULL → loss default; "m1" → mean of m(z,1); float → that value).
    #' @param random_state Random seed.
    initialize = function(estimand,
                          backend = NULL, loss = NULL,
                          n_estimators = 200L, learning_rate = 0.05,
                          max_depth = 4L, reg_lambda = 1.0, subsample = 1.0,
                          early_stopping_rounds = NULL,
                          validation_fraction = 0.0,
                          init = NULL,
                          random_state = 0L) {
      args <- list(
        estimand = estimand,
        n_estimators = as.integer(n_estimators),
        learning_rate = learning_rate,
        max_depth = as.integer(max_depth),
        reg_lambda = reg_lambda,
        subsample = subsample,
        validation_fraction = validation_fraction,
        random_state = as.integer(random_state)
      )
      if (!is.null(backend)) args$backend <- backend
      if (!is.null(loss)) args$loss <- loss
      if (!is.null(early_stopping_rounds))
        args$early_stopping_rounds <- as.integer(early_stopping_rounds)
      if (!is.null(init)) args$init <- init
      py_object <- do.call(.module()$RieszBooster, args)
      super$initialize(py_object = py_object, estimand = estimand)
    }
  )
)


#' Load a RieszBooster from a directory written by `RieszBooster$save()`.
#'
#' For built-in estimands, fully reconstructs the estimand from the metadata.
#' For custom estimands (Python-only), pass `estimand=` explicitly.
#' @param path Directory path.
#' @param estimand Optional user-supplied `Estimand` (required for custom m).
#' @export
load_riesz_booster <- function(path, estimand = NULL) {
  args <- list(path = path)
  if (!is.null(estimand)) args$estimand <- estimand
  py_obj <- do.call(.module()$RieszBooster$load, args)
  rb <- RieszBooster$new(estimand = py_obj$estimand,
                         n_estimators = 1L)  # dummy, replaced below
  rb$py <- py_obj
  rb$estimand <- py_obj$estimand
  rb
}


# Estimand and loss factories are re-exported from rieszreg via NAMESPACE
# (importFrom + export), so `library(rieszboost); ATE(...)` keeps working.
