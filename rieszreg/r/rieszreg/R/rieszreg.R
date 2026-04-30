#' rieszreg: shared abstractions for the Riesz regression family.
#'
#' Holds the estimand factories, Bregman-Riesz losses, and the base R6
#' estimator class used by every implementation package (rieszboost, krrr).
#' Implementation packages subclass [RieszEstimatorR6] and add their own
#' backend factories.
#'
#' @keywords internal
"_PACKAGE"


.rr <- new.env(parent = emptyenv())


#' Configure the Python interpreter that holds the rieszreg module.
#'
#' Call this once per session (or set `RETICULATE_PYTHON` before R starts).
#' Implementation packages have their own `use_python_<pkg>()` helpers; you
#' only need to call one of them per session.
#'
#' @param python Path to the Python interpreter or virtualenv directory.
#' @param required Whether reticulate should fail if the Python is unavailable.
#' @export
use_python_rieszreg <- function(python = NULL, required = TRUE) {
  if (!is.null(python)) {
    if (dir.exists(python)) {
      reticulate::use_virtualenv(python, required = required)
    } else {
      reticulate::use_python(python, required = required)
    }
  }
  .rr$mod <- reticulate::import("rieszreg", convert = FALSE)
  invisible(.rr$mod)
}


.module <- function() {
  if (is.null(.rr$mod)) {
    .rr$mod <- reticulate::import("rieszreg", convert = FALSE)
  }
  .rr$mod
}


#' Convert an R data.frame to a pandas DataFrame, preserving list-columns.
#'
#' List-columns (e.g. `shift_samples` for stochastic-intervention estimands)
#' contain a numeric vector per row; element-wise conversion keeps them as
#' Python lists rather than collapsing them.
#'
#' @param data An R data.frame.
#' @return A pandas DataFrame (Python object, `convert = FALSE`).
#' @export
df_to_py <- function(data) {
  cols <- colnames(data)
  py_dict <- list()
  for (k in cols) {
    v <- data[[k]]
    if (is.list(v)) {
      py_dict[[k]] <- lapply(v, function(x) reticulate::r_to_py(as.numeric(x)))
    } else {
      py_dict[[k]] <- as.numeric(v)
    }
  }
  pd <- reticulate::import("pandas", convert = FALSE)
  pd$DataFrame(reticulate::r_to_py(py_dict))
}


# ---- Estimand factories (return opaque Python Estimand instances) ----

#' Average treatment effect estimand: m(alpha)(z) = alpha(1, x) - alpha(0, x).
#' @param treatment Name of the treatment column.
#' @param covariates Character vector of covariate column names.
#' @return A Python `Estimand` object, suitable to pass to any RieszReg
#'   estimator's constructor via `estimand=`.
#' @export
ATE <- function(treatment = "a", covariates = "x") {
  .module()$ATE(treatment = treatment, covariates = as.list(covariates))
}


#' ATT *partial-estimand* surface: m(alpha)(z) = a*(alpha(1,x) - alpha(0,x)).
#'
#' Full ATT divides by P(A=1) and is not a Riesz functional — combine
#' alpha_partial with a delta-method EIF (Hubbard 2011) downstream.
#' @inheritParams ATE
#' @export
ATT <- function(treatment = "a", covariates = "x") {
  .module()$ATT(treatment = treatment, covariates = as.list(covariates))
}


#' Treatment-specific mean: m(alpha)(z) = alpha(level, x).
#' @param level Fixed treatment value.
#' @inheritParams ATE
#' @export
TSM <- function(level, treatment = "a", covariates = "x") {
  .module()$TSM(level = level, treatment = treatment,
                covariates = as.list(covariates))
}


#' Additive shift effect: m(alpha)(z) = alpha(a + delta, x) - alpha(a, x).
#' @param delta Shift magnitude.
#' @inheritParams ATE
#' @export
AdditiveShift <- function(delta, treatment = "a", covariates = "x") {
  .module()$AdditiveShift(delta = delta, treatment = treatment,
                          covariates = as.list(covariates))
}


#' LASE *partial-estimand* surface. Full LASE divides by P(A < threshold)
#' and is not a Riesz functional.
#' @param delta Shift magnitude.
#' @param threshold Cutoff; only rows with `a < threshold` get shifted.
#' @inheritParams ATE
#' @export
LocalShift <- function(delta, threshold, treatment = "a", covariates = "x") {
  .module()$LocalShift(delta = delta, threshold = threshold,
                       treatment = treatment, covariates = as.list(covariates))
}


#' Stochastic intervention via pre-computed Monte Carlo samples per row.
#' Pass a data.frame with a list-column under `samples_key` containing
#' numeric vectors of shift samples.
#' @inheritParams ATE
#' @param samples_key Column holding the per-row sample vectors.
#' @export
StochasticIntervention <- function(samples_key = "shift_samples",
                                   treatment = "a", covariates = "x") {
  .module()$StochasticIntervention(samples_key = samples_key,
                                   treatment = treatment,
                                   covariates = as.list(covariates))
}


# ---- Loss specs ----

#' Squared Riesz loss (default — the standard Lee-Schuler / Chernozhukov objective).
#' @export
SquaredLoss <- function() {
  .module()$SquaredLoss()
}

#' KL-Bregman loss (phi = t log t with exp link). Suitable for density-ratio
#' estimands like TSM / IPSI; requires non-negative m-coefficients.
#' @param max_eta Clip on η before applying the exponential link (numerical safety).
#' @export
KLLoss <- function(max_eta = 50.0) {
  .module()$KLLoss(max_eta = max_eta)
}

#' Bernoulli-Bregman loss (phi = t log t + (1-t) log(1-t), sigmoid link).
#'
#' Forces predictions into (0, 1) — useful when alpha_0 is known to lie
#' there by problem structure.
#' @param max_abs_eta Clip on |η| before applying the sigmoid link.
#' @export
BernoulliLoss <- function(max_abs_eta = 30.0) {
  .module()$BernoulliLoss(max_abs_eta = max_abs_eta)
}

#' Squared Riesz loss with predictions clipped into `(lo, hi)` via a
#' sigmoid-scaled link. Useful for representers with hard prior bounds
#' (e.g. trimmed propensity ratios). Pick bounds tightly around alpha_0;
#' very generous bounds saturate the link and slow boosting.
#' @param lo,hi Lower and upper bounds of the prediction range.
#' @param max_abs_eta Clip on |η| before the sigmoid (numerical safety).
#' @export
BoundedSquaredLoss <- function(lo, hi, max_abs_eta = 30.0) {
  .module()$BoundedSquaredLoss(lo = lo, hi = hi, max_abs_eta = max_abs_eta)
}


# ---- Base R6 class --------------------------------------------------

#' Base R6 class for Riesz regression estimators.
#'
#' Implementation packages (rieszboost, krrr) subclass this and override
#' `initialize` to construct their concrete Python estimator. The shared
#' methods (`fit`, `predict`, `score`, `riesz_loss`, `save`, `diagnose`,
#' `print`) operate on `self$py` (the Python object) and `self$estimand`.
#'
#' Subclasses typically look like:
#'
#' \preformatted{
#' RieszBooster <- R6::R6Class(
#'   "RieszBooster",
#'   inherit = rieszreg::RieszEstimatorR6,
#'   public = list(
#'     initialize = function(estimand, n_estimators = 200L, ...) {
#'       py_obj <- .module()$RieszBooster(estimand = estimand,
#'                                        n_estimators = as.integer(n_estimators), ...)
#'       super$initialize(py_object = py_obj, estimand = estimand)
#'     }
#'   )
#' )
#' }
#'
#' @export
RieszEstimatorR6 <- R6::R6Class(
  "RieszEstimatorR6",
  public = list(
    py = NULL,
    estimand = NULL,

    #' @param py_object The constructed Python estimator (subclasses build this).
    #' @param estimand The Python `Estimand` object that was passed to the constructor.
    initialize = function(py_object, estimand) {
      self$py <- py_object
      self$estimand <- estimand
      invisible(self)
    },

    #' Fit the estimator on a data.frame.
    #' @param data Training data (R data.frame; converted to pandas).
    #' @param eval_set Optional held-out data for early stopping / λ selection.
    fit = function(data, eval_set = NULL) {
      X <- df_to_py(data)
      args <- list(X = X)
      if (!is.null(eval_set)) {
        args$eval_set <- df_to_py(eval_set)
      }
      do.call(self$py$fit, args)
      invisible(self)
    },

    #' Predict α̂ on a data.frame. Returns a numeric vector.
    predict = function(data) {
      preds <- self$py$predict(df_to_py(data))
      as.numeric(reticulate::py_to_r(preds))
    },

    #' Negative held-out Riesz loss on a data.frame (sklearn higher-is-better).
    score = function(data) {
      reticulate::py_to_r(self$py$score(df_to_py(data)))
    },

    #' Held-out per-row Riesz loss.
    riesz_loss = function(data) {
      reticulate::py_to_r(self$py$riesz_loss(df_to_py(data)))
    },

    #' Save the fitted estimator to a directory.
    save = function(path) {
      self$py$save(path)
      invisible(self)
    },

    #' Diagnostics. Returns a list mirroring the Python `Diagnostics` dataclass.
    diagnose = function(data, ...) {
      d <- self$py$diagnose(df_to_py(data), ...)
      list(
        n = reticulate::py_to_r(d$n),
        rms = reticulate::py_to_r(d$rms),
        mean = reticulate::py_to_r(d$mean),
        min = reticulate::py_to_r(d$min),
        max = reticulate::py_to_r(d$max),
        abs_quantiles = reticulate::py_to_r(d$abs_quantiles),
        n_extreme = reticulate::py_to_r(d$n_extreme),
        extreme_fraction = reticulate::py_to_r(d$extreme_fraction),
        extreme_threshold = reticulate::py_to_r(d$extreme_threshold),
        riesz_loss = reticulate::py_to_r(d$riesz_loss),
        warnings = as.character(reticulate::py_to_r(d$warnings)),
        summary = reticulate::py_to_r(d$summary())
      )
    },

    print = function(...) {
      cat("<", class(self)[[1]], ">\n", sep = "")
      cat("  estimand   :", reticulate::py_to_r(self$estimand$name), "\n")
      best_iter <- tryCatch(reticulate::py_to_r(self$py$best_iteration_),
                            error = function(e) NULL)
      if (!is.null(best_iter)) {
        cat("  best_iter  :", best_iter, "\n")
        bs <- tryCatch(reticulate::py_to_r(self$py$best_score_),
                       error = function(e) NULL)
        if (!is.null(bs)) cat("  best_score :", bs, "\n")
      } else {
        cat("  status     : unfitted\n")
      }
      invisible(self)
    }
  )
)
