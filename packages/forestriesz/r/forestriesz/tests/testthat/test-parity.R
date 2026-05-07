test_that("R-side and Python-side predictions match on a small TSM problem", {
  skip_if_not(reticulate::py_module_available("forestriesz"))

  set.seed(0)
  n <- 200
  x <- runif(n, 0, 1)
  pi <- 1 / (1 + exp(-(0.5 * x - 0.3)))
  a <- as.numeric(rbinom(n, 1, pi))
  df <- data.frame(a = a, x = x)

  # R side: locally constant on TSM (the only flavor R supports in v1)
  fr <- ForestRieszRegressor$new(
    estimand = TSM(level = 1L, treatment = "a", covariates = "x"),
    n_estimators = 16L,
    min_samples_leaf = 10L,
    random_state = 0L
  )
  fr$fit(df)
  alpha_R <- fr$predict(df)

  # Python side directly
  fr_py <- reticulate::import("forestriesz", convert = FALSE)
  pd <- reticulate::import("pandas", convert = FALSE)
  py_df <- pd$DataFrame(reticulate::r_to_py(list(a = a, x = x)))

  py_fr <- fr_py$ForestRieszRegressor(
    estimand = fr_py$TSM(level = 1L, treatment = "a", covariates = list("x")),
    n_estimators = 16L,
    min_samples_leaf = 10L,
    random_state = 0L
  )
  py_fr$fit(py_df)
  alpha_py <- as.numeric(reticulate::py_to_r(py_fr$predict(py_df)))

  expect_equal(alpha_R, alpha_py, tolerance = 1e-12)
})
