setup_gamma_problem <- function() {
  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(1, -1, 0.5, rep(0, p - 3))
  Y <- X %*% beta_true + rnorm(n)
  list(X = X, Y = Y, beta_true = beta_true, n = n, p = p)
}


# =====================================================================
# 1. gamma_hyperprior_tau_alpha = FALSE is identical to default
# =====================================================================

test_that("gamma_hyperprior_tau_alpha = FALSE gives identical results to default", {
  dat <- setup_gamma_problem()

  fit_default <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 50, tol = 1e-4, seed = 123
  )

  fit_false <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 50, tol = 1e-4, seed = 123,
    gamma_hyperprior_tau_alpha = FALSE
  )

  expect_equal(fit_default$omega, fit_false$omega, tolerance = 1e-10)
  expect_equal(fit_default$mu, fit_false$mu, tolerance = 1e-10)
  expect_equal(fit_default$beta, fit_false$beta, tolerance = 1e-10)
  expect_equal(fit_default$exploded_elbo, fit_false$exploded_elbo, tolerance = 1e-10)
  expect_equal(fit_default$iterations, fit_false$iterations)
  expect_equal(fit_default$tau_alpha, fit_false$tau_alpha, tolerance = 1e-10)
  expect_null(fit_false$tau_alpha_history)
})


# =====================================================================
# 2. gamma_hyperprior_tau_alpha = TRUE converges
# =====================================================================

test_that("gamma_hyperprior_tau_alpha = TRUE converges on basic problem", {
  dat <- setup_gamma_problem()

  fit <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 200, tol = 1e-4, seed = 123,
    gamma_hyperprior_tau_alpha = TRUE
  )

  expect_true(fit$converged)
  expect_true(fit$iterations > 1)
  expect_true(fit$iterations <= 200)
  expect_equal(length(fit$mu), dat$p)
  expect_equal(length(fit$omega), dat$p)
  expect_true(all(fit$omega >= 0 & fit$omega <= 1))
  expect_true(all(is.finite(fit$beta)))
  expect_true(is.finite(fit$exploded_elbo))
})


# =====================================================================
# 3. tau_alpha_history is monotonically stabilizing
# =====================================================================

test_that("tau_alpha_history stabilizes (not oscillating wildly)", {
  dat <- setup_gamma_problem()

  fit <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 200, tol = 1e-6, seed = 123,
    gamma_hyperprior_tau_alpha = TRUE
  )

  tau_hist <- fit$tau_alpha_history
  expect_true(length(tau_hist) >= 2)

  diffs <- abs(diff(tau_hist))
  late_diffs <- tail(diffs, max(1, length(diffs) %/% 2))
  early_diffs <- head(diffs, max(1, length(diffs) %/% 2))

  expect_true(
    mean(late_diffs) <= mean(early_diffs) + 1e-6,
    info = "tau_alpha should stabilize: late changes <= early changes"
  )
})


# =====================================================================
# 4. tau_alpha_history length equals iterations
# =====================================================================

test_that("tau_alpha_history length equals iterations", {
  dat <- setup_gamma_problem()

  fit <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 30, tol = 1e-3, seed = 123,
    gamma_hyperprior_tau_alpha = TRUE
  )

  expect_equal(length(fit$tau_alpha_history), fit$iterations)
})


# =====================================================================
# 5. Final tau_alpha is finite and positive
# =====================================================================

test_that("final tau_alpha is finite and positive", {
  dat <- setup_gamma_problem()

  fit <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 100, tol = 1e-4, seed = 123,
    gamma_hyperprior_tau_alpha = TRUE
  )

  expect_true(is.finite(fit$tau_alpha))
  expect_true(fit$tau_alpha > 0)
  expect_true(all(is.finite(fit$tau_alpha_history)))
  expect_true(all(fit$tau_alpha_history > 0))
})


# =====================================================================
# 6. ELBO is approximately monotone with gamma_hyperprior_tau_alpha
# =====================================================================

test_that("ELBO is approximately monotone with gamma_hyperprior_tau_alpha", {
  dat <- setup_gamma_problem()

  fit <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 200, tol = 1e-6, seed = 123,
    gamma_hyperprior_tau_alpha = TRUE,
    save_history = FALSE
  )

  elbo_hist <- as.numeric(fit$approximate_posterior$elbo_history)
  diffs <- diff(elbo_hist)
  violations <- sum(diffs < -1e-4)

  expect_true(
    violations <= 3,
    info = sprintf(
      "ELBO monotonicity: %d violations > 1e-4, worst = %.2e",
      violations, if (length(diffs) > 0 && any(diffs < 0)) min(diffs) else 0
    )
  )
})


# =====================================================================
# 7. tune_spxlvb with gamma_hyperprior_tau_alpha + criterion = "elbo"
# =====================================================================

test_that("tune_spxlvb works with gamma_hyperprior_tau_alpha + elbo criterion", {
  dat <- setup_gamma_problem()

  result <- suppressWarnings(suppressMessages(
    tune_spxlvb(
      X = dat$X, Y = dat$Y,
      criterion = "elbo",
      b_prior_precision_grid = c(0.5, 1, 5),
      max_iter = 30, tol = 1e-3,
      gamma_hyperprior_tau_alpha = TRUE,
      parallel = FALSE, verbose = FALSE,
      seed = 123
    )
  ))

  expect_type(result, "list")
  expect_equal(result$criterion, "elbo")
  expect_true(is.finite(result$fit$tau_alpha))
  expect_true(result$fit$tau_alpha > 0)
  expect_equal(nrow(result$tuning_grid), 3)
})


# =====================================================================
# 8. tune_spxlvb with gamma_hyperprior_tau_alpha + criterion = "cv"
# =====================================================================

test_that("tune_spxlvb works with gamma_hyperprior_tau_alpha + cv criterion", {
  dat <- setup_gamma_problem()

  result <- suppressWarnings(suppressMessages(
    tune_spxlvb(
      X = dat$X, Y = dat$Y,
      criterion = "cv", k = 3L,
      b_prior_precision_grid = c(0.5, 1, 5),
      max_iter = 30, tol = 1e-3,
      gamma_hyperprior_tau_alpha = TRUE,
      parallel = FALSE, verbose = FALSE,
      seed = 123
    )
  ))

  expect_type(result, "list")
  expect_equal(result$criterion, "cv")
  expect_true(is.finite(result$fit$tau_alpha))
  expect_true(result$fit$tau_alpha > 0)
  expect_equal(nrow(result$tuning_grid), 3)
})


# =====================================================================
# 9. Results differ from fixed tau_alpha (gamma actually adapts)
# =====================================================================

test_that("gamma_hyperprior_tau_alpha produces different results than fixed tau_alpha", {
  dat <- setup_gamma_problem()

  fit_fixed <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 100, tol = 1e-5, seed = 123,
    gamma_hyperprior_tau_alpha = FALSE
  )

  fit_gamma <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 100, tol = 1e-5, seed = 123,
    gamma_hyperprior_tau_alpha = TRUE
  )

  expect_false(
    isTRUE(all.equal(fit_fixed$omega, fit_gamma$omega, tolerance = 1e-6)),
    info = "gamma_hyperprior_tau_alpha should produce different omega than fixed"
  )
})


# =====================================================================
# 10. Works with disable_global_alpha = TRUE
# =====================================================================

test_that("gamma_hyperprior_tau_alpha works with disable_global_alpha = TRUE", {
  dat <- setup_gamma_problem()

  fit <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 100, tol = 1e-4, seed = 123,
    gamma_hyperprior_tau_alpha = TRUE,
    disable_global_alpha = TRUE
  )

  expect_true(fit$converged || fit$iterations == 100)
  expect_true(is.finite(fit$tau_alpha))
  expect_true(fit$tau_alpha > 0)
  expect_equal(length(fit$tau_alpha_history), fit$iterations)
  expect_true(all(is.finite(fit$tau_alpha_history)))
  expect_equal(length(fit$mu), dat$p)
  expect_equal(length(fit$omega), dat$p)
  expect_true(all(fit$omega >= 0 & fit$omega <= 1))
  expect_true(all(is.finite(fit$beta)))
})


# =====================================================================
# 11. Custom r_alpha and d_alpha are respected
# =====================================================================

test_that("custom r_alpha and d_alpha produce different tau_alpha", {
  dat <- setup_gamma_problem()

  fit_default_prior <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 50, tol = 1e-4, seed = 123,
    gamma_hyperprior_tau_alpha = TRUE
  )

  fit_strong_prior <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 50, tol = 1e-4, seed = 123,
    gamma_hyperprior_tau_alpha = TRUE,
    r_alpha = 1, d_alpha = 100
  )

  expect_false(
    isTRUE(all.equal(
      fit_default_prior$tau_alpha,
      fit_strong_prior$tau_alpha,
      tolerance = 1e-2
    )),
    info = "Different hyperprior params should yield different final tau_alpha"
  )
})


# =====================================================================
# 12. gamma_hyperprior_tau_b = FALSE is identical to default
# =====================================================================

test_that("gamma_hyperprior_tau_b = FALSE gives identical results to default", {
  dat <- setup_gamma_problem()

  fit_default <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 50, tol = 1e-4, seed = 123
  )

  fit_false <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 50, tol = 1e-4, seed = 123,
    gamma_hyperprior_tau_b = FALSE
  )

  expect_equal(fit_default$omega, fit_false$omega, tolerance = 1e-10)
  expect_equal(fit_default$mu, fit_false$mu, tolerance = 1e-10)
  expect_equal(fit_default$beta, fit_false$beta, tolerance = 1e-10)
  expect_equal(fit_default$exploded_elbo, fit_false$exploded_elbo, tolerance = 1e-10)
  expect_equal(fit_default$iterations, fit_false$iterations)
  expect_null(fit_false$tau_b_common)
  expect_null(fit_false$tau_b_common_history)
})


# =====================================================================
# 13. gamma_hyperprior_tau_b = TRUE converges
# =====================================================================

test_that("gamma_hyperprior_tau_b = TRUE converges on basic problem", {
  dat <- setup_gamma_problem()

  fit <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 200, tol = 1e-4, seed = 123,
    gamma_hyperprior_tau_b = TRUE
  )

  expect_true(fit$converged)
  expect_true(fit$iterations > 1)
  expect_true(fit$iterations <= 200)
  expect_equal(length(fit$mu), dat$p)
  expect_equal(length(fit$omega), dat$p)
  expect_true(all(fit$omega >= 0 & fit$omega <= 1))
  expect_true(all(is.finite(fit$beta)))
  expect_true(is.finite(fit$exploded_elbo))
})


# =====================================================================
# 14. tau_b_common is finite and positive
# =====================================================================

test_that("tau_b_common is finite and positive", {
  dat <- setup_gamma_problem()

  fit <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 100, tol = 1e-4, seed = 123,
    gamma_hyperprior_tau_b = TRUE
  )

  expect_true(is.finite(fit$tau_b_common))
  expect_true(fit$tau_b_common > 0)
  expect_true(all(is.finite(fit$tau_b_common_history)))
  expect_true(all(fit$tau_b_common_history > 0))
})


# =====================================================================
# 15. tau_b_common_history length equals iterations
# =====================================================================

test_that("tau_b_common_history length equals iterations", {
  dat <- setup_gamma_problem()

  fit <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 30, tol = 1e-3, seed = 123,
    gamma_hyperprior_tau_b = TRUE
  )

  expect_equal(length(fit$tau_b_common_history), fit$iterations)
})


# =====================================================================
# 16. tau_b_common_history stabilizes
# =====================================================================

test_that("tau_b_common_history stabilizes", {
  dat <- setup_gamma_problem()

  fit <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 200, tol = 1e-6, seed = 123,
    gamma_hyperprior_tau_b = TRUE
  )

  tau_hist <- fit$tau_b_common_history
  expect_true(length(tau_hist) >= 2)

  diffs <- abs(diff(tau_hist))
  late_diffs <- tail(diffs, max(1, length(diffs) %/% 2))
  early_diffs <- head(diffs, max(1, length(diffs) %/% 2))

  expect_true(
    mean(late_diffs) <= mean(early_diffs) + 1e-6,
    info = "tau_b_common should stabilize: late changes <= early changes"
  )
})


# =====================================================================
# 17. gamma_hyperprior_tau_b produces different results than fixed tau_b
# =====================================================================

test_that("gamma_hyperprior_tau_b produces different results than fixed tau_b", {
  dat <- setup_gamma_problem()

  fit_fixed <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 100, tol = 1e-5, seed = 123,
    gamma_hyperprior_tau_b = FALSE
  )

  fit_gamma <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 100, tol = 1e-5, seed = 123,
    gamma_hyperprior_tau_b = TRUE
  )

  expect_false(
    isTRUE(all.equal(fit_fixed$omega, fit_gamma$omega, tolerance = 1e-6)),
    info = "gamma_hyperprior_tau_b should produce different omega than fixed"
  )
})


# =====================================================================
# 18. Both hyperpriors together converge
# =====================================================================

test_that("both gamma hyperpriors together converge", {
  dat <- setup_gamma_problem()

  fit <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 200, tol = 1e-4, seed = 123,
    gamma_hyperprior_tau_alpha = TRUE,
    gamma_hyperprior_tau_b = TRUE
  )

  expect_true(fit$converged)
  expect_true(is.finite(fit$tau_alpha))
  expect_true(fit$tau_alpha > 0)
  expect_true(is.finite(fit$tau_b_common))
  expect_true(fit$tau_b_common > 0)
  expect_equal(length(fit$tau_alpha_history), fit$iterations)
  expect_equal(length(fit$tau_b_common_history), fit$iterations)
  expect_equal(length(fit$mu), dat$p)
  expect_equal(length(fit$omega), dat$p)
  expect_true(all(fit$omega >= 0 & fit$omega <= 1))
  expect_true(all(is.finite(fit$beta)))
})


# =====================================================================
# 19. Both hyperpriors: ELBO approximately monotone
# =====================================================================

test_that("ELBO is approximately monotone with both hyperpriors", {
  dat <- setup_gamma_problem()

  fit <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 200, tol = 1e-6, seed = 123,
    gamma_hyperprior_tau_alpha = TRUE,
    gamma_hyperprior_tau_b = TRUE,
    save_history = FALSE
  )

  elbo_hist <- as.numeric(fit$approximate_posterior$elbo_history)
  diffs <- diff(elbo_hist)
  violations <- sum(diffs < -1e-4)

  expect_true(
    violations <= 3,
    info = sprintf(
      "ELBO monotonicity (both): %d violations > 1e-4, worst = %.2e",
      violations, if (length(diffs) > 0 && any(diffs < 0)) min(diffs) else 0
    )
  )
})


# =====================================================================
# 20. Custom r_b and d_b are respected
# =====================================================================

test_that("custom r_b and d_b produce different tau_b_common", {
  dat <- setup_gamma_problem()

  fit_default_prior <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 50, tol = 1e-4, seed = 123,
    gamma_hyperprior_tau_b = TRUE
  )

  fit_strong_prior <- spxlvb(
    X = dat$X, Y = dat$Y,
    max_iter = 50, tol = 1e-4, seed = 123,
    gamma_hyperprior_tau_b = TRUE,
    r_b = 1, d_b = 100
  )

  expect_false(
    isTRUE(all.equal(
      fit_default_prior$tau_b_common,
      fit_strong_prior$tau_b_common,
      tolerance = 1e-2
    )),
    info = "Different hyperprior params should yield different final tau_b_common"
  )
})


# =====================================================================
# 21. tune_spxlvb with both hyperpriors (zero grid search)
# =====================================================================

test_that("tune_spxlvb with both hyperpriors produces valid result", {
  dat <- setup_gamma_problem()

  result <- suppressWarnings(suppressMessages(
    tune_spxlvb(
      X = dat$X, Y = dat$Y,
      criterion = "elbo",
      max_iter = 30, tol = 1e-3,
      gamma_hyperprior_tau_alpha = TRUE,
      gamma_hyperprior_tau_b = TRUE,
      parallel = FALSE, verbose = FALSE,
      seed = 123
    )
  ))

  expect_type(result, "list")
  expect_equal(result$criterion, "elbo")
  expect_true(is.finite(result$fit$tau_alpha))
  expect_true(result$fit$tau_alpha > 0)
  expect_true(is.finite(result$fit$tau_b_common))
  expect_true(result$fit$tau_b_common > 0)
  expect_equal(nrow(result$tuning_grid), 1)
})
