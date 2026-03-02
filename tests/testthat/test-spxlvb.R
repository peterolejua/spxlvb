# Smoke tests for the core spxlvb model fit

# Shared fixture: small p >> n problem
setup_small_problem <- function() {
    p <- 100
    n <- 30
    L <- get_L(p = p, range = 2.0, smoothness = 1.5)
    dat <- matern_data_gen(
        seed_val = 42, n = n, p = p, pi_0 = 0.05,
        L = L, sd_beta = 4, SNR = 2
    )
    dat
}

test_that("spxlvb runs and returns valid structure", {
    dat <- setup_small_problem()
    p <- ncol(dat$X)

    fit <- spxlvb(
        X = dat$X, Y = dat$Y,
        max_iter = 50, tol = 1e-3
    )

    # Core structure
    expect_type(fit, "list")
    expect_true(is.logical(fit$converged))
    expect_true(fit$iterations > 0)
    expect_true(fit$iterations <= 50)

    # Coefficient dimensions (with intercept by default)
    expect_equal(length(fit$beta), p + 1)
    expect_equal(length(fit$mu), p)
    expect_equal(length(fit$omega), p)

    # Omega values in [0, 1]
    expect_true(all(fit$omega >= 0 & fit$omega <= 1))

    # tau_e positive
    expect_true(fit$tau_e > 0)
})

test_that("spxlvb works without intercept", {
    dat <- setup_small_problem()
    p <- ncol(dat$X)

    fit <- spxlvb(
        X = dat$X, Y = dat$Y,
        intercept = FALSE,
        max_iter = 50, tol = 1e-3
    )

    # No intercept: beta length == p (not p+1)
    expect_equal(length(fit$beta), p)
})

test_that("spxlvb works without standardization", {
    dat <- setup_small_problem()

    fit <- spxlvb(
        X = dat$X, Y = dat$Y,
        standardize = FALSE, intercept = FALSE,
        max_iter = 50, tol = 1e-3
    )

    expect_type(fit, "list")
    expect_true(is.logical(fit$converged))
})

test_that("save_history = TRUE returns history matrices", {
    dat <- setup_small_problem()
    p <- ncol(dat$X)

    fit <- spxlvb(
        X = dat$X, Y = dat$Y,
        max_iter = 50, tol = 1e-3,
        save_history = TRUE
    )

    ap <- fit$approximate_posterior

    # History matrices should exist
    expect_true("mu_history" %in% names(ap))
    expect_true("omega_history" %in% names(ap))
    expect_true("sigma_history" %in% names(ap))

    # History columns == number of iterations
    expect_equal(ncol(ap$mu_history), fit$iterations)
    expect_equal(nrow(ap$mu_history), p)
})

test_that("spxlvb warns when max_iter is reached without convergence", {
    dat <- setup_small_problem()

    expect_warning(
        spxlvb(
            X = dat$X, Y = dat$Y,
            max_iter = 1, tol = 1e-12
        ),
        "did not converge"
    )
})

test_that("spxlvb accepts initialization parameter for each strategy", {
    dat <- setup_small_problem()
    p <- ncol(dat$X)

    for (strat in c("lasso", "ridge", "lasso_ridge", "null")) {
        fit <- spxlvb(
            X = dat$X, Y = dat$Y,
            initialization = strat,
            max_iter = 50, tol = 1e-3
        )

        expect_type(fit, "list")
        expect_true(is.logical(fit$converged))
        expect_equal(length(fit$beta), p + 1)
        expect_true(all(fit$omega >= 0 & fit$omega <= 1))
    }
})

test_that("save_history = FALSE omits history matrices", {
    dat <- setup_small_problem()

    fit <- spxlvb(
        X = dat$X, Y = dat$Y,
        max_iter = 50, tol = 1e-3,
        save_history = FALSE
    )

    ap <- fit$approximate_posterior

    # History matrices should NOT exist
    expect_false("mu_history" %in% names(ap))
    expect_false("omega_history" %in% names(ap))
    expect_false("sigma_history" %in% names(ap))

    # But convergence/elbo history always present
    expect_true("convergence_history" %in% names(ap))
    expect_true("elbo_history" %in% names(ap))

    # Core fit should still work
    expect_true(is.logical(fit$converged))
    expect_true(length(fit$beta) > 0)
})
