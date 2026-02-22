setup_cv_problem <- function() {
    set.seed(1)
    n <- 50
    p <- 20
    X <- matrix(rnorm(n * p), n, p)
    Y <- X[, 1:3] %*% c(1, -1, 0.5) + rnorm(n)
    list(X = X, Y = Y)
}

test_that("cv.spxlvb 1D returns correct structure", {
    dat <- setup_cv_problem()
    alpha_grid <- c(100, 1000)

    res <- cv.spxlvb(
        k = 3, X = dat$X, Y = dat$Y,
        alpha_prior_precision_grid = alpha_grid,
        max_iter = 30, tol = 1e-2, parallel = FALSE, verbose = FALSE
    )

    expect_type(res, "list")
    expect_equal(res$ordered_alpha_prior_precision_grid, sort(alpha_grid))
    expect_true(is.matrix(res$epe_test_k))
    expect_equal(dim(res$epe_test_k), c(3L, length(alpha_grid)))
    expect_equal(length(res$CVE), length(alpha_grid))
    expect_true(res$alpha_prior_precision_grid_opt %in% alpha_grid)
    expect_null(res$ordered_b_prior_precision_grid)
    expect_null(res$b_prior_precision_grid_opt)
})

test_that("cv.spxlvb 2D returns correct structure", {
    dat <- setup_cv_problem()
    alpha_grid <- c(100, 1000)
    b_grid <- c(1, 5)

    res <- cv.spxlvb(
        k = 3, X = dat$X, Y = dat$Y,
        alpha_prior_precision_grid = alpha_grid,
        b_prior_precision_grid = b_grid,
        max_iter = 30, tol = 1e-2, parallel = FALSE, verbose = FALSE
    )

    expect_type(res, "list")
    expect_equal(res$ordered_alpha_prior_precision_grid, sort(alpha_grid))
    expect_equal(res$ordered_b_prior_precision_grid, sort(b_grid))

    expect_true(is.array(res$epe_test_k))
    expect_equal(dim(res$epe_test_k), c(3L, length(alpha_grid), length(b_grid)))

    expect_true(is.matrix(res$CVE))
    expect_equal(dim(res$CVE), c(length(alpha_grid), length(b_grid)))

    expect_true(res$alpha_prior_precision_grid_opt %in% alpha_grid)
    expect_true(res$b_prior_precision_grid_opt %in% b_grid)
})

test_that("cv.spxlvb.fit 1D returns fit object", {
    dat <- setup_cv_problem()
    alpha_grid <- c(100, 1000)

    res <- cv.spxlvb.fit(
        k = 3, X = dat$X, Y = dat$Y,
        alpha_prior_precision_grid = alpha_grid,
        max_iter = 30, tol = 1e-2, parallel = FALSE, verbose = FALSE
    )

    expect_type(res, "list")
    expect_true("fit_spxlvb" %in% names(res))
    expect_true("alpha_prior_precision_grid_opt" %in% names(res))
    expect_false("b_prior_precision_grid_opt" %in% names(res))
    expect_equal(length(res$fit_spxlvb$beta), ncol(dat$X) + 1)
})

test_that("cv.spxlvb.fit 2D returns fit object with both optimal hyperparameters", {
    dat <- setup_cv_problem()
    alpha_grid <- c(100, 1000)
    b_grid <- c(1, 5)

    res <- cv.spxlvb.fit(
        k = 3, X = dat$X, Y = dat$Y,
        alpha_prior_precision_grid = alpha_grid,
        b_prior_precision_grid = b_grid,
        max_iter = 30, tol = 1e-2, parallel = FALSE, verbose = FALSE
    )

    expect_type(res, "list")
    expect_true("fit_spxlvb" %in% names(res))
    expect_true("alpha_prior_precision_grid_opt" %in% names(res))
    expect_true("b_prior_precision_grid_opt" %in% names(res))
    expect_true(res$b_prior_precision_grid_opt %in% b_grid)
    expect_equal(length(res$fit_spxlvb$beta), ncol(dat$X) + 1)
})

test_that("cv.spxlvb rejects b_prior_precision_grid with fewer than 2 values", {
    dat <- setup_cv_problem()
    expect_error(
        cv.spxlvb(
            k = 3, X = dat$X, Y = dat$Y,
            b_prior_precision_grid = 1,
            parallel = FALSE, verbose = FALSE
        ),
        "at least two values"
    )
})
