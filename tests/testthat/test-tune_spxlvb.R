setup_tune_problem <- function() {
    set.seed(1)
    n <- 50
    p <- 20
    X <- matrix(rnorm(n * p), n, p)
    beta_true <- c(1, -1, 0.5, rep(0, p - 3))
    Y <- X %*% beta_true + rnorm(n)
    list(X = X, Y = Y, beta_true = beta_true, n = n, p = p)
}

# =====================================================================
# ELBO criterion
# =====================================================================

test_that("tune_spxlvb with criterion='elbo' returns correct structure", {
    dat <- setup_tune_problem()
    alpha_grid <- c(100, 1000)
    b_grid <- c(1, 5)

    result <- suppressWarnings(tune_spxlvb(
        X = dat$X, Y = dat$Y,
        criterion = "elbo",
        alpha_prior_precision_grid = alpha_grid,
        b_prior_precision_grid = b_grid,
        max_iter = 30, tol = 1e-2,
        parallel = FALSE, verbose = FALSE
    ))

    expect_type(result, "list")
    expect_named(
        result,
        c("fit", "criterion", "optimal", "tuning_grid",
          "tuning_details", "refitted_on"),
        ignore.order = TRUE
    )
    expect_equal(result$criterion, "elbo")
    expect_equal(result$refitted_on, "training")
    expect_true(result$optimal$alpha_prior_precision %in% alpha_grid)
    expect_true(result$optimal$b_prior_precision %in% b_grid)
    expect_true("elbo" %in% names(result$tuning_grid))
    expect_equal(nrow(result$tuning_grid), length(alpha_grid) * length(b_grid))
    expect_equal(length(result$fit$beta), dat$p + 1)
})

test_that("tune_spxlvb ELBO 1D search works (b_prior_precision_grid = NULL)", {
    dat <- setup_tune_problem()
    alpha_grid <- c(100, 1000)

    result <- suppressWarnings(tune_spxlvb(
        X = dat$X, Y = dat$Y,
        criterion = "elbo",
        alpha_prior_precision_grid = alpha_grid,
        max_iter = 30, tol = 1e-2,
        parallel = FALSE, verbose = FALSE
    ))

    expect_equal(result$criterion, "elbo")
    expect_true(result$optimal$alpha_prior_precision %in% alpha_grid)
    expect_true(is.na(result$optimal$b_prior_precision))
    expect_equal(nrow(result$tuning_grid), length(alpha_grid))
})

test_that("tune_spxlvb ELBO selects the fit with highest ELBO", {
    dat <- setup_tune_problem()
    alpha_grid <- c(100, 1000)
    b_grid <- c(1, 5)

    result <- suppressWarnings(tune_spxlvb(
        X = dat$X, Y = dat$Y,
        criterion = "elbo",
        alpha_prior_precision_grid = alpha_grid,
        b_prior_precision_grid = b_grid,
        max_iter = 30, tol = 1e-2,
        parallel = FALSE, verbose = FALSE
    ))

    best_idx <- which.max(result$tuning_grid$elbo)
    expect_equal(
        result$optimal$alpha_prior_precision,
        result$tuning_grid$alpha_prior_precision[best_idx]
    )
    expect_equal(
        result$optimal$b_prior_precision,
        result$tuning_grid$b_prior_precision[best_idx]
    )
})

# =====================================================================
# CV criterion
# =====================================================================

test_that("tune_spxlvb with criterion='cv' returns correct structure", {
    dat <- setup_tune_problem()
    alpha_grid <- c(100, 1000)

    result <- suppressWarnings(tune_spxlvb(
        X = dat$X, Y = dat$Y,
        criterion = "cv", k = 3,
        alpha_prior_precision_grid = alpha_grid,
        max_iter = 30, tol = 1e-2,
        parallel = FALSE, verbose = FALSE
    ))

    expect_equal(result$criterion, "cv")
    expect_equal(result$refitted_on, "training")
    expect_true(result$optimal$alpha_prior_precision %in% alpha_grid)
    expect_true("mean_mspe" %in% names(result$tuning_grid))
    expect_true("per_fold_mspe" %in% names(result$tuning_details))
    expect_equal(result$tuning_details$k, 3)
    expect_equal(nrow(result$tuning_details$per_fold_mspe), 3)
    expect_equal(length(result$fit$beta), dat$p + 1)
})

test_that("tune_spxlvb CV 2D search works", {
    dat <- setup_tune_problem()
    alpha_grid <- c(100, 1000)
    b_grid <- c(1, 5)

    result <- suppressWarnings(tune_spxlvb(
        X = dat$X, Y = dat$Y,
        criterion = "cv", k = 3,
        alpha_prior_precision_grid = alpha_grid,
        b_prior_precision_grid = b_grid,
        max_iter = 30, tol = 1e-2,
        parallel = FALSE, verbose = FALSE
    ))

    expect_equal(result$criterion, "cv")
    expect_true(result$optimal$alpha_prior_precision %in% alpha_grid)
    expect_true(result$optimal$b_prior_precision %in% b_grid)
    expect_equal(
        nrow(result$tuning_grid),
        length(alpha_grid) * length(b_grid)
    )
})

# =====================================================================
# Validation criterion
# =====================================================================

test_that("tune_spxlvb with criterion='validation' returns correct structure", {
    dat <- setup_tune_problem()
    set.seed(2)
    X_val <- matrix(rnorm(30 * dat$p), 30, dat$p)
    Y_val <- X_val %*% dat$beta_true + rnorm(30)
    alpha_grid <- c(100, 1000)
    b_grid <- c(1, 5)

    result <- suppressWarnings(tune_spxlvb(
        X = dat$X, Y = dat$Y,
        criterion = "validation",
        X_validation = X_val, Y_validation = Y_val,
        alpha_prior_precision_grid = alpha_grid,
        b_prior_precision_grid = b_grid,
        max_iter = 30, tol = 1e-2,
        parallel = FALSE, verbose = FALSE
    ))

    expect_equal(result$criterion, "validation")
    expect_equal(result$refitted_on, "training_plus_validation")
    expect_true(result$optimal$alpha_prior_precision %in% alpha_grid)
    expect_true(result$optimal$b_prior_precision %in% b_grid)
    expect_true("validation_mspe" %in% names(result$tuning_grid))
})

test_that("tune_spxlvb validation with beta_true computes oracle MSPE", {
    dat <- setup_tune_problem()
    set.seed(2)
    X_val <- matrix(rnorm(30 * dat$p), 30, dat$p)
    Y_val <- X_val %*% dat$beta_true + rnorm(30)

    result <- suppressWarnings(tune_spxlvb(
        X = dat$X, Y = dat$Y,
        criterion = "validation",
        X_validation = X_val, Y_validation = Y_val,
        beta_true = dat$beta_true,
        alpha_prior_precision_grid = c(100, 1000),
        b_prior_precision_grid = c(1, 5),
        max_iter = 30, tol = 1e-2,
        parallel = FALSE, verbose = FALSE
    ))

    expect_true("oracle_mspe" %in% names(result$tuning_grid))
    expect_true(all(!is.na(result$tuning_grid$oracle_mspe)))
    expect_true("oracle_mspe" %in% names(result$tuning_details))
})

# =====================================================================
# Argument validation
# =====================================================================

test_that("tune_spxlvb errors when validation criterion lacks validation data", {
    dat <- setup_tune_problem()
    expect_error(
        tune_spxlvb(
            X = dat$X, Y = dat$Y,
            criterion = "validation",
            parallel = FALSE, verbose = FALSE
        ),
        "requires X_validation and Y_validation"
    )
})

test_that("tune_spxlvb warns when CV criterion gets validation data", {
    dat <- setup_tune_problem()
    X_val <- matrix(rnorm(30 * dat$p), 30, dat$p)
    Y_val <- rnorm(30)

    expect_warning(
        withCallingHandlers(
            tune_spxlvb(
                X = dat$X, Y = dat$Y,
                criterion = "cv", k = 3,
                X_validation = X_val, Y_validation = Y_val,
                alpha_prior_precision_grid = c(100, 1000),
                max_iter = 20, tol = 1e-1,
                parallel = FALSE, verbose = FALSE
            ),
            warning = function(w) {
                if (grepl("search grid", conditionMessage(w)))
                    invokeRestart("muffleWarning")
            }
        ),
        "ignored when criterion"
    )
})

test_that("tune_spxlvb warns when ELBO criterion gets validation data", {
    dat <- setup_tune_problem()
    X_val <- matrix(rnorm(30 * dat$p), 30, dat$p)
    Y_val <- rnorm(30)

    expect_warning(
        withCallingHandlers(
            tune_spxlvb(
                X = dat$X, Y = dat$Y,
                criterion = "elbo",
                X_validation = X_val, Y_validation = Y_val,
                alpha_prior_precision_grid = c(100, 1000),
                max_iter = 20, tol = 1e-1,
                parallel = FALSE, verbose = FALSE
            ),
            warning = function(w) {
                if (grepl("search grid", conditionMessage(w)))
                    invokeRestart("muffleWarning")
            }
        ),
        "ignored when criterion"
    )
})

test_that("tune_spxlvb CV rejects k < 3", {
    dat <- setup_tune_problem()
    expect_error(
        tune_spxlvb(
            X = dat$X, Y = dat$Y,
            criterion = "cv", k = 2,
            alpha_prior_precision_grid = c(100, 1000),
            parallel = FALSE, verbose = FALSE
        ),
        "k must be at least 3"
    )
})

# =====================================================================
# Deprecated wrappers still work
# =====================================================================

test_that("grid_search_spxlvb_fit still works with deprecation warning", {
    dat <- setup_tune_problem()

    result <- suppressWarnings(expect_warning(
        grid_search_spxlvb_fit(
            X = dat$X, Y = dat$Y,
            alpha_prior_precision_grid = c(100, 1000),
            b_prior_precision_grid = c(1, 5),
            max_iter = 30, tol = 1e-2,
            parallel = FALSE, verbose = FALSE
        ),
        "deprecated"
    ))

    expect_true("fit_spxlvb" %in% names(result))
    expect_true("hyper_grid" %in% names(result))
    expect_true("optimal_hyper" %in% names(result))
    expect_equal(length(result$fit_spxlvb$beta), dat$p + 1)
})

test_that("cv_spxlvb_fit still works with deprecation warning", {
    dat <- setup_tune_problem()

    result <- suppressWarnings(expect_warning(
        cv_spxlvb_fit(
            k = 3, X = dat$X, Y = dat$Y,
            alpha_prior_precision_grid = c(100, 1000),
            max_iter = 30, tol = 1e-2,
            parallel = FALSE, verbose = FALSE
        ),
        "deprecated"
    ))

    expect_true("fit_spxlvb" %in% names(result))
    expect_true("alpha_prior_precision_grid_opt" %in% names(result))
})
