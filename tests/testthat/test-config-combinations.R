# Integration tests: verify all configurable option combinations run without error
# and produce valid output structure.

setup_easy_problem <- function() {
    set.seed(1)
    n <- 50
    p <- 20
    X <- matrix(rnorm(n * p), n, p)
    beta_true <- c(1, -1, 0.5, rep(0, p - 3))
    Y <- X %*% beta_true + rnorm(n)
    list(X = X, Y = Y, beta_true = beta_true, n = n, p = p)
}

validate_fit <- function(fit, p, has_intercept = TRUE) {
    expect_type(fit, "list")
    expect_true(is.logical(fit$converged))
    expect_true(fit$iterations > 0)
    expect_equal(length(fit$mu), p)
    expect_equal(length(fit$omega), p)
    expect_true(all(fit$omega >= 0 & fit$omega <= 1))
    expect_true(fit$tau_e > 0)
    expect_true(is.finite(fit$exploded_elbo))
    expect_true(is.finite(fit$alpha_stripped_elbo))
    expected_beta_len <- if (has_intercept) p + 1 else p
    expect_equal(length(fit$beta), expected_beta_len)
    expect_true(all(is.finite(fit$beta)))
}


# =====================================================================
# 1. Full factorial: convergence x update_pi x include_exploded_elbo_constants x disable_global_alpha
# =====================================================================

test_that("all convergence x update_pi x include_exploded_elbo_constants x disable_global_alpha combinations work", {
    dat <- setup_easy_problem()

    configs <- expand.grid(
        convergence = c("elbo_relative", "elbo_absolute", "chisq", "entropy"),
        update_pi = c(FALSE, TRUE),
        include_exploded_elbo_constants = c(FALSE, TRUE),
        disable_global_alpha = c(FALSE, TRUE),
        stringsAsFactors = FALSE
    )

    for (i in seq_len(nrow(configs))) {
        cfg <- configs[i, ]
        label <- paste(
            cfg$convergence,
            if (cfg$update_pi) "pi" else "nopi",
            if (cfg$include_exploded_elbo_constants) "full" else "partial",
            if (cfg$disable_global_alpha) "noalpha" else "alpha",
            sep = "_"
        )

        fit <- suppressWarnings(spxlvb(
            X = dat$X, Y = dat$Y,
            max_iter = 30, tol = 1e-2,
            convergence = cfg$convergence,
            update_pi = cfg$update_pi,
            include_exploded_elbo_constants = cfg$include_exploded_elbo_constants,
            disable_global_alpha = cfg$disable_global_alpha,
            save_history = FALSE,
            seed = 123
        ))

        validate_fit(fit, dat$p, has_intercept = TRUE)

        ap <- fit$approximate_posterior
        expect_true(length(ap$elbo_history) == fit$iterations,
            info = paste("ELBO history length mismatch for", label))
        expect_true(length(ap$alpha_stripped_elbo_history) == fit$iterations,
            info = paste("Alpha-stripped ELBO history length mismatch for", label))
    }
})


# =====================================================================
# 2. save_history interacts correctly with all convergence methods
# =====================================================================

test_that("save_history works with all convergence methods", {
    dat <- setup_easy_problem()

    for (conv in c("elbo_relative", "elbo_absolute", "chisq", "entropy")) {
        for (save in c(TRUE, FALSE)) {
            fit <- suppressWarnings(spxlvb(
                X = dat$X, Y = dat$Y,
                max_iter = 20, tol = 1e-2,
                convergence = conv,
                save_history = save,
                seed = 123
            ))

            ap <- fit$approximate_posterior

            if (save) {
                expect_true("mu_history" %in% names(ap),
                    info = paste("Missing mu_history:", conv, save))
                expect_equal(ncol(ap$mu_history), fit$iterations)
            } else {
                expect_false("mu_history" %in% names(ap),
                    info = paste("Unexpected mu_history:", conv, save))
            }

            expect_true("elbo_history" %in% names(ap))
            expect_true("convergence_history" %in% names(ap))
        }
    }
})


# =====================================================================
# 3. standardize/intercept combinations
# =====================================================================

test_that("standardize/intercept combinations work with all core options", {
    dat <- setup_easy_problem()

    std_int_configs <- list(
        list(standardize = TRUE, intercept = TRUE),
        list(standardize = TRUE, intercept = FALSE),
        list(standardize = FALSE, intercept = FALSE)
    )

    for (si in std_int_configs) {
        has_int <- si$intercept

        for (conv in c("elbo_relative", "elbo_absolute", "chisq", "entropy")) {
            fit <- suppressWarnings(spxlvb(
                X = dat$X, Y = dat$Y,
                max_iter = 20, tol = 1e-2,
                convergence = conv,
                standardize = si$standardize,
                intercept = si$intercept,
                save_history = FALSE,
                seed = 123
            ))

            validate_fit(fit, dat$p, has_intercept = has_int)
        }
    }
})


# =====================================================================
# 4. Tolerance variation doesn't break anything
# =====================================================================

test_that("different tolerance values produce valid fits", {
    dat <- setup_easy_problem()

    for (tol_val in c(1e-1, 1e-3, 1e-6)) {
        for (conv in c("elbo_relative", "elbo_absolute", "chisq", "entropy")) {
            fit <- suppressWarnings(spxlvb(
                X = dat$X, Y = dat$Y,
                max_iter = 100, tol = tol_val,
                convergence = conv,
                save_history = FALSE,
                seed = 123
            ))

            validate_fit(fit, dat$p)
        }
    }
})


# =====================================================================
# 5. alpha_prior_precision variation
# =====================================================================

test_that("different alpha_prior_precision values work with all options", {
    dat <- setup_easy_problem()

    for (tau_a in c(0, 1, 100, 1e6)) {
        for (dga in c(FALSE, TRUE)) {
            fit <- suppressWarnings(spxlvb(
                X = dat$X, Y = dat$Y,
                alpha_prior_precision = tau_a,
                max_iter = 30, tol = 1e-2,
                disable_global_alpha = dga,
                save_history = FALSE,
                seed = 123
            ))

            expect_type(fit, "list")
            expect_true(is.logical(fit$converged))
            expect_true(fit$iterations > 0)
            expect_equal(length(fit$mu), dat$p)
            expect_equal(length(fit$omega), dat$p)
            expect_true(all(fit$omega >= 0 & fit$omega <= 1))
            expect_true(all(is.finite(fit$beta)))

            if (tau_a > 0) {
                expect_true(is.finite(fit$exploded_elbo),
                    info = sprintf("tau_a=%g, dga=%s", tau_a, dga))
            }
        }
    }
})


# =====================================================================
# 6. disable_global_alpha produces different results from enabled
# =====================================================================

test_that("disable_global_alpha changes the fit", {
    dat <- setup_easy_problem()

    fit_on <- spxlvb(
        X = dat$X, Y = dat$Y,
        max_iter = 50, tol = 1e-4,
        disable_global_alpha = FALSE,
        save_history = FALSE, seed = 123
    )

    fit_off <- spxlvb(
        X = dat$X, Y = dat$Y,
        max_iter = 50, tol = 1e-4,
        disable_global_alpha = TRUE,
        save_history = FALSE, seed = 123
    )

    expect_false(
        isTRUE(all.equal(fit_on$omega, fit_off$omega, tolerance = 1e-10)),
        info = "Enabling/disabling global alpha should produce different omega"
    )
})


# =====================================================================
# 7. ELBO monotonicity under ELBO convergence criterion
# =====================================================================

test_that("ELBO is non-decreasing under ELBO convergence for all config combos", {
    dat <- setup_easy_problem()

    key_configs <- list(
        list(update_pi = FALSE, include_exploded_elbo_constants = FALSE, dga = FALSE),
        list(update_pi = TRUE, include_exploded_elbo_constants = FALSE, dga = FALSE),
        list(update_pi = FALSE, include_exploded_elbo_constants = TRUE, dga = FALSE),
        list(update_pi = FALSE, include_exploded_elbo_constants = FALSE, dga = TRUE),
        list(update_pi = TRUE, include_exploded_elbo_constants = TRUE, dga = FALSE),
        list(update_pi = TRUE, include_exploded_elbo_constants = TRUE, dga = TRUE)
    )

    for (cfg in key_configs) {
        label <- paste(
            if (cfg$update_pi) "pi" else "nopi",
            if (cfg$include_exploded_elbo_constants) "full" else "partial",
            if (cfg$dga) "noalpha" else "alpha",
            sep = "_"
        )

        fit <- suppressWarnings(spxlvb(
            X = dat$X, Y = dat$Y,
            max_iter = 100, tol = 1e-6,
            convergence = "elbo_relative",
            update_pi = cfg$update_pi,
            include_exploded_elbo_constants = cfg$include_exploded_elbo_constants,
            disable_global_alpha = cfg$dga,
            save_history = FALSE,
            seed = 123
        ))

        elbo_hist <- as.numeric(fit$approximate_posterior$elbo_history)
        diffs <- diff(elbo_hist)
        violations <- sum(diffs < -1e-6)

        expect_true(
            violations == 0,
            info = sprintf(
                "ELBO monotonicity violated for %s: %d violations, worst = %.2e",
                label, violations, if (violations > 0) min(diffs) else 0
            )
        )
    }
})


# =====================================================================
# 8. tune_spxlvb with different criteria and options
# =====================================================================

test_that("tune_spxlvb works with all criteria x core options", {
    dat <- setup_easy_problem()
    small_grid <- c(100, 1000)

    for (crit in c("elbo", "cv")) {
        for (up in c(FALSE, TRUE)) {
            for (dga in c(FALSE, TRUE)) {
                label <- paste(crit, if (up) "pi" else "nopi",
                    if (dga) "noalpha" else "alpha", sep = "_")

                result <- suppressWarnings(suppressMessages(
                    tune_spxlvb(
                        X = dat$X, Y = dat$Y,
                        criterion = crit,
                        alpha_prior_precision_grid = small_grid,
                        max_iter = 20, tol = 1e-2,
                        update_pi = up,
                        disable_global_alpha = dga,
                        parallel = FALSE, verbose = FALSE,
                        k = 3L,
                        seed = 123
                    )
                ))

                expect_type(result, "list")
                expect_equal(result$criterion, crit,
                    info = paste("Wrong criterion for", label))
                expect_true(
                    result$optimal$alpha_prior_precision %in% small_grid,
                    info = paste("Optimal not in grid for", label)
                )

                validate_fit(result$fit, dat$p)
            }
        }
    }
})


# =====================================================================
# 9. tune_spxlvb selection_elbo variants
# =====================================================================

test_that("tune_spxlvb selection_elbo variants both work", {
    dat <- setup_easy_problem()
    small_grid <- c(100, 1000)

    for (sel in c("exploded_model", "alpha_stripped")) {
        result <- suppressWarnings(suppressMessages(
            tune_spxlvb(
                X = dat$X, Y = dat$Y,
                criterion = "elbo",
                alpha_prior_precision_grid = small_grid,
                max_iter = 20, tol = 1e-2,
                selection_elbo = sel,
                parallel = FALSE, verbose = FALSE,
                seed = 123
            )
        ))

        expect_type(result, "list")
        expect_true(
            "exploded_elbo" %in% names(result$tuning_grid),
            info = paste("Missing exploded_elbo column for", sel)
        )
        expect_true(
            "alpha_stripped_elbo" %in% names(result$tuning_grid),
            info = paste("Missing alpha_stripped_elbo column for", sel)
        )
        validate_fit(result$fit, dat$p)
    }
})


# =====================================================================
# 10. tune_spxlvb with validation criterion
# =====================================================================

test_that("tune_spxlvb validation criterion works with core options", {
    dat <- setup_easy_problem()

    set.seed(2)
    n_val <- 20
    X_val <- matrix(rnorm(n_val * dat$p), n_val, dat$p)
    Y_val <- X_val %*% dat$beta_true + rnorm(n_val)
    small_grid <- c(100, 1000)

    for (dga in c(FALSE, TRUE)) {
        result <- suppressWarnings(suppressMessages(
            tune_spxlvb(
                X = dat$X, Y = dat$Y,
                criterion = "validation",
                X_validation = X_val, Y_validation = Y_val,
                alpha_prior_precision_grid = small_grid,
                max_iter = 20, tol = 1e-2,
                disable_global_alpha = dga,
                parallel = FALSE, verbose = FALSE,
                seed = 123
            )
        ))

        expect_equal(result$criterion, "validation")
        expect_equal(result$refitted_on, "training_plus_validation")
        n_combined <- dat$n + n_val
        validate_fit(result$fit, dat$p)
    }
})


# =====================================================================
# 11. gamma_hyperprior_tau_alpha x convergence x disable_global_alpha combinations
# =====================================================================

test_that("gamma_hyperprior_tau_alpha works with all convergence x disable_global_alpha combos", {
    dat <- setup_easy_problem()

    for (conv in c("elbo_relative", "elbo_absolute", "chisq", "entropy")) {
        for (dga in c(FALSE, TRUE)) {
            label <- paste("gamma", conv,
                if (dga) "noalpha" else "alpha", sep = "_")

            fit <- suppressWarnings(spxlvb(
                X = dat$X, Y = dat$Y,
                max_iter = 30, tol = 1e-2,
                convergence = conv,
                disable_global_alpha = dga,
                gamma_hyperprior_tau_alpha = TRUE,
                save_history = FALSE,
                seed = 123
            ))

            validate_fit(fit, dat$p, has_intercept = TRUE)
            expect_true(is.finite(fit$tau_alpha),
                info = paste("tau_alpha not finite for", label))
            expect_true(fit$tau_alpha > 0,
                info = paste("tau_alpha not positive for", label))
            expect_equal(
                length(fit$tau_alpha_history), fit$iterations,
                info = paste("tau_alpha_history length mismatch for", label)
            )
        }
    }
})


# =====================================================================
# 12. gamma_hyperprior_tau_b x convergence x disable_global_alpha combinations
# =====================================================================

test_that("gamma_hyperprior_tau_b works with all convergence x disable_global_alpha combos", {
    dat <- setup_easy_problem()

    for (conv in c("elbo_relative", "elbo_absolute", "chisq", "entropy")) {
        for (dga in c(FALSE, TRUE)) {
            label <- paste("gamma_b", conv,
                if (dga) "noalpha" else "alpha", sep = "_")

            fit <- suppressWarnings(spxlvb(
                X = dat$X, Y = dat$Y,
                max_iter = 30, tol = 1e-2,
                convergence = conv,
                disable_global_alpha = dga,
                gamma_hyperprior_tau_b = TRUE,
                save_history = FALSE,
                seed = 123
            ))

            validate_fit(fit, dat$p, has_intercept = TRUE)
            expect_true(is.finite(fit$tau_b_common),
                info = paste("tau_b_common not finite for", label))
            expect_true(fit$tau_b_common > 0,
                info = paste("tau_b_common not positive for", label))
            expect_equal(
                length(fit$tau_b_common_history), fit$iterations,
                info = paste("tau_b_common_history length mismatch for", label)
            )
        }
    }
})


# =====================================================================
# 13. Both gamma hyperpriors x convergence combinations
# =====================================================================

test_that("both gamma hyperpriors work with all convergence methods", {
    dat <- setup_easy_problem()

    for (conv in c("elbo_relative", "elbo_absolute", "chisq", "entropy")) {
        label <- paste("both", conv, sep = "_")

        fit <- suppressWarnings(spxlvb(
            X = dat$X, Y = dat$Y,
            max_iter = 30, tol = 1e-2,
            convergence = conv,
            gamma_hyperprior_tau_alpha = TRUE,
            gamma_hyperprior_tau_b = TRUE,
            save_history = FALSE,
            seed = 123
        ))

        validate_fit(fit, dat$p, has_intercept = TRUE)
        expect_true(is.finite(fit$tau_alpha),
            info = paste("tau_alpha not finite for", label))
        expect_true(is.finite(fit$tau_b_common),
            info = paste("tau_b_common not finite for", label))
        expect_equal(
            length(fit$tau_alpha_history), fit$iterations,
            info = paste("tau_alpha_history length mismatch for", label)
        )
        expect_equal(
            length(fit$tau_b_common_history), fit$iterations,
            info = paste("tau_b_common_history length mismatch for", label)
        )
    }
})
