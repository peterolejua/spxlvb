#' @title Tune Hyperparameters for spxlvb
#'
#' @description
#' Selects optimal hyperparameters for the \code{\link{spxlvb}} model and
#' returns the final fitted model. Three tuning criteria are supported:
#' ELBO maximisation, k-fold cross-validation, and held-out validation.
#'
#' @param X Numeric matrix (\eqn{n \times p}). Training design matrix.
#' @param Y Numeric vector of length \eqn{n}. Training response.
#' @param criterion Character string specifying the tuning strategy.
#'   One of:
#'   \describe{
#'     \item{\code{"elbo"} (default)}{Fit the model on the full training
#'       data for every grid combination and select the hyperparameters
#'       that maximise the Evidence Lower Bound (ELBO). No data splitting
#'       is performed. This is the recommended criterion when no external
#'       validation set is available; see the paper Appendix for an
#'       empirical comparison with cross-validation.}
#'     \item{\code{"cv"}}{Perform k-fold cross-validation. For each
#'       grid combination, the model is fit on \eqn{k-1} folds and
#'       evaluated via mean squared prediction error (MSPE) on the
#'       held-out fold. The combination with the lowest mean MSPE is
#'       selected, and the final model is refit on the full training
#'       data.}
#'     \item{\code{"validation"}}{Fit on the training data, evaluate
#'       MSPE on the held-out validation set (\code{X_validation},
#'       \code{Y_validation}). The final model is refit on the
#'       combined training + validation data.}
#'   }
#' @param alpha_prior_precision_grid Numeric vector. Grid of expansion
#'   prior precision (\eqn{\tau_\alpha}) values to search over.
#'   Default: \code{c(0, 10^(3:7))}.
#' @param b_prior_precision_grid Optional numeric vector. Grid of scalar
#'   slab prior precisions (\eqn{\tau_b}) to search over. When
#'   \code{NULL} (default), only \code{alpha_prior_precision_grid} is
#'   searched (1D tuning), and the fixed \code{b_prior_precision} is
#'   used for every fit. When non-\code{NULL}, a 2D grid search over
#'   all combinations of \eqn{(\tau_\alpha, \tau_b)} is performed. Each
#'   scalar value is expanded to a constant vector of length \eqn{p} via
#'   \code{rep(value, p)}.
#' @param b_prior_precision Numeric vector of length \eqn{p}.
#'   Fixed slab prior precisions used when \code{b_prior_precision_grid}
#'   is \code{NULL}. Default: \code{rep(1, ncol(X))}.
#' @param k Integer. Number of folds for cross-validation. Only used
#'   when \code{criterion = "cv"}. Must be at least 3. Default: 5.
#' @param X_validation Optional numeric matrix. Validation design matrix.
#'   Required when \code{criterion = "validation"}.
#' @param Y_validation Optional numeric vector. Validation response.
#'   Required when \code{criterion = "validation"}.
#' @param beta_true Optional numeric vector of true coefficients (length
#'   \eqn{p} or \eqn{p+1}). When supplied alongside
#'   \code{criterion = "validation"}, the oracle linear-predictor MSE
#'   (\eqn{\|X\beta_{true} - X\hat\beta\|^2/n}) is computed for each
#'   grid point and reported in \code{tuning_details$grid}. Selection
#'   still uses MSPE against \code{Y_validation}; the oracle metric is
#'   informational only.
#' @param mu_0 Optional numeric vector. Initial variational means.
#' @param omega_0 Optional numeric vector. Initial variational
#'   inclusion probabilities.
#' @param c_pi_0 Optional numeric. Prior Beta shape1 for \eqn{\pi}.
#' @param d_pi_0 Optional numeric. Prior Beta shape2 for \eqn{\pi}.
#' @param tau_e Optional numeric. Initial error precision.
#' @param update_order Optional integer vector. Coordinate update
#'   order (0-indexed for C++).
#' @param mu_alpha Numeric vector of length \eqn{p+1}. Prior means for
#'   the expansion parameters (see \code{\link{spxlvb}}). Default:
#'   a vector of ones.
#' @param standardize Logical. Center Y and center + scale X.
#'   Default: \code{TRUE}.
#' @param intercept Logical. Include an intercept (requires
#'   \code{standardize = TRUE}). Default: \code{TRUE}.
#' @param max_iter Integer. Maximum VB iterations per fit.
#'   Default: 100.
#' @param tol Numeric. Convergence tolerance. Default: 1e-5.
#' @param seed Integer. Seed for reproducibility. Default: 12376.
#' @param verbose Logical. Print progress messages. Default: \code{TRUE}.
#' @param parallel Logical. Run grid evaluation in parallel via
#'   \code{foreach}. A parallel backend (e.g.,
#'   \code{doParallel::registerDoParallel()}) must be registered.
#'   Default: \code{TRUE}.
#' @param save_history Logical. Store per-iteration parameter histories
#'   in the final fit. Default: \code{FALSE}.
#'
#' @return A list with elements:
#'   \describe{
#'     \item{\code{fit}}{The final \code{spxlvb} model object (as
#'       returned by \code{\link{spxlvb}}).}
#'     \item{\code{criterion}}{Character string: the criterion used
#'       (\code{"elbo"}, \code{"cv"}, or \code{"validation"}).}
#'     \item{\code{optimal}}{Named list with elements
#'       \code{alpha_prior_precision} and \code{b_prior_precision}
#'       (the selected scalar values).}
#'     \item{\code{tuning_grid}}{Data frame with one row per grid
#'       combination and columns \code{alpha_prior_precision},
#'       \code{b_prior_precision}, and the tuning score(s) (ELBO
#'       for \code{"elbo"}; mean MSPE for \code{"cv"} and
#'       \code{"validation"}).}
#'     \item{\code{tuning_details}}{A list of criterion-specific
#'       diagnostics:
#'       \itemize{
#'         \item For \code{"cv"}: \code{per_fold_mspe} (array of
#'           per-fold prediction errors), \code{k} (number of folds).
#'         \item For \code{"validation"}: \code{oracle_mspe} (if
#'           \code{beta_true} was supplied).
#'         \item For \code{"elbo"}: empty list.
#'       }}
#'     \item{\code{refitted_on}}{Character string:
#'       \code{"training"} for \code{"elbo"} and \code{"cv"};
#'       \code{"training_plus_validation"} for \code{"validation"}.}
#'   }
#'
#' @details
#' \strong{How tuning works.}
#'
#' The function searches over a grid of hyperparameter values. When only
#' \code{alpha_prior_precision_grid} is specified (and
#' \code{b_prior_precision_grid = NULL}), a 1D search is performed and
#' the fixed \code{b_prior_precision} vector is used throughout. When
#' both grids are specified, all combinations are evaluated (2D search).
#'
#' For \code{criterion = "elbo"}, the model is fit once per grid point
#' on the full training data, and the fit with the highest ELBO is
#' returned directly (no redundant refit). This makes ELBO tuning
#' faster than CV, since the winning fit is already on the full data.
#'
#' For \code{criterion = "cv"}, data are split into \code{k} folds
#' using \code{caret::createFolds}. Each grid point is evaluated on all
#' folds, the mean MSPE is computed, and the optimal hyperparameters are
#' used for a final refit on the full training data.
#'
#' For \code{criterion = "validation"}, the model is fit on the training
#' data for each grid point, evaluated on the validation set, and the
#' final model is refit on the combined (training + validation) data.
#'
#' \strong{Initialisation.}
#'
#' Variational parameters are initialised via LASSO (see
#' \code{\link{get_initials_spxlvb}}). For \code{criterion = "cv"},
#' initialisations are computed once on the full training data and
#' shared across all folds and grid points, avoiding redundant
#' \code{cv.glmnet} calls.
#'
#' \strong{Parallelism.}
#'
#' When \code{parallel = TRUE}, grid points (or fold x grid-point
#' combinations for CV) are distributed via \code{foreach}. Register a
#' backend before calling:
#' \code{doParallel::registerDoParallel(cores = 4)}.
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 50; p <- 20
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- X[, 1:3] %*% c(1, -1, 0.5) + rnorm(n)
#'
#' # ELBO tuning (default, recommended)
#' result <- tune_spxlvb(X, Y,
#'   criterion = "elbo",
#'   alpha_prior_precision_grid = c(100, 1000),
#'   b_prior_precision_grid = c(1, 5),
#'   parallel = FALSE)
#' result$optimal
#' result$criterion
#'
#' # Cross-validation tuning
#' result_cv <- tune_spxlvb(X, Y,
#'   criterion = "cv", k = 3,
#'   alpha_prior_precision_grid = c(100, 1000),
#'   parallel = FALSE)
#' result_cv$optimal
#' }
#'
#' @seealso \code{\link{spxlvb}} for the low-level fitter with fixed
#'   hyperparameters.
#' @importFrom foreach foreach %do% %dopar%
#' @importFrom caret createFolds
#' @export
tune_spxlvb <- function(
    X,
    Y,
    criterion = c("elbo", "cv", "validation"),
    alpha_prior_precision_grid = c(0, 10^(3:7)),
    b_prior_precision_grid = NULL,
    b_prior_precision = rep(1, ncol(X)),
    k = 5L,
    X_validation = NULL,
    Y_validation = NULL,
    beta_true = NULL,
    mu_0 = NULL,
    omega_0 = NULL,
    c_pi_0 = NULL,
    d_pi_0 = NULL,
    tau_e = NULL,
    update_order = NULL,
    mu_alpha = NULL,
    standardize = TRUE,
    intercept = TRUE,
    max_iter = 100L,
    tol = 1e-5,
    seed = 12376,
    verbose = TRUE,
    parallel = TRUE,
    save_history = FALSE
) {
    criterion <- match.arg(criterion)
    p <- ncol(X)
    if (is.null(mu_alpha)) mu_alpha <- rep(1, p + 1)
    is_2d <- !is.null(b_prior_precision_grid)

    # --- Argument validation ---
    if (criterion == "validation") {
        if (is.null(X_validation) || is.null(Y_validation)) {
            stop("criterion = \"validation\" requires X_validation and Y_validation.")
        }
    }
    if (criterion == "cv") {
        if (k < 3L) stop("k must be at least 3 for cross-validation.")
        if (!is.null(X_validation) || !is.null(Y_validation)) {
            warning(
                "X_validation/Y_validation ignored when criterion = \"cv\". ",
                "Use criterion = \"validation\" to tune on a held-out set."
            )
        }
    }
    if (criterion == "elbo") {
        if (!is.null(X_validation) || !is.null(Y_validation)) {
            warning(
                "X_validation/Y_validation ignored when criterion = \"elbo\". ",
                "Use criterion = \"validation\" to tune on a held-out set."
            )
        }
    }

    if (parallel && !foreach::getDoParRegistered()) {
        if (verbose) {
            warning("No parallel backend registered. Falling back to sequential.")
        }
        parallel <- FALSE
    }
    `%loop_op%` <- if (parallel) foreach::`%dopar%` else foreach::`%do%`

    # --- Initialise on full training data ---
    set.seed(seed)

    scaled <- standardize_data(X, Y, standardize)

    initials <- get_initials_spxlvb(
        X = scaled$X_cs,
        Y = scaled$Y_c,
        mu_0 = mu_0,
        omega_0 = omega_0,
        c_pi_0 = c_pi_0,
        d_pi_0 = d_pi_0,
        tau_e = tau_e,
        update_order = update_order,
        seed = seed
    )

    # --- Build hyperparameter grid ---
    b_grid_values <- if (is_2d) b_prior_precision_grid else NA_real_
    hyper_grid <- expand.grid(
        alpha_prior_precision = alpha_prior_precision_grid,
        b_prior_precision = b_grid_values,
        stringsAsFactors = FALSE
    )

    if (verbose) {
        message(sprintf(
            "tune_spxlvb: criterion = \"%s\", %d grid points%s",
            criterion, nrow(hyper_grid),
            if (criterion == "cv") sprintf(", k = %d folds", k) else ""
        ))
    }

    # --- Dispatch by criterion ---
    result <- switch(criterion,
        elbo = tune_elbo(
            X = X, Y = Y, hyper_grid = hyper_grid, is_2d = is_2d,
            b_prior_precision = b_prior_precision,
            initials = initials, mu_alpha = mu_alpha,
            standardize = standardize, intercept = intercept,
            max_iter = max_iter, tol = tol, seed = seed,
            save_history = save_history,
            loop_op = `%loop_op%`
        ),
        cv = tune_cv(
            X = X, Y = Y, hyper_grid = hyper_grid, is_2d = is_2d,
            b_prior_precision = b_prior_precision,
            k = k, initials = initials, mu_alpha = mu_alpha,
            standardize = standardize, intercept = intercept,
            max_iter = max_iter, tol = tol, seed = seed,
            save_history = save_history,
            loop_op = `%loop_op%`
        ),
        validation = tune_validation(
            X = X, Y = Y,
            X_validation = X_validation, Y_validation = Y_validation,
            beta_true = beta_true,
            hyper_grid = hyper_grid, is_2d = is_2d,
            b_prior_precision = b_prior_precision,
            initials = initials, mu_alpha = mu_alpha,
            standardize = standardize, intercept = intercept,
            max_iter = max_iter, tol = tol, seed = seed,
            save_history = save_history,
            loop_op = `%loop_op%`
        )
    )

    result
}


# =====================================================================
# Internal helpers — one per criterion
# =====================================================================

#' @noRd
tune_elbo <- function(
    X, Y, hyper_grid, is_2d, b_prior_precision,
    initials, mu_alpha, standardize, intercept,
    max_iter, tol, seed, save_history, loop_op
) {
    `%loop_op%` <- loop_op
    p <- ncol(X)
    i <- NULL

    grid_fits <- foreach::foreach(
        i = seq_len(nrow(hyper_grid)),
        .packages = "spxlvb"
    ) %loop_op% {
        b_prec <- if (is_2d) {
            rep(hyper_grid$b_prior_precision[i], p)
        } else {
            b_prior_precision
        }

        fit_i <- spxlvb(
            X = X, Y = Y,
            mu_0 = initials$mu_0,
            omega_0 = initials$omega_0,
            c_pi_0 = initials$c_pi_0,
            d_pi_0 = initials$d_pi_0,
            tau_e = initials$tau_e,
            update_order = initials$update_order,
            mu_alpha = mu_alpha,
            alpha_prior_precision = hyper_grid$alpha_prior_precision[i],
            b_prior_precision = b_prec,
            standardize = standardize,
            intercept = intercept,
            max_iter = max_iter,
            tol = tol,
            save_history = save_history,
            seed = seed
        )

        list(elbo = fit_i$elbo, fit = fit_i)
    }

    elbos <- vapply(grid_fits, function(x) x$elbo, numeric(1))
    optimal_idx <- which.max(elbos)

    hyper_grid$elbo <- elbos
    optimal_alpha <- hyper_grid$alpha_prior_precision[optimal_idx]
    optimal_b <- if (is_2d) {
        hyper_grid$b_prior_precision[optimal_idx]
    } else {
        NA_real_
    }

    warn_grid_boundary(
        optimal_alpha, hyper_grid$alpha_prior_precision,
        is_2d, optimal_b,
        if (is_2d) hyper_grid$b_prior_precision else numeric(0)
    )

    list(
        fit = grid_fits[[optimal_idx]]$fit,
        criterion = "elbo",
        optimal = list(
            alpha_prior_precision = optimal_alpha,
            b_prior_precision = optimal_b
        ),
        tuning_grid = hyper_grid,
        tuning_details = list(),
        refitted_on = "training"
    )
}


#' @noRd
tune_cv <- function(
    X, Y, hyper_grid, is_2d, b_prior_precision,
    k, initials, mu_alpha, standardize, intercept,
    max_iter, tol, seed, save_history, loop_op
) {
    `%loop_op%` <- loop_op
    p <- ncol(X)
    set.seed(seed)
    fold_indices <- caret::createFolds(Y, k = k, list = TRUE)

    task_grid <- expand.grid(
        fold_id = seq_len(k),
        grid_idx = seq_len(nrow(hyper_grid)),
        stringsAsFactors = FALSE
    )

    idx <- NULL

    all_mspe <- foreach::foreach(
        idx = seq_len(nrow(task_grid)),
        .combine = "c",
        .packages = "spxlvb"
    ) %loop_op% {
        task <- task_grid[idx, ]
        train_idx <- unlist(fold_indices[-task$fold_id])
        test_idx <- unlist(fold_indices[task$fold_id])
        gi <- task$grid_idx

        b_prec <- if (is_2d) {
            rep(hyper_grid$b_prior_precision[gi], p)
        } else {
            b_prior_precision
        }

        fold_fit <- tryCatch(
            spxlvb(
                X = X[train_idx, , drop = FALSE],
                Y = Y[train_idx],
                mu_0 = initials$mu_0,
                omega_0 = initials$omega_0,
                c_pi_0 = initials$c_pi_0,
                d_pi_0 = initials$d_pi_0,
                tau_e = initials$tau_e,
                update_order = initials$update_order,
                mu_alpha = mu_alpha,
                alpha_prior_precision = hyper_grid$alpha_prior_precision[gi],
                b_prior_precision = b_prec,
                standardize = standardize,
                intercept = intercept,
                max_iter = max_iter,
                tol = tol,
                save_history = FALSE,
                seed = seed
            ),
            error = function(e) NULL
        )

        if (is.null(fold_fit) || is.null(fold_fit$beta)) return(NA_real_)

        y_pred <- if (intercept) {
            fold_fit$beta[1] + X[test_idx, , drop = FALSE] %*% fold_fit$beta[-1]
        } else {
            X[test_idx, , drop = FALSE] %*% fold_fit$beta
        }
        mean((Y[test_idx] - y_pred)^2)
    }

    per_fold_mspe <- matrix(
        all_mspe,
        nrow = k,
        ncol = nrow(hyper_grid),
        dimnames = list(
            paste0("fold_", seq_len(k)),
            paste0("grid_", seq_len(nrow(hyper_grid)))
        )
    )

    mean_mspe <- colMeans(per_fold_mspe, na.rm = TRUE)
    optimal_idx <- which.min(mean_mspe)

    if (length(optimal_idx) == 0L) {
        stop("Cross-validation produced no valid MSPE. Check model convergence.")
    }

    hyper_grid$mean_mspe <- mean_mspe

    optimal_alpha <- hyper_grid$alpha_prior_precision[optimal_idx]
    optimal_b_scalar <- if (is_2d) {
        hyper_grid$b_prior_precision[optimal_idx]
    } else {
        NA_real_
    }
    b_prec_final <- if (is_2d) rep(optimal_b_scalar, p) else b_prior_precision

    warn_grid_boundary(
        optimal_alpha, hyper_grid$alpha_prior_precision,
        is_2d, optimal_b_scalar,
        if (is_2d) hyper_grid$b_prior_precision else numeric(0)
    )

    final_fit <- spxlvb(
        X = X, Y = Y,
        mu_0 = initials$mu_0,
        omega_0 = initials$omega_0,
        c_pi_0 = initials$c_pi_0,
        d_pi_0 = initials$d_pi_0,
        tau_e = initials$tau_e,
        update_order = initials$update_order,
        mu_alpha = mu_alpha,
        alpha_prior_precision = optimal_alpha,
        b_prior_precision = b_prec_final,
        standardize = standardize,
        intercept = intercept,
        max_iter = max_iter,
        tol = tol,
        save_history = save_history,
        seed = seed
    )

    list(
        fit = final_fit,
        criterion = "cv",
        optimal = list(
            alpha_prior_precision = optimal_alpha,
            b_prior_precision = optimal_b_scalar
        ),
        tuning_grid = hyper_grid,
        tuning_details = list(
            per_fold_mspe = per_fold_mspe,
            k = k
        ),
        refitted_on = "training"
    )
}


#' @noRd
tune_validation <- function(
    X, Y, X_validation, Y_validation, beta_true,
    hyper_grid, is_2d, b_prior_precision,
    initials, mu_alpha, standardize, intercept,
    max_iter, tol, seed, save_history, loop_op
) {
    `%loop_op%` <- loop_op
    p <- ncol(X)
    i <- NULL

    grid_results <- foreach::foreach(
        i = seq_len(nrow(hyper_grid)),
        .combine = "rbind",
        .packages = "spxlvb"
    ) %loop_op% {
        b_prec <- if (is_2d) {
            rep(hyper_grid$b_prior_precision[i], p)
        } else {
            b_prior_precision
        }

        fit_i <- spxlvb(
            X = X, Y = Y,
            mu_0 = initials$mu_0,
            omega_0 = initials$omega_0,
            c_pi_0 = initials$c_pi_0,
            d_pi_0 = initials$d_pi_0,
            tau_e = initials$tau_e,
            update_order = initials$update_order,
            mu_alpha = mu_alpha,
            alpha_prior_precision = hyper_grid$alpha_prior_precision[i],
            b_prior_precision = b_prec,
            standardize = standardize,
            intercept = intercept,
            max_iter = max_iter,
            tol = tol,
            save_history = FALSE,
            seed = seed
        )

        y_val_hat <- if (intercept) {
            fit_i$beta[1] + X_validation %*% fit_i$beta[-1]
        } else {
            X_validation %*% fit_i$beta
        }

        mspe <- mean((Y_validation - y_val_hat)^2)

        oracle_mspe <- NA_real_
        if (!is.null(beta_true)) {
            eta_true <- if (length(beta_true) == p) {
                X_validation %*% beta_true
            } else if (length(beta_true) == (p + 1)) {
                beta_true[1] + X_validation %*% beta_true[-1]
            } else {
                stop("beta_true must have length p or p+1.")
            }
            oracle_mspe <- mean((eta_true - y_val_hat)^2)
        }

        data.frame(
            validation_mspe = mspe,
            oracle_mspe = oracle_mspe,
            elbo = fit_i$elbo
        )
    }

    hyper_grid <- cbind(hyper_grid, grid_results)
    optimal_idx <- which.min(hyper_grid$validation_mspe)

    optimal_alpha <- hyper_grid$alpha_prior_precision[optimal_idx]
    optimal_b_scalar <- if (is_2d) {
        hyper_grid$b_prior_precision[optimal_idx]
    } else {
        NA_real_
    }
    b_prec_final <- if (is_2d) rep(optimal_b_scalar, p) else b_prior_precision

    warn_grid_boundary(
        optimal_alpha, hyper_grid$alpha_prior_precision,
        is_2d, optimal_b_scalar,
        if (is_2d) hyper_grid$b_prior_precision else numeric(0)
    )

    X_combined <- rbind(X, X_validation)
    Y_combined <- c(as.numeric(Y), as.numeric(Y_validation))

    final_fit <- spxlvb(
        X = X_combined,
        Y = Y_combined,
        mu_alpha = mu_alpha,
        alpha_prior_precision = optimal_alpha,
        b_prior_precision = b_prec_final,
        standardize = standardize,
        intercept = intercept,
        max_iter = max_iter,
        tol = tol,
        save_history = save_history,
        seed = seed
    )

    details <- list()
    if (!is.null(beta_true)) {
        details$oracle_mspe <- hyper_grid$oracle_mspe
    }

    list(
        fit = final_fit,
        criterion = "validation",
        optimal = list(
            alpha_prior_precision = optimal_alpha,
            b_prior_precision = optimal_b_scalar
        ),
        tuning_grid = hyper_grid,
        tuning_details = details,
        refitted_on = "training_plus_validation"
    )
}

warn_grid_boundary <- function(optimal_alpha, alpha_grid, is_2d,
                               optimal_b, b_grid) {
    alpha_at_min <- optimal_alpha == min(alpha_grid)
    alpha_at_max <- optimal_alpha == max(alpha_grid)
    if (alpha_at_min) {
        warning(
            "Optimal alpha_prior_precision (", optimal_alpha,
            ") is at the minimum of the search grid. ",
            "Consider extending the grid to smaller values.",
            call. = FALSE
        )
    }
    if (alpha_at_max) {
        warning(
            "Optimal alpha_prior_precision (", optimal_alpha,
            ") is at the maximum of the search grid. ",
            "Consider extending the grid to larger values.",
            call. = FALSE
        )
    }
    if (is_2d) {
        b_at_min <- optimal_b == min(b_grid)
        b_at_max <- optimal_b == max(b_grid)
        if (b_at_min) {
            warning(
                "Optimal b_prior_precision (", optimal_b,
                ") is at the minimum of the search grid. ",
                "Consider extending the grid to smaller values.",
                call. = FALSE
            )
        }
        if (b_at_max) {
            warning(
                "Optimal b_prior_precision (", optimal_b,
                ") is at the maximum of the search grid. ",
                "Consider extending the grid to larger values.",
                call. = FALSE
            )
        }
    }
}

utils::globalVariables(c("i", "idx"))
