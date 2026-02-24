#' @title Cross-Validation for spxlvb (Deprecated)
#'
#' @description
#' \ifelse{html}{\out{<span style="color:red">[Deprecated]</span>}}{\strong{[Deprecated]}}
#' Use \code{\link{tune_spxlvb}(criterion = "cv")} instead.
#'
#' Performs k-fold cross-validation for the \code{spxlvb} model to
#' optimise hyperparameters. Returns only the CV diagnostics (no final
#' refit). For CV + refit, use \code{\link{tune_spxlvb}(criterion = "cv")}.
#'
#' @param k Integer, the number of folds. Must be at least 3.
#' @param X Numeric design matrix (\eqn{n \times p}).
#' @param Y Numeric response vector of length \eqn{n}.
#' @param mu_0 Optional initial variational means.
#' @param omega_0 Optional initial variational probabilities.
#' @param c_pi_0 Optional prior shape1 for \eqn{\pi}.
#' @param d_pi_0 Optional prior shape2 for \eqn{\pi}.
#' @param tau_e Optional initial error precision.
#' @param update_order Optional coordinate update order (0-indexed).
#' @param mu_alpha Optional prior means for expansion parameters.
#' @param alpha_prior_precision_grid Grid of \eqn{\tau_\alpha} values.
#' @param b_prior_precision_grid Optional grid of \eqn{\tau_b} values
#'   for 2D search.
#' @param b_prior_precision Fixed slab precisions (used when
#'   \code{b_prior_precision_grid = NULL}).
#' @param standardize Logical. Default \code{TRUE}.
#' @param intercept Logical. Default \code{TRUE}.
#' @param max_iter Maximum iterations per fit.
#' @param tol Convergence tolerance.
#' @param seed Seed for reproducibility.
#' @param verbose Logical.
#' @param parallel Logical.
#'
#' @return A list with CV diagnostics (no fitted model).
#'
#' @seealso \code{\link{tune_spxlvb}}
#' @importFrom caret createFolds
#' @importFrom foreach foreach %do% %dopar%
#' @importFrom utils globalVariables
#' @export
cv_spxlvb <- function(
    k = 5,
    X,
    Y,
    mu_0 = NULL,
    omega_0 = NULL,
    c_pi_0 = NULL,
    d_pi_0 = NULL,
    tau_e = NULL,
    update_order = NULL,
    mu_alpha = NULL,
    alpha_prior_precision_grid = c(0, 10^(3:7)),
    b_prior_precision_grid = NULL,
    b_prior_precision = rep(1, ncol(X)),
    standardize = TRUE,
    intercept = TRUE,
    max_iter = 100L,
    tol = 1e-5,
    seed = 12376,
    verbose = TRUE,
    parallel = TRUE
) {
    .Deprecated("tune_spxlvb")

    idx <- NULL

    if (k < 3) stop("The number of folds 'k' must be at least 3.")
    if (length(alpha_prior_precision_grid) < 2) {
        stop("alpha_prior_precision_grid must contain at least two values for cross-validation.")
    }

    is_2d <- !is.null(b_prior_precision_grid)
    if (is_2d && length(b_prior_precision_grid) < 2) {
        stop("b_prior_precision_grid must contain at least two values for cross-validation.")
    }

    set.seed(seed)
    p <- ncol(X)

    if (is.null(mu_alpha)) mu_alpha <- rep(1, p + 1)
    ordered_alpha_grid <- sort(alpha_prior_precision_grid)
    ordered_b_grid <- if (is_2d) sort(b_prior_precision_grid) else NULL

    fold_indices <- caret::createFolds(Y, k = k, list = TRUE)

    if (is_2d) {
        task_grid <- expand.grid(
            fold_id = seq_len(k),
            alpha_val = ordered_alpha_grid,
            b_val = ordered_b_grid,
            stringsAsFactors = FALSE
        )
    } else {
        task_grid <- expand.grid(
            fold_id = seq_len(k),
            alpha_val = ordered_alpha_grid,
            stringsAsFactors = FALSE
        )
    }

    if (parallel && !foreach::getDoParRegistered()) {
        if (verbose) {
            warning("Parallel execution requested but no parallel backend is registered. Falling back to sequential.")
        }
        parallel <- FALSE
    }

    `%loop_op%` <- if (parallel) foreach::`%dopar%` else foreach::`%do%`

    if (verbose) {
        message(sprintf(
            "Starting CV: %d folds x %d grid points = %d total fits.",
            k,
            if (is_2d) length(ordered_alpha_grid) * length(ordered_b_grid)
            else length(ordered_alpha_grid),
            nrow(task_grid)
        ))
    }

    all_results <- foreach::foreach(
        idx = seq_len(nrow(task_grid)),
        .combine = "c",
        .packages = c("spxlvb")
    ) %loop_op% {
        current_task <- task_grid[idx, ]
        train_idx <- unlist(fold_indices[-current_task$fold_id])
        test_idx <- unlist(fold_indices[current_task$fold_id])

        b_prec <- if (is_2d) rep(current_task$b_val, p) else b_prior_precision

        fit <- tryCatch(
            {
                spxlvb(
                    X = X[train_idx, , drop = FALSE],
                    Y = Y[train_idx],
                    mu_0 = mu_0,
                    omega_0 = omega_0,
                    c_pi_0 = c_pi_0,
                    d_pi_0 = d_pi_0,
                    tau_e = tau_e,
                    update_order = update_order,
                    mu_alpha = mu_alpha,
                    alpha_prior_precision = current_task$alpha_val,
                    b_prior_precision = b_prec,
                    standardize = standardize,
                    intercept = intercept,
                    max_iter = max_iter,
                    tol = tol,
                    save_history = FALSE,
                    seed = seed
                )
            },
            error = function(e) NULL
        )

        if (is.null(fit) || is.null(fit$beta)) {
            return(NA_real_)
        } else {
            X_test_mat <- X[test_idx, , drop = FALSE]
            y_pred <- if (intercept) {
                fit$beta[1] + X_test_mat %*% fit$beta[-1]
            } else {
                X_test_mat %*% fit$beta
            }
            return(mean((Y[test_idx] - y_pred)^2, na.rm = TRUE))
        }
    }

    if (is_2d) {
        n_alpha <- length(ordered_alpha_grid)
        n_b <- length(ordered_b_grid)
        epe_test_k <- array(
            all_results,
            dim = c(k, n_alpha, n_b),
            dimnames = list(
                paste0("Fold", seq_len(k)),
                as.character(ordered_alpha_grid),
                as.character(ordered_b_grid)
            )
        )
        CVE <- apply(epe_test_k, c(2, 3), mean, na.rm = TRUE)
        opt_idx <- which(CVE == min(CVE, na.rm = TRUE), arr.ind = TRUE)[1, ]

        if (length(opt_idx) == 0) {
            stop("Cross-validation failed to produce valid MSE results. Check model convergence.")
        }

        list(
            ordered_alpha_prior_precision_grid = ordered_alpha_grid,
            ordered_b_prior_precision_grid = ordered_b_grid,
            epe_test_k = epe_test_k,
            CVE = CVE,
            alpha_prior_precision_grid_opt = ordered_alpha_grid[opt_idx[1]],
            b_prior_precision_grid_opt = ordered_b_grid[opt_idx[2]]
        )
    } else {
        epe_test_k <- matrix(
            all_results,
            nrow = k,
            ncol = length(ordered_alpha_grid),
            dimnames = list(
                paste0("Fold", seq_len(k)),
                as.character(ordered_alpha_grid)
            )
        )
        CVE <- colMeans(epe_test_k, na.rm = TRUE)
        opt_idx <- which.min(CVE)

        if (length(opt_idx) == 0) {
            stop("Cross-validation failed to produce valid MSE results. Check model convergence.")
        }

        list(
            ordered_alpha_prior_precision_grid = ordered_alpha_grid,
            epe_test_k = epe_test_k,
            CVE = CVE,
            alpha_prior_precision_grid_opt = ordered_alpha_grid[opt_idx]
        )
    }
}

#' @title Cross-Validation and Final Model Fitting for spxlvb (Deprecated)
#'
#' @description
#' \ifelse{html}{\out{<span style="color:red">[Deprecated]</span>}}{\strong{[Deprecated]}}
#' Use \code{\link{tune_spxlvb}(criterion = "cv")} instead.
#'
#' Performs k-fold cross-validation and refits on the full dataset.
#'
#' @inheritParams cv_spxlvb
#' @param save_history Logical. Store per-iteration histories.
#'
#' @return A list with CV results and the final fitted model.
#' @seealso \code{\link{tune_spxlvb}}
#' @export
cv_spxlvb_fit <- function(
    k = 5,
    X,
    Y,
    mu_0 = NULL,
    omega_0 = NULL,
    c_pi_0 = NULL,
    d_pi_0 = NULL,
    tau_e = NULL,
    update_order = NULL,
    mu_alpha = NULL,
    alpha_prior_precision_grid = c(0, 10^(3:7)),
    b_prior_precision_grid = NULL,
    b_prior_precision = rep(1, ncol(X)),
    standardize = TRUE,
    intercept = TRUE,
    max_iter = 100L,
    tol = 1e-5,
    seed = 12376,
    verbose = TRUE,
    parallel = TRUE,
    save_history = FALSE
) {
    .Deprecated("tune_spxlvb")

    result <- tune_spxlvb(
        X = X, Y = Y,
        criterion = "cv",
        alpha_prior_precision_grid = alpha_prior_precision_grid,
        b_prior_precision_grid = b_prior_precision_grid,
        b_prior_precision = b_prior_precision,
        k = k,
        mu_0 = mu_0,
        omega_0 = omega_0,
        c_pi_0 = c_pi_0,
        d_pi_0 = d_pi_0,
        tau_e = tau_e,
        update_order = update_order,
        mu_alpha = mu_alpha,
        standardize = standardize,
        intercept = intercept,
        max_iter = max_iter,
        tol = tol,
        seed = seed,
        verbose = verbose,
        parallel = parallel,
        save_history = save_history
    )

    # Reshape to legacy return structure
    is_2d <- !is.null(b_prior_precision_grid)
    ordered_alpha_grid <- sort(alpha_prior_precision_grid)
    ordered_b_grid <- if (is_2d) sort(b_prior_precision_grid) else NULL

    legacy <- list(
        ordered_alpha_prior_precision_grid = ordered_alpha_grid,
        epe_test_k = result$tuning_details$per_fold_mspe,
        CVE = result$tuning_grid$mean_mspe,
        alpha_prior_precision_grid_opt = result$optimal$alpha_prior_precision,
        fit_spxlvb = result$fit
    )

    if (is_2d) {
        legacy$ordered_b_prior_precision_grid <- ordered_b_grid
        legacy$b_prior_precision_grid_opt <- result$optimal$b_prior_precision
    }

    legacy
}

utils::globalVariables(c("idx", "i"))
