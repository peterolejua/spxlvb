#' @title Grid Search and Final Model Fitting for spxlvb (Deprecated)
#'
#' @description
#' \ifelse{html}{\out{<span style="color:red">[Deprecated]</span>}}{\strong{[Deprecated]}}
#' Use \code{\link{tune_spxlvb}} instead.
#'
#' This function performs grid search to determine optimal hyperparameters.
#' It has been replaced by \code{\link{tune_spxlvb}}, which provides an
#' explicit \code{criterion} argument (\code{"elbo"}, \code{"cv"}, or
#' \code{"validation"}).
#'
#' @inheritParams tune_spxlvb
#' @param X_validation Optional validation design matrix.
#' @param Y_validation Optional validation response.
#' @param beta_true Optional true coefficients for oracle MSE.
#' @param save_history Logical. Store per-iteration histories.
#'
#' @return A list matching the legacy return structure. See
#'   \code{\link{tune_spxlvb}} for the recommended interface.
#'
#' @seealso \code{\link{tune_spxlvb}}
#' @importFrom foreach foreach %do% %dopar%
#' @export
grid_search_spxlvb_fit <- function(
    X,
    Y,
    X_validation = NULL,
    Y_validation = NULL,
    beta_true = NULL,
    mu_0 = NULL,
    omega_0 = NULL,
    c_pi_0 = NULL,
    d_pi_0 = NULL,
    tau_e = NULL,
    update_order = NULL,
    mu_alpha = rep(1, ncol(X) + 1),
    alpha_prior_precision_grid = c(10, 50, 100, 400, 1000),
    b_prior_precision_grid = seq(0.001, 5, length.out = 5),
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

    has_validation <- !is.null(X_validation) && !is.null(Y_validation)
    selected_criterion <- if (has_validation) "validation" else "elbo"

    result <- tune_spxlvb(
        X = X, Y = Y,
        criterion = selected_criterion,
        alpha_prior_precision_grid = alpha_prior_precision_grid,
        b_prior_precision_grid = b_prior_precision_grid,
        X_validation = X_validation,
        Y_validation = Y_validation,
        beta_true = beta_true,
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
    legacy_grid <- result$tuning_grid
    if (!"elbo" %in% names(legacy_grid)) legacy_grid$elbo <- NA_real_
    if (!"mse_validation_y" %in% names(legacy_grid)) {
        legacy_grid$mse_validation_y <- if ("validation_mspe" %in% names(legacy_grid)) {
            legacy_grid$validation_mspe
        } else {
            NA_real_
        }
    }
    if (!"mse_validation_Xbeta" %in% names(legacy_grid)) {
        legacy_grid$mse_validation_Xbeta <- if ("oracle_mspe" %in% names(legacy_grid)) {
            legacy_grid$oracle_mspe
        } else {
            NA_real_
        }
    }

    list(
        hyper_grid = legacy_grid,
        optimal_hyper = c(
            result$optimal$alpha_prior_precision,
            result$optimal$b_prior_precision
        ),
        fit_spxlvb = result$fit,
        selection_criterion = selected_criterion,
        refitted_on_combined = (result$refitted_on == "training_plus_validation")
    )
}
