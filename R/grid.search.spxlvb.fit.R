#' @title Grid search and Final Model Fitting for spxlvb
#' @description This function performs grid search to determine optimal hyperparameters.
#' Optimality is determined by: 1. mse_validation_Xbeta (if available), 2. mse_validation_y (if available), 3. elbo.
#' @param X A design matrix (training).
#' @param Y A response vector (training).
#' @param X_validation Optional design matrix for the validation set.
#' @param Y_validation Optional response vector for the validation set.
#' @param beta_true Optional vector of true coefficients (length p or p+1).
#' @param mu_0 Initial variational mean.
#' @param omega_0 Initial variational probability.
#' @param c_pi_0 Prior parameter for pi (shape1).
#' @param d_pi_0 Prior parameter for pi (shape2).
#' @param tau_e Initial precision of errors.
#' @param update_order Numeric vector for update order.
#' @param mu_alpha Prior mean for alpha.
#' @param alpha_prior_precision_grid Grid for alpha prior precision.
#' @param b_prior_precision_grid Grid for slab prior precision.
#' @param standardize Logical. Default is TRUE.
#' @param intercept Logical. Default is TRUE.
#' @param max_iter Maximum iterations.
#' @param tol Convergence tolerance.
#' @param seed Seed for reproducibility.
#' @param verbose Logical, if TRUE, prints progress.
#' @param parallel Logical, if TRUE, search in parallel.
#' @return A list with elements \code{hyper_grid} (data frame of all grid
#'   combinations and their metrics), \code{optimal_hyper} (the selected
#'   hyperparameter values), \code{fit_spxlvb} (the final fitted model),
#'   \code{selection_criterion} (the criterion used), and
#'   \code{refitted_on_combined} (logical).
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 50; p <- 20
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- X[, 1:3] %*% c(1, -1, 0.5) + rnorm(n)
#' result <- grid.search.spxlvb.fit(X = X, Y = Y,
#'   alpha_prior_precision_grid = c(100, 1000),
#'   b_prior_precision_grid = c(1, 5),
#'   parallel = FALSE)
#' }
#' @importFrom foreach foreach %do% %dopar%
#' @export
grid.search.spxlvb.fit <- function(
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
  parallel = TRUE
) {
  set.seed(seed)

  # Standardize data
  std <- standardize_data(X, Y, standardize)
  X_cs <- std$X_cs
  Y_c <- std$Y_c

  # Get initials for the full dataset
  initials <- get.initials.spxlvb(
    X = X_cs,
    Y = Y_c,
    mu_0 = mu_0,
    omega_0 = omega_0,
    c_pi_0 = c_pi_0,
    d_pi_0 = d_pi_0,
    tau_e = tau_e,
    update_order = update_order,
    seed = seed
  )

  hyper_grid <- expand.grid(
    alpha_prior_precision = alpha_prior_precision_grid,
    b_prior_precision = b_prior_precision_grid
  )

  # --- MODIFIED PARALLEL LOGIC ---
  # Define the loop operator based on the 'parallel' argument
  `%loop_op%` <- if (parallel) foreach::`%dopar%` else foreach::`%do%`

  if (verbose) {
    message(sprintf(
      "Starting grid search over %d combinations (Parallel: %s)",
      nrow(hyper_grid), parallel
    ))
  }
  # Grid search loop
  # Fix for "no visible binding for global variable" in R CMD check
  i <- NULL

  grid_results <- foreach::foreach(
    i = seq_len(nrow(hyper_grid)),
    .combine = "rbind",
    .packages = c("spxlvb")
  ) %loop_op% {
    fit_temp <- spxlvb(
      X = X,
      Y = Y,
      mu_0 = initials$mu_0,
      omega_0 = initials$omega_0,
      c_pi_0 = initials$c_pi_0,
      d_pi_0 = initials$d_pi_0,
      tau_e = initials$tau_e,
      update_order = initials$update_order,
      mu_alpha = mu_alpha,
      alpha_prior_precision = hyper_grid[i, 1],
      b_prior_precision = rep(hyper_grid[i, 2], ncol(X)),
      standardize = standardize,
      intercept = intercept,
      max_iter = max_iter,
      tol = tol,
      seed = seed
    )

    # Base metrics
    res_row <- data.frame(
      elbo = fit_temp$elbo,
      mse_validation_y = NA,
      mse_validation_Xbeta = NA
    )

    # If test features are provided, compute predictions
    if (!is.null(X_validation)) {
      y_validation_hat <- if (intercept) {
        fit_temp$beta[1] + X_validation %*% fit_temp$beta[-1]
      } else {
        X_validation %*% fit_temp$beta
      }

      # 1. Prediction MSE
      if (!is.null(Y_validation)) {
        res_row$mse_validation_y <- mean((Y_validation - y_validation_hat)^2)
      }

      # 2. Linear Predictor MSE (Xbeta)
      if (!is.null(beta_true)) {
        eta_true <- if (length(beta_true) == ncol(X)) {
          X_validation %*% beta_true
        } else if (length(beta_true) == (ncol(X) + 1)) {
          beta_true[1] + X_validation %*% beta_true[-1]
        } else {
          stop("Dimension mismatch: beta_true must be length p or p+1")
        }
        res_row$mse_validation_Xbeta <- mean((eta_true - y_validation_hat)^2)
      }
    }

    return(res_row)
  }

  # Merge metrics back into the grid
  hyper_grid <- cbind(hyper_grid, grid_results)

  # Determine Optimal Index based on hierarchy
  # Priority: mse_validation_Xbeta (min) > mse_validation_y (min) > elbo (max)
  if (any(!is.na(hyper_grid$mse_validation_Xbeta))) {
    optimal_idx <- which.min(hyper_grid$mse_validation_Xbeta)
    criterion_used <- "mse_validation_Xbeta"
  } else if (any(!is.na(hyper_grid$mse_validation_y))) {
    optimal_idx <- which.min(hyper_grid$mse_validation_y)
    criterion_used <- "mse_validation_y"
  } else {
    optimal_idx <- which.max(hyper_grid$elbo)
    criterion_used <- "elbo"
  }

  optimal_hyper <- as.numeric(hyper_grid[optimal_idx, c("alpha_prior_precision", "b_prior_precision")])

  # --- REFIT STEP ON COMBINED DATA ---
  if (!is.null(X_validation) && !is.null(Y_validation)) {
    if (verbose) message("Refitting final model on combined Training + Validation data...")
    X_final <- rbind(X, X_validation)
    Y_final <- if (is.matrix(Y)) rbind(Y, Y_validation) else c(Y, Y_validation)
    refitted <- TRUE
  } else {
    if (verbose) message("No validation data provided for refit. Using training data only.")
    X_final <- X
    Y_final <- Y
    refitted <- FALSE
  }


  # Final fit with optimal values
  fit_spxlvb <- spxlvb(
    X = X_final,
    Y = Y_final,
    mu_alpha = mu_alpha,
    alpha_prior_precision = optimal_hyper[1],
    b_prior_precision = rep(optimal_hyper[2], ncol(X_final)),
    standardize = standardize,
    intercept = intercept,
    max_iter = max_iter,
    tol = tol,
    seed = seed
  )

  if (verbose) {
    message(paste0("Optimal model chosen based on: ", criterion_used))
  }

  return(list(
    "hyper_grid" = hyper_grid,
    "optimal_hyper" = optimal_hyper,
    "fit_spxlvb" = fit_spxlvb,
    "selection_criterion" = criterion_used,
    "refitted_on_combined" = refitted
  ))
}
