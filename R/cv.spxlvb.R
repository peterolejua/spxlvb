#' @title Cross-validation for Sparse Parameter Exploded Variational Bayes (spxlvb)
#'
#' @description Performs k-fold cross-validation for the \code{spxlvb} model to
#'   optimize the \code{alpha_prior_precision} hyperparameter. This implementation
#'   is optimized for high-performance computing by flattening the cross-validation
#'   folds and hyperparameter grid into a single parallelizable task list.
#'
#' @param k Integer, the number of folds for cross-validation. Must be at least 3.
#' @param X A numeric design matrix of dimension \eqn{n \times p}.
#' @param Y A numeric response vector of length \eqn{n}.
#' @param mu_0 Optional numeric vector of length \eqn{p}. Initial variational means.
#' @param omega_0 Optional numeric vector of length \eqn{p}. Initial variational probabilities.
#' @param c_pi_0 Optional numeric. Prior shape1 parameter for \eqn{\pi}.
#' @param d_pi_0 Optional numeric. Prior shape2 parameter for \eqn{\pi}.
#' @param tau_e Optional numeric. Initial precision of errors.
#' @param update_order Optional integer vector specifying the coordinate update sequence (0-indexed for C++).
#' @param mu_alpha Optional numeric vector of length \eqn{p+1}. Prior means for the expansion parameters.
#' @param alpha_prior_precision_grid A numeric vector of values to cross-validate over.
#'   Defaults to \code{c(0, 10^(3:7))}.
#' @param b_prior_precision Numeric. Initial prior precision for the slab component. Defaults to 400.
#' @param standardize Logical. Should the design matrix be standardized? Defaults to TRUE.
#' @param intercept Logical. Should an intercept be included in the model? Defaults to TRUE.
#' @param max_iter Integer. Maximum number of iterations for each model fit.
#' @param tol Numeric. Convergence tolerance for the Variational Bayes algorithm.
#' @param seed Integer. Seed for reproducibility of data splitting.
#' @param verbose Logical. If TRUE, progress messages are printed to the console.
#' @param parallel Logical. If TRUE, execution is performed in parallel using the
#'   currently registered backend (e.g., via \code{doParallel}).
#'
#' @return A list containing:
#' \item{ordered_alpha_prior_precision_grid}{The sorted hyperparameter grid used.}
#' \item{epe_test_k}{A matrix of prediction errors for each fold and grid point.}
#' \item{CVE}{Mean Cross-Validation Error for each grid point.}
#' \item{alpha_prior_precision_grid_opt}{The value from the grid that minimized the CVE.}
#'
#' @details To use parallel processing, a backend must be registered before calling
#' this function. For example: \code{doParallel::registerDoParallel(cores = 4)}.
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 50; p <- 20
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- X[, 1:3] %*% c(1, -1, 0.5) + rnorm(n)
#' result <- cv.spxlvb(k = 3, X = X, Y = Y,
#'   alpha_prior_precision_grid = c(100, 1000), parallel = FALSE)
#' }
#' @importFrom caret createFolds
#' @importFrom foreach foreach %do% %dopar%
#' @importFrom utils globalVariables
#' @export

cv.spxlvb <- function(
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
  b_prior_precision = rep(1, ncol(X)),
  standardize = TRUE,
  intercept = TRUE,
  max_iter = 100L,
  tol = 1e-5,
  seed = 12376,
  verbose = TRUE,
  parallel = TRUE
) {
  # Fix R CMD check notes for global variables used in foreach
  idx <- NULL

  # 1. Validation and Setup
  if (k < 3) stop("The number of folds 'k' must be at least 3.")
  if (length(alpha_prior_precision_grid) < 2) {
    stop("alpha_prior_precision_grid must contain at least two values for cross-validation.")
  }

  set.seed(seed)
  p <- ncol(X)

  if (is.null(mu_alpha)) mu_alpha <- rep(1, p + 1)
  ordered_grid <- sort(alpha_prior_precision_grid)

  # 2. Data Partitioning
  fold_indices <- caret::createFolds(Y, k = k, list = TRUE)

  # Flattening the grid for maximum parallel throughput
  task_grid <- expand.grid(
    fold_id = seq_len(k),
    alpha_val = ordered_grid,
    stringsAsFactors = FALSE
  )

  # 3. Parallel Logic Check
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
      k, length(ordered_grid), nrow(task_grid)
    ))
  }

  # 4. Execution
  all_results <- foreach::foreach(
    idx = seq_len(nrow(task_grid)),
    .combine = "c",
    .packages = c("spxlvb")
  ) %loop_op% {
    current_task <- task_grid[idx, ]
    train_idx <- unlist(fold_indices[-current_task$fold_id])
    test_idx <- unlist(fold_indices[current_task$fold_id])

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
          b_prior_precision = b_prior_precision,
          standardize = standardize,
          intercept = intercept,
          max_iter = max_iter,
          tol = tol,
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
      # mean is a base function, no stats:: prefix needed
      return(mean((Y[test_idx] - y_pred)^2, na.rm = TRUE))
    }
  }

  # 5. Result Reconstruction
  epe_test_k <- matrix(
    all_results,
    nrow = k,
    ncol = length(ordered_grid),
    dimnames = list(paste0("Fold", seq_len(k)), as.character(ordered_grid))
  )

  CVE <- colMeans(epe_test_k, na.rm = TRUE)
  opt_idx <- which.min(CVE)

  if (length(opt_idx) == 0) {
    stop("Cross-validation failed to produce valid MSE results. Check model convergence.")
  }

  list(
    ordered_alpha_prior_precision_grid = ordered_grid,
    epe_test_k = epe_test_k,
    CVE = CVE,
    alpha_prior_precision_grid_opt = ordered_grid[opt_idx]
  )
}

#' @title Cross-validation and Final Model Fitting for spxlvb
#'
#' @description Performs k-fold cross-validation to determine the optimal
#'   \code{alpha_prior_precision} and then fits a final \code{spxlvb} model
#'   to the full dataset using the identified optimal hyperparameter.
#'
#' @inheritParams cv.spxlvb
#' @return The final fitted \code{spxlvb} model object based on the full dataset.
#'   If the final fit fails, the cross-validation results are returned with a warning.
#'
#' @details This function orchestrates the cross-validation process and the final model fit.
#'   It first identifies initial values for the full dataset, uses \code{cv.spxlvb} to find
#'   the optimal hyperparameter, and finally fits the model to the complete dataset.
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 50; p <- 20
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- X[, 1:3] %*% c(1, -1, 0.5) + rnorm(n)
#' fit <- cv.spxlvb.fit(k = 3, X = X, Y = Y,
#'   alpha_prior_precision_grid = c(100, 1000), parallel = FALSE)
#' }
#' @seealso \code{\link{cv.spxlvb}}, \code{\link{spxlvb}}
#' @export

cv.spxlvb.fit <- function(
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
  b_prior_precision = rep(1, ncol(X)),
  standardize = TRUE,
  intercept = TRUE,
  max_iter = 100L,
  tol = 1e-5,
  seed = 12376,
  verbose = TRUE,
  parallel = TRUE
) {
  set.seed(seed)

  # Standardize data before generating initials
  std <- standardize_data(X, Y, standardize)
  X_cs <- std$X_cs
  Y_c <- std$Y_c

  # 1. Generate initials based on the correctly scaled full dataset
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

  # 2. Perform cross-validation to find the optimal hyperparameter
  cv_results <- cv.spxlvb(
    k = k,
    X = X, # Pass raw X, as spxlvb will internally scale the folds
    Y = Y, # Pass raw Y
    mu_0 = initials$mu_0,
    omega_0 = initials$omega_0,
    c_pi_0 = initials$c_pi_0,
    d_pi_0 = initials$d_pi_0,
    tau_e = initials$tau_e,
    update_order = initials$update_order,
    mu_alpha = mu_alpha,
    alpha_prior_precision_grid = alpha_prior_precision_grid,
    b_prior_precision = b_prior_precision,
    standardize = standardize,
    intercept = intercept,
    max_iter = max_iter,
    tol = tol,
    seed = seed,
    verbose = verbose,
    parallel = parallel
  )

  # 3. Fit final model on the complete dataset using optimal hyperparameter
  if (verbose) {
    message(paste(
      "Fitting final model with optimal alpha_prior_precision:",
      cv_results$alpha_prior_precision_grid_opt
    ))
  }

  fit_spxlvb <- tryCatch(
    {
      spxlvb(
        X = X,
        Y = Y,
        mu_0 = initials$mu_0,
        omega_0 = initials$omega_0,
        c_pi_0 = initials$c_pi_0,
        d_pi_0 = initials$d_pi_0,
        tau_e = initials$tau_e,
        update_order = initials$update_order,
        mu_alpha = mu_alpha,
        alpha_prior_precision = cv_results$alpha_prior_precision_grid_opt,
        b_prior_precision = b_prior_precision,
        standardize = standardize,
        intercept = intercept,
        max_iter = max_iter,
        tol = tol,
        seed = seed
      )
    },
    error = function(e) {
      warning("Final spxlvb model fit failed. Returning cross-validation summary.")
      return(NULL)
    }
  )

  if (is.null(fit_spxlvb)) {
    return(cv_results)
  }

  return(
    list(
      ordered_alpha_prior_precision_grid = cv_results$ordered_alpha_prior_precision_grid,
      epe_test_k = cv_results$epe_test_k,
      CVE = cv_results$CVE,
      alpha_prior_precision_grid_opt = cv_results$alpha_prior_precision_grid_opt,
      fit_spxlvb = fit_spxlvb
    )
  )
}

# Declaring global variables for CRAN
utils::globalVariables(c("idx", "i"))
