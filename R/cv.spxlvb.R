#' @title Cross-validation for Sparse Parameter Exploded Variational Bayes (spxlvb)
#'
#' @description Performs k-fold cross-validation for the \code{spxlvb} model to
#'   optimize hyperparameters. By default, only
#'   \code{alpha_prior_precision} is searched (1D CV). When
#'   \code{b_prior_precision_grid} is supplied, a joint 2D search over both
#'   \code{alpha_prior_precision} and \code{b_prior_precision} is performed.
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
#' @param b_prior_precision_grid Optional numeric vector of scalar slab prior
#'   precisions to cross-validate over. When \code{NULL} (default), the scalar
#'   \code{b_prior_precision} is used for every fit (1D search over
#'   \code{alpha_prior_precision} only). When non-\code{NULL}, a 2D grid
#'   search is performed over all combinations of
#'   \code{alpha_prior_precision_grid} and \code{b_prior_precision_grid}.
#'   Each grid value is expanded to a constant vector of length \eqn{p}.
#' @param b_prior_precision Numeric vector of length \eqn{p}.
#'   Coordinate-specific slab prior precisions
#'   (see \code{\link{spxlvb}} for details). Used only when
#'   \code{b_prior_precision_grid} is \code{NULL}.
#'   Defaults to a vector of ones of length \eqn{p} (determined
#'   automatically from \code{X}).
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
#' \item{ordered_alpha_prior_precision_grid}{The sorted alpha hyperparameter grid.}
#' \item{ordered_b_prior_precision_grid}{The sorted b hyperparameter grid
#'   (only present when \code{b_prior_precision_grid} is non-\code{NULL}).}
#' \item{epe_test_k}{Prediction errors per fold. A matrix (folds x alpha grid)
#'   for 1D search, or a 3D array (folds x alpha grid x b grid) for 2D search.}
#' \item{CVE}{Mean CVE. A named vector for 1D search, or a matrix
#'   (alpha grid x b grid) for 2D search.}
#' \item{alpha_prior_precision_grid_opt}{Optimal alpha value.}
#' \item{b_prior_precision_grid_opt}{Optimal b value
#'   (only present when \code{b_prior_precision_grid} is non-\code{NULL}).}
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

#' @title Cross-validation and Final Model Fitting for spxlvb
#'
#' @description Performs k-fold cross-validation to determine the optimal
#'   hyperparameters and then fits a final \code{spxlvb} model to the full
#'   dataset. When \code{b_prior_precision_grid} is supplied, a joint 2D
#'   search over both \code{alpha_prior_precision} and
#'   \code{b_prior_precision} is performed.
#'
#' @inheritParams cv.spxlvb
#' @return A list containing the cross-validation results and the final
#'   fitted \code{spxlvb} model (\code{fit_spxlvb}). If the final fit
#'   fails, only the CV results are returned with a warning.
#'
#' @details This function orchestrates the cross-validation process and the final model fit.
#'   It first identifies initial values for the full dataset, uses \code{cv.spxlvb} to find
#'   the optimal hyperparameters, and finally fits the model to the complete dataset.
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
  set.seed(seed)
  p <- ncol(X)
  is_2d <- !is.null(b_prior_precision_grid)

  std <- standardize_data(X, Y, standardize)
  X_cs <- std$X_cs
  Y_c <- std$Y_c

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

  cv_results <- cv.spxlvb(
    k = k,
    X = X,
    Y = Y,
    mu_0 = initials$mu_0,
    omega_0 = initials$omega_0,
    c_pi_0 = initials$c_pi_0,
    d_pi_0 = initials$d_pi_0,
    tau_e = initials$tau_e,
    update_order = initials$update_order,
    mu_alpha = mu_alpha,
    alpha_prior_precision_grid = alpha_prior_precision_grid,
    b_prior_precision_grid = b_prior_precision_grid,
    b_prior_precision = b_prior_precision,
    standardize = standardize,
    intercept = intercept,
    max_iter = max_iter,
    tol = tol,
    seed = seed,
    verbose = verbose,
    parallel = parallel
  )

  if (verbose) {
    msg <- sprintf(
      "Fitting final model with optimal alpha_prior_precision: %g",
      cv_results$alpha_prior_precision_grid_opt
    )
    if (is_2d) {
      msg <- sprintf(
        "%s, b_prior_precision: %g",
        msg, cv_results$b_prior_precision_grid_opt
      )
    }
    message(msg)
  }

  b_prec_final <- if (is_2d) {
    rep(cv_results$b_prior_precision_grid_opt, p)
  } else {
    b_prior_precision
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
        b_prior_precision = b_prec_final,
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

  result <- list(
    ordered_alpha_prior_precision_grid = cv_results$ordered_alpha_prior_precision_grid,
    epe_test_k = cv_results$epe_test_k,
    CVE = cv_results$CVE,
    alpha_prior_precision_grid_opt = cv_results$alpha_prior_precision_grid_opt,
    fit_spxlvb = fit_spxlvb
  )

  if (is_2d) {
    result$ordered_b_prior_precision_grid <- cv_results$ordered_b_prior_precision_grid
    result$b_prior_precision_grid_opt <- cv_results$b_prior_precision_grid_opt
  }

  result
}

# Declaring global variables for CRAN
utils::globalVariables(c("idx", "i"))
