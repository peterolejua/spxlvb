#' @title Cross-validation for Sparse Paramaeter Expanded Variational Bayes (spxlvb)
#' @description Performs k-fold cross-validation for the spxlvb model,
#'   allowing for evaluation of model performance across different tau_alpha values.
#' @param k Integer, the number of folds for cross-validation. Must be greater than 2.
#' @param X A design matrix.
#' @param Y A response vector.
#' @param mu Initial variational mean (posterior expectation of beta_j | s_j = 1). If NULL, initialized automatically.
#' @param omega Initial variational probability (posterior expectation of s_j). If NULL, initialized automatically.
#' @param c_pi Prior parameter for pi (beta distribution shape1). If NULL, initialized automatically.
#' @param d_pi Prior parameter for pi (beta distribution shape2). If NULL, initialized automatically.
#' @param tau_e Initial precision of errors. If NULL, initialized automatically.
#' @param update_order A numeric vector specifying the order of updates for coefficients. If NULL, initialized automatically.
#' @param mu_alpha Prior mean for alpha. Default is mu_alpha_j = 1.
#' @param tau_alpha A numeric vector of tau_alpha values to cross-validate over. Must have at least two values.
#' @param tau_b Initial precision for beta_j (when s_j = 1). Default is a tau_b_j = 400.
#' @param intercept Logical, indicating whether an intercept term should be included in the model.
#' @param max_iter Maximum number of outer loop iterations for each spxlvb fit.
#' @param tol Convergence tolerance for each spxlvb fit.
#' @param seed Seed for reproducibility of data splitting and `glmnet` initials.
#' @param verbose Logical, if TRUE, prints progress messages during cross-validation.
#' @return A list containing cross-validation results:
#'   \item{ordered_tau_alpha}{The sorted vector of tau_alpha values used.}
#'   \item{epe_test_k}{A matrix of prediction errors (MSE) for each fold (rows) and each tau_alpha (columns).}
#'   \item{CVE}{Cross-Validation Error (mean MSE) for each tau_alpha.}
#'   \item{tau_alpha_opt}{The tau_alpha value that minimizes the CVE.}
#' @details This function performs k-fold cross-validation to find the optimal `tau_alpha`
#'   for the `spxlvb` model. It iterates through different `tau_alpha` values, trains
#'   the model on training folds, and evaluates performance on the held-out test fold.
#'   To leverage parallel processing, ensure a parallel backend (e.g., from `doParallel` or `doSNOW` packages)
#'   is registered using `registerDoParallel()` or similar before calling this function.
#' @importFrom caret createFolds
#' @importFrom foreach foreach %do% %dopar%
#' @importFrom spxlvb spxlvb
#' @importFrom stats sd
#' @importFrom stats setNames
#' @export
cv.spxlvb <- function(
    k = 5, #the number of folds to use
    X, # design matrix
    Y, # response vector
    mu = NULL, # Variational Normal mean estimated beta coefficient from lasso, posterior expectation of bj|sj = 1
    omega = NULL, # Variational probability, expectation that the coefficient from lasso is not zero, the posterior expectation of sj
    c_pi = NULL, # π ∼ Beta(aπ, bπ), known/estimated
    d_pi = NULL, # π ∼ Beta(aπ, bπ), known/estimated
    tau_e = NULL, # errors iid N(0, tau_e^{-1}), known/estimated
    update_order = NULL,
    mu_alpha = rep(1, ncol(X)), # alpha_j is N(mu_alpha_j, (tau_e*tau_alpha)^{-1})
    tau_alpha = c(0,10^(3:7)), # Can be a vector now
    tau_b = rep(400, ncol(X)), # initial. b_j is N(0, (tau_e*tau_b_j)^{-1}), known/estimated
    intercept = FALSE,
    max_iter = 100L, # Ensure it's an integer literal
    tol = 1e-5,
    seed = 12376, # seed for cv.glmnet initials
    verbose = TRUE
){

  set.seed(seed)

  ## verification before start -----------------

  ### check whether tau_alpha is a vector ----
  if (!is.null(tau_alpha) && length(tau_alpha) < 2){
    stop("Need more than one value of tau_alpha for cross validation.")
  }

  if( sum(tau_alpha < 0) > 0 ){
    stop("Error Message: Some tau_alpha is less than 0.")
  }


  ### check the number of folds is a whole number of at least two ----
  if (!is.numeric(k) || length(k) != 1 || k != as.integer(k) ){ # Changed check
    stop("The number of folds must be an integer.")
  }

  if(k < 3){
    stop("The number of folds must be greater than two.")
  }

  ### dimensions ----
  n.y <- length(Y)
  n.x <- nrow(X)
  p <- ncol(X)
  if(n.x == n.y){
    n <- n.x
  }else{
    stop("The dimensions for X and Y matrix do not match.")
  }

  if( tol < 0 ){
    stop("Error Message: tol is less than 0.")
  }

  if( max_iter < 0 || !is.integer(max_iter)){ # Changed check
    stop("Error Message: max_iter must be a positive whole number")
  }

  # order the tau_alpha
  ordered_tau_alpha <- sort(tau_alpha, decreasing = FALSE) # Changed to FALSE

  ## Folds creation for k-fold Cross Validation ----
  data <- data.frame(X = X, Y = Y) # Create data.frame from X and Y

  # Create indices for k-fold cross-validation
  folds <- caret::createFolds(data$Y, k = k, list = TRUE)

  # Use %dopar% for parallel execution if a backend is registered
  epe_test_k <- foreach(
    i = 1:k,
    .combine = 'rbind',
    .packages = c('foreach','glmnet', 'caret', 'spxlvb') # Explicitly list necessary packages
  ) %dopar% { # Changed from %do% to %dopar%
    train_indices <- unlist(folds[-i])
    test_indices <- unlist(folds[i])

    # Ensure data is matrix/vector for spxlvb and C++
    X_train = as.matrix(data[train_indices, -ncol(data)]) # design matrix
    Y_train = as.vector(data[train_indices, ncol(data)]) # response vector

    X_test = as.matrix(data[test_indices, -ncol(data)]) # design matrix
    Y_test = as.vector(data[test_indices, ncol(data)]) # response vector

    # Train the model on the training set
    fold_mses <- foreach(
      current_tau_alpha = ordered_tau_alpha, # Renamed to avoid conflict with outer parameter
      .combine = 'c' # Combine into a vector for this fold
    ) %do% { # Changed from %do% to %dopar%
      if (verbose) {
        message(paste("Fold:", i, "tau_alpha:", current_tau_alpha)) # Use message for package
      }

      fit_spxlvb <- spxlvb(
        X = X_train, # design matrix
        Y = Y_train, # response vector
        mu = mu, # Pass initial mu, omega if provided
        omega = omega,
        c_pi = c_pi,
        d_pi = d_pi,
        mu_alpha = mu_alpha,
        tau_alpha = current_tau_alpha, # Use the current tau_alpha from iteration
        tau_b = tau_b,
        intercept = intercept,
        tau_e = tau_e,
        update_order = update_order,
        max_iter = max_iter,
        tol = tol,
        seed = seed # Pass seed for reproducibility of internal initializations
      )

      # betas and prediction
      if (is.null(fit_spxlvb) || is.null(fit_spxlvb$mu) || is.null(fit_spxlvb$omega)) {
        # If fit failed or returned NULL/incomplete results, return NA for MSE
        if (verbose) {
          message(paste("Warning: spxlvb fit failed or returned incomplete results for Fold:", i, ", tau_alpha:", current_tau_alpha))
        }
        NA_real_
      } else {
        beta_hat_coefficients <- fit_spxlvb$mu # This contains the 'p' coefficients for X

        y_pred <- if (intercept) {
          # Explicitly add intercept if it was included in the model
          X_test %*% beta_hat_coefficients + fit_spxlvb$intercept
        } else {
          X_test %*% beta_hat_coefficients
        }
        mean((Y_test - y_pred)^2)
      }
    }
    names(fold_mses) <- as.character(ordered_tau_alpha) # Name columns by tau_alpha
    fold_mses
  } # End of outer foreach

  # Calculate CVE (Cross-Validation Error)
  CVE <- colMeans(epe_test_k, na.rm = TRUE)

  #--------------------- output ----------------------
  cv_results <- list(
    "ordered_tau_alpha" = ordered_tau_alpha,
    "epe_test_k" = epe_test_k,
    "CVE" = CVE,
    "tau_alpha_opt" = ordered_tau_alpha[ which.min(CVE)]
  )

  return(cv_results)
}
