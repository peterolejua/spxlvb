#' @title Cross-validation and Final Model Fitting for spxlvb
#' @description This function performs k-fold cross-validation to determine the optimal
#'   `tau_alpha` parameter for the `spxlvb` model, and then fits a final `spxlvb` model
#'   to the full dataset using this optimal `tau_alpha`. Initial values for the final
#'   model are also derived from the full dataset.
#' @param k Integer, the number of folds to use for cross-validation. Must be greater than 2.
#' @param X A design matrix.
#' @param Y A response vector.
#' @param mu Initial variational mean (posterior expectation of beta_j | s_j = 1).
#'   If NULL, initialized automatically by `get.initials`.
#' @param omega Initial variational probability (posterior expectation of s_j).
#'   If NULL, initialized automatically by `get.initials`.
#' @param c_pi Prior parameter for pi (beta distribution shape1).
#'   If NULL, initialized automatically by `get.initials`.
#' @param d_pi Prior parameter for pi (beta distribution shape2).
#'   If NULL, initialized automatically by `get.initials`.
#' @param tau_e Initial precision of errors.
#'   If NULL, initialized automatically by `get.initials`.
#' @param update_order A numeric vector specifying the order of updates for coefficients.
#'   If NULL, initialized automatically by `get.initials`.
#' @param mu_alpha Prior mean for alpha. Default is mu_alpha_j = 1.
#' @param tau_alpha A numeric vector of `tau_alpha` values to cross-validate over.
#'   Must have at least two values.
#' @param tau_b Initial precision for beta_j (when s_j = 1). Default is a tau_b_j = 400.
#' @param intercept Logical, indicating whether an intercept term should be included in the model.
#' @param max_iter Maximum number of outer loop iterations for both CV fits and the final fit.
#' @param tol Convergence tolerance for both CV fits and the final fit.
#' @param seed Seed for reproducibility of data splitting and `glmnet` initials.
#' @param verbose Logical, if TRUE, prints progress messages during cross-validation.
#' @return The final fitted `spxlvb` model, which is a list containing the approximate
#'   posterior parameters and convergence information for the full dataset using the
#'   optimal `tau_alpha` determined by cross-validation.
#' @details This function orchestrates the cross-validation process and the final model fit.
#'   It first gets initial values for the full dataset, then uses `cv.spxlvb` to find
#'   the `tau_alpha` that minimizes cross-validation error, and finally calls `spxlvb`
#'   on the complete dataset with the chosen `tau_alpha`.
#' @seealso \code{\link{cv.spxlvb}}, \code{\link{spxlvb}}
#' @importFrom spxlvb spxlvb cv.spxlvb get.initials
#' @export
cv.spxlvb.fit <- function(
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

  # get initials for the *full* dataset
  initials <- spxlvb::get.initials(
    X = X, # design matrix
    Y = Y, # response vector
    mu = mu, # Variational Normal mean estimated beta coefficient from lasso, posterior expectation of bj|sj = 1
    omega = omega, # Variational probability, expectation that the coefficient from lasso is not zero, the posterior expectation of sj
    c_pi = c_pi, # π ∼ Beta(aπ, bπ), known/estimated
    d_pi = d_pi, # π ∼ Beta(aπ, bπ), known/estimated
    tau_e = tau_e, # errors iid N(0, tau_e^{-1}), known/estimated
    update_order = update_order,
    intercept = intercept, # Pass intercept here so initials are consistent
    seed = seed # seed for cv
  )

  # Perform cross-validation to find optimal tau_alpha
  cv_results <- spxlvb::cv.spxlvb(
    k = k, #the number of folds to use
    X = X, # design matrix
    Y = Y, # response vector
    mu = initials$mu_0, # Use initials from full data for CV, but note that CV itself will re-initialize per fold
    omega = initials$omega_0, # This argument is passed to cv.spxlvb, which then passes it to spxlvb.
    # Note: cv.spxlvb re-initializes within each fold using get.initials
    # so these `mu` and `omega` from `initials` might not be directly used *within* the CV folds,
    # unless you explicitly changed `get.initials` to pass them through.
    # However, it's good to pass them here as part of the consistent parameter set for cv.spxlvb.
    c_pi = initials$c_pi,
    d_pi = initials$d_pi,
    tau_e = initials$tau_e,
    update_order = initials$update_order,
    mu_alpha = mu_alpha,
    tau_alpha = tau_alpha,
    tau_b = tau_b,
    intercept = intercept,
    max_iter = max_iter,
    tol = tol,
    seed = seed,
    verbose = verbose
  )

  # Fit the final model with the optimal tau_alpha on the full dataset
  # Using try() to gracefully handle potential errors during the final fit
  fit_spxlvb <- try(
    spxlvb::spxlvb(
      X = X, # design matrix
      Y = Y, # response vector
      mu = initials$mu_0, # Use the initial values derived from the full dataset
      omega = initials$omega_0,
      c_pi = initials$c_pi,
      d_pi = initials$d_pi,
      tau_e = initials$tau_e,
      update_order = initials$update_order,
      mu_alpha = mu_alpha,
      tau_alpha = cv_results$tau_alpha_opt, # Use the optimal tau_alpha from CV
      tau_b = tau_b,
      intercept = intercept,
      max_iter = max_iter,
      tol = tol,
      seed = seed
    ),
    silent = TRUE # Prevents printing error messages directly from try()
  )

  if (inherits(fit_spxlvb, "try-error")) {
    warning("Final spxlvb model fit failed with optimal tau_alpha. Returning CV results only.")
    return(cv_results)
  }

  if (verbose) {
    message(paste("Best tau_alpha selected by CV:", cv_results$tau_alpha_opt))
  }

  # Return the final fitted model
  return(fit_spxlvb)
}
