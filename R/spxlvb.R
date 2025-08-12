#' @title Parameter Exploded Variational Bayes for Well-Calibrated High-Dimensional Linear Regression with Spike-and-Slab Priors
#' @description Fits a sparse linear regression model using variational inference with parameter explosion.
#' The model uses spike-and-slab priors.
#' @param X A numeric matrix. The design matrix (n observations × p predictors).
#' @param Y A numeric vector. The response vector of length n.
#' @param mu Optional numeric vector. Initial variational means for regression coefficients.
#' @param omega Optional numeric vector. Initial spike probabilities.
#' @param c_pi Optional numeric. Prior Beta(a, b) parameter a for the spike probability.
#' @param d_pi Optional numeric. Prior Beta(a, b) parameter b for the spike probability.
#' @param tau_e Optional numeric. Known or estimated error precision.
#' @param update_order Optional integer vector. The coordinate update order (0-indexed for C++).
#' @param mu_alpha Prior mean for alpha. Default is mu_alpha_j = 1.
#' @param tau_alpha Prior precision for alpha. Default is 1000.
#' @param tau_b Slab prior precision. Default is a tau_b_j = 400.
#' @param intercept Logical. Whether to include an intercept. Default is FALSE.
#' @param max_iter Maximum number of iterations for the variational update. Default is 1000.
#' @param tol Convergence threshold for entropy and alpha change. Default is 1e-5.
#' @param seed Integer seed for cross-validation in glmnet. Default is 12376.
#'
#' @return A list with posterior summaries including estimated coefficients (`mu`),
#' inclusion probabilities (`omega`), intercept (if applicable), alpha path, convergence status, etc.
#' @details This function acts as a wrapper for various C++ implementations of the SPXLVB algorithm.
#'   It handles initial parameter setup and dynamically dispatches to the chosen C++ version.
#' @examples
#' \dontrun{
#' # Example usage (assuming X and Y are defined)
#' # result <- spxlvb(X = my_X, Y = my_Y)
#' }
#' @useDynLib spxlvb, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom glmnet cv.glmnet
#' @importFrom stats predict coef
#' @export
spxlvb <- function(
    X, # design matrix
    Y, # response vector
    mu = NULL, # Variational Normal mean estimated beta coefficient from lasso, posterior expectation of bj|sj = 1
    omega = NULL, # Variational probability, expectation that the coefficient from lasso is not zero, the posterior expectation of sj
    c_pi = NULL, # π ∼ Beta(aπ, bπ), known/estimated
    d_pi = NULL, # π ∼ Beta(aπ, bπ), known/estimated
    tau_e = NULL, # errors iid N(0, tau_e^{-1}), known/estimated
    update_order = NULL,
    mu_alpha = rep(1, ncol(X)), # alpha_j is N(mu_alpha_j, (tau_e*tau_alpha)^{-1})
    tau_alpha = 1000,
    tau_b = rep(400, ncol(X)), # initial. b_j is N(0, (tau_e*tau_b_j)^{-1}), known/estimated
    intercept = FALSE,
    max_iter = 1000,
    tol = 1e-5,
    seed = 12376 # seed for cv.glmnet initials
) {

  #extract problem dimensions
  n = nrow(X)
  p = ncol(X)

  #rescale data if necessary
  X_sc = X
  Y_sc = Y


  # get.initials.spxlvb is in R/ directory and is automatically available
  initials <- get.initials(
    X = X_sc, # design matrix
    Y = Y_sc, # response vector
    mu = mu, # Variational Normal mean estimated beta coefficient from lasso, posterior expectation of bj|sj = 1
    omega = omega, # Variational probability, expectation that the coefficient from lasso is not zero, the posterior expectation of sj
    c_pi = c_pi, # π ∼ Beta(aπ, bπ), known/estimated
    d_pi = d_pi, # π ∼ Beta(aπ, bπ), known/estimated
    intercept = intercept,
    tau_e = tau_e, # errors iid N(0, tau_e^{-1}), known/estimated
    update_order = update_order,
    seed = seed # seed for cv
  )

  mu_0 = initials$mu_0
  omega_0 = initials$omega_0
  c_pi = initials$c_pi
  d_pi = initials$d_pi
  tau_e = initials$tau_e
  update_order = initials$update_order


  # #add intercept
  if(intercept){
    X_sc = cbind(X_sc, rep(1, n))
  }


  #match internal function call and generate list of arguments
  arg = list(
    X_sc,
    Y_sc,
    mu_0,
    omega_0,
    c_pi,
    d_pi,
    tau_e,
    update_order,
    mu_alpha,
    tau_alpha,
    tau_b,
    max_iter,
    tol
  )


  fn <- "fit_linear_exploded"

  approximate_posterior = do.call(fn, arg)


  test <- list(
    converged = as.logical(approximate_posterior$converged),
    tau_alpha = tau_alpha,
    tau_b_0 = tau_b,
    tau_b = approximate_posterior$tau_b,
    tau_e = tau_e,
    mu_0 = mu_0,
    mu = as.numeric(approximate_posterior$mu[1:p]),
    omega_0 = omega_0,
    omega = as.numeric(approximate_posterior$omega[1:p]),
    c_pi_0 = c_pi,
    c_pi_p = as.numeric(approximate_posterior$c_pi_p),
    d_pi_0 = d_pi,
    d_pi_p = as.numeric(approximate_posterior$d_pi_p),
    intercept = if(intercept) as.numeric(approximate_posterior$mu[p+1]) else 0,
    approximate_posterior = lapply(approximate_posterior, as.numeric),
    alpha_vec = as.numeric(approximate_posterior$alpha_vec),
    alpha = as.numeric(approximate_posterior$alpha_vec[approximate_posterior$iterations]),
    iterations = as.numeric(approximate_posterior$iterations),
    convergence_criterion = as.numeric(approximate_posterior$convergence_criterion),
    update_order = update_order
  )
  return(test)
}
