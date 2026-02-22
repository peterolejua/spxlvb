#' @title Parameter Exploded Variational Bayes for Well-Calibrated High-Dimensional Linear Regression with Spike-and-Slab Priors
#' @description Fits a sparse linear regression model using variational inference with parameter explosion.
#' The model uses spike-and-slab priors.
#' @param X A numeric matrix. The design matrix (n observations × p predictors).
#' @param Y A numeric vector. The response vector of length n.
#' @param mu_0 Optional numeric vector. Initial variational means for regression coefficients.
#' @param omega_0 Optional numeric vector. Initial spike probabilities.
#' @param c_pi_0 Optional numeric. Prior Beta(a, b) parameter a for the spike probability.
#' @param d_pi_0 Optional numeric. Prior Beta(a, b) parameter b for the spike probability.
#' @param tau_e Optional numeric. Known or estimated error precision.
#' @param update_order Optional integer vector. The coordinate update order (0-indexed for C++).
#' @param mu_alpha Numeric vector of length \eqn{p+1}. Prior means for the
#'   expansion parameters \eqn{\alpha_1,\ldots,\alpha_{p+1}}.
#'   Elements 1 to \eqn{p} are the per-coordinate expansion prior means;
#'   element \eqn{p+1} is the prior mean for the global expansion
#'   parameter \eqn{\alpha_{p+1}} (applied after each full coordinate sweep).
#'   The \eqn{p+1} dimension comes from the global expansion parameter,
#'   not from an intercept term.
#'   Defaults to a vector of ones of length \eqn{p+1} (determined
#'   automatically from \code{X}), centering all expansion parameters
#'   at 1 (no rescaling a priori).
#' @param alpha_prior_precision Numeric scalar. Shared prior precision
#'   \eqn{\tau_\alpha} for all \eqn{p+1} expansion parameters.
#'   Each \eqn{\alpha_j \sim N(\mu_{\alpha,j},\;
#'   (\tau_\epsilon \tau_\alpha)^{-1})}. Larger values shrink the
#'   expansion parameters toward their prior means (closer to standard VB).
#'   Default is 1000.
#' @param b_prior_precision Numeric vector of length \eqn{p}.
#'   Coordinate-specific slab prior precisions
#'   \eqn{\tau_{b,1},\ldots,\tau_{b,p}}. Each slab component has
#'   \eqn{b_j \mid s_j=1 \sim N(0,\; (\tau_\epsilon \tau_{b,j})^{-1})}.
#'   These are the slab precisions for the regression coefficients,
#'   not for the expansion parameters.
#'   Defaults to a vector of ones of length \eqn{p} (determined
#'   automatically from \code{X}).
#' @param standardize Logical. Center Y, and center and scale X. Default is TRUE.
#' @param intercept Logical. Whether to include an intercept. Default is TRUE. After the model is fit on the centered and scaled data, the final coefficients are "unscaled" to put them back on the original scale of your data. The intercept is then calculated separately using the means and the final coefficients.
#' @param max_iter Maximum number of iterations for the variational update. Default is 1000.
#' @param tol Convergence threshold for entropy and alpha change. Default is 1e-5.
#' @param save_history Logical. If TRUE (default), per-iteration parameter histories are stored and returned. Set to FALSE to save memory in large-scale simulations.
#' @param convergence Character string specifying the convergence criterion.
#'   One of \code{"elbo"} (default), \code{"chisq"}, or \code{"entropy"}.
#'   \code{"elbo"} stops when the relative change in the Evidence Lower
#'   Bound falls below \code{tol}.
#'   \code{"chisq"} uses a chi-squared test on normalised changes in the
#'   linear predictor.
#'   \code{"entropy"} stops when the maximum absolute change in
#'   per-coordinate Bernoulli entropy of the inclusion probabilities
#'   falls below \code{tol}, following the criterion used by
#'   Ray and Szabo (2022).
#'   See Appendix for a comparison of the three criteria.
#' @param seed Integer seed for cross-validation in glmnet. Default is 12376.
#' @return A list with posterior summaries including estimated coefficients (`mu`),
#' inclusion probabilities (`omega`), intercept (if applicable), alpha path, convergence status, etc.
#' @details
#' \strong{Parameter explosion.}
#' The algorithm introduces \eqn{p+1} expansion parameters
#' \eqn{\alpha_1,\ldots,\alpha_p,\alpha_{p+1}}. The \eqn{+1} comes
#' from the global expansion parameter \eqn{\alpha_{p+1}}, not from an
#' intercept. At each coordinate update \eqn{j}, the optimal
#' \eqn{\alpha_j} rescales all other variational parameters to improve
#' calibration. After a full sweep through all \eqn{p} coordinates, a
#' global \eqn{\alpha_{p+1}} rescaling is applied. When all expansion
#' parameters equal 1,
#' the algorithm reduces to standard coordinate-ascent VB.
#'
#' The key user-facing parameters governing the explosion are
#' \code{mu_alpha} (length \eqn{p+1}) and \code{alpha_prior_precision}
#' (scalar, shared). The slab prior precisions \code{b_prior_precision}
#' (length \eqn{p}) are separate and control the spike-and-slab
#' component, not the expansion.
#'
#' \strong{Intercept handling.}
#' When \code{intercept = TRUE} (requires \code{standardize = TRUE}),
#' the model is fit on centered-and-scaled data (no intercept column is
#' added to \code{X}). After convergence, the coefficients are unscaled
#' to the original data scale, and the intercept is computed as
#' \eqn{\hat\beta_0 = \bar Y - \sum_{j=1}^{p} \hat\beta_j \bar X_j},
#' where \eqn{\bar Y} and \eqn{\bar X_j} are the original sample means.
#' The returned \code{beta} vector has length \eqn{p+1} (intercept
#' first), but this \eqn{+1} is unrelated to the expansion parameter
#' dimension.
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 50; p <- 20
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- X[, 1:3] %*% c(1, -1, 0.5) + rnorm(n)
#' fit <- spxlvb(X = X, Y = Y, max_iter = 50)
#' }
#' @useDynLib spxlvb, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom glmnet cv.glmnet
#' @importFrom stats predict coef
#' @export
spxlvb <- function(
  X, # design matrix
  Y, # response vector
  mu_0 = NULL, # Variational Normal mean estimated beta coefficient from lasso, posterior expectation of bj|sj = 1
  omega_0 = NULL, # Variational probability, expectation that the coefficient from lasso is not zero, the posterior expectation of sj
  c_pi_0 = NULL, # π ∼ Beta(aπ, bπ), known/estimated
  d_pi_0 = NULL, # π ∼ Beta(aπ, bπ), known/estimated
  tau_e = NULL, # errors iid N(0, tau_e^{-1}), known/estimated
  update_order = NULL,
  mu_alpha = rep(1, ncol(X) + 1), # alpha_j is N(mu_alpha_j, (tau_e*tau_alpha)^{-1})
  alpha_prior_precision = 1000,
  b_prior_precision = rep(1, ncol(X)),
  standardize = TRUE,
  intercept = TRUE,
  max_iter = 1000,
  tol = 1e-3,
  save_history = TRUE,
  convergence = c("elbo", "chisq", "entropy"),
  seed = 12376 # seed for cv.glmnet initials
) {
  convergence <- match.arg(convergence)
  convergence_method <- match(convergence, c("elbo", "chisq", "entropy")) - 1L

  if (intercept && !standardize) {
    stop("intercept = TRUE requires standardize = TRUE")
  }

  p <- ncol(X)

  # Standardize data
  std <- standardize_data(X, Y, standardize)
  X_cs <- std$X_cs
  Y_c <- std$Y_c
  X_means <- std$X_means
  sigma_estimate <- std$sigma_estimate
  Y_mean <- std$Y_mean

  # get.initials.spxlvb is in R/ directory and is automatically available
  # if null they are calculated
  # if given the function is still called but skipped when not needed.
  initials <- get.initials.spxlvb(
    X = X_cs, # design matrix
    Y = Y_c, # response vector
    mu_0 = mu_0, # Variational Normal mean estimated beta coefficient from lasso, posterior expectation of bj|sj = 1
    omega_0 = omega_0, # Variational probability, expectation that the coefficient from lasso is not zero, the posterior expectation of sj
    c_pi_0 = c_pi_0, # π ∼ Beta(aπ, bπ), known/estimated
    d_pi_0 = d_pi_0, # π ∼ Beta(aπ, bπ), known/estimated
    tau_e = tau_e, # errors iid N(0, tau_e^{-1}), known/estimated
    update_order = update_order,
    seed = seed # seed for cv
  )

  mu_0 <- initials$mu_0
  omega_0 <- initials$omega_0
  c_pi_0 <- initials$c_pi_0
  d_pi_0 <- initials$d_pi_0
  tau_e <- initials$tau_e
  update_order <- initials$update_order

  # match internal function call and generate list of arguments
  arg <- list(
    X_cs,
    Y_c,
    mu_0,
    omega_0,
    c_pi_0,
    d_pi_0,
    tau_e,
    update_order,
    mu_alpha,
    alpha_prior_precision / tau_e, # = tau_alpha (scalar)
    b_prior_precision / tau_e, # = tau_b (vector)
    max_iter,
    tol,
    save_history,
    convergence_method
  )
  fn <- "run_vb_updates_cpp"

  approximate_posterior <- do.call(fn, arg)

  # Unscale solution
  if (standardize) {
    beta <- approximate_posterior$mu /
      sigma_estimate *
      approximate_posterior$omega
  } else {
    beta <- approximate_posterior$mu * approximate_posterior$omega
  }

  # add intercept
  if (intercept) {
    beta <- c(
      beta0 = Y_mean - sum(beta * X_means),
      beta
    )
  }

  wrapper_results <- list(
    converged = as.logical(approximate_posterior$converged),
    iterations = as.numeric(approximate_posterior$iterations),
    convergence_criterion = as.numeric(
      approximate_posterior$convergence_criterion
    ),
    elbo = as.numeric(approximate_posterior$elbo_history)[length(as.numeric(
      approximate_posterior$elbo_history
    ))],
    tau_alpha = alpha_prior_precision / tau_e,
    tau_b_0 = b_prior_precision / tau_e,
    tau_b = approximate_posterior$tau_b,
    tau_e = tau_e,
    mu_0 = mu_0,
    mu = if (standardize) {
      as.numeric(approximate_posterior$mu[1:p]) / sigma_estimate
    } else {
      as.numeric(approximate_posterior$mu[1:p])
    }, # unscale mu
    omega_0 = omega_0,
    omega = as.numeric(approximate_posterior$omega[1:p]),
    beta = beta,
    update_order = update_order,
    approximate_posterior = approximate_posterior
  )
  return(wrapper_results)
}
