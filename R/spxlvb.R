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
#' @param initialization Character string specifying the initialization
#'   strategy. One of \code{"lasso"} (default), \code{"ridge"},
#'   \code{"lasso_ridge"}, or \code{"null"}. See
#'   \code{\link{get_initials_spxlvb}} for details. Pre-supplied values
#'   (\code{mu_0}, \code{omega_0}, etc.) always override the strategy.
#' @param mu_alpha Numeric vector of length \eqn{p+1}. Prior means for the
#'   explosion parameters \eqn{\alpha_1,\ldots,\alpha_{p+1}}.
#'   Elements 1 to \eqn{p} are the per-coordinate explosion prior means;
#'   element \eqn{p+1} is the prior mean for the global explosion
#'   parameter \eqn{\alpha_{p+1}} (applied after each full coordinate sweep).
#'   The \eqn{p+1} dimension comes from the global explosion parameter,
#'   not from an intercept term.
#'   Defaults to a vector of ones of length \eqn{p+1} (determined
#'   automatically from \code{X}), centering all explosion parameters
#'   at 1 (no rescaling a priori).
#' @param alpha_prior_precision Numeric scalar. Shared prior precision
#'   \eqn{\tau_\alpha} for all \eqn{p+1} explosion parameters.
#'   Each \eqn{\alpha_j \sim N(\mu_{\alpha,j},\;
#'   (\tau_\epsilon \tau_\alpha)^{-1})}. Larger values shrink the
#'   explosion parameters toward their prior means (closer to standard VB).
#'   Default is 1000.
#' @param b_prior_precision Numeric vector of length \eqn{p}.
#'   Coordinate-specific slab prior precisions
#'   \eqn{\tau_{b,1},\ldots,\tau_{b,p}}. Each slab component has
#'   \eqn{b_j \mid s_j=1 \sim N(0,\; (\tau_\epsilon \tau_{b,j})^{-1})}.
#'   These are the slab precisions for the regression coefficients,
#'   not for the explosion parameters.
#'   Defaults to a vector of ones of length \eqn{p} (determined
#'   automatically from \code{X}).
#' @param standardize Logical. Center Y, and center and scale X. Default is TRUE.
#' @param intercept Logical. Whether to include an intercept. Default is TRUE. After the model is fit on the centered and scaled data, the final coefficients are "unscaled" to put them back on the original scale of your data. The intercept is then calculated separately using the means and the final coefficients.
#' @param max_iter Maximum number of iterations for the variational update. Default is 1000.
#' @param tol Convergence threshold for entropy and alpha change. Default is 1e-5.
#' @param save_history Logical. If TRUE (default), per-iteration parameter histories are stored and returned. Set to FALSE to save memory in large-scale simulations.
#' @param convergence Character string specifying the convergence criterion.
#'   One of \code{"elbo_relative"} (default), \code{"elbo_absolute"},
#'   \code{"chisq"}, or \code{"entropy"}.
#'   \code{"elbo_relative"} stops when the relative change in the exploded
#'   ELBO, \eqn{|\Delta \mathrm{ELBO}| / (|\mathrm{ELBO}| + 10^{-10})},
#'   falls below \code{tol}.
#'   \code{"elbo_absolute"} stops when the absolute change in the exploded
#'   ELBO, \eqn{|\Delta \mathrm{ELBO}|}, falls below \code{tol}.
#'   \code{"chisq"} uses a chi-squared test on normalised changes in the
#'   linear predictor.
#'   \code{"entropy"} stops when the maximum absolute change in
#'   per-coordinate Bernoulli entropy of the inclusion probabilities
#'   falls below \code{tol}, following the criterion used by
#'   Ray and Szabo (2022).
#'   See Appendix for a comparison of the four criteria.
#' @param update_pi Logical. If \code{TRUE}, treat \eqn{\pi} as a variational
#'   parameter with \eqn{q(\pi) = \text{Beta}(\tilde{c}_\pi, \tilde{d}_\pi)},
#'   updated each iteration via the conjugate Beta--Bernoulli update.
#'   If \code{FALSE} (default), \eqn{\pi} is fixed at
#'   \eqn{c_\pi / (c_\pi + d_\pi)}.
#' @param include_exploded_elbo_constants Logical. If \code{TRUE}, include all
#'   normalizing constants in the exploded ELBO (likelihood normalization,
#'   \eqn{2\pi} factors, Gaussian entropy constants). These terms are
#'   independent of variational parameters and tuned hyperparameters, so they
#'   cannot affect hyperparameter selection. However, they shift the absolute
#'   ELBO level, which can affect the relative-change convergence criterion.
#'   Default: \code{FALSE}.
#' @param disable_global_alpha Logical. If \code{TRUE}, skip the global
#'   \eqn{\alpha_{p+1}} rescaling step after each full coordinate sweep.
#'   Per-coordinate \eqn{\alpha_j} rescaling still occurs. This reduces
#'   the parameter explosion to coordinate-level only and is used for
#'   ablation studies (see the paper Appendix, Section C.2.1).
#'   Default: \code{FALSE}.
#' @param track_coordinate_exploded_elbo Logical. If \code{TRUE}, compute the
#'   exploded ELBO (data fit, slab normalisation, slab penalty, alpha
#'   normalisation, alpha penalty, pi logodds, slab entropy, spike entropy)
#'   and the \eqn{\alpha}-stripped ELBO (exploded ELBO minus alpha terms)
#'   after each individual coordinate update within the inner loop and check
#'   both for monotonicity violations.
#'   Returns \code{coordinate_exploded_elbo_violations} and
#'   \code{coordinate_exploded_elbo_worst_drop} for the exploded ELBO,
#'   plus \code{coordinate_alpha_stripped_elbo_violations} and
#'   \code{coordinate_alpha_stripped_elbo_worst_drop} for the
#'   \eqn{\alpha}-stripped ELBO.
#'   Intended for diagnostic use only as it adds
#'   \eqn{O(p^2 + np)} work per outer iteration.
#'   Default: \code{FALSE}.
#' @param track_all_criteria Logical. If \code{TRUE}, compute and store
#'   per-iteration values of all four convergence criteria (ELBO,
#'   chi-squared p-value, and maximum entropy change) regardless of which
#'   criterion is used for stopping.  This enables post-hoc extraction of
#'   the iteration at which any criterion would have triggered convergence
#'   at any tolerance, without rerunning the algorithm.  Requires
#'   materialising the \eqn{n \times p} element-wise squared design matrix
#'   and adds \eqn{O(np)} work per iteration for the chi-squared statistic.
#'   Returns \code{chisq_history} and \code{entropy_change_history} in
#'   addition to the always-returned \code{elbo_history} and
#'   \code{alpha_stripped_elbo_history}.
#'   Default: \code{FALSE}.
#' @param gamma_hyperprior_tau_alpha Logical. If \code{TRUE}, place a conjugate
#'   \eqn{\text{Gamma}(r_\alpha, d_\alpha)} hyperprior on
#'   \eqn{\tau_\alpha} and update it within the VB loop. The variational
#'   update is \eqn{q(\tau_\alpha) = \text{Gamma}(r_{\text{post}},
#'   d_{\text{post}})} with \eqn{r_{\text{post}} = r_\alpha + (p+1)/2}
#'   and \eqn{d_{\text{post}} = d_\alpha + (\tau_\epsilon / 2)
#'   \sum_k E_q[(\alpha_k - \mu_{\alpha,k})^2]}. The posterior mean
#'   \eqn{E[\tau_\alpha] = r_{\text{post}} / d_{\text{post}}} replaces
#'   the fixed \code{alpha_prior_precision / tau_e} at each iteration.
#'   This eliminates the need to grid-search over \eqn{\tau_\alpha},
#'   reducing tuning to a 1D search over \eqn{\tau_b} only.
#'   The hyperprior parameters \code{r_alpha} and \code{d_alpha} are not
#'   sensitive: with \eqn{(p+1)/2} effective observations from the alpha
#'   posteriors, even a vague prior is quickly overwhelmed by the data.
#'   Default: \code{FALSE}.
#' @param r_alpha Numeric scalar. Shape parameter for the Gamma hyperprior
#'   on \eqn{\tau_\alpha}. Only used when \code{gamma_hyperprior_tau_alpha = TRUE}.
#'   Default: \code{alpha_prior_precision / tau_e} (matches the initial
#'   \eqn{\tau_\alpha} from the fixed-precision parameterisation when
#'   \code{d_alpha = 1}).
#' @param d_alpha Numeric scalar. Rate parameter for the Gamma hyperprior
#'   on \eqn{\tau_\alpha}. Only used when \code{gamma_hyperprior_tau_alpha = TRUE}.
#'   Default: 1.
#' @param gamma_hyperprior_tau_b Logical. If \code{TRUE}, place a conjugate
#'   \eqn{\text{Gamma}(r_b, d_b)} hyperprior on the shared slab precision
#'   \eqn{\tau_b} and update it within the VB loop.  At each iteration,
#'   \eqn{E[\tau_b] = r_{b,\text{post}} / d_{b,\text{post}}} is computed
#'   and all per-coordinate \eqn{\tau_{b,j}} are reset to this common
#'   value.  Combined with \code{gamma_hyperprior_tau_alpha = TRUE}, this makes the
#'   algorithm fully tuning-free (no grid search needed).
#'   Default: \code{FALSE}.
#' @param r_b Numeric scalar. Shape parameter for the Gamma hyperprior
#'   on \eqn{\tau_b}. Only used when \code{gamma_hyperprior_tau_b = TRUE}.
#'   Default: \code{b_prior_precision[1] / tau_e} (matches initial
#'   \eqn{\tau_b}).
#' @param d_b Numeric scalar. Rate parameter for the Gamma hyperprior
#'   on \eqn{\tau_b}. Only used when \code{gamma_hyperprior_tau_b = TRUE}.
#'   Default: 1.
#' @param seed Integer seed for cross-validation in glmnet. Default is 12376.
#' @return A list with posterior summaries including estimated coefficients (`mu`),
#' inclusion probabilities (`omega`), intercept (if applicable), alpha path, convergence status, etc.
#' @details
#' \strong{Parameter explosion.}
#' The algorithm introduces \eqn{p+1} explosion parameters
#' \eqn{\alpha_1,\ldots,\alpha_p,\alpha_{p+1}}. The \eqn{+1} comes
#' from the global explosion parameter \eqn{\alpha_{p+1}}, not from an
#' intercept. At each coordinate update \eqn{j}, the optimal
#' \eqn{\alpha_j} rescales all other variational parameters to improve
#' calibration. After a full sweep through all \eqn{p} coordinates, a
#' global \eqn{\alpha_{p+1}} rescaling is applied. When all explosion
#' parameters equal 1,
#' the algorithm reduces to standard coordinate-ascent VB.
#'
#' The key user-facing parameters governing the explosion are
#' \code{mu_alpha} (length \eqn{p+1}) and \code{alpha_prior_precision}
#' (scalar, shared). The slab prior precisions \code{b_prior_precision}
#' (length \eqn{p}) are separate and control the spike-and-slab
#' component, not the explosion.
#'
#' \strong{Intercept handling.}
#' When \code{intercept = TRUE} (requires \code{standardize = TRUE}),
#' the model is fit on centered-and-scaled data (no intercept column is
#' added to \code{X}). After convergence, the coefficients are unscaled
#' to the original data scale, and the intercept is computed as
#' \eqn{\hat\beta_0 = \bar Y - \sum_{j=1}^{p} \hat\beta_j \bar X_j},
#' where \eqn{\bar Y} and \eqn{\bar X_j} are the original sample means.
#' The returned \code{beta} vector has length \eqn{p+1} (intercept
#' first), but this \eqn{+1} is unrelated to the explosion parameter
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
  initialization = c("lasso", "ridge", "lasso_ridge", "null"),
  mu_alpha = rep(1, ncol(X) + 1), # alpha_j is N(mu_alpha_j, (tau_e*tau_alpha)^{-1})
  alpha_prior_precision = 1000,
  b_prior_precision = rep(1, ncol(X)),
  standardize = TRUE,
  intercept = TRUE,
  max_iter = 1000,
  tol = 1e-3,
  save_history = TRUE,
  convergence = c("elbo_relative", "elbo_absolute", "chisq", "entropy"),
  update_pi = FALSE,
  include_exploded_elbo_constants = FALSE,
  disable_global_alpha = FALSE,
  track_coordinate_exploded_elbo = FALSE,
  track_all_criteria = FALSE,
  gamma_hyperprior_tau_alpha = FALSE,
  r_alpha = NULL,
  d_alpha = NULL,
  gamma_hyperprior_tau_b = FALSE,
  r_b = NULL,
  d_b = NULL,
  seed = 12376 # seed for cv.glmnet initials
) {
  initialization <- match.arg(initialization)
  convergence <- match.arg(convergence)
  convergence_method <- match(
    convergence,
    c("elbo_relative", "chisq", "entropy", "elbo_absolute")
  ) - 1L

  if (intercept && !standardize) {
    stop("intercept = TRUE requires standardize = TRUE")
  }

  p <- ncol(X)
  if (is.null(mu_alpha)) mu_alpha <- rep(1, p + 1)

  # Standardize data
  std <- standardize_data(X, Y, standardize)
  X_cs <- std$X_cs
  Y_c <- std$Y_c
  X_means <- std$X_means
  sigma_estimate <- std$sigma_estimate
  Y_mean <- std$Y_mean

  # get_initials_spxlvb is in R/ directory and is automatically available
  # if null they are calculated
  # if given the function is still called but skipped when not needed.
  initials <- get_initials_spxlvb(
    X = X_cs,
    Y = Y_c,
    mu_0 = mu_0,
    omega_0 = omega_0,
    c_pi_0 = c_pi_0,
    d_pi_0 = d_pi_0,
    tau_e = tau_e,
    update_order = update_order,
    initialization = initialization,
    seed = seed
  )

  mu_0 <- initials$mu_0
  omega_0 <- initials$omega_0
  c_pi_0 <- initials$c_pi_0
  d_pi_0 <- initials$d_pi_0
  tau_e <- initials$tau_e
  update_order <- initials$update_order

  if (gamma_hyperprior_tau_alpha) {
    if (is.null(r_alpha)) r_alpha <- alpha_prior_precision / tau_e
    if (is.null(d_alpha)) d_alpha <- 1
  }

  if (gamma_hyperprior_tau_b) {
    if (is.null(r_b)) r_b <- b_prior_precision[1] / tau_e
    if (is.null(d_b)) d_b <- 1
  }

  # match internal function call and generate list of arguments
  elbo_offset <- 0.0
  if (include_exploded_elbo_constants) {
    n <- nrow(X_cs)
    pi_fixed <- c_pi_0 / (c_pi_0 + d_pi_0)
    elbo_offset <- -n / 2 * log(2 * pi) +
      n / 2 * log(tau_e) +
      p / 2 +
      (p + 1) / 2
    if (!update_pi) {
      elbo_offset <- elbo_offset +
        p * log(1 - pi_fixed) +
        (c_pi_0 - 1) * log(pi_fixed) +
        (d_pi_0 - 1) * log(1 - pi_fixed) -
        lbeta(c_pi_0, d_pi_0)
    }
  }

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
    convergence_method,
    update_pi,
    elbo_offset,
    disable_global_alpha,
    track_coordinate_exploded_elbo,
    track_all_criteria,
    gamma_hyperprior_tau_alpha,
    if (gamma_hyperprior_tau_alpha) r_alpha else 0,
    if (gamma_hyperprior_tau_alpha) d_alpha else 0,
    gamma_hyperprior_tau_b,
    if (gamma_hyperprior_tau_b) r_b else 0,
    if (gamma_hyperprior_tau_b) d_b else 0
  )
  fn <- "run_vb_updates_cpp"

  approximate_posterior <- do.call(fn, arg)

  if (!approximate_posterior$converged) {
    warning(
      sprintf(
        "spxlvb did not converge within max_iter = %d iterations. Consider increasing max_iter or relaxing tol (current: %g).",
        max_iter, tol
      ),
      call. = FALSE
    )
  }

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
    exploded_elbo = as.numeric(approximate_posterior$elbo_history)[length(
      as.numeric(approximate_posterior$elbo_history)
    )],
    alpha_stripped_elbo = as.numeric(
      approximate_posterior$alpha_stripped_elbo_history
    )[length(as.numeric(approximate_posterior$alpha_stripped_elbo_history))],
    tau_alpha = if (gamma_hyperprior_tau_alpha) {
      approximate_posterior$tau_alpha
    } else {
      alpha_prior_precision / tau_e
    },
    tau_alpha_history = if (gamma_hyperprior_tau_alpha) {
      as.numeric(approximate_posterior$tau_alpha_history)
    },
    tau_b_common = if (gamma_hyperprior_tau_b) {
      approximate_posterior$tau_b[1]
    },
    tau_b_common_history = if (gamma_hyperprior_tau_b) {
      as.numeric(approximate_posterior$tau_b_common_history)
    },
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
    c_pi_tilde = approximate_posterior$c_pi_tilde,
    d_pi_tilde = approximate_posterior$d_pi_tilde,
    coordinate_exploded_elbo_violations =
      approximate_posterior$coordinate_exploded_elbo_violations,
    coordinate_exploded_elbo_worst_drop =
      approximate_posterior$coordinate_exploded_elbo_worst_drop,
    coordinate_alpha_stripped_elbo_violations =
      approximate_posterior$coordinate_alpha_stripped_elbo_violations,
    coordinate_alpha_stripped_elbo_worst_drop =
      approximate_posterior$coordinate_alpha_stripped_elbo_worst_drop,
    chisq_history = if (track_all_criteria) {
      as.numeric(approximate_posterior$chisq_history)
    },
    entropy_change_history = if (track_all_criteria) {
      as.numeric(approximate_posterior$entropy_change_history)
    },
    approximate_posterior = approximate_posterior
  )
  return(wrapper_results)
}
