#' Generate Initial Values for Variational Inference in Sparse Regression
#'
#' Estimates initial values for variational parameters (regression
#' coefficients \code{mu_0}, inclusion probabilities \code{omega_0},
#' error precision \code{tau_e}, and Beta prior parameters \code{c_pi_0},
#' \code{d_pi_0}) using one of four initialization strategies.
#' Pre-supplied values always override the strategy.
#'
#' @param X A design matrix (assumed already standardized when called
#'   from \code{\link{spxlvb}}).
#' @param Y A response vector (assumed already centered).
#' @param mu_0 Optional numeric vector. Initial variational means.
#' @param omega_0 Optional numeric vector. Initial inclusion probabilities.
#' @param c_pi_0 Optional numeric. Prior Beta shape1 for \eqn{\pi}.
#' @param d_pi_0 Optional numeric. Prior Beta shape2 for \eqn{\pi}.
#' @param tau_e Optional numeric. Error precision.
#' @param update_order Optional integer vector or \code{"random"}.
#'   Coordinate update order (0-indexed for C++).
#' @param initialization Character string specifying the initialization
#'   strategy. One of:
#'   \describe{
#'     \item{\code{"lasso"} (default)}{LASSO via \code{cv.glmnet(alpha=1)}.
#'       Coefficients are de-shrunk by OLS rescaling. Inclusion probabilities
#'       are set high for LASSO-selected variables. Update order is ascending
#'       \eqn{|\mu_{0,j}|}.}
#'     \item{\code{"ridge"}}{Ridge via \code{cv.glmnet(alpha=0)}.
#'       The top 10\% of coefficients by absolute value get high inclusion
#'       probability. Update order is ascending \eqn{|\mu_{0,j}|}.}
#'     \item{\code{"lasso_ridge"}}{LASSO for variable selection (omega)
#'       and noise estimation, ridge for coefficient magnitudes (mu).
#'       Combines the sparsity structure from LASSO with the less-shrunk
#'       coefficients from ridge, following the spirit of
#'       Ray and Szabo (2022). Update order is ascending \eqn{|\mu_{0,j}|}.}
#'     \item{\code{"null"}}{Prior-based initialization with no data-driven
#'       regression. Sets \eqn{\mu_0 = 0}, \eqn{\omega_0 = \hat\pi}
#'       (prior mean), and \eqn{\tau_e = 1/\mathrm{Var}(Y)}. Update order
#'       is random. Does not call \code{cv.glmnet}, making it the fastest
#'       strategy.}
#'   }
#' @param seed Seed for reproducibility (passed to \code{cv.glmnet} and
#'   random update order generation).
#' @return A named list with elements \code{mu_0}, \code{omega_0},
#'   \code{c_pi_0}, \code{d_pi_0}, \code{tau_e}, and \code{update_order}.
#' @examples
#' \donttest{
#' set.seed(1)
#' X <- matrix(rnorm(30 * 20), 30, 20)
#' Y <- rnorm(30)
#' init <- get_initials_spxlvb(X = X, Y = Y)
#' init_null <- get_initials_spxlvb(X = X, Y = Y, initialization = "null")
#' }
#' @importFrom glmnet cv.glmnet
#' @importFrom stats predict coef var
#' @export
get_initials_spxlvb <- function(
  X,
  Y,
  mu_0 = NULL,
  omega_0 = NULL,
  c_pi_0 = NULL,
  d_pi_0 = NULL,
  tau_e = NULL,
  update_order = NULL,
  initialization = c("lasso", "ridge", "lasso_ridge", "null"),
  seed = 12376
) {
  initialization <- match.arg(initialization)
  set.seed(seed)

  p <- ncol(X)

  all_supplied <- !is.null(mu_0) && !is.null(omega_0) && !is.null(c_pi_0) &&
    !is.null(d_pi_0) && !is.null(tau_e) && !is.null(update_order)

  if (!all_supplied) {
    strategy_result <- switch(initialization,
      lasso = init_lasso_strategy(X, Y, p),
      ridge = init_ridge_strategy(X, Y, p),
      lasso_ridge = init_lasso_ridge_strategy(X, Y, p),
      null = init_null_strategy(X, Y, p)
    )

    if (is.null(mu_0)) mu_0 <- strategy_result$mu_0
    if (is.null(omega_0)) omega_0 <- strategy_result$omega_0
    if (is.null(c_pi_0)) c_pi_0 <- strategy_result$c_pi_0
    if (is.null(d_pi_0)) d_pi_0 <- strategy_result$d_pi_0
    if (is.null(tau_e)) tau_e <- strategy_result$tau_e
  }

  if (!is.null(update_order) && length(update_order) < 2) {
    if (update_order == "random") {
      update_order <- sample(seq_len(p)) - 1L
    }
  }
  if (is.null(update_order)) {
    if (initialization == "null" && all(mu_0 == 0)) {
      update_order <- sample(seq_len(p)) - 1L
    } else {
      update_order <- order(abs(mu_0), decreasing = FALSE) - 1L
    }
  }

  list(
    mu_0 = mu_0,
    omega_0 = omega_0,
    c_pi_0 = c_pi_0,
    d_pi_0 = d_pi_0,
    tau_e = tau_e,
    update_order = update_order
  )
}


# =====================================================================
# Internal strategy helpers (unexported)
# =====================================================================

init_lasso_strategy <- function(X, Y, p) {
  lasso_cv <- glmnet::cv.glmnet(
    X, Y,
    alpha = 1, family = "gaussian",
    standardize = FALSE, standardize.response = FALSE,
    intercept = FALSE, parallel = TRUE
  )

  nz_idx <- predict(lasso_cv, s = "lambda.min", type = "nonzero")$lambda.min
  nz_min <- min(length(nz_idx), length(Y) - 2)

  yhat <- predict(lasso_cv, X, s = "lambda.min", type = "response")
  noise_sd <- sqrt(sum((Y - yhat)^2) / (length(Y) - nz_min - 1))
  tau_e <- 1 / noise_sd^2

  s_hat <- max(nz_min, 1)

  c_pi_0 <- s_hat * exp(0.5)
  d_pi_0 <- p - c_pi_0

  omega_0 <- rep(s_hat / p, p)
  if (length(nz_idx) > 0) {
    omega_0[nz_idx] <- 1 - s_hat / p
  }

  mu_0_raw <- as.numeric(coef(lasso_cv, s = "lambda.min"))[-1]
  appx_mn_0 <- X %*% mu_0_raw
  denom <- sum(appx_mn_0^2)

  if (denom > 0) {
    scaling_factor <- sum(appx_mn_0 * Y) / denom
    mu_0 <- mu_0_raw * scaling_factor
  } else {
    mu_0 <- mu_0_raw
  }

  list(mu_0 = mu_0, omega_0 = omega_0, tau_e = tau_e,
       c_pi_0 = c_pi_0, d_pi_0 = d_pi_0)
}


init_ridge_strategy <- function(X, Y, p) {
  ridge_cv <- glmnet::cv.glmnet(
    X, Y,
    alpha = 0, family = "gaussian",
    standardize = FALSE, intercept = FALSE, parallel = TRUE
  )

  mu_0 <- as.numeric(coef(ridge_cv, s = "lambda.min"))[-1]

  yhat <- predict(ridge_cv, X, s = "lambda.min", type = "response")
  tau_e <- 1 / mean((Y - yhat)^2)

  abs_mu <- abs(mu_0)
  threshold <- stats::quantile(abs_mu, 0.9)
  omega_0 <- ifelse(abs_mu > threshold, 0.8, 0.1)
  s_hat <- max(sum(abs_mu > threshold), 1)

  c_pi_0 <- s_hat * exp(0.5)
  d_pi_0 <- p - c_pi_0

  list(mu_0 = mu_0, omega_0 = omega_0, tau_e = tau_e,
       c_pi_0 = c_pi_0, d_pi_0 = d_pi_0)
}


init_lasso_ridge_strategy <- function(X, Y, p) {
  lasso_cv <- glmnet::cv.glmnet(
    X, Y,
    alpha = 1, family = "gaussian",
    standardize = FALSE, intercept = FALSE, parallel = TRUE
  )

  nz_idx <- predict(lasso_cv, s = "lambda.min", type = "nonzero")$lambda.min
  nz_min <- min(length(nz_idx), length(Y) - 2)
  s_hat <- max(nz_min, 1)

  omega_0 <- rep(s_hat / p, p)
  if (length(nz_idx) > 0) omega_0[nz_idx] <- 1 - s_hat / p

  yhat <- predict(lasso_cv, X, s = "lambda.min", type = "response")
  tau_e <- 1 / mean((Y - yhat)^2)

  c_pi_0 <- s_hat * exp(0.5)
  d_pi_0 <- p - c_pi_0

  ridge_cv <- glmnet::cv.glmnet(
    X, Y,
    alpha = 0, family = "gaussian",
    standardize = FALSE, intercept = FALSE, parallel = TRUE
  )
  mu_0 <- as.numeric(coef(ridge_cv, s = "lambda.min"))[-1]

  list(mu_0 = mu_0, omega_0 = omega_0, tau_e = tau_e,
       c_pi_0 = c_pi_0, d_pi_0 = d_pi_0)
}


init_null_strategy <- function(X, Y, p) {
  mu_0 <- rep(0, p)
  tau_e <- 1 / var(as.numeric(Y))

  s_hat <- max(ceiling(sqrt(p)), 1)
  c_pi_0 <- s_hat * exp(0.5)
  d_pi_0 <- p - c_pi_0
  pi_hat <- c_pi_0 / (c_pi_0 + d_pi_0)
  omega_0 <- rep(pi_hat, p)

  list(mu_0 = mu_0, omega_0 = omega_0, tau_e = tau_e,
       c_pi_0 = c_pi_0, d_pi_0 = d_pi_0)
}
