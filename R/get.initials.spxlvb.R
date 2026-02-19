#' Generate Initial Values for Variational Inference in Sparse Regression
#'
#' This helper function estimates initial values for variational parameters such as
#' regression coefficients (`mu`), spike probabilities (`omega`), and hyperparameters
#' like `tau_e`, `c_pi`, and `d_pi` using LASSO and Ridge regression fits.
#'
#' @title Get initial values for spxlvb
#' @description This function initializes parameters for the spxlvb model.
#' @param X A design matrix.
#' @param Y A response vector.
#' @param mu_0 Initial mean.
#' @param omega_0 Initial omega.
#' @param c_pi_0 Initial c_pi.
#' @param d_pi_0 Initial d_pi.
#' @param tau_e Initial tau_e.
#' @param update_order Initial update order.
#' @param seed Seed for reproducibility.
#' @return A named list with elements \code{mu_0}, \code{omega_0},
#'   \code{c_pi_0}, \code{d_pi_0}, \code{tau_e}, and \code{update_order}.
#' @examples
#' \donttest{
#' set.seed(1)
#' X <- matrix(rnorm(30 * 20), 30, 20)
#' Y <- rnorm(30)
#' init <- get.initials.spxlvb(X = X, Y = Y)
#' }
#' @importFrom glmnet cv.glmnet
#' @importFrom stats predict coef
#' @export
get.initials.spxlvb <- function(
  X, # design matrix
  Y, # response vector
  mu_0 = NULL, # Variational Normal mean estimated beta coefficient from lasso, posterior expectation of bj|sj = 1
  omega_0 = NULL, # Variational probability, expectation that the coefficient from lasso is not zero, the posterior expectation of sj
  c_pi_0 = NULL, # π ∼ Beta(aπ, bπ), known/estimated
  d_pi_0 = NULL, # π ∼ Beta(aπ, bπ), known/estimated
  tau_e = NULL, # errors iid N(0, tau_e^{-1}), known/estimated
  update_order = NULL,
  seed = 12376 # seed for cv
) {
  set.seed(seed)

  ### dimensions ----
  p <- ncol(X)

  lasso_cv <- NULL

  if (any(c(is.null(tau_e), is.null(c_pi_0), is.null(d_pi_0), is.null(omega_0)))) {
    lasso_cv <- glmnet::cv.glmnet(
      X,
      Y,
      alpha = 1,
      family = "gaussian",
      standardize = FALSE,
      standardize.response = FALSE,
      intercept = FALSE,
      parallel = TRUE
    )

    nz_ind_lambda.min <- predict(
      lasso_cv,
      s = "lambda.min",
      type = "nonzero"
    )$lambda.min

    nz_min <- min(
      length(nz_ind_lambda.min),
      length(Y) - 2
    )

    yhat <- predict(lasso_cv, X, s = "lambda.min", type = "response")

    noise_sd <- sqrt(sum((Y - yhat)^2) / (length(Y) - nz_min - 1))

    tau_e <- 1 / noise_sd^2

    s_hat <- max(c(nz_min, 1))

    if (is.null(c_pi_0)) {
      c_pi_0 <- s_hat * exp(0.5)
    }

    if (is.null(d_pi_0)) {
      d_pi_0 <- p - c_pi_0
    }

    if (is.null(omega_0)) {
      omega_0 <- rep(s_hat / p, p)
      if (length(nz_ind_lambda.min) > 0) {
        omega_0[nz_ind_lambda.min] <- 1 - s_hat / p
      }
    }
  }

  if (is.null(mu_0)) {
    if (is.null(lasso_cv)) {
      lasso_cv <- glmnet::cv.glmnet(
        X, Y,
        alpha = 1, family = "gaussian",
        standardize = FALSE, intercept = FALSE, parallel = TRUE
      )
    }
    mu_0_raw <- as.numeric(coef(lasso_cv, s = "lambda.min"))[-1]

    # Calculate scaling factor to adjust Lasso shrinkage
    appx_mn_0 <- X %*% mu_0_raw
    denom <- sum(appx_mn_0^2)

    if (denom > 0) {
      scaling_factor <- sum(appx_mn_0 * Y) / denom
      mu_0 <- mu_0_raw * scaling_factor
    } else {
      mu_0 <- mu_0_raw
    }
  }

  # generate prioritized updating order

  if (!is.null(update_order) && length(update_order) < 2) {
    if (update_order == "random") {
      update_order <- sample(1:p, p) - 1
    }
  }
  if (is.null(update_order)) {
    # Order by absolute value of mu, increasingly, for features only
    update_order <- order(abs(mu_0), decreasing = FALSE)
    update_order <- update_order - 1 # Convert to 0-based indexing for C++
  }

  return(
    list(
      mu_0 = mu_0, # Variational Normal mean estimated beta coefficient from lasso, posterior expectation of bj|sj = 1
      omega_0 = omega_0, # Variational probability, expectation that the coefficient from lasso is not zero, the posterior expectaion of sj
      c_pi_0 = c_pi_0, # π ∼ Beta(aπ, bπ), known/estimated
      d_pi_0 = d_pi_0, # π ∼ Beta(aπ, bπ), known/estimated
      tau_e = tau_e, # errors iid N(0, tau_e^{-1}), known/estimated
      update_order = update_order
    )
  )
}
