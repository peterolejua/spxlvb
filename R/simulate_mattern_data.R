#' @title Compute Cholesky Factor for Matern Correlation
#' @description Pre-computes the Lower Cholesky factor (L) for a 1D Matern
#' covariance matrix. This should be called once per simulation setting to
#' save computational time.
#' @param p An integer. The number of predictors (dimension).
#' @param range A numeric value. Controls the distance at which correlation decays.
#' @param smoothness A numeric value (nu). Controls differentiability (0.5, 1.5, or 2.5).
#' @param phi A numeric value. The marginal variance. Defaults to 1.0.
#' @importFrom stats dist
#' @export
get.L <- function(p, range, smoothness, phi = 1.0) {
  # 1. Generate Covariance Structure
  dist_matrix <- as.matrix(dist(1:p))
  cov_matrix <- Matern_Internal(dist_matrix, range, smoothness, phi)

  # 2. Cholesky decomposition: cov = L %*% t(L)
  # t(chol()) gives the Lower triangular matrix
  L <- t(chol(cov_matrix))

  # Attribute parameters so they can be retrieved by the data gen if needed
  attr(L, "range") <- range
  attr(L, "smoothness") <- smoothness

  return(L)
}

#' @title Simulate Matern Correlated High-Dimensional Data
#' @description This function generates a synthetic dataset where the predictors (X)
#' follow a spatial correlation structure defined by a pre-computed L matrix.
#' @param seed_val An integer. The random seed for reproducibility.
#' @param n An integer. The number of observations.
#' @param p An integer. The number of predictors.
#' @param pi_0 A numeric value between 0 and 1. The proportion of non-zero coefficients.
#' @param L The Lower Cholesky matrix provided by \code{get.L}.
#' @param sd_beta A numeric value. The standard deviation of the non-zero coefficients.
#' @param SNR A numeric value. The signal-to-noise ratio.
#' @param n_test An integer. Number of test observations.
#' @param n_validation An integer. Number of validation observations.
#' @importFrom stats rnorm var
#' @export
matern.data.gen <- function(
    seed_val,
    n,
    p,
    pi_0,
    L = NULL,
    sd_beta = 1,
    SNR = 2,
    n_test = NULL,
    n_validation = NULL
) {

  if (is.null(L)) {
    stop("The L matrix must be provided. Use get.L() to generate it first.")
  }

  set.seed(seed_val)

  # 1. Generate Training Data
  X <- gen_correlated_X(n, L, p)

  # 2. Locations of signals (contiguous block)
  s <- floor(p * pi_0)
  s_dat <- sample(seq(floor(s/2) + 1, floor(p - s/2)), 1)
  active_indices <- seq(s_dat - floor(s/2), s_dat + floor(s/2) - 1 * (s %% 2 == 0))

  # 3. Generate Sparse Beta
  beta <- numeric(p)
  beta[active_indices] <- rnorm(s, mean = 0, sd = sd_beta)

  # 4. Generate Response Y
  true_eta <- X %*% beta
  sd_noise <- sqrt(var(as.vector(true_eta)) / SNR)
  Y <- true_eta + rnorm(n, sd = sd_noise)

  # Build results list
  results <- list(
    Y = Y,
    X = X,
    beta = beta,
    sd_noise = sd_noise,
    range = attr(L, "range"),
    smoothness = attr(L, "smoothness"),
    SNR = SNR,
    pi_0 = pi_0,
    active_indices = active_indices
  )

  # 5. Generate sets if requested
  if (!is.null(n_test)) {
    results$X_test <- gen_correlated_X(n_test, L, p)
    results$Y_test <- (results$X_test %*% beta) + rnorm(n_test, sd = sd_noise)
  }

  if (!is.null(n_validation)) {
    results$X_validation <- gen_correlated_X(n_validation, L, p)
    results$Y_validation <- (results$X_validation %*% beta) + rnorm(n_validation, sd = sd_noise)
  }

  return(results)
}

# --- INTERNAL HELPERS (Not Exported) ---

Matern_Internal <- function(d, range, smoothness, phi) {
  alpha <- 1 / range
  nu <- smoothness
  d <- d * alpha

  if (nu == 0.5) return(phi * exp(-d))
  if (nu == 1.5) return(phi * (1 + d) * exp(-d))
  if (nu == 2.5) return(phi * (1 + d + d^2 / 3) * exp(-d))

  d[d == 0] <- 1e-10
  res <- phi * (2^(1 - nu) / gamma(nu)) * (d^nu) * besselK(d, nu)
  res[is.na(res)] <- phi
  return(res)
}

gen_correlated_X <- function(num_rows, L, p) {
  Z <- matrix(rnorm(num_rows * p), num_rows, p)
  return(Z %*% t(L))
}
