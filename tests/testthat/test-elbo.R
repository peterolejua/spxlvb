# Tests for ELBO correctness: interior optimum, consistency, boundary warnings

setup_elbo_problem <- function() {
    set.seed(1)
    n <- 50
    p <- 20
    X <- matrix(rnorm(n * p), n, p)
    beta_true <- c(1, -1, 0.5, rep(0, p - 3))
    Y <- X %*% beta_true + rnorm(n)
    list(X = X, Y = Y, beta_true = beta_true, n = n, p = p)
}

# =====================================================================
# 1. Regression test: 6-term ELBO is monotone increasing in tau_alpha
# =====================================================================

test_that("6-term ELBO is monotone increasing in tau_alpha", {
    dat <- setup_elbo_problem()

    tau_grid <- c(1, 100, 10000, 1e6)
    elbos <- numeric(length(tau_grid))

    for (i in seq_along(tau_grid)) {
        fit <- spxlvb(
            X = dat$X, Y = dat$Y,
            alpha_prior_precision = tau_grid[i],
            tol = 1e-5, max_iter = 500,
            convergence = "elbo_relative",
            save_history = FALSE,
            intercept = TRUE, seed = 123
        )
        elbos[i] <- fit$exploded_elbo
    }

    diffs <- diff(elbos)
    expect_true(
        all(diffs >= -1e-6),
        info = paste(
            "6-term ELBO should be monotone increasing in tau_alpha.",
            "Values:", paste(sprintf("%.2f", elbos), collapse = ", "),
            "at tau_alpha:", paste(tau_grid, collapse = ", ")
        )
    )
})

# =====================================================================
# 2. ELBO component consistency: parts sum to total
# =====================================================================

test_that("ELBO components sum to the reported total", {
    dat <- setup_elbo_problem()

    fit <- spxlvb(
        X = dat$X, Y = dat$Y,
        alpha_prior_precision = 1000,
        tol = 1e-5, max_iter = 500,
        convergence = "elbo_relative",
        save_history = FALSE,
        intercept = TRUE, seed = 123,
        standardize = TRUE
    )

    post <- fit$approximate_posterior
    mu <- as.numeric(post$mu)
    sigma <- as.numeric(post$sigma)
    omega <- as.numeric(post$omega)
    mu_alpha <- as.numeric(post$mu_alpha)
    tau_b <- as.numeric(post$tau_b)

    scaled <- standardize_data(dat$X, dat$Y, TRUE)
    initials <- get_initials_spxlvb(
        X = scaled$X_cs, Y = scaled$Y_c, seed = 123
    )
    tau_e <- initials$tau_e
    tau_alpha_internal <- 1000 / tau_e
    pi_fixed <- initials$c_pi_0 / (initials$c_pi_0 + initials$d_pi_0)

    eta_bar <- scaled$X_cs %*% (omega * mu)
    y_sq <- sum(scaled$Y_c^2)
    y_dot_eta_bar <- sum(scaled$Y_c * eta_bar)
    g <- mu^2 * omega * (1 - omega) + sigma^2 * omega
    X_col_sq <- colSums(scaled$X_cs^2)
    var_eta <- sum(X_col_sq * g)
    zeta <- var_eta + sum(eta_bar^2)

    elbo_parts <- compute_elbo_cpp(
        mu, sigma, omega, tau_b, mu_alpha,
        y_sq, y_dot_eta_bar, zeta, tau_alpha_internal, tau_e,
        initials$c_pi_0, initials$d_pi_0
    )

    reconstructed <- elbo_parts$slab_entropy + elbo_parts$spike_entropy +
        elbo_parts$data_fit + elbo_parts$slab_normalisation +
        elbo_parts$slab_penalty +
        elbo_parts$alpha_normalisation - elbo_parts$alpha_penalty +
        elbo_parts$pi_posterior + elbo_parts$pi_normalisation

    expect_equal(
        elbo_parts$ELBO, reconstructed,
        tolerance = 1e-8,
        info = "ELBO should equal sum of its 9 component terms"
    )

    expect_true(
        elbo_parts$alpha_normalisation > 0,
        info = "Alpha normalisation (G) should be positive"
    )
    expect_true(
        elbo_parts$alpha_penalty >= 0,
        info = "Alpha penalty (H) should be non-negative"
    )
})

test_that("ELBO computed in R matches C++ compute_elbo_cpp", {
    dat <- setup_elbo_problem()
    p <- dat$p

    fit <- spxlvb(
        X = dat$X, Y = dat$Y,
        alpha_prior_precision = 1000,
        tol = 1e-5, max_iter = 500,
        convergence = "elbo_relative",
        save_history = FALSE,
        intercept = TRUE, seed = 123,
        standardize = TRUE
    )

    post <- fit$approximate_posterior
    mu <- as.numeric(post$mu)
    sigma <- as.numeric(post$sigma)
    omega <- as.numeric(post$omega)
    mu_alpha <- as.numeric(post$mu_alpha)
    tau_b <- as.numeric(post$tau_b)

    scaled <- standardize_data(dat$X, dat$Y, TRUE)
    initials <- get_initials_spxlvb(
        X = scaled$X_cs, Y = scaled$Y_c, seed = 123
    )
    tau_e <- initials$tau_e
    tau_alpha <- 1000 / tau_e
    c_pi <- initials$c_pi_0
    d_pi <- initials$d_pi_0

    eta_bar <- scaled$X_cs %*% (omega * mu)
    y_sq <- sum(scaled$Y_c^2)
    y_dot_eta_bar <- sum(scaled$Y_c * eta_bar)
    X_col_sq <- colSums(scaled$X_cs^2)
    g <- mu^2 * omega * (1 - omega) + sigma^2 * omega
    var_eta <- sum(X_col_sq * g)
    zeta <- var_eta + sum(eta_bar^2)

    sigma2 <- sigma^2
    l_omega <- pmax(log(omega), -500)
    l_omega_m1 <- pmax(log(1 - omega), -500)

    slab_entropy_r <- sum(omega * (log(sigma) - l_omega))

    log_slab_prec <- log(tau_e * tau_b)
    spike_entropy_r <- -sum((1 - omega) * (0.5 * log_slab_prec + l_omega_m1))

    expected_b_sq <- omega * (mu^2 + sigma2) + (1 - omega) / (tau_e * tau_b)
    residual_sq <- y_sq - 2 * y_dot_eta_bar + zeta

    data_fit_r <- -0.5 * tau_e * residual_sq
    slab_normalisation_r <- 0.5 * sum(log_slab_prec)
    slab_penalty_r <- -0.5 * tau_e * sum(tau_b * expected_b_sq)
    alpha_normalisation_r <- 0.5 * (p + 1) * log(tau_e * tau_alpha)
    alpha_penalty_r <- 0.5 * tau_e * tau_alpha * sum((1 - mu_alpha)^2)

    sum_omega <- sum(omega)
    pi_posterior_r <- lbeta(c_pi + sum_omega, d_pi + (p - sum_omega))
    pi_normalisation_r <- -lbeta(c_pi, d_pi)

    elbo_r <- slab_entropy_r + spike_entropy_r +
        data_fit_r + slab_normalisation_r + slab_penalty_r +
        alpha_normalisation_r - alpha_penalty_r +
        pi_posterior_r + pi_normalisation_r

    elbo_cpp <- compute_elbo_cpp(
        mu, sigma, omega, tau_b, mu_alpha,
        y_sq, y_dot_eta_bar, zeta, tau_alpha, tau_e, c_pi, d_pi
    )

    expect_equal(elbo_r, elbo_cpp$ELBO, tolerance = 1e-6,
        info = "R and C++ ELBO computations should match")
})

# =====================================================================
# 3. Grid boundary warnings
# =====================================================================

test_that("tune_spxlvb warns when optimal is at grid maximum", {
    dat <- setup_elbo_problem()

    # 6-term ELBO is monotone increasing in tau_alpha, so the optimal
    # is always at the grid maximum. A small grid triggers the warning.
    expect_warning(
        tune_spxlvb(
            X = dat$X, Y = dat$Y,
            criterion = "elbo",
            alpha_prior_precision_grid = c(0.001, 0.01),
            max_iter = 30, tol = 1e-2,
            parallel = FALSE, verbose = FALSE
        ),
        "maximum of the search grid"
    )
})

test_that("tune_spxlvb warns at grid maximum for any grid (monotone ELBO)", {
    dat <- setup_elbo_problem()

    # With the 6-term ELBO monotone in tau_alpha, any finite grid
    # will have its optimum at the maximum.
    expect_warning(
        suppressMessages(
            tune_spxlvb(
                X = dat$X, Y = dat$Y,
                criterion = "elbo",
                alpha_prior_precision_grid = c(100, 1000, 10000, 1e5),
                max_iter = 50, tol = 1e-3,
                parallel = FALSE, verbose = FALSE
            )
        ),
        "maximum of the search grid"
    )
})
