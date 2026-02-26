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
# 1. Regression test: ELBO has non-monotone interior optimum for tau_alpha
# =====================================================================

test_that("ELBO has interior optimum in tau_alpha (not monotone)", {
    dat <- setup_elbo_problem()

    tau_grid <- c(1, 100, 10000, 1e6)
    elbos <- numeric(length(tau_grid))

    for (i in seq_along(tau_grid)) {
        fit <- spxlvb(
            X = dat$X, Y = dat$Y,
            alpha_prior_precision = tau_grid[i],
            tol = 1e-5, max_iter = 500,
            convergence = "elbo",
            save_history = FALSE,
            intercept = TRUE, seed = 123
        )
        elbos[i] <- fit$elbo
    }

    diffs <- diff(elbos)
    is_monotone_inc <- all(diffs >= 0)
    is_monotone_dec <- all(diffs <= 0)

    expect_false(
        is_monotone_inc || is_monotone_dec,
        info = paste(
            "ELBO should be non-monotone in tau_alpha.",
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
        convergence = "elbo",
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

    W <- scaled$X_cs %*% (omega * mu)
    Y2 <- sum(scaled$Y_c^2)
    t_YW <- sum(scaled$Y_c * W)
    g <- mu^2 * omega * (1 - omega) + sigma^2 * omega
    X2_col_sums <- colSums(scaled$X_cs^2)
    var_W <- sum(X2_col_sums * g)
    t_W2 <- var_W + sum(W^2)

    elbo_parts <- compute_elbo_cpp(
        mu, sigma, omega, tau_b, mu_alpha,
        Y2, t_YW, t_W2, tau_alpha_internal, tau_e, pi_fixed
    )

    reconstructed <- elbo_parts$Sum_a + elbo_parts$Sum_b +
        elbo_parts$Datafit + elbo_parts$term_norm + elbo_parts$pi_term

    expect_equal(
        elbo_parts$ELBO, reconstructed,
        tolerance = 1e-8,
        info = "ELBO should equal sum of its component terms"
    )

    expect_equal(
        elbo_parts$sum_taua + elbo_parts$sum_taub,
        elbo_parts$term_norm,
        tolerance = 1e-10,
        info = "term_norm should equal sum_taua + sum_taub"
    )

    expect_true(
        elbo_parts$sum_taua > 0,
        info = "Alpha prior normalization (sum_taua) should be positive"
    )
})

test_that("ELBO computed in R matches C++ compute_elbo_cpp", {
    dat <- setup_elbo_problem()
    p <- dat$p

    fit <- spxlvb(
        X = dat$X, Y = dat$Y,
        alpha_prior_precision = 1000,
        tol = 1e-5, max_iter = 500,
        convergence = "elbo",
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
    pi_fixed <- initials$c_pi_0 / (initials$c_pi_0 + initials$d_pi_0)

    W <- scaled$X_cs %*% (omega * mu)
    Y2 <- sum(scaled$Y_c^2)
    t_YW <- sum(scaled$Y_c * W)
    X2_col_sums <- colSums(scaled$X_cs^2)
    g <- mu^2 * omega * (1 - omega) + sigma^2 * omega
    var_W <- sum(X2_col_sums * g)
    t_W2 <- var_W + sum(W^2)

    sigma2 <- sigma^2
    l_omega <- pmax(log(omega), -500)
    l_omega_m1 <- pmax(log(1 - omega), -500)

    sum_a_r <- sum(omega * (log(sigma) - l_omega))

    log_tau_e_tau_b <- log(tau_e * tau_b)
    sum_b_r <- -sum((1 - omega) * (0.5 * log_tau_e_tau_b + l_omega_m1))

    inside <- omega * (mu^2 + sigma2) + (1 - omega) / (tau_e * tau_b)
    sum_taub_inside <- sum(tau_b * inside)
    resid_term <- Y2 - 2 * t_YW + t_W2
    alpha_penalty <- tau_alpha * sum((1 - mu_alpha)^2)
    datafit_r <- -0.5 * tau_e * (resid_term + sum_taub_inside + alpha_penalty)

    term_norm_r <- 0.5 * sum(log_tau_e_tau_b) +
        0.5 * (p + 1) * log(tau_e * tau_alpha)

    logodds <- log(pi_fixed / (1 - pi_fixed))
    pi_term_r <- logodds * sum(omega)

    elbo_r <- sum_a_r + sum_b_r + datafit_r + term_norm_r + pi_term_r

    elbo_cpp <- compute_elbo_cpp(
        mu, sigma, omega, tau_b, mu_alpha,
        Y2, t_YW, t_W2, tau_alpha, tau_e, pi_fixed
    )

    expect_equal(elbo_r, elbo_cpp$ELBO, tolerance = 1e-6,
        info = "R and C++ ELBO computations should match")
})

# =====================================================================
# 3. Grid boundary warnings
# =====================================================================

test_that("tune_spxlvb warns when optimal is at grid minimum", {
    dat <- setup_elbo_problem()

    expect_warning(
        tune_spxlvb(
            X = dat$X, Y = dat$Y,
            criterion = "elbo",
            alpha_prior_precision_grid = c(1e6, 1e7),
            max_iter = 30, tol = 1e-2,
            parallel = FALSE, verbose = FALSE
        ),
        "minimum of the search grid"
    )
})

test_that("tune_spxlvb warns when optimal is at grid maximum", {
    dat <- setup_elbo_problem()

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

test_that("tune_spxlvb does NOT warn when optimal is interior", {
    dat <- setup_elbo_problem()

    expect_no_warning(
        suppressMessages(
            tune_spxlvb(
                X = dat$X, Y = dat$Y,
                criterion = "elbo",
                alpha_prior_precision_grid = c(100, 1000, 10000, 1e5),
                max_iter = 50, tol = 1e-3,
                parallel = FALSE, verbose = FALSE
            )
        )
    )
})
