# Smoke tests for get.initials.spxlvb

test_that("get.initials.spxlvb returns all expected components", {
    set.seed(123)
    p <- 100
    n <- 30
    X <- matrix(rnorm(n * p), n, p)
    Y <- rnorm(n)

    initials <- get.initials.spxlvb(X = X, Y = Y)

    # All list elements exist
    expect_true(all(c(
        "mu_0", "omega_0", "c_pi_0", "d_pi_0",
        "tau_e", "update_order"
    ) %in% names(initials)))

    # Correct dimensions
    expect_equal(length(initials$mu_0), p)
    expect_equal(length(initials$omega_0), p)
    expect_equal(length(initials$update_order), p)

    # Scalars
    expect_length(initials$c_pi_0, 1)
    expect_length(initials$d_pi_0, 1)
    expect_length(initials$tau_e, 1)

    # tau_e should be positive
    expect_true(initials$tau_e > 0)

    # omega values in [0, 1]
    expect_true(all(initials$omega_0 >= 0 & initials$omega_0 <= 1))

    # update_order is 0-indexed for C++
    expect_true(all(initials$update_order >= 0))
    expect_true(all(initials$update_order < p))
})

test_that("get.initials.spxlvb respects pre-supplied values", {
    set.seed(456)
    p <- 50
    n <- 20
    X <- matrix(rnorm(n * p), n, p)
    Y <- rnorm(n)

    custom_omega <- rep(0.5, p)
    custom_tau <- 2.0

    initials <- get.initials.spxlvb(
        X = X, Y = Y,
        omega_0 = custom_omega,
        tau_e = custom_tau,
        c_pi_0 = 5,
        d_pi_0 = 95
    )

    # Pre-supplied values should pass through unchanged
    expect_equal(initials$omega_0, custom_omega)
    expect_equal(initials$tau_e, custom_tau)
    expect_equal(initials$c_pi_0, 5)
    expect_equal(initials$d_pi_0, 95)
})
