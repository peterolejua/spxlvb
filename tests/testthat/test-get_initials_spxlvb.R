# Smoke tests for get_initials_spxlvb

setup_init_data <- function() {
    set.seed(123)
    p <- 100
    n <- 30
    X <- matrix(rnorm(n * p), n, p)
    Y <- rnorm(n)
    list(X = X, Y = Y, n = n, p = p)
}

expected_names <- c("mu_0", "omega_0", "c_pi_0", "d_pi_0",
                    "tau_e", "update_order")

validate_init_structure <- function(initials, p) {
    expect_true(all(expected_names %in% names(initials)))
    expect_equal(length(initials$mu_0), p)
    expect_equal(length(initials$omega_0), p)
    expect_equal(length(initials$update_order), p)
    expect_length(initials$c_pi_0, 1)
    expect_length(initials$d_pi_0, 1)
    expect_length(initials$tau_e, 1)
    expect_true(initials$tau_e > 0)
    expect_true(all(initials$omega_0 >= 0 & initials$omega_0 <= 1))
    expect_true(all(initials$update_order >= 0))
    expect_true(all(initials$update_order < p))
}

test_that("each initialization strategy returns valid structure", {
    dat <- setup_init_data()
    strategies <- c("lasso", "ridge", "lasso_ridge", "null")

    for (strat in strategies) {
        initials <- get_initials_spxlvb(
            X = dat$X, Y = dat$Y,
            initialization = strat
        )
        validate_init_structure(initials, dat$p)
    }
})

test_that("default initialization is lasso (backward compatible)", {
    dat <- setup_init_data()

    init_default <- get_initials_spxlvb(X = dat$X, Y = dat$Y)
    init_lasso <- get_initials_spxlvb(
        X = dat$X, Y = dat$Y, initialization = "lasso"
    )

    expect_equal(init_default$mu_0, init_lasso$mu_0)
    expect_equal(init_default$omega_0, init_lasso$omega_0)
    expect_equal(init_default$tau_e, init_lasso$tau_e)
    expect_equal(init_default$c_pi_0, init_lasso$c_pi_0)
    expect_equal(init_default$d_pi_0, init_lasso$d_pi_0)
    expect_equal(init_default$update_order, init_lasso$update_order)
})

test_that("null strategy sets mu_0 to zero", {
    dat <- setup_init_data()
    initials <- get_initials_spxlvb(
        X = dat$X, Y = dat$Y, initialization = "null"
    )
    expect_true(all(initials$mu_0 == 0))
})

test_that("null strategy uses random update order", {
    dat <- setup_init_data()
    init1 <- get_initials_spxlvb(
        X = dat$X, Y = dat$Y, initialization = "null", seed = 1
    )
    init2 <- get_initials_spxlvb(
        X = dat$X, Y = dat$Y, initialization = "null", seed = 2
    )
    expect_false(identical(init1$update_order, init2$update_order))
})

test_that("pre-supplied values override any strategy", {
    dat <- setup_init_data()
    custom_mu <- rep(99, dat$p)
    custom_omega <- rep(0.5, dat$p)
    custom_tau <- 2.0
    custom_order <- seq(0, dat$p - 1)

    for (strat in c("lasso", "ridge", "lasso_ridge", "null")) {
        initials <- get_initials_spxlvb(
            X = dat$X, Y = dat$Y,
            mu_0 = custom_mu,
            omega_0 = custom_omega,
            c_pi_0 = 5,
            d_pi_0 = 95,
            tau_e = custom_tau,
            update_order = custom_order,
            initialization = strat
        )

        expect_equal(initials$mu_0, custom_mu)
        expect_equal(initials$omega_0, custom_omega)
        expect_equal(initials$tau_e, custom_tau)
        expect_equal(initials$c_pi_0, 5)
        expect_equal(initials$d_pi_0, 95)
        expect_equal(initials$update_order, custom_order)
    }
})

test_that("ridge strategy uses top-10% threshold for omega", {
    dat <- setup_init_data()
    initials <- get_initials_spxlvb(
        X = dat$X, Y = dat$Y, initialization = "ridge"
    )
    high_omega <- sum(initials$omega_0 > 0.5)
    expect_true(high_omega > 0 && high_omega <= ceiling(0.1 * dat$p) + 1)
})

test_that("lasso_ridge strategy uses LASSO omega with ridge mu", {
    dat <- setup_init_data()
    init_lr <- get_initials_spxlvb(
        X = dat$X, Y = dat$Y, initialization = "lasso_ridge"
    )
    init_lasso <- get_initials_spxlvb(
        X = dat$X, Y = dat$Y, initialization = "lasso"
    )

    expect_equal(init_lr$omega_0, init_lasso$omega_0)
    expect_false(identical(init_lr$mu_0, init_lasso$mu_0))
})
