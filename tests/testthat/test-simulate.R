# Smoke tests for data generation utilities

test_that("get.L returns valid Cholesky factor", {
    p <- 50
    L <- get.L(p = p, range = 2.0, smoothness = 1.5)

    expect_true(is.matrix(L))
    expect_equal(dim(L), c(p, p))
    # Lower triangular: upper triangle should be all zeros
    expect_true(all(L[upper.tri(L)] == 0))
    # Attributes stored

    expect_equal(attr(L, "range"), 2.0)
    expect_equal(attr(L, "smoothness"), 1.5)
})

test_that("matern.data.gen produces correct dimensions (p >> n)", {
    p <- 100
    n <- 30
    pi_0 <- 0.10

    L <- get.L(p = p, range = 2.0, smoothness = 1.5)
    dat <- matern.data.gen(
        seed_val = 42, n = n, p = p, pi_0 = pi_0,
        L = L, sd_beta = 1, SNR = 2,
        n_test = 20, n_validation = 15
    )

    # Training data

    expect_equal(dim(dat$X), c(n, p))
    expect_equal(length(dat$Y), n)
    expect_equal(length(dat$beta), p)

    # Sparsity check
    s_active <- sum(dat$beta != 0)
    expect_true(s_active > 0)
    expect_true(s_active <= ceiling(p * pi_0) + 1)

    # Test set
    expect_equal(dim(dat$X_test), c(20, p))
    expect_equal(length(dat$Y_test), 20)

    # Validation set
    expect_equal(dim(dat$X_validation), c(15, p))
    expect_equal(length(dat$Y_validation), 15)
})

test_that("matern.data.gen errors without L matrix", {
    expect_error(
        matern.data.gen(seed_val = 1, n = 10, p = 20, pi_0 = 0.1),
        "L matrix must be provided"
    )
})
