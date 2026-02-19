# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

`spxlvb` implements **Parameter Exploded Variational Bayes** (SPXLVB) for sparse linear regression with spike-and-slab priors. The key idea is a "parameter explosion" trick: at each coordinate update, a scalar `alpha_j` is optimally chosen to rescale all other variational parameters, then a global `alpha_{p+1}` is applied across all coordinates. This improves convergence over standard coordinate-ascent VB.

## Build & Test Commands

```r
devtools::load_all()                      # load for interactive dev
devtools::document()                      # rebuild docs + NAMESPACE (always before commit)
devtools::test()                          # run full test suite
devtools::test(filter = "spxlvb")        # run tests/testthat/test-spxlvb.R only
devtools::check()                         # full CRAN check
Rcpp::compileAttributes()                 # regenerate RcppExports after changing C++ exports
```

After adding or removing `// [[Rcpp::export]]` in any `.cpp` file, run `devtools::document()` (which calls `compileAttributes()` internally).

## Architecture & Data Flow

**Entry points (in order of complexity):**

| Function | File | Purpose |
|---|---|---|
| `spxlvb()` | `R/spxlvb.R` | Main fitter: standardize → init → C++ VB loop → unscale |
| `cv.spxlvb()` | `R/cv.spxlvb.R` | k-fold CV over `alpha_prior_precision` grid |
| `cv.spxlvb.fit()` | `R/cv.spxlvb.R` | CV + final refit on full data |
| `grid.search.spxlvb.fit()` | `R/grid.search.spxlvb.fit.R` | 2D grid search over `alpha_prior_precision` × `b_prior_precision`; can use validation set or ELBO |

**Internal helpers:**

- `get.initials.spxlvb()` (`R/get.initials.spxlvb.R`): Runs `cv.glmnet` (LASSO) to initialise `mu_0`, `omega_0`, `tau_e`, `c_pi_0`, `d_pi_0`, and `update_order` (sorted by `|mu_0|` ascending, 0-indexed for C++).
- `standardize_data()` (`R/utils.R`): Centers Y, centers+scales X (using population SDs, not sample SDs via `colMeans(X_c^2)`). Called by `spxlvb()` and grid/CV wrappers before passing scaled data to `get.initials.spxlvb()`.

**C++ backend (`src/`):**

- `run_vb_updates_cpp()` (`fit_linear_gaussian_exploded.cpp`): Core coordinate-ascent VB loop. Precomputes `X'Y`, `X^2` column sums, `dot(Y,Y)`. Calls `calculate_lambda_eta_sigma_update_cpp` twice per coordinate (for `s_j=0` and `s_j=1`), computes optimal `alpha_j`, then propagates the rescaling to `mu`, `sigma`, `tau_b`, `mu_alpha`, `W`, `var_W`.
- `calculate_lambda_eta_sigma_update_cpp()` (`common_helpers.cpp`): Assembles the 2×2 precision matrix `Lambda_j` and its inverse for a single (j, s_j) combination; returns `Lambda_j` and the posterior mean `eta_j`.
- `compute_elbo_cpp()` (`common_helpers.cpp`): ELBO computed once per outer iteration (not used for convergence checking—see below).
- `sigmoid_cpp()` (`common_helpers.cpp`): Numerically stable sigmoid, clamped at ±32.

**Convergence criterion:** Not ELBO-based. Uses a chi-squared test on the normalised change in the linear predictor `W = X(ω⊙μ)`: `max_j((W_old - W)^2 / var_W_j) / log(n)`, compared against chi-sq(1) CDF. Convergence when `p-value < tol`.

## Key Parameter Relationships

- Slab prior: `b_j | s_j=1 ~ N(0, (τ_ε τ_b_j)^{-1})` — `b_prior_precision` is `τ_b_j` (scalar or vector of length p).
- Alpha prior: `α_j ~ N(μ_α_j, (τ_ε τ_α)^{-1})` — `alpha_prior_precision` is `τ_α` (scalar).
- Both are divided by `tau_e` internally before being passed to C++ as `tau_alpha` and `tau_b`.
- The `mu_alpha` vector has length `p+1`; the `(p+1)`-th element governs the global rescaling step.

## Parallelism

CV and grid search use `foreach` with `%dopar%`/`%do%`. To enable parallel execution, register a backend before calling:

```r
doParallel::registerDoParallel(cores = 4)
```

Both functions fall back to sequential with a warning if `parallel = TRUE` but no backend is registered.

## Test Fixtures

Tests use `matern.data.gen()` and `get.L()` from `R/simulate_mattern_data.R` to generate correlated, sparse-signal regression problems. The shared fixture in `test-spxlvb.R` uses `p=100, n=30` (high-dimensional) with a Matérn covariance structure.
