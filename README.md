
# spxlvb

<!-- badges: start -->
[![R-CMD-check](https://img.shields.io/badge/R--CMD--check-passing-brightgreen)](https://github.com/peterolejua/spxlvb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
<!-- badges: end -->

**Parameter Exploded Variational Bayes** for sparse linear regression
with spike-and-slab priors.

Standard mean-field Variational Bayes (VB) for high-dimensional
regression tends to underestimate posterior variance and produce poorly
calibrated credible intervals. `spxlvb` addresses this by introducing a
*parameter explosion* step: at each coordinate update an optimal scalar
rescaling is applied to all other variational parameters, improving both
convergence speed and calibration.

## Features

- Coordinate-ascent VB with per-coordinate and global parameter
  explosion steps
- Spike-and-slab prior for automatic variable selection
- LASSO-based initialisation via `glmnet`
- ELBO and chi-squared convergence criteria
- Cross-validation (`cv.spxlvb.fit`) and 2-D grid search
  (`grid.search.spxlvb.fit`) for hyperparameter tuning
- C++ backend via `RcppArmadillo` for fast linear algebra
- Parallel evaluation of tuning grids via `foreach`

## Parameter structure

The "parameter explosion" introduces **p+1 expansion parameters**
α₁, …, αₚ, α_{p+1}. The +1 comes from the global expansion
parameter, **not** from an intercept term.

| R argument | Dimension | Default | Role |
|---|---|---|---|
| `mu_alpha` | p+1 | all ones | Prior means for expansion parameters α₁,…,αₚ (per-coordinate) and α_{p+1} (global, applied after each full sweep via `use_global_alpha`). The +1 is for the global parameter, **not** an intercept. |
| `alpha_prior_precision` | scalar | `1000` | Shared prior precision τ_α for all p+1 expansion parameters. Larger = closer to standard VB. |
| `b_prior_precision` | p | all ones | Slab prior precisions τ_{b,j} for the regression coefficients (**not** the expansion parameters). |

When all expansion parameters equal 1, the algorithm reduces to
standard coordinate-ascent VB.

**Intercept:** When `intercept = TRUE`, the model is fit on
centered-and-scaled data. After convergence, the intercept is
recovered as β₀ = Ȳ − Σⱼ β̂ⱼ X̄ⱼ and prepended to the `beta`
vector. The returned `beta` has length p+1 (intercept + p
coefficients), but this +1 is unrelated to the expansion parameter
dimension.

## Installation

Install the development version from GitHub:

```r
# install.packages("remotes")
remotes::install_github("peterolejua/spxlvb")
```

## Quick example

```r
library(spxlvb)

# Simulate a sparse regression problem
set.seed(42)
n <- 100; p <- 200
X <- matrix(rnorm(n * p), n, p)
beta_true <- c(rep(2, 5), rep(0, p - 5))
Y <- X %*% beta_true + rnorm(n)

# Fit the model
fit <- spxlvb(X, Y)

# Estimated coefficients (posterior mean x inclusion probability)
head(fit$beta[-1], 10)   # first element is the intercept

# Posterior inclusion probabilities
head(fit$omega, 10)

# Convergence
cat("Converged:", fit$converged, "in", fit$iterations, "iterations\n")
```

## Hyperparameter tuning

`spxlvb` exposes two prior precision parameters that can be tuned:

- `alpha_prior_precision` controls how tightly the expansion
  parameter is centred around 1 (higher = closer to standard VB).
- `b_prior_precision` controls the slab width of the spike-and-slab
  prior.

Use **cross-validation** when no validation set is available:

```r
cv_fit <- cv.spxlvb.fit(
  k = 5, X = X, Y = Y,
  alpha_prior_precision_grid = c(0, 10, 100, 1000),
  parallel = FALSE
)
cv_fit$alpha_prior_precision_grid_opt
```

Or a **grid search** with a held-out validation set:

```r
grid_fit <- grid.search.spxlvb.fit(
  X = X, Y = Y,
  X_validation = X_val, Y_validation = Y_val,
  alpha_prior_precision_grid = c(10, 100, 1000),
  b_prior_precision_grid = c(0.5, 1, 5),
  parallel = FALSE
)
```

## Documentation

See the package vignette for a full worked example:

```r
vignette("spxlvb-tutorial", package = "spxlvb")
```

## Citation

If you use `spxlvb` in published work, please cite:

> Olejua, P. and McLain, A. (2025). Parameter Exploded Variational Bayes
> for High-Dimensional Linear Regression with Spike-and-Slab Priors.
> *Working paper.*

## License

MIT
