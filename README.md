
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
- Cross-validation (`cv_spxlvb_fit`) and 2-D grid search
  (`grid_search_spxlvb_fit`) for hyperparameter tuning
- C++ backend via `RcppArmadillo` for fast linear algebra
- Parallel evaluation of tuning grids via `foreach`

## Parameter structure

The "parameter explosion" introduces **p+1 expansion parameters**
α₁, …, αₚ, α_{p+1}. The +1 comes from the global expansion
parameter, **not** from an intercept term.

| R argument | Dimension | Default | Role |
|---|---|---|---|
| `mu_alpha` | p+1 | all ones | Prior means for expansion parameters α₁,…,αₚ (per-coordinate) and α_{p+1} (global, applied after each full sweep). The +1 is for the global parameter, **not** an intercept. |
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

The recommended approach is **ELBO-based grid search**, which selects
the hyperparameter combination that maximises the converged evidence
lower bound. This requires only one fit per grid point (no
cross-validation loop), making it substantially faster than CV-based
tuning — and empirically at least as accurate (see the paper appendix
for details):

```r
fit <- grid_search_spxlvb_fit(
  X = X, Y = Y,
  alpha_prior_precision_grid = c(0, 30, 300, 3000, 30000),
  b_prior_precision_grid = c(0.01, 0.1, 1, 10),
  parallel = FALSE
)
```

Cross-validation is also available via `cv_spxlvb_fit()`, which
supports both 1-D search over `alpha_prior_precision` and 2-D search
over `(alpha_prior_precision, b_prior_precision)`. See the vignette
for details.

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
