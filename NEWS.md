# spxlvb 0.1.0

## Initial release

* Core fitting function `spxlvb()` implementing parameter-exploded
  coordinate-ascent VB for spike-and-slab linear regression.
* Per-coordinate expansion parameter (`alpha_j`) and global rescaling
  step (`alpha_{p+1}`).
* ELBO-based convergence criterion (default) and chi-squared alternative.
* LASSO-based initialisation via `glmnet`.
* Cross-validation wrapper `cv.spxlvb.fit()` for tuning
  `alpha_prior_precision`.
* Two-dimensional grid search `grid.search.spxlvb.fit()` over
  `(alpha_prior_precision, b_prior_precision)` using validation-set MSE
  or ELBO.
* C++ backend via RcppArmadillo for the inner VB loop.
* Parallel grid evaluation via `foreach`.
* Matern data generation utilities (`get.L`, `matern.data.gen`) for
  simulation studies.
