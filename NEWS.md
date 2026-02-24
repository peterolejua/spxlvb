# spxlvb 0.0.0.9002

## API refactor: unified tuning interface

* New `tune_spxlvb()` function replaces `grid_search_spxlvb_fit()` and
  `cv_spxlvb_fit()` with an explicit `criterion` argument:
  - `criterion = "elbo"` (default): ELBO maximisation over hyperparameter grid.
  - `criterion = "cv"`: k-fold cross-validation with MSPE selection.
  - `criterion = "validation"`: held-out validation set with MSPE selection.
* ELBO tuning no longer performs a redundant refit: the winning fit from
  the grid search is returned directly.
* `grid_search_spxlvb_fit()`, `cv_spxlvb_fit()`, and `cv_spxlvb()` are
  deprecated. They still work (delegating to `tune_spxlvb()` internally)
  but emit deprecation warnings.

# spxlvb 0.0.0.9001

## Initial release

* Core fitting function `spxlvb()` implementing parameter-exploded
  coordinate-ascent VB for spike-and-slab linear regression.
* Per-coordinate expansion parameter (`alpha_j`) and global rescaling
  step (`alpha_{p+1}`).
* ELBO-based convergence criterion (default) and chi-squared alternative.
* LASSO-based initialisation via `glmnet`.
* Cross-validation wrapper `cv_spxlvb_fit()` for tuning
  `alpha_prior_precision`.
* Two-dimensional grid search `grid_search_spxlvb_fit()` over
  `(alpha_prior_precision, b_prior_precision)` using validation-set MSE
  or ELBO.
* C++ backend via RcppArmadillo for the inner VB loop.
* Parallel grid evaluation via `foreach`.
* Matern data generation utilities (`get_L`, `matern_data_gen`) for
  simulation studies.
