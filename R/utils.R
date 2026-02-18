#' Internal: Standardize X and Y
#'
#' Centers Y and centers + scales X. Returns a list with
#' standardized data and the original means/scales needed to
#' back-transform coefficients.
#'
#' @param X Numeric matrix.
#' @param Y Numeric vector.
#' @param standardize Logical.
#' @return A list with `X_cs`, `Y_c`, `X_means`, `sigma_estimate`,
#'   `Y_mean` (last three only when `standardize = TRUE`).
#' @keywords internal
#' @noRd
standardize_data <- function(X, Y, standardize = TRUE) {
    if (standardize) {
        X_means <- colMeans(X)
        X_c <- scale(X, center = X_means, scale = FALSE)
        sigma_estimate <- sqrt(colMeans(X_c^2))
        X_cs <- scale(X_c, center = FALSE, scale = sigma_estimate)

        Y_mean <- mean(Y)
        Y_c <- Y - Y_mean

        list(
            X_cs = X_cs,
            Y_c = Y_c,
            X_means = X_means,
            sigma_estimate = sigma_estimate,
            Y_mean = Y_mean
        )
    } else {
        list(
            X_cs = X,
            Y_c = Y,
            X_means = NULL,
            sigma_estimate = NULL,
            Y_mean = NULL
        )
    }
}
