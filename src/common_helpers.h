#ifndef COMMON_HELPERS_H
#define COMMON_HELPERS_H

#include <RcppArmadillo.h>
#include <tuple>

// Function declaration for compute_elbo()
Rcpp::List compute_elbo_cpp(
    const arma::vec& mu,
    const arma::vec& sigma,
    const arma::vec& omega,
    const arma::vec& tau_b,
    const arma::vec& mu_alpha,
    double Y2,
    double t_YW,
    double t_W2,
    double tau_alpha,
    double tau_e,
    double pi_fixed
);


// Function declaration for sigmoid()
double sigmoid_cpp(const double &x);

// Function declaration for calculate_lambda_eta_sigma()
Rcpp::List calculate_lambda_eta_sigma_update_cpp(
    const arma::mat& X,
    const arma::vec& X_2_col_sums,   // precomputed column sums of X^2
    const arma::vec& YX_vec,  // precomputed t(X) %*% Y
    const arma::vec& Y,
    const arma::vec& W_j,
    double W_j_squared,
    double s_j_val,
    unsigned int j,           // 0-based index
    double tau_e,
    const arma::vec& tau_b,
    double tau_alpha,
    const arma::vec& mu_alpha
);

#endif // COMMON_HELPERS_H
