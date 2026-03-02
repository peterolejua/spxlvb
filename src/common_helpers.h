#ifndef COMMON_HELPERS_H
#define COMMON_HELPERS_H

#include <RcppArmadillo.h>

Rcpp::List compute_elbo_cpp(
    const arma::vec& mu,
    const arma::vec& sigma,
    const arma::vec& omega,
    const arma::vec& tau_b,
    const arma::vec& mu_alpha,
    double y_sq,
    double y_dot_eta_bar,
    double zeta,
    double tau_alpha,
    double tau_e,
    double c_pi,
    double d_pi
);

double compute_elbo_scalar(
    const arma::vec& mu,
    const arma::vec& sigma,
    const arma::vec& omega,
    const arma::vec& tau_b,
    const arma::vec& mu_alpha,
    double y_sq,
    double y_dot_eta_bar,
    double zeta,
    double tau_alpha,
    double tau_e,
    double pi_fixed
);

double sigmoid_cpp(const double &x);

#endif // COMMON_HELPERS_H
