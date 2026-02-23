#ifndef COMMON_HELPERS_H
#define COMMON_HELPERS_H

#include <RcppArmadillo.h>

struct VBUpdate2x2 {
  double L00, L01, L11;
  double eta0, eta1;
};

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

double compute_elbo_scalar(
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

double sigmoid_cpp(const double &x);

VBUpdate2x2 compute_vb_update_2x2(
    double X_j_sq,
    double dot_Xj_Wj,
    double YXj,
    double dot_Y_Wj,
    double W_j_squared,
    double s_j,
    double tau_e,
    double tau_b_j,
    double tau_alpha,
    double mu_alpha_j
);

#endif // COMMON_HELPERS_H
