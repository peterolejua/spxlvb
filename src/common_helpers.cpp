// common_helpers.cpp
#include "common_helpers.h" // Include its own header first
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <cmath>
#include <algorithm> // for std::max, std::min

// -----------------------------------------------------------------------------
// compute_elbo: Numerically Robust ELBO Calculation
// -----------------------------------------------------------------------------
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
) {
  int p = mu.n_elem;

  // double pi = arma::datum::pi;
  // double log2pi = std::log(2.0 * pi);
  arma::vec sigma2 = arma::square(sigma);
  arma::vec one_minus_omega = 1.0 - omega;

  arma::vec l_omega = arma::log(omega);
  arma::vec l_omega_m1 = arma::log(1 - omega);

  for (int i = 0; i < p; ++i) {
    if (!R_finite(l_omega(i)))     l_omega(i)     = -500.0;
    if (!R_finite(l_omega_m1(i)))  l_omega_m1(i)  = -500.0;
  }

  // arma::vec term_a = omega % (0.5 + 0.5*log2pi + arma::log(sigma) - l_omega);
  arma::vec term_a = omega % (arma::log(sigma) - l_omega);
  double sum_term_a = arma::accu(term_a);

  arma::vec term_b = one_minus_omega %
    (arma::log(arma::sqrt(tau_e * tau_b)) + l_omega_m1); //
  double sum_term_b = -arma::accu(term_b);

  arma::vec inside = omega % (arma::square(mu) + sigma2) +
    one_minus_omega % (1.0 / (tau_e * tau_b));
  arma::vec taub_times_inside = tau_b % inside;
  double sum_taub_inside = arma::accu(taub_times_inside);

  // arma::vec one_minus_mu_alpha = 1.0 - mu_alpha;
  // double sum_alpha_term = tau_alpha * arma::accu(arma::square(one_minus_mu_alpha));

  double resid_term = Y2 - 2.0 * t_YW + t_W2;

  double bigurly = resid_term + sum_taub_inside; // + sum_alpha_term;
  double datafit_term = -0.5 * tau_e * bigurly;

  double term_norm = 0.5 * (arma::accu(arma::log(tau_e * tau_b))); // +
  // (p + 1) * std::log(tau_e * tau_alpha));

  double logodds = std::log(pi_fixed / (1.0 - pi_fixed));
  double pi_term = logodds * arma::accu(omega);

  double elbo = sum_term_a + sum_term_b + datafit_term + term_norm + pi_term;

  double sum_taua = (p + 1) * std::log(tau_e * tau_alpha) / 2.0;
  double sum_taub = arma::accu(arma::log(tau_e * tau_b)) / 2.0;
  double SSE = tau_e * (Y2 - 2.0 * t_YW + t_W2);

  return Rcpp::List::create(
    Named("ELBO")      = elbo,
    Named("Sum_a")     = sum_term_a,
    Named("Sum_b")     = sum_term_b,
    Named("Datafit")   = datafit_term,
    Named("Resid_term")= resid_term,
    Named("sum_taua")  = sum_taua,
    Named("sum_taub")  = sum_taub,
    Named("SSE")       = SSE,
    Named("term_norm") = term_norm,
    Named("pi_term")   = pi_term
  );
}

double sigmoid_cpp(const double &x) {
  if (x > 32.0) {
    return 1;
  } else if (x < -32.0) {
    return 0;
  } else {
    return 1 / (1 + std::exp(-x));
  }
}

// Calculates Lambda_j, Sigma_j, and eta_j
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
) {
  double s_j = s_j_val;
  arma::vec X_j = X.col(j);

  arma::mat Lambda_j(2,2, fill::zeros);

  double X_j_sq_val = X_2_col_sums(j);
  double dot_Xj_Wj = arma::dot(X_j, W_j);

  Lambda_j(0,0) = tau_e * s_j * X_j_sq_val + tau_e * tau_b(j);
  Lambda_j(0,1) = tau_e * s_j * dot_Xj_Wj;
  Lambda_j(1,0) = tau_e * s_j * dot_Xj_Wj;
  Lambda_j(1,1) = tau_e * W_j_squared + tau_e * tau_alpha;

  arma::mat Sigma_j(2,2);
  bool ok = arma::inv(Sigma_j, Lambda_j);
  if (!ok) {
    arma::mat Lambda_j_pert = Lambda_j +
      std::numeric_limits<double>::epsilon() * arma::eye<arma::mat>(2,2);
    arma::inv(Sigma_j, Lambda_j_pert);
  }

  arma::vec eta(2, fill::zeros);
  eta(0) = s_j * tau_e * YX_vec(j);
  eta(1) = tau_e * arma::dot(Y, W_j) + tau_e * tau_alpha * mu_alpha(j);

  arma::vec eta_out = Sigma_j * eta;

  return Rcpp::List::create(
    Named("Lambda_j") = Lambda_j,
    Named("Sigma_j")  = Sigma_j,
    Named("eta_j")    = eta_out
  );
}

