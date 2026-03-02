#include "common_helpers.h"
#include <cmath>
#include <algorithm>

using namespace arma;

// [[Rcpp::export]]
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
) {
  int p = mu.n_elem;

  arma::vec sigma2 = arma::square(sigma);
  arma::vec one_minus_omega = 1.0 - omega;

  arma::vec l_omega = arma::log(omega);
  arma::vec l_omega_m1 = arma::log(1 - omega);

  for (int i = 0; i < p; ++i) {
    if (!R_finite(l_omega(i)))     l_omega(i)     = -500.0;
    if (!R_finite(l_omega_m1(i)))  l_omega_m1(i)  = -500.0;
  }

  arma::vec slab_entropy_terms = omega % (arma::log(sigma) - l_omega);
  double slab_entropy = arma::accu(slab_entropy_terms);

  arma::vec spike_entropy_terms = one_minus_omega %
    (arma::log(arma::sqrt(tau_e * tau_b)) + l_omega_m1);
  double spike_entropy = -arma::accu(spike_entropy_terms);

  arma::vec expected_b_sq = omega % (arma::square(mu) + sigma2) +
    one_minus_omega % (1.0 / (tau_e * tau_b));

  double residual_sq = y_sq - 2.0 * y_dot_eta_bar + zeta;

  double data_fit = -0.5 * tau_e * residual_sq;

  double slab_normalisation = 0.5 * arma::accu(arma::log(tau_e * tau_b));
  double slab_penalty = -0.5 * tau_e * arma::accu(tau_b % expected_b_sq);

  double alpha_normalisation = 0.5 * (p + 1) * std::log(tau_e * tau_alpha);
  double alpha_penalty = 0.5 * tau_e * tau_alpha * arma::accu(arma::square(1.0 - mu_alpha));

  double sum_omega = arma::accu(omega);
  double pi_posterior = R::lbeta(c_pi + sum_omega, d_pi + (p - sum_omega));
  double pi_normalisation = -R::lbeta(c_pi, d_pi);

  double elbo = slab_entropy + spike_entropy +
                data_fit + slab_normalisation + slab_penalty +
                alpha_normalisation - alpha_penalty +
                pi_posterior + pi_normalisation;

  return Rcpp::List::create(
    Rcpp::Named("ELBO")                  = elbo,
    Rcpp::Named("slab_entropy")          = slab_entropy,
    Rcpp::Named("spike_entropy")         = spike_entropy,
    Rcpp::Named("data_fit")              = data_fit,
    Rcpp::Named("slab_normalisation")    = slab_normalisation,
    Rcpp::Named("slab_penalty")          = slab_penalty,
    Rcpp::Named("alpha_normalisation")   = alpha_normalisation,
    Rcpp::Named("alpha_penalty")         = alpha_penalty,
    Rcpp::Named("pi_posterior")          = pi_posterior,
    Rcpp::Named("pi_normalisation")      = pi_normalisation
  );
}

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
) {
  int p = mu.n_elem;

  arma::vec sigma2 = arma::square(sigma);
  arma::vec one_minus_omega = 1.0 - omega;

  arma::vec l_omega = arma::log(omega);
  arma::vec l_omega_m1 = arma::log(one_minus_omega);

  for (int i = 0; i < p; ++i) {
    if (!R_finite(l_omega(i)))    l_omega(i)    = -500.0;
    if (!R_finite(l_omega_m1(i))) l_omega_m1(i) = -500.0;
  }

  double slab_entropy = arma::accu(omega % (arma::log(sigma) - l_omega));

  arma::vec log_slab_prec = arma::log(tau_e * tau_b);
  double spike_entropy = -arma::accu(one_minus_omega %
    (0.5 * log_slab_prec + l_omega_m1));

  arma::vec expected_b_sq = omega % (arma::square(mu) + sigma2) +
    one_minus_omega % (1.0 / (tau_e * tau_b));

  double residual_sq = y_sq - 2.0 * y_dot_eta_bar + zeta;

  double data_fit = -0.5 * tau_e * residual_sq;

  double slab_normalisation = 0.5 * arma::accu(log_slab_prec);
  double slab_penalty = -0.5 * tau_e * arma::accu(tau_b % expected_b_sq);

  double alpha_normalisation = 0.5 * (p + 1) * std::log(tau_e * tau_alpha);
  arma::vec alpha_diff = 1.0 - mu_alpha;
  double alpha_penalty = 0.5 * tau_e * tau_alpha * arma::dot(alpha_diff, alpha_diff);

  double logit_pi = std::log(pi_fixed / (1.0 - pi_fixed));
  double pi_posterior = logit_pi * arma::accu(omega);

  return slab_entropy + spike_entropy +
         data_fit + slab_normalisation + slab_penalty +
         alpha_normalisation - alpha_penalty + pi_posterior;
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
