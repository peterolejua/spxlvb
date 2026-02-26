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
    double Y2,
    double t_YW,
    double t_W2,
    double tau_alpha,
    double tau_e,
    double pi_fixed
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

  arma::vec term_a = omega % (arma::log(sigma) - l_omega);
  double sum_term_a = arma::accu(term_a);

  arma::vec term_b = one_minus_omega %
    (arma::log(arma::sqrt(tau_e * tau_b)) + l_omega_m1);
  double sum_term_b = -arma::accu(term_b);

  arma::vec inside = omega % (arma::square(mu) + sigma2) +
    one_minus_omega % (1.0 / (tau_e * tau_b));
  double slab_penalty = arma::accu(tau_b % inside);

  double resid_term = Y2 - 2.0 * t_YW + t_W2;
  double alpha_penalty = tau_alpha * arma::accu(arma::square(1.0 - mu_alpha));

  double data_fit = -0.5 * tau_e * resid_term;

  double slab_prior = 0.5 * arma::accu(arma::log(tau_e * tau_b))
                    - 0.5 * tau_e * slab_penalty;

  double alpha_prior = 0.5 * (p + 1) * std::log(tau_e * tau_alpha)
                     - 0.5 * tau_e * alpha_penalty;

  double logodds = std::log(pi_fixed / (1.0 - pi_fixed));
  double spike_prior = logodds * arma::accu(omega);

  double elbo = sum_term_a + sum_term_b +
                data_fit + slab_prior + alpha_prior + spike_prior;

  return Rcpp::List::create(
    Rcpp::Named("ELBO")           = elbo,
    Rcpp::Named("slab_entropy")   = sum_term_a,
    Rcpp::Named("spike_entropy")  = sum_term_b,
    Rcpp::Named("data_fit")       = data_fit,
    Rcpp::Named("slab_prior")     = slab_prior,
    Rcpp::Named("alpha_prior")    = alpha_prior,
    Rcpp::Named("spike_prior")    = spike_prior
  );
}

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

  double sum_term_a = arma::accu(omega % (arma::log(sigma) - l_omega));

  arma::vec log_tau_e_tau_b = arma::log(tau_e * tau_b);
  double sum_term_b = -arma::accu(one_minus_omega %
    (0.5 * log_tau_e_tau_b + l_omega_m1));

  arma::vec inside = omega % (arma::square(mu) + sigma2) +
    one_minus_omega % (1.0 / (tau_e * tau_b));
  double slab_penalty = arma::accu(tau_b % inside);

  double resid_term = Y2 - 2.0 * t_YW + t_W2;

  arma::vec alpha_diff = 1.0 - mu_alpha;
  double alpha_penalty = tau_alpha * arma::dot(alpha_diff, alpha_diff);

  double data_fit = -0.5 * tau_e * resid_term;

  double slab_prior = 0.5 * arma::accu(log_tau_e_tau_b)
                    - 0.5 * tau_e * slab_penalty;

  double alpha_prior = 0.5 * (p + 1) * std::log(tau_e * tau_alpha)
                     - 0.5 * tau_e * alpha_penalty;

  double logodds = std::log(pi_fixed / (1.0 - pi_fixed));
  double spike_prior = logodds * arma::accu(omega);

  return sum_term_a + sum_term_b +
         data_fit + slab_prior + alpha_prior + spike_prior;
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
) {
  VBUpdate2x2 res;

  res.L00 = tau_e * s_j * X_j_sq + tau_e * tau_b_j;
  res.L01 = tau_e * s_j * dot_Xj_Wj;
  res.L11 = tau_e * W_j_squared + tau_e * tau_alpha;

  double rhs0 = s_j * tau_e * YXj;
  double rhs1 = tau_e * dot_Y_Wj + tau_e * tau_alpha * mu_alpha_j;

  double det = res.L00 * res.L11 - res.L01 * res.L01;
  double eps = std::numeric_limits<double>::epsilon();
  if (det < eps) det = eps;
  double inv_det = 1.0 / det;

  res.eta0 = inv_det * (res.L11 * rhs0 - res.L01 * rhs1);
  res.eta1 = inv_det * (-res.L01 * rhs0 + res.L00 * rhs1);

  return res;
}
