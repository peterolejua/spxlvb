// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include <cmath>
#include <vector>

#include "common_helpers.h"

using namespace arma;

// [[Rcpp::export]]
Rcpp::List run_vb_updates_cpp(
    const arma::mat& X,
    const arma::vec& Y,
    arma::vec mu,
    arma::vec omega,
    double c_pi,
    double d_pi,
    double tau_e,
    const arma::uvec& update_order,
    arma::vec mu_alpha,
    double tau_alpha,
    arma::vec tau_b,
    int max_iter,
    double tol,
    bool save_history = true
) {
  int n = X.n_rows;
  int p = X.n_cols;

  arma::vec YX_vec = X.t() * Y;
  arma::mat X_2 = square(X);
  arma::vec X_2_col_sums = arma::sum(X_2, 0).t();
  double Y2 = arma::dot(Y, Y);

  arma::vec sigma = 1.0 / arma::sqrt(tau_e * (X_2_col_sums + tau_b));
  arma::vec alpha_j_optimal(p + 1, fill::ones);
  arma::vec logit_phi = arma::log(omega / (1.0 - omega));

  arma::vec W = X * (omega % mu);
  arma::vec var_W_vec = X_2 * (arma::square(mu) % omega % (1.0 - omega) +
                                arma::square(sigma) % omega);
  double var_W = arma::accu(var_W_vec);

  double pi_fixed = c_pi / (c_pi + d_pi);
  double E_logit_pi = std::log(pi_fixed) - std::log(1 - pi_fixed);

  std::vector<arma::vec> mu_hist, omega_hist, sigma_hist;
  std::vector<arma::vec> tau_b_hist, mu_alpha_hist, alpha_hist;
  std::vector<double> conv_hist, elbo_hist;

  if (save_history) {
    mu_hist.reserve(max_iter);
    omega_hist.reserve(max_iter);
    sigma_hist.reserve(max_iter);
    tau_b_hist.reserve(max_iter);
    mu_alpha_hist.reserve(max_iter);
    alpha_hist.reserve(max_iter);
  }
  conv_hist.reserve(max_iter);
  elbo_hist.reserve(max_iter);

  bool converged = false;
  int last_iter = 0;
  for (int iter = 0; iter < max_iter; ++iter) {
    Rcpp::checkUserInterrupt();

    arma::vec W_old = W;

    for (int k = 0; k < p; ++k) {
      unsigned int j = update_order(k);

      arma::vec W_j = W - (omega(j) * mu(j)) * X.col(j);
      double coeff = mu(j) * mu(j) * omega(j) * (1.0 - omega(j)) +
                     sigma(j) * sigma(j) * omega(j);
      double var_W_j = var_W - X_2_col_sums(j) * coeff;
      double W_j_squared = var_W_j + arma::dot(W_j, W_j);

      double dot_Xj_Wj = arma::dot(X.col(j), W_j);
      double dot_Y_Wj  = arma::dot(Y, W_j);

      VBUpdate2x2 u0 = compute_vb_update_2x2(
        X_2_col_sums(j), dot_Xj_Wj, YX_vec(j), dot_Y_Wj,
        W_j_squared, 0.0, tau_e, tau_b(j), tau_alpha, mu_alpha(j)
      );

      VBUpdate2x2 u1 = compute_vb_update_2x2(
        X_2_col_sums(j), dot_Xj_Wj, YX_vec(j), dot_Y_Wj,
        W_j_squared, 1.0, tau_e, tau_b(j), tau_alpha, mu_alpha(j)
      );

      double det0 = u0.L00 * u0.L11 - u0.L01 * u0.L01;
      double det1 = u1.L00 * u1.L11 - u1.L01 * u1.L01;

      double eps = std::numeric_limits<double>::epsilon();
      det0 = std::max(det0, eps);
      det1 = std::max(det1, eps);

      double logit_phi_j =
        E_logit_pi +
        0.5 * std::log(det0 / det1) -
        0.5 * u0.eta1 * u0.eta1 * u0.L11 +
        0.5 * u1.eta0 * u1.eta0 * u1.L00 +
        u1.eta0 * u1.eta1 * u1.L01 +
        0.5 * u1.eta1 * u1.eta1 * u1.L11;

      logit_phi(j) = logit_phi_j;

      mu(j) = u1.eta0 - u1.L01 / u1.L00 * (1.0 - u1.eta1);
      sigma(j) = std::sqrt(1.0 / u1.L00);

      double L_0 = u0.L11 - u0.L01 * u0.L01 / u0.L00;
      double L_1 = u1.L11 - u1.L01 * u1.L01 / u1.L00;

      double diff0 = 1.0 - u0.eta1;
      double diff1 = 1.0 - u1.eta1;
      double g_0 = -0.5 * L_0 * diff0 * diff0;
      double g_1 = -0.5 * L_1 * diff1 * diff1;

      omega(j) = sigmoid_cpp(logit_phi_j + (g_1 - g_0));

      double R_1 = u1.L01 * u1.L01 / u1.L00;

      double sigma2_alpha_0 = 1.0 / L_0;
      double sigma2_alpha_1 = 1.0 / (L_1 + R_1);

      double mu_alpha_0 = u0.eta1;
      double mu_alpha_1 = (L_1 * u1.eta1 + R_1) * sigma2_alpha_1;

      double D_j = (omega(j) * sigma2_alpha_0) /
                   (omega(j) * sigma2_alpha_0 +
                    (1.0 - omega(j)) * sigma2_alpha_1);

      double optimal_alpha_j = D_j * mu_alpha_1 + (1.0 - D_j) * mu_alpha_0;
      alpha_j_optimal(j) = optimal_alpha_j;

      double mu_j_saved = mu(j);
      double sigma_j_saved = sigma(j);
      double tau_b_j_saved = tau_b(j);

      mu *= optimal_alpha_j;
      sigma *= std::fabs(optimal_alpha_j);
      tau_b /= (optimal_alpha_j * optimal_alpha_j);

      mu(j)       = mu_j_saved;
      sigma(j)    = sigma_j_saved;
      tau_b(j)    = tau_b_j_saved;
      mu_alpha(j) = 1.0 - (optimal_alpha_j - mu_alpha(j));

      W       = optimal_alpha_j * W_j + (omega(j) * mu(j)) * X.col(j);
      coeff   = mu(j) * mu(j) * omega(j) * (1.0 - omega(j)) +
                sigma(j) * sigma(j) * omega(j);
      var_W   = optimal_alpha_j * optimal_alpha_j * var_W_j +
                X_2_col_sums(j) * coeff;
    }

    double t_YW = arma::dot(W, Y);
    double t_W2 = var_W + arma::dot(W, W);

    int idx_p1 = p;
    double optimal_alpha_p1 = (t_YW + tau_alpha * mu_alpha(idx_p1)) /
                              (t_W2 + tau_alpha);
    alpha_j_optimal(idx_p1) = optimal_alpha_p1;

    mu        *= optimal_alpha_p1;
    sigma     *= std::fabs(optimal_alpha_p1);
    tau_b     /= (optimal_alpha_p1 * optimal_alpha_p1);
    mu_alpha(idx_p1) = 1.0 - (optimal_alpha_p1 - mu_alpha(idx_p1));

    W         *= optimal_alpha_p1;
    var_W     *= optimal_alpha_p1 * optimal_alpha_p1;

    t_YW = arma::dot(W, Y);
    t_W2 = var_W + arma::dot(W, W);

    double convg2 = 1.0;

    if (iter > 0) {
      var_W_vec = X_2 * (arma::square(mu) % omega % (1.0 - omega) +
                          arma::square(sigma) % omega);
      var_W = arma::accu(var_W_vec);
      arma::vec tmp = W_old - W;
      arma::vec stat_vec = arma::square(tmp) / var_W_vec;
      double max_stat = stat_vec.max() / std::log((double)n);
      convg2 = R::pchisq(max_stat, 1.0, 1, 0);
    }

    Rcpp::List elbo_res = compute_elbo_cpp(
      mu, sigma, omega, tau_b, mu_alpha,
      Y2, t_YW, t_W2, tau_alpha, tau_e, pi_fixed
    );
    double current_elbo = elbo_res["ELBO"];

    if (save_history) {
      mu_hist.push_back(mu);
      omega_hist.push_back(omega);
      sigma_hist.push_back(sigma);
      tau_b_hist.push_back(tau_b);
      mu_alpha_hist.push_back(mu_alpha);
      alpha_hist.push_back(alpha_j_optimal);
    }
    conv_hist.push_back(convg2);
    elbo_hist.push_back(current_elbo);

    last_iter = iter;
    if (convg2 < tol) {
      converged = true;
      break;
    }
  }

  arma::vec conv_vec(conv_hist);
  arma::vec elbo_vec(elbo_hist);

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("converged") = converged,
    Rcpp::Named("iterations") = last_iter + 1,
    Rcpp::Named("convergence_criterion") = conv_hist.back(),
    Rcpp::Named("convergence_history") = conv_vec,
    Rcpp::Named("elbo_history") = elbo_vec,
    Rcpp::Named("mu") = mu,
    Rcpp::Named("omega") = omega,
    Rcpp::Named("sigma") = sigma,
    Rcpp::Named("tau_b") = tau_b,
    Rcpp::Named("mu_alpha") = mu_alpha
  );

  if (save_history) {
    int niters = static_cast<int>(mu_hist.size());
    arma::mat m_mu(p, niters), m_omega(p, niters), m_sigma(p, niters);
    arma::mat m_tau_b(p, niters);
    arma::mat m_mu_alpha(p + 1, niters), m_alpha(p + 1, niters);
    for (int i = 0; i < niters; ++i) {
      m_mu.col(i)       = mu_hist[i];
      m_omega.col(i)    = omega_hist[i];
      m_sigma.col(i)    = sigma_hist[i];
      m_tau_b.col(i)    = tau_b_hist[i];
      m_mu_alpha.col(i) = mu_alpha_hist[i];
      m_alpha.col(i)    = alpha_hist[i];
    }
    result["mu_history"]       = m_mu;
    result["omega_history"]    = m_omega;
    result["sigma_history"]    = m_sigma;
    result["tau_b_history"]    = m_tau_b;
    result["mu_alpha_history"] = m_mu_alpha;
    result["alpha_history"]    = m_alpha;
  }

  return result;
}
