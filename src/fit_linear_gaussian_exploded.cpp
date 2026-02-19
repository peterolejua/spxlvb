// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include <cmath>
#include <vector>

#include "common_helpers.h"

using namespace arma;

// [[Rcpp::export]]
Rcpp::List run_vb_updates_cpp(
    const arma::mat& X, // Design matrix
    const arma::vec& Y, // Response vector
    arma::vec mu, // estimated beta coefficient from lasso
    arma::vec omega, // Expectation that the coefficient from lasso is not zero
    double c_pi,    // Parameter of the Beta prior for pi
    double d_pi,    // Parameter of the Beta prior for pi
    double tau_e, // Precision of the errors
    const arma::uvec& update_order, // Order in which to update the coefficients
    arma::vec mu_alpha, // alpha_j is N(mu_alpha_j, (tau_e*tau_alpha)^{-1}), known/estimated
    double tau_alpha, // Precision of the Normal prior for the expansion parameters (alpha_j)
    arma::vec tau_b, // Precision of the prior for the coefficients (b_j)
    int max_iter, // Maximum number of iterations
    double tol, // Tolerance for convergence
    bool save_history = true // Whether to store per-iteration history
) {
  int n = X.n_rows;
  int p = X.n_cols;

  // Precomputations
  arma::vec YX_vec = X.t() * Y;
  arma::mat X_2 = square(X);
  arma::vec X_2_col_sums = arma::sum(X_2, 0).t();
  double Y2 = arma::dot(Y, Y);

  // Posterior sigma initialization
  arma::vec sigma = 1.0 / arma::sqrt(tau_e * (X_2_col_sums + tau_b));
  arma::vec alpha_j_optimal(p + 1, fill::ones);
  arma::vec logit_phi = arma::log(omega / (1.0 - omega));

  arma::vec mu_tilde = mu;

  arma::vec sigma_tilde = sigma;
  arma::vec tau_b_tilde = tau_b;
  arma::vec mu_alpha_tilde = mu_alpha;
  // Trackers for convergence and "exploded" logic
  arma::vec omega_old = omega;
  arma::vec W = X * (omega % mu);
  arma::vec var_W_vec = X_2 * (arma::square(mu) % omega % (1.0 - omega) + arma::square(sigma) % omega);

  arma::vec alpha_prod(p, fill::ones);

  double var_W = arma::accu(var_W_vec);

  // Initial prior expectation for pi
  double pi_fixed = c_pi / (c_pi + d_pi);
  double E_logit_pi = std::log(pi_fixed) - std::log(1-pi_fixed);

  // History Storage — dynamic vectors, pushed per iteration
  std::vector<arma::vec> mu_hist, omega_hist, sigma_hist;
  std::vector<arma::vec> tau_b_hist, mu_alpha_hist, alpha_hist;
  std::vector<double> conv_hist, elbo_hist;

  bool converged = false;
  int last_iter = 0;
  for (int iter = 0; iter < max_iter; ++iter) {

    arma::vec new_entr(p, fill::zeros);
    alpha_prod.ones();
    arma::vec W_old = W;

    arma::vec mu_old, sigma_old, tau_b_old, mu_alpha_old;

    for (int k = 0; k < p; ++k) {
      Rcpp::checkUserInterrupt();
      mu_old = mu;
      sigma_old = sigma;
      tau_b_old = tau_b;
      mu_alpha_old = mu_alpha;
      omega_old = omega;
      arma::vec logit_phi_old = logit_phi;

      unsigned int j = update_order(k);  // 0-based index

      arma::vec W_j = W - (omega(j) * mu(j)) * X.col(j);
      double coeff = (mu(j) * mu(j) * omega(j) * (1.0 - omega(j)) + sigma(j) * sigma(j) * omega(j));
      double var_W_j = var_W - X_2_col_sums(j) * coeff;
      arma::vec var_W_vec_j = var_W_vec - X_2.col(j) * coeff;
      double W_j_squared = var_W_j + arma::dot(W_j, W_j);


      Rcpp::List res0 = calculate_lambda_eta_sigma_update_cpp(
        X,
        X_2_col_sums,
        YX_vec,
        Y,
        W_j,
        W_j_squared,
        0.0,
        j,
        tau_e,
        tau_b,
        tau_alpha,
        mu_alpha
      );
      arma::mat Lambda_j_0 = res0["Lambda_j"];
      arma::vec eta_j_0    = res0["eta_j"];

      Rcpp::List res1 = calculate_lambda_eta_sigma_update_cpp(
        X,
        X_2_col_sums,
        YX_vec,
        Y,
        W_j,
        W_j_squared,
        1.0,
        j,
        tau_e,
        tau_b,
        tau_alpha,
        mu_alpha
      );
      arma::mat Lambda_j_1 = res1["Lambda_j"];
      arma::vec eta_j_1    = res1["eta_j"];

      double det0 = Lambda_j_0(0,0) * Lambda_j_0(1,1) - std::pow(Lambda_j_0(0,1), 2.0);
      double det1 = Lambda_j_1(0,0) * Lambda_j_1(1,1) - std::pow(Lambda_j_1(0,1), 2.0);

      double eps = std::numeric_limits<double>::epsilon();
      det0 = std::max(det0, eps);
      det1 = std::max(det1, eps);

      double logit_phi_j =
        E_logit_pi +
        0.5 * std::log(det0 / det1) -
        0.5 * std::pow(eta_j_0(1), 2.0) * Lambda_j_0(1,1) +
        0.5 * std::pow(eta_j_1(0), 2.0) * Lambda_j_1(0,0) +
        eta_j_1(0) * eta_j_1(1) * Lambda_j_1(0,1) +
        0.5 * std::pow(eta_j_1(1), 2.0) * Lambda_j_1(1,1);

      logit_phi(j) = logit_phi_j;

      mu(j) = eta_j_1(0) - Lambda_j_1(0,1) / Lambda_j_1(0,0) * (1.0 - eta_j_1(1));
      sigma(j) = std::sqrt(1.0 / Lambda_j_1(0,0));

      double L_0 = Lambda_j_0(1,1) - std::pow(Lambda_j_0(0,1), 2.0) / Lambda_j_0(0,0);
      double L_1 = Lambda_j_1(1,1) - std::pow(Lambda_j_1(0,1), 2.0) / Lambda_j_1(0,0);

      double g_0 = -0.5 * L_0 * std::pow(1.0 - eta_j_0(1), 2.0);
      double g_1 = -0.5 * L_1 * std::pow(1.0 - eta_j_1(1), 2.0);

      omega(j) = sigmoid_cpp(logit_phi_j + (g_1 - g_0));

      double R_1 = std::pow(Lambda_j_1(0,1), 2.0) / Lambda_j_1(0,0);

      double sigma2_alpha_0 = 1.0 / L_0;
      double sigma2_alpha_1 = 1.0 / (L_1 + R_1);

      double mu_alpha_0 = eta_j_0(1);
      double mu_alpha_1 = (L_1 * eta_j_1(1) + R_1) * sigma2_alpha_1;

      double D_j = (omega(j) * sigma2_alpha_0) / (omega(j) * sigma2_alpha_0 + (1.0 - omega(j)) * sigma2_alpha_1);

      double optimal_alpha_j = D_j * mu_alpha_1 + (1.0 - D_j) * mu_alpha_0;
      alpha_j_optimal(j) = optimal_alpha_j;

      mu_tilde        = mu * optimal_alpha_j;
      sigma_tilde     = sigma * std::fabs(optimal_alpha_j);
      tau_b_tilde     = tau_b / (optimal_alpha_j * optimal_alpha_j);
      mu_alpha_tilde  = mu_alpha;

      mu_tilde(j)       = mu(j);
      sigma_tilde(j)    = sigma(j);
      tau_b_tilde(j)    = tau_b(j);
      mu_alpha_tilde(j) = 1.0 - (optimal_alpha_j - mu_alpha(j));

      mu       = mu_tilde;
      sigma    = sigma_tilde;
      tau_b    = tau_b_tilde;
      mu_alpha = mu_alpha_tilde;

      W       = optimal_alpha_j * W_j + (omega(j) * mu(j)) * X.col(j);
      coeff   = (mu(j) * mu(j) * omega(j) * (1.0 - omega(j)) + sigma(j) * sigma(j) * omega(j));
      var_W   = optimal_alpha_j * optimal_alpha_j * var_W_j + X_2_col_sums(j) * coeff;
      var_W_vec = optimal_alpha_j * optimal_alpha_j * var_W_vec_j + X_2.col(j) * coeff;
    }

    double t_YW = arma::dot(W, Y);
    double t_W2 = var_W + arma::dot(W, W);

    int idx_p1 = p;
    double optimal_alpha_p1 = (t_YW + tau_alpha * mu_alpha(idx_p1)) / (t_W2 + tau_alpha);
    alpha_j_optimal(idx_p1) = optimal_alpha_p1;

    mu        = optimal_alpha_p1 * mu;
    sigma     = std::fabs(optimal_alpha_p1) * sigma;
    tau_b     = tau_b / (optimal_alpha_p1 * optimal_alpha_p1);
    mu_alpha(idx_p1) = 1.0 - (optimal_alpha_p1 - mu_alpha(idx_p1));

    W         = optimal_alpha_p1 * W;
    var_W     = optimal_alpha_p1 * optimal_alpha_p1 * var_W;
    var_W_vec = optimal_alpha_p1 * optimal_alpha_p1 * var_W_vec;

    t_YW = arma::dot(W, Y);
    t_W2 = var_W + arma::dot(W, W);


    double convg2 = 1.0;

    if (iter > 0) {

      arma::vec tmp = (W_old - W);
      arma::vec stat_vec = arma::square(tmp) / var_W_vec;
      double max_stat = stat_vec.max() / std::log((double)n);
      convg2 = R::pchisq(max_stat, 1.0, 1, 0);  // lower.tail=TRUE, log.p=FALSE

    }

    // Compute ELBO using current parameters
    Rcpp::List elbo_res = compute_elbo_cpp(
      mu,
      sigma,
      omega,
      tau_b,
      mu_alpha,
      Y2,
      t_YW,
      t_W2,
      tau_alpha,
      tau_e,
      pi_fixed
    );
    double current_elbo = elbo_res["ELBO"];

    // Save history — push only what's needed
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

    // --- Convergence Checks --
    last_iter = iter;
    if (convg2 < tol) {
      converged = true;
      break;
    }
  }

  // Convert scalar histories to arma::vec
  arma::vec conv_vec(conv_hist);
  arma::vec elbo_vec(elbo_hist);

  // Build the result list
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

  // Append per-iteration history matrices only if requested
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
