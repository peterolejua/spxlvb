// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include <cmath>
#include <vector>

#include "common_helpers.h"

using namespace arma;

// [[Rcpp::export]]
Rcpp::List run_vb_path_c_cpp(
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
    bool save_history = true,
    int convergence_method = 0,
    bool update_pi = false,
    double elbo_offset = 0.0,
    bool disable_global_alpha = false,
    bool track_all_criteria = false,
    bool use_gamma_hyperprior_tau_alpha = false,
    double r_alpha = 0.0,
    double d_alpha = 0.0,
    bool use_gamma_hyperprior_tau_b = false,
    double r_b = 0.0,
    double d_b = 0.0,
    bool simultaneous_remap = false,
    bool use_annealing = false,
    int anneal_iterations = 50
) {
  int n = X.n_rows;
  int p = X.n_cols;

  arma::vec Xty = X.t() * Y;
  double y_sq = arma::dot(Y, Y);

  arma::mat X_sq;
  arma::vec X_col_sq(p);
  if (convergence_method == 1 || track_all_criteria) {
    X_sq = arma::square(X);
    X_col_sq = arma::sum(X_sq, 0).t();
  } else {
    for (int j = 0; j < p; ++j) {
      X_col_sq(j) = arma::dot(X.col(j), X.col(j));
    }
  }

  arma::vec sigma = 1.0 / arma::sqrt(tau_e * (X_col_sq + tau_b));
  arma::vec alpha_hat(p + 1, fill::ones);
  arma::vec logit_omega = arma::log(omega / (1.0 - omega));

  arma::vec eta_bar = X * (omega % mu);
  double var_eta;
  {
    arma::vec d_var = arma::square(mu) % omega % (1.0 - omega) +
                  arma::square(sigma) % omega;
    var_eta = (convergence_method == 1 || track_all_criteria)
      ? arma::accu(X_sq * d_var)
      : arma::dot(X_col_sq, d_var);
  }
  double zeta = var_eta + arma::dot(eta_bar, eta_bar);
  double y_dot_eta_bar = arma::dot(Y, eta_bar);

  double pi_fixed = c_pi / (c_pi + d_pi);
  double E_logit_pi = update_pi
    ? R::digamma(c_pi) - R::digamma(d_pi)
    : std::log(pi_fixed) - std::log(1.0 - pi_fixed);

  std::vector<arma::vec> mu_hist, omega_hist, sigma_hist;
  std::vector<arma::vec> tau_b_hist, mu_alpha_hist, alpha_hist;
  std::vector<double> conv_hist, elbo_hist, alpha_stripped_elbo_hist;
  std::vector<double> chisq_hist, entropy_change_hist;

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
  alpha_stripped_elbo_hist.reserve(max_iter);
  if (track_all_criteria) {
    chisq_hist.reserve(max_iter);
    entropy_change_hist.reserve(max_iter);
  }

  arma::vec sigma_sq_alpha(p + 1, fill::zeros);
  double r_alpha_post = 0.0;
  double d_alpha_post = 0.0;
  std::vector<double> tau_alpha_hist;
  if (use_gamma_hyperprior_tau_alpha) {
    r_alpha_post = r_alpha + (p + 1.0) / 2.0;
    d_alpha_post = d_alpha;
    tau_alpha_hist.reserve(max_iter);
  }

  double tau_b_common = 0.0;
  std::vector<double> tau_b_common_hist;
  if (use_gamma_hyperprior_tau_b) {
    tau_b_common = tau_b(0);
    tau_b_common_hist.reserve(max_iter);
  }

  bool converged = false;
  int last_iter = 0;
  double prev_elbo = R_NegInf;
  arma::vec omega_old_entropy = omega;
  arma::vec eta_bar_prev((convergence_method == 1 || track_all_criteria) ? n : 0);

  for (int iter = 0; iter < max_iter; ++iter) {
    Rcpp::checkUserInterrupt();

    // Annealing: beta ramps from ~0 to 1 over anneal_iterations
    // beta scales the likelihood contribution to logit(omega)
    double beta = 1.0;
    if (use_annealing && iter < anneal_iterations) {
      beta = static_cast<double>(iter + 1) / anneal_iterations;
    }

    if (convergence_method == 1 || track_all_criteria) eta_bar_prev = eta_bar;

    // ================================================================
    // PHASE 1: Standard CAVI sweep (no alpha involvement)
    // Carbonetto & Stephens (2012) updates: sigma, mu, omega
    // With annealing: likelihood terms in logit(omega) scaled by beta
    // ================================================================
    for (int k = 0; k < p; ++k) {
      unsigned int j = update_order(k);

      double eta_bar_contrib_old = omega(j) * mu(j);
      double Xj_eta_bar_mj = arma::dot(X.col(j), eta_bar)
                              - eta_bar_contrib_old * X_col_sq(j);
      double second_moment_j = omega(j)
                                * (sigma(j) * sigma(j) + mu(j) * mu(j));
      double zeta_mj = zeta - X_col_sq(j) * second_moment_j
                       - 2.0 * eta_bar_contrib_old * Xj_eta_bar_mj;
      double y_dot_eta_bar_mj = y_dot_eta_bar
                                 - eta_bar_contrib_old * Xty(j);

      double denom_j = X_col_sq(j) + tau_b(j);
      double residual_j = Xty(j) - Xj_eta_bar_mj;

      sigma(j) = 1.0 / std::sqrt(tau_e * denom_j);
      mu(j) = residual_j / denom_j;

      // Annealing: scale the likelihood Bayes factor by beta
      // logit(omega) = log(pi/(1-pi)) + beta * [slab normalisation + data fit]
      double logit_omega_j = E_logit_pi
        + beta * (0.5 * std::log(tau_b(j) / denom_j)
                  + 0.5 * tau_e * residual_j * residual_j / denom_j);

      logit_omega(j) = logit_omega_j;
      omega(j) = sigmoid_cpp(logit_omega_j);

      double eta_bar_contrib_new = omega(j) * mu(j);
      eta_bar += (eta_bar_contrib_new - eta_bar_contrib_old) * X.col(j);
      double new_second_moment = omega(j)
                                  * (sigma(j) * sigma(j) + mu(j) * mu(j));
      zeta = zeta_mj + X_col_sq(j) * new_second_moment
             + 2.0 * eta_bar_contrib_new * Xj_eta_bar_mj;
      y_dot_eta_bar = y_dot_eta_bar_mj + eta_bar_contrib_new * Xty(j);
    }

    // ================================================================
    // PHASE 2 + 3: Alpha optimization and remap
    // ================================================================
    if (simultaneous_remap) {
      // --- Joint optimization: solve p×p system for optimal alpha ---
      //
      // The ELBO as a function of alpha (holding q(b,s) fixed) is quadratic:
      //   f(alpha) = tau_e * y' X(alpha ⊙ v) - tau_e/2 ||X(alpha ⊙ v)||^2
      //              - tau_e/2 sum_j alpha_j^2 omega_j sigma_j^2 ||X_j||^2
      //              - tau_alpha/2 ||alpha - mu_alpha||^2
      //
      // where v_j = omega_j * mu_j. Setting gradient to zero gives:
      //   [tau_e * V X'X V + tau_e * D + tau_alpha * I] alpha
      //       = tau_e * V X'y + tau_alpha * mu_alpha
      //
      // with V = diag(v), D_jj = omega_j * sigma_j^2 * ||X_j||^2.

      arma::vec v = omega % mu;  // p-vector
      arma::vec diag_d = omega % arma::square(sigma) % X_col_sq;  // p-vector

      // Form X_scaled = X * diag(v), then compute X_scaled' * X_scaled
      // This is the VX'XV term. Cost: O(np) for scaling, O(np^2) for crossproduct.
      arma::mat X_scaled = X.each_row() % v.t();  // n x p
      arma::mat system_matrix = tau_e * (X_scaled.t() * X_scaled);  // p x p

      // Add diagonal terms: tau_e * D + tau_alpha * I
      system_matrix.diag() += tau_e * diag_d + tau_alpha;

      // RHS: tau_e * V * X'y + tau_alpha * mu_alpha(0:p-1)
      arma::vec rhs = tau_e * (v % Xty) + tau_alpha * mu_alpha.head(p);

      // Solve symmetric positive definite system via Cholesky
      arma::vec alpha_joint = arma::solve(system_matrix, rhs,
                                          arma::solve_opts::likely_sympd);

      // Store alpha_hat for history
      for (int j = 0; j < p; ++j) {
        alpha_hat(j) = alpha_joint(j);
      }

      // Remap: mu_k *= alpha_k, sigma_k *= |alpha_k|, tau_b_k /= alpha_k^2
      for (int j = 0; j < p; ++j) {
        mu(j) *= alpha_joint(j);
        sigma(j) *= std::fabs(alpha_joint(j));
        tau_b(j) /= (alpha_joint(j) * alpha_joint(j));
        mu_alpha(j) = 1.0 - (alpha_joint(j) - mu_alpha(j));
      }

      if (use_gamma_hyperprior_tau_alpha) {
        // Posterior variance of alpha_j from the joint system
        // sigma_sq_alpha = diag(system_matrix^{-1}) / tau_e
        // But inverting is expensive; use diagonal approximation instead
        for (int j = 0; j < p; ++j) {
          sigma_sq_alpha(j) = 1.0 / (tau_e * system_matrix(j, j));
        }
      }

      // Recompute eta_bar, zeta, y_dot_eta_bar from scratch
      eta_bar = X * (omega % mu);
      {
        arma::vec d_var = arma::square(mu) % omega % (1.0 - omega) +
                      arma::square(sigma) % omega;
        var_eta = (convergence_method == 1 || track_all_criteria)
          ? arma::accu(X_sq * d_var)
          : arma::dot(X_col_sq, d_var);
      }
      zeta = var_eta + arma::dot(eta_bar, eta_bar);
      y_dot_eta_bar = arma::dot(Y, eta_bar);

    } else {
      // --- Sequential: compute alpha_hat(j), remap, one at a time ---

      double A_running = 1.0;
      arma::vec A_snap(p, fill::ones);

      for (int k = 0; k < p; ++k) {
        unsigned int j = update_order(k);

        double scale_j = A_running / A_snap(j);
        double true_mu_j = mu(j) * scale_j;
        double true_sigma_j = sigma(j) * std::fabs(scale_j);
        double true_tau_b_j = tau_b(j) / (scale_j * scale_j);

        double eta_bar_contrib_j = omega(j) * true_mu_j;
        double Xj_eta_bar_mj = arma::dot(X.col(j), eta_bar)
                                - eta_bar_contrib_j * X_col_sq(j);
        double second_moment_j = omega(j)
          * (true_sigma_j * true_sigma_j + true_mu_j * true_mu_j);
        double zeta_mj = zeta - X_col_sq(j) * second_moment_j
                         - 2.0 * eta_bar_contrib_j * Xj_eta_bar_mj;
        double y_dot_eta_bar_mj = y_dot_eta_bar
                                   - eta_bar_contrib_j * Xty(j);

        double optimal_alpha_j = (y_dot_eta_bar_mj
                                   - eta_bar_contrib_j * Xj_eta_bar_mj
                                   + tau_alpha * mu_alpha(j))
                                  / (zeta_mj + tau_alpha);

        alpha_hat(j) = optimal_alpha_j;

        if (use_gamma_hyperprior_tau_alpha) {
          sigma_sq_alpha(j) = 1.0 / (tau_e * (zeta_mj + tau_alpha));
        }

        mu(j) = true_mu_j;
        sigma(j) = true_sigma_j;
        tau_b(j) = true_tau_b_j;
        A_running *= optimal_alpha_j;
        A_snap(j) = A_running;

        mu_alpha(j) = 1.0 - (optimal_alpha_j - mu_alpha(j));

        // Update eta_bar, zeta, y_dot_eta_bar for the remap
        eta_bar *= optimal_alpha_j;
        eta_bar += (1.0 - optimal_alpha_j) * eta_bar_contrib_j * X.col(j);

        zeta = optimal_alpha_j * optimal_alpha_j * zeta_mj
               + X_col_sq(j) * second_moment_j
               + 2.0 * optimal_alpha_j * eta_bar_contrib_j * Xj_eta_bar_mj;

        y_dot_eta_bar = optimal_alpha_j * y_dot_eta_bar_mj
                         + eta_bar_contrib_j * Xty(j);
      }

      // Flush deferred remap
      for (int j_flush = 0; j_flush < p; ++j_flush) {
        double sc = A_running / A_snap(j_flush);
        mu(j_flush) *= sc;
        sigma(j_flush) *= std::fabs(sc);
        tau_b(j_flush) /= (sc * sc);
      }
    }

    // ================================================================
    // PHASE 4: Global alpha_{p+1}
    // ================================================================
    int idx_p1 = p;
    if (!disable_global_alpha) {
      double optimal_alpha_p1 = (y_dot_eta_bar
                                  + tau_alpha * mu_alpha(idx_p1))
                                 / (zeta + tau_alpha);
      alpha_hat(idx_p1) = optimal_alpha_p1;

      mu    *= optimal_alpha_p1;
      sigma *= std::fabs(optimal_alpha_p1);
      tau_b /= (optimal_alpha_p1 * optimal_alpha_p1);

      mu_alpha(idx_p1) = 1.0 - (optimal_alpha_p1 - mu_alpha(idx_p1));

      eta_bar       *= optimal_alpha_p1;
      zeta          *= optimal_alpha_p1 * optimal_alpha_p1;
      y_dot_eta_bar *= optimal_alpha_p1;

      if (use_gamma_hyperprior_tau_alpha) {
        sigma_sq_alpha(idx_p1) = 1.0 / (zeta + tau_alpha);
      }
    }

    if (use_gamma_hyperprior_tau_alpha && disable_global_alpha) {
      sigma_sq_alpha(idx_p1) = 1.0 / (zeta + tau_alpha);
    }

    // ================================================================
    // PHASE 5: q(pi) update
    // ================================================================
    double c_pi_tilde = c_pi;
    double d_pi_tilde = d_pi;
    if (update_pi) {
      double sum_omega = arma::accu(omega);
      c_pi_tilde = c_pi + sum_omega;
      d_pi_tilde = d_pi + (p - sum_omega);
      E_logit_pi = R::digamma(c_pi_tilde) - R::digamma(d_pi_tilde);
    }

    // ================================================================
    // Recompute var_eta, zeta, y_dot_eta_bar from scratch (numerical
    // stability) and compute convergence criteria
    // ================================================================
    double convergence_value = 1.0;
    double chisq_pvalue = 1.0;
    double max_entropy_change_val = 0.0;

    if (iter > 0) {
      arma::vec d_var = arma::square(mu) % omega % (1.0 - omega) +
                    arma::square(sigma) % omega;
      if (convergence_method == 1 || track_all_criteria) {
        arma::vec var_eta_vec = X_sq * d_var;
        var_eta = arma::accu(var_eta_vec);
        arma::vec tmp = eta_bar_prev - eta_bar;
        arma::vec stat_vec = arma::square(tmp) / var_eta_vec;
        double max_stat = stat_vec.max() / std::log((double)n);
        chisq_pvalue = R::pchisq(max_stat, 1.0, 1, 0);
        if (convergence_method == 1) convergence_value = chisq_pvalue;
      } else {
        var_eta = arma::dot(X_col_sq, d_var);
      }
      zeta = var_eta + arma::dot(eta_bar, eta_bar);
      y_dot_eta_bar = arma::dot(Y, eta_bar);

      if (convergence_method == 2 || track_all_criteria) {
        arma::vec h_old = -omega_old_entropy % arma::log(
          arma::clamp(omega_old_entropy, 1e-15, 1.0 - 1e-15)) -
          (1.0 - omega_old_entropy) % arma::log(
          arma::clamp(1.0 - omega_old_entropy, 1e-15, 1.0 - 1e-15));
        arma::vec h_new = -omega % arma::log(
          arma::clamp(omega, 1e-15, 1.0 - 1e-15)) -
          (1.0 - omega) % arma::log(
          arma::clamp(1.0 - omega, 1e-15, 1.0 - 1e-15));
        max_entropy_change_val = arma::max(arma::abs(h_new - h_old));
        omega_old_entropy = omega;
      }
    }

    // Gamma hyperprior updates
    if (use_gamma_hyperprior_tau_alpha) {
      arma::vec alpha_diff = 1.0 - mu_alpha;
      arma::vec E_alpha_sq = arma::square(alpha_diff) + sigma_sq_alpha;
      d_alpha_post = d_alpha + 0.5 * tau_e * arma::accu(E_alpha_sq);
      tau_alpha = r_alpha_post / d_alpha_post;
      tau_alpha_hist.push_back(tau_alpha);
    }

    if (use_gamma_hyperprior_tau_b) {
      double sum_omega = arma::accu(omega);
      arma::vec E_b_sq = arma::square(mu) + arma::square(sigma);
      double weighted_sum = arma::dot(omega, E_b_sq);
      double r_b_post = r_b + sum_omega / 2.0;
      double d_b_post = d_b + 0.5 * tau_e * weighted_sum;
      tau_b_common = r_b_post / d_b_post;
      tau_b.fill(tau_b_common);
      tau_b_common_hist.push_back(tau_b_common);
    }

    // ELBO computation
    double current_elbo = compute_elbo_scalar(
      mu, sigma, omega, tau_b, mu_alpha,
      y_sq, y_dot_eta_bar, zeta, tau_alpha, tau_e, pi_fixed
    );

    double alpha_normalisation = 0.0;
    double alpha_penalty = 0.0;
    if (tau_alpha > 0.0) {
      arma::vec alpha_diff = 1.0 - mu_alpha;
      alpha_normalisation = 0.5 * (p + 1) * std::log(tau_e * tau_alpha);
      alpha_penalty = 0.5 * tau_e * tau_alpha
                       * arma::dot(alpha_diff, alpha_diff);
    }

    if (update_pi) {
      double sum_omega = arma::accu(omega);
      double old_pi_contribution = std::log(pi_fixed
                                             / (1.0 - pi_fixed))
                                    * sum_omega;
      double pi_posterior = R::lbeta(c_pi_tilde, d_pi_tilde);
      double pi_normalisation = -R::lbeta(c_pi, d_pi);
      current_elbo += (pi_posterior + pi_normalisation)
                       - old_pi_contribution;
    }

    current_elbo += elbo_offset;

    double alpha_stripped_elbo = current_elbo - alpha_normalisation
                                 + alpha_penalty;

    // History storage
    if (save_history) {
      mu_hist.push_back(mu);
      omega_hist.push_back(omega);
      sigma_hist.push_back(sigma);
      tau_b_hist.push_back(tau_b);
      mu_alpha_hist.push_back(mu_alpha);
      alpha_hist.push_back(alpha_hat);
    }
    conv_hist.push_back(convergence_value);
    elbo_hist.push_back(current_elbo);
    alpha_stripped_elbo_hist.push_back(alpha_stripped_elbo);
    if (track_all_criteria) {
      chisq_hist.push_back(chisq_pvalue);
      entropy_change_hist.push_back(max_entropy_change_val);
    }

    last_iter = iter;

    // Convergence check
    if (convergence_method == 0 || convergence_method == 3) {
      if (iter > 0) {
        double abs_change = std::fabs(current_elbo - prev_elbo);
        double rel_change = abs_change / (std::fabs(prev_elbo) + 1e-10);
        if (convergence_method == 0) {
          convergence_value = rel_change;
          if (rel_change < tol) {
            converged = true;
            prev_elbo = current_elbo;
            break;
          }
        } else {
          convergence_value = abs_change;
          if (abs_change < tol) {
            converged = true;
            prev_elbo = current_elbo;
            break;
          }
        }
      }
      prev_elbo = current_elbo;
    } else if (convergence_method == 1) {
      if (convergence_value < tol) {
        converged = true;
        break;
      }
    } else {
      if (iter > 0 && max_entropy_change_val < tol) {
        converged = true;
        break;
      }
    }
  }

  // ================================================================
  // Build return list (same structure as run_vb_updates_cpp)
  // ================================================================
  arma::vec conv_vec(conv_hist);
  arma::vec elbo_vec(elbo_hist);
  arma::vec alpha_stripped_elbo_vec(alpha_stripped_elbo_hist);

  double final_c_pi_tilde = c_pi;
  double final_d_pi_tilde = d_pi;
  if (update_pi) {
    double sum_omega = arma::accu(omega);
    final_c_pi_tilde = c_pi + sum_omega;
    final_d_pi_tilde = d_pi + (p - sum_omega);
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("converged") = converged,
    Rcpp::Named("iterations") = last_iter + 1,
    Rcpp::Named("convergence_criterion") = conv_hist.back(),
    Rcpp::Named("convergence_history") = conv_vec,
    Rcpp::Named("elbo_history") = elbo_vec,
    Rcpp::Named("alpha_stripped_elbo_history") = alpha_stripped_elbo_vec,
    Rcpp::Named("mu") = mu,
    Rcpp::Named("omega") = omega,
    Rcpp::Named("sigma") = sigma,
    Rcpp::Named("tau_b") = tau_b,
    Rcpp::Named("mu_alpha") = mu_alpha,
    Rcpp::Named("c_pi_tilde") = final_c_pi_tilde,
    Rcpp::Named("d_pi_tilde") = final_d_pi_tilde,
    Rcpp::Named("coordinate_exploded_elbo_violations") = 0,
    Rcpp::Named("coordinate_exploded_elbo_worst_drop") = 0.0,
    Rcpp::Named("coordinate_alpha_stripped_elbo_violations") = 0,
    Rcpp::Named("coordinate_alpha_stripped_elbo_worst_drop") = 0.0,
    Rcpp::Named("tau_alpha") = tau_alpha
  );

  if (use_gamma_hyperprior_tau_alpha) {
    arma::vec tau_alpha_vec(tau_alpha_hist);
    result["tau_alpha_history"] = tau_alpha_vec;
  }

  if (use_gamma_hyperprior_tau_b) {
    arma::vec tau_b_common_vec(tau_b_common_hist);
    result["tau_b_common_history"] = tau_b_common_vec;
  }

  if (track_all_criteria) {
    arma::vec chisq_vec(chisq_hist);
    arma::vec entropy_change_vec(entropy_change_hist);
    result["chisq_history"] = chisq_vec;
    result["entropy_change_history"] = entropy_change_vec;
  }

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
