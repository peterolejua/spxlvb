// [[Rcpp::depends(RcppArmadillo)]]

#include <fstream>
#include <iostream>
#include <RcppArmadillo.h>
#include <Rcpp.h> // Explicitly include Rcpp.h
#include <cmath>  // For std::sqrt, std::log, std::abs

#include "common_helpers.h" // Include your new general helpers header

// [[Rcpp::export(fit_linear_exploded)]]
List fit_linear_exploded(
    const arma::mat &X, // Design matrix
    const arma::vec &Y, // Response vector
    arma::vec mu,         // estimated beta coefficient from lasso
    arma::vec omega,      // Expectation that the coefficient from lasso is not zero
    double c_pi,    // Parameter of the Beta prior for pi
    double d_pi,    // Parameter of the Beta prior for pi
    double tau_e,   // Precision of the errors
    const arma::uvec &update_order, // Order in which to update the coefficients
    arma::vec mu_alpha,   // alpha_j is N(mu_alpha_j, (tau_e*tau_alpha)^{-1}), known/estimated
    double tau_alpha, // Precision of the Normal prior for the expansion parameters (alpha_j)
    arma::vec tau_b,      // Precision of the prior for the coefficients (b_j)
    const size_t &max_iter,   // Maximum number of iterations
    const double &tol        // Tolerance for convergence
) {

  // dimensions
  double p = X.n_cols;


  // initializations
  arma::vec old_entr = entropy(omega);

  // R1 update specific global sum variable (initialized only if use_r1_update is true)
  arma::rowvec YX_vec = Y.t() * X;
  arma::mat XtX = X.t() * X; // Precompute X^t X
  arma::vec half_diag = gram_diag(X);
  arma::vec approx_mean = omega % mu;

  arma::vec alpha_j_optimal(p, fill::ones);
  arma::vec mu_tilde = mu;
  arma::vec sigma = 1 / arma::sqrt(tau_e * (half_diag + tau_b));
  arma::vec sigma_tilde = sigma;
  arma::vec tau_b_tilde = tau_b;
  arma::vec mu_alpha_tilde = mu_alpha;

  arma::vec W = X * approx_mean;
  arma::mat X_2 = arma::square(X);

  double a_pi = c_pi;
  double b_pi = d_pi;
  double E_logit_pi = r_digamma(a_pi) - r_digamma(b_pi);

  bool converged = false;
  double convergence_criterion = -999.0;
  size_t iter = 0;
  // std::cout << "iter: " << iter << std::endl;
  for (iter = 0; iter < max_iter; ++iter) {
    // std::cout << "Iteration: " << iter << std::endl;

    vec new_entr = zeros<vec>(p);

    for (size_t k = 0; k < p; ++k) {
      Rcpp::checkUserInterrupt();
      uword j = update_order(k);
      // std::cout << "j: " << j << std::endl;


      // Calculate W_j using the online update optimization
      // std::cout << "Calculate W_j using the online update optimization" << std::endl;
      arma::vec W_j = W - approx_mean(j) * X.col(j);
      arma::vec var_W = X_2 * (arma::square(mu) % omega % (1 - omega) + sigma % sigma % omega);
      arma::vec var_W_j = var_W - X_2.col(j) * (mu(j)* mu(j) * omega(j) * (1 - omega(j)) + sigma(j) * sigma(j) * omega(j));
      double W_j_squared = arma::sum(var_W_j + arma::square(W_j));

      // std::cout << "Calculate Lambda" << std::endl;
      // Calculate Lambda and eta for s_j = 0
      auto lambda_eta_sigma_0 = calculate_lambda_eta_sigma(
        X,
        XtX,
        YX_vec,
        Y,
        W_j,
        W_j_squared,
        0,
        j,
        tau_e,
        tau_b,
        tau_alpha,
        mu_alpha
        );

      arma::mat Lambda_j_0 = std::get<0>(lambda_eta_sigma_0);
      arma::mat Sigma_j_0 = std::get<1>(lambda_eta_sigma_0);
      arma::vec eta_j_0 = std::get<2>(lambda_eta_sigma_0);

      // Calculate Lambda and eta for s_j = 1
      auto lambda_eta_sigma_1 = calculate_lambda_eta_sigma(
        X,
        XtX,
        YX_vec,
        Y,
        W_j,
        W_j_squared,
        1,
        j,
        tau_e,
        tau_b,
        tau_alpha,
        mu_alpha
        );

      arma::mat Lambda_j_1 = std::get<0>(lambda_eta_sigma_1);
      arma::mat Sigma_j_1 = std::get<1>(lambda_eta_sigma_1);
      arma::vec eta_j_1 = std::get<2>(lambda_eta_sigma_1);

      // Calculate logit(phi_j)
      // std::cout << "Calculate logit(phi_j)" << std::endl;
      double det_Lambda_j_0 = Lambda_j_0(0, 0) * Lambda_j_0(1, 1) - pow(Lambda_j_0(0, 1), 2);
      double det_Lambda_j_1 = Lambda_j_1(0, 0) * Lambda_j_1(1, 1) - pow(Lambda_j_1(0, 1), 2);

      // Safeguard determinants for log(): ensure they are positive and not too small
      det_Lambda_j_0 = std::max(arma::datum::eps, det_Lambda_j_0);
      det_Lambda_j_1 = std::max(arma::datum::eps, det_Lambda_j_1);

      double logit_phi_j = E_logit_pi +
        0.5 * log( det_Lambda_j_0 / det_Lambda_j_1 ) -
        0.5 * pow(eta_j_0(1), 2) * Lambda_j_0(1, 1) +
        0.5 * pow(eta_j_1(0), 2) * Lambda_j_1(0, 0) +
        eta_j_1(0) * eta_j_1(1) * Lambda_j_1(0, 1) +
        0.5 * pow(eta_j_1(1), 2) * Lambda_j_1(1, 1);

      double phi_j = sigmoid(logit_phi_j);

      // std::cout << "optimal alpha_j" << std::endl;
      // Calculate optimal alpha_j using a simple grid search optimizer ---
      double var_alpha_0 = Sigma_j_0(1,1);
      double var_alpha_1 = Sigma_j_1(1,1);

      // Safeguard variances for division (ensure they are positive and not too small)
      if (var_alpha_0 < arma::datum::eps) var_alpha_0 = arma::datum::eps;
      if (var_alpha_1 < arma::datum::eps) var_alpha_1 = arma::datum::eps;

      // Define a search interval around the means of the two distributions
      double mean_min = std::min(eta_j_0(1), eta_j_1(1));
      double mean_max = std::max(eta_j_0(1), eta_j_1(1));
      double search_range = std::max(mean_max - mean_min, 1.0); // Ensure a reasonable range
      double lower_bound = mean_min - search_range;
      double upper_bound = mean_max + search_range;

      double max_density = -1.0;
      double optimal_alpha_j = 0.0;

      // Perform a grid search with a fine resolution
      const int num_steps = 1000;
      double step_size = (upper_bound - lower_bound) / num_steps;
      for (int i = 0; i <= num_steps; ++i) {
        double alpha_val = lower_bound + i * step_size;
        double current_density = mixture_pdf(alpha_val, phi_j, eta_j_0(1), var_alpha_0, eta_j_1(1), var_alpha_1);
        if (current_density > max_density) {
          max_density = current_density;
          optimal_alpha_j = alpha_val;
        }
      }

      alpha_j_optimal(j) = optimal_alpha_j;

      // Update omega_j (posterior expectation of s_j)
      double f_alpha_0 = gaussian_pdf(optimal_alpha_j, eta_j_0(1), var_alpha_0);
      double f_alpha_1 = gaussian_pdf(optimal_alpha_j, eta_j_1(1), var_alpha_1);

      // Safeguard denominator in omega(j) calculation
      double omega_denominator = phi_j * f_alpha_1 + (1 - phi_j) * f_alpha_0;
      if (omega_denominator < arma::datum::eps) { // Check if denominator is too small
        omega(j) = arma::datum::nan; // Propagate NaN if numerically unstable
      } else {
        omega(j) = (phi_j * f_alpha_1) / omega_denominator;
      }

      // After calculating omega(j)
      if (!arma::is_finite(omega(j))) {
        Rcpp::stop("omega(j) became NaN or Inf at j = %u, iter = %zu. Denominator: %f", j, iter, omega_denominator);
      }

      // And before using old_E_bs_j:
      if (!arma::is_finite(mu(j)) || !arma::is_finite(omega(j))) {
        Rcpp::stop("mu(j) or omega(j) is not finite before old_E_bs_j calculation at j = %u, iter = %zu", j, iter);
      }

      // Also check W_j and W_j_squared:
      if (!W_j.is_finite()) {
        Rcpp::stop("W_j became NaN or Inf at j = %u, iter = %zu", j, iter);
      }
      if (!arma::is_finite(W_j_squared)) {
        Rcpp::stop("W_j_squared became NaN or Inf at j = %u, iter = %zu", j, iter);
      }

      // Update mu_j (posterior mean of b_j when s_j = 1)
      mu(j) = eta_j_1(0) - Sigma_j_1(0, 1) / Sigma_j_1(1, 1) * (optimal_alpha_j - eta_j_1(1));

      // Update sigma_j^2 (variance of b_j when s_j = 1)
      sigma(j) = 1.0 / (tau_e * XtX(j, j) + tau_e * tau_b(j));

      // --- Remapping step ---
      mu_tilde = mu;
      sigma_tilde = sigma;
      tau_b_tilde = tau_b;
      mu_alpha_tilde = mu_alpha;

      for (uword l = 0; l < p; ++l) {
        if (l != j) {
          mu_tilde(l) *= optimal_alpha_j;
          sigma_tilde(l) *= pow(optimal_alpha_j, 2);
          tau_b_tilde(l) /= pow(optimal_alpha_j, 2);
        }
        if (l == j) {
          mu_alpha_tilde(l) = 1.0 - (optimal_alpha_j - mu_alpha(l));
        } else {
          mu_alpha_tilde(l) = mu_alpha(l);
        }
      }

      // --- Set expanded to original ---
      mu = mu_tilde;
      sigma = sigma_tilde;
      tau_b = tau_b_tilde;
      mu_alpha = mu_alpha_tilde;

      // Update q(pi)
      double M = sum(omega);
      a_pi = c_pi + M;
      b_pi = d_pi + p - M;
      E_logit_pi = r_digamma(a_pi) - r_digamma(b_pi);

      // Update W with the new mu and omega
      // approx_mean = omega % mu;
      // var_W = optimal_alpha_j*optimal_alpha_j*var_W_j + X_2.col(j) * (mu(j)* mu(j) * omega(j) * (1 - omega(j)) + sigma(j) * sigma(j) * omega(j));
      // W = optimal_alpha_j*W_j + approx_mean(j) * X.col(j);

      // Redefine W O(np)
      approx_mean = (omega % mu);
      W = X * approx_mean;

    } // end of for k loop


    // Check for convergence
    new_entr = entropy(omega);

    if ( norm(new_entr - old_entr, "inf") <= tol ) {
      converged = true;
      convergence_criterion = norm(new_entr - old_entr, "inf");
      break;
    } else {
      convergence_criterion = norm(new_entr - old_entr, "inf");
      old_entr = new_entr;
    }
  } // end of for iter loop

  return Rcpp::List::create(
    Rcpp::Named("mu") = mu,
    Rcpp::Named("omega") = omega,
    Rcpp::Named("sigma") = sigma,
    Rcpp::Named("mu_alpha") = mu_alpha,
    Rcpp::Named("iterations") = iter + 1,
    Rcpp::Named("tau_alpha") = tau_alpha,
    Rcpp::Named("tau_b") = tau_b,
    Rcpp::Named("E_logit_pi") = E_logit_pi,
    Rcpp::Named("a_pi") = a_pi,
    Rcpp::Named("b_pi") = b_pi,
    Rcpp::Named("converged") = converged,
    Rcpp::Named("convergence_criterion") = convergence_criterion
  );
}
