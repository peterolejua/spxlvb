// common_helpers.cpp
#include "common_helpers.h" // Include its own header first

// Define helper functions here
arma::vec entropy(const arma::vec &x) {
  arma::vec ent(x.n_elem, arma::fill::zeros);
  for (arma::uword j = 0; j < x.n_elem; ++j) {
    // clamp values to avoid -Inf
    if ((x(j) > 1e-10) && (x(j) < 1 - 1e-10)) {
      ent(j) -= x(j) * std::log2(x(j)) + (1 - x(j)) * std::log2(1 - x(j));
    }
  }
  return ent;
}

double sigmoid(const double &x) {
  if (x > 32.0) {
    return 1;
  } else if (x < -32.0) {
    return 0;
  } else {
    return 1 / (1 + std::exp(-x));
  }
}

arma::vec gram_diag(const arma::mat &X) {
  arma::vec diag(X.n_cols);

  for (arma::uword i = 0; i < diag.n_elem; ++i) {
    diag(i) = std::pow(arma::norm(X.col(i)), 2);
  }
  return diag;
}

// Function to calculate the digamma function using R's digamma
double r_digamma(double x) {
  static Rcpp::Function digamma("digamma");
  return Rcpp::as<double>(digamma(Rcpp::Named("x") = x));
}


// Calculates Lambda_j, Sigma_j, and eta_j
std::tuple<mat, mat, vec> calculate_lambda_eta_sigma(
    const mat &X,
    const mat &XtX, // precomputed XtX
    const rowvec &YX_vec, // precomputed XtX
    const vec &Y,
    const vec &W_j,
    double W_j_squared,
    int s_j_val,
    uword j,
    double tau_e,
    const vec &tau_b,
    double tau_alpha,
    const vec &mu_alpha
) {
  double s_j = static_cast<double>(s_j_val);
  vec X_j = X.col(j);

  mat Lambda_j(2, 2);
  // Ensure that X_j squared is finite and positive
  double X_j_sq_val = XtX(j,j);
  if (!arma::is_finite(X_j_sq_val) || X_j_sq_val < 0) X_j_sq_val = arma::datum::eps; // safeguard

  Lambda_j(0, 0) = tau_e * s_j * X_j_sq_val + tau_e * tau_b(j);
  Lambda_j(0, 1) = tau_e * s_j * dot(X_j, W_j);
  Lambda_j(1, 0) = tau_e * s_j * dot(X_j, W_j);
  Lambda_j(1, 1) = tau_e * W_j_squared + tau_e * tau_alpha;

  mat Sigma_j = inv(Lambda_j);

  vec eta_j(2);
  eta_j(0) = tau_e * YX_vec(j);
  eta_j(1) = tau_e * dot(Y, W_j) + tau_e * tau_alpha * mu_alpha(j);
  eta_j = Sigma_j * eta_j;

  return std::make_tuple(Lambda_j, Sigma_j, eta_j);
}

// Gaussian probability density function
double gaussian_pdf(double x, double mean, double variance) {
  // Ensure variance is positive for sqrt
  if (variance <= 0) {
    return 0.0;
  }
  double diff = x - mean;
  return (1.0 / sqrt(2.0 * M_PI * variance)) * exp(-0.5 * diff * diff / variance);
}

// Objective function to maximize (the mixture PDF)
double mixture_pdf(double alpha, double phi, double mean0, double var0, double mean1, double var1) {
  return (1.0 - phi) * gaussian_pdf(alpha, mean0, var0) + phi * gaussian_pdf(alpha, mean1, var1);
}
