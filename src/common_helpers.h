#ifndef COMMON_HELPERS_H
#define COMMON_HELPERS_H

#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <tuple>

using namespace arma;
using namespace Rcpp;

// Function declaration for entropy()
arma::vec entropy(const arma::vec &x);

// Function declaration for sigmoid()
double sigmoid(const double &x);

// Function declaration for gram_diag()
arma::vec gram_diag(const arma::mat &X);

// Function declaration for r_digamma()
double r_digamma(double x);

// Function declaration for calculate_lambda_eta_sigma()
std::tuple<mat, mat, vec> calculate_lambda_eta_sigma(
    const mat &X,
    const mat &XtX,
    const rowvec &YX_vec,
    const vec &Y,
    const vec &W_j,
    double W_j_squared,
    int s_j_val,
    uword j,
    double tau_e,
    const vec &tau_b,
    double tau_alpha,
    const vec &mu_alpha
);

// Function declaration for gaussian_pdf()
double gaussian_pdf(double x, double mean, double variance);

// Function declaration for mixture_pdf()
double mixture_pdf(double alpha, double phi, double mean0, double var0, double mean1, double var1);

#endif // COMMON_HELPERS_H
