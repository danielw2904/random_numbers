#include <RcppArmadillo.h>
#include <omp.h>
#include <random>
#include <iostream>
#include <boost/random.hpp>
#include <boost/random/gamma_distribution.hpp>
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat rdirichlet(int n, arma::rowvec alpha) {
  unsigned int l = alpha.size();
  arma::mat dirich(n, l);
  for(int i=0; i<l; i++){
    dirich.col(i) = Rcpp::as<arma::colvec>(Rcpp::rgamma(n, alpha[i], 1));
  } 
  arma::vec V = sum(dirich,1);
  dirich.each_col() /= V;
  return dirich;
}

// [[Rcpp::export]]
arma::mat rd_armaMC(int n, arma::rowvec alpha, int threads) { 
  omp_set_num_threads(threads);
  unsigned int l = alpha.size();
  double b = 1;
  arma::mat dirich(n, l);
  int i;
  #pragma omp for schedule(auto) 
  for(i=0; i<l; i++){
    double ai = alpha[i];
    dirich.col(i) = arma::randg<arma::colvec>(n, arma::distr_param(ai, b));
  }
  arma::vec V = sum(dirich, 1);
  dirich.each_col() /= V;
  return dirich;
}

// [[Rcpp::export]]
arma::mat rd_arma(int n, arma::rowvec alpha) { 
    unsigned int l = alpha.size();
    arma::mat dirich(n, l);
    double b = 1; // call to arma::distr_param ambiguous if this is an int
    for(int i=0; i<l; i++){
      double ai = alpha[i];
      dirich.col(i) = arma::randg<arma::colvec>(n, arma::distr_param(ai, b));
    } 
    arma::vec V = sum(dirich, 1);
    dirich.each_col() /= V;
    return dirich;
  }

// [[Rcpp::export]]
arma::mat rd_std(int n, arma::rowvec alpha){ // this is the fastest!
  omp_set_num_threads(4);
  
  int l = alpha.size();
  arma::mat x(n, l);
  std::random_device r;
#pragma omp parallel for schedule(auto)
  for(int i=0; i<l; i++){
    std::mt19937 rng(r());
    std::gamma_distribution<double> gam(alpha[i], 1);
    arma::vec y(n);
    auto draw = std::bind(gam, rng);
    std::generate(y.begin(), y.end(), draw);
    x.col(i) = y;
  }
  arma::vec V = sum(x, 1);
  x.each_col() /= V;
  return x;
}


// [[Rcpp::export]]
arma::mat rd_boost(int n, arma::rowvec alpha) {
  int l = alpha.size();
  arma::mat x(n, l);
  std::random_device r;
  boost::mt19937 rng(r());
  for(int i=0; i<l;i++){
    boost::gamma_distribution<double> gam(alpha[i]);
    boost::variate_generator<boost::mt19937&, boost::gamma_distribution<double> >  gen(rng, gam);
    arma::vec y(n);
    std::generate(y.begin(), y.end(), gen);
    x.col(i) = y;
  }
  arma::vec V = sum(x, 1);
  x.each_col() /= V;
  return x;
}
// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R
# From
# https://github.com/cran/MCMCpack/blob/ba7a28d7fb083ce6fdb17a8a1c0bb9d06394d3d3/R/distn.R
rdir <- function(n, alpha) {
    l <- length(alpha)
    x <- matrix(rgamma(l*n,alpha),ncol=l,byrow=TRUE)
    sm <- x%*%rep(1,l)
    return(x/as.vector(sm))
}
###

chk_draws <- function(arr_draws, alpha){
  print("Sum of differences with R equivalent:")
  print(sum(R_dir - arr_draws))
  R_dir_mean <- apply(R_dir, c(1,2), mean)
  other_mean <- apply(arr_draws, c(1,2), mean)
  other_var <- apply(arr_draws, c(1,2), var)
  print("Differences in means with R equivalent:")
  print(R_dir_mean - other_mean)
  ex <- alpha/sum(alpha)
  print("Difference of mean to theoretical expectation:")
  print(apply(other_mean, 1, function(x)x-ex))
  print("Sum of difference of mean to theoretical expectation:")
  print(apply((apply(other_mean, 1, function(x)x-ex)), 2, sum))
  print("Difference of var to theoretical var:")
  thvar <- (alpha * (sum(alpha) - alpha)) / (sum(alpha)^2 * (sum(alpha + 1)))
  print(apply(other_var, 1, function(x)x-thvar))
}

n <- 4
alpha_size <- 5
alpha <- sample(c(1:4), alpha_size, replace = TRUE)

nrep <- 1e4
R_dir <- Rcpp_dir <- boost_dir <- arma_dir <- arma_mcdir <- std_dir <- array(NA, c(4, alpha_size, nrep))

for (i in 1:nrep){
  R_dir[,,i] <- rdir(n, alpha)
  Rcpp_dir[,,i] <- rdirichlet(n, alpha)
  boost_dir[,,i] <- rd_boost(n, alpha)
  arma_dir[,,i] <- rd_arma(n, alpha)
  arma_mcdir[,,i] <- rd_armaMC(n, alpha, 4)
  std_dir[,,i] <- rd_std(n, alpha)
}

chk_draws(R_dir, alpha)
chk_draws(Rcpp_dir, alpha)
chk_draws(boost_dir, alpha)
chk_draws(arma_dir, alpha)
chk_draws(armamc_dir, alpha)
chk_draws(std_dir, alpha)

library(microbenchmark)
m <- 1000
alphan <- sample(1:10, 1000, replace = T)
microbenchmark(
  y <- rdir(m, alphan)
)
microbenchmark(
  y <- rdirichlet(m, alphan)
)
microbenchmark(
  y <- rd_arma(m, alphan)
)
microbenchmark(
  y <- rd_boost(m, alphan)
)
microbenchmark(
  y <- rd_armaMC(m, alphan, 4)
)
microbenchmark(
  y <- rd_std(m, alphan)
)
*/
