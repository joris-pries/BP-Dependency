library(speedglm) # speedlm
library(Rfast) # dcor bcdcor
library(dHSIC) # dHSIC
library(expm) # sqrtm



# Vanilla utility functions -----------------------------------------------

# R squared
R2 <- function(y, X) {if (length(X) == 0) {0} else {summary(speedlm(y~X))$r.squared}}
# Distance correlation
DC <- function(y, X){if (length(X) == 0) {return(0)} else {
  dc <- dcor(y,X)$dcor
  if (is.nan(dc)) {return(0)} else {return(dc)}
  }}
# Bias corrected distance correlation
BCDC <- function(y, X){if (length(X) == 0) {0} else {bcdcor(y,X)}}
# Affine invariant distance correlation
AIDC <- function(y,X){if (length(X) == 0) {0} else {
  dcor(y %*% sqrtm(solve(cov(y))), X %*% sqrtm(solve(cov(X))))$dcor}
}
# Hilbert Schmidt Independence Criterion
HSIC <- function(y,X){if (length(X) == 0) {0} else {hsic(y,X)}}



# res vs fits utility functions -------------------------------------------

DC_rf <- function(y, X, model){if (length(X) == 0) {0} else {
    preds <- predict(model, X)
    if (zero_range(preds)) {return(0)}
    dcor(y-preds, preds)$dcor
  }
}


# UTILITY FUNCTION HELPERS ------------------------------------------------
zero_range <- function(x, tol = .Machine$double.eps ^ 0.5) {
  if (length(x) == 1) return(TRUE)
  x <- range(x, na.rm = T) / mean(x, na.rm = T)
  isTRUE(all.equal(x[1], x[2], tolerance = tol))
}


gaussianK <- function(X) {
  n <- nrow(X); d <- ncol(X)
  bandwidth <- dHSIC:::median_bandwidth_rcpp(X, n, d)
  if (bandwidth == 0) {bandwidth <- .Machine$double.eps}
  return(dHSIC:::gaussian_grammat_rcpp(X, bandwidth, n, d))
}
hsic <- function(y,X) {
  k <- gaussianK(X); l <- gaussianK(y)
  n <- nrow(k)
  dterm1 <- sum(k*l)
  dterm2 <- 1/(n^4)*sum(k)*sum(l)
  dterm3 <- 2/(n^3)*sum(k %*% l)
  return(1/(n^2)*dterm1 + dterm2 - dterm3)
}
