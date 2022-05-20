## Parameters
# y: Response vector y or, if X is missing, then matrix cbind(y,X). 
#    Coerced if y is not a matrix. Ignored if CF is specified.
# X: Feature matrix X or NULL. Coerced if X is not a matrix. Ignored if CF is specified.
# utility: The utility function. Ignored if CF is specified.
# v: Vector of feature indices (column numbers of X) to calculate shapley value of.
#    If missing then assumed all.
# CF: Characteristic function. Estimated (using utility) if missing.
shapley <- function(y, X, utility, v, CF, drop_method = "actual", ...) {

  if ( !is.matrix(y) ) {y <- as.matrix(y)}
  if (any(!is.finite(y))) {stop(
    paste0("shapley can only handle finite numbers at this time, ",
           "please check y for NA, NaN or Inf"))}
  if (!missing(CF)) {
    if (missing(v)) {
      v <- attr(CF,"players")
    }
    return(shapley_vec(CF, v))
  }
  if ( ncol(y) > 1 & missing(X) ) {
    X <- y[,-1, drop = F]
    y <- y[, 1, drop = F]
  }
  if ( !is.matrix(X) ) {X <- as.matrix(X)}
  if ( any(!is.finite(X)) ) {stop(
    paste0("shapley can only handle finite numbers at this time, ",
           "please check X for NA, NaN or Inf"))}
  if (missing(v)) {v <- 1:ncol(X)}
  
  CF <- estimate_CF(X, utility, drop_method = drop_method, y = y, ...)
  sv <- shapley_vec(CF, v)
  names(sv) <- colnames(X)
  return(sv)
}

# We don't know the population characteristic function,
# so we use the utility function to estimate the 
# characteristic function from the data X.
estimate_CF <- function(X, utility, drop_method = "actual", ...) {
  values <- list()
  players <- 1:ncol(X)
  num_players <- length(players)
  team_sizes <- 0:num_players
  
  # We now precompute all the
  # possible values of the utility function
  if ( tolower(drop_method) == "actual" ) {
    for ( s in team_sizes ) {
      teams_of_size_s <- combn( players, s, simplify = F )
      for ( team in teams_of_size_s ) {
        Xs <- X[,team,drop = F]
        values[[access_string(team)]] <- utility(Xs, ...) 
      }
    }
  } else if ( tolower(drop_method) == "mean" ) {
    for ( s in team_sizes ) {
      teams_of_size_s <- combn( players, s, simplify = F )
      for ( team in teams_of_size_s ) {
        Xs <- mean_drop(X, team)
        values[[access_string(team)]] <- utility(Xs, ...) 
      }
    }
  }
  # We created some bindings in this environment 
  # and we are now returning a function that 
  # permantently has access to this environment,
  # so we can access this environment from anywhere
  CF <- function(t){values[[access_string(t)]]}
  attr(CF, "players") <- players
  return(CF)
}

mean_drop <- function(X, team) {
  d <- ncol(X)
  if (length(team) == d) {return(X)}
  Xr <- if ( length(team) > 0 ) {X[, -team, drop = F]} else {X}
  Er <- apply(Xr, FUN = mean, MARGIN = 2)
  for (nam in names(Er)) {X[,nam] <- Er[nam]}
  return(X)
}


# The Shapley value of a player can be broken into
# the mean of the average utility of that player
# within each team size.
shapley_v <- function(CF, v) {
  players <- environment(CF)$players[-v]
  num_players <- length(players)
  team_sizes <- 0:num_players
  value <- 0
  for ( s in team_sizes ) {
    value_s <- 0
    teams_of_size_s <- if (length(players) != 1) {
      combn(players, s, simplify = F)} else if (s == 1) 
      {list(players)} else {list(integer())}
    for ( team in teams_of_size_s ) {
      value_in_team <- CF(c(v,team)) - CF(team)
      value_s <- value_s + value_in_team
    }
    average_value_s <- value_s/length(teams_of_size_s)
    value <- value + average_value_s
  }
  average_value <- value/length(team_sizes)
  return(average_value)
}

shapley_vec <- Vectorize(shapley_v, "v")

# This function converts teams into strings so we can look
# them up in the characteristic function, a bit like a dictionary.
access_string <- function(team) {paste0("-", sort(team), collapse = "")}
