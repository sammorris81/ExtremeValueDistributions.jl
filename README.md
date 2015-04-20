# ExtremeValueDistributions.jl

A Julia package to extend [**Distributions**](https://github.com/JuliaStats/Distributions.jl) to include the generalized extreme value distribution (GEV) and generalized Pareto distribution (GPD).

## Requires

This package requires two unregistered packages that are used for fitting the distributions using MCMC methods. To install these packages open julia and run

```julia
Pkg.clone("https://github.com/sammorris81/MetropolisUpdaters.jl.git")
Pkg.clone("https://github.com/sammorris81/DataTransformations.jl.git")
```

## Installation

```julia
Pkg.clone("https://github.com/sammorris81/ExtremeValueDistributions.jl.git")
```

## Interface

The interface for ExtremeValueDistributions utilizes the common interface of Distributions as much as possible.

### Generalized extreme value distribution

To generate a generalized extreme value distribution, use the function `GeneralizedExtremeValue(μ, σ, ξ)` where
* `μ` is the location parameter
* `σ` is the scale parameter
* `ξ` is the shape parameter

### Generalized Pareto distribution

To generate a generalized Pareto distribution, use the function `GeneralizedPareto(μ, σ, ξ)` where
* `μ` is the location parameter
* `σ` is the scale parameter
* `ξ` is the shape parameter

## Available methods

Let `d` be a distribution of type `GeneralizedExtremeValue` or `GeneralizedPareto`:

### Parameter retrieval

* `params(d)` returns a tuple of parameters
* `location(d)` returns the location parameter
* `scale(d)` returns the location parameter
* `shape(d)` returns the shape parameter

### Computation of statistics

* `mean(d)` returns the expectation of distribution `d`
* `var(d)` returns the variance of distribution `d`
* `std(d)` returns the standard deviation of disitribution `d`, i.e. `sqrt(var(d))`
* `median(d)` returns the median value of distribution `d`
* `mode(d)` returns the mode of distribution `d`
* `skewness(d)` returns the skewness of distribution `d`
* `kurtosis(d)` returns the excess kurtosis of distribution `d`
* `entropy(d)` returns the entropy of distribution `d`

### Probability evaluation

* `insupport(d, x)` returns whether `x` is within the support of `d`
* `pdf(d, x)` returns the pdf value evaluated at `x`
* `logpdf(d, x)` returns the logarithm of the pdf value evaluated at `x`, i.e. `log(pdf(d, x))`
* `cdf(d, x)` returns the cumulative distribution function evaluated at `x`
* `logcdf(d, x)` returns the logarithm of the cumulative distribution function evaluated at `x`
* `ccdf(d, x)` returns the complementary cumulative function evaluated at `x`, i.e. `1 - cdf(d, x)`
* `logccdf(d, x)` returns the logarithm of the complementary cumulative function evaluated at `x`
* `quantile(d, q)` returns the qth quantile value
* `cquantile(d, q)` returns the complementary quantile value, i.e. `quantile(d, 1 - q)`

### Sampling (Random number generation)

* `rand(d)` draws a single sample from `d`
* `rand(d, n)` draws a vector comprised of `n` independent samples from the distribution `d`

### Model fitting

* [Maximum likelihood] (./src/mle/)
* [MCMC] (./src/bayes/)
