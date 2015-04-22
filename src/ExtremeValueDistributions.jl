module ExtremeValueDistributions

using Distributions
using Compat
using MetropolisUpdaters  # types for Bayes fitting
using RDatasets           # for readtable
using DataFrames          # to convert DataArrays to Arrays

# get methods from Base
import Base.Random
import Base: size, eltype, length, full, convert, show, getindex, scale, rand, rand!
import Base: sum, mean, median, maximum, minimum, quantile, std, var, cov, cor
# get methods from StatsBase
import StatsBase: kurtosis, skewness, entropy, mode, modes, randi, fit, kldivergence
# get methods from Distributions
import Distributions: ccdf, cdf, cquantile, isbounded, islowerbounded
import Distributions: isupperbounded, hasfinitesupport, insupport, invlogccdf, invlogcdf, location
import Distributions: logccdf, logcdf, mean, median, params, logpdf, pdf, quantile
import Distributions: scale, shape, skewness, support, var, sample
# get types from Distributions
import Distributions: ContinuousUnivariateDistribution
# removing NAs at the moment
import DataFrames: deleterows!, isna

# for mcmc
# import Base.LinAlg.BLAS.gemv!

export
  # distribution types
  GeneralizedPareto,
  GeneralizedExtremeValue,
  # mcmc types
  ExtremePosterior,
  GeneralizedExtremeValuePosterior,
  GeneralizedParetoPosterior,

  # methods
  ccdf,
  cdf,
  cquantile,
  invlogccdf,
  invlogcdf,
  insupport,
  isbounded,
  islowerbounded,
  isupperbounded,
  hasfinitesupport,
  kurtosis,
  location,
  logccdf,
  logcdf,
  logpdf,
  mean,
  median,
  minimum,
  maximum,
  mode,
  params,
  pdf,
  quantile,
  scale,
  shape,
  skewness,
  support,
  var,
  sample,

  # fitting
  fit_mcmc,

  # load data
  extremedata


### source files
include("density/generalizedpareto.jl")
include("density/generalizedextreme.jl")
include("bayes/mcmc.jl")
include("bayes/generalizedextreme.jl")
include("bayes/generalizedpareto.jl")
include("loaddata.jl")

end # module
