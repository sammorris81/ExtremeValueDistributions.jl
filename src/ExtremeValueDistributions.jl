module ExtremeValueDistributions

using Distributions
using Compat

import Base.Random
# get methods from StatsBase
import StatsBase: kurtosis, skewness, entropy, mode, modes, randi, fit, kldivergence
# get methods from Distributions
import Distributions: ccdf, cdf, cquantile, invlogccdf, invlogcdf, location
import Distributions: logccdf, logcdf, mean, median, params, logpdf, pdf, quantile
import Distributions: scale, shape, skewness, var, sample
# import Distributions: @distr_support
# get types from Distributions
import Distributions: ContinuousUnivariateDistribution

export
  # distribution types
  GeneralizedPareto,
  GeneralizedExtremeValue,

  # methods
  ccdf,
  cdf,
  cquantile,
  invlogccdf,
  invlogcdf,
  kurtosis,
  location,
  logccdf,
  logcdf,
  logpdf,
  mean,
  median,
  mode,
  params,
  pdf,
  quantile,
  scale,
  shape,
  skewness,
  var,
  sample

### source files
include("support.jl")
include("density/generalizedpareto.jl")
include("density/generalizedextreme.jl")

end # module
