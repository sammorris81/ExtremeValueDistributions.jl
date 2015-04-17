immutable GeneralizedExtremeValue <: ContinuousUnivariateDistribution
  μ::Float64
  σ::Float64
  ξ::Float64

  function GeneralizedExtremeValue(μ::Real, σ::Real, ξ::Real)
    σ > zero(σ) || error("Scale must be positive")
    @compat new(Float64(μ), Float64(σ), Float64(ξ))
  end

  GeneralizedExtremeValue() = new(1.0, 1.0, 1.0)
end

# create support functions
hasfinitesupport(d::GeneralizedExtremeValue) = false
islowerbounded(d::GeneralizedExtremeValue) = (ξ = d.ξ; ξ > 0.0 ? true : false)
isupperbounded(d::GeneralizedExtremeValue) = (ξ = d.ξ; ξ < 0.0 ? true : false)
isbounded(d::GeneralizedExtremeValue) = islowerbounded(d) && isupperbounded(d)
minimum(d::GeneralizedExtremeValue) = (d.ξ > 0.0 ? d.μ - d.σ / d.ξ : -Inf)
maximum(d::GeneralizedExtremeValue) = (d.ξ < 0.0 ? d.μ - d.σ / d.ξ : Inf)
support(d::GeneralizedExtremeValue) = RealInterval(minimum(d), maximum(d))
insupport(d::GeneralizedExtremeValue, x::Real) = (minimum(d) <= x <= maximum(d))


#### Parameters

location(d::GeneralizedExtremeValue) = d.μ
scale(d::GeneralizedExtremeValue) = d.σ
shape(d::GeneralizedExtremeValue) = d.ξ
params(d::GeneralizedExtremeValue) = (d.μ, d.σ, d.ξ)

#### Statistics

function mean(d::GeneralizedExtremeValue)
  ξ = d.ξ
  if ξ != 0.0
    return d.μ + d.σ * 0.57721566490153286
  elseif ξ < 1.0
    return d.μ + d.σ * (gamma(1.0 - ξ) - 1.0) / ξ
  else
    return Inf
  end
end

function median(d::GeneralizedExtremeValue)
  ξ = d.ξ
  if ξ != 0.0
    return d.μ + d.σ * (log(2.0)^(-ξ) - 1.0) / ξ
  else
    return d.μ - d.σ * log(log(2.0))
  end
end

function mode(d::GeneralizedExtremeValue)
  ξ = d.ξ
  if ξ != 0.0
    return d.μ + d.σ * ((1.0 + ξ)^(-ξ) - 1.0) / ξ
  else
    return d.μ
  end
end

function var(d::GeneralizedExtremeValue)
  ξ = d.ξ
  if ξ == 0.0
    return d.σ^2.0 * 1.6449340668482264
  elseif ξ < 0.5
    return d.σ^2 * (gamma(1.0 - 2.0 * ξ) - (gamma(1.0 - ξ))^2.0) / ξ^2.0
  else
    return Inf
  end
end

function skewness(d::GeneralizedExtremeValue)

end

function kurtosis(d::GeneralizedExtremeValue)
  ξ = d.ξ
  if ξ == 0.0
    return 2.4
  elseif ξ < 0.25
    return
  else
    return Inf
  end
end

entropy(d::GeneralizedExtremeValue) = log(d.σ) + 0.57721566490153286 * d.ξ + 1.57721566490153286


#### Evaluation
zval(d::GeneralizedExtremeValue, x::Float64) = (x - d.μ) / d.σ
xval(d::GeneralizedExtremeValue, z::Float64) = z * d.σ + d.μ

function logpdf(d::GeneralizedExtremeValue, x::Float64)
  (μ, σ, ξ) = params(d)
  z = zval(d, x)
  if ξ == 0.0
    return -log(σ) - z - exp(-z)
  else
    if z * ξ < -1.0
      return -Inf
    else
      return -log(σ) - (ξ + 1.0) / ξ * (1.0 + z * ξ)
    end
  end
end

function pdf(d::GeneralizedExtremeValue, x::Float64)
  return exp(logpdf(d, x))
end

function logcdf(d::GeneralizedExtremeValue, x::Float64)
  ξ = d.ξ
  z = zval(d, x)
  if ξ == 0.0
    return exp(-z)
  else
    if z * ξ >= -1.0
      return -(1.0 + z * ξ)^(-1.0 / ξ)
    else
      return ξ < 0.0 ? 0.0 : -Inf
    end
  end
end
cdf(d::GeneralizedExtremeValue, x::Float64) = expm1(logcdf(d, x)) + 1.0
logccdf(d::GeneralizedExtremeValue, x::Float64) = log1p(-cdf(d, x))
ccdf(d::GeneralizedExtremeValue, x::Float64) = -expm1(logcdf(d, x))

function quantile(d::GeneralizedExtremeValue, p::Float64)
  ξ = d.ξ
  if ξ == 0.0
    return xval(d, -log(-log(p)))
  else
    return xval(d, ((-log(p))^(-ξ) - 1.0) / ξ)
  end
end

function cquantile(d::GeneralizedExtremeValue, p::Float64)
  ξ = d.ξ
  if ξ == 0.0
    return xval(d, -log(-log1p(-p)))
  else
    return xval(d, ((-log1p(-p))^(-ξ) - 1.0) / ξ)
  end
end

#### Sampling
function rand(d::GeneralizedExtremeValue)
  ξ = d.ξ
  if ξ == 0.0
    return xval(d, -log(randexp()))
  else
    return xval(d, (randexp()^(-ξ) - 1) / ξ)
  end
end
