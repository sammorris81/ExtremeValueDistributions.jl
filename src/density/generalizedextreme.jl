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
islowerbounded(d::GeneralizedExtremeValue) = (d.ξ > 0.0 ? true : false)
isupperbounded(d::GeneralizedExtremeValue) = (d.ξ < 0.0 ? true : false)
isbounded(d::GeneralizedExtremeValue) = islowerbounded(d) && isupperbounded(d)
minimum(d::GeneralizedExtremeValue) = (ξ = d.ξ; ξ > 0.0 ? d.μ - d.σ / ξ : -Inf)
maximum(d::GeneralizedExtremeValue) = (ξ = d.ξ; ξ < 0.0 ? d.μ - d.σ / ξ : Inf)
support(d::GeneralizedExtremeValue) = RealInterval(minimum(d), maximum(d))
insupport(d::GeneralizedExtremeValue, x::Real) = (minimum(d) <= x <= maximum(d))

#### Parameters
location(d::GeneralizedExtremeValue) = d.μ
scale(d::GeneralizedExtremeValue) = d.σ
shape(d::GeneralizedExtremeValue) = d.ξ
params(d::GeneralizedExtremeValue) = (d.μ, d.σ, d.ξ)

#### Statistics
g(d::GeneralizedExtremeValue, k::Real) = gamma(1 - k * d.ξ)

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
    return d.σ^2.0 * (g(d, 2.0) - g(d, 1.0)^2.0) / ξ^2.0
  else
    return Inf
  end
end

function skewness(d::GeneralizedExtremeValue)
  ξ = d.ξ
  if ξ == 0.0
    # these are constant values from 12 * sqrt(6) zeta(3) / pi^3
    return 29.393876913398135 * 1.2020569031595951 / 31.006276680299816
  else
    g1 = g(d, 1)
    g2 = g(d, 2)
    g3 = g(d, 3)
    return sign(ξ) * (g3 - 3.0 * g1 * g2 + 2.0 * g1^3.0) / (g2 - g1^2.0)^(1.5)
  end
end

function kurtosis(d::GeneralizedExtremeValue)
  ξ = d.ξ
  if ξ == 0.0
    return 2.4
  elseif ξ < 0.25
    g1 = g(d, 1)
    g2 = g(d, 2)
    g3 = g(d, 3)
    g4 = g(d, 4)
    return (g4 - 4.0 * g1 * g3 + 6.0 * g2 * g1^2.0 - 3.0 * g1^4.0) / (g2 - g1^2.0)^2.0 - 3.0
  else
    return Inf
  end
end

# constant values are γ (Euler's constant)
entropy(d::GeneralizedExtremeValue) = log(d.σ) + 0.57721566490153286 * d.ξ + 1.57721566490153286


#### Evaluation
zval(d::GeneralizedExtremeValue, x::Float64) = (x - d.μ) / d.σ
xval(d::GeneralizedExtremeValue, z::Float64) = z * d.σ + d.μ

function logpdf(d::GeneralizedExtremeValue, x::Float64)
  (μ, σ, ξ) = params(d)
  if x == -Inf || x == Inf  # numerical stability to avoid NaN
    return -Inf
  else
    if insupport(d, x)
      z = zval(d, x)
      if ξ == 0.0
        return -log(σ) - z - exp(-z)
      else
        if z * ξ == -1.0  # numerical stability to avoid NaN
          return -Inf
        else
          t = (1.0 + z * ξ)^(-1.0 / ξ)
          return -log(σ) + (ξ + 1.0) * log(t) - t
        end
      end  # cases for ξ
    else  # insupport
      return -Inf
    end
  end  # evaluating at -Inf or Inf
end

function pdf(d::GeneralizedExtremeValue, x::Float64)
  return exp(logpdf(d, x))
end

function logcdf(d::GeneralizedExtremeValue, x::Float64)
  ξ = d.ξ
  if insupport(d, x)
    z = zval(d, x)
    if ξ == 0.0
      return -exp(-z)
    else
      return -(1.0 + z * ξ)^(-1.0 / ξ)
    end
  else
    return ξ < 0.0 ? 0.0 : -Inf # when ξ < 0, we are in this case when above max(d)
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
