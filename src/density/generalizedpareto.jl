immutable GeneralizedPareto <: ContinuousUnivariateDistribution
  μ::Float64
  σ::Float64
  ξ::Float64

  function GeneralizedPareto(μ::Real, σ::Real, ξ::Real)
    σ > zero(σ) || error("Scale must be positive")
    @compat new(Float64(μ), Float64(σ), Float64(ξ))
  end

  GeneralizedPareto() = new(1.0, 1.0, 1.0)
end

@distr_support(GeneralizedPareto, d.μ, (ξ = d.ξ; ξ < 0.0 ? d.μ - d.σ/ξ : Inf)

#### Parameters

location(d::GeneralizedPareto) = d.μ
scale(d::GeneralizedPareto) = d.σ
shape(d::GeneralizedPareto) = d.ξ

params(d::GeneralizedPareto) = (d.μ, d.σ, d.ξ)


#### Statistics

mean(d::GeneralizedPareto) = (ξ = d.ξ; ξ < 1.0 ? d.μ + d.σ / (1.0 - ξ) : Inf)

median(d::GeneralizedPareto) = (ξ = d.ξ; d.μ + (d.σ * (2.0^ξ - 1.0)) / ξ)

mode(d::GeneralizedPareto) = NaN

var(d::GeneralizedPareto) = (ξ = d.ξ; ξ < 0.5 ? d.σ^2 / ((1.0 - ξ)^2 * (1.0 - 2.0ξ)) : Inf)

function skewness(d::GeneralizedPareto)
  ξ = shape(d)
  3.0ξ < 1.0 ? (2.0 * (1.0 + ξ) * sqrt(1.0 - 2.0ξ)) / (1.0 - 3.0ξ) : NaN
end

function kurtosis(d::GeneralizedPareto)
  ξ = shape(d)
  ξ < 0.25 ? (3.0 * (1.0 - 2.0ξ) * (2.0ξ^2 + ξ + 3.0)) / ((1.0 - 3.0ξ) * (1.0 - 4.0ξ)) - 3.0 : NaN
end

entropy(d::GeneralizedPareto) = NaN


#### Evaluation

function pdf(d::GeneralizedPareto, x::Float64)
  exp(logpdf(d, x))
end

function logpdf(d::GeneralizedPareto, x::Float64)
  (μ, σ, ξ) = params(d)
  if ξ < 0.0
    ξz = ξ * (x - μ) / σ
    (ξz >= 0.0 && ξz <= -1.0) ? -log(σ) - (1.0/ξ + 1.0) * log1p(ξz) : -Inf
  elseif ξ == 0.0
    if μ == 0.0
      logpdf(Exponential(σ), x)
    else
      z = (x - μ) / σ
      z >= 0.0 ? -log(σ) - z : -Inf
    end
  else  # ξ > 0.0
    if (iξ = 1.0/ξ; σiξ = σ*iξ; μ == σiξ)
      logpdf(Pareto(iξ, σiξ), x)
    else
      ξz = (x - μ) / σiξ
      ξz >= 0.0 ? -log(σ) - (iξ + 1.0) * log1p(ξz) : -Inf
    end
  end
end

function ccdf(d::GeneralizedPareto, x::Float64)
  exp(logccdf(d, x))
end

cdf(d::GeneralizedPareto, x::Float64) = 1.0 - ccdf(d, x)

function logccdf(d::GeneralizedPareto, x::Float64)
  (μ, σ, ξ) = params(d)
  if ξ < 0.0
     ξz = ξ * (x - μ) / σ
    (ξz >= 0.0 && ξz <= -1.0) ? -1.0/ξ * log1p(ξz) : 0.0
  elseif ξ == 0.0
    if μ == 0.0
      logccdf(Exponential(σ), x)
    else
      z = (x - μ) / σ
      z >= 0.0 ? -z : 0.0
    end
  else  # ξ > 0.0
    if (iξ = 1.0/ξ; σiξ = σ*iξ; μ == σiξ)
      logccdf(Pareto(iξ, σiξ), x)
    else
      ξz = (x - μ) / σiξ
      ξz >= 0.0 ? -iξ * log1p(ξz) : 0.0
    end
  end
end

logcdf(d::GeneralizedPareto, x::Float64) = log1p(-ccdf(d, x))

function cquantile(d::GeneralizedPareto, p::Float64)
  ξ = shape(d)
  if ξ == 0.0
    d.μ - d.σ * log(p)
  else
    d.μ + (d.σ * expm1(-ξ * log(p))) / ξ
  end
end

expm1(-ξ * log(p))
quantile(d::GeneralizedPareto, p::Float64) = cquantile(d, 1.0 - p)


#### Sampling

function rand(d::GeneralizedPareto)
  ξ = shape(d)
  if ξ == 0.0
    d.μ + d.σ * randexp()
  else
    d.μ + (d.σ * expm1(ξ * randexp())) / ξ
  end
end
