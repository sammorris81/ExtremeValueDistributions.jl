import Distributions

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

@distr_support GeneralizedExtremeValue -Inf Inf

#### Parameters

location(d::GeneralizedExtremeValue) = d.μ
scale(d::GeneralizedExtremeValue) = d.σ
shape(d::GeneralizedExtremeValue) = d.ξ
params(d::GeneralizedExtremeValue) = (d.μ, d.σ, d.ξ)

#### Statistics
# TODO: Add in cases

#### Evaluation



