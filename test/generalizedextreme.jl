# Tests for GeneralizedPareto

using Distributions
using ExtremeValueDistributions
using Base.Test

d1 = GeneralizedExtremeValue(0.0, 1.0, 1.1)  # μ = 0.0, σ = 1.0, ξ = 1.1 (infinite mean)
d2 = GeneralizedExtremeValue(0.0, 1.0, 0.6)  # μ = 0.0, σ = 1.0, ξ = 0.6 (infinite var)
d3 = GeneralizedExtremeValue(0.0, 1.0, 0.3)  # μ = 0.0, σ = 1.0, ξ = 0.3 (infinite kurtosis)
d4 = GeneralizedExtremeValue(0.0, 1.0, 0.1)  # μ = 0.0, σ = 1.0, ξ = 0.1 (bounded below)
d5 = GeneralizedExtremeValue(0.0, 1.0, 0.0)  # μ = 0.0, σ = 1.0, ξ = 0.0 (unbounded)
d6 = GeneralizedExtremeValue(0.0, 1.0, -0.3)  # μ = 0.0, σ = 1.0, ξ = -0.3 (bounded above)

dd = [d1, d2, d3, d4, d5, d6]

# pdfs, cdfs
for d in dd
  min_d = minimum(d) == -Inf ? -999999 : minimum(d)
  max_d = maximum(d) == Inf ? 999999 : maximum(d)
  x_insupport = rand(Uniform(min_d, max_d), 1)[1]
  ϵ = 1e-5

  # cdf
  @test cdf(d, maximum(d)) == 1.0
  @test cdf(d, x_insupport) <= 1.0 && cdf(d, x_insupport) >= 0.0
  @test cdf(d, minimum(d)) == 0.0
  if islowerbounded(d)
    @test cdf(d, min_d - ϵ) == 0.0
  elseif isupperbounded(d)
    @test cdf(d, max_d + ϵ) == 1.0
  end

  # pdf
  @test pdf(d, maximum(d)) == 0.0
  @test pdf(d, x_insupport) >= 0.0
  @test pdf(d, minimum(d)) == 0.0
  if islowerbounded(d)
    @test pdf(d, min_d - ϵ) == 0.0
  elseif isupperbounded(d)
    @test pdf(d, max_d + ϵ) == 0.0
  end

  # random sampling
  @test all(bool([insupport(d, x) for x = rand(d, 10000)]))
end
