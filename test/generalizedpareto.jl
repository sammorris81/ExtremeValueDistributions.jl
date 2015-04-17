# Tests for GeneralizedPareto

using ExtremeValueDistributions
using Base.Test

d1 = GeneralizedPareto(0.0, 1.0, -1.0)  # ξ < 0.0 and μ = 0.0
d2 = GeneralizedPareto(1.0, 1.0, -1.0)  # ξ < 0.0 and μ = 1.0
d3 = GeneralizedPareto(0.0, 1.0, 0.0)   # ξ = 0.0 and μ = 0.0 (reduces to Exponential)
d4 = GeneralizedPareto(1.0, 1.0, 0.0)   # ξ = 0.0 and μ != 0.0
d5 = GeneralizedPareto(0.0, 1.0, 1.0)   # ξ > 0.0 and μ != σ/ξ
d6 = GeneralizedPareto(1.0, 1.0, 1.0)   # ξ > 0.0 and μ = σ/ξ (reduces to Pareto)

dd = [d1, d2, d3, d4, d5, d6]

for d in dd
  # check cumulative function
  @test cdf(d, -Inf) == 0.0
  @test cdf(d, minimum(d)) == 0.0
  @test 0.0 <= cdf(d, rand(d)) <= 1.0
  @test cdf(d, maximum(d)) == 1.0
  @test cdf(d, Inf) == 1.0

  # check pdf over the support
  ϵ = 1e-5
  @test pdf(d, minimum(d) - ϵ) == 0.0  # below lowerbound
  @test pdf(d, minimum(d)) == (d.ξ < 0.0 ? 1/d.σ : 1.0)        # at lowerbound
  @test pdf(d, maximum(d)) == 0.0        # at upperbound
  @test pdf(d, maximum(d) + ϵ) == 0.0  # above upperbound

  # randomly sampled values remain in support?
  @test all(bool([insupport(d, x) for x = rand(d, 10000)]))
end

