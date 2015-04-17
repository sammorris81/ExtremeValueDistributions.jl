# Tests for GeneralizedPareto

using ExtremeValueDistributions
using Base.Test

d1 = GeneralizedPareto(0.0, 1.0, -1.0)  # ξ < 0.0
d2 = GeneralizedPareto(0.0, 1.0, 0.0)   # ξ = 0.0 and μ = 0.0 (reduces to Exponential)
d3 = GeneralizedPareto(1.0, 1.0, 0.0)   # ξ = 0.0 and μ != 0.0
d4 = GeneralizedPareto(0.0, 1.0, 1.0)   # ξ > 0.0 and μ != σ/ξ
d5 = GeneralizedPareto(1.0, 1.0, 1.0)   # ξ > 0.0 and μ = σ/ξ (reduces to Pareto)

dd = [d2, d3, d4, d5]

# Basics

for d in dd
  @test cdf(d, -Inf) == 0.0
  @test cdf(d, Inf) == 1.0
end


# random sampling

# pdfs, cdfs

