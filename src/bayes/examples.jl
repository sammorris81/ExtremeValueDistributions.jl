using ExtremeValueDistributions

# Simulated example - GEV Data
#   Z ~ GeneralizedExtremeValue(0.0, 1.0, 0.2)
# priors:
#   μ ~ Normal(0, 100)
#   σ ~ InvGamma(0.1, 0.1)
#   ξ ~ Normal(0, 0.5)
# generate covariate data and simulated observations
n = 1000
μₐ = 1.0
σₐ = 2.0
ξₐ = 0.1
z = reshape(rand(GeneralizedExtremeValue(μₐ, σₐ, ξₐ), n), n, 1)

# gives tuple of (βμ, βσ, βξ)
results = fit_mcmc(GeneralizedExtremeValue, z, iters=10000, burn=8000,
                   verbose=true, report=500)

using Gadfly
plot(x = 1:10000, y=results.βμpost, Geom.line)
plot(x = 1:10000, y=exp(results.βσpost), Geom.line)
plot(x = 1:10000, y=results.βξpost, Geom.line)


# Simulated example - GEV Data
#   Z ~ GeneralizedPareto(0.0, 1.0, 0.2)
# priors:
#   σ ~ InvGamma(0.1, 0.1)
#   ξ ~ Normal(0, 0.5)
# generate covariate data and simulated observations
n = 1000
μₐ = 1.0
σₐ = 2.0
ξₐ = 0.1
z = reshape(rand(GeneralizedPareto(μₐ, σₐ, ξₐ), n), n, 1)

# gives tuple of (βσ, βξ)
results = fit_mcmc(GeneralizedPareto, z, μ=1.0, iters=10000, burn=8000,
                   verbose=true, report=500)

using Gadfly
plot(x = 1:10000, y=exp(results.βσpost), Geom.line)
plot(x = 1:10000, y=results.βξpost, Geom.line)