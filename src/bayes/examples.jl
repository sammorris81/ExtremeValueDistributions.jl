using Distributions
using ExtremeValueDistributions

# Simulated example - GEV Data
#   Y ~ GeneralizedExtremeValue(0.0, 1.0, 0.2)
# priors:
#        μ ~ Normal(0, 100)
#   log(σ) ~ Normal(0, 100)
#        ξ ~ Normal(0, 1.0)
# generate covariate data and simulated observations
srand(1000)  # set seed
n = 1000
μₐ = 1.0
σₐ = 2.0
ξₐ = 0.1
y = reshape(rand(GeneralizedExtremeValue(μₐ, σₐ, ξₐ), n), n, 1)

# returns GeneralizedExtremeValuePosterior object
results = fit_mcmc(GeneralizedExtremeValue, y, iters=10000, burn=8000,
                   verbose=true, report=500)

using Gadfly
plot(x = 1:10000, y=results.βμpost, Geom.line)
plot(x = 1:10000, y=exp(results.βσpost), Geom.line)
plot(x = 1:10000, y=results.βξpost, Geom.line)

# Simulated example - GEV Data
#   Y ~ GeneralizedExtremeValue(μ, σ, ξ)
#        μ = 1.0 + 2.0 * N(0, 1)
#   log(σ) = 2.0 + 1.3 * N(0, 1)
#        ξ = 0.1
# priors:
#   βμ ~iid Normal(0, 100)
#   βσ ~iid Normal(0, 50)
#    ξ ~ Normal(0, 0.5)
srand(1000)  # set seed
n = 1000
X = hcat(ones(n), rand(Normal(0, 1), n))
βμₐ = [1.0, 2.0]
μₐ  = X * βμₐ
βσₐ = [2.0, 1.3]
σₐ  = exp(X * βσₐ)
βξₐ = 0.1
ξₐ  = 0.1
y = reshape([rand(GeneralizedExtremeValue(μₐ[i], σₐ[i], ξₐ), 1)[1] for i = 1:n], n, 1)

# to include covariates for μ, σ, or ξ, you need to include arguments
# Xμ, Xσ, and Xξ
# to change prior standard deviation, you need to include arguments
# βμsd, βσsd, or βξsd
# to update all β terms for a parameter in a block, you need to set
# βμseq = false, βσseq = false, βξseq = false
results = fit_mcmc(GeneralizedExtremeValue, y,
                   Xμ = X, Xσ = X, βμsd = 100.0, βσsd = 50.0, βξsd = 1.0,
                   βμseq = false, βσseq = false, βξseq = false,
                   iters=10000, burn=8000,
                   verbose=true, report=500)


using Gadfly
plot(x = 1:10000, y=results.βμpost[:, 1], Geom.line)
plot(x = 1:10000, y=results.βμpost[:, 2], Geom.line)
plot(x = 1:10000, y=results.βσpost[:, 1], Geom.line)
plot(x = 1:10000, y=results.βσpost[:, 2], Geom.line)
plot(x = 1:10000, y=results.βξpost, Geom.line)


# Simulated example - GPD Data
#   Y ~ GeneralizedPareto(0.0, 1.0, 0.2)
# priors:
#   σ ~ InvGamma(0.1, 0.1)
#   ξ ~ Normal(0, 0.5)
# generate covariate data and simulated observations
srand(1000)
n = 1000
μₐ = 1.0
σₐ = 2.0
ξₐ = 0.1
y = reshape(rand(GeneralizedPareto(μₐ, σₐ, ξₐ), n), n, 1)

# returns GeneralizedParetoPosterior object
results = fit_mcmc(GeneralizedPareto, y, μ=1.0, iters=10000, burn=8000,
                   verbose=true, report=500)

using Gadfly
plot(x = 1:10000, y=exp(results.βσpost), Geom.line)
plot(x = 1:10000, y=results.βξpost, Geom.line)

# Simulated example - GPD Data
#   Y ~ GeneralizedPareto(0.0, σ, ξ)
#   log(σ) = 2.0 + 1.3 * N(0, 1)
#        ξ = 0.1
# priors:
#   βσ ~iid Normal(0, 50)
#    ξ ~ Normal(0, 0.5)
srand(1000)
n = 1000
X = hcat(ones(n), rand(Normal(0, 1), n))
βσₐ = [2.0, 1.3]
σₐ  = exp(X * βσₐ)
βξₐ = 0.1
ξₐ  = 0.1
y = reshape([rand(GeneralizedPareto(0.0, σₐ[i], ξₐ), 1)[1] for i = 1:n], n, 1)


# to include covariates for σ, or ξ, you need to include arguments Xσ, and Xξ
# to change prior standard deviation, you need to include arguments βσsd, or βξsd
# to update all β terms for a parameter in a block, you need to set
# βσseq = false, βξseq = false
results = fit_mcmc(GeneralizedPareto, y,
                   Xσ = X, βσsd = 50.0, βξsd = 1.0,
                   βσseq = false, βξseq = false,
                   iters=10000, burn=8000,
                   verbose=true, report=500)


using Gadfly
plot(x = 1:10000, y=results.βσpost[:, 1], Geom.line)
plot(x = 1:10000, y=results.βσpost[:, 2], Geom.line)
plot(x = 1:10000, y=results.βξpost, Geom.line)

# Port Pirie data analysis
using DataFrames
using ExtremeValueDistributions
df = readtable(Pkg.dir()"/ExtremeValueDistributions/data/portpirie.csv")

y = reshape(array(df[:SeaLevel], 2), 65, 1)
results = fit_mcmc(GeneralizedExtremeValue, y,
                   iters = 20000, burn = 18000, verbose = true, report = 2000)
using Gadfly
plot(x = 1:20000, y = results.βμpost, Geom.line)
plot(x = 1:20000, y = exp(results.βσpost), Geom.line)
plot(x = 1:20000, y = results.βξpost, Geom.line)
