using Distributions
using ExtremeValueDistributions

# Simulated example - GEV Data
#   Y ~ GeneralizedExtremeValue(0.0, 1.0, 0.2)
# generate covariate data and simulated observations
srand(1000)  # set seed
n = 1000
μₐ = 1.0
σₐ = 2.0
ξₐ = 0.1
y = reshape(rand(GeneralizedExtremeValue(μₐ, σₐ, ξₐ), n), n, 1)

# returns GeneralizedExtremeValue object with parameters as Max. Like. Estimates
results = fit_mle_optim(GeneralizedExtremeValue, y, [0.0, 0.0, 0.0])


# Simulated example - GEV Data
#   Y ~ GeneralizedExtremeValue(μ, σ, ξ)
#        μ = 1.0 + 2.0 * N(0, 1)
#   log(σ) = 2.0 + 1.3 * N(0, 1)
#        ξ = 0.1
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
results = fit_mle_optim(GeneralizedExtremeValue, vec(y), [0.5, 0.5, 0.5], Xμ = X, Xσ = X)

# Simulated example - GPD Data
#   Y ~ GeneralizedPareto(0.0, 1.0, 0.2)
# generate covariate data and simulated observations
using ExtremeValueDistributions
srand(1000)
n = 1000
μₐ = 1.0
σₐ = 2.0
ξₐ = 0.1
y = reshape(rand(GeneralizedPareto(μₐ, σₐ, ξₐ), n), n, 1)
μ = fill(1.0, size(y, 1))

# returns GeneralizedParetoPosterior object
results = fit_mle_optim(GeneralizedPareto, vec(y), [1.0, 0.5, 0.5])

# Simulated example - GPD Data
#   Y ~ GeneralizedPareto(0.0, σ, ξ)
#   log(σ) = 2.0 + 1.3 * N(0, 1)
#        ξ = 0.1
using ExtremeValueDistributions
using Distributions
srand(1000)
n = 1000
X = hcat(ones(n), rand(Normal(0, 1), n))
βσₐ = [2.0, 1.3]
σₐ  = exp(X * βσₐ)
βξₐ = 0.1
ξₐ  = 0.1
y = reshape([rand(GeneralizedPareto(0.0, σₐ[i], ξₐ), 1)[1] for i = 1:n], n, 1)


# to include covariates for σ, or ξ, you need to include arguments Xσ, and Xξ
results = fit_mle_optim(GeneralizedPareto, vec(y), [0.0, 1.0, 0.1], Xσ = X)


# Port Pirie data analysis
using ExtremeValueDistributions
df = extremedata("portpirie")
results = fit_mle_optim(GeneralizedExtremeValue, df[:SeaLevel], [0.5, 0.5, 0.5])

# Rainfall analysis
using ExtremeValueDistributions
df = extremedata("rainfall")
results = fit_mle_optim(GeneralizedPareto, df[:rainfall], [40.0, 0.0, 0.0])
