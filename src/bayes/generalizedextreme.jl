using MetropolisUpdaters
using ExtremeValueDistributions

function mcmc_gev(y::Array{Float64};
                  Xμ::Array{Float64}=ones(y), Xσ::Array{Float64}=ones(y),
                  Xξ::Array{Float64}=ones(y),
                  βμsd::Real=100.0, βσsd::Real=100.0, βξsd::Real=1.0,
                  βμtune::Real=1.0, βσtune::Real=1.0, βξtune::Real=1.0,
                  iters::Integer=30000, burn::Integer=10000, thin::Integer=1,
                  verbose::Bool=false, report::Integer=1000)
  # arguments:
    # y: data
    # Xμ: covariates for μ
    # Xσ: covariates for σ
    # Xξ: covariates for ξ
    # βμsd: prior standard deviation for β parameters for μ
    # βσsd: prior standard deviation for β parameters for σ
    # βξsd: prior standard deviation for β parameters for ξ
    # βμtune: metropolis jump size for candidates βμ
    # βσtune: metropolis jump size for candidates βσ
    # βξtune: metropolis jump size for candidates βξ
    # iters: number of iterations to run the mcmc
    # burn: length of burnin period
    # thin: thinning length
    # verbose: do we want to print out periodic updates
    # report: how often to print out updates

  # returns:
    # (βμ, βσ, βξ): tuple of the posterior samples for the GEV parameters

  # model being fit:
    # y ~ GeneralizedExtremeValue(μ, σ, ξ)
    # μ = Xμ * βμ
    # σ = Xσ * βσ
    # ξ = Xξ * βξ

  ns = size(y, 1)
  nt = size(y, 2)
  npμ = nt == 1 ? size(Xμ, 2) : size(Xμ, 3)
  npσ = nt == 1 ? size(Xσ, 2) : size(Xσ, 3)
  npξ = nt == 1 ? size(Xξ, 2) : size(Xξ, 3)

  # parameters for the mcmc
  βμ = createmetropolis(npμ, prior=Distributions.Normal(0.0, βμsd), tune=βμtune)
  βσ = createmetropolis(npσ, prior=Distributions.Normal(0.0, βσsd), tune=βσtune)
  βξ = createmetropolis(npξ, prior=Distributions.Normal(0.0, βξsd), tune=βξtune)

  # functions update calculated values
  function updateXβ!(Xβ::CalculatedValues, X::Array{Float64, 2}, β::MetropolisVector)
    # there are two cases in which X only has 2 dimensions: np = 1 and nt = 1
    activevalue(Xβ)[:, :] = X[:, :] * activevalue(β)
  end


  function updatelly!(ll::CalculatedValuesMatrix, y::Array{Float64},
                      μ::CalculatedValues, σ::CalculatedValues, ξ::CalculatedValues)
    this_μ = activevalue(μ)
    this_σ = exp(activevalue(σ))
    this_ξ = activevalue(ξ)

    for j = 1:ll.ncols, i = 1:ll.nrows
      activevalue(ll)[i, j] = logpdf(GeneralizedExtremeValue(this_μ[i, j], this_σ[i, j], this_ξ[i, j]), y[i, j])
    end
  end

  # storage for calculated values
  Xβμ = createcalculatedvalues(ns, nt, updater=updateXβ!, requires=(Xμ, βμ))
  Xβσ = createcalculatedvalues(ns, nt, updater=updateXβ!, requires=(Xσ, βσ))
  Xβξ = createcalculatedvalues(ns, nt, updater=updateXβ!, requires=(Xξ, βξ))
  ll  = createcalculatedvalues(ns, nt, updater=updatelly!, requires=(y, Xβμ, Xβσ, Xβξ))

  # set impacts
  βμ.impacts = [Xβμ]
  βσ.impacts = [Xβσ]
  βξ.impacts = [Xβξ]

  # assign ll
  βμ.ll = [ll]
  βσ.ll = [ll]
  βξ.ll = [ll]

  # storage for posterior samples
  βμ_keep = fill(0.0, iters, βμ.length)
  βσ_keep = fill(0.0, iters, βσ.length)
  βξ_keep = fill(0.0, iters, βξ.length)

  for iter = 1:iters
    for ttt = 1:thin
      updatemhseq!(βμ)
      updatemhseq!(βσ)
      updatemhseq!(βξ)
    end  # end thin

    if iter < (burn / 2)
      updatestepsize!(βμ)
      updatestepsize!(βσ)
      updatestepsize!(βξ)
    end

    βμ_keep[iter, :] = βμ.cur
    βσ_keep[iter, :] = βσ.cur
    βξ_keep[iter, :] = βξ.cur

    if iter % report == 0 && verbose
      println("Iter: $iter")
    end
  end

  return(βμ_keep, βσ_keep, βξ_keep)
end

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
results = mcmc_gev(z, iters=10000, burn=8000, verbose=true, report=500)

using Gadfly
plot(x = 1:10000, y=results[1], Geom.line)
plot(x = 1:10000, y=exp(results[2]), Geom.line)
plot(x = 1:10000, y=results[3], Geom.line)
