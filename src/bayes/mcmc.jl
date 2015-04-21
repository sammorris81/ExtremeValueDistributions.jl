abstract ExtremePosterior

type GeneralizedExtremeValuePosterior <: ExtremePosterior
  y::Array{Float64}        # response variable
  ns::Integer              # number of responses per day
  nt::Integer              # number of days
  Xμ::Array{Float64}       # covariates for fitting μ
  Xσ::Array{Float64}       # covariates for fitting σ
  Xξ::Array{Float64}       # covariates for fitting ξ
  βμ::MetropolisParameter  # metropolis parameters for βμ
  βσ::MetropolisParameter  # metropolis parameters for βσ
  βξ::MetropolisParameter  # metropolis parameters for βξ
  βμpost::Array{Float64}   # posterior samples for βμ
  βσpost::Array{Float64}   # posterior samples for βσ
  βξpost::Array{Float64}   # posterior samples for βξ

  iters::Integer           # length of MCMC chain
  burn::Integer            # length of burnin period
  thin::Integer            # how much thinning

  GeneralizedExtremeValuePosterior() = new()
end

type GeneralizedParetoPosterior <: ExtremePosterior
  y::Array{Float64}        # response variable
  ns::Integer              # number of responses per day
  nt::Integer              # number of days
  μ::Real                  # GPD location parameter
  Xσ::Array{Float64}       # covariates for fitting σ
  Xξ::Array{Float64}       # covariates for fitting ξ
  βσ::MetropolisParameter  # metropolis parameters for βσ
  βξ::MetropolisParameter  # metropolis parameters for βξ
  βσpost::Array{Float64}   # posterior samples for βσ
  βξpost::Array{Float64}   # posterior samples for βξ

  iters::Integer           # length of MCMC chain
  burn::Integer            # length of burnin period
  thin::Integer            # how much thinning

  GeneralizedParetoPosterior() = new()
end

function fit_mcmc(::Type{GeneralizedExtremeValue}, y::Array{Float64};
                  Xμ::Array{Float64}=ones(y), Xσ::Array{Float64}=ones(y),
                  Xξ::Array{Float64}=ones(y),
                  βμsd::Real=100.0, βσsd::Real=100.0, βξsd::Real=1.0,
                  βμtune::Real=1.0, βσtune::Real=1.0, βξtune::Real=1.0,
                  βμseq::Bool=true, βσseq::Bool=true, βξseq::Bool=true,
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
    # βμseq: update β parameters for μ sequentially (true) or block (false)
    # βσseq: update β parameters for σ sequentially (true) or block (false)
    # βξseq: update β parameters for ξ sequentially (true) or block (false)
    # iters: number of iterations to run the mcmc
    # burn: length of burnin period
    # thin: thinning length
    # verbose: do we want to print out periodic updates
    # report: how often to print out updates

  # returns:
    # gevfit: GeneralizedExtremeValuePosterior type with posterior samples

  # model being fit:
    # y ~ GeneralizedExtremeValue(μ, σ, ξ)
    # μ = Xμ * βμ
    # σ = Xσ * βσ
    # ξ = Xξ * βξ

  gevfit = GeneralizedExtremeValuePosterior()

  gevfit.ns = size(y, 1)
  gevfit.nt = size(y, 2)
  npμ = gevfit.nt == 1 ? size(Xμ, 2) : size(Xμ, 3)
  npσ = gevfit.nt == 1 ? size(Xσ, 2) : size(Xσ, 3)
  npξ = gevfit.nt == 1 ? size(Xξ, 2) : size(Xξ, 3)

  # data and covariates for MCMC
  gevfit.y  = y
  gevfit.Xμ = Xμ
  gevfit.Xσ = Xσ
  gevfit.Xξ = Xξ

  # parameters for the mcmc
  gevfit.βμ = createmetropolis(npμ, prior=Distributions.Normal(0.0, βμsd),
                               tune=βμtune, seq=βμseq)
  gevfit.βσ = createmetropolis(npσ, prior=Distributions.Normal(0.0, βσsd),
                               tune=βσtune, seq=βσseq)
  gevfit.βξ = createmetropolis(npξ, prior=Distributions.Normal(0.0, βξsd),
                               tune=βξtune, seq=βξseq)

  # storage for the posterior samples
  gevfit.βμpost = fill(0.0, iters, gevfit.βμ.length)
  gevfit.βσpost = fill(0.0, iters, gevfit.βσ.length)
  gevfit.βξpost = fill(0.0, iters, gevfit.βξ.length)

  # chain description
  gevfit.iters = iters
  gevfit.burn  = burn
  gevfit.thin  = thin

  # get samples
  mcmc_gev!(gevfit, verbose, report)

  return gevfit
end


function fit_mcmc(::Type{GeneralizedPareto}, y::Array{Float64};
                  μ::Real=0.0, Xσ::Array{Float64}=ones(y),
                  Xξ::Array{Float64}=ones(y),
                  βσsd::Real=100.0, βξsd::Real=1.0,
                  βσtune::Real=1.0, βξtune::Real=1.0,
                  βσseq::Bool=true, βξseq::Bool=true,
                  iters::Integer=30000, burn::Integer=10000, thin::Integer=1,
                  verbose::Bool=false, report::Integer=1000)
  # arguments:
    # y: data
    # Xσ: covariates for σ
    # Xξ: covariates for ξ
    # βσsd: prior standard deviation for β parameters for σ
    # βξsd: prior standard deviation for β parameters for ξ
    # βσtune: metropolis jump size for candidates βσ
    # βξtune: metropolis jump size for candidates βξ
    # βσseq: update β parameters for σ sequentially (true) or block (false)
    # βξseq: update β parameters for ξ sequentially (true) or block (false)
    # iters: number of iterations to run the mcmc
    # burn: length of burnin period
    # thin: thinning length
    # verbose: do we want to print out periodic updates
    # report: how often to print out updates

  # returns:
    # gevfit: GeneralizedExtremeValuePosterior type with posterior samples

  # model being fit:
    # y ~ GeneralizedPareto(μ, σ, ξ)
    # σ = Xσ * βσ
    # ξ = Xξ * βξ

  gpdfit = GeneralizedParetoPosterior()

  gpdfit.ns = size(y, 1)
  gpdfit.nt = size(y, 2)
  npσ = gpdfit.nt == 1 ? size(Xσ, 2) : size(Xσ, 3)
  npξ = gpdfit.nt == 1 ? size(Xξ, 2) : size(Xξ, 3)

  # data and covariates for MCMC
  gpdfit.y  = y
  gpdfit.μ  = μ
  gpdfit.Xσ = Xσ
  gpdfit.Xξ = Xξ

  # parameters for the mcmc
  gpdfit.βσ = createmetropolis(npσ, prior=Distributions.Normal(0.0, βσsd),
                               tune=βσtune, seq=βσseq)
  gpdfit.βξ = createmetropolis(npξ, prior=Distributions.Normal(0.0, βξsd),
                               tune=βξtune, seq=βξseq)

  # storage for the posterior samples
  gpdfit.βσpost = fill(0.0, iters, gpdfit.βσ.length)
  gpdfit.βξpost = fill(0.0, iters, gpdfit.βξ.length)

  # chain description
  gpdfit.iters = iters
  gpdfit.burn  = burn
  gpdfit.thin  = thin

  mcmc_gpd!(gpdfit, verbose, report)

  return gpdfit
end


# functions update calculated values
function updateXβ!(Xβ::CalculatedValues, X::Array{Float64, 2}, β::MetropolisVector)
  # there are two cases in which X only has 2 dimensions: np = 1 and nt = 1
  activevalue(Xβ)[:, :] = X[:, :] * activevalue(β)
end

function updatellgev!(ll::CalculatedValuesMatrix, y::Array{Float64},
                      μ::CalculatedValues, σ::CalculatedValues, ξ::CalculatedValues)
  this_μ = activevalue(μ)
  this_σ = exp(activevalue(σ))
  this_ξ = activevalue(ξ)

  for j = 1:ll.ncols, i = 1:ll.nrows
    activevalue(ll)[i, j] = logpdf(GeneralizedExtremeValue(this_μ[i, j], this_σ[i, j], this_ξ[i, j]), y[i, j])
  end
end

function updatellgpd!(ll::CalculatedValuesMatrix, y::Array{Float64},
                      μ::Real, σ::CalculatedValues, ξ::CalculatedValues)
  this_σ = exp(activevalue(σ))
  this_ξ = activevalue(ξ)

  for j = 1:ll.ncols, i = 1:ll.nrows
    activevalue(ll)[i, j] = logpdf(GeneralizedPareto(μ, this_σ[i, j], this_ξ[i, j]), y[i, j])
  end
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
