function fit_mle_optim(::Type{GeneralizedExtremeValue}, y::Array{Float64}, init::Vector;
                       Xμ::Array{Float64}=ones(reshape(y, length(y), 1)), Xσ::Array{Float64}=ones(reshape(y, length(y), 1)),
                       Xξ::Array{Float64}=ones(reshape(y, length(y), 1)), attempts::Int64=10, verbose::Bool=true)
  ## must first define function to minimize
  function negloglikelihood(params::Vector; maxval::Float64=1.0e10, tol::Float64=1.0e-3)
    βμ = params[1:pμ]
    βσ = params[(pμ + 1):(pμ + pσ)]
    βξ = params[(pμ + pσ + 1):end]
    μ = Xμ * βμ
    logσ = Xσ * βσ
    ξ = Xξ * βξ
   if any(abs(logσ) .> 100.0) return maxval end
    n = length(y)
    z = zeros(n)
    iξ = ones(n)
    iξp1 = ones(n)
    ξzp1 = ones(n)
    negiξlogξzp1 = ones(n)
    negll = sum(logσ)
    σ = exp(logσ)
    for i = 1:n
      if abs(ξ[i]) < tol  # if ξ ϵ (-tol, tol), use Gumbel limit of the GEV for stability
        z[i] = (y[i] - μ[i]) / σ[i]
        if z[i] <= 0.0 return maxval end
        negll += z[i] + exp(-z[i])
      else  # otherwise, use standard GEV likelihood
        iξ[i] = 1.0/ξ[i]
        iξp1[i] = iξ[i] + 1.0
        ξzp1[i] = 1.0 + ξ[i] * (y[i] - μ[i]) / σ[i]
        if ξzp1[i] <= 0.0 return maxval end
        negiξlogξzp1[i] = -iξ[i] * log(ξzp1[i])
        if abs(negiξlogξzp1[i]) > 100.0 return maxval end
        negll += iξp1[i] * log(ξzp1[i]) + exp(negiξlogξzp1[i])
      end
    end
    return negll
  end
  pμ, pσ, pξ = [size(X, 2) for X in (Xμ, Xσ, Xξ)]
  βμ = zeros(pμ)
  βσ = zeros(pσ)
  βξ = zeros(pξ)
  βμ[1], βσ[1], βξ[1] = init
  init = [βμ, βσ, βξ]  # redefine init in terms of βμ, βσ, βξ
  inits = Array(Vector{Float64}, attempts)
  opts = Array(Any, attempts)
  for i in 1:attempts  # minimize log-likelihood over varying initial conditions
    if verbose println("Minimizing negative log-likelihood: attempt $i of $attempts") end
    inits[i] = init + 2.0 * rand(length(init)) - 1.0
    opts[i] = optimize(negloglikelihood, inits[i])  # minimizing negative ll is equivalent to maximizing positive ll
  end
  opts = opts[bool([opt.f_converged for opt in opts])]  # remove opts that failed to converge
  if isempty(opts)  # no MLEs obtained
    println("Failed to converge in $attempts attempts around initial values $init; returning initial values")
    return init
  else  # otherwise, return best maximizers of log-like (equiv, best minimizers of negative log-like)
    opt = opts[findmin([opt.f_minimum for opt in opts])[2]]
    return opt.minimum
  end
end

function fit_mle_optim(::Type{GeneralizedExtremeValue}, y::DataArray{Float64}, init::Vector{Float64};
                       Xμ::DataArray{Float64} = ones(reshape(y, length(y), 1)), Xσ::DataArray{Float64} = ones(reshape(y, length(y), 1)),
                       Xξ::DataArray{Float64} = ones(reshape(y, length(y), 1)), attempts::Int64=10, verbose::Bool=true)
  # remove NAs
  these = find(!isna(y))
  if length(these) != size(y, 1)
    if verbose println("Keeping $(length(these)) out of $(length(y)) observations. Remaining observations removed due to NA") end
    y = y[these]
    Xμ = size(Xμ, 2) == 1 ? Xμ[these] : Xμ[these, :]
    Xσ = size(Xσ, 2) == 1 ? Xσ[these] : Xσ[these, :]
    Xξ = size(Xξ, 2) == 1 ? Xξ[these] : Xξ[these, :]
  end

  # make sure that n matches for y, Xμ, Xσ, and Xξ
  @assert size(y, 1) == size(Xμ, 1)
  @assert size(y, 1) == size(Xσ, 1)
  @assert size(y, 1) == size(Xξ, 1)

  # basic functionality for dataframes
  return fit_mle_optim(GeneralizedExtremeValue, array(y), init; Xμ = array(Xμ),
                       Xσ = array(Xσ), Xξ = array(Xξ), attempts = attempts, verbose = verbose)
end



function fit_mle_optim(::Type{GeneralizedPareto}, y::Vector, init::Vector;
                       Xσ::Array{Float64}=ones(reshape(y, length(y), 1)), Xξ::Array{Float64}=ones(reshape(y, length(y), 1)),
                       attempts::Int64=10, verbose::Bool=true)
  function negloglikelihood(params::Vector; maxval::Float64=1.0e10, tol::Float64=1.0e-4)
    (σ, ξ) = params
    βσ = params[1:pσ]
    βξ = params[(pσ + 1):end]
    logσ = Xσ * βσ
    ξ = Xξ * βξ
    if any(abs(logσ) .> 100.0) return maxval end
    n = length(y)
    z = zeros(n)
    ξzp1 = ones(n)
    negll = sum(logσ)
    σ = exp(logσ)
    for i = 1:n
      if abs(ξ[i]) < tol  # if ξ ϵ (-tol, tol), use Gumbel limit of the GEV for stability
        z[i] = y[i] / σ[i]
        if z[i] < tol return maxval end
        negll += z[i]
      else  # otherwise, use standard GEV likelihood
        ξzp1[i] = 1.0 + ξ[i] * y[i] / σ[i]
        if ξzp1[i] <= 0.0 return maxval end
        negll += (1/ξ[i] + 1.0) * log(ξzp1[i])
      end
    end
    return negll
  end
  μ = init[1]  # threshold must be fixed for GeneralizedPareto
  excesses = y .> μ
  y = y[excesses]  # restrict to excesses over threshold
  Xσ = Xσ[excesses, :]
  Xξ = Xξ[excesses, :]
  pσ, pξ = [size(X, 2) for X in (Xσ, Xξ)]
  βσ = zeros(pσ)
  βξ = zeros(pξ)
  βσ[1], βξ[1] = init[2:3]
  init = [βσ, βξ]  # redefine init in terms of βσ, βξ
  inits = Array(Vector{Float64}, attempts)
  opts = Array(Any, attempts)
  for i in 1:attempts  # minimize log-likelihood over varying initial conditions
    if verbose println("Minimizing negative log-likelihood: attempt $i of $attempts") end
    inits[i] = init + 2.0 * rand(length(init)) - 1.0
    opts[i] = optimize(negloglikelihood, inits[i])  # minimizing negative ll is equivalent to maximizing positive ll
  end
  opts = opts[bool([opt.f_converged for opt in opts])]  # remove opts that failed to converge
  if isempty(opts)  # no MLEs obtained
    println("Failed to converge in $attempts attempts around initial values $init; returning initial values")
    return [μ, init]
  else  # otherwise, return best maximizers of log-like (equiv, best minimizers of negative log-like)
    opt = opts[findmin([opt.f_minimum for opt in opts])[2]]
    return [μ, opt.minimum]
  end
end

function fit_mle_optim(::Type{GeneralizedPareto}, y::DataArray{Float64}, init::Vector{Float64};
                       Xσ::DataArray{Float64} = ones(reshape(y, length(y), 1)), Xξ::DataArray{Float64} = ones(reshape(y, length(y), 1)),
                       attempts::Int64=10, verbose::Bool=true)
  # remove NAs
  these = find(!isna(y))
  if length(these) != size(y, 1)
    if verbose println("Keeping $(length(these)) out of $(length(y)) observations. Remaining observations removed due to NA") end
    y = y[these]
    Xσ = size(Xσ, 2) == 1 ? Xσ[these] : Xσ[these, :]
    Xξ = size(Xξ, 2) == 1 ? Xξ[these] : Xξ[these, :]
  end

  # make sure that n matches for y, Xσ, and Xξ
  @assert size(y, 1) == size(Xσ, 1)
  @assert size(y, 1) == size(Xξ, 1)

  # basic functionality for dataframes
  return fit_mle_optim(GeneralizedPareto, array(y), init; Xσ = array(Xσ), Xξ = array(Xξ), attempts = attempts, verbose = verbose)
end
