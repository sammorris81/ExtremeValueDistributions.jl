# Maximum Likelihood Estimation for GeneralizedExtremeValue

using ExtremeValueDistributions
using Optim

function fit_mle(::Type{GeneralizedExtremeValue}, x::Vector, init::Vector)
  function negloglikelihood(params::Vector; maxval::Float64=1.0e10, tol::Float64=1.0e-4)
  (μ, σ, ξ) = params
  logσ = log(σ)
  if abs(logσ) > 20.0 return maxval end
  n = length(x)
  negll = float64(n) * logσ
  if abs(ξ) < tol  # if ξ ϵ (-tol, tol), use Gumbel limit of the GEV for stability
    for i = 1:n
      z = (x[i] - μ) / σ
      if z <= 0.0 return maxval end
      negll += z + exp(-z)
    end
  else  # otherwise, GEV
    iξ = 1.0/ξ
    iξp1 = iξ + 1.0
    for i = 1:n
      ξzp1 = 1.0 + ξ * (x[i] - μ) / σ
      if ξzp1 <= 0.0 return maxval end
      negiξlogξzp1 = -iξ * log(ξzp1)
      if abs(negiξlogξzp1) > 20.0 return maxval end
      negll += iξp1 * log(ξzp1) + exp(negiξlogξzp1)
    end
  end
  return negll
end
  opt = optimize(negloglikelihood, init)  # minimizing negative ll is equivalent to maximizing positive ll
  GeneralizedExtremeValue(opt.minimum...)
end


## check if MLE method produces estimates close to the true value
parmlist = [[0.0, 1.0, ξ] for ξ = [-0.6:0.2:1.2]]
for parm in parmlist
  init = parm .+ (rand(3) .- 0.5)  # actual parm values wiggled w/ Uniform(-0.5, 0.5) error
  d = GeneralizedExtremeValue(parm...)
  x = rand(d, 100)
  fit = fit_mle(GeneralizedExtremeValue, x, init)
  params(fit)
  println("Actual: $parm")
  println("MLEs: $([round(params(fit)[i], 3) for i=1:length(params(fit))])")
end
