# Maximum Likelihood Estimation for GeneralizedPareto

using ExtremeValueDistributions
using Optim

function negloglikelihood(params::Vector; maxval::Float64=1.0e10, tol::Float64=1.0e-4)
  (σ, ξ) = params
  logσ = log(σ)
  if abs(logσ) > 20.0 return maxval end
  n = length(x)
  negll = float64(n) * logσ
  if abs(ξ) < tol  # if ξ ϵ (-tol, tol), use Gumbel limit of the GEV for stability
    for i = 1:n
      z = x[i] / σ
      #if z <= 0.0 return maxval end
      negll += z
    end
  else  # otherwise, GEV
    iξ = 1.0/ξ
    iξp1 = iξ + 1.0
    for i = 1:n
      ξzp1 = 1.0 + ξ * x[i] / σ
      if ξzp1 <= 0.0 return maxval end
      negll += iξp1 * log(ξzp1)
    end
  end
  return negll
end

function fit_mle(::Type{GeneralizedPareto}, x::Vector, init::Vector)
  μ = init[1]  # threshold must be fixed for GeneralizedPareto
  x = x[x .> μ]  # restrict to excesses over threshold
  opt = optimize(negloglikelihood, init[2:3])  # minimizing negative ll is equivalent to maximizing positive ll
  GeneralizedPareto([μ, opt.minimum]...)
end


## check if MLE method produces estimates close to the true value
parmlist = [[0.0, 1.0, ξ] for ξ = [-0.6:0.2:1.2]]
parm = parmlist[2]
init = [parm[1], parm[2] + (rand() - 0.5), parm[3] + rand() - 0.5]  # actual parm values wiggled w/ Uniform(-0.5, 0.5) error
d = GeneralizedPareto(parm...)
x = rand(d, 100)
fit = fit_mle(GeneralizedPareto, x, init)
params(fit)
println("Actual: $parm")
println("MLEs: $([round(params(fit)[i], 3) for i=1:length(params(fit))])")

