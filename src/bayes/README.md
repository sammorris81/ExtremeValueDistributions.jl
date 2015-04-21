# ExtremeValueDistributions.jl

We have implemented a random walk metropolis hastings MCMC sampler to fit model parameters for the generalized extreme value distribution (GEV) and generalized Pareto distribution (GPD). We use an adaptive sampler that adjusts the standard deviation of the candidate distribution until the acceptance rate is between 0.25 and 0.50. The method `fit_mcmc()` is used to fit both types of distributions.

## Common interface

Let `y` be an *n* x *1* vector of responses. The method `fit_mcmc()` is used to fit the GEV or GPD distribution. By default `fit_mcmc(GeneralizedExtremeValue, y)` fits a GEV(μ, σ, ξ) distribution to the data, and `fit_mcmc(GeneralizedPareto, y)` fits a GPD(0.0, σ, ξ) distribution. Optional named arguments include:
* `Xμ`: matrix of covariates for μ (*GEV only*)
* `μ`: threshold value (*GPD only*)
* `Xσ`: matrix of covariates for σ
* `Xξ`: matrix of covariates for ξ
* `βμsd`: prior standard deviation for β parameters for μ (Default = 100.0, *GEV only*)
* `βσsd`: prior standard deviation for β parameters for σ (Default = 100.0)
* `βξsd`: prior standard deviation for β parameters for ξ (Default = 1.0)
* `βμtune`: starting metropolis jump size for candidates βμ (Default = 1.0, *GEV only*)
* `βσtune`: starting metropolis jump size for candidates βσ (Default = 1.0)
* `βξtune`: starting metropolis jump size for candidates βξ (Default = 1.0)
* `βμseq`: update β parameters for μ sequentially (true) or block (false) (*GEV only*)
* `βσseq`: update β parameters for σ sequentially (true) or block (false)
* `βξseq`: update β parameters for ξ sequentially (true) or block (false)
* `iters`: number of iterations to run the mcmc
* `burn`: length of burnin period
* `thin`: thinning length
* `verbose`: do we want to print out periodic updates
* `report`: how often to print out updates

The results from fitting the model using MCMC are of type `GeneralizedExtremeValuePosterior` or `GeneralizedParetoPosterior` depending on the type of distribution fit.

## Results

Let `results` be a type of `GeneralizedExtremeValuePosterior` or `GeneralizedParetoPosterior`.
The full list of fields is
* `results.y`: Response variable
* `results.ns`: Number of responses per day
* `results.nt`: Number of days
* `results.Xμ`: Covariates for fitting μ (*GEV only*)
* `results.Xσ`: Covariates for fitting σ
* `results.Xξ`: Covariates for fitting ξ
* `results.βμ`: MetropolisParameter type for regression coefficients for μ. (*GEV only*)
* `results.βσ`: MetropolisParameter type for regression coefficients for σ.
* `results.βξ`: MetropolisParameter type for regression coefficients for ξ.
* `results.βμpost`: Posterior samples for βμ (*GEV only*)
* `results.βσpost`: Posterior samples for βσ
* `results.βξpost`: Posterior samples for βξ
* `results.iters`: Number of iterations in the MCMC
* `results.burn`: Length of burnin period
* `results.thin`: How much thinning was used

### Posterior samples

Posterior samples are available as matrices in `results.βμpost`, `results.βσpost`, and `results.βξpost`. Each iteration is stored as a row in the matrix.

### MetropolisParameters

Three MetropolisParameter types `results.βμ`, `results.βσ`, and `results.βξ` are included in the results from the MCMC. This type is still under development, but we have included some basic documentation here. The following fields give information about the prior distributions used along with information about final candidate standard deviation and acceptance rates. Here are some of the more useful fields in the MetropolisParameter type.
* Post-burnin acceptance rates: `results.βμ.acc ./ results.βμ.att`
* Prior distribution: `results.βμ.prior`
* Sequential update: `results.βμ.seq`
