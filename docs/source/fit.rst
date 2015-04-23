Distribution Fitting
====================

Maximum Likelihood fitting for Extreme Value Distributions
----------------------------------------------------------

Maximum Likelihood Estimates (MLEs) of the model parameters for the generalized extreme value distribution (GEV) and generalized Pareto distribution (GPD) must be obtained via numerical optimization routines. The method ``fit_mle_optim()`` is fits both types of distributions by calling ``optimize`` from ``Optim.jl`` to minimize the negative log-likelihood function (i.e., maximized the log-likelihood) with respect to the parameters.

Common interface
~~~~~~~~~~~~~~~~

.. function:: fit_mle_optim()

Let ``y`` be an ``n`` x 1 vector of responses. The method ``fit_mle_optim()`` is used to fit the GEV or GPD distribution. By default ``fit_mle_optim(GeneralizedExtremeValue, y, init)`` fits a GEV (μ, σ, ξ) distribution to the data, and ``fit_mle_optim(GeneralizedPareto, y, init)`` fits a GPD (init[1], σ, ξ) distribution. Optional named arguments include:

* ``Xμ``: matrix of covariates for μ (Default = ``ones(y)``, *GEV only*)
* ``μ``: threshold value (Default = ``0.0``, *GPD only*)
* ``Xσ``: matrix of covariates for σ (Default = ``ones(y)``)
* ``Xξ``: matrix of covariates for ξ (Default = ``ones(y)``)
* ``verbose``: do we want to print out periodic updates (Default = ``false``)
* ``attempts``: number of times to vary initial conditions and attempt to maximize the likelihood (Default = ``10``)

Missing data
""""""""""""

When ``y`` is a ``DataFrame``, then the user can include ``NA`` values for ``fit_mle_optim()``. In the current version of the package, ``NA`` values are assumed to be missing at random and are removed from the dataset.

Results
"""""""

After iterating to convergence (or divergence) from ``attempts`` different initial values, ``fit_mle_optim`` returns the best maximizers of the log-likelihood, ``[βμ, βσ, βξ]``, where ``μ = Xμ * βμ``, ``logσ = Xσ * βσ``, and ``ξ = Xξ * βξ``.

Simulated Example: Generalized Extreme Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We generate the following generalized extreme value distribution to demonstrate the capabilities of ``fit_mle_optim()``. Let

.. math::

    Z \sim \text{GEV}(\mu, \sigma, \xi)

where

.. math::

  \mu &= 1\\
  \sigma &= 2 \\
  \xi &= 0.1 \\

.. code-block:: julia

  # generate data
  srand(1000)  # set seed
  n = 1000
  μ = 1.0
  σ = 2.0
  ξ = 0.1
  y = reshape(rand(GeneralizedExtremeValue(μ, σ, ξ), n), n, 1)

  # returns GeneralizedExtremeValue object with parameters as Max. Like. Estimates
  results = fit_mle_optim(GeneralizedExtremeValue, vec(y), [0.5, 0.5, 0.5])
  println("μ = $(results[1])")
  println("σ = $(exp(results[2]))")
  println("ξ = $(results[3])")


We can also allow for linear trends in the parameters of the GEV. Let

.. math::

  Z \sim \text{GEV}(\mu, \sigma, \xi)

where

.. math::

  \mu &= 1 + 2 X\\
  \log(\sigma) &= 2 + 1.3 * X\\
  \xi &= 0.1 \\
  X &~\sim N(0, 1) \\

.. code-block:: julia
  # generate the data
  using ExtremeValueDistributions
  using Distributions
  srand(100)
  n = 1000
  X = hcat(ones(n), rand(Normal(0, 1), n))
  βμ = [1.0, 2.0]
  μ  = X * βμ
  βσ = [2.0, 1.3]
  σ  = exp(X * βσ)
  ξ  = 0.1
  y = reshape([rand(GeneralizedExtremeValue(μ[i], σ[i], ξ), 1)[1] for i = 1:n], n, 1)

  # fit the model
  results = fit_mle_optim(GeneralizedExtremeValue, vec(y), [0.5, 0.5, 0.5], Xμ = X, Xσ = X)
  println(results)  # [βμ, βσ, βξ]


Simulated Example: Generalized Pareto Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We generate the following generalized Pareto distribution to demonstrate the capabilities of ``fit_mle_optim()``. Let

.. math::

    Z \sim \text{GPD}(0, \sigma, \xi)

where

.. math::

    \log(\sigma) &= 2 + 1.3x\\
    \xi &= 0.1 \\
    X &~\sim N(0, 1) \\

.. code-block:: julia

    # generate the data
    using ExtremeValueDistributions
    using Distributions
    srand(100)
    n = 1000
    X = hcat(ones(n), rand(Normal(0, 1), n))
    βσ = [2.0, 1.3]
    σ  = exp(X * βσ)
    ξ  = 0.1
    y = reshape([rand(GeneralizedExtremeValue(0.0, σ[i], ξ), 1)[1] for i = 1:n], n, 1)

    # fit the model
    results = fit_mle_optim(GeneralizedPareto, vec(y), [0.0, 0.5, 0.5], Xσ = X)
    println(results)  # [μ, βσ, βξ]


MCMC fitting for Extreme Value Distributions
--------------------------------------------

We have implemented a random walk metropolis hastings MCMC sampler to fit model parameters for the generalized extreme value distribution (GEV) and generalized Pareto distribution (GPD). We use an adaptive sampler that adjusts the standard deviation of the candidate distribution until the acceptance rate is between 0.25 and 0.50. The method ``fit_mcmc()`` is used to fit both types of distributions.

Common interface
~~~~~~~~~~~~~~~~

.. function:: fit_mcmc()

Let ``y`` be an ``n`` x 1 vector of responses. The method ``fit_mcmc()`` is used to fit the GEV or GPD distribution. By default ``fit_mcmc(GeneralizedExtremeValue, y)`` fits a GEV (μ, σ, ξ) distribution to the data, and ``fit_mcmc(GeneralizedPareto, y)`` fits a GPD (0.0, σ, ξ) distribution. Optional named arguments include:

* ``Xμ``: matrix of covariates for μ (Default = ``ones(y)``, *GEV only*)
* ``μ``: threshold value (Default = ``0.0``, *GPD only*)
* ``Xσ``: matrix of covariates for σ (Default = ``ones(y)``)
* ``Xξ``: matrix of covariates for ξ (Default = ``ones(y)``)
* ``βμsd``: prior standard deviation for β parameters for μ (Default = ``100.0``, *GEV only*)
* ``βσsd``: prior standard deviation for β parameters for σ (Default = ``100.0``)
* ``βξsd``: prior standard deviation for β parameters for ξ (Default = ``1.0``)
* ``βμtune``: starting metropolis jump size for candidates βμ (Default = ``1.0``, *GEV only*)
* ``βσtune``: starting metropolis jump size for candidates βσ (Default = ``1.0``)
* ``βξtune``: starting metropolis jump size for candidates βξ (Default = ``1.0``)
* ``βμseq``: update β parameters for μ sequentially (true) or block (false) (Default = ``true``, *GEV only*)
* ``βσseq``: update β parameters for σ sequentially (true) or block (false) (Default = ``true``)
* ``βξseq``: update β parameters for ξ sequentially (true) or block (false) (Default = ``true``)
* ``iters``: number of iterations to run the mcmc (Default = ``30000``)
* ``burn``: length of burnin period (Default = ``10000``)
* ``thin``: thinning length (Default = ``1``)
* ``verbose``: do we want to print out periodic updates (Default = ``false``)
* ``report``: how often to print out updates (Default = ``1000``)

The results from fitting the model using MCMC are of type ``GeneralizedExtremeValuePosterior`` or ``GeneralizedParetoPosterior`` depending on the type of distribution fit.

Missing data
""""""""""""

When ``y`` is a ``DataFrame``, then the user can include ``NA`` values for ``fit_mcmc()``. In the current version of the package, ``NA`` values are assumed to be missing at random and are removed from the dataset.

Results
~~~~~~~

Let ``results`` be a type of ``GeneralizedExtremeValuePosterior`` or ``GeneralizedParetoPosterior``. The full list of available fields is

* ``results.y``: Response variable
* ``results.ns``: Number of responses per day
* ``results.nt``: Number of days
* ``results.Xμ``: Covariates for fitting μ (*GEV only*)
* ``results.Xσ``: Covariates for fitting σ
* ``results.Xξ``: Covariates for fitting ξ
* ``results.βμ``: ``MetropolisParameter`` type for regression coefficients for μ. (*GEV only*)
* ``results.βσ``: ``MetropolisParameter`` type for regression coefficients for σ.
* ``results.βξ``: ``MetropolisParameter`` type for regression coefficients for ξ.
* ``results.βμpost``: Posterior samples for βμ (*GEV only*)
* ``results.βσpost``: Posterior samples for βσ
* ``results.βξpost``: Posterior samples for βξ
* ``results.iters``: Number of iterations in the MCMC
* ``results.burn``: Length of burnin period
* ``results.thin``: How much thinning was used

Posterior samples
"""""""""""""""""

Posterior samples are available as matrices in ``results.βμpost``, ``results.βσpost``, and ``results.βξpost``. Each iteration is stored as a row in the matrix.

MetropolisParameters
""""""""""""""""""""

The following three results fields are ``MetropolisParameter`` types: a) ``results.βμ``, b) ``results.βσ``, and c) ``results.βξ``. This type is still under development, but we have included some basic documentation here. The following fields give information about the prior distributions used along with information about final candidate standard deviation and acceptance rates. Here are some of the more useful fields in the ``MetropolisParameter`` type.

* Post-burnin acceptance rates: ``results.βμ.acc ./ results.βμ.att``
* Prior distribution: ``results.βμ.prior``
* Sequential update: ``results.βμ.seq``

Simulated Example: Generalized Extreme Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We generate the following generalized extreme value distribution to demonstrate the capabilities of ``fit_mcmc()``. Let

.. math::

    Z \sim \text{GEV}(\mu, \sigma, \xi)

where

.. math::

    \mu &= 1 + 2 x\\
    \log(\sigma) &= 2 + 1.3x\\
    \xi &= 0.1 \\
    X &~\sim N(0, 1) \\

.. code-block:: julia

    # generate the data
    using ExtremeValueDistributions
    using Distributions
    srand(100)
    n = 1000
    X = hcat(ones(n), rand(Normal(0, 1), n))
    βμ = [1.0, 2.0]
    μ  = X * βμ
    βσ = [2.0, 1.3]
    σ  = exp(X * βσ)
    ξ  = 0.1
    y = reshape([rand(GeneralizedExtremeValue(μ[i], σ[i], ξ), 1)[1] for i = 1:n], n, 1)

    # fit the model
    results = fit_mcmc(GeneralizedExtremeValue, y,
                       Xμ = X, Xσ = X, βμsd = 100.0, βσsd = 50.0, βξsd = 1.0,
                       βμseq = false, βσseq = false, βξseq = false,
                       iters=10000, burn=8000,
                       verbose=true, report=500)

    # plot the posterior distribution
    using Gadfly
    plot(x = 1:10000, y=results.βμpost[:, 1], Geom.line)
    plot(x = 1:10000, y=results.βμpost[:, 2], Geom.line)
    plot(x = 1:10000, y=results.βσpost[:, 1], Geom.line)
    plot(x = 1:10000, y=results.βσpost[:, 2], Geom.line)
    plot(x = 1:10000, y=results.βξpost, Geom.line)


Simulated Example: Generalized Pareto Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We generate the following generalized Pareto distribution to demonstrate the capabilities of ``fit_mcmc()``. Let

.. math::

    Z \sim \text{GPD}(0, \sigma, \xi)

where

.. math::

    \log(\sigma) &= 2 + 1.3x\\
    \xi &= 0.1 \\
    X &~\sim N(0, 1) \\

.. code-block:: julia

    # generate the data
    using ExtremeValueDistributions
    using Distributions
    srand(100)
    n = 1000
    X = hcat(ones(n), rand(Normal(0, 1), n))
    βσ = [2.0, 1.3]
    σ  = exp(X * βσ)
    ξ  = 0.1
    y = reshape([rand(GeneralizedExtremeValue(0.0, σ[i], ξ), 1)[1] for i = 1:n], n, 1)

    # fit the model
    results = fit_mcmc(GeneralizedPareto, y, 0.0,
                       Xσ = X, βσsd = 50.0, βξsd = 1.0,
                       βσseq = false, βξseq = false,
                       iters=10000, burn=8000,
                       verbose=true, report=500)

    # plot the posterior distribution
    using Gadfly
    plot(x = 1:10000, y=results.βσpost[:, 1], Geom.line)
    plot(x = 1:10000, y=results.βσpost[:, 2], Geom.line)
    plot(x = 1:10000, y=results.βξpost, Geom.line)

Data analysis
-------------

Port Pirie sea level data
~~~~~~~~~~~~~~~~~~~~~~~~~

The dataset ``portpirie`` consists of annual maximum sea levels (in meters) from Port Pirie, South Australia, from 1928 to 1987. This dataset comes from the ``evdbayes`` package in ``R``. Data can be loaded into ``Julia`` using ``extremedata("portpirie")``.

MLE data analysis
"""""""""""""""""

.. code-block:: julia

    # import the data
    using ExtremeValueDistributions
    df = extremedata("portpirie")
    results = fit_mle_optim(GeneralizedExtremeValue, df[:SeaLevel], [0.5, 0.5, 0.5])
    println("μ = $(results[1])")
    println("σ = $(exp(results[2]))")
    println("ξ = $(results[3])")

MCMC data analysis
""""""""""""""""""

We illustrate how to fit the ``portpirie`` dataset using a generalized extreme value distribution. The data are fit using 20000 iterations with 18000 burnin.

.. code-block:: julia

  # import the data
  using ExtremeValueDistributions
  df = extremedata("portpirie")
  results = fit_mcmc(GeneralizedExtremeValue, df[:SeaLevel],
                     iters = 20000, burn = 18000, verbose = true, report = 2000)

  # plot the posterior distributions
  using Gadfly
  plot(x = 1:20000, y = results.βμpost, Geom.line)
  plot(x = 1:20000, y = exp(results.βσpost), Geom.line)
  plot(x = 1:20000, y = results.βξpost, Geom.line)

Rainfall analysis
~~~~~~~~~~~~~~~~~

The dataset ``rainfall`` contains 20820 daily rainfall observations (in mm) recorded at a rain gauge in England over 57 years. Three of the years contain only ``NA`` values, and of the remaining observations 54, are ``NA`` values. This dataset comes from the ``evdbayes`` package in ``R``.

MLE data analysis
"""""""""""""""""

.. code-block:: julia

    # import the data
    using ExtremeValueDistributions
    df = extremedata("rainfall")
    results = fit_mle_optim(GeneralizedPareto, df[:rainfall], [40.0, 0.0, 0.0])
    println("μ = $(results[1])")  # threshold fixed by user
    println("σ = $(exp(results[2]))")
    println("ξ = $(results[3])")

MCMC data analysis
""""""""""""""""""

We illustrate how to fit the ``rainfall`` dataset using a generalized Pareto distribution with a threshold set at 40mm. The data are fit using 20000 iterations with 18000 burnin.

.. code-block:: julia

    # import the data
    using ExtremeValueDistributions
    df = extremedata("rainfall")
    results = fit_mcmc(GeneralizedPareto, df[:rainfall], 40.0, iters = 20000, burn = 18000,
                       verbose = true, report = 1000)

    # plot the posterior distributions
    using Gadfly
    plot(x = 1:20000, y = exp(results.βσpost), Geom.line)
    plot(x = 1:20000, y = results.βξpost, Geom.line)

