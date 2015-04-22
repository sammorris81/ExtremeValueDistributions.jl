function mcmc_gpd!(obj::GeneralizedParetoPosterior, verbose::Bool, report::Integer)

  # storage for calculated values
  Xβσ = createcalculatedvalues(obj.n, updater=updateXβ!, requires=(obj.Xσ, obj.βσ))
  Xβξ = createcalculatedvalues(obj.n, updater=updateXβ!, requires=(obj.Xξ, obj.βξ))
  ll  = createcalculatedvalues(obj.n, updater=updatellgpd!,
                               requires=(obj.y, obj.μ, Xβσ, Xβξ))

  # set impacts
  obj.βσ.impacts = [Xβσ]
  obj.βξ.impacts = [Xβξ]

  # assign ll
  obj.βσ.ll = [ll]
  obj.βξ.ll = [ll]

  for iter = 1:obj.iters
    for ttt = 1:obj.thin
      updatemhseq!(obj.βσ)
      updatemhseq!(obj.βξ)
    end  # end thin

    if iter < (obj.burn / 2)
      updatestepsize!(obj.βσ)
      updatestepsize!(obj.βξ)
    end

    obj.βσpost[iter, :] = obj.βσ.cur
    obj.βξpost[iter, :] = obj.βξ.cur

    if iter % report == 0 && verbose
      println("Iter: $iter")
    end
  end
end
