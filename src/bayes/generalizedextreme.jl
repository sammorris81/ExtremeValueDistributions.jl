function mcmc_gev!(obj::GeneralizedExtremeValuePosterior, verbose::Bool, report::Integer)

  # storage for calculated values
  Xβμ = createcalculatedvalues(obj.ns, obj.nt, updater=updateXβ!, requires=(obj.Xμ, obj.βμ))
  Xβσ = createcalculatedvalues(obj.ns, obj.nt, updater=updateXβ!, requires=(obj.Xσ, obj.βσ))
  Xβξ = createcalculatedvalues(obj.ns, obj.nt, updater=updateXβ!, requires=(obj.Xξ, obj.βξ))
  ll  = createcalculatedvalues(obj.ns, obj.nt, updater=updatellgev!,
                               requires=(obj.y, Xβμ, Xβσ, Xβξ))

  # set impacts
  obj.βμ.impacts = [Xβμ]
  obj.βσ.impacts = [Xβσ]
  obj.βξ.impacts = [Xβξ]

  # assign ll
  obj.βμ.ll = [ll]
  obj.βσ.ll = [ll]
  obj.βξ.ll = [ll]

  for iter = 1:obj.iters
    for ttt = 1:obj.thin
      updatemhseq!(obj.βμ)
      updatemhseq!(obj.βσ)
      updatemhseq!(obj.βξ)
    end  # end thin

    if iter < (obj.burn / 2)
      updatestepsize!(obj.βμ)
      updatestepsize!(obj.βσ)
      updatestepsize!(obj.βξ)
    end

    obj.βμpost[iter, :] = obj.βμ.cur
    obj.βσpost[iter, :] = obj.βσ.cur
    obj.βξpost[iter, :] = obj.βξ.cur

    if iter % report == 0 && verbose
      println("Iter: $iter")
    end
  end
end
