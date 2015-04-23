function extremedata(dataset_name::String)
  filename = joinpath(Pkg.dir("ExtremeValueDistributions", "data"),
                      string(dataset_name, ".csv.gz"))

  if !isfile(filename)
    error(@sprintf "Unable to locate dataset %s in ExtremeValueDistributions\n" filename)
  else
    return readtable(filename)
  end

end

function coercematrix(X::Array{Float64}, n::Integer)
  @assert size(X, 1) == n
  if ndims(X) == 1
    X = reshape(X, n, 1)
  end
  return X
end
