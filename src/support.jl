minimum(r::RealInterval) = r.lb
maximum(r::RealInterval) = r.ub
@compat in(x::Real, r::RealInterval) = (r.lb <= Float64(x) <= r.ub)

isbounded(d::UnivariateDistribution) = isupperbounded(d) && islowerbounded(d)
hasfinitesupport(d::DiscreteUnivariateDistribution) = isbounded(d)
hasfinitesupport(d::ContinuousUnivariateDistribution) = false

function insupport!{D<:UnivariateDistribution}(r::AbstractArray, d::Union(D,Type{D}), X::AbstractArray)
    length(r) == length(X) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    for i in 1 : length(X)
        @inbounds r[i] = insupport(d, X[i])
    end
    return r
end

insupport{D<:UnivariateDistribution}(d::Union(D,Type{D}), X::AbstractArray) =
     insupport!(BitArray(size(X)), d, X)

# This macro is copied directly from Distributions.jl, If we just import it from the Distributions package,
# it doesn't let us evaluate the macro using either GeneralizedExtremeValue or GeneralizedPareto.
macro distr_support(D, lb, ub)
    Dty = eval(D)
    @assert Dty <: UnivariateDistribution

    # determine whether is it upper & lower bounded
    D_is_lbounded = !(lb == :(-Inf))
    D_is_ubounded = !(ub == :Inf)
    D_is_bounded = D_is_lbounded && D_is_ubounded

    D_has_constantbounds = (isa(ub, Number) || ub == :Inf) &&
                           (isa(lb, Number) || lb == :(-Inf))

    paramdecl = D_has_constantbounds ? :(::Union($D, Type{$D})) : :(d::$D)

    insuppcomp = (D_is_lbounded && D_is_ubounded)  ? :(($lb) <= x <= $(ub)) :
                 (D_is_lbounded && !D_is_ubounded) ? :(x >= $(lb)) :
                 (!D_is_lbounded && D_is_ubounded) ? :(x <= $(ub)) : :true

    support_funs =

    support_funs = if Dty <: DiscreteUnivariateDistribution
        if D_is_bounded
            quote
                support($(paramdecl)) = round(Int, $lb):round(Int, $ub)
            end
        end
    else
        quote
            support($(paramdecl)) = RealInterval($lb, $ub)
        end
    end

    insupport_funs = if Dty <: DiscreteUnivariateDistribution
        quote
            insupport($(paramdecl), x::Real) = isinteger(x) && ($insuppcomp)
            insupport($(paramdecl), x::Integer) = $insuppcomp
        end
    else
        @assert Dty <: ContinuousUnivariateDistribution
        quote
            insupport($(paramdecl), x::Real) = $insuppcomp
        end
    end

    # overall
    esc(quote
        islowerbounded(::Union($D, Type{$D})) = $(D_is_lbounded)
        isupperbounded(::Union($D, Type{$D})) = $(D_is_ubounded)
        isbounded(::Union($D, Type{$D})) = $(D_is_bounded)
        minimum(d::$D) = $lb
        maximum(d::$D) = $ub
        $(support_funs)
        $(insupport_funs)
    end)
end