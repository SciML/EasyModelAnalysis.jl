
struct IndexKeyMap
    indices::Vector{Int}
end

# probs support
function IndexKeyMap(prob, keys)
    params = ModelingToolkit.parameters(prob.f.sys)
    indices = Vector{Int}(undef, length(keys))
    for i in eachindex(keys)
        indices[i] = findfirst(Base.Fix1(isequal, keys[i]), params)
    end
    return IndexKeyMap(indices)
end

Base.@propagate_inbounds function (ikm::IndexKeyMap)(prob::SciMLBase.AbstractDEProblem,
    v::AbstractVector)
    @boundscheck checkbounds(v, length(ikm.indices))
    def = prob.p
    ret = Vector{Base.promote_eltype(v, def)}(undef, length(def))
    copyto!(ret, def)
    for (i, j) in enumerate(ikm.indices)
        @inbounds ret[j] = v[i]
    end
    return ret
end
function _remake(prob, tspan, ikm::IndexKeyMap, pprior)
    p = ikm(prob, pprior)
    remake(prob; tspan, p)
end

# data support
function IndexKeyMap(prob, data::AbstractVector{<:Pair})
    states = ModelingToolkit.states(prob.f.sys)
    indices = Vector{Int}(undef, length(data))
    for i in eachindex(data)
        indices[i] = findfirst(Base.Fix1(isequal, data[i].first), states)
    end
    return IndexKeyMap(indices)
end
function (ikm::IndexKeyMap)(sol::SciMLBase.AbstractTimeseriesSolution)
    (@view(sol[i, :]) for i in ikm.indices)
end
