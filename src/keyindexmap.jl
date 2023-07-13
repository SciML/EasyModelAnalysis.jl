
struct IndexKeyMap
    indices::Vector{Int}
end

function IndexKeyMap(prob, keys)
    params = ModelingToolkit.parameters(prob.f.sys)
    indices = Vector{Int}(undef, length(keys))
    for i in eachindex(keys)
        indices[i] = findfirst(Base.Fix1(isequal, keys[i]), params)
    end
    return IndexKeyMap(indices)
end

Base.@propagate_inbounds function (ikm::IndexKeyMap)(prob, v::AbstractVector)
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
