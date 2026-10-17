function _get_sensitivity(prob, t, x, pbounds; samples)
    boundvals = getfield.(pbounds, :second)
    boundkeys = getfield.(pbounds, :first)
    f = function (p)
        prob_func(prob, i::Integer, repeat) = remake(prob; p = Pair.(boundkeys, p[:, i]))
        prob_func(prob, ctx) = remake(prob; p = Pair.(boundkeys, p[:, ctx.sim_id]))
        # `prob_func` returns a remade problem and never mutates its argument, so
        # the per-trajectory safety copy is unnecessary. It is also broken here:
        # remaking the copy that `safetycopy` makes of a ModelingToolkit problem
        # throws in MTK's initialization/tearing path, while the original is fine.
        ensemble_prob = EnsembleProblem(
            prob, prob_func = prob_func, safetycopy = false
        )
        sol = solve(
            ensemble_prob, nothing, EnsembleThreads(); saveat = t,
            trajectories = size(p, 2)
        )
        out = zeros(size(p, 2))
        if x isa Function
            for i in 1:size(p, 2)
                out[i] = x(sol.u[i])
            end
        else
            for i in 1:size(p, 2)
                out[i] = sol.u[i](t; idxs = x)
            end
        end
        return out
    end
    return GlobalSensitivity.gsa(
        f, Sobol(; order = [0, 1, 2]), boundvals; samples,
        batch = true
    )
end

"""
    get_sensitivity(prob, t, x, pbounds; samples = 1000) -> Dict{Symbol, Float64}

Return [Sobol indices](https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis)
that quantify how parameter bounds in `pbounds` affect observation `x` at time `t`.

# Arguments

  - `prob`: a SciML problem to solve for each sampled parameter vector.
  - `t`: time of observation; each solution is saved at this time to obtain `x`.
  - `x`: The observation symbolic expression or a function that acts on the solution object.
  - `pbounds`: parameter-expression-to-`[lower, upper]` bound pairs.

# Keywords

  - `samples::Integer = 1000`: number of Sobol samples.

# Returns

  - A dictionary containing first-, second-, and total-order indices keyed by parameter
    name.

# Examples

```julia
indices = get_sensitivity(prob, 10.0, x, [k => [0.8, 1.2]]; samples = 1_000)
```
"""
function get_sensitivity(prob, t, x, pbounds; samples = 1000)
    sensres = _get_sensitivity(prob, t, x, pbounds; samples)
    boundvals = getfield.(pbounds, :second)
    boundkeys = getfield.(pbounds, :first)
    res_dict = Dict{Symbol, Float64}()
    for i in eachindex(boundkeys)
        res_dict[Symbol(boundkeys[i], "_first_order")] = sensres.S1[i]
        res_dict[Symbol(boundkeys[i], "_total_order")] = sensres.ST[i]
    end
    for i in eachindex(boundkeys)
        for j in (i + 1):length(boundkeys)
            res_dict[Symbol(boundkeys[i], "_", boundkeys[j], "_second_order")] = sensres.S2[
                i,
                j,
            ]
        end
    end
    return res_dict
end

"""
    get_sensitivity_of_maximum(prob, t, x, pbounds; samples = 1000)

Return Sobol indices for the maximum value of `x` over each sampled solution.

# Arguments

  - `prob`: a SciML problem to solve for each sampled parameter vector.
  - `t`: saved time(s) used when solving each sampled problem.
  - `x`: the symbolic observation whose maximum is analyzed.
  - `pbounds`: parameter-to-`[lower, upper]` bound pairs.

# Keywords

  - `samples::Integer = 1000`: number of Sobol samples.

# Returns

  - A dictionary containing first-, second-, and total-order indices keyed by parameter
    name.

# Examples

```julia
indices = get_sensitivity_of_maximum(prob, 0.0:0.1:10.0, x, [k => [0.8, 1.2]])
```
"""
function get_sensitivity_of_maximum(prob, t, x, pbounds; samples = 1000)
    return get_sensitivity(prob, t, sol -> get_max_t(sol, x)[2], pbounds, samples = samples)
end

"""
    create_sensitivity_plot(prob, t, x, pbounds; samples = 1000) -> Plots.Plot

Create bar plots of first-, second-, and total-order Sobol indices for the solution at time
`t` and state `x`.

# Arguments

  - `prob`: a SciML problem to solve for each sampled parameter vector.
  - `t`: saved time(s) used when solving each sampled problem.
  - `x`: the symbolic observation to analyze.
  - `pbounds`: parameter-to-`[lower, upper]` bound pairs.

# Keywords

  - `samples::Integer = 1000`: number of Sobol samples.

# Returns

  - A three-panel plot of total-, first-, and second-order Sobol indices.

# Examples

```julia
p = create_sensitivity_plot(prob, 10.0, x, [k => [0.8, 1.2]])
```

See also [`get_sensitivity`](@ref).
"""
function create_sensitivity_plot(prob, t, x, pbounds; samples = 1000)
    sensres = _get_sensitivity(prob, t, x, pbounds; samples)
    paramnames = String.(Symbol.(getfield.(pbounds, :first)))
    p1 = bar(
        paramnames, sensres.ST,
        title = "Total Order Indices", legend = false
    )
    p2 = bar(
        paramnames, sensres.S1,
        title = "First Order Indices", legend = false
    )
    p3 = bar(
        [
            paramnames[i] * "_" * paramnames[j] for i in eachindex(paramnames)
                for j in (i + 1):length(paramnames)
        ],
        [
            sensres.S2[i, j] for i in eachindex(paramnames)
                for j in (i + 1):length(paramnames)
        ],
        title = "Second Order Indices", legend = false
    )
    l = @layout [a b; c]
    return plot(p2, p3, p1; layout = l, ylims = (0, 1))
end

"""
    create_sensitivity_plot(sensres, pbounds, total_only = false; kw...) -> Plots.Plot

Creates bar plots of the first, second and total order Sobol indices from the
result of `get_sensitivity` and `pbounds`.

# Arguments

  - `sensres::Dict{Symbol}`: result returned by [`get_sensitivity`](@ref).
  - `pbounds`: parameter-to-`[lower, upper]` bound pairs used for `sensres`.
  - `total_only::Bool = false`: return only total-order indices when `true`.

# Keywords

  - `kw...`: keyword arguments forwarded to the bar plots.

# Returns

  - A total-order bar plot, or a three-panel Sobol-index plot.

# Examples

```julia
p = create_sensitivity_plot(indices, [k => [0.8, 1.2]]; total_only = true)
```

See also [`get_sensitivity`](@ref).
"""
function create_sensitivity_plot(sensres::Dict{Symbol}, pbounds, total_only = false; kw...)
    paramnames = String.(Symbol.(getfield.(pbounds, :first)))
    st = getindex.((sensres,), Symbol.(paramnames .* "_total_order"))
    idxs = sortperm(st, by = abs, rev = true)
    p1 = bar(
        paramnames[idxs], st[idxs];
        title = "Total Order Indices", legend = false, xrot = 90, kw...
    )
    total_only && return p1
    s1 = getindex.((sensres,), Symbol.(paramnames .* "_first_order"))
    idxs = sortperm(s1, by = abs, rev = true)
    p2 = bar(
        paramnames[idxs], s1[idxs];
        title = "First Order Indices", legend = false, xrot = 90, kw...
    )
    names = [
        paramnames[i] * "_" * paramnames[j] for i in eachindex(paramnames)
            for j in (i + 1):length(paramnames)
    ]
    s2 = getindex.((sensres,), Symbol.(names, "_second_order"))
    idxs = sortperm(s2, by = abs, rev = true)
    p3 = bar(
        names[idxs], s2[idxs];
        title = "Second Order Indices", legend = false, xrot = 90, kw...
    )
    l = @layout [a b; c]
    return plot(p2, p3, p1; layout = l, ylims = (0, 1))
end
