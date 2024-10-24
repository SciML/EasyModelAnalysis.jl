function _get_sensitivity(prob, t, x, pbounds; samples)
    boundvals = getfield.(pbounds, :second)
    boundkeys = getfield.(pbounds, :first)
    f = function (p)
        prob_func(prob, i, repeat) = remake(prob; p = Pair.(boundkeys, p[:, i]))
        ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
        sol = solve(ensemble_prob, nothing, EnsembleThreads(); saveat = t,
            trajectories = size(p, 2))
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
        out
    end
    return GlobalSensitivity.gsa(f, Sobol(; order = [0, 1, 2]), boundvals; samples,
        batch = true)
end

"""
    get_sensitivity(prob, t, x, pbounds)

Returns the [Sobol Indices](https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis) that quantify the uncertainty of the solution at time `t` and observation `x` to the parameters in `pbounds`.

## Arguments

  - `t`: The time of observation, the solution is stored at this time to obtain the relevant observed variable.
  - `x`: The observation symbolic expression or a function that acts on the solution object.
  - `pbounds`: An array with the bounds for each parameter, passed as a pair of parameter expression and a vector with the upper and lower bound.

## Keyword Arguments

  - `samples`: Number of samples for running the global sensitivity analysis.

# Returns

  - A dictionary with the first, second and total order indices for all parameters (and pairs in case of second order).
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
                j]
        end
    end
    return res_dict
end

"""
    get_sensitivity_of_maximum(prob, t, x, pbounds; samples = 1000)

Returns the [Sobol Indices](https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis) that
quantify the uncertainty of the solution at time `t` and maximum of observation `x` to the
parameters in `pbounds`.

## Arguments

  - `t`: The time of observation, the solution is stored at this time to obtain the relevant observed variable.
  - `x`: The observation symbolic expression.
  - `pbounds`: An array with the bounds for each parameter, passed as a pair of parameter expression and a vector with the upper and lower bound.

## Keyword Arguments

  - `samples`: Number of samples for running the global sensitivity analysis.

# Returns

  - A dictionary with the first, second and total order indices for all parameters (and pairs incase of second order).
"""
function get_sensitivity_of_maximum(prob, t, x, pbounds; samples = 1000)
    get_sensitivity(prob, t, sol -> get_max_t(sol, x)[2], pbounds, samples = samples)
end

"""
    create_sensitivity_plot(prob, t, x, pbounds)

Creates bar plots of the first, second and total order Sobol indices that quantify sensitivity of the solution
at time `t` and state `x` to the parameters in `pbounds`.

See also [`get_sensitivity`](@ref)
"""
function create_sensitivity_plot(prob, t, x, pbounds; samples = 1000)
    sensres = _get_sensitivity(prob, t, x, pbounds; samples)
    paramnames = String.(Symbol.(getfield.(pbounds, :first)))
    p1 = bar(paramnames, sensres.ST,
        title = "Total Order Indices", legend = false)
    p2 = bar(paramnames, sensres.S1,
        title = "First Order Indices", legend = false)
    p3 = bar(
        [paramnames[i] * "_" * paramnames[j] for i in eachindex(paramnames)
         for j in (i + 1):length(paramnames)],
        [sensres.S2[i, j] for i in eachindex(paramnames)
         for j in (i + 1):length(paramnames)],
        title = "Second Order Indices", legend = false)
    l = @layout [a b; c]
    plot(p2, p3, p1; layout = l, ylims = (0, 1))
end

"""
    create_sensitivity_plot(sensres, pbounds, total_only = false; kw...)

Creates bar plots of the first, second and total order Sobol indices from the
result of `get_sensitivity` and `pbounds`.

See also [`get_sensitivity`](@ref)
"""
function create_sensitivity_plot(sensres::Dict{Symbol}, pbounds, total_only = false; kw...)
    paramnames = String.(Symbol.(getfield.(pbounds, :first)))
    st = getindex.((sensres,), Symbol.(paramnames .* "_total_order"))
    idxs = sortperm(st, by = abs, rev = true)
    p1 = bar(paramnames[idxs], st[idxs];
        title = "Total Order Indices", legend = false, xrot = 90, kw...)
    total_only && return p1
    s1 = getindex.((sensres,), Symbol.(paramnames .* "_first_order"))
    idxs = sortperm(s1, by = abs, rev = true)
    p2 = bar(paramnames[idxs], s1[idxs];
        title = "First Order Indices", legend = false, xrot = 90, kw...)
    names = [paramnames[i] * "_" * paramnames[j] for i in eachindex(paramnames)
             for j in (i + 1):length(paramnames)]
    s2 = getindex.((sensres,), Symbol.(names, "_second_order"))
    idxs = sortperm(s2, by = abs, rev = true)
    p3 = bar(names[idxs], s2[idxs];
        title = "Second Order Indices", legend = false, xrot = 90, kw...)
    l = @layout [a b; c]
    plot(p2, p3, p1; layout = l, ylims = (0, 1))
end
