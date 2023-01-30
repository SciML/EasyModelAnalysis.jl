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
                out[i] = x(sol[i])
            end
        else
            for i in 1:size(p, 2)
                out[i] = sol[i](t; idxs = x)
            end
        end
        out
    end
    return GlobalSensitivity.gsa(f, Sobol(; order = [0, 1, 2]), boundvals; samples,
                                 batch = true)
end

"""
    get_sensitivity(prob, t, x, pbounds)

Returns the [Sobol Indices](https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis) that quanitfy the uncertainity of the solution at time `t` and observation `x` to the parameters in `pbounds`.


## Arguments
  - `t`: The time of observation, the solution is stored at this time to obtain the relevant observed variable.
  - `x`: The observation symbolic expression or a function that acts on the solution object.
  - `pbounds`: An array with the bounds for each parameter, passed as a pair of parameter expression and a vector with the upper and lower bound.

## Keyword Arguments
  - `samples`: Number of samples for running the global sensitivity analysis.

# Returns
  - A dictionary with the first, second and total order indices for all parameters (and pairs incase of second order).
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
            res_dict[Symbol(boundkeys[i], "_", boundkeys[j], "_second_order")] = sensres.S2[i,
                                                                                            j]
        end
    end
    return res_dict
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
    p3 = bar([paramnames[i] * "_" * paramnames[j] for i in eachindex(paramnames)
              for j in (i + 1):length(paramnames)],
             [sensres.S2[i, j] for i in eachindex(paramnames)
              for j in (i + 1):length(paramnames)],
             title = "Second Order Indices", legend = false)
    l = @layout [a b; c]
    plot(p2, p3, p1; layout = l, ylims = (0, 1))
end
