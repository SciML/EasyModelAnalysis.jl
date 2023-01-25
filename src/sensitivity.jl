function _get_sensitivity(prob, t, x, pbounds)
    boundvals = getfield.(pbounds, :second)
    boundkeys = getfield.(pbounds, :first)
    function f(p)
        prob = remake(prob; p = Pair.(boundkeys, p))
        sol = solve(prob, saveat = t)
        sol(t; idxs = x)
    end
    return GlobalSensitivity.gsa(f, Sobol(; order = [0, 1, 2]), boundvals;
                                 samples = 1000)
end

"""
    get_sensitivity(prob, t, x, pbounds)

Returns the sensitivity of the solution at time `t` and state `x` to the parameters in `pbounds`.
"""
function get_sensitivity(prob, t, x, pbounds)
    sensres = _get_sensitivity(prob, t, x, pbounds)
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
"""
function create_sensitivity_plot(prob, t, x, pbounds)
    sensres = _get_sensitivity(prob, t, x, pbounds)
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
