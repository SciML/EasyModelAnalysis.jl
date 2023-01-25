module EasyModelAnalysis

using Reexport
@reexport using DifferentialEquations
@reexport using ModelingToolkit
using Optimization, OptimizationBBO
using GlobalSensitivity

"""
    get_timeseries(prob, sym, t)

Get the time-series of state `sym` evaluated at times `t`.
"""
function get_timeseries(prob, sym, t)
    prob = remake(prob, tspan = (min(prob.tspan[1], t[1]), max(prob.tspan[2], t[end])))
    sol = solve(prob, saveat = t)
    sol[sym]
end

"""
    get_min_t(prob, sym)

Returns the minimum of state `sym` in the interval `prob.tspan`.
"""
function get_min_t(prob, sym)
    sol = solve(prob)
    f(t, _) = sol(t[1]; idxs = sym)
    oprob = OptimizationProblem(f, [(prob.tspan[2] - prob.tspan[1]) / 2],
                                lb = [prob.tspan[1]],
                                ub = [prob.tspan[end]])
    res = solve(oprob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = 10000)
    res.u[1]
end

"""
    get_max_t(prob, sym)

Returns the maximum of state `sym` in the interval `prob.tspan`.
"""
function get_max_t(prob, sym)
    sol = solve(prob)
    f(t, _) = -sol(t[1]; idxs = sym)
    oprob = OptimizationProblem(f, [(prob.tspan[2] - prob.tspan[1]) / 2],
                                lb = [prob.tspan[1]],
                                ub = [prob.tspan[end]])
    res = solve(oprob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = 10000)
    res.u[1]
end

"""
    get_sensitivity(prob, t, x, pbounds)

Returns the sensitivity of the solution at time `t` and state `x` to the parameters in `pbounds`.
"""
function get_sensitivity(prob, t, x, pbounds)
    boundvals = getfield.(pbounds, :second)
    boundkeys = getfield.(pbounds, :first)
    function f(p)
        prob = remake(prob; p = Pair.(boundkeys, p))
        sol = solve(prob, saveat = t)
        sol(t; idxs = x)
    end
    sensres = GlobalSensitivity.gsa(f, Sobol(; order = [0, 1, 2]), boundvals;
                                    samples = 1000)
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

export get_timeseries, get_min_t, get_max_t, get_sensitivity

end
