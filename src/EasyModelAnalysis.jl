module EasyModelAnalysis

using Reexport
@reexport using DifferentialEquations
@reexport using ModelingToolkit
using Optimization, OptimizationBBO

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

export get_timeseries, get_min_t, get_max_t

end
