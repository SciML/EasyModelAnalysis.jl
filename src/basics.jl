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

Returns `(t,min)` where `t` is the timepoint where `sym` reaches its minimum `min` in the interval `prob.tspan`.
"""
function get_min_t(prob, sym)
    sol = solve(prob)
    f(t, _) = sol(t[1]; idxs = sym)
    oprob = OptimizationProblem(f, [(prob.tspan[2] - prob.tspan[1]) / 2],
                                lb = [prob.tspan[1]],
                                ub = [prob.tspan[end]])
    res = solve(oprob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = 10000)
    res.u[1], f(res.u[1])
end

"""
    get_max_t(prob, sym)

Returns `(t,max)` where `t` is the timepoint where `sym` reaches its maximum `max` in the interval `prob.tspan`.
"""
function get_max_t(prob, sym)
    sol = solve(prob)
    f(t, _) = -sol(t[1]; idxs = sym)
    oprob = OptimizationProblem(f, [(prob.tspan[2] - prob.tspan[1]) / 2],
                                lb = [prob.tspan[1]],
                                ub = [prob.tspan[end]])
    res = solve(oprob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = 10000)
    res.u[1], -f(res.u[1])
end

"""
    plot_extrema(prob, sym)

Plots the solution of the observable `sym` along with showcasing time time points where it obtains its maximum and minimum values.
"""
function plot_extrema(prob, sym)
    xmin, xminval = get_min_t(prob, sym)
    xmax, xmaxval = get_max_t(prob, sym)
    plot(sol, idxs = x)
    scatter!([xmin], [xminval])
    scatter!([xmax], [xmaxval])
end

"""
    phaseplot_extrema(prob, sym, plotsyms)

Plots the phase plot solution of the observable `sym` along with showcasing time time points where it 
obtains its maximum and minimum values. `plotsyms` should be given as the tuple of symbols for the
observables that define the axis of the phase plot.
"""
function phaseplot_extrema(prob, sym, plotsyms)
    sol = solve(prob)
    xmin, xminval = get_min_t(prob, sym)
    xmax, xmaxval = get_max_t(prob, sym)
    plot(sol, idxs = plotsyms)
    scatter!([[sol(xmin; idxs = plotsyms[i])] for i in plotsyms]...)
    scatter!([[sol(xmax; idxs = plotsyms[i])] for i in plotsyms]...)
end
