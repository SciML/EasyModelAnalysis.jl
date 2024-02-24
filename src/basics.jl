"""
    get_timeseries(prob, sym, t)

Get the time-series of state `sym` evaluated at times `t`.
"""
function get_timeseries(prob, sym, t)
    @assert t[1] >= prob.tspan[1]
    prob = remake(prob, tspan = (prob.tspan[1], min(prob.tspan[2], t[end])))
    sol = solve(prob, saveat = t)
    sol[sym]
end

"""
    get_min_t(prob, sym)
    get_min_t(sol, sym)

Returns `(t,min)` where `t` is the timepoint where `sym` reaches its minimum `min` in the interval `prob.tspan`.
"""
function get_min_t(prob, sym)
    if prob isa ODESolution
        sol = prob
        prob = sol.prob
    else
        sol = solve(prob)
    end
    f(t, _) = sol(t[1]; idxs = sym)
    oprob = OptimizationProblem(f, [(prob.tspan[2] - prob.tspan[1]) / 2],
        lb = [prob.tspan[1]],
        ub = [prob.tspan[end]])
    res = solve(oprob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = 10000)
    res.u[1], f(res.u[1], nothing)
end

"""
    get_max_t(prob, sym)
    get_max_t(sol, sym)

Returns `(t,max)` where `t` is the timepoint where `sym` reaches its maximum `max` in the interval `prob.tspan`.
"""
function get_max_t(prob, sym)
    if prob isa ODESolution
        sol = prob
        prob = sol.prob
    else
        sol = solve(prob)
    end
    f(t, _) = -sol(t[1]; idxs = sym)
    oprob = OptimizationProblem(f, [(prob.tspan[2] - prob.tspan[1]) / 2],
        lb = [prob.tspan[1]],
        ub = [prob.tspan[end]])
    res = solve(oprob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = 10000)
    res.u[1], -f(res.u[1], nothing)
end

"""
    plot_extrema(prob, sym)

Plots the solution of the observable `sym` along with showcasing time points where it obtains its maximum and minimum values.
"""
function plot_extrema(prob, sym)
    xmin, xminval = get_min_t(prob, sym)
    xmax, xmaxval = get_max_t(prob, sym)
    sol = solve(prob)
    plot(sol, idxs = sym)
    scatter!([xmin], [xminval])
    scatter!([xmax], [xmaxval])
end

"""
    phaseplot_extrema(prob, sym, plotsyms)

Plots the phase plot solution of the observable `sym` along with showcasing time points where it
obtains its maximum and minimum values. `plotsyms` should be given as the tuple of symbols for the
observables that define the axis of the phase plot.
"""
function phaseplot_extrema(prob, sym, plotsyms)
    sol = solve(prob)
    xmin, xminval = get_min_t(prob, sym)
    xmax, xmaxval = get_max_t(prob, sym)
    plot(sol, idxs = plotsyms)
    scatter!([[sol(xmin; idxs = x)] for x in plotsyms]...)
    scatter!([[sol(xmax; idxs = x)] for x in plotsyms]...)
end

"""
    get_uncertainty_forecast(prob, sym, t, uncertainp, samples)

Get the ensemble of time-series of state `sym` evaluated at times `t` for solutions with
uncertain parameters specified according to the distributions in `uncertainp`. The distributions
are specified in the form `[sym1 => dist1, sym2 => dist2]` where `dist` is a Distributions.jl
distribution. Samples is the number of trajectories to run.
"""
function get_uncertainty_forecast(prob, sym, t, uncertainp, samples)
    @assert t[1] >= prob.tspan[1]
    function prob_func(prob, i, reset)
        ps = getindex.(uncertainp, 1) .=> rand.(getindex.(uncertainp, 2))
        prob = remake(prob, tspan = (prob.tspan[1], min(prob.tspan[2], t[end])),
            p = ps)
    end
    eprob = EnsembleProblem(prob, prob_func = prob_func)
    esol = solve(eprob, nothing, EnsembleSerial(), saveat = t, trajectories = samples)
    Array.(reduce.(hcat, [esol[i][sym] for i in 1:samples]))
end

"""
get_uncertainty_forecast_quantiles(prob, sym, t, uncertainp, samples, quants = (0.05, 0.95))

Get the ensemble of time-series of state `sym` evaluated at times `t` for solutions with
uncertain parameters specified according to the distributions in `uncertainp`. The distributions
are specified in the form `[sym1 => dist1, sym2 => dist2]` where `dist` is a Distributions.jl
distribution. Samples is the number of trajectories to run.

Returns a tuple of arrays for the quantiles `quants` which defaults to the 95% confidence intervals.
"""
function get_uncertainty_forecast_quantiles(prob, sym, t, uncertainp, samples,
        quants = (0.05, 0.95))
    @assert t[1] >= prob.tspan[1]
    function prob_func(prob, i, reset)
        ps = getindex.(uncertainp, 1) .=> rand.(getindex.(uncertainp, 2))
        prob = remake(prob, tspan = (prob.tspan[1], min(prob.tspan[2], t[end])),
            p = ps)
    end
    eprob = EnsembleProblem(prob, prob_func = prob_func)

    indexof(sym, syms) = indexin(Symbol.(sym), Symbol.(syms))
    idx = indexof(sym, states(prob.f.sys))

    esol = solve(eprob, nothing, EnsembleSerial(), saveat = t, trajectories = samples,
        save_idxs = idx)
    [Array(reduce(hcat, SciMLBase.EnsembleAnalysis.timeseries_steps_quantile(esol, q).u)')
     for q in quants]
end

"""
    plot_uncertainty_forecast(prob, sym, t, uncertainp, samples)
"""
function plot_uncertainty_forecast(prob, sym, t, uncertainp, samples;
        label = reshape(string.(Symbol.(sym)), 1, length(sym)),
        kwargs...)
    esol = get_uncertainty_forecast(prob, sym, t, uncertainp, samples)
    p = plot(Array(esol[1]'), idxs = sym; label = label, kwargs...)
    for i in 2:samples
        plot!(p, Array(esol[i]'), idxs = sym; label = false, kwargs...)
    end
    display(p)
end

"""
    plot_uncertainty_forecast_quantiles(prob, sym, t, uncertainp, samples, quants = (0.05, 0.95))
"""
function plot_uncertainty_forecast_quantiles(prob, sym, t, uncertainp, samples,
        quants = (0.05, 0.95); label = false,
        kwargs...)
    qs = get_uncertainty_forecast_quantiles(prob, sym, t, uncertainp, samples, quants)
    plot(t, qs[1]; label = label, kwargs...)
    plot!(t, qs[2]; label = false, kwargs...)
end
