module EasyModelAnalysis

using LinearAlgebra
using Reexport
@reexport using DifferentialEquations
@reexport using ModelingToolkit
@reexport using Distributions
using Optimization, OptimizationBBO, OptimizationNLopt
using GlobalSensitivity, Plots, Turing

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
    xmin,xminval = get_min_t(prob, sym)
    xmax,xmaxval = get_max_t(prob, sym)
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
    xmin,xminval = get_min_t(prob, sym)
    xmax,xmaxval = get_max_t(prob, sym)
    plot(sol, idxs = plotsyms)
    scatter!([[sol(xmin; idxs = plotsyms[i])] for i in plotsyms]...)
    scatter!([[sol(xmax; idxs = plotsyms[i])] for i in plotsyms]...)
end

function l2loss(pvals, (prob, pkeys, t, data))
    p = Pair.(pkeys, pvals)
    prob = remake(prob, tspan = (prob.tspan[1], t[end]), p = p)
    sol = solve(prob, saveat = t)
    tot_loss = 0.0
    for pairs in data
        tot_loss += sum((sol[pairs.first] .- pairs.second) .^ 2)
    end
    return tot_loss
end
"""
    datafit(prob,  p, t, data)

Fit paramters `p` to `data` measured at times `t`.
"""
function datafit(prob, p, t, data)
    pvals = getfield.(p, :second)
    pkeys = getfield.(p, :first)
    oprob = OptimizationProblem(l2loss, pvals,
                                lb = fill(-Inf, length(p)),
                                ub = fill(Inf, length(p)), (prob, pkeys, t, data))
    res = solve(oprob, NLopt.LN_SBPLX())
    Pair.(pkeys, res.u)
end

@model function bayesianODE(prob, t, p, data)
    σ ~ InverseGamma(2, 3)
    pdist = getfield.(p, :second)
    pkeys = getfield.(p, :first)
    pprior ~ Product(pdist)

    prob = remake(prob, tspan = (prob.tspan[1], t[end]), p = Pair.(pkeys, pprior))
    sol = solve(prob, saveat = t)
    failure = size(sol, 2) < length(t)
    if failure
        Turing.DynamicPPL.acclogp!!(__varinfo__, -Inf)
        return nothing
    end
    for i in eachindex(data)
        data[i].second ~ MvNormal(sol[data[i].first], σ^2 * I)
    end
    return nothing
end
"""
    bayesian_datafit(prob,  p, t, data)

Calculate posterior distribution for paramters `p` given `data` measured at times `t`.
"""
function bayesian_datafit(prob, p, t, data)
    pdist = getfield.(p, :second)
    pkeys = getfield.(p, :first)

    model = bayesianODE(prob, t, p, data)
    chain = sample(model, NUTS(0.65), MCMCSerial(), 1000, 3; progress = false)
    [Pair(pkeys[i], collect(chain["pprior[" * string(i) * "]"])[:])
     for i in eachindex(pkeys)]
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

export get_timeseries, get_min_t, get_max_t, datafit, bayesian_datafit, get_sensitivity
end
