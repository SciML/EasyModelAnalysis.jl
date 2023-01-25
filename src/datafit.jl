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
