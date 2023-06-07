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

function relative_l2loss(pvals, (prob, pkeys, t, data))
    p = Pair.(pkeys, pvals)
    _prob = remake(prob, tspan = (prob.tspan[1], t[end]), p = p)
    sol = solve(_prob, saveat = t)
    tot_loss = 0.0
    for pairs in data
        tot_loss += sum(((sol[pairs.first] .- pairs.second) ./ sol[pairs.first]) .^ 2)
    end
    return tot_loss
end

"""
    datafit(prob, p, t, data)

Fit parameters `p` to `data` measured at times `t`.

## Arguments

  - `prob`: ODEProblem
  - `p`: Vector of pairs of symbolic parameters and initial guesses for the parameters.
  - `t`: Vector of time-points
  - `data`: Vector of pairs of symbolic states and measurements of these states at times `t`.

## Keyword Arguments

    - `loss`: the loss function used for fitting. Defaults to `EasyModelAnalysis.l2loss`,
      with an alternative being `EasyModelAnalysis.relative_l2loss` for relative weighted error.
    - `lb`: lower bounds for the parameters. Defaults to `fill(-Inf, length(p))`.
    - `ub`: upper bounds for the parameters. Defaults to `fill(Inf, length(p))`.
    - `alg`: the optimization algorithm to use. Defaults to `NLopt.LN_SBPLX()`.
    - `solve_kws`: keyword arguments to pass to `solve`.

`p` does not have to contain all the parameters required to solve `prob`,
it can be a subset of parameters. Other parameters necessary to solve `prob`
default to the parameter values found in `prob.p`.
Similarly, not all states must be measured.
"""
function datafit(prob, p::Vector{Pair{Num, Float64}}, t, data; loss = l2loss,
                 lb = fill(-Inf, length(p)),
                 ub = fill(Inf, length(p)), alg = NLopt.LN_SBPLX(), solve_kws = (;))
    pvals = getfield.(p, :second)
    pkeys = getfield.(p, :first)
    oprob = OptimizationProblem(loss, pvals, (prob, pkeys, t, data);
                                lb,
                                ub)
    res = solve(oprob, alg; solve_kws...)
    Pair.(pkeys, res.u)
end

"""
    global_datafit(prob, pbounds, t, data; maxiters = 10000)

Fit parameters `p` to `data` measured at times `t`.

## Arguments

  - `prob`: ODEProblem
  - `pbounds`: Vector of pairs of symbolic parameters to vectors of lower and upper bounds for the parameters.
  - `t`: Vector of time-points
  - `data`: Vector of pairs of symbolic states and measurements of these states at times `t`.

## Keyword Arguments

  - `maxiters`: how long to run the optimization for. Defaults to 10000. Larger values are slower but more
    robust.
  - `loss`: the loss function used for fitting. Defaults to `EasyModelAnalysis.l2loss`, with an alternative
    being `EasyModelAnalysis.relative_l2loss` for relative weighted error.
  - `alg`: the optimization algorithm to use. Defaults to `BBO_adaptive_de_rand_1_bin_radiuslimited`.
  - `u0`: initial guess for the parameters. Defaults to the midpoint of the bounds.
  - `solve_kws`: keyword arguments to pass to `solve`.

`p` does not have to contain all the parameters required to solve `prob`,
it can be a subset of parameters. Other parameters necessary to solve `prob`
default to the parameter values found in `prob.p`.
Similarly, not all states must be measured.
"""
function global_datafit(prob, pbounds, t, data; maxiters = 10000, loss = l2loss,
                        alg = BBO_adaptive_de_rand_1_bin_radiuslimited(),
                        u0 = nothing,
                        solve_kws = (;))
    plb = getindex.(getfield.(pbounds, :second), 1)
    pub = getindex.(getfield.(pbounds, :second), 2)
    pkeys = getfield.(pbounds, :first)
    u0 = isnothing(u0) ? (pub .+ plb) ./ 2 : u0
    oprob = OptimizationProblem(loss, u0, (prob, pkeys, t, data);
                                lb = plb, ub = pub)
    res = solve(oprob, alg; maxiters, solve_kws...)
    Pair.(pkeys, res.u)
end

Turing.@model function bayesianODE(prob, t, p, data, noise_prior)
    σ ~ noise_prior
    pdist = getfield.(p, :second)
    pkeys = getfield.(p, :first)
    pprior ~ product_distribution(pdist)

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

Calculate posterior distribution for parameters `p` given `data` measured at times `t`.
"""
function bayesian_datafit(prob, p, t, data; noise_prior = InverseGamma(2, 3))
    pdist = getfield.(p, :second)
    pkeys = getfield.(p, :first)

    model = bayesianODE(prob, t, p, data, noise_prior)
    chain = sample(model, NUTS(0.65), MCMCSerial(), 1000, 3; progress = true)
    [Pair(pkeys[i], collect(chain["pprior[" * string(i) * "]"])[:])
     for i in eachindex(pkeys)]
end

"""
    model_forecast_score(probs::AbstractVector, ts::AbstractVector, dataset::AbstractVector{<:Pair})

Compute the L2 distance between each problem and the dataset.

Arguments:

  - `probs`: a vector of problems to simulate.
  - `ts`: time points of the dataset.
  - `dataset`: dataset of the form of `[S => zeros(n), I => zeros(n)]`.

Output: the L2 distance from the dataset for each problem.
"""
function model_forecast_score(probs::AbstractVector, ts::AbstractVector,
                              dataset::AbstractVector{<:Pair})
    obs = map(first, dataset)
    data = map(last, dataset)
    map(probs) do prob
        sol = solve(prob, saveat = ts)
        sum(enumerate(obs)) do (i, o)
            norm(sol[o] - data[i])
        end
    end
end
