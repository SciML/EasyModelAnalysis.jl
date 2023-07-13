function l2loss(pvals, (prob, pkeys, t, data)::Tuple{Vararg{Any, 4}})
    p = Pair.(pkeys, pvals)
    prob = remake(prob, tspan = (prob.tspan[1], t[end]), p = p)
    sol = solve(prob, saveat = t)
    tot_loss = 0.0
    for pairs in data
        tot_loss += sum((sol[pairs.first] .- pairs.second) .^ 2)
    end
    return tot_loss
end

function l2loss(pvals, (prob, pkeys, data)::Tuple{Vararg{Any, 3}})
    p = Pair.(pkeys, pvals)
    ts = first.(last.(data))
    lastt = maximum(last.(ts))
    timeseries = last.(last.(data))
    datakeys = first.(data)

    prob = remake(prob, tspan = (prob.tspan[1], lastt), p = p)
    sol = solve(prob)
    tot_loss = 0.0
    for i in 1:length(ts)
        tot_loss += sum((sol(ts[i]; idxs = datakeys[i]) .- timeseries[i]) .^ 2)
    end
    return tot_loss
end

function relative_l2loss(pvals, (prob, pkeys, t, data)::Tuple{Vararg{Any, 4}})
    p = Pair.(pkeys, pvals)
    _prob = remake(prob, tspan = (prob.tspan[1], t[end]), p = p)
    sol = solve(_prob, saveat = t)
    tot_loss = 0.0
    for pairs in data
        tot_loss += sum(((sol[pairs.first] .- pairs.second) ./ sol[pairs.first]) .^ 2)
    end
    return tot_loss
end

function relative_l2loss(pvals, (prob, pkeys, data)::Tuple{Vararg{Any, 3}})
    p = Pair.(pkeys, pvals)
    ts = first.(last.(data))
    lastt = maximum(last.(ts))
    timeseries = last.(last.(data))
    datakeys = first.(data)

    prob = remake(prob, tspan = (prob.tspan[1], lastt), p = p)
    sol = solve(prob)
    tot_loss = 0.0
    for i in 1:length(ts)
        vals = sol(ts[i]; idxs = datakeys[i])
        tot_loss += sum(((vals .- timeseries[i]) ./ vals) .^ 2)
    end
    return tot_loss
end

"""
    datafit(prob, p, t, data)
    datafit(prob, p, data)

Fit parameters `p` to `data` measured at times `t`.

## Arguments

  - `prob`: ODEProblem
  - `p`: Vector of pairs of symbolic parameters and initial guesses for the parameters.
  - `t`: Vector of time-points
  - `data`: Vector of pairs of symbolic states and measurements of these states at times `t`.

## Keyword Arguments

    - `loss`: the loss function used for fitting. Defaults to `EasyModelAnalysis.l2loss`,
      with an alternative being `EasyModelAnalysis.relative_l2loss` for relative weighted error.

`p` does not have to contain all the parameters required to solve `prob`,
it can be a subset of parameters. Other parameters necessary to solve `prob`
default to the parameter values found in `prob.p`.
Similarly, not all states must be measured.

## Data Definition

The data definition is given as a vctor of pairs. If `t` is specified globally for the datafit,
then those time series correspond to the time points specified. For example, 

```julia
[
x => [11.352378507900013, 11.818374125301172, -10.72999081810307]
z => [2.005502877055581, 13.626953144513832, 5.382984515620634, 12.232084518374545]
]
```

then if `datafit(prob, p, t, data)`, `t` must be length 3 and these values correspond to `x(t[i])`.

If `datafit(prob, p, data)`, then the data must be a tuple of (t, timeseries), for example:

```julia
[
x => ([1.0, 2.0, 3.0], [11.352378507900013, 11.818374125301172, -10.72999081810307])
z => ([0.5, 1.5, 2.5, 3.5], [2.005502877055581, 13.626953144513832, 5.382984515620634, 12.232084518374545])
]
```

where this means x(2.0) == 11.81...
"""
function datafit(prob, p::Vector{Pair{Num, Float64}}, t, data; loss = l2loss)
    pvals = getfield.(p, :second)
    pkeys = getfield.(p, :first)
    oprob = OptimizationProblem(loss, pvals,
        lb = fill(-Inf, length(p)),
        ub = fill(Inf, length(p)), (prob, pkeys, t, data))
    res = solve(oprob, NLopt.LN_SBPLX())
    Pair.(pkeys, res.u)
end

function datafit(prob, p::Vector{Pair{Num, Float64}}, data; loss = l2loss)
    pvals = getfield.(p, :second)
    pkeys = getfield.(p, :first)
    oprob = OptimizationProblem(loss, pvals,
        lb = fill(-Inf, length(p)),
        ub = fill(Inf, length(p)), (prob, pkeys, data))
    l2loss(last.(p), (prob, pkeys, data))
    res = solve(oprob, NLopt.LN_SBPLX())
    Pair.(pkeys, res.u)
end

"""
    global_datafit(prob, pbounds, t, data; maxiters = 10000)
    global_datafit(prob, pbounds, data; maxiters = 10000)

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

`p` does not have to contain all the parameters required to solve `prob`,
it can be a subset of parameters. Other parameters necessary to solve `prob`
default to the parameter values found in `prob.p`.
Similarly, not all states must be measured.

## Data Definition

The data definition is given as a vctor of pairs. If `t` is specified globally for the datafit,
then those time series correspond to the time points specified. For example, 

```julia
[
x => [11.352378507900013, 11.818374125301172, -10.72999081810307]
z => [2.005502877055581, 13.626953144513832, 5.382984515620634, 12.232084518374545]
]
```

then if `datafit(prob, p, t, data)`, `t` must be length 3 and these values correspond to `x(t[i])`.

If `datafit(prob, p, data)`, then the data must be a tuple of (t, timeseries), for example:

```julia
[
x => ([1.0, 2.0, 3.0], [11.352378507900013, 11.818374125301172, -10.72999081810307])
z => ([0.5, 1.5, 2.5, 3.5], [2.005502877055581, 13.626953144513832, 5.382984515620634, 12.232084518374545])
]
```

where this means x(2.0) == 11.81...
"""
function global_datafit(prob, pbounds, t, data; maxiters = 10000, loss = l2loss)
    plb = getindex.(getfield.(pbounds, :second), 1)
    pub = getindex.(getfield.(pbounds, :second), 2)
    pkeys = getfield.(pbounds, :first)
    oprob = OptimizationProblem(loss, (pub .+ plb) ./ 2,
        lb = plb, ub = pub, (prob, pkeys, t, data))
    res = solve(oprob, BBO_adaptive_de_rand_1_bin_radiuslimited(); maxiters)
    Pair.(pkeys, res.u)
end

function global_datafit(prob, pbounds, data; maxiters = 10000, loss = l2loss)
    plb = getindex.(getfield.(pbounds, :second), 1)
    pub = getindex.(getfield.(pbounds, :second), 2)
    pkeys = getfield.(pbounds, :first)
    oprob = OptimizationProblem(loss, (pub .+ plb) ./ 2,
        lb = plb, ub = pub, (prob, pkeys, data))
    res = solve(oprob, BBO_adaptive_de_rand_1_bin_radiuslimited(); maxiters)
    Pair.(pkeys, res.u)
end

function bayes_unpack_data(p::AbstractVector{<:Pair}, data)
    pdist, pkeys = bayes_unpack_data(p)
    ts = first.(last.(data))
    lastt = maximum(last, ts)
    timeseries = last.(last.(data))
    datakeys = first.(data)
    (pdist, pkeys, ts, lastt, timeseries, datakeys)
end
function bayes_unpack_data(p::AbstractVector{<:Pair})
    pdist = getfield.(p, :second)
    pkeys = getfield.(p, :first)
    (pdist, pkeys)
end

Turing.@model function bayesianODE(prob, t, pdist, pkeys, data, noise_prior)
    σ ~ noise_prior

    pprior ~ product_distribution(pdist)

    prob = remake(prob, tspan = (prob.tspan[1], t[end]), p = Pair.(pkeys, pprior))
    sol = solve(prob, saveat = t)
    if !SciMLBase.successful_retcode(sol)
        Turing.DynamicPPL.acclogp!!(__varinfo__, -Inf)
        return nothing
    end
    for i in eachindex(data)
        data[i].second ~ MvNormal(sol[data[i].first], σ^2 * I)
    end
    return nothing
end

Turing.@model function bayesianODE(prob,
    pdist,
    pkeys,
    ts,
    lastt,
    timeseries,
    datakeys,
    noise_prior)
    σ ~ noise_prior

    pprior ~ product_distribution(pdist)

    prob = remake(prob, tspan = (prob.tspan[1], lastt), p = Pair.(pkeys, pprior))
    sol = solve(prob)
    if !SciMLBase.successful_retcode(sol)
        Turing.DynamicPPL.acclogp!!(__varinfo__, -Inf)
        return nothing
    end
    for i in eachindex(datakeys)
        vals = sol(ts[i]; idxs = datakeys[i])
        timeseries[i] ~ MvNormal(Array(vals), σ^2 * I)
    end
    return nothing
end

"""
Weights can be unbounded. Length of weights must be one less than the length of sols, to apply a sum-to-1 constraint.
Last `sol` is given the weight `1 - sum(weights)`.
"""
struct WeightedSol{T, S <: Tuple{Vararg{AbstractVector{T}}}, W} <: AbstractVector{T}
    sols::S
    weights::W
    function WeightedSol{T}(sols::S,
        weights::W) where {T, S <: Tuple{Vararg{AbstractVector{T}}}, W}
        @assert length(sols) == length(weights) + 1
        new{T, S, W}(sols, weights)
    end
end
Base.length(ws::WeightedSol) = length(first(ws.sols))
Base.size(ws::WeightedSol) = (length(first(ws.sols)),)
function Base.getindex(ws::WeightedSol{T}, i::Int) where {T}
    s = zero(T)
    w = zero(T)
    for j in eachindex(ws.weights)
        w += ws.weights[j]
        s += ws.weights[j] * ws.sols[j][i]
    end
    return s + (one(T) - w) * ws.sols[end][i]
end
function WeightedSol(sols, select, weights)
    T = eltype(weights)
    s = map(Base.Fix2(getindex, select), sols)
    WeightedSol{T}(s, weights)
end
function bayes_unpack_data(p::Tuple{Vararg{<:AbstractVector{<:Pair}}}, data)
    pdist, pkeys = bayes_unpack_data(p)
    ts = first.(last.(data))
    lastt = maximum(last, ts)
    timeseries = last.(last.(data))
    datakeys = first.(data)
    (pdist, pkeys, ts, lastt, timeseries, datakeys)
end
function bayes_unpack_data(p::Tuple{Vararg{<:AbstractVector{<:Pair}}})
    unpacked = map(bayes_unpack_data, p)
    map(first, unpacked), map(last, unpacked)
end

struct Grouper{N}
    sizes::NTuple{N, Int}
end
function (g::Grouper)(x)
    i = Ref(0)
    map(g.sizes) do N
        _i = i[]
        i[] = _i + N
        view(x, (_i + 1):(_i + N))
    end
end
function flatten(x::Tuple)
    reduce(vcat, x), Grouper(map(length, x))
end

function getsols(probs, probspkeys, ppriors, t::AbstractArray)
    map(probs, probspkeys, ppriors) do prob, pkeys, pprior
        solve(remake(prob, tspan = (prob.tspan[1], t[end]), p = Pair.(pkeys, pprior)),
            saveat = t)
    end
end
function getsols(probs, probspkeys, ppriors, lastt::Number)
    map(probs, probspkeys, ppriors) do prob, pkeys, pprior
        solve(remake(prob, tspan = (prob.tspan[1], lastt), p = Pair.(pkeys, pprior)))
    end
end

Turing.@model function ensemblebayesianODE(probs::Union{Tuple, AbstractVector},
    t,
    pdist,
    grouppriorsfunc,
    probspkeys,
    data,
    noise_prior)
    σ ~ noise_prior
    ppriors ~ product_distribution(pdist)

    Nprobs = length(probs)
    Nprobs⁻¹ = inv(Nprobs)
    weights ~ MvNormal(Distributions.Fill(Nprobs⁻¹, Nprobs - 1), Nprobs⁻¹)
    sols = getsols(probs, probspkeys, grouppriorsfunc(ppriors), t)
    if !all(SciMLBase.successful_retcode, sols)
        Turing.DynamicPPL.acclogp!!(__varinfo__, -Inf)
        return nothing
    end
    for i in eachindex(data)
        data[i].second ~ MvNormal(WeightedSol(sols, data[i].first, weights), σ^2 * I)
    end
    return nothing
end
Turing.@model function ensemblebayesianODE(probs::Union{Tuple, AbstractVector},
    pdist,
    grouppriorsfunc,
    probspkeys,
    ts,
    lastt,
    timeseries,
    datakeys,
    noise_prior)
    σ ~ noise_prior
    ppriors ~ product_distribution(pdist)

    sols = getsols(probs, probspkeys, grouppriorsfunc(ppriors), lastt)

    Nprobs = length(probs)
    Nprobs⁻¹ = inv(Nprobs)
    weights ~ MvNormal(Distributions.Fill(Nprobs⁻¹, Nprobs - 1), Nprobs⁻¹)
    if !all(SciMLBase.successful_retcode, sols)
        Turing.DynamicPPL.acclogp!!(__varinfo__, -Inf)
        return nothing
    end
    for i in eachindex(datakeys)
        vals = map(sols) do sol
            sol(ts[i]; idxs = datakeys[i])
        end
        timeseries[i] ~ MvNormal(WeightedSol(vals, weights), σ^2 * I)
    end
    return nothing
end

"""
    bayesian_datafit(prob, p, t, data)
    bayesian_datafit(prob, p, data)

Calculate posterior distribution for parameters `p` given `data` measured at times `t`.

## Data Definition

The data definition is given as a vctor of pairs. If `t` is specified globally for the datafit,
then those time series correspond to the time points specified. For example, 

```julia
[
x => [11.352378507900013, 11.818374125301172, -10.72999081810307]
z => [2.005502877055581, 13.626953144513832, 5.382984515620634, 12.232084518374545]
]
```

then if `datafit(prob, p, t, data)`, `t` must be length 3 and these values correspond to `x(t[i])`.

If `datafit(prob, p, data)`, then the data must be a tuple of (t, timeseries), for example:

```julia
[
x => ([1.0, 2.0, 3.0], [11.352378507900013, 11.818374125301172, -10.72999081810307])
z => ([0.5, 1.5, 2.5, 3.5], [2.005502877055581, 13.626953144513832, 5.382984515620634, 12.232084518374545])
]
```

where this means x(2.0) == 11.81...
"""
function bayesian_datafit(prob,
    p,
    t,
    data;
    noise_prior = InverseGamma(2, 3),
    mcmcensemble::AbstractMCMC.AbstractMCMCEnsemble = Turing.MCMCThreads(),
    nchains = 4,
    niter = 1000)
    (pdist, pkeys) = bayes_unpack_data(p)
    model = bayesianODE(prob, t, pdist, pkeys, data, noise_prior)
    chain = Turing.sample(model,
        Turing.NUTS(0.65),
        mcmcensemble,
        niter,
        nchains;
        progress = true)
    [Pair(p[i].first, collect(chain["pprior[" * string(i) * "]"])[:])
     for i in eachindex(p)]
end

function bayesian_datafit(prob,
    p,
    data;
    noise_prior = InverseGamma(2, 3),
    mcmcensemble::AbstractMCMC.AbstractMCMCEnsemble = Turing.MCMCThreads(),
    nchains = 4,
    niter = 1_000)
    pdist, pkeys, ts, lastt, timeseries, datakeys = bayes_unpack_data(p, data)
    model = bayesianODE(prob, pdist, pkeys, ts, lastt, timeseries, datakeys, noise_prior)
    chain = Turing.sample(model,
        Turing.NUTS(0.65),
        mcmcensemble,
        niter,
        nchains;
        progress = true)
    [Pair(p[i].first, collect(chain["pprior[" * string(i) * "]"])[:])
     for i in eachindex(p)]
end
function bayesian_datafit(probs::Union{Tuple, AbstractVector},
    p::Tuple{Vararg{<:AbstractVector{<:Pair}}},
    t,
    data;
    noise_prior = InverseGamma(2, 3),
    mcmcensemble::AbstractMCMC.AbstractMCMCEnsemble = Turing.MCMCThreads(),
    nchains = 4,
    niter = 1000)
    (pdist_, pkeys) = bayes_unpack_data(p)
    pdist, grouppriorsfunc = flatten(pdist_)

    model = ensemblebayesianODE(probs, t, pdist, grouppriorsfunc, pkeys, data, noise_prior)
    chain = Turing.sample(model,
        Turing.NUTS(0.65),
        mcmcensemble,
        niter,
        nchains;
        progress = true)
    [Pair(p[i].first, collect(chain["pprior[" * string(i) * "]"])[:])
     for i in eachindex(p)]
end

function bayesian_datafit(probs::Union{Tuple, AbstractVector},
    p::Tuple{Vararg{<:AbstractVector{<:Pair}}},
    data;
    noise_prior = InverseGamma(2, 3),
    mcmcensemble::AbstractMCMC.AbstractMCMCEnsemble = Turing.MCMCThreads(),
    nchains = 4,
    niter = 1_000)
    pdist_, pkeys, ts, lastt, timeseries, datakeys = bayes_unpack_data(p, data)
    pdist, grouppriorsfunc = flatten(pdist_)
    model = ensemblebayesianODE(probs,
        pdist,
        grouppriorsfunc,
        pkeys,
        ts,
        lastt,
        timeseries,
        datakeys,
        noise_prior)
    chain = Turing.sample(model,
        Turing.NUTS(0.65),
        mcmcensemble,
        niter,
        nchains;
        progress = true)
    [Pair(p[i].first, collect(chain["pprior[" * string(i) * "]"])[:])
     for i in eachindex(p)]
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
