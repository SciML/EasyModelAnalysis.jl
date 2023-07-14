function naivemap(f, ::EnsembleThreads, arg0, args...)
    t = Vector{Task}(undef, length(arg0))
    for (n, a) in enumerate(arg0)
        t[n] = let an = map(Base.Fix2(Base.getindex, n), args)
            Threads.@spawn f(a, an...)
        end
    end
    return identity.(map(fetch, t))
end
function naivemap(f, ::EnsembleSerial, args...)
    map(f, args...)
end

getsolmean(sol, s) = sol[s]
function getsolmean(sol::EnsembleSolution, s)
  acc = sol[1][s]
  N = length(sol)
  for i = 2:N
    acc .+= sol[i][s]
  end
  acc ./= N
end

"""
    ensemble_weights(sol::EnsembleSolution, data_ensem)


Returns the weights for a linear combination of the models
so that the prediction = sum(weight[i] * model_prediction[i])
where `sol` is the ensemble solution and `data_ensem` is the
dataset on which the ensembler should be trained on.

## Arguments

- `sol`: the ensemble solution of the prediction data
- `data_ensem`: a vector of pairs from the symbolic states to the measurements

!!! note

    This function currently assumes that `sol.t` matches the time points of all measurements
    in `data_ensem`!
"""
function ensemble_weights(sol::EnsembleSolution, data_ensem)
    obs = first.(data_ensem)
    predictions = reduce(vcat,
        reduce(hcat, [getsolmean(sol[i],s) for i in 1:length(sol)]) for s in obs)
    data = reduce(vcat,
        [data_ensem[i][2] isa Tuple ? data_ensem[i][2][2] : data_ensem[i][2]
         for i in 1:length(data_ensem)])
    weights = predictions \ data
end

function bayesian_ensemble(probs, ps, datas;
    noise_prior = InverseGamma(2, 3),
    ensemblealg::SciMLBase.BasicEnsembleAlgorithm = EnsembleThreads(),
    nchains = 4,
    niter = 1_000,
    keep = 100)
    models = naivemap(ensemblealg, probs, ps, datas) do prob, p, data
        bayesian_datafit(prob, p, data; noise_prior, ensemblealg, nchains, niter)
    end

    @info "Calibrations are complete"

    all_probs = reduce(vcat, models)

    @info "$(length(all_probs)) total models"

    enprob = EnsembleProblem(all_probs)
end
