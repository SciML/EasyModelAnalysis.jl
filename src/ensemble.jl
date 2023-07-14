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
function ensemble_weights(sol::EnsembleSolution, data_ensem; rank = size(data_ensem,2))
    obs = first.(data_ensem)
    predictions = reduce(vcat, reduce(hcat,[sol[i][s] for i in 1:length(sol)]) for s in obs)
    data = reduce(vcat, [data_ensem[i][2] isa Tuple ? data_ensem[i][2][2] : data_ensem[i][2] for i in 1:length(data_ensem)])
    F = svd(data)
    # Truncate SVD
    U, S, V = F.U[:, 1:rank], F.S[1:rank], F.V[:, 1:rank]
    # Compute pseudo-inverse of A from truncated SVD
    pinv = (V * Diagonal(1 ./ S) * U')
    weights = data*A_pinv
end

function bayesian_ensemble(probs, ps, datas;
    noise_prior = InverseGamma(2, 3),
    mcmcensemble::AbstractMCMC.AbstractMCMCEnsemble = Turing.MCMCSerial(),
    nchains = 4,
    niter = 1_000,
    keep = 100)

    fits = map(probs, ps, datas) do prob, p, data
        bayesian_datafit(prob, p, data; noise_prior, mcmcensemble, nchains, niter)
    end

    models = map(probs, fits) do prob, fit
        [remake(prob, p = Pair.(first.(fit), getindex.(last.(fit), i))) for i in length(fit[1][2])-keep:length(fit[1][2])]
    end

    @info "Calibrations are complete"

    all_probs = reduce(vcat,models)

    @info "$(length(all_probs)) total models"

    enprob = EnsembleProblem(all_probs)
end
