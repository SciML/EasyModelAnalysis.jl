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
    predictions = reduce(vcat, stack([sol[i][s] for i in 1:length(sol)]) for s in obs)
    data = reduce(vcat, [data_ensem[i][2] for i in 1:length(sol)])
    predictions \ data
end
