using AlgebraicPetri, DataFrames, DifferentialEquations, ModelingToolkit, Symbolics, EasyModelAnalysis, Catlab, Catlab.CategoricalAlgebra, JSON3, UnPack, Downloads, URIs, CSV, MathML, NLopt, Plots, StatsPlots, OptimizationBBO, QuasiMonteCarlo
@info "usings"
MTK = ModelingToolkit
EMA = EasyModelAnalysis
meqs = MTK.equations
dd = "/Users/anand/.julia/dev/EasyModelAnalysis/data"
include("demo_functions.jl")

df, dfc, dfd, dfh, covidhub = get_dataframes()
plot_covidhub(df)

petri_fns = [
    "BIOMD0000000955_miranet.json",
    "BIOMD0000000960_miranet.json",
    "BIOMD0000000983_miranet.json",
]

abs_fns = [joinpath(dd, fn) for fn in petri_fns]
T_PLRN = PropertyLabelledReactionNet{Number,Number,Dict}

petris = read_json_acset.((T_PLRN,), abs_fns)
syss = structural_simplify.(ODESystem.(petris))
@info "syss"

defs = map(x -> ModelingToolkit.defaults(x), syss)




all_ts = df.t
dfs = select_timeperiods(df, N_weeks; step=5)
non_overlapdfs = select_timeperiods(df, N_weeks; step=N_weeks)
split_dfs = [train_test_split(df; train_weeks=10) for df in dfs]
train_dfs, test_dfs = unzip(split_dfs)

petris, syss, defs = load_ensemble()
all_syms = [sys_syms(sys) for sys in syss]

# adjust the defaults to be in terms of the total population. now all 3 models have defaults in terms of pop
# this is a weak link in the workflow
# TODO: scale the outputs for loss instead of the u0
total_pop = 300_000_000
N_weeks = 20
global opt_step_count = 0
for i in 1:2 # only the first two are in proportions 
    for st in states(syss[i])
        defs[i][st] *= total_pop # this mutates the return of ModelingToolkit.defaults
    end
end

sys = syss[1]
syms = all_syms[1]

@unpack Deaths, Hospitalizations, Cases = sys
obs_sts = [Deaths, Hospitalizations, Cases]
mapping = Dict([Deaths => :deaths, Cases => :cases, Hospitalizations => :hosp])

prob = ODEProblem(sys, [], extrema(all_ts))
sol = solve(prob)
plot!(sol; idxs=obs_sts .* 1e6)

single_model_fits = global_ensemble_fit([prob], train_dfs, mapping; maxiters=5000)

# TODO: to separate into train and test losses, i need to simulate train up until t0 of test, remake with that u0, then forecast for the test period
losses, remade_probs, remade_solutions = calculate_losses_and_solutions(single_model_fits, [prob], dfs);
fitdf = fitvec_to_df(single_model_fits[1], syms)

# this is a bit buggy if the step size in `select_timeperiods` is not == N_weeks
model1_df, model1_obs = forecast_stitch(df, remade_solutions[1][1:4:end])
plot(model1_df.timestamp, model1_obs[:, 1]; label="deaths")
plot!(df.t, df.deaths; label="ground truth deaths")
plt = forecast_plot(df, remade_solutions[1][1:4:end])
plot(fitdf.beta; label="beta")


# now multiple models 
odeprobs = [ODEProblem(sys, [], extrema(all_ts)) for sys in syss];
ensemble_fits = global_ensemble_fit(odeprobs, train_dfs, mapping; maxiters=2000) # 2000 for speed, the API can parallelize, so I won't do it here
losses, ensemble_remade_probs, ensemble_sols = calculate_losses_and_solutions(ensemble_fits, odeprobs, dfs);
fitdfs = [fitvec_to_df(fit, all_syms[i]) for (i, fit) in enumerate(ensemble_fits)]

forecast_plts = [forecast_plot(df, ensemble_sol) for ensemble_sol in ensemble_sols]
# this plot shows that the second model consistently underperforms 
ensemble_loss_plot(losses)


dfi = dfs[1]
prbs = ensemble_remade_probs[:, 1]
# optimize linear conbination weights for the ensemble for first timeperiod
weights = optimize_ensemble_weights(prbs, dfi.t, Matrix(dfi[:, 2:end]); maxiters=1000)

# interface for solving collections of models at a time, used to do the second phase of optimization
eprob = EnsembleProblem(prbs; prob_func=(probs, i, reset) -> probs[i])
esol = solve(eprob; trajectories=length(prbs), saveat=dfi.t)


all_weights = [optimize_ensemble_weights(prbs, dfi.t, Matrix(dfi[:, 2:end]); maxiters=1000) for (dfi, prbs) in zip(dfs, eachcol(ensemble_remade_probs))]
weights_df = DataFrame(stack(map(x -> x.u, all_weights))', :auto)

plot(weights_df.x1)
plot(weights_df.x2)
plot(weights_df.x3)


weighted_ensemble_df = build_weighted_ensemble_df(weights, esol)
plot_covidhub(weighted_ensemble_df)
plot_covidhub(dfi)


eo = esol[obs_sts]
plot(esol[1]; idxs=obs_sts)
plot(esol[2]; idxs=obs_sts)
plot(esol[3]; idxs=obs_sts)

# equal weighting 
# ensemble_ps = 1/3
avg_weighting = stack(mean(eo))'
plt = plot_covidhub(dfi)
plot!(dfi.t, avg_weighting[:, 1]; label="ensemble avg deaths")
plot!(dfi.t, avg_weighting[:, 2]; label="ensemble avg hosp")
plot!(dfi.t, avg_weighting[:, 3]; label="ensemble avg cases")
