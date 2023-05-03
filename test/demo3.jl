using AlgebraicPetri, DataFrames, DifferentialEquations, ModelingToolkit, Symbolics,
      EasyModelAnalysis, Catlab, Catlab.CategoricalAlgebra, JSON3, UnPack, Downloads, URIs,
      CSV, MathML, NLopt, Plots, StatsPlots, OptimizationBBO
@info "usings"
MTK = ModelingToolkit
EMA = EasyModelAnalysis
meqs = MTK.equations
total_pop = 30_000_000
dd = "/Users/anand/.julia/dev/EasyModelAnalysis/data"
include("/Users/anand/.julia/dev/EasyModelAnalysis/test/demo_functions.jl")
df, dfc, dfd, dfh, covidhub = get_dataframes()
plot_covidhub(df)
N_weeks = 20;
period_step = 10;
train_weeks = 10; # 10 weeks of training data, 10 weeks of testing
all_ts = df.t
dfs = select_timeperiods(df, N_weeks; step = period_step)
split_dfs = [train_test_split(df; train_weeks = train_weeks) for df in dfs]
train_dfs, test_dfs = unzip(split_dfs)
petri_fns = [
    "BIOMD0000000955_miranet.json",
    "BIOMD0000000960_miranet.json",
    "BIOMD0000000983_miranet.json",
]

petris, syss, defs = load_ensemble(petri_fns)
all_syms = [sys_syms(sys) for sys in syss];
petris[1]
# adjust the defaults to be in terms of the total population. now all 3 models have defaults in terms of pop
# this is a weak link in the workflow
# TODO: scale the outputs for loss instead of the u0

global opt_step_count = 0
for i in 1:2 # only the first two are in proportions 
    for st in states(syss[i])
        defs[i][st] *= total_pop # this mutates the return of ModelingToolkit.defaults
    end
end
# modifiying the observed to be *= total_pop will be the fix for above, since they dont affect dynamics.
observed.(syss)
sys = syss[1]
syms = all_syms[1]

@unpack Deaths, Hospitalizations, Cases = sys
obs_sts = [Deaths, Hospitalizations, Cases]
mapping = Dict([Deaths => :deaths, Cases => :cases, Hospitalizations => :hosp])

prob = ODEProblem(sys, defs[1], extrema(all_ts), defs[1])
sol = solve(prob; saveat = all_ts)
plt = plot_covidhub(df)
plot!(plt, sol; idxs = obs_sts)
single_model_fits = global_ensemble_fit([prob], train_dfs, mapping; maxiters = 5000,
                                        doplot = true);
# there is one plot/fit per timeperiod per model (so 12)
# TODO: to separate into train and test losses, i need to simulate train up until t0 of test, remake with that u0, then forecast for the test period
losses, test_losses, remade_probs, remade_solutions = calculate_losses_and_solutions(single_model_fits,
                                                                                     [prob],
                                                                                     dfs,
                                                                                     mapping);
plot(losses[1, :]; label = "train losses");
xaxis!("timeperiod")

plot(test_losses[1, :]; label = "test losses");
xaxis!("timeperiod")
# one thing that might be useful is seeing how the test loss increases as we forecast further out, since we are attempting to forecast 10 weeks here
forecast_plot(dfs[1], [remade_solutions[1][1]])
fitdf = fitvec_to_df(single_model_fits[1], syms)
plt = forecast_plot(df, remade_solutions[1][1:2:end])
# plot how beta was fit for each timeperiod for model 1
plot(fitdf.beta; label = "beta", xaxis = "timeperiod")
# now multiple models 
odeprobs = [ODEProblem(sys, [], extrema(all_ts)) for sys in syss];
ensemble_fits = global_ensemble_fit(odeprobs, train_dfs, mapping; maxiters = 2000,
                                    doplot = false) # 2000 for speed, the API can parallelize, so I won't do it here
ensemble_losses, ensemble_test_losses, ensemble_remade_probs, ensemble_sols = calculate_losses_and_solutions(ensemble_fits,
                                                                                                             odeprobs,
                                                                                                             dfs,
                                                                                                             mapping);
fitdfs = [fitvec_to_df(fit, all_syms[i]) for (i, fit) in enumerate(ensemble_fits)]
forecast_plts = [forecast_plot(df, ensemble_sol[1:2:end]) for ensemble_sol in ensemble_sols]
display.(forecast_plts);
# this plot shows that the second model consistently underperforms for all timeperiods
ensemble_loss_plot(losses)
dfi = dfs[1]
Matrix(dfi[:, 2:end])
dfi = dfs[1]
prbs = ensemble_remade_probs[:, 1]
# optimize linear conbination weights for the ensemble for first timeperiod
weights = optimize_ensemble_weights(prbs, dfi.t, Matrix(dfi[:, 2:end]); maxiters = 1000)

obs_sts
# interface for solving collections of models at a time, used to do the second phase of optimization
eprob = EnsembleProblem(prbs; prob_func = (probs, i, reset) -> probs[i])
esol = solve(eprob; trajectories = length(prbs), saveat = dfi.t)
esol[obs_sts] # we can index into the ensemble
# this recreates the covidhub df from the weighted ensemble for a single timeperiod
equal_ensemble_df = build_weighted_ensemble_df(fill(1 / 3, 3), esol)
weighted_ensemble_df = build_weighted_ensemble_df(weights, esol)
display(plot_covidhub(weighted_ensemble_df))
# this optimizes the weights for all timeperiods, returning a dataframe for the optimized weight for each model for each timeperiod
weights_df = build_all_weights_df(ensemble_remade_probs, dfs)
# see if the weights are consistent over time (which is consistent with ensemble_loss_plot(losses) above)
plt = plot(; xaxis = "timeperiod")
plot!(plt, weights_df.x1; label = "model 1 weight");
plot!(plt, weights_df.x2; label = "model 2 weight");
plot!(plt, weights_df.x3; label = "model 3 weight");
display(plt)
# to compare with below
plot_covidhub(df)
# plot the weighted ensemble stitched together into a covidhub df
xx = stitched_ensemble_df(ensemble_remade_probs, dfs);
full_weighted_ensemble = reduce(vcat, xx[1:2:end])
plt = plot_covidhub(full_weighted_ensemble;
                    labs = [
                        "weighted ensemble deaths",
                        "weighted ensemble hosp",
                        "weighted ensemble cases",
                    ]);
xaxis!("week")
# single model plots again
forecast_plot(df, ensemble_sols[1][1:2:end])
# single model plots again
forecast_plot(df, ensemble_sols[2][1:2:end])
forecast_plot(df, ensemble_sols[3][1:2:end])

sir_fn = "/Users/anand/.julia/dev/ASKEM_Evaluation_Staging/docs/src/Scenario3/sir.json"
pn = read_json_acset(LabelledPetriNet, sir_fn)
sys = ODESystem(pn)
eqs = ModelingToolkit.equations(sys)
sts = states(sys)
ps = parameters(sys)
u0 = sts .=> [0.99, 0.01, 0.0]
@unpack S, I, R, inf, rec = sys

# conservation_eq = 1 ~ S + I + R
all_eqs = [eqs; conservation_eq]

p_ps = ps .=> [0.5, 0.25]
defaults = [u0; p_ps]
tspan = (0, 100.0)

@named sir = ODESystem(all_eqs; defaults, tspan)
sir = structural_simplify(sir)
prob = ODEProblem(sir)
sol = solve(prob)
plot(sol)
