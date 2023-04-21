t0 = time()
using AlgebraicPetri, DataFrames, DifferentialEquations, ModelingToolkit, Symbolics,
      EasyModelAnalysis, Catlab, Catlab.CategoricalAlgebra, JSON3, UnPack, CSV, DataFrames,
      Downloads, URIs, CSV, DataFrames, MathML, NLopt
using EasyModelAnalysis: NLopt
@info "usings"
MTK = ModelingToolkit
meqs = MTK.equations

include("demo_functions.jl")

urls = [
    "https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Incident%20Cases.csv",
    "https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Incident%20Deaths.csv",
    "https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Incident%20Hospitalizations.csv",
]

filenames = [joinpath(@__DIR__, "../data/", URIs.unescapeuri(split(url, "/")[end]))
             for url in urls]
download_covidhub_data(urls, filenames)

# Read the local CSV files into DataFrames
dfc = CSV.read(filenames[1], DataFrame)
dfd = CSV.read(filenames[2], DataFrame)
dfh = CSV.read(filenames[3], DataFrame)
covidhub = calibration_data(dfc, dfh, dfd, use_hosp = true)
df = groupby_week(covidhub)
plot_covidhub(df)
@info "data"

petri_fns = [
    "BIOMD0000000955_miranet.json",
    "BIOMD0000000960_miranet.json",
    "BIOMD0000000983_miranet.json",
]

abs_fns = [joinpath(@__DIR__, "../data/", fn) for fn in petri_fns]
T_PLRN = PropertyLabelledReactionNet{Number, Number, Dict}

petris = read_json_acset.((T_PLRN,), abs_fns)
syss = structural_simplify.(ODESystem.(petris))
defs = map(x -> ModelingToolkit.defaults(x), syss)
@info "syss"

total_pop = 300_000_000
N_weeks = 10

# adjust the defaults to be in terms of the total population. now all 3 models have defaults in terms of pop
for i in 1:2
    for st in states(syss[i])
        defs[i][st] *= total_pop # this mutates the return of ModelingToolkit.defaults
    end
end

probs = map(x -> ODEProblem(x, [], (0, 100)), syss); # goes after default canonicalization

dfs = select_timeperiods(df, N_weeks)

# train/test split
dfi = dfs[1]

dfx, dfy = train_test_split(dfi, train_weeks = 5)
ts = dfx.t

@unpack Deaths, Hospitalizations, Cases = syss[1]

# this mapping is the same for all in the ensemble
mapping = Dict([Deaths => :deaths, Cases => :cases, Hospitalizations => :hosp])

osols = []

t1 = time()
xscores = []
yscores = []

ress = []

calibrated_probs = []

# takes too long, but is needed to make the nice plot from the doc
# for dfi in dfs
#     @info "" dfi
#     # train/test split, calibrate, reforecast, plot
#     dfx = dfi[1:(N_weeks ÷ 2), :]
#     dfy = dfi[((N_weeks ÷ 2) + 1):end, :]
#     xdata = to_data(dfx, mapping)
#     ydata = to_data(dfy, mapping)

#     ensemble_res = [calibrate(prob, dfx, mapping) for prob in probs]
#     push!(ress, ensemble_res)

#     new_probs = [remake(prob, u0 = res.u, p = res.u)
#                  for (prob, res) in zip(probs, ensemble_res)]

#     push!(calibrated_probs, new_probs)
#     push!(xscores, EasyModelAnalysis.model_forecast_score(new_probs, dfx.t, xdata))
#     push!(yscores, EasyModelAnalysis.model_forecast_score(new_probs, dfy.t, ydata))

#     ts = dfi.t
#     data = to_data(dfi, mapping)

#     for prob in new_probs
#         sol = solve(prob; saveat = ts)
#         plt = plot_covidhub(dfi)
#         # todo change the color of the forecasted points
#         scatter!(plt, sol, idxs = [Deaths, Cases, Hospitalizations])
#         display(plt)
#     end
# end

