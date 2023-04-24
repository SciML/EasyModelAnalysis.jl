using Catlab, AlgebraicPetri, ModelingToolkit, DifferentialEquations, UnPack, SciMLBase,
      Distributions, Symbolics, DiffEqBase, Plots, EasyModelAnalysis, NLopt
using DifferentialEquations.EnsembleAnalysis
using Catlab.CategoricalAlgebra, Catlab.Graphics
using CSV, DataFrames
using Optimization
@info "usings"

state_defs(m, sys) = states(sys) .=> m[:concentration]#last.(collect(m.subparts.concentration.m))
function param_defs(m, sys)

    # parameters(sys) .=> convert(Vector{Any}, (m[:rate]))#m[:rate] #last.(collect(m.subparts.rate.m))
    parameters(sys) .=> parse.((Float64,), string.(m[:rate]))
    #m[:rate] #last.(collect(m.subparts.rate.m))
end

make_defaults(m) = make_defaults(m, ODESystem(m))

"rates are nothing in some miranets, so this currently only works for BIOMD955"
function make_defaults(m, sys)
    [state_defs(m, sys); param_defs(m, sys)]
end

function calibration_data(; use_hosp = false)
    df = CSV.read("cases.csv", DataFrame)
    dfd = CSV.read("deaths.csv", DataFrame)
    dfh = CSV.read("hosp.csv", DataFrame)

    us_ = df[df.location .== "US", :]
    usd_ = dfd[dfd.location .== "US", :]
    ush_ = dfh[dfh.location .== "US", :]

    rename!(us_, :value => :cases)
    rename!(usd_, :value => :deaths)
    rename!(ush_, :value => :hosp)

    if use_hosp
        d_ = innerjoin(us_, usd_, ush_, on = :date, makeunique = true)
        d = d_[:, [:date, :cases, :deaths, :hosp]]
    else
        d_ = innerjoin(us_, usd_, on = :date, makeunique = true)
        d = d_[:, [:date, :cases, :deaths]]
    end

    us_ = d
    sort!(us_, :date)
    us = deepcopy(us_)
    # us[!, :unix] = datetime2unix.(DateTime.(us.date))
    insertcols!(us, 1, :unix => datetime2unix.(DateTime.(us.date)))
    # us
end

function plot_covidhub(df)
    plt = plot()
    plot!(plt, df.date, df.cases; label = "cases")
    plot!(plt, df.date, df.deaths; label = "deaths")
    plot!(plt, df.date, df.hosp; label = "hosp")
    plt
    # plot(df.date, df.hosp; label = "hosp")
end
# these dispatches improve the Symbolic interface for EnsembleSolutions
Base.@propagate_inbounds function Base.getindex(A::SciMLBase.AbstractEnsembleSolution, sym)
    map(sol -> getindex(sol, sym), A.u)
end

Base.@propagate_inbounds function Base.getindex(A::SciMLBase.AbstractEnsembleSolution{T, N},
                                                i::Int) where {T, N}
    A.u[i]
end

df1 = df = calibration_data()
df2 = calibration_data(; use_hosp = true)
@assert all(x -> x ∈ df.date, df2.date)

# missing hosp data is only in 2020 
@assert all(x -> year(x) == 2020, setdiff(df.date, df2.date))
df = df2

@info "data loaded"
dir = "/Users/anand/code/python/mira/notebooks/ensemble"
fns = filter(endswith("json"), readdir(dir; join = true))
miras = []

for fn in fns
    push!(miras, read_json_acset(PropertyLabelledReactionNet{Symbol, Number, Dict}, fn))
end

m = miras[1]
mm = deepcopy(m)
all_snames = union(snames.(miras)...)
sys = ODESystem(m)

sys = ODESystem(m; tspan = (0, 100), defaults = make_defaults(m))
prob = ODEProblem(sys)
sol = solve(prob)
plot(sol)

# ensemble stuff
# all_defs = map(make_defaults, miras)

# mira_syss = syss = map(x -> ODESystem(x; defaults = make_defaults(x)), miras)
# all_sts = states.(syss)
# all_ps = parameters.(syss)

# ists = intersect(all_sts...)
# ips = intersect(all_ps...)

# stss = union(all_sts...)
# pss = union(all_ps...)

# defkeys = [stss; pss]
# tspan = (0, 100)

# defs = defkeys .=> rand(length(defkeys))

# syss = ODESystem.(miras; tspan, defaults = defs);
# eprob = EnsembleProblem(odeprobs; prob_func = (probs, i, repeat) -> probs[i])
# @time odeprobs = map(x -> ODEProblem(x; config...), syss);
# sols = map(solve, odeprobs)

# sim = solve(eprob, EnsembleThreads(); trajectories = length(odeprobs))

@info "systems initialized with defaults"

ssys = structural_simplify(sys)
@unpack Susceptible, Infected, Diagnosed, Ailing, Recognized, Threatened, Healed, Extinct = ssys
S, I, D, A, R, T, H, E = Susceptible, Infected, Diagnosed, Ailing, Recognized, Threatened,
                         Healed, Extinct
@unpack t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17 = sys
@parameters t
@variables Hosp(t) Cases(t)

observed_eqs = [
    Hosp ~ R + T,
    Cases ~ D + R + T,
]
hosp = R + T
defs = ModelingToolkit.defaults(sys)
map(x -> x => defs[x], states(ssys))

nts = []
for i in 1:7:(nrow(df) - 6)
    push!(nts,
          (date = df.date[i], hosp = sum(df.hosp[i:(i + 7)]),
           cases = sum(df.cases[i:(i + 7)]), deaths = sum(df.deaths[i:(i + 7)])))
end
df = dfw = DataFrame(nts)
plot_covidhub(df)
N = 300_000_000
# past_n = 50
# train = df[(end - (past_n - 1)):end, :]
past_n = nrow(df)
train = df
all_to_norm = df[:, [:cases, :deaths, :hosp]]
norm_df = map(x -> collect(x) ./ N, eachrow(all_to_norm))
plot_covidhub(train)

to_norm = train[:, [:cases, :deaths, :hosp]]
u_df = map(x -> collect(x) ./ N, eachrow(to_norm)) # (Infected, Extinct, Hospitilized)

@info "since we only have hospitalizations, we need to make a choice on how to split it into R and T (since H ~ R + T)"
u0 = u_df[1]
defs = ModelingToolkit.defaults(ssys)
# defs = Num.(first.(defs)) .=> last.(defs)
for x in states(ssys)
    y = Num(x)
    defs[y] = 0
end

defs[Infected] = u0[1]
defs[Extinct] = u0[2]
defs[Recognized] = u0[3]
defs[Susceptible] = 1 - sum(u0)

ts = 0:1:past_n
config = (; abstol = 1e-8, reltol = 1e-8, saveat = 1)

prob = ODEProblem(ssys, defs, (0, past_n - 1), defs; config...)
sol = solve(prob)
plot(sol)
plt = plot()
plot!(plt, sol, idxs = [hosp])
plot!(plt, 0:(past_n - 1), train.hosp ./ N; label = "hosp")
p = fit_p = param_defs(m, ssys)[1:10]

data = [
    # hosp => train.hosp,
    Extinct => train.deaths ./ N,
    Diagnosed + Recognized + Threatened => train.cases ./ N,
]

# function datafit(prob, p::Vector{Pair{Num, Float64}}, t, data; loss = l2loss)
pvals = getfield.(p, :second)
pkeys = getfield.(p, :first)
oprob = OptimizationProblem(EasyModelAnalysis.l2loss, pvals,
                            lb = fill(0, length(p)),
                            ub = fill(Inf, length(p)), (prob, pkeys, 0:(past_n - 1), data))

res = solve(oprob, NLopt.LN_SBPLX(); callback = (p, l) -> (@show l; false))

# res = solve(oprob, NLopt.LN_SBPLX())
pdefs = Pair.(pkeys, res.u)
# end
sdefs = state_defs(m, ssys)
defs = [pdefs; sdefs]
prob2 = remake(prob, p = defs, u0 = defs, tspan = (0, 100))
prob2 = remake(prob, p = defs, u0 = defs)
sol2 = solve(prob2; saveat = 0:0.001:0.1)
plot(sol2)

# is there a timescale problem?

# rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
# x0 = zeros(2)
# _p = [1.0, 100.0]

# f = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
# l1 = rosenbrock(x0, _p)
# prob = OptimizationProblem(f, x0, _p, lb=[-5,-5], ub=[100.,100])
# res = solve(prob, OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(); callback=(l, pred)->false)
# res = solve(prob, NLopt.LN_SBPLX(); callback=(l, pred)->false)
function add_things!(p, ts::Vararg{Union{Pair, Tuple}})
    n = snames(p)
    state_idx = AlgebraicPetri.state_dict(n)
    for (name, (ins, outs)) in ts
        i = add_transition!(p, tname = name)
        ins = AlgebraicPetri.vectorify(ins)
        outs = AlgebraicPetri.vectorify(outs)
        add_inputs!(p, length(ins), repeat([i], length(ins)),
                    map(x -> state_idx[x], collect(ins)))
        add_outputs!(p, length(outs), repeat([i], length(outs)),
                     map(x -> state_idx[x], collect(outs)))
    end
    p
end

function get_things(p)
    is_ = inputs.((p,), 1:nt(p))
    os_ = outputs.((p,), 1:nt(p))
    tns = tnames(p)
    sns = snames(p)
    (sns, [tns[i] => (sns[is_[i]] => sns[os_[i]]) for i in 1:nt(p)])
end

mm = deepcopy(m)
# add_transition!(mm; tname=:t17)
# add_things!(mm, :t17 => (:Healed => :Susceptible))
i = add_transition!(mm; tname = :t17, rate = Symbol("0.5"))
ins = (:Healed,)
outs = (:Susceptible,)
state_idx = AlgebraicPetri.state_dict(snames(mm))

add_inputs!(mm, length(ins), repeat([i], length(ins)),
            map(x -> state_idx[x], collect(ins)))
add_outputs!(mm, length(outs), repeat([i], length(outs)),
             map(x -> state_idx[x], collect(outs)))
AlgebraicPetri.Graph(mm)
# sys = ODESystem(mm)
@unpack t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17 = sys
# defs = ModelingToolkit.defaults(sys)
# defps = collect(defs)
# defnps = Num.(first.(defps)) .=> parse.((Float64,), string.(last.(defps)))
# push!(defnps, t17 => 10)
# sysm = ODESystem(mm)
sys2 = ODESystem(mm;defaults=make_defaults(mm, sysm), tspan=(0,100))

prob2 = ODEProblem(sys2)
sol2 = solve(prob2)
plot(sol2)

p = fit_p = param_defs(mm, sys2)
pvals = getfield.(p, :second)
pkeys = getfield.(p, :first)
oprob = OptimizationProblem(EasyModelAnalysis.l2loss, pvals,
                            lb = fill(0, length(p)),
                            ub = fill(Inf, length(p)), (prob, pkeys, 0:(past_n - 1), data))

res = solve(oprob, NLopt.LN_SBPLX())#; callback = (p, l) -> (@show l; false))

prob3 = remake(prob, p = defs, u0 = defs, tspan = (0, 100))
prob2 = remake(prob, p = defs, u0 = defs)
sol2 = solve(prob2; saveat = 0:0.001:0.1)
plot(sol2)


using Downloads, URIs
using CSV
using DataFrames

urls = ["https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Incident%20Cases.csv",
        "https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Incident%20Deaths.csv",
        "https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Incident%20Hospitalizations.csv"]

filenames = [URIs.unescapeuri(split(url, "/")[end]) for url in urls]
for (url, filename) in zip(urls, filenames)
    if !isfile(filename)
        Downloads.download(url, filename)
    end
end

# Read the local CSV files into DataFrames
dfc = CSV.read(filenames[1], DataFrame)
dfd = CSV.read(filenames[2], DataFrame)
dfh = CSV.read(filenames[3], DataFrame)
