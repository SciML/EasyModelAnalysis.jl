timmeee = time()
using Catlab, AlgebraicPetri, ModelingToolkit, DifferentialEquations, UnPack, SciMLBase,
      Distributions, Symbolics, DiffEqBase, Plots, EasyModelAnalysis
using DifferentialEquations.EnsembleAnalysis
using Catlab.CategoricalAlgebra
using OpenAIReplMode
using CSV, DataFrames
@info "usings"

Base.@propagate_inbounds function Base.getindex(A::SciMLBase.AbstractEnsembleSolution, sym)
    map(sol -> getindex(sol, sym), A.u)
end

Base.@propagate_inbounds function Base.getindex(A::SciMLBase.AbstractEnsembleSolution{T, N},
                                                i::Int) where {T, N}
    A.u[i]
end

"checks that ODEProblem can be made from a system"
function can_make_prob(sys; defaults = ModelingToolkit.defaults(sys))
    isempty(setdiff([states(sys); parameters(sys)], keys(defaults))) &&
        ModelingToolkit.has_tspan(sys)
end

unzip(d::Dict) = (collect(keys(d)), collect(values(d)))
unzip(ps) = first.(ps), last.(ps)

"""
you have a bunch of ODESystems, this gives you a list of ODEProblems

the ideal thing is that ODEProblem(sys;kws...) works, meaning all defaults and tspan are set.

for now I assume that there is meaning to different models having states/parameters with the same name.

we then require that the `defaults` map has the union of all the keys of the systems.

the most basic use case we can think of here is to take the infected counts of all the models, taking the mean and standard deviation.

i dont particularly like this setup because we want to make the system and have that be the whole thing that allows us to simulate
"""
function setup_probs(syss, tspan, defaults; prob_kws...)
    ks, vs = unzip(defaults)
    sts = union(states.(syss)...)
    ps = union(parameters.(syss)...)
    map(system -> ODEProblem(system, defaults, tspan, defaults; prob_kws...), syss)
end
all_fns = readdir("/Users/anand/.julia/dev/ASKEM_Evaluation_Staging/docs/src/Scenario3";
                  join = true)
fns = filter(endswith("json"), all_fns)

pns = read_json_acset.(LabelledPetriNet, fns[1:4])
ts = union(tnames.(pns)...)
st = union(snames.(pns)...)

syss = syss3 = ODESystem.(pns);
@info "systems initialized"
all_sts = states.(syss)
all_ps = parameters.(syss)

ists = intersect(all_sts...)
ips = intersect(all_ps...)

stss = union(all_sts...)
pss = union(all_ps...)

defkeys = [stss; pss]
tspan = (0, 100)

# dist = Uniform(0, 1)
# defs = defkeys .=> dist
defs = defkeys .=> rand(length(defkeys))

syss = ODESystem.(pns; tspan, defaults = defs); # this is the ideal case
@info "systems initialized with defaults"

@test all(x -> can_make_prob(x), syss)

config = (; abstol = 1e-8, reltol = 1e-8, saveat = 1)

sir_sys = sys = syss[1]
@unpack S, I, R = sys
# @time odeprobs = setup_probs(syss, (0, 100), defs);
# 202.225800 seconds (250.18 M allocations: 13.628 GiB, 33.33% gc time, 99.32% compilation time)

@time odeprobs = map(x -> ODEProblem(x; config...), syss);
sir_prob = odeprobs[1]
@info "problems initialized"

# probs = map(x->remake(x;saveat=1), probs)
my_prob_func(probs, i, repeat) = probs[i]
eprob = EnsembleProblem(odeprobs; prob_func = my_prob_func)
sim = esol = solve(eprob, EnsembleThreads(); trajectories = length(odeprobs))

sim[ists]
# summ = EnsembleSummary(sim)

plot(sim, legend = true)
plot(sim, idxs = [I], legend = true)

# try on a stratified model
# vector{observed} I ~ sum(I*)

# this demonstrates
# strat_fn = "/Users/anand/.julia/dev/ASKEM_Evaluation_Staging/docs/src/Scenario3/sirhd_renew_vax_age11.json"

strat_fn = "/Users/anand/.julia/dev/ASKEM_Evaluation_Staging/docs/src/Scenario3/sirhd_renew_vax.json"
sp = read_json_acset(LabelledPetriNet, strat_fn)
@which ODESystem(sp)
st_sys = ODESystem(sp)
st_sts = states(st_sys)
st_ps = parameters(st_sys)

# this section demonstrates the difficulty of quickly building observed maps from stratified petrinets

strat_st_vecs = eval.(Meta.parse.(String.(Symbolics.getname.(st_sts))))
vsts = st_sts[findall(==("V"), last.(strat_st_vecs))]
usts = st_sts[findall(==("U"), last.(strat_st_vecs))]
ists = st_sts[findall(==("I"), first.(strat_st_vecs))]

# create the observed expressions
@parameters t
@variables V(t) U(t) I(t)

vobs_ex = sum(vsts)
uobs_ex = sum(usts)
iobs_ex = sum(ists)

defs = [st_sts; st_ps] .=> rand(21)
st_prob = ODEProblem(st_sys, defs, (0, 10), defs)
st_sol = solve(st_prob)

vaxd = st_sol[vobs_ex]
unvax = st_sol[uobs_ex]
tot_inf = st_sol[iobs_ex]

plot(st_sol, idxs = [vobs_ex, uobs_ex]; legend = true)


plt = plot()
plot!(plt, vaxd; label = "vax")
plot!(plt, unvax; label = "unvax")
plot!(plt, tot_inf; label = "total infected")
plot(st_sol, idxs=ists)
timmeee2 = time()
@info timmeee2 - timmeee # 373.21696496009827 seconds just to get started

# redo example from before MTK/
# alternatively change the ODESystem(pn) constructor to make hiererarchical sys. 
# sys.V.[V1, ... Vn]
# sys.U.[U1, ... Un]


# so for now i show we can write math expressions with [ists; ips] and indexing will work fine

prob1 = ODEProblem(sys, defs, (0, 100), defs; saveat = 1)
sol = solve(prob1)

# ep = EnsembleProblem(prob; prob_func)
# esim = solve(ep, EnsembleThreads(); trajectories = 10, progress = true)
# esol1 = collect(get_timestep(esol, 1));

# plot(esol[[S + 1]])
# plot(esol; vars = [[S + 1], log(S)])
# plot(esol; vars = [[S + 1], log(S)])
# @which sol[[u + 1, u - 1]]
# @unpack I, inf = sys
# exs = [S + 1, S - 1, log(S), I * inf]
# sol[exs]

# sols = map(solve, probs);
# @test DataFrame(sols[1]) == DataFrame(esol.u[1])

# plt = plot()
# map(x -> plot!(plt, sol[x]; label = string(x)), exs)
# display(plt)

# Base.@propagate_inbounds function Base.getindex(A::SciMLBase.AbstractEnsembleSolution{T, N}, i::Int) where {T, N}
#     map(sol->getindex(sol, i), A.u)
# end

# extending a lot of the functionality of EMA to ensemble
# 1. indexing
# 2. plotting
# 3. ensemble statistics
# ets = collect(get_timestep(esol, 1));
# emea  = collect(timestep_mean(esol,1));

# 4. ensemble sensitivity analysis
# structural identifyiability before fitting

# multiple shooting
# fit, then simulate next week
# data = sum((1:4) .* esol[S])

# corm = rand(3, 3)

# @which solve(eprob, EnsembleThreads(); trajectories = length(probs))
# @which DiffEqBase.__solve(prob, nothing, EnsembleThreads(); trajectories = length(probs))

# summ = EnsembleSummary(sim, 0:0.1:10)
# plot(summ, fillalpha = 0.5)


df = CSV.read("cases.csv", DataFrame)
dfd = CSV.read("deaths.csv", DataFrame)
@info "data"
us_ = df[df.location .== "US", :]
usd_ = dfd[dfd.location .== "US", :]
rename!(us_, :value => :cases)
rename!(usd_, :value => :deaths)
d_ = innerjoin(us_, usd_, on = :date, makeunique=true)
d = d_[:, [:date, :cases, :deaths]]
plt = plot()
plot!(plt, d.date, d.cases; label = "cases")
plot!(plt, d.date, d.deaths; label = "deaths")
us_ = d
sort!(us_, :date)
# us100 = us_[1:600, :]
# plot(us100.date, us100.value)
us = deepcopy(us_)
us[!, :date] = datetime2unix.(DateTime.(us.date))
ts = us.date
ts_offset = ts .- ts[1]

sir_prob_date = remake(sir_prob; u0=rand(3), tspan=(ts[1],ts[end]))
sir_sol = solve(sir_prob_date)

I ~ I*
SIRD 
sir_prob_date = remake(sir_prob; u0=rand(3), tspan=(us.date[1],us.date[28]), saveat=us.date[1:28])
[I => us.cases, S=>us.sus, R=>us.rec]
plot(sir_sol;xlims=sir_prob_date.tspan)

train_itv = Week(1)
predict_itv = Week(3)
# tensorboard integration
for itv in itvs
    for prob in probs 
        sir_fit = datafit(prob, psub_ini, us.date, data)
        l2loss(data, solve(remake(prob; p=sir_fit)))
        # solve()
    end
end

data = [I => us.value[1:28]]
n = 1
us[(1*n):(28*n), :]
p_sub = Num.(parameters(sir_prob.f.sys)) .=> rand(2)
datafit(sir_prob_date, p_sub, us.date[1*n:28*n], data)

300_000

# using EasyModelAnalysis

# @parameters t σ ρ β
# @variables x(t) y(t) z(t)
# D = Differential(t)

# eqs = [D(D(x)) ~ σ * (y - x),
#     D(y) ~ x * (ρ - z) - y,
#     D(z) ~ x * y - β * z]

# @named sys = ODESystem(eqs)
# sys = structural_simplify(sys)

# u0 = [D(x) => 2.0,
#     x => 1.0,
#     y => 0.0,
#     z => 0.0]

# p = [σ => 28.0,
#     ρ => 10.0,
#     β => 8 / 3]

# tspan = (0.0, 100.0)
# prob = ODEProblem(sys, u0, tspan, p, jac = true)
# sol = solve(prob)

# tsave = [1.0, 2.0, 3.0]
# data = [x => get_timeseries(prob, x, tsave), z => get_timeseries(prob, z, tsave)]

# psub_ini = [σ => 27.0, β => 3.0]
# fit = datafit(prob, psub_ini, tsave, data)


# function ensemble_datafit(eprob, p::Vector{Pair{Num, Float64}}, t, data; loss = l2loss)
#     pvals = getfield.(p, :second)
#     pkeys = getfield.(p, :first)
#     oprobs = OptimizationProblem(loss, pvals,
#                                 lb = fill(-Inf, length(p)),
#                                 ub = fill(Inf, length(p)), (prob, pkeys, t, data))
#     res = solve(oprob, NLopt.LN_SBPLX())
#     Pair.(pkeys, res.u)
# end