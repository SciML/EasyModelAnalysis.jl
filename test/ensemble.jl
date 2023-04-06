using Catlab, AlgebraicPetri, ModelingToolkit, DifferentialEquations, UnPack, SciMLBase, Distributions, Symbolics, DiffEqBase, Plots, EasyModelAnalysis
using DifferentialEquations.EnsembleAnalysis
using Catlab.CategoricalAlgebra
using OpenAIReplMode
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
all_fns = readdir("/Users/anand/.julia/dev/ASKEM_Evaluation_Staging/docs/src/Scenario3";join = true)
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

sys = syss[1]
@unpack S, I, R = sys
@time odeprobs = setup_probs(syss, (0, 100), defs);
# 202.225800 seconds (250.18 M allocations: 13.628 GiB, 33.33% gc time, 99.32% compilation time)

odeprobs = map(x -> ODEProblem(x; config...), syss);
@info "problems initialized"

# probs = map(x->remake(x;saveat=1), probs)
eprob = EnsembleProblem(odeprobs; prob_func = (probs, i, repeat) -> probs[i])
sim = esol = solve(eprob, EnsembleThreads(); trajectories = length(odeprobs))

sim[ists]
# summ = EnsembleSummary(sim)

plot(sim, legend=true)
plot(sim, idxs=[S+I], legend=true)
# try on a stratified model
# vector{observed} I ~ sum(I*)

# this demonstrates 
strat_fn = "/Users/anand/.julia/dev/ASKEM_Evaluation_Staging/docs/src/Scenario3/sirhd_renew_vax_age11.json"
strat_fn = "/Users/anand/.julia/dev/ASKEM_Evaluation_Staging/docs/src/Scenario3/sirhd_renew_vax.json"
sp = read_json_acset(LabelledPetriNet, strat_fn)
st_sys = ODESystem(sp)
st_sts = states(st_sys)
st_ps = parameters(st_sys)

# this section demonstrates the difficulty of quickly building observed maps from stratified petrinets
strat_st_vecs = eval.(Meta.parse.(String.(Symbolics.getname.(st_sts))))
vsts = st_sts[findall(==("V"), last.(strat_st_vecs))]
usts = st_sts[findall(==("U"), last.(strat_st_vecs))]

@parameters t
@variables V(t) U(t)
vobs_ex = sum(vsts)
uobs_ex = sum(usts)
defs = [st_sts; st_ps] .=> rand(21)
st_prob = ODEProblem(st_sys, defs, (0, 10), defs)
st_sol = solve(st_prob)
vaxd = st_sol[vobs_ex]
unvax = st_sol[uobs_ex]
plot(st_sol, idxs=[vobs_ex, uobs_ex];legend=true)
plt = plot()
plot!(plt, vaxd;label="vax")
plot!(plt, unvax;label="unvax")

# so for now i show we can write math expressions with [ists; ips] and indexing will work fine

prob1 = ODEProblem(sys, defs, (0, 100), defs; saveat = 1)
sol = solve(prob1)

ep = EnsembleProblem(prob; prob_func)
esim = solve(ep, EnsembleThreads(); trajectories = 10, progress = true)
esol1 = collect(get_timestep(esol, 1));

plot(esol[[S + 1]])
plot(esol; vars = [[S + 1], log(S)])
plot(esol; vars = [[S + 1], log(S)])
@which sol[[u + 1, u - 1]]
@unpack I, inf = sys
exs = [S + 1, S - 1, log(S), I * inf]
sol[exs]

sols = map(solve, probs);
@test DataFrame(sols[1]) == DataFrame(esol.u[1])

plt = plot()
map(x -> plot!(plt, sol[x]; label = string(x)), exs)
display(plt)

# Base.@propagate_inbounds function Base.getindex(A::SciMLBase.AbstractEnsembleSolution{T, N}, i::Int) where {T, N}
#     map(sol->getindex(sol, i), A.u)
# end

# extending a lot of the functionality of EMA to ensemble
# 1. indexing
# 2. plotting
# 3. ensemble statistics
ets = collect(get_timestep(esol, 1));
# emea  = collect(timestep_mean(esol,1));

# 4. ensemble sensitivity analysis

# multiple shooting
# fit, then simulate next week
data = sum((1:4) .* esol[S])


corm = rand(3,3)

# @which solve(eprob, EnsembleThreads(); trajectories = length(probs))
# @which DiffEqBase.__solve(prob, nothing, EnsembleThreads(); trajectories = length(probs))

# summ = EnsembleSummary(sim, 0:0.1:10)
# plot(summ, fillalpha = 0.5)
