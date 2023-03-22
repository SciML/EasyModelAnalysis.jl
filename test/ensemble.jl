using Catlab, AlgebraicPetri, ModelingToolkit, DifferentialEquations
using UnPack
using DifferentialEquations.EnsembleAnalysis
using SciMLBase
using Catlab.CategoricalAlgebra

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
    # @assert all(sts .∈ ks) && all(ps .∈ ks)
    map(system -> ODEProblem(system, defaults, tspan, defaults; prob_kws...), syss)
    # sols = map(prob -> solve(prob; solve_kws...), probs) # make this parallel 
end

fns = filter(endswith("json"), readdir("/Users/anand/.julia/dev/ASKEM_Evaluation_Staging/docs/src/Scenario3";join=true))
pns = read_json_acset.(LabelledPetriNet, fns[1:4])
ts = union(tnames.(pns)...)
st = union(snames.(pns)...)
syss= syss3 = ODESystem.(pns);
all_sts = states.(syss)
all_ps = parameters.(syss)
stss = union(all_sts...)
pss = union(all_ps...)
defkeys = [stss; pss]
tspan = (0, 100)
defs = defkeys .=> rand(length(defkeys))
syss = ODESystem.(pns; tspan, defaults=defs); # this is the ideal case
config = (;abstol=1e-8, reltol=1e-8, saveat=1)

sys = syss[1]
probs = setup_probs(syss, (0, 100), defs)
"checks that ODEProblem can be made from a system"
function can_make_prob(sys)
    isempty(setdiff([states(sys); parameters(sys)], keys(ModelingToolkit.defaults(sys)))) && ModelingToolkit.has_tspan(sys)
end

@test all(x->can_make_prob(x), syss)
probs = map(x->ODEProblem(x;config...), syss)
# probs = map(x->remake(x;saveat=1), probs)
eprob = EnsembleProblem(probs; prob_func=(probs, i, repeat) -> probs[i])
esol = solve(eprob, EnsembleThreads(); trajectories=length(probs))


ists = intersect(all_sts...)
ips = intersect(all_ps...)
# so for now i show we can write math expressions with [ists; ips] and indexing will work fine

prob1 = ODEProblem(sys, defs, (0, 100), defs;saveat=1)
sol = solve(prob1)
@unpack I = sys
Is = map(sol->sol[I], esol.u)
plt = plot()
for sol in esol.u
    plot!(plt, sol[I])
end
display(plt)

ep = EnsembleProblem(prob; prob_func)
esim = solve(ep, EnsembleThreads(); trajectories=10, progress=true)

# we still want a lot more of the machinery already built out in https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/ to work
# @variables t u(t)
# D = Differential(t)
# eqs = [D(u) ~ -u]
# prob = ODEProblem(ODESystem(eqs;name=:foo), [1.0], (0.0,1.0), [1.0])
# function prob_func(prob,i,repeat)
#     @. prob.u0 = rand()*prob.u0
#     prob
#   end
# esim[u]
# @which esim[u]
# sol = solve(prob)
# @which sol[[u+1, u-1]]

"this breaks plot(esol)"
Base.@propagate_inbounds function Base.getindex(A::SciMLBase.AbstractEnsembleSolution, sym)
    map(sol->getindex(sol, sym), A.u)
end

Base.@propagate_inbounds function Base.getindex(A::SciMLBase.AbstractEnsembleSolution{T, N}, i::Int) where {T, N}
    A.u[i]
end

esol1 = collect(get_timestep(esol,1));

# Base.@propagate_inbounds function Base.getindex(A::SciMLBase.AbstractEnsembleSolution{T, N, AbstractArray{O}}, i::Int) where {T, N, O <: SciMLBase.AbstractTimeseriesSolution}
#     map(sol->getindex(sol, i), A.u)
# end

plot(esol[[S + 1]])
plot(esol; vars=[[S + 1], log(S)])
plot(esol; vars=[[S + 1], log(S)])
@which sol[[u+1, u-1]]
@unpack I, inf = sys
exs = [S+1, S-1, log(S), I*inf]
sol[exs]

sols = map(solve, probs);
@test DataFrame(sols[1]) == DataFrame(esol.u[1])

plt = plot()
map(x->plot!(plt, sol[x]; label=string(x)), exs)
display(plt)

# Base.@propagate_inbounds function Base.getindex(A::SciMLBase.AbstractEnsembleSolution{T, N}, i::Int) where {T, N}
#     map(sol->getindex(sol, i), A.u)
# end

# extending a lot of the functionality of EMA to ensemble
# 1. indexing
# 2. plotting
# 3. ensemble statistics
ets  = collect(get_timestep(esol,1));
# emea  = collect(timestep_mean(esol,1));

# 4. ensemble sensitivity analysis

data = sum((1:4) .* esol[S])



