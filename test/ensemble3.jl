dir = "/Users/anand/code/python/mira/notebooks/ensemble"
fns = filter(endswith("json"), readdir(dir; join = true))
sch_fn = "/Users/anand/code/python/algj/py-acsets/src/acsets/schemas/catlab/PropertyLabelledReactionNet.json"

SchPLRN = read_json_acset_schema(sch_fn)

@acset_type LPRN3(SchPRN) <: AbstractLabelledReactionNet
miras = []
for fn in fns
    push!(miras, read_json_acset(LPRN3{Symbol, Symbol, Any, Any}, fn))
end
m = miras[1]
all_snames = union(snames.(miras)...)
syss = ODESystem.(miras)
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

syss = ODESystem.(miras; tspan, defaults = defs); # this is the ideal case
@info "systems initialized with defaults"

config = (; abstol = 1e-8, reltol = 1e-8, saveat = 1)
@unpack Susceptible, Infected, Diagnosed, Ailing, Recognized, Threatened, Healed, Extinct = syss[1]
S = Susceptible
I = Infected
D = Diagnosed
A = Ailing
R = Recognized
T = Threatened
H = Healed
E = Extinct

# @time odeprobs = setup_probs(syss, (0, 100), defs);
# 202.225800 seconds (250.18 M allocations: 13.628 GiB, 33.33% gc time, 99.32% compilation time)

@time odeprobs = map(x -> ODEProblem(x; config...), syss);
eprob = EnsembleProblem(odeprobs; prob_func = (probs, i, repeat) -> probs[i])
sim = solve(eprob, EnsembleThreads(); trajectories=length(odeprobs))
plot(sim)
plot(sim[1])
plot(sim[I])