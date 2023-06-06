# @time @time_imports using EasyModelAnalysis
using EasyModelAnalysis
using DataFrames, AlgebraicPetri, Catlab, Setfield, MathML, JSON3, CommonSolve
using Downloads, CSV, URIs, DataFrames, Dates, AlgebraicPetri, MathML, Setfield
using Catlab.CategoricalAlgebra: read_json_acset
import Catlab.ACSetInterface: has_subpart
EMA = EasyModelAnalysis
datadir = joinpath(@__DIR__, "../data/")
mkpath(datadir)

function generate_sys_args(p::AbstractPetriNet)
    t = first(@variables t)
    sname′(i) =
        if has_subpart(p, :sname)
            sname(p, i)
        else
            Symbol("S", i)
        end
    tname′(i) =
        if has_subpart(p, :tname)
            tname(p, i)
        else
            Symbol("r", i)
        end

    S = [first(@variables $Si(t)) for Si in sname′.(1:ns(p))]
    r = [first(@parameters $ri) for ri in tname′.(1:nt(p))]
    D = Differential(t)

    tm = TransitionMatrices(p)

    coefficients = tm.output - tm.input

    transition_rates = [r[tr] * prod(S[s]^tm.input[tr, s] for s in 1:ns(p))
                        for tr in 1:nt(p)]

    eqs = [D(S[s]) ~ transition_rates' * coefficients[:, s] for s in 1:ns(p)]

    eqs, t, S, r
end

function ModelingToolkit.ODESystem(rn::AbstractLabelledReactionNet; name = :ReactionNet,
                                   kws...)
    sys = ODESystem(generate_sys_args(rn)...; name = name, kws...)
    defaults = get_rn_defaults(sys, rn)
    @set! sys.defaults = defaults
    sys
end

function CommonSolve.solve(sys::ODESystem; prob_kws = (;), solve_kws = (;), kws...)
    solve(ODEProblem(sys; prob_kws..., kws...); solve_kws..., kws...)
end

function CommonSolve.solve(sys::SDESystem; prob_kws = (;), solve_kws = (;), kws...)
    solve(SDEProblem(sys; prob_kws..., kws...); solve_kws..., kws...)
end

to_ssys(sys::ODESystem) = complete(structural_simplify(sys))
to_ssys(pn) = to_ssys(ODESystem(pn))

EMA.solve(pn::AbstractPetriNet; kws...) = solve(to_ssys(pn); kws...)
getsys(sol) = sol.prob.f.sys
getsys(prob::ODEProblem) = prob.f.sys

ModelingToolkit.parameters(prob::ODEProblem) = parameters(getsys(prob))
ModelingToolkit.states(prob::ODEProblem) = states(getsys(prob))

"this doesn't necesarily get everything, use with caution"
sys_syms(sys) = [states(sys); parameters(sys)]

function st_defs(sys)
    filter(x -> !ModelingToolkit.isparameter(x[1]),
           collect(ModelingToolkit.defaults(sys)))
end

function p_defs(sys)
    filter(x -> ModelingToolkit.isparameter(x[1]),
           collect(ModelingToolkit.defaults(sys)))
end

to_data(df, mapping) = [k => df[:, v] for (k, v) in mapping]

"""
Separate keys and values    
"""
_unzip(d::Dict) = (collect(keys(d)), collect(values(d)))
"""
Unzip a collection of pairs    
"""
unzip(ps) = first.(ps), last.(ps)

remove_t(x) = Symbol(replace(String(x), "(t)" => ""))

function set_sys_defaults(sys, pn; kws...)
    pn_defs = get_defaults(pn)
    syms = sys_syms(sys)
    defs = _symbolize_args(pn_defs, syms)
    sys = ODESystem(pn; tspan = ModelingToolkit.get_tspan(sys), defaults = defs, kws...)
end

"""
Transform list of args into Symbolics variables 
```julia
@parameters sig

_symbolize_args([:sig => 1], [sig])

Dict{Num, Int64} with 1 entry:
  sig => 1
```
"""
function _symbolize_args(incoming_values, sys_vars)
    pairs = collect(incoming_values)
    ks, values = unzip(pairs)
    symbols = Symbol.(ks)
    vars_as_symbols = Symbolics.getname.(sys_vars)
    symbols_to_vars = Dict(vars_as_symbols .=> sys_vars)
    Dict([symbols_to_vars[vars_as_symbols[findfirst(x -> x == symbol, vars_as_symbols)]]
          for symbol in symbols] .=> values)
end

function get_defaults(pn)
    [snames(pn) .=> collect(pn[:concentration]); tnames(pn) .=> collect(pn[:rate])]
end

function get_rn_defaults(sys, rn)
    _symbolize_args(get_defaults(rn), sys_syms(sys))
end

function to_data(sol::ODESolution; sts = states(getsys(sol)))
    sts .=> [sol[x] for x in sts]
end

"this should take sts "
function sol_df_to_t_data(df)
    @parameters t
    sts = [only(@variables $x(t)) for x in Symbol.(remove_t.(names(df)[2:end]))]
    df.timestamp, sts .=> collect(eachcol(df)[2:end])
end

"helper to easily "
function petri_bounds(prob; ps = parameters(getsys(prob)),
                      ranges = fill((0.0, 1.0), length(ps)))
    ps .=> ranges
end

global losses = []
global logged_p = []
global opt_step = 0

callback = function (p, l)
    global opt_step += 1
    if opt_step % 100 == 0
        @show opt_step, l
        push!(losses, deepcopy(l))
        push!(logged_p, deepcopy(p))
        # display(plot(losses))
    end
    return false
end
# fns = readdir(datadir; join = true)
fns = joinpath.((datadir,), (["sir", "sirh", "sird", "sirhd"] .* ".json"))
pns = [read_json_acset(LabelledPetriNet, fn) for fn in fns]
p = pns[1]
sir, sird, sirh, sirhd = pns

tspan = (0.0, 100.0)
saveat = 1

s_defs = [
    :S => 0.99,
    :I => 0.01,
    :R => 0.0,
    :H => 0.0,
    :D => 0.0,
]

ps_defs = [
    :inf => 0.6,
    :rec => 0.2,
    :hosp => 0.1,
    :ideath => 0.01,
    :hrec => 0.05,
    :death => 0.01,
]
sd = Dict(s_defs)
psd = Dict(ps_defs)
defs = merge(sd, psd)

rn = LabelledReactionNet{Number, Number}(sirhd, s_defs, ps_defs)
sol = solve(rn; tspan, saveat) # shortcut 

sys = ODESystem(rn)
ssys = to_ssys(sys)
prob = ODEProblem(ssys, [], tspan)
sol = solve(prob; saveat = 1)
df = DataFrame(sol)
sts = states(ssys)

rns = [LabelledReactionNet{Number, Number}(pn, s_defs, ps_defs) for pn in pns]
syss = [to_ssys(rn) for rn in rns]
sir, sird, sirh, sirhd = syss
rprobs = [ODEProblem(sys, [], tspan) for sys in syss] # why does Distributions export `probs`?
sols = solve.(rprobs; saveat)

ists = intersect(states.(syss)...)
data = to_data(sol; sts = ists)

rprob = rprobs[1]
fit = EMA.global_datafit(rprob, petri_bounds(rprob), df.timestamp, data;
                         solve_kws = (; callback))
before_loss = EMA.l2loss(rprob.p, (rprob, parameters(rprob), df.timestamp, data))
after_loss = EMA.l2loss(last.(fit), (rprob, first.(fit), df.timestamp, data))

scores = [:sir, :sird, :sirh, :sirhd] .=>
    EMA.model_forecast_score(rprobs, df.timestamp, data)

fits = []
for prob in rprobs
    fit = EMA.global_datafit(prob, petri_bounds(prob), df.timestamp, data)
    push!(fits, fit)
end

fits = [EMA.global_datafit(prob, petri_bounds(prob), df.timestamp, data) for prob in rprobs]
# this was the "easy" case because state and parameter names overlapped, and allowed us to treat them as the same, using the same data across all fits

# to get around this for the case where we have a model that has states Sus, Inf, Rec, and parameters infection_rate, recovery_rate
sir2_rn = LabelledReactionNet{Number, Number}((:Sus => 0.99, :Inf => 0.01, :Rec => 0),
                                              (:infection_rate, 0.3 / 1000) => ((:Sus, :Inf) => (:Inf,
                                                                                                 :Inf)),
                                              (:recovery_rate, 0.2) => (:Inf => :Rec))

sir2 = complete(ODESystem(sir2_rn))
prob2 = ODEProblem(sir2, [], tspan)
# we need to provide a mapping to the data, so that we can use the same data across all fits
# so what we do is create pairs that map states to fit against to DataFrame column names, which allows us to construct `data`


function Base.getindex(sys::ODESystem, s::Symbol)
    syms = sys_syms(sys)
    syms[findfirst(==(s), Symbolics.getname.(syms))]
end

function Base.getindex(sys::ODESystem, syms::AbstractArray{Symbol})
    [sys[s] for s in syms]
end

# datad = Dict(data)
fit_syms = Symbol.(ists)
fit_ns = Symbolics.getname.(ists)

mapping2 = [
    sir2.Sus => Symbol("S(t)"),
    sir2.Inf => Symbol("I(t)"),
    sir2.Rec => Symbol("R(t)"),
]
data2 = to_data(df, mapping2)

mapping = sys[fit_ns] .=> fit_syms
prob_mapping_ps = []

bounds = parameters(getsys(prob2)) .=> ((0.0, 1.0),)

fit = EMA.global_datafit(prob2, bounds, df.timestamp, data2)

# function EMA.global_datafit(probs)

# function amr_to_odesys(fn)