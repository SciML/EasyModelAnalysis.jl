# @time @time_imports using EasyModelAnalysis
using EasyModelAnalysis
using DataFrames, AlgebraicPetri, Catlab, Setfield, MathML, JSON3, CommonSolve
using Catlab.CategoricalAlgebra: read_json_acset
import Catlab.ACSetInterface: has_subpart

EMA = EasyModelAnalysis
# rescale data to be proportion of population
# function scale_df!(df)
#     for c in names(df)[2:end]
#         df[!, c] = df[!, c] ./ total_pop
#     end
# end

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

to_ssys(sys::ODESystem) = complete(structural_simplify(sys))
to_ssys(pn) = to_ssys(ODESystem(pn))

EMA.solve(pn::AbstractPetriNet; kws...) = solve(to_ssys(pn); kws...)
getsys(sol) = sol.prob.f.sys
getsys(prob::ODEProblem) = prob.f.sys

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
# gi(xs, y) = map(x -> x[y], xs)
# cv(x) = collect(values(x))
# read_replace_write(fn, rs) = write(fn, replace(read(fn, String), rs...))
# function fits_to_df(fits)
#     DataFrame(namedtuple.([Symbolics.getname.(ks) .=> vs for (ks, vs) in EMA.unzip.(fits)]))
# end

# function logged_p_df(pkeys, logged_p)
#     DataFrame(stack(logged_p)', Symbolics.getname.(pkeys))
# end

# function fit_plot(sol, df, sts)
#     plt = EMA.plot_covidhub(df)
#     plt = scatter!(plt, sol; idxs = sts)
#     display(plt)
# end

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

EMA = EasyModelAnalysis
datadir = joinpath(@__DIR__, "../data/")
mkpath(datadir)

fns = readdir(datadir; join = true)
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

all_bounds = [parameters(sys) .=> ((0.0, 1.0),) for sys in syss]

ists = intersect(states.(syss)...)

data = to_data(sol; sts = ists)

EMA.global_datafit(rprobs[1], bounds, df.timestamp, data)

scores = [:sir, :sird, :sirh, :sirhd] .=> EMA.model_forecast_score(rprobs, df.timestamp, data)

fits = []
for (prob, bounds) in zip(rprobs, all_bounds)
    fit = EMA.global_datafit(prob, bounds, df.timestamp, data)
    push!(fits, fit)
end

# #
# sir = LabelledReactionNet((:S => 0.99, :I => 0.01, :R => 0),
#                           (:inf, 0.3 / 1000) => ((:S, :I) => (:I, :I)),
#                           (:rec, 0.2) => (:I => :R))

# sirh = LabelledReactionNet((:S => 0.99, :I => 0.01, :R => 0),
#                            (:inf, 0.3 / 1000) => ((:S, :I) => (:I, :I)),
#                            (:rec, 0.2) => (:I => :R))

# sirhd = LabelledReactionNet((:S => 0.99, :I => 0.01, :R => 0),
#                             (:inf, 0.3 / 1000) => ((:S, :I) => (:I, :I)),
#                             (:rec, 0.2) => (:I => :R))

# sird = LabelledReactionNet((:S => 0.99, :I => 0.01, :R => 0),
#                            (:inf, 0.3 / 1000) => ((:S, :I) => (:I, :I)),
#                            (:rec, 0.2) => (:I => :R))
