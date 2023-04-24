t1 = time()
using AlgebraicPetri, DataFrames, DifferentialEquations, ModelingToolkit, Symbolics,
      EasyModelAnalysis, Catlab, Catlab.CategoricalAlgebra, JSON3, UnPack, CSV, DataFrames,
      Downloads, URIs, CSV, DataFrames, MathML, NLopt
using EasyModelAnalysis: NLopt

MTK = ModelingToolkit
meqs = MTK.equations

function download_covidhub_data(urls, filenames)
    begin for (url, filename) in zip(urls, filenames)
        if !isfile(filename)
            Downloads.download(url, filename)
        end
    end end
end

function calibration_data(dfc, dfd, dfh; use_hosp = false, location = "US")
    us_ = dfc[dfc.location .== location, :]
    usd_ = dfd[dfd.location .== location, :]
    ush_ = dfh[dfh.location .== location, :]

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
    insertcols!(us, 1, :unix => datetime2unix.(DateTime.(us.date)))
end

function groupby_week(df)
    first_monday = first(df.date) - Day(dayofweek(first(df.date)) - 2) % 7
    df.t = (Dates.value.(df.date .- first_monday) .+ 1) .÷ 7

    weekly_summary = combine(groupby(df, :t),
                             :cases => sum,
                             :deaths => sum,
                             :hosp => sum)

    rename!(weekly_summary, [:t, :cases, :deaths, :hosp])
    weekly_summary
end

function plot_covidhub(df)
    plt = plot()
    plot!(plt, df.t, df.cases; label = "incident cases")
    plot!(plt, df.t, df.deaths; label = "incident deaths")
    plot!(plt, df.t, df.hosp; label = "incident hosp")
    plt
end

function select_timeperiods(df::DataFrame, split_length::Int)
    if split_length < 1
        error("Split length must be a positive integer.")
    end
    return [df[i:(i + split_length - 1), :]
            for i in 1:split_length:(nrow(df) - split_length + 1)]
end

"""
Transform list of args into Symbolics variables     
"""
function symbolize_args(incoming_values, sys_vars)
    pairs = collect(incoming_values)
    ks, values = unzip(pairs)
    symbols = Symbol.(ks)
    vars_as_symbols = Symbolics.getname.(sys_vars)
    symbols_to_vars = Dict(vars_as_symbols .=> sys_vars)
    Dict([symbols_to_vars[vars_as_symbols[findfirst(x -> x == symbol, vars_as_symbols)]]
          for symbol in symbols] .=> values)
end
sys_syms(sys) = [states(sys); parameters(sys)]

function ModelingToolkit.ODESystem(p::PropertyLabelledReactionNet{Number, Number, Dict};
                                   name = :PropMiraNet, kws...)
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
    S_ = [first(@variables $Si) for Si in sname′.(1:ns(p))] # MathML doesn't know whether a Num should be dependent on t, so we use this to substitute 

    # we have rate parameters and then the mira_parameters
    r = [first(@parameters $ri) for ri in tname′.(1:nt(p))]

    js = [JSON3.read(tprop(p, ti)["mira_parameters"]) for ti in 1:nt(p)]
    for si in 1:ns(p)
        x = get(sprop(p, si), "mira_parameters", nothing)
        isnothing(x) && continue
        push!(js, JSON3.read(x))
    end

    mira_ps = merge(js...)

    mira_st_ps = [first(@variables $k = v) for (k, v) in mira_ps]
    mira_p = MTK.toparam.(mira_st_ps)

    ps_sub_map = mira_st_ps .=> mira_p
    # mira_p = [first(@parameters $k = v) for (k, v) in mira_ps]

    D = Differential(t)

    tm = TransitionMatrices(p)

    coefficients = tm.output - tm.input
    st_sub_map = S_ .=> S
    sym_rate_exprs = [substitute(MathML.parse_str(tprop(p, tr)["mira_rate_law_mathml"]),
                                 st_sub_map) for tr in 1:nt(p)]

    mrl_vars = union(Symbolics.get_variables.(sym_rate_exprs)...)
    sts_that_should_be_ps = setdiff(mrl_vars, S)
    # sts_that_should_be_ps2 = setdiff(sts_that_should_be_ps, mira_st_ps)
    # append!(mira_p, MTK.toparam.(sts_that_should_be_ps2)) 
    # append!(ps_sub_map, sts_that_should_be_ps2 .=> MTK.toparam.(sts_that_should_be_ps2)) # why ben, why!! XXlambdaXX

    default_p = [r .=> p[:rate]; mira_p .=> last.(collect(mira_ps))]
    default_u0 = S .=> p[:concentration]
    defaults = [default_p; default_u0]
    # to_ps_names = Symbolics.getname.(sts_that_should_be_ps)
    # ps_sub_map = sts_that_should_be_ps .=> MTK.toparam.(sts_that_should_be_ps)

    full_sub_map = [st_sub_map; ps_sub_map]
    sym_rate_exprs = [substitute(sym_rate_expr, ps_sub_map)
                      for sym_rate_expr in sym_rate_exprs]

    tm = TransitionMatrices(p)

    coefficients = tm.output - tm.input

    transition_rates = [r[tr] * prod(S[s]^tm.input[tr, s] for s in 1:ns(p))
                        for tr in 1:nt(p)]

    # disabling this for now
    # transition_rates = [sym_rate_exprs[tr] for tr in 1:nt(p)]

    observable_species_idxs = filter(i -> sprop(p, i)["is_observable"], 1:ns(p))
    observable_species_names = Symbolics.getname.(S[observable_species_idxs])

    # todo, this really needs to be validated, since idk if it's correct, update: pretty sure its busted.
    deqs = [D(S[s]) ~ transition_rates' * coefficients[:, s]
            for s in 1:ns(p) if Symbolics.getname(S[s]) ∉ observable_species_names]

    obs_eqs = [substitute(S[i] ~ Symbolics.parse_expr_to_symbolic(Meta.parse(sprop(p, i)["expression"]),
                                                                  @__MODULE__),
                          Dict(full_sub_map))
               for i in observable_species_idxs]
    eqs = [deqs; obs_eqs]
    sys = ODESystem(eqs, t, S, first.(default_p); name = name, defaults, kws...)
end

function u0_defs(sys)
    begin filter(x -> !ModelingToolkit.isparameter(x[1]),
                 collect(ModelingToolkit.defaults(sys))) end
end
function p_defs(sys)
    begin filter(x -> ModelingToolkit.isparameter(x[1]),
                 collect(ModelingToolkit.defaults(sys))) end
end

"this differs from EMA l2 in that we also are optimizing for the u0 to account for there not being a 1-1 way to map the observed dataset onto the u0 of the system"
function myloss(pvals, (prob, pkeys, t, data))
    new_defs = pkeys .=> pvals
    prob = remake(prob, tspan = (prob.tspan[1], t[end]), p = new_defs, u0 = new_defs)
    sol = solve(prob, saveat = t)
    tot_loss = 0.0
    for pairs in data
        tot_loss += sum((sol[pairs.first] .- pairs.second) .^ 2)
    end
    return tot_loss
end

function mycallback(p, l)
    # return true
    global opt_step_count += 1
    @info ((l, opt_step_count))
    push!(losses, l)
    if opt_step_count % 10 == 0
        plt = plot(losses, yaxis = :log, ylabel = "Loss", xlabel = "Iteration",
                   legend = false)
        display(plt)
    end

    if opt_step_count > 100
        return true
    end

    return false
end

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

# MODEL MODIFCATION LOG
# modified all of the  "rate": null, to be 0.1 just so that we can make defaults easier to handle
# deleted all the uses of the XXlambdaXX parameter, since it doesn't get defined anywhere and seems to be a loose end
# noting that BIOMD983 is giving ┌ Warning: Internal error: Variable Quarantined(t) was marked as being in Differential(t)(Quarantined(t)) ~ beta*sigma*Infected_unreported(t)*Susceptible_unconfined(t) + beta*n*sigma*Infected_reported(t)*Susceptible_unconfined(t) - theta*(1.0 - Quarantined(t)) - theta*Quarantined(t), but was actually zero

petri_fns = [
    "BIOMD0000000955_miranet.json",
    "BIOMD0000000960_miranet.json",
    "BIOMD0000000983_miranet.json",
]

abs_fns = [joinpath(@__DIR__, "../data/", fn) for fn in petri_fns]
T_PLRN = PropertyLabelledReactionNet{Number, Number, Dict}

for fn in abs_fns
    @info fn
    p = read_json_acset(T_PLRN, fn)
    sys = structural_simplify(ODESystem(p))
    prob = ODEProblem(sys, [], (0, 100))
    sol = solve(prob)
    display(plot(sol))
    @test sol.retcode == ReturnCode.Success
end

petris = read_json_acset.((T_PLRN,), abs_fns)
syss = structural_simplify.(ODESystem.(petris))
probs = map(x -> ODEProblem(x, [], (0, 100)), syss)

petri = petris[1]
sys = syss[1]
prob = probs[1]
sol = solve(prob)
prob = remake(prob; tspan = (0, 1e4))
sol = solve(prob)

petri = petris[1]
lrn = LabelledReactionNet{Number, Number}(petri)
sys = ODESystem(petri)
sidarthe = ssys = structural_simplify(sys)
sym_vars = syms = sys_syms(sys)

incoming_values = [snames(petri) .=> petri[:concentration]; tnames(petri) .=> petri[:rate]]
defs = symbolize_args(incoming_values, sym_vars)

x = @which ODESystem(lrn; defaults = defs)
@test AlgebraicPetri.ModelingToolkitInterop == x.module
sys2 = structural_simplify(ODESystem(lrn; defaults = defs))
prob = ODEProblem(ssys, [], (0, 100))
prob2 = ODEProblem(sys2, [], (0, 100))
sol = solve(prob)
sol2 = solve(prob2)

# this is testing that the mira mml rate laws are all mass action and solve the same as if we used the normal petri net dispatch
# it doesn't really validate all that much for the non mass action laws 
# i disabled rate laws 
@test isapprox(sol.u[end], sol2.u[end][1:8]; rtol = 1e-4)

petridefs = [snames(petri) .=> petri[:concentration]; tnames(petri) .=> petri[:rate]]
# sidarthe_old_obs = sidarthe = sys = ODESystem(petri) # this is just so we can get the syms out, the real sys used is recreated below

petri = petris[2]
lrn = LabelledReactionNet{Number, Number}(petri)
sys = ODESystem(petri)
ssys = structural_simplify(sys)
sym_vars = syms = sys_syms(sys)

incoming_values = [snames(petri) .=> petri[:concentration]; tnames(petri) .=> petri[:rate]]
defs = symbolize_args(incoming_values, sym_vars)

x = @which ODESystem(lrn; defaults = defs)
@test AlgebraicPetri.ModelingToolkitInterop == x.module
sys2 = structural_simplify(ODESystem(lrn; defaults = defs))
prob = ODEProblem(ssys, [], (0, 100))
prob2 = ODEProblem(sys2, [], (0, 100))
sol = solve(prob)
sol2 = solve(prob2)
plot(sol)
plot(sol2)

# actual demo stuff
petri = petris[1]
sys = ODESystem(petri)
sidarthe = ssys = structural_simplify(sys)
sym_vars = syms = sys_syms(sys)

@unpack Susceptible, Infected, Diagnosed, Ailing, Recognized, Threatened, Healed, Extinct, Deaths, Hospitalizations, Cases = sidarthe
S, I, D, A, R, T, H, E, De, Ho, Ca = Susceptible, Infected, Diagnosed, Ailing, Recognized,
                                     Threatened,
                                     Healed, Extinct, Deaths, Hospitalizations, Cases

tspan = (0.0, 100.0)
defaults = ModelingToolkit.defaults(sidarthe)
for st in [Susceptible, Infected, Diagnosed, Ailing, Recognized, Threatened, Healed, Extinct]
    defaults[st] *= total_pop # this actually mutates the return of ModelingToolkit.defaults
end

# prob = ODEProblem(sidarthe, defaults, tspan, defaults)
prob = ODEProblem(sidarthe, [], tspan)
sol = solve(prob)
plot(sol)

sys = ODESystem(petris[2])
seiahrd = structural_simplify(sys)
prob = ODEProblem(seiahrd, [], tspan)
sol = solve(prob)
plot(sol)
defaults = ModelingToolkit.defaults(seiahrd)
for st in states(seiahrd)
    defaults[st] *= total_pop # this actually mutates the return of ModelingToolkit.defaults
end

sys = ODESystem(petris[3])
sys3 = structural_simplify(sys)
prob = ODEProblem(sys3, [], tspan)
sol = solve(prob)
plot(sol)
defaults = ModelingToolkit.defaults(sys3)
# sys3 has defaults in population, not proportion
# for st in states(seiahrd)
#     defaults[st] *= total_pop # this actually mutates the return of ModelingToolkit.defaults
# end

""" 
if the data are observables, it's not necesarily true that a consistent assignment for u0 exists
    in the BIOMD0000000955 example, 
    * Hos ~ Recog + Thr
    * Case ~ Detec + Recog + Thr
    * Deaths ~ Extinct

    we can solve for D:
    D ~ C - H 

    we can only determine u0 for D, but there are infinite solutions for H and C
"""
nothing
# ddefaults = Dict(defaults)

total_pop = 300_000_000

N_weeks = 10

dfs = select_timeperiods(df, N_weeks)
dfi = dfs[1]
dfx = dfi[1:(N_weeks÷2), :]
dfy = dfi[((N_weeks÷2)+1):end, :]

# p = p_defs(sys)
# p = collect(defaults)
ts = dfx.t
data = [Deaths => dfx.deaths, Cases => dfx.cases, Hospitalizations => dfx.hosp]

# all_fits = []
# for dfi in dfs
@info dfi
p = p_defs(sys)
p = collect(defaults)
pkeys = getfield.(p, :first)
pvals = getfield.(p, :second)
# oargs = (prob, first.(p), ts, data);
# myloss(pvals, oargs)
# t2 = time()

# how best should the domain knowledge that the transition rates for a petri net are always ∈ [0, Inf) be encoded?
global opt_step_count = 0
losses = Float64[]
oprob = OptimizationProblem(myloss, pvals,
                            lb = fill(0, length(p)),
                            ub = fill(Inf, length(p)), (prob, pkeys, ts, data))
@time res = solve(oprob, NLopt.LN_SBPLX(); maxiters=1000)

fitps = Pair.(pkeys, res.u)
rprob = remake(prob; u0 = fitps, p = fitps)
plt = plot_covidhub(dfi)
rsol = solve(rprob; saveat = dfi.t)
scatter!(plt, rsol, vars = [Deaths, Cases, Hospitalizations])
