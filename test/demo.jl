using AlgebraicPetri, DataFrames, DifferentialEquations, ModelingToolkit, Symbolics,
      EasyModelAnalysis, Catlab, Catlab.CategoricalAlgebra, JSON3, UnPack, CSV, DataFrames,
      Downloads, URIs, CSV, DataFrames, MathML, NLopt

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
    # append!(ps_sub_map, sts_that_should_be_ps2 .=> MTK.toparam.(sts_that_should_be_ps2))

    default_p = [r .=> p[:rate]; mira_p .=> last.(collect(mira_ps))]
    default_u0 = S .=> p[:concentration]
    defaults = [default_p; default_u0]
    # to_ps_names = Symbolics.getname.(sts_that_should_be_ps)
    # ps_sub_map = sts_that_should_be_ps .=> MTK.toparam.(sts_that_should_be_ps)

    full_sub_map = [st_sub_map; ps_sub_map]
    sym_rate_exprs = [substitute(sym_rate_expr, ps_sub_map)
                      for sym_rate_expr in sym_rate_exprs]
    transition_rates = [sym_rate_exprs[tr] for tr in 1:nt(p)]

    observable_species_idxs = filter(i -> sprop(p, i)["is_observable"], 1:ns(p))
    observable_species_names = Symbolics.getname.(S[observable_species_idxs])

    # todo, this really needs to be validated, since idk if it's correct
    deqs = [D(S[s]) ~ transition_rates' * coefficients[:, s]
            for s in 1:ns(p) if Symbolics.getname(S[s]) ∉ observable_species_names]

    obs_eqs = [substitute(S[i] ~ Symbolics.parse_expr_to_symbolic(Meta.parse(sprop(p, i)["expression"]),
                                                                  @__MODULE__),
                          Dict(full_sub_map))
               for i in observable_species_idxs]
    eqs = [deqs; obs_eqs]
    sys = ODESystem(eqs, t, S, first.(default_p); name = name, defaults, kws...)
end


urls = [
    "https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Incident%20Cases.csv",
    "https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Incident%20Deaths.csv",
    "https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Incident%20Hospitalizations.csv",
]

filenames = [URIs.unescapeuri(split(url, "/")[end]) for url in urls]
download_covidhub_data(urls, filenames)
# Read the local CSV files into DataFrames
dfc = CSV.read(filenames[1], DataFrame)
dfd = CSV.read(filenames[2], DataFrame)
dfh = CSV.read(filenames[3], DataFrame)
covidhub = calibration_data(dfc, dfh, dfd, use_hosp = true)
df = groupby_week(covidhub)
plot_covidhub(df)

# NOTE: i modified all of the  "rate": null, to be 0.1 just so that we can make defaults easier to handle
petri_fns = [
    "BIOMD0000000955_miranet.json",
    "BIOMD0000000960_miranet.json",
    "BIOMD0000000983_miranet.json",
]
abs_fns = [joinpath(@__DIR__, "../data/", fn) for fn in petri_fns]
T_PLRN = PropertyLabelledReactionNet{Number, Number, Dict}
petris = read_json_acset.((T_PLRN,), abs_fns)
p = petri = petris[3]
sys = ODESystem(petri)
sys2 = ODESystem(p)
sys2 = structural_simplify(ODESystem(p))
prob = ODEProblem(sys2, [], (0,100))
sol = solve(prob)
plot(sol)


sidarthe_old_obs = sidarthe = sys = ODESystem(petri) # this is just so we can get the syms out, the real sys used is recreated below

@unpack Susceptible, Infected, Diagnosed, Ailing, Recognized, Threatened, Healed, Extinct = sidarthe
@variables t Hospitalizations(t) Cases(t) Deaths(t)
S, I, D, A, R, T, H, E = Susceptible, Infected, Diagnosed, Ailing, Recognized, Threatened,
                         Healed, Extinct

ssnames = string.(snames(petri))
psnames = string.(tnames(petri))

ps = psnames .=> petri[:rate]
u0 = ssnames .=> (petri[:concentration] .* 300_000_000)

syms = sys_syms(sys)
symsnames = symbolize_args(u0, syms)
sympsnames = symbolize_args(ps, syms)
u0syms = first.(collect(symsnames))
pssyms = first.(collect(sympsnames))
defaults = collect(merge(symsnames, sympsnames))
defaults = Num.(first.(defaults)) .=> last.(defaults)
tspan = (0.0, 100.0)

# here is where we're adding in the observed 
eqs = [ModelingToolkit.equations(sys)[1:(end - 3)];
       Hospitalizations ~ Recognized + Threatened;
       Cases ~ Diagnosed + Recognized + Threatened; Deaths ~ Extinct]

sidarthe = sys = structural_simplify(ODESystem(eqs, t, u0syms, pssyms; defaults, tspan,
                                               name = :sidarthe))
prob = ODEProblem(sys)
sol = solve(prob)
plot(sol)

total_pop = 300_000_000

dfs = select_timeperiods(df, 6)
function u0_defs(sys)
    begin filter(x -> !ModelingToolkit.isparameter(x[1]),
                 collect(ModelingToolkit.defaults(sys))) end
end
function p_defs(sys)
    begin filter(x -> ModelingToolkit.isparameter(x[1]),
                 collect(ModelingToolkit.defaults(sys))) end
end

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
ddefaults = Dict(defaults)
ddefaults[Num(Cases)] = u01.cases

remake(pro)

dfi = dfs[1]
dfx = dfi[1:3, :]
dfy = dfi[4:end, :]

p = collect(sympsnames)
ts = dfx.t
data = [Deaths => dfx.deaths, Cases => dfx.cases, Hospitalizations => dfx.hosp]
loss = EasyModelAnalysis.l2loss
function mycallback2(p, l)
    display((p, l, opt_step_count))
    global opt_step_count += 1
    if opt_step_count > 100
        return true
    end
    return false
end
global opt_step_count = 0
losses = Float64[]
function mycallback(p, l)
    return true
    global opt_step_count += 1
    @info ((p, l, opt_step_count))
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

using EasyModelAnalysis: NLopt
pvals = getfield.(p, :second)
pkeys = getfield.(p, :first)

# how best should the domain knowledge that the transition rates for a petri net are always ∈ [0, Inf) be encoded?
oprob = OptimizationProblem(loss, pvals,
                            lb = fill(0, length(p)),
                            ub = fill(Inf, length(p)), (prob, pkeys, ts, data))
res = solve(oprob, NLopt.LN_SBPLX(); callback = mycallback)
fitps = Pair.(pkeys, res.u)

rprob = remake(prob, p = fitps)
plt = covidhub_plot(dfi)
rsol = solve(rprob; saveat = dfi.ts)

using Optim

# Extract initial values from the first row of the dataframe
initial_cases = dfs[1].cases[1]
initial_deaths = dfs[1].deaths[1]
initial_hosp = dfs[1].hosp[1]

# Define the objective function
function objective_function(initial_conditions)
    sol = solve(remake(prob; u0 = initial_conditions))
    sum(abs2,
        sol[[Cases, Deaths, Hospitalizations]][1] .-
        [initial_cases, initial_deaths, initial_hosp])
end

# Optimize the initial conditions
# initial_guess = [sys.u0[:Susceptible], sys.u0[:Infected], sys.u0[:Diagnosed], sys.u0[:Ailing], sys.u0[:Recognized], sys.u0[:Healed], sys.u0[:Threatened], sys.u0[:Extinct]]
result = optimize(objective_function, prob.u0, BFGS())

# Update the ODE system with the optimized initial conditions
optimized_initial_conditions = result.minimizer
sys.u0[:Susceptible] = optimized_initial_conditions[1]
sys.u0[:Infected] = optimized_initial_conditions[2]
sys.u0[:Diagnosed] = optimized_initial_conditions[3]
sys.u0[:Ailing] = optimized_initial_conditions[4]
sys.u0[:Recognized] = optimized_initial_conditions[5]
sys.u0[:Healed] = optimized_initial_conditions[6]
sys.u0[:Threatened] = optimized_initial_conditions[7]
sys.u0[:Extinct] = optimized_initial_conditions[8]
