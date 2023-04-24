function download_covidhub_data(urls, filenames)
    for (url, filename) in zip(urls, filenames)
        if !isfile(filename)
            Downloads.download(url, filename)
        end
    end
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
    mira_p = ModelingToolkit.toparam.(mira_st_ps)

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
    # append!(mira_p, ModelingToolkit.toparam.(sts_that_should_be_ps2)) 
    # append!(ps_sub_map, sts_that_should_be_ps2 .=> ModelingToolkit.toparam.(sts_that_should_be_ps2)) # why ben, why!! XXlambdaXX

    default_p = [r .=> p[:rate]; mira_p .=> last.(collect(mira_ps))]
    default_u0 = S .=> p[:concentration]
    defaults = [default_p; default_u0]
    # to_ps_names = Symbolics.getname.(sts_that_should_be_ps)
    # ps_sub_map = sts_that_should_be_ps .=> ModelingToolkit.toparam.(sts_that_should_be_ps)

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
    global opt_step_count += 1
    @info ((l, opt_step_count))
    return true
    push!(losses, l)
    if opt_step_count % 10 == 0
        plt = plot(losses, yaxis = :log, ylabel = "Loss", xlabel = "Iteration",
                   legend = false)
        display(plt)
    end

    # todo find out why this doesn't trigger early stopping
    if opt_step_count > 100
        return true
    end

    return false
end

function mycalibrate(prob, train_df, mapping;
                   p = collect(ModelingToolkit.defaults(prob.f.sys)))
    ts = train_df.t
    data = [k => train_df[:, v] for (k, v) in mapping]
    prob = remake(prob; tspan = extrema(ts))

    pkeys = getfield.(p, :first)
    pvals = getfield.(p, :second)

    # this needs to get reset before each call to solve
    global opt_step_count = 0
    losses = Float64[]

    # how best should the domain knowledge that the transition rates for a petri net are always ∈ [0, Inf) be encoded?
    # EMA.datafit hardcoded to [-Inf, Inf]
    oprob = OptimizationProblem(myloss, pvals,
                                lb = fill(0, length(p)),
                                ub = fill(Inf, length(p)), (prob, pkeys, ts, data))
    solve(oprob, NLopt.LN_SBPLX(); callback = mycallback, maxiters=1000) 
end

function l2loss_from_sol(sol, df, mapping)
    ts = df.t
    data = [k => df[:, v] for (k, v) in mapping]
    tot_loss = 0.0
    for pairs in data
        tot_loss += sum((sol(ts)[pairs.first] .- pairs.second) .^ 2)
    end
    return tot_loss
end

to_data(df, mapping) = [k => df[:, v] for (k, v) in mapping]

"specify how many weeks out of the df to use for calibration/datafitting"
function train_test_split(dfi; train_weeks = nrow(dfi) ÷ 2)
    @assert train_weeks < nrow(dfi)
    dfi[1:train_weeks, :], dfi[(train_weeks + 1):end, :]
end
