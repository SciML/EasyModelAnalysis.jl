MTK = ModelingToolkit
EMA = EasyModelAnalysis
meqs = MTK.equations
dd = "/Users/anand/.julia/dev/EasyModelAnalysis/data"
function download_covidhub_data(urls, filenames)
    for (url, filename) in zip(urls, filenames)
        if !isfile(filename)
            Downloads.download(url, filename)
        end
    end
end

function load_ensemble()
    petri_fns = [
        "BIOMD0000000955_miranet.json",
        "BIOMD0000000960_miranet.json",
        "BIOMD0000000983_miranet.json",
    ]

    abs_fns = [joinpath(dd, fn) for fn in petri_fns]
    T_PLRN = PropertyLabelledReactionNet{Number, Number, Dict}

    petris = read_json_acset.((T_PLRN,), abs_fns)
    syss = structural_simplify.(ODESystem.(petris))
    defs = map(x -> ModelingToolkit.defaults(x), syss)
    petris, syss, defs
end

function get_dataframes(; dd = "/Users/anand/.julia/dev/EasyModelAnalysis/data")
    urls = [
        "https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Incident%20Cases.csv",
        "https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Incident%20Deaths.csv",
        "https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Incident%20Hospitalizations.csv",
    ]
    filenames = [joinpath(dd, URIs.unescapeuri(split(url, "/")[end]))
                 for url in urls]
    download_covidhub_data(urls, filenames)

    # Read the local CSV files into DataFrames
    dfc = CSV.read(filenames[1], DataFrame)
    dfd = CSV.read(filenames[2], DataFrame)
    dfh = CSV.read(filenames[3], DataFrame)
    covidhub = calibration_data(dfc, dfh, dfd, use_hosp = true)
    # adjust_covid_data(covidhub)
    df = groupby_week(covidhub)
    df, dfc, dfd, dfh, covidhub
end

"TODO validate that this is correct"
function adjust_covid_data(df::DataFrame; infection_duration = 10, hosp_duration = 12)
    n_weeks_infect = ceil(Int, infection_duration / 7)
    n_weeks_hosp = ceil(Int, hosp_duration / 7)

    new_df = copy(df)

    for i in 1:nrow(df)
        start_infect = max(1, i - n_weeks_infect + 1)
        start_hosp = max(1, i - n_weeks_hosp + 1)
        new_df.cases[i] = sum(df.cases[start_infect:i])
        new_df.hosp[i] = sum(df.hosp[start_hosp:i])
    end

    return new_df
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
    plot!(plt, df.t, df.deaths; label = "incident deaths", color = "blue")
    plot!(plt, df.t, df.hosp; label = "incident hosp", color = "orange")
    plot!(plt, df.t, df.cases; label = "incident cases", color = "green")
    plt
end

function select_timeperiods(df::DataFrame, split_length::Int; step::Int = split_length)
    if split_length < 1
        error("Split length must be a positive integer.")
    end
    if step < 1
        error("Step must be a positive integer.")
    end
    return [df[i:(i + split_length - 1), :]
            for i in 1:step:(nrow(df) - split_length + 1)]
end

"""
Transform list of args into Symbolics variables     
"""
function symbolize_args(incoming_values, sys_vars)
    pairs = collect(incoming_values)
    ks, values = unzip(pairs)
    symbols = Symbol.(remove_t.(String.(ks)))
    vars_as_symbols = Symbolics.getname.(sys_vars)
    symbols_to_vars = Dict(vars_as_symbols .=> sys_vars)
    Dict([symbols_to_vars[vars_as_symbols[findfirst(x -> x == symbol, vars_as_symbols)]]
          for symbol in symbols] .=> values)
end
sys_syms(sys) = [states(sys); parameters(sys)]
remove_t(x) = Symbol(replace(String(x), "(t)" => ""))

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
    # @info ((l, opt_step_count))

    if opt_step_count % 100 == 0
        push!(losses, l)
        #     plt = plot(losses, yaxis = :log, ylabel = "Loss", xlabel = "Iteration",
        #                legend = false)
        #     display(plt)
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
    solve(oprob, NLopt.LN_SBPLX(); callback = mycallback, maxiters = 1000)
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

"""
    global_datafit(prob, pbounds, t, data; maxiters = 1000)

Fit parameters `p` to `data` measured at times `t`.

## Arguments

  - `prob`: ODEProblem
  - `pbounds`: Vector of pairs of symbolic parameters to vectors of lower and upper bounds for the parameters.
  - `t`: Vector of time-points
  - `data`: Vector of pairs of symbolic states and measurements of these states at times `t`.

## Keyword Arguments

  - `maxiters`: how long to run the optimization for. Defaults to 10000. Larger values are slower but more
    robust.
  - `loss`: the loss function used for fitting. Defaults to `EasyModelAnalysis.l2loss`, with an alternative
    being `EasyModelAnalysis.relative_l2loss` for relative weighted error.

`p` does not have to contain all the parameters required to solve `prob`,
it can be a subset of parameters. Other parameters necessary to solve `prob`
default to the parameter values found in `prob.p`.
Similarly, not all states must be measured.
"""
function my_global_datafit(prob, pbounds, t, data; maxiters = 1000, loss = EMA.l2loss)
    plb = getindex.(getfield.(pbounds, :second), 1)
    pub = getindex.(getfield.(pbounds, :second), 2)
    pkeys = getfield.(pbounds, :first)
    oprob = OptimizationProblem(loss, (pub .+ plb) ./ 2,
                                lb = plb, ub = pub, (prob, pkeys, t, data))
    global losses = []
    res = solve(oprob, BBO_adaptive_de_rand_1_bin_radiuslimited(); maxiters,
                callback = mycallback)

    plt = plot(losses, yaxis = :log, ylabel = "Loss", xlabel = "Iteration",
               legend = false)
    display(plt)
    res
    # Pair.(pkeys, res.u)
end

Base.@propagate_inbounds function Base.getindex(A::SciMLBase.AbstractEnsembleSolution, sym)
    map(sol -> getindex(sol, sym), A.u)
end

Base.@propagate_inbounds function Base.getindex(A::SciMLBase.AbstractEnsembleSolution{T, N},
                                                i::Int) where {T, N}
    A.u[i]
end

function make_bounds(sys; st_bound = (0.0, 1e8), p_bound = (0.0, 1.0))
    st_space = states(sys) .=> (st_bound,)
    p_space = parameters(sys) .=> (p_bound,)
    [st_space; p_space]
end

""
function global_ensemble_fit(odeprobs, dfs, mapping; kws...)
    all_gress = []
    for prob in odeprobs
        sys = prob.f.sys
        gress = []
        bounds = make_bounds(sys)
        for dfi in dfs
            saveat = dfi.t
            gres = my_global_datafit(prob, bounds, saveat, to_data(dfi, mapping);
                                     maxiters = 1000, kws...)
            push!(gress, gres)
        end
        push!(all_gress, gress)
    end
    return all_gress
end

function calculate_losses_and_solutions(all_gress, odeprobs, dfs)
    loss_mat = zeros(length(odeprobs), length(dfs))
    prob_mat = Matrix(undef, length(odeprobs), length(dfs))

    all_sols = []
    for (i, (gress, prob)) in enumerate(zip(all_gress, odeprobs))
        sols = []
        for (j, (dfi, gres)) in enumerate(zip(dfs, gress))
            # @info 
            saveat = dfi.t
            sys = prob.f.sys
            ndefs = first.(make_bounds(syss[i])) .=> gres.u

            np = remake(prob; u0 = ndefs, p = ndefs, tspan = extrema(saveat))
            prob_mat[i, j] = np
            sol = solve(np; saveat = saveat)
            loss_mat[i, j] = gres.objective
            push!(sols, sol)
        end
        push!(all_sols, sols)
    end
    return loss_mat, prob_mat, all_sols
end

function forecast_stitch(df, sols)
    all_obs = map(x -> x[obs_sts], sols)
    obsvecs = reduce(vcat, all_obs)
    obsm = stack(obsvecs)'

    soldf = reduce(vcat, DataFrame.(sols))
    soldf, obsm
end

function forecast_plot(df, sols)
    soldf, obsm = forecast_stitch(df, sols)
    cs = ["blue", "orange", "green"]
    plt = plot_covidhub(df)
    for (j, c) in enumerate(eachcol(obsm))
        display(scatter!(plt, soldf.timestamp, c; ms = 2, label = string(obs_sts[j]),
                         color = cs[j]))
    end
    # end
    plt
end

function fitvec_to_df(fits, syms)
    DataFrame(stack(map(x -> x.u, fits))',
              string.(Symbolics.getname.(syms)))
end

function ensemble_loss_plot(loss_mat)
    plt = plot()
    [plot!(plt, loss_i; label = "$i", yscale = :log10, xaxis = "timeperiod",
                   yaxis = "log l2 loss") for (i, loss_i) in enumerate(eachrow(loss_mat))]
    plt
end

function optimize_ensemble_weights(odeprobs, t, data; maxiters = 100)
    eprob = EnsembleProblem(odeprobs; prob_func = (odeprobs, i, reset) -> odeprobs[i])
    esol = solve(eprob; trajectories = length(odeprobs), saveat = t)
    eo = esol[obs_sts]
    initial_guess = fill(1 / length(odeprobs), length(odeprobs))
    oprob = OptimizationProblem(ensemble_l2loss, initial_guess,
                                lb = zeros(length(odeprobs)), ub = ones(length(odeprobs)),
                                (eo, data))
    global losses = []
    res = solve(oprob, BBO_adaptive_de_rand_1_bin_radiuslimited(); maxiters,
                callback = mycallback)
    # Pair.(pkeys, res.u)
end

# a*eo[1] + b*eo[2] + c*eo[3] = data
function ensemble_l2loss(pvals, (eo, data))
    sum(abs2, data .- stack(sum(pvals .* eo))')
end

function build_weighted_ensemble_df(weights, esol)
    weighted_ensemble_df = DataFrame(stack(sum(weights .* esol[obs_sts]))', [:deaths, :hosp, :cases])
    insertcols!(weighted_ensemble_df, 1, :t => dfi.t)
    weighted_ensemble_df
end
