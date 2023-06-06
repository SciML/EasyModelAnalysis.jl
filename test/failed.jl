
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

function download_data(url, dd)
    filename = joinpath(dd, URIs.unescapeuri(split(url, "/")[end]))
    if !isfile(filename)
        Downloads.download(url, filename)
    end
    filename
end

function get_covidhub_data(url, dd)
    return CSV.read(download_data(url, dd), DataFrame)
end

select_location(df, location) = df[df.location .== location, :]
select_location(location) = df -> select_location(df, location)

function date_join(colnames, dfs...)
    d_ = innerjoin(dfs..., on = :date, makeunique = true)
    d = d_[:, colnames]
    return sort!(d, :date)
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

dfc = get_covidhub_data("https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Cumulative%20Cases.csv",
                            datadir)
dfd = get_covidhub_data("https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Cumulative%20Deaths.csv",
                            datadir)
dfh = get_covidhub_data("https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Cumulative%20Hospitalizations.csv",
                            datadir)
                            # select location to use
dfc, dfd, dfh = map(select_location("US"), [dfc, dfd, dfh])

# "US" isn't in the hospitalization data 
# rename to synchronize dataset column names with models
rename!(dfc, :value => :cases)
rename!(dfd, :value => :deaths)
rename!(dfh, :value => :hosp)

# create combined dataframe joined on date
covidhub = EMA.date_join([:date, :cases, :deaths, :hosp], dfc, dfd, dfh)

# aggregate to week-level data
df = EMA.groupby_week(covidhub)
                            
"rescale data to be proportion of population"
function scale_df!(df)
    for c in names(df)[2:end]
        df[!, c] = df[!, c] ./ total_pop
    end
end


# how do you validate that the exprs in data will not error when sol[x] is called?
struct Ensemble
    df::Any
    prob_mapping_pairs::Any
end

# another way of doing this canonicalization is to add observed equations 
# this means that a single `data` is used across all fits
data_sts = first.(data)
@unpack Sus, Inf, Rec = sir2
@unpack S, I, R = sys

obs_eqs = [
    S ~ Sus,
    I ~ Inf,
    R ~ Rec,
]
@set! sir2.observed = obs_eqs
observed(sir2)
sol2 = solve(sir2; tspan, saveat)
sol2[S]
prob2 = ODEProblem(sir2)
sol2 = solve(prob2; tspan=(0., 1e5), saveat)
sol2[S]
plot(sol2; idxs=data_sts)
plot(sol2)

fit = EMA.global_datafit(prob2, petri_bounds(prob2), df.timestamp, data)
# so now we can use a 
all_probs = [rprobs; [prob2]]
fits = []
for prob in all_probs
    bounds = parameters(getsys(prob)) .=> ((0.0, 1.0),)
    fit = EMA.global_datafit(prob, bounds, df.timestamp, data)
    push!(fits, fit)
end



@named sdesys2 = SDESystem(sys, []; tspan)
sp = SDEProblem(sdesys2)
ssol = Array(solve(sp; saveat))
ssol2 = Array(solve(ODEProblem(sdesys2); saveat))
@test isapprox(ssol, ssol2; rtol = 1e-4)

plot(ssol)
ssol = solve(sdesys)
# what is the right assertion for the df to be a timeseries?
#  first column is riskier but more general than assuming its :t or :timestamp
function ModelingToolkit.SDESystem(sys::ODESystem, neqs; kwargs...)
    SDESystem(ModelingToolkit.equations(sys), neqs, ModelingToolkit.get_iv(sys),
              ModelingToolkit.states(sys), ModelingToolkit.parameters(sys);
              tspan = ModelingToolkit.get_tspan(sys),
              defaults = ModelingToolkit.defaults(sys), kwargs...)
end

@named sdesys = SDESystem(sys, 1 .* states(sys); tspan = (0, 1e3))
ssol = solve(sdesys; saveat)
plot(ssol; idxs = sdesys.H)