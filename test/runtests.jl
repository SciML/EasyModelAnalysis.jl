using EasyModelAnalysis, Test

@parameters t σ ρ β
@variables x(t) y(t) z(t)
D = Differential(t)

eqs = [D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@named sys = ODESystem(eqs)
sys = structural_simplify(sys)

u0 = [D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0]

p = [σ => 28.0,
    ρ => 10.0,
    β => 8 / 3]

tspan = (0.0, 100.0)
prob = ODEProblem(sys, u0, tspan, p, jac = true)
sol = solve(prob)

t_measure = [0.0, 1.0, 2.0]
x_series = get_timeseries(prob, x, t_measure)
@test sol(t_measure; idxs = x).u ≈ x_series

t_measure2 = [0.0, 1.0, 2.0, 200.0] # go past original tspan
x_series = get_timeseries(prob, x, t_measure)
@test sol(t_measure2; idxs = x).t[end] >= prob.tspan[2]
@test sol(t_measure2; idxs = x).t[end] ≈ t_measure2[end]

xmin, xminval = get_min_t(prob, x)
@test sol(xmin; idxs = x) == xminval
@test sol(xmin; idxs = x) <= minimum(sol[x])

xmax, xmaxval = get_max_t(prob, x)
@test sol(xmax; idxs = x) == xmaxval
@test sol(xmax; idxs = x) >= maximum(sol[x])

pbounds = [ρ => [0.0, 20.0], β => [0.0, 100.0]]
sensres = get_sensitivity(prob, 100.0, y, pbounds)
@test length(sensres) == 5
@test collect(keys(sensres)) ==
      [:ρ_first_order, :β_first_order, :ρ_total_order, :β_total_order, :ρ_β_second_order]

tsave = [1.0, 2.0, 3.0]
sol_data = solve(prob, saveat = tsave)
data = [x => sol_data[x], z => sol_data[z]]
psub_ini = [σ => 27.0, β => 3.0]
fit = datafit(prob, psub_ini, tsave, data)
pvals_fit = getfield.(fit, :second)
pvals = getfield.(p, :second)[[1, 3]]
@test isapprox(pvals, pvals_fit, atol = 1e-4, rtol = 1e-4)

# Threshold
@variables t x(t)
D = Differential(t)
eqs = [D(x) ~ x]
@named sys = ODESystem(eqs)
prob = ODEProblem(sys, [x => 0.01], (0.0, Inf))
sol = stop_at_threshold(prob, x^2, 0.1)
@test sol.u[end][1]^2≈0.1 atol=1e-5

p_prior = [σ => Normal(27.0, 1.0), β => Normal(3.0, 0.1)]
@test_broken p_posterior = bayesian_datafit(prob, p_prior, tsave, data)

# Intervention
@variables t x(t)
@parameters p
D = Differential(t)
eqs = [D(x) ~ p * x]
@named sys = ODESystem(eqs)
prob = ODEProblem(sys, [x => 0.01], (0.0, Inf), [p => 1.0])
opt_tspan, (s1, s2, s3), ret = optimal_threshold_intervention(prob, [p => -1.0], x, 3, 50);
@test -(-(opt_tspan...)) < 25

# plot(s1, lab = "pre-intervention")
# plot!(s2, lab = "intervention")
# plot!(s3, xlims = (0, s3.t[end]), ylims = (0, 5), lab = "post-intervention", dpi = 300)
