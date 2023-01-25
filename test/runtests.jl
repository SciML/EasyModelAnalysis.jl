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

xmin = get_min_t(prob, x)
@test sol(xmin; idxs = x) <= minimum(sol[x])

xmax = get_max_t(prob, x)
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