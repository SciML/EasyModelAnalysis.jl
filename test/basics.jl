using EasyModelAnalysis, Test

@parameters t σ ρ β
@variables x(t) y(t) z(t)
D = Differential(t)

eqs = [D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@named sys = ODESystem(eqs,t)
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
sol2 = solve(remake(prob, tspan = (0.0, 2.0)))
@test sol2(t_measure; idxs = x).u ≈ x_series

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
