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

# Threshold
@variables t x(t)
D = Differential(t)
eqs = [D(x) ~ x]
@named sys = ODESystem(eqs)
prob = ODEProblem(sys, [x => 0.01], (0.0, Inf))
sol = stop_at_threshold(prob, x^2, 0.1)
@test sol.u[end][1]^2≈0.1 atol=1e-5

# Intervention
@variables t x(t)
@parameters p
D = Differential(t)
eqs = [D(x) ~ p * x]
@named sys = ODESystem(eqs)
prob = ODEProblem(sys, [x => 0.01], (0.0, Inf), [p => 1.0])
opt_tspan, (s1, s2, s3), ret = optimal_threshold_intervention(prob, [p => -1.0], x, 3, 50);
@test -(-(opt_tspan...)) < 25
