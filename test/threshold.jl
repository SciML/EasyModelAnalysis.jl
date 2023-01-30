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
prob = ODEProblem(sys, [x => 0.01], (0.0, 50), [p => 1.0])
opt_tspan, (s1, s2, s3), ret = optimal_threshold_intervention(prob, [p => -1.0], x, 3, 50);
@test -(-(opt_tspan...)) < 25

opt_ps, (s1, s2, s3), ret = optimal_parameter_intervention_for_threshold(prob, x, 3,
                                                                         abs(p)^2, [p],
                                                                         [-5.0], [0.0],
                                                                         (2.0, 45.0), 50);
@test abs(opt_ps[p]) < 0.04
opt_ps, (s1, s2, s3), ret = optimal_parameter_intervention_for_threshold(prob, x, 3,
                                                                         -p, [p],
                                                                         [-1.0], [1.0]);
@test abs(opt_ps[p]) > 0.110
opt_ps, s2, ret = optimal_parameter_threshold(prob, x, 3,
                                              -p, [p],
                                              [-1.0], [1.0]);
@test abs(opt_ps[p]) > 0.110
opt_ps, s2, ret = optimal_parameter_threshold(prob, x, 3,
                                              -p, [p],
                                              [-1.0], [1.0]);
@variables t x(t) y(t)
@parameters p1 p2
D = Differential(t)
eqs = [D(x) ~ p1 * abs(x) + p2 * y
       D(y) ~ p1 * abs(x) + p2 * y]
@named sys = ODESystem(eqs)
prob = ODEProblem(sys, [x => 0.01, y => 1], (0.0, 50), [p1 => 0.5, p2 => 0.2])
opt_ps, s2, ret = optimal_parameter_threshold(prob, x, 2, p1 - p2, [p1, p2],
                                              [-2.0, -2.0], [2.0, 2],
                                              ineq_cons
                                              = [abs(p2) - abs(p1) + 0.1]);
@test s2.u[end][1] < 2
@test abs(opt_ps[p2]) - abs(opt_ps[p1]) + 0.1 < 0

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

tresholds = [x > 10.0, y < -5.0]
p_prior = [
    σ => truncated(Normal(28.0, 1.0), 20.0, 40.0),
    β => truncated(Normal(2.7, 0.1), 2.0, 4.0),
]
@test prob_violating_threshold(prob, p_prior, tresholds) > 0.99

tresholds = [x > Inf, y < -Inf]
p_prior = [
    σ => truncated(Normal(28.0, 1.0), 20.0, 40.0),
    β => truncated(Normal(2.7, 0.1), 2.0, 4.0),
]
@test prob_violating_threshold(prob, p_prior, tresholds) < 0.01
