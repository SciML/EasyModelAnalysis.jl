using EasyModelAnalysis, Test
using ModelingToolkit: t_nounits as t, D_nounits as D

@parameters σ ρ β
@variables x(t) y(t) z(t)

eqs = [
    D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z,
]

@mtkbuild sys = ODESystem(eqs, t)

u0 = [
    D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0,
]

p = [
    σ => 28.0,
    ρ => 10.0,
    β => 8 / 3,
]

tspan = (0.0, 100.0)
prob = ODEProblem(sys, u0, tspan, p, jac = true)
sol = solve(prob)

pbounds = [ρ => [0.0, 20.0], β => [0.0, 100.0]]
sensres = get_sensitivity(prob, 100.0, y, pbounds)
@test_nowarn create_sensitivity_plot(sensres, pbounds)

@test length(sensres) == 5
@test collect(keys(sensres)) ==
    [:ρ_first_order, :β_first_order, :ρ_total_order, :β_total_order, :ρ_β_second_order]

sensres_max = get_sensitivity_of_maximum(prob, 100.0, y, pbounds, samples = 50)
@test length(sensres) == 5
