# Calibrating Models to Data

```@example datafitting
using EasyModelAnalysis

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
```

```@example datafitting
tsave = [1.0, 2.0, 3.0]
sol_data = solve(prob, saveat = tsave)
data = [x => sol_data[x], z => sol_data[z]]
psub_ini = [σ => 27.0, β => 3.0]
fit = datafit(prob, psub_ini, tsave, data)
pvals_fit = getfield.(fit, :second)
pvals = getfield.(p, :second)[[1, 3]]
isapprox(pvals, pvals_fit, atol = 1e-4, rtol = 1e-4)
```