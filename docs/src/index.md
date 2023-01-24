# EasyModelAnalysis.jl

```@example analysis
using EasyModelAnalysis, Plots

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

```@example analysis
get_timeseries(prob, x, [0.0, 1.0, 2.0])
```

```@example analysis
xmin = get_min_t(prob, x)
```

```@example analysis
xmax = get_max_t(prob, x)
```

```@example analysis
plot(sol, idxs = (x,y))
scatter!([sol(xmin;idxs=x)], [sol(xmin;idxs=y)])
scatter!([sol(xmax;idxs=x)], [sol(xmax;idxs=y)])
```

```@example analysis
plot(sol, idxs = x)
scatter!([xmin],[sol(xmin;idxs=x)])
scatter!([xmax],[sol(xmax;idxs=x)])
```