# Getting Started with EasyModelAnalysis

EasyModelAnalysis.jl is made to work with a [ModelingToolkit.jl model](https://docs.sciml.ai/ModelingToolkit/stable/).
If one is unfamiliar with ModelingToolkit, check out its tutorial before getting started. We will start by defining
our model as the ModelingToolkit's README example:

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

EasyModelAnalysis.jl then makes it easy to do complex queries about the model with simple one line commands.
For example, let's the values of `x` at times `[0.0, 1.0, 2.0]`:

```@example analysis
get_timeseries(prob, x, [0.0, 1.0, 2.0])
```

That's too simple, so now let's grab the time points where `x` achieves its maximum and minimum:

```@example analysis
xmin,xminval = get_min_t(prob, x)
```

```@example analysis
xmax,xmaxval = get_max_t(prob, x)
```

Was that simple? Let's see what `x` looks like:

```@example analysis
phaseplot_extrema(prob, x, (x,y))
```

```@example analysis
plot_extrema(prob, x)
```

and boom, it grabbed the correct value in something that's relatively difficult. That's the core
of EasyModelAnalysis!
