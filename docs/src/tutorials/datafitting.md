# Calibrating Models to Data

In this tutorial we will showcase the tooling for fitting models to data. Let's take our favorite 2nd order Lorenz equation form
as our model:

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

Let's create a dataset with some set of parameters, and show how the `datafit` function can be used to discover the parameters
that generated the data. To start, let's show the data format by generating a dataset. A dataset contains an array `t` of the
time points for the data, and maps `[observable => timeseries]` where the timeseries is an array of values for the observable
at the time points of `t`. We can use the `get_timeseries` function to generate a dataset like:

```@example datafitting
tsave = [1.0, 2.0, 3.0]
data = [x => get_timeseries(prob, x, tsave), z => get_timeseries(prob, z, tsave)]
```

Now let's do a datafit. We need to choose initial parameters for the fitting process and call the datafit:

```@example datafitting
psub_ini = [σ => 27.0, β => 3.0]
fit = datafit(prob, psub_ini, tsave, data)
```

Recall that our starting parameters, the parameters the dataset was generated from, was `[σ => 28.0, ρ => 10.0, β => 8 / 3]`.
Looks like this did a good job at recovering them!
