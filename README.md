# EasyModelAnalysis.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://docs.sciml.ai/EasyModelAnalysis/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://docs.sciml.ai/EasyModelAnalysis/dev/)

EasyModelAnalysis does exactly what it says: it makes model analysis easy. Want to know the first time
the number of infected individuals is about 1000? What is the probability that more than 50 people will
be infected given probability distributions for the parameters? What variables are the most sensitive?
Please find the parameters that best fit the model to the data. All of these, and more, given as simple
one-liner queries over SciML-defined differential equation models.

## Tutorials and Documentation

For information on using the package, see the [stable documentation](https://docs.sciml.ai/EasyModelAnalysis/stable/). 
Use the [in-development documentation](https://docs.sciml.ai/EasyModelAnalysis/dev/) for the version of the documentation 
which contains the unreleased features.

## Quick Demonstration

```julia
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

phaseplot_extrema(prob, x, (x, y))
plot_extrema(prob, x)
```

![](https://user-images.githubusercontent.com/1814174/214805423-2f79eb2b-a243-4c69-9aec-90cd16d67218.png)
![](https://user-images.githubusercontent.com/1814174/214805420-f1192965-e49e-458a-9c45-5fe86fdd3c80.png)
