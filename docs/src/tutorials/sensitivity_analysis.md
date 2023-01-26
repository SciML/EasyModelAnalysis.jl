# Sensitivity Analysis

In this tutorial we will showcase how to perform global sensitivity analysis of a model. To get started, let's first pull in our
modified second order ODE Lorenz equation model from before:

```@example sensitivity
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

On this model we wish to perform sensitivity analyses. Global sensitivity analysis requires the specification of two things:

1. Global sensitivity of what? What is the value to be measured from the model that you want to assess the sensitivity of?
2. Over what space of parameter values? This is a box of potential values for the parameters of the model.

The function `get_sensitivity(prob, t, obs, bounds)` first takes `(t,obs)`, where `t` is the time point to take measurements from
the model and `y` is the desired observable to measure. `bounds` is specified as an array of pairs which maps parameter symbols
to arrays specifying the lower and upper bounds of its parameter range.

Thus for example, let's calculate the sensitivity of `y(100)` over the parameters `(ρ, β)` where 
``\rho \in [0,20]`` and ``\beta \in [0,100]``: 

```@example sensitivity
pbounds = [ρ => [0.0, 20.0], β => [0.0, 100.0]]
sensres = get_sensitivity(prob, 100.0, y, pbounds)
```

The output shows values of `first_order`, `second_order` and `total_order` sensitivities. These are quantities that define the
nonlinear effect of the variables on the output. The first order values can be thought of as the independent interaction effects,
similar to the values of a linear regression for ``R^2``, i.e. it's the variance explained by independent effects of a given
parameter. The first indices are the scaled variance terms, for example:

```julia
ρ_first_order = V[y(100) changing only ρ] / V[y(100)]
```

where `V` denotes the variance. I.e., `ρ_first_order` is the percentage of variance explained by only changing `ρ`. Being normalized,
if the model is linear then `ρ_first_order + β_first_order == 1`, and thus its total summation tells us the degree of nonlinearity.
Our simulation here has the sum of first indices as `<0.2`, an indication of a high degree of nonlinear interaction between the measured
output and the parameters.

The second order indices then say how much can be attributed to changes of combinations of variables. I.e.:

```julia
ρ_β_second_order = V[y(100) changing only ρ and β] / V[y(100)]
```

which thus gives the percentage of variance explained by the nonlinearities of ρ and β combined. These sensitivity functions only 
output up to the second indices since there is a combinatorial explosion in the number of terms that need to be computed for models
with more parameters. However, in this tutorial there are only two variables, and thus all variance should be explained by just these
two parameters. This means that `ρ_first_order + β_first_order + ρ_β_second_order` should be approximately equal to 1, as all variance
should be explained by the linearity or second order interactions between the two variables. Let's check this in action:

```@example sensitivity
sensres[:ρ_first_order] + sensres[:β_first_order] + sensres[:ρ_β_second_order]
```

This is not exactly equal to 1 due to the numerical error in the integral approximations, but you can see theory in action!
(Also, this is a good test for correctness of the implementation).

Now if you had more than two variables, is there a good way to get a sense of the "sensitivity due to ρ"? Using the first indices
is a bad approximation to this value since it's only the measurement of the independent or linear sensitivity of the output due
to ρ. Instead what one would want to do is say get the sum of the first order index `ρ_first_order`, plus all second order effects
which include `ρ`, plus all third order effects that include `ρ`, plus ... all the way to the `N`th order for `N` variables.
Surprisingly, this summation can be computed without requiring the computation of all of the higher order indices. This is known
as the "total order index" of a variable. In a two parameter model, we can see this in action:

```@example sensitivity
sensres[:ρ_first_order] + sensres[:ρ_β_second_order], sensres[:ρ_total_order] 
```

```@example sensitivity
sensres[:β_first_order] + sensres[:ρ_β_second_order], sensres[:β_total_order]
```

Thus the total indices are a good measurement of the relative size of the total effect of each parameter on the solution of the
model. 

In summary:

- First order indices showcase the amount of linearity and the direct linear attributions to each variable
- The second order indices show the linear correlations in the outputs
- The total indices measure the total effect a given variable has on the variance of the output

and notably, all values are normalized relative quantities.

Thus we can finally use the `create_sensitivity_plot` function to visualize the field of sensitivity results:

```@example sensitivity
create_sensitivity_plot(prob, 100.0, y, pbounds)
```

which shows the relative sizes of the values in plots for the first, second, and total index values.