# Analysis of Interventions On The SEIRHD Epidemic Model

First, let's implement the classic SEIRHD epidemic model with ModelingToolkit:

```@example seirhd
using EasyModelAnalysis
@variables t
Dₜ = Differential(t)
@variables S(t)=0.9 E(t)=0.05 I(t)=0.01 R(t)=0.2 H(t)=0.1 D(t)=0.01
@variables T(t)=0.0 η(t)=0.0 cumulative_I(t)=0.0
@parameters β₁=0.6 β₂=0.143 β₃=0.055 α=0.003 γ₁=0.007 γ₂=0.011 δ=0.1 μ=0.14
eqs = [T ~ S + E + I + R + H + D
       η ~ (β₁ * E + β₂ * I + β₃ * H) / T
       Dₜ(S) ~ -η * S
       Dₜ(E) ~ η * S - α * E
       Dₜ(I) ~ α * E - (γ₁ + δ) * I
       Dₜ(cumulative_I) ~ I
       Dₜ(R) ~ γ₁ * I + γ₂ * H
       Dₜ(H) ~ δ * I - (μ + γ₂) * H
       Dₜ(D) ~ μ * H];
@named seirhd = ODESystem(eqs, t)
seirhd = structural_simplify(seirhd)
prob = ODEProblem(seirhd, [], (0, 110.0))
sol = solve(prob)
plot(sol)
```

Let's solve a few problems:

> Provide a forecast of cumulative Covid-19 cases and deaths over the 6-week period from May 1 – June 15, 2020 under no interventions, including 90% prediction intervals in your forecasts. Compare the accuracy of the forecasts with true data over the six-week timespan.

```@example seirhd
get_uncertainty_forecast(prob, [cumulative_I], 0:100, [β₁ => Uniform(0.0, 1.0)], 6 * 7)
```

```@example seirhd
plot_uncertainty_forecast(prob, [cumulative_I], 0:100, [β₁ => Uniform(0.0, 1.0)], 6 * 7)
```

```@example seirhd
get_uncertainty_forecast_quantiles(prob, [cumulative_I], 0:100, [β₁ => Uniform(0.0, 1.0)],
    6 * 7)
```

```@example seirhd
plot_uncertainty_forecast_quantiles(prob, [cumulative_I], 0:100, [β₁ => Uniform(0.0, 1.0)],
    6 * 7)
```

> Based on the forecasts, do we need additional interventions to keep cumulative Covid deaths under 6000 total? Provide a probability that the cumulative number of Covid deaths will stay under 6000 for the next 6 weeks without any additional interventions.
