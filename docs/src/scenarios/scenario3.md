# Scenario 3: Limiting Deaths

## Generate the Model and Dataset

```@example scenario3
prob = nothing
p_init = nothing # or box constraints
tsave, data = nothing, nothing
```

### Sample Model

```@example scenario3
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
@named seirhd = ODESystem(eqs)
seirhd = structural_simplify(seirhd)
prob = ODEProblem(seirhd, [], (0, 110.0))
sol = solve(prob)
plot(sol)
```

## Model Analysis

### Question 1

> Provide a forecast of cumulative Covid-19 cases and deaths over the 6-week period from May 1 – June 15, 2020 under no interventions, including 90% prediction intervals in your forecasts. Compare the accuracy of the forecasts with true data over the six-week timespan.

```@example seirhd
get_uncertainty_forecast(prob, cumulative_I, 0:100, [β₁ => Uniform(0.0, 1.0)], 6 * 7)
```

```@example seirhd
plot_uncertainty_forecast(prob, cumulative_I, 0:100, [β₁ => Uniform(0.0, 1.0)], 6 * 7)
```

```@example seirhd
get_uncertainty_forecast_quantiles(prob, cumulative_I, 0:100, [β₁ => Uniform(0.0, 1.0)],
                                   6 * 7)
```

```@example seirhd
plot_uncertainty_forecast_quantiles(prob, cumulative_I, 0:100, [β₁ => Uniform(0.0, 1.0)],
                                    6 * 7)
```

### Question 2

> Based on the forecasts, do we need additional interventions to keep cumulative Covid deaths under 6000 total? Provide a probability that the cumulative number of Covid deaths will stay under 6000 for the next 6 weeks without any additional interventions.

```@example seirhd
_prob = remake(prob, tspan = (0.0, 6*7.0))
prob_violating_treshold(_prob, [β₁ => Uniform(0.0, 1.0)], [cumulative_I > 0.4])
```

### Question 3

> We are interested in determining how effective it would be to institute a mandatory mask mandate for the duration of the next six weeks. What is the probability of staying below 6000 cumulative deaths if we institute an indefinite mask mandate starting May 1, 2020?

```@example seirhd
_prob = remake(_prob, p=[β₂ => 0.02])
prob_violating_treshold(_prob, [β₁ => Uniform(0.0, 1.0)], [cumulative_I > 0.4])
```

### Question 4

> We are interested in determining how detection rate can affect the accuracy and uncertainty in our forecasts. In particular, suppose we can improve the baseline detection rate by 20%, and the detection rate stays constant throughout the duration of the forecast. Assuming no additional interventions (ignoring Question 3), does that increase the amount of cumulative forecasted cases and deaths after six weeks? How does an increase in the detection rate affect the uncertainty in our estimates? Can you characterize the relationship between detection rate and our forecasts and their uncertainties, and comment on whether improving detection rates would provide decision-makers with better information (i.e., more accurate forecasts and/or narrower prediction intervals)?

> Compute the accuracy of the forecast assuming no mask mandate (ignoring Question 3) in the same way as you did in Question 1 and determine if improving the detection rate improves forecast accuracy.

### Question 5

> Convert the MechBayes SEIRHD model to an SIRHD model by removing the E compartment. Compute the same six-week forecast that you had done in Question 1a and compare the accuracy of the six-week forecasts with the forecasts done in Question 1a.

> Further modify the MechBayes SEIRHD model and do a model space exploration and model selection from the following models, based on comparing forecasts of cases and deaths to actual data: SEIRD, SEIRHD, and SIRHD models. Use data from April 1, 2020 – April 30, 2020 from the scenario location (Massachusetts) for fitting these models.  Then make out-of-sample forecasts from the same 6-week period from May 1 – June 15, 2020, and compare with actual data. Comment on the quality of the fit for each of these models.

> Do a 3-way structural model comparison between the SEIRD, SEIRHD, and SIRHD models.

### Question 7

> What is the latest date we can impose a mandatory mask mandate over the next six weeks to ensure, with 90% probability, that cumulative deaths do not exceed 6000? Can you characterize the following relationship: for every day that we delay implementing a mask mandate, we expect cumulative deaths (over the six-week timeframe) to go up by X?
