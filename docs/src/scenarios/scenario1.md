# Scenario 1: Vaccination

## Generate the Model and Dataset

```@example scenario1
prob = nothing
p_init = nothing # or box constraints
tsave, data = nothing, nothing
```

## Model Analysis

> Parameterize model either using data from the previous two months (October 28th – December 28th, 2021), or with relevant parameter values from the literature. 

```julia
fit = datafit(prob, p_init, tsave, data)
```

> Forecast Covid cases and hospitalizations over the next 3 months under no interventions.