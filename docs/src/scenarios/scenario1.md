# Scenario 1: Vaccination

## Generate the Model and Dataset

```@example scenario1
prob = nothing
p_init = nothing # or box constraints
tsave, data = nothing, nothing
```

## Model Analysis

### Question 3 Numerical Comparison

> Compare simulation outputs between the three models, for the following two scenarios. Assume initial values and parameter values are consistent (to the extent possible) with Table 1 in https://biomedres.us/pdfs/BJSTR.MS.ID.007413.pdf. For initial values that are not specified, choose reasonable values and ensure they are the same between the three models being compared.
> i.	Vaccine efficacy = 75%, population vaccinated = 10%
> ii.	Vaccine efficacy = 75%, population vaccinated = 80%

### Question 4

> Create an equally weighted ensemble model using the three models in 3b, and replicate the scenarios in 3.c.i and 3.c.ii. How does the ensemble model output compare to the output from the individual component models?

### Question 5

> For any of the models in question 3, conduct a sensitivity analysis to determine which intervention parameters should be prioritized in the model, for having the greatest impact on deaths – NPIs, or vaccine-related interventions?

### Question 6

> With the age-stratified model, simulate the following situations. You may choose initial values that seem reasonable given the location and time, and you can reuse values from any of the publications referenced):
> i.	High vaccination rate among older populations 65 years and older (e.g. 80%+), and low vaccination rate among all other age groups (e.g. below 15%)
> ii.	High vaccination rate among all age groups
> iii.	Repeat d.i and d.ii, but now add a social distancing policy at schools, that decreases contact rates by 20% for school-aged children only.
> iv.	Compare and summarize simulation outputs for d.i-d.iii
