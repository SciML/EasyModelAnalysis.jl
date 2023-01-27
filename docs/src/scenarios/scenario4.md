# Scenario 4: Testing and Return to Campus

## Generate the Model and Dataset

```@example scenario4
prob = nothing
p_init = nothing # or box constraints
tsave, data = nothing, nothing
```

## Model Analysis

### Question 1

> Define a return-to-campus strategy that minimizes total testing while maintaining infections below the initial isolation bed capacity of 430. The testing scheme can include an arrival testing strategy in addition to unique testing approaches within time periods of the simulation. Cohorts can have unique testing strategies defined by test type and number per week.

#### https://github.com/SciML/EasyModelAnalysis.jl/issues/88

### Question 2

> The model will need to include cohort stratification and appropriate treatment of the testing campaign as an intervention. The user will need to produce and maintain distinct cohorts for undergraduate student population, graduate/professional student population, and employee population.

> The testing campaign needs to be implemented on a weekly cadence with the ability to modulate the number and type of tests applied across cohort. For a simplification, we can start with just one test type.
