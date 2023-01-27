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

### Question 4

> Challenge question: assume that antigen tests are one fifth the cost of PCR tests but also much less (~half) as sensitive. Incorporate the cost of the testing program into your recommendations.

#### https://github.com/SciML/EasyModelAnalysis.jl/issues/88