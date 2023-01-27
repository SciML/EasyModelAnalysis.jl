# Scenario 2: Limiting Hospitalizations

## Generate the Model and Dataset

```@example scenario2
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
prob = ODEProblem(seirhd, [], (0, 60.0), saveat = 1.0)
sol = solve(prob)
plot(sol)

data = [I => sol[I], R => sol[R], H => sol[H], D => sol[D]]
```

## Model Analysis

> Parameterize model either using data from the previous two months (October 28th – December 28th, 2021), or with relevant parameter values from the literature.

```@example scenario2
p_priors = [β₁=>Normal(0.6, 0.1), β₂=>Normal(0.143, 0.01), β₃=>Normal(0.055, 0.01), α=>Normal(0.003, 0.001), γ₁=>Normal(0.007, 0.001), γ₂=>Normal(0.011, 0.01), δ=>Normal(0.1, 0.05), μ=>Normal(0.14, 0.05)]
tsave = collect(0.0:1.0:2*30)
fit = bayesian_datafit(prob, p_priors, tsave, data)
prob = remake(prob; u0 = sol[end, :], p = Pair.(getfield.(fit, :first), mean.(getfield.(fit, :second))))
```

### Question 1

> Forecast Covid cases and hospitalizations over the next 3 months under no interventions.

```@example scenario2
forecast_threemonths = solve(prob, tspan = (0.0, 90.0), saveat = 1.0)
```

### Question 2

> Based on the forecast, do we need interventions to keep total Covid hospitalizations under a threshold of 3000 on any given day? If there is uncertainty in the model parameters, express the answer probabilistically, i.e., what is the likelihood or probability that the number of Covid hospitalizations will stay under this threshold for the next 3 months without interventions?

```@example scenario2
need_intervention = any(x -> x > 3000, forecast_threemonths[H])

posterior = Pair.(getfield.(fit,:first),Normal.(mean.(getfield.(fit,:second)),var.(getfield.(fit,:second))
prob_violating_treshold(prob, posterior, [H>3000])
```

### Question 3

> Assume a consistent policy of social distancing/masking will be implemented, resulting in a 50% decrease from baseline transmission. Assume that we want to minimize the time that the policy is in place, and once it has been put in place and then ended, it can't be re-implemented. Looking forward from “today’s” date of Dec. 28, 2021, what are the optimal start and end dates for this policy, to keep projections below the hospitalization threshold over the entire 3-month period? How many fewer hospitalizations and cases does this policy result in?

```@example scenario2
#root_eqs = [H ~ 3000] # or 80% of 3000 
#affect   = [η ~ η/2]
#@named mask_system = ODESystem(eqs, t; continuous_events = root_eqs => affect)

start_intervention = (t == tstart) => [η ~ η/2]
stop_intervention = (t == tstop) => [η ~ 2*η]
@named opttime_sys = ODESystem(eqs, t; discrete_events = [start_intervention, stop_intervention])
function f(ts, p = nothing)
  ts[2] - ts[1]
end

function g(res, ts, p = nothing)
  tstart = ts[1]
  tstop = ts[2]
  prob = ODEProblem(opttime_sys, [], (0.0, 90.0), saveat = 1.0)
  sol = solve(prob, idxs = [H])
  res .= copy(H) 
end

optprob = OptimizationFunction(f, AutoModelingToolkit(), cons = g)
prob = OptimizationProblem(optprob, [0.0, 90.0], lcons = fill(-Inf, 90), ucons = fill(3000.0, 90))
```


### Question 4

> Assume there is a protocol to kick in mitigation policies when hospitalizations rise above 80% of the hospitalization threshold (i.e. 80% of 3000). When hospitalizations fall back below 80% of the threshold, these policies expire.

> When do we expect these policies to first kick in?

> What is the minimum impact on transmission rate these mitigation policies need to have the first time they kick in, to (1) ensure that we don't reach the hospitalization threshold at any time during the 3-month period, and (2) ensure that the policies only need to be implemented once, and potentially expired later, but never reimplemented? Express this in terms of change in baseline transmission levels (e.g. 10% decrease, 50% decrease, etc.).

```@example scenario2

```

### Question 5

> Now assume that instead of NPIs, the Board wants to focus all their resources on an aggressive vaccination campaign to increase the fraction of the total population that is vaccinated. What is the minimum intervention with vaccinations required in order for this intervention to have the same impact on cases and hospitalizations, as your optimal answer from question 3? Depending on the model you use, this may be represented as an increase in total vaccinated population, or increase in daily vaccination rate (% of eligible people vaccinated each day), or some other representation.

```@example scenario2
@variables S(t)=0.9 E(t)=0.05 I(t)=0.01 R(t)=0.2 H(t)=0.1 D(t)=0.01
@variables T(t)=0.0 η(t)=0.0 cumulative_I(t)=0.0
@parameters β₁=0.6 β₂=0.143 β₃=0.055 β₄=0.8 α=0.003 γ₁=0.007 γ₂=0.011 δ=0.1 μ=0.14 ρ₁=0.5 ρ₂=0.5  
eqs = [T ~ S + E + I + R + H + D
       η ~ (β₁ * E + β₂ * I + β₃ * H - β₄ * V) / T
       Dₜ(S) ~ -η * S
       Dₜ(E) ~ η * S - α * E
       Dₜ(I) ~ α * E - (γ₁ + δ) * I
       Dₜ(cumulative_I) ~ I
       Dₜ(R) ~ γ₁ * I + γ₂ * H
       Dₜ(H) ~ δ * I - (μ + γ₂) * H
       Dₜ(D) ~ μ * H
       Dₜ(V) ~ ρ₁ * S – ρ₂ * V * I - V];
@named seirhd = ODESystem(eqs)
seirhd = structural_simplify(seirhd)
prob = ODEProblem(seirhd, [], (0, 60.0), saveat = 1.0)
sol = solve(prob; idxs = [I, R, H, D])
plot(sol)
```
