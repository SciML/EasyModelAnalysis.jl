# Scenario 3: Limiting Deaths

Load packages:

```@example scenario3
using EasyModelAnalysis
using AlgebraicPetri
using UnPack
```

## Generate the Model and Dataset

```@example scenario3
function formSEIRHD()
    SEIRHD = LabelledPetriNet([:S, :E, :I, :R, :H, :D],
        :expo => ((:S, :I) => (:E, :I)),
        :conv => (:E => :I),
        :rec => (:I => :R),
        :hosp => (:I => :H),
        :death => (:H => :D))
    return SEIRHD
end

seirhd = formSEIRHD()
sys1 = ODESystem(seirhd)
```

```@example scenario3
function formSEIRD()
    SEIRD = LabelledPetriNet([:S, :E, :I, :R, :D],
        :expo => ((:S, :I) => (:E, :I)),
        :conv => (:E => :I),
        :rec => (:I => :R),
        :death => (:I => :D))
    return SEIRD
end

seird = formSEIRD()
sys2 = ODESystem(seird)
```

```@example scenario3
function formSIRHD()
    SIRHD = LabelledPetriNet([:S, :I, :R, :H, :D],
        :expo => ((:S, :I) => (:I, :I)),
        :rec => (:I => :R),
        :hosp => (:I => :H),
        :death => (:H => :D))
    return SIRHD
end

sirhd = formSIRHD()
sys3 = ODESystem(sirhd)
```

```@example scenario3
function form_seird_renew()
    seird_renew = LabelledPetriNet([:S, :E, :I, :R, :D],
        :expo => ((:S, :I) => (:E, :I)),
        :conv => (:E => :I),
        :rec => (:I => :R),
        :death => (:I => :D),
        :renew => (:R => :S))
    return seird_renew
end

seird_renew = form_seird_renew()
sys4 = ODESystem(seird_renew)
```

```julia
using ASKEM # Hack, remove when merged
max_e_h = mca(seird, sirhd)
AlgebraicPetri.Graph(max_e_h[1])
```

```julia
max_3way = mca(max_e_h[1], seirhd)
AlgebraicPetri.Graph(max_3way[1])
```

```julia
max_seird_renew = mca(seird, seird_renew)
AlgebraicPetri.Graph(max_seird_renew[1])
```

```@example scenario3
t = ModelingToolkit.get_iv(sys1)
@unpack S, E, I, R, H, D = sys1
@unpack expo, conv, rec, hosp, death = sys1
NN = 10.0
@parameters u_expo=0.2 * NN u_conv=0.2 * NN u_rec=0.8 * NN u_hosp=0.2 * NN u_death=0.1 * NN N=NN
translate_params = [expo => u_expo / N,
    conv => u_conv / N,
    rec => u_rec / N,
    hosp => u_hosp / N,
    death => u_death / N]
subed_sys = substitute(sys1, translate_params)
sys = add_accumulations(subed_sys, [I])
@unpack accumulation_I = sys
```

```@example scenario3
u0init = [
    S => 0.9 * NN,
    E => 0.05 * NN,
    I => 0.01 * NN,
    R => 0.02 * NN,
    H => 0.01 * NN,
    D => 0.01 * NN
]

tend = 6 * 7
ts = 0:tend
prob = ODEProblem(sys, u0init, (0.0, tend))
sol = solve(prob)
plot(sol)
```

## Model Analysis

### Question 1

> Provide a forecast of cumulative Covid-19 cases and deaths over the 6-week period from May 1 – June 15, 2020 under no interventions, including 90% prediction intervals in your forecasts. Compare the accuracy of the forecasts with true data over the six-week timespan.

```@example scenario3
get_uncertainty_forecast(prob, [accumulation_I], ts, [u_conv => Uniform(0.0, 1.0)], 6 * 7)
```

```@example scenario3
plot_uncertainty_forecast(prob, [accumulation_I], ts, [u_conv => Uniform(0.0, 1.0)], 6 * 7)
```

```@example scenario3
get_uncertainty_forecast_quantiles(prob, [accumulation_I], ts,
    [u_conv => Uniform(0.0, 1.0)],
    6 * 7)
```

```@example scenario3
plot_uncertainty_forecast_quantiles(prob, [accumulation_I], ts,
    [u_conv => Uniform(0.0, 1.0)],
    6 * 7)
```

### Question 2

> Based on the forecasts, do we need additional interventions to keep cumulative Covid deaths under 6000 total? Provide a probability that the cumulative number of Covid deaths will stay under 6000 for the next 6 weeks without any additional interventions.

```@example scenario3
_prob = remake(prob, tspan = (0.0, 6 * 7.0))
prob_violating_threshold(_prob, [u_conv => Uniform(0.0, 1.0)], [accumulation_I > 0.4 * NN]) # TODO: explain 0.4*NN
```

### Question 3

> We are interested in determining how effective it would be to institute a mandatory mask mandate for the duration of the next six weeks. What is the probability of staying below 6000 cumulative deaths if we institute an indefinite mask mandate starting May 1, 2020?

```@example scenario3
_prob = remake(_prob, p = [u_expo => 0.02])
prob_violating_threshold(_prob, [u_conv => Uniform(0.0, 1.0)], [accumulation_I > 0.4 * NN])
```

### Question 4

> We are interested in determining how detection rate can affect the accuracy and uncertainty in our forecasts. In particular, suppose we can improve the baseline detection rate by 20%, and the detection rate stays constant throughout the duration of the forecast. Assuming no additional interventions (ignoring Question 3), does that increase the amount of cumulative forecasted cases and deaths after six weeks? How does an increase in the detection rate affect the uncertainty in our estimates? Can you characterize the relationship between detection rate and our forecasts and their uncertainties, and comment on whether improving detection rates would provide decision-makers with better information (i.e., more accurate forecasts and/or narrower prediction intervals)?

```@example scenario3
# these new equations add I->D and H->R  to the model.
# this says now, that all I are undetected and u_hosp is the detection rate.
# this assumes there is always hospital capacity
eqs2 = [Differential(t)(S) ~ -(u_expo / N) * I * S
        Differential(t)(E) ~ (u_expo / N) * I * S - (u_conv / N) * E
        Differential(t)(I) ~ (u_conv / N) * E - (u_hosp / N) * I - (u_rec / N) * I -
                             (u_death / N) * I
        Differential(t)(R) ~ (u_rec / N) * I + (u_rec / N) * H
        Differential(t)(H) ~ (u_hosp / N) * I - (u_death / N) * H - (u_rec / N) * H
        Differential(t)(D) ~ (u_death / N) * H + (u_death / N) * I]
@named seirhd_detect = ODESystem(eqs2)
sys2 = add_accumulations(seirhd_detect, [I])
u0init2 = [
    S => 0.9 * NN,
    E => 0.05 * NN,
    I => 0.01 * NN,
    R => 0.02 * NN,
    H => 0.01 * NN,
    D => 0.01 * NN
]
sys2_ = structural_simplify(sys2)
@unpack accumulation_I = sys2_

probd = ODEProblem(sys2_, u0init2, (0.0, tend))
sold = solve(probd; saveat = ts)
plot(sold)
```

```julia
sols = []
u_detects = 0:0.1:1
for x in u_detects
    probd = remake(probd, p = [u_hosp => x])
    sold = solve(probd; saveat = sold.t)
    push!(sols, sold)
end

# demonstrate that the total infected count is strictly decreasing with increasing detection rate
is = map(x -> x[accumulation_I][end], sols)
plot(is)
@test issorted(is; rev = true)

# deaths decrease with increasing detection rate
ds = map(x -> x[D][end], sols)
plot(ds)
@test issorted(ds; rev = true)
```

```julia
get_uncertainty_forecast(_prob, accumulation_I, 0:100,
    [u_hosp => Uniform(0.0, 1.0), u_conv => Uniform(0.0, 1.0)],
    6 * 7)

plot_uncertainty_forecast(probd, accumulation_I, 0:100,
    [
        u_hosp => Uniform(0.0, 1.0),
        u_conv => Uniform(0.0, 1.0)
    ],
    6 * 7)
```

> Compute the accuracy of the forecast assuming no mask mandate (ignoring Question 3) in the same way as you did in Question 1 and determine if improving the detection rate improves forecast accuracy.

### Question 5

> Convert the MechBayes SEIRHD model to an SIRHD model by removing the E compartment. Compute the same six-week forecast that you had done in Question 1a and compare the accuracy of the six-week forecasts with the forecasts done in Question 1a.

```julia
prob2 = prob
get_uncertainty_forecast(prob2, [accumulation_I], 0:100, [u_conv => Uniform(0.0, 1.0)],
    6 * 7)
```

```julia
plot_uncertainty_forecast(prob2, [accumulation_I], 0:100, [u_conv => Uniform(0.0, 1.0)],
    6 * 7)
```

```julia
get_uncertainty_forecast_quantiles(prob2, [accumulation_I], 0:100,
    [u_conv => Uniform(0.0, 1.0)],
    6 * 7)
```

```julia
plot_uncertainty_forecast_quantiles(prob2, [accumulation_I], 0:100,
    [u_conv => Uniform(0.0, 1.0)],
    6 * 7)
```

> Further modify the MechBayes SEIRHD model and do a model space exploration and model selection from the following models, based on comparing forecasts of cases and deaths to actual data: SEIRD, SEIRHD, and SIRHD models. Use data from April 1, 2020 – April 30, 2020 from the scenario location (Massachusetts) for fitting these models.  Then make out-of-sample forecasts from the same 6-week period from May 1 – June 15, 2020, and compare with actual data. Comment on the quality of the fit for each of these models.

```julia
prob3 = prob
get_uncertainty_forecast(prob2, [accumulation_I], 0:100, [u_conv => Uniform(0.0, 1.0)],
    6 * 7)
```

```julia
plot_uncertainty_forecast(prob2, [accumulation_I], 0:100, [u_conv => Uniform(0.0, 1.0)],
    6 * 7)
```

```julia
get_uncertainty_forecast_quantiles(prob2, [accumulation_I], 0:100,
    [u_conv => Uniform(0.0, 1.0)],
    6 * 7)
```

```julia
plot_uncertainty_forecast_quantiles(prob2, [accumulation_I], 0:100,
    [u_conv => Uniform(0.0, 1.0)],
    6 * 7)
```

> Do a 3-way structural model comparison between the SEIRD, SEIRHD, and SIRHD models.

```@example scenario3
#
```

### https://github.com/SciML/EasyModelAnalysis.jl/issues/22

### Question 7

> What is the latest date we can impose a mandatory mask mandate over the next six weeks to ensure, with 90% probability, that cumulative deaths do not exceed 6000? Can you characterize the following relationship: for every day that we delay implementing a mask mandate, we expect cumulative deaths (over the six-week timeframe) to go up by X?
