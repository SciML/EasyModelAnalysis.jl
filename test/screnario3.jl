@time_imports using EasyModelAnalysis, AlgebraicPetri, UnPack # 91197.80000000003 ms

# sys equations
# Differential(t)(S(t)) ~ -(u_expo / N)*I(t)*S(t)
# Differential(t)(E(t)) ~ (u_expo / N)*I(t)*S(t) - (u_conv / N)*E(t)
# Differential(t)(I(t)) ~ (u_conv / N)*E(t) - (u_hosp / N)*I(t) - (u_rec / N)*I(t)
# Differential(t)(R(t)) ~ (u_rec / N)*I(t)
# Differential(t)(H(t)) ~ (u_hosp / N)*I(t) - (u_death / N)*H(t)
# Differential(t)(D(t)) ~ (u_death / N)*H(t)
# Differential(t)(accumulation_I(t)) ~ I(t)

function formSEIRHD()
    SEIRHD = LabelledPetriNet([:S, :E, :I, :R, :H, :D],
                              :expo => ((:S, :I) => (:E, :I)),
                              :conv => (:E => :I),
                              :rec => (:I => :R),
                              :hosp => (:I => :H),
                              :death => (:H => :D))
    return SEIRHD
end
sys1 = ODESystem(formSEIRHD())

@unpack S, E, I, R, H, D = sys1
@unpack expo, conv, rec, hosp, death = sys1
NN = 10.0
# u_conv, 
@parameters u_expo=0.2 u_conv=0.2 u_rec=0.8 u_hosp=0.2 u_death=0.1 N=NN
translate_params = [expo => u_expo / N,
    conv => u_conv / N,
    rec => u_rec / N,
    hosp => u_hosp / N,
    death => u_death / N]
subed_sys = substitute(sys1, translate_params)
sys = add_accumulations(subed_sys, [I])
@unpack accumulation_I = sys

u0init = [
    S => 0.9 * NN,
    E => 0.05 * NN,
    I => 0.01 * NN,
    R => 0.02 * NN,
    H => 0.01 * NN,
    D => 0.01 * NN,
]

# question 1 
# > Provide a forecast of cumulative Covid-19 cases and deaths over the 6-week period from 
# May 1 – June 15, 2020 under no interventions, including 90% prediction intervals in your forecasts. 
# Compare the accuracy of the forecasts with true data over the six-week timespan.

tend = 6 * 7 # 6 weeks
ts = 0:tend
prob = ODEProblem(sys, u0init, (0.0, tend))
sol = solve(prob)
plot(sol)

uncert_forecast = get_uncertainty_forecast(prob, accumulation_I, ts, [u_conv => Uniform(0.0, 1.0)], 6 * 7)

plot_uncertainty_forecast(prob, accumulation_I, ts, [u_conv => Uniform(0.0, 1.0)], 6 * 7)

qtiles = get_uncertainty_forecast_quantiles(prob, accumulation_I, ts, [u_conv => Uniform(0.0, 1.0)],
                                   6 * 7)
                                   
plt = plot_uncertainty_forecast_quantiles(prob, accumulation_I, ts, [u_conv => Uniform(0.0, 1.0)],
                                    6 * 7)
plot!(plt, sol, vars = [accumulation_I])

# question 2 
# > Based on the forecasts, do we need additional interventions to keep cumulative Covid deaths under 6000 total? 
# Provide a probability that the cumulative number of Covid deaths will stay under 6000 for the next 6 weeks 
# without any additional interventions.

_prob = remake(prob, tspan = (0.0, 6 * 7.0))
prob_violating_threshold(_prob, [u_conv => Uniform(0.0, 1.0)], [accumulation_I > 0.4 * NN]) # TODO: explain 0.4*NN

# question 3
# > We are interested in determining how effective it would be to institute a mandatory mask mandate for 
# the duration of the next six weeks. What is the probability of staying below 6000 cumulative deaths if 
# we institute an indefinite mask mandate starting May 1, 2020?

_prob = remake(_prob, p = [u_expo => 0.02]) # here we assume the exposure parameter was decreased 10 times (0.2 -> 0.02)
prob_violating_threshold(_prob, [u_conv => Uniform(0.0, 1.0)], [accumulation_I > 0.4 * NN])

# question 4
# > We are interested in determining how detection rate can affect the accuracy and uncertainty in our forecasts. 
# In particular, suppose we can improve the baseline detection rate by 20%, and the detection rate stays constant 
# throughout the duration of the forecast. 

# Assuming no additional interventions (ignoring Question 3), does that increase the amount of cumulative forecasted 
# cases and deaths after six weeks? 

# How does an increase in the detection rate affect the uncertainty in our estimates? 

# Can you characterize the relationship between detection rate and our forecasts and their uncertainties, 
# and comment on whether improving detection rates would provide decision-makers with better information 
# (i.e., more accurate forecasts and/or narrower prediction intervals)?

_prob = remake(prob, p = [β₃ => 0.015])
get_uncertainty_forecast(_prob, accumulation_I, 0:100, [u_conv => Uniform(0.0, 1.0)], 6 * 7)

plot_uncertainty_forecast(_prob, accumulation_I, 0:100, [u_conv => Uniform(0.0, 1.0)],
                          6 * 7)

# question 5
prob2 = prob
get_uncertainty_forecast(prob2, accumulation_I, 0:100, [u_conv => Uniform(0.0, 1.0)], 6 * 7)

plot_uncertainty_forecast(prob2, accumulation_I, 0:100, [u_conv => Uniform(0.0, 1.0)],
                          6 * 7)

get_uncertainty_forecast_quantiles(prob2, accumulation_I, 0:100,
                                   [u_conv => Uniform(0.0, 1.0)],
                                   6 * 7)
plot_uncertainty_forecast_quantiles(prob2, accumulation_I, 0:100,
                                    [u_conv => Uniform(0.0, 1.0)],
                                    6 * 7)

prob3 = prob
get_uncertainty_forecast(prob2, accumulation_I, 0:100, [u_conv => Uniform(0.0, 1.0)], 6 * 7)

plot_uncertainty_forecast(prob2, accumulation_I, 0:100, [u_conv => Uniform(0.0, 1.0)],
                          6 * 7)

get_uncertainty_forecast_quantiles(prob2, accumulation_I, 0:100,
                                   [u_conv => Uniform(0.0, 1.0)],
                                   6 * 7)

plot_uncertainty_forecast_quantiles(prob2, accumulation_I, 0:100,
                                    [u_conv => Uniform(0.0, 1.0)],
                                    6 * 7)