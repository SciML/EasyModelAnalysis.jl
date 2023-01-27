@time_imports using EasyModelAnalysis, AlgebraicPetri, UnPack # 91197.80000000003 ms
using Plots
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

@parameters u_expo=0.2 u_conv=0.2 u_rec=0.8 u_hosp=0.2 u_death=0.1 u_detect=0.5 N=NN
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

uncert_forecast = get_uncertainty_forecast(prob, accumulation_I, ts,
                                           [u_conv => Uniform(0.0, 1.0)], 6 * 7)

plot_uncertainty_forecast(prob, accumulation_I, ts, [u_conv => Uniform(0.0, 1.0)], 6 * 7)

qtiles = get_uncertainty_forecast_quantiles(prob, accumulation_I, ts,
                                            [u_conv => Uniform(0.0, 1.0)],
                                            6 * 7)

plt = plot_uncertainty_forecast_quantiles(prob, accumulation_I, ts,
                                          [u_conv => Uniform(0.0, 1.0)],
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

@parameters t

@variables I_undetected(t) I_detected(t)

seirhd_detect_eqs = [Differential(t)(S) ~ -(u_expo / N) * I * S
                     Differential(t)(E) ~ (u_expo / N) * I * S - (u_conv / N) * E
                     Differential(t)(I_undetected) ~ (u_conv / N) * E - (u_rec / N) * I -
                                                     (u_detect / N) * I_undetected
                     Differential(t)(I_detected) ~ (u_detect / N) * I_undetected -
                                                   (u_hosp / N) * I_detected
                     I ~ I_detected + I_undetected
                     Differential(t)(R) ~ (u_rec / N) * I
                     Differential(t)(H) ~ (u_hosp / N) * I_detected - (u_death / N) * H
                     Differential(t)(D) ~ (u_death / N) * H]
#  Differential(t)(accumulation_I) ~ I]

@named seirhd_detect = ODESystem(seirhd_detect_eqs)
sys2 = add_accumulations(seirhd_detect, [I])
u0init2 = [
    S => 0.9 * NN,
    E => 0.05 * NN,
    I_undetected => 0.01 * NN,
    I_detected => 0.0,
    R => 0.02 * NN,
    H => 0.01 * NN,
    D => 0.01 * NN,
]
sys2_ = structural_simplify(sys2)
probd = ODEProblem(sys2_, u0init2, (0.0, tend))
ufd = get_uncertainty_forecast(_prob, accumulation_I, 0:100, [u_conv => Uniform(0.0, 1.0)],
                               6 * 7)

plot_uncertainty_forecast(probd, accumulation_I, 0:100, [u_conv => Uniform(0.0, 1.0)],
                          6 * 7)

sold = solve(probd)
ts = sold.t
plot(sold)

# comparing infected counts between the two models
plot(sold[I] .- sol[I])

# a couple of cases to sanity check 
probd2 = remake(probd, p = [u_detect => Symbolics.getdefaultval(u_detect) * 1.2])
sold2 = solve(probd2; saveat = sold.t)

probd3 = remake(probd, p = [u_detect => 1])
sold3 = solve(probd3; saveat = sold.t)

probd4 = remake(probd, p = [u_detect => 0])
sold4 = solve(probd4; saveat = sold.t)

sols = []
ufs = []
ufqs = []
u_detecs = 0:0.1:1
for x in u_detecs
    probd = remake(probd, p = [u_detect => x])
    sold = solve(probd; saveat = sold.t)
    uf = get_uncertainty_forecast(probd, accumulation_I, 0:100,
                                  [u_conv => Uniform(0.0, 1.0)],
                                  6 * 7)
    q = get_uncertainty_forecast_quantiles(probd, accumulation_I, ts,
                                            [u_conv => Uniform(0.0, 1.0)],
                                            6 * 7)
    push!(ufs, uf)
    push!(ufqs, q)
    push!(sols, sold)
end

is = map(x -> x[accumulation_I][end], sols)
plot(is)
# demonstrate that the total infected count is strictly decreasing with increasing detection rate
@test issorted(is; rev = true)

spans = map(x->x[2] .- x[1], ufqs)
end_t_uncert = last.(spans)

# 👀: i think im doing something wrong here. i would assume the uncertainty should be decreasing with increasing detection rate
plot(end_t_uncert)
sortperm(end_t_uncert)

q1 = get_uncertainty_forecast_quantiles(probd, accumulation_I, ts,
                                        [u_conv => Uniform(0.0, 1.0)],
                                        6 * 7)
pq1 = plot_uncertainty_forecast_quantiles(probd, accumulation_I, ts,
                                          [u_conv => Uniform(0.0, 1.0)],
                                          6 * 7)
title!(pq1)
q2 = get_uncertainty_forecast_quantiles(probd2, accumulation_I, ts,
                                        [u_conv => Uniform(0.0, 1.0)],
                                        6 * 7)
pq2 = plot_uncertainty_forecast_quantiles(probd2, accumulation_I, ts,
                                          [u_conv => Uniform(0.0, 1.0)],
                                          6 * 7)
q3 = get_uncertainty_forecast_quantiles(probd3, accumulation_I, 0:100,
                                        [u_conv => Uniform(0.0, 1.0)],
                                        6 * 7)
pq3 = plot_uncertainty_forecast_quantiles(probd3, accumulation_I, ts,
                                          [u_conv => Uniform(0.0, 1.0)],
                                          6 * 7)
q4 = get_uncertainty_forecast_quantiles(probd4, accumulation_I, 0:100,
                                        [u_conv => Uniform(0.0, 1.0)],
                                        6 * 7)

ufd2 = get_uncertainty_forecast(probd, accumulation_I, 0:100, [u_conv => Uniform(0.0, 1.0)],
                                6 * 7)

plot_uncertainty_forecast(probd, accumulation_I, 0:100, [u_conv => Uniform(0.0, 1.0)],
                          6 * 7)

# todo: compare with the previous forecast uncertainties 

# question 5
# > Convert the MechBayes SEIRHD model to an SIRHD model by removing the E compartment. 
# Compute the same six-week forecast that you had done in Question 1a and compare the accuracy of the six-week 
# forecasts with the forecasts done in Question 1a.

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
# question 6

# > Further modify the MechBayes SEIRHD model and do a model space exploration and model selection from the following models, 
# based on comparing forecasts of cases and deaths to actual data: SEIRD, SEIRHD, and SIRHD models. 

# Use data from April 1, 2020 – April 30, 2020 from the scenario location (Massachusetts) for fitting these models.  

# Then make out-of-sample forecasts from the same 6-week period from May 1 – June 15, 2020, and compare with actual data. 
# Comment on the quality of the fit for each of these models.
# > Do a 3-way structural model comparison between the SEIRD, SEIRHD, and SIRHD models.

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

# question 7
# > What is the latest date we can impose a mandatory mask mandate over the next six weeks to ensure, with 90% probability, 
# that cumulative deaths do not exceed 6000? 
# Can you characterize the following relationship: for every day that we delay implementing a mask mandate, 
# we expect cumulative deaths (over the six-week timeframe) to go up by X?
