using EasyModelAnalysis, Test
using ModelingToolkit: t_nounits as t, D_nounits as D

@parameters α β γ δ
@variables x(t) y(t)

eqs = [D(x) ~ α * x - β * x * y,
    D(y) ~ -γ * y + δ * x * y]

@mtkbuild sys = ODESystem(eqs, t)

u0 = [x => 1.0,
    y => 1.0]

p = [α => 2 / 3,
    β => 4 / 3,
    γ => 1,
    δ => 1]

tspan = (0.0, 50.0)
prob = ODEProblem(sys, u0, tspan, p, jac = true)
sol = solve(prob)

tsave = collect(1.0:1:10)
sol_data = solve(prob, saveat = tsave)
data = [x => sol_data[x], y => sol_data[y]]
psub_ini = [α => 1.0,
    β => 1.2,
    γ => 0.9,
    δ => 0.8]
fit = datafit(prob, psub_ini, tsave, data)
pvals_fit = getfield.(fit, :second)
pvals = getfield.(p, :second)
@test isapprox(pvals, pvals_fit, atol = 1e-4, rtol = 1e-4)

tsave1 = collect(1.0:1:10)
sol_data1 = solve(prob, saveat = tsave1)
tsave2 = collect(0.5:1:5)
sol_data2 = solve(prob, saveat = tsave2)
data_with_t = [x => (tsave1, sol_data1[x]), y => (tsave2, sol_data2[y])]

fit = datafit(prob, psub_ini, data_with_t)
pvals_fit = getfield.(fit, :second)
pvals = getfield.(p, :second)
@test isapprox(pvals, pvals_fit, atol = 1e-4, rtol = 1e-4)

prob2 = remake(prob, p = fit)
prob3 = remake(prob, p = psub_ini)
scores = model_forecast_score([prob, prob2, prob3], tsave, data)
@test scores[1] == 0
@test scores[2] < 2e-3
@test scores[3] > 2

psub_ini = [
    α => [0.5, 0.9],
    β => [0.9, 1.5],
    γ => [0.7, 1.4],
    δ => [0.5, 1.5]
]
fit = global_datafit(prob, psub_ini, tsave, data)
pvals_fit = getfield.(fit, :second)
pvals = getfield.(p, :second)
@test isapprox(pvals, pvals_fit, atol = 1e-4, rtol = 1e-4)

fit = global_datafit(prob, psub_ini, data_with_t)
pvals_fit = getfield.(fit, :second)
pvals = getfield.(p, :second)
@test isapprox(pvals, pvals_fit, atol = 1e-4, rtol = 1e-4)

@parameters α β γ δ
@variables x(t) y(t) x_2(t)

eqs_obs = [D(x) ~ α * x - β * x * y,
    D(y) ~ -γ * y + δ * x * y,
    x_2 ~ 2 * x]

@mtkbuild sys_obs = ODESystem(eqs_obs, t)

u0_obs = [
    x => 1.0,
    y => 1.0,
    x_2 => 2.0]

prob_obs = ODEProblem(sys_obs, u0_obs, tspan, p, jac = true)
sol_data_obs = solve(prob_obs, saveat = tsave)
data_obs = [x_2 => sol_data_obs[x_2], y => sol_data_obs[y]]
fit_obs = global_datafit(prob_obs, psub_ini, tsave, data_obs)
pvals_fit_obs = getfield.(fit_obs, :second)
@test isapprox(pvals, pvals_fit_obs, atol = 1e-4, rtol = 1e-4)

tsave = collect(1.0:1.0:10.0)
sol_data = solve(prob, saveat = tsave)
data = [x => sol_data[x], y => sol_data[y]]

p_prior = [α => Normal(2 / 3, 0.1), β => Normal(4 / 3, 0.1),
    γ => Normal(1, 0.1), δ => Normal(1, 0.1)]
p_posterior = @time bayesian_datafit(prob, p_prior, tsave, data, niter = 3000)
@test var.(getfield.(p_prior, :second)) >= var.(getfield.(p_posterior, :second))

tsave1 = collect(1.0:1.0:10.0)
sol_data1 = solve(prob, saveat = tsave1)
tsave2 = collect(1.0:2.0:10.0)
sol_data2 = solve(prob, saveat = tsave2)
data_with_t = [x => (tsave1, sol_data1[x]), y => (tsave2, sol_data2[y])]

p_posterior = @time bayesian_datafit(prob, p_prior, data_with_t, niter = 5000)
@test var.(getfield.(p_prior, :second)) >= var.(getfield.(p_posterior, :second))
