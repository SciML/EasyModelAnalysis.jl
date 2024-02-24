using EasyModelAnalysis, Test

@parameters t σ ρ β
@variables x(t) y(t) z(t)
D = Differential(t)

eqs = [D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@named sys = ODESystem(eqs)
sys = structural_simplify(sys)

u0 = [D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0]

p = [σ => 28.0,
    ρ => 10.0,
    β => 8 / 3]

tspan = (0.0, 100.0)
prob = ODEProblem(sys, u0, tspan, p, jac = true)
sol = solve(prob)

tsave = [1.0, 2.0, 3.0]
sol_data = solve(prob, saveat = tsave)
data = [x => sol_data[x], z => sol_data[z]]
psub_ini = [σ => 27.0, β => 3.0]
fit = datafit(prob, psub_ini, tsave, data)
pvals_fit = getfield.(fit, :second)
pvals = getfield.(p, :second)[[1, 3]]
@test isapprox(pvals, pvals_fit, atol = 1e-4, rtol = 1e-4)

tsave1 = [1.0, 2.0, 3.0]
sol_data1 = solve(prob, saveat = tsave1)
tsave2 = [0.5, 1.5, 2.5, 3.5]
sol_data2 = solve(prob, saveat = tsave2)
data_with_t = [x => (tsave1, sol_data1[x]), z => (tsave2, sol_data2[z])]

fit = datafit(prob, psub_ini, data_with_t)
pvals_fit = getfield.(fit, :second)
pvals = getfield.(p, :second)[[1, 3]]
@test isapprox(pvals, pvals_fit, atol = 1e-4, rtol = 1e-4)

prob2 = remake(prob, p = fit)
prob3 = remake(prob, p = psub_ini)
scores = model_forecast_score([prob, prob2, prob3], tsave, data)
@test scores[1] == 0
@test scores[2] < 2e-3
@test scores[3] > 10

psub_ini = [σ => [27.0, 29.0], β => [2.0, 3.0]]
fit = global_datafit(prob, psub_ini, tsave, data)
pvals_fit = getfield.(fit, :second)
pvals = getfield.(p, :second)[[1, 3]]
@test isapprox(pvals, pvals_fit, atol = 1e-4, rtol = 1e-4)

fit = global_datafit(prob, psub_ini, data_with_t)
pvals_fit = getfield.(fit, :second)
pvals = getfield.(p, :second)[[1, 3]]
@test isapprox(pvals, pvals_fit, atol = 1e-4, rtol = 1e-4)

@variables x_2(t)
eqs_obs = [D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z,
    x_2 ~ 2 * x]

@named sys_obs = ODESystem(eqs_obs)
sys_obs = structural_simplify(sys_obs)

u0_obs = [D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0,
    x_2 => 2.0]

prob_obs = ODEProblem(sys_obs, u0_obs, tspan, p, jac = true)
sol_data_obs = solve(prob_obs, saveat = tsave)
data_obs = [x_2 => sol_data_obs[x_2], z => sol_data_obs[z]]
fit_obs = global_datafit(prob_obs, psub_ini, tsave, data_obs)
pvals_fit_obs = getfield.(fit_obs, :second)
@test isapprox(pvals, pvals_fit_obs, atol = 1e-4, rtol = 1e-4)

tsave = collect(10.0:10.0:100.0)
sol_data = solve(prob, saveat = tsave)
data = [x => sol_data[x], z => sol_data[z]]
p_prior = [σ => Normal(26.8, 0.1), β => Normal(2.7, 0.1)]
p_posterior = @time bayesian_datafit(prob, p_prior, tsave, data)
@test var.(getfield.(p_prior, :second)) >= var.(getfield.(p_posterior, :second))

tsave1 = collect(10.0:10.0:100.0)
sol_data1 = solve(prob, saveat = tsave1)
tsave2 = collect(10.0:13.5:100.0)
sol_data2 = solve(prob, saveat = tsave2)
data_with_t = [x => (tsave1, sol_data1[x]), z => (tsave2, sol_data2[z])]

p_posterior = @time bayesian_datafit(prob, p_prior, data_with_t)
@test var.(getfield.(p_prior, :second)) >= var.(getfield.(p_posterior, :second))
