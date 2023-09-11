using EasyModelAnalysis, LinearAlgebra, Test

@parameters t β=0.05 c=10.0 γ=0.25
@variables S(t)=990.0 I(t)=10.0 R(t)=0.0
∂ = Differential(t)
N = S + I + R # This is recognized as a derived variable
eqs = [∂(S) ~ -β * c * I / N * S,
    ∂(I) ~ β * c * I / N * S - γ * I,
    ∂(R) ~ γ * I];

@named sys = ODESystem(eqs);
tspan = (0, 30)
prob = ODEProblem(sys, [], tspan);

@parameters t β=0.1 c=10.0 γ=0.25 ρ=0.1 h=0.1 d=0.1 r=0.1
@variables S(t)=990.0 I(t)=10.0 R(t)=0.0 H(t)=0.0 D(t)=0.0
∂ = Differential(t)
N = S + I + R + H + D # This is recognized as a derived variable
eqs = [∂(S) ~ -β * c * I / N * S,
    ∂(I) ~ β * c * I / N * S - γ * I - h * I - ρ * I,
    ∂(R) ~ γ * I + r * H,
    ∂(H) ~ h * I - r * H - d * H,
    ∂(D) ~ ρ * I + d * H];

@named sys2 = ODESystem(eqs);

prob2 = ODEProblem(sys2, [], tspan);

@parameters t β=0.1 c=10.0 γ=0.25 ρ=0.1 h=0.1 d=0.1 r=0.1 v=0.1
@parameters t β2=0.1 c2=10.0 ρ2=0.1 h2=0.1 d2=0.1 r2=0.1
@variables S(t)=990.0 I(t)=10.0 R(t)=0.0 H(t)=0.0 D(t)=0.0
@variables Sv(t)=0.0 Iv(t)=0.0 Rv(t)=0.0 Hv(t)=0.0 Dv(t)=0.0
@variables I_total(t)

∂ = Differential(t)
N = S + I + R + H + D + Sv + Iv + Rv + Hv + Dv # This is recognized as a derived variable
eqs = [∂(S) ~ -β * c * I_total / N * S - v * Sv,
    ∂(I) ~ β * c * I_total / N * S - γ * I - h * I - ρ * I,
    ∂(R) ~ γ * I + r * H,
    ∂(H) ~ h * I - r * H - d * H,
    ∂(D) ~ ρ * I + d * H,
    ∂(Sv) ~ -β2 * c2 * I_total / N * Sv + v * Sv,
    ∂(Iv) ~ β2 * c2 * I_total / N * Sv - γ * Iv - h2 * Iv - ρ2 * Iv,
    ∂(Rv) ~ γ * I + r2 * H,
    ∂(Hv) ~ h2 * I - r2 * H - d2 * H,
    ∂(Dv) ~ ρ2 * I + d2 * H,
    I_total ~ I + Iv,
];

@named sys3 = ODESystem(eqs)
sys3 = structural_simplify(sys3)
prob3 = ODEProblem(sys3, [], tspan);
enprob = EnsembleProblem([prob, prob2, prob3])

sol = solve(enprob; saveat = 1);

weights = [0.2, 0.5, 0.3]

fullS = vec(sum(stack(weights .* sol[:, S]), dims = 2))
fullI = vec(sum(stack(weights .* sol[:, I]), dims = 2))
fullR = vec(sum(stack(weights .* sol[:, R]), dims = 2))

t_train = 0:14
data_train = [
    S => (t_train, fullS[1:15]),
    I => (t_train, fullI[1:15]),
    R => (t_train, fullR[1:15]),
]
t_ensem = 0:21
data_ensem = [
    S => (t_ensem, fullS[1:22]),
    I => (t_ensem, fullI[1:22]),
    R => (t_ensem, fullR[1:22]),
]
t_forecast = 0:30
data_forecast = [
    S => (t_forecast, fullS),
    I => (t_forecast, fullI),
    R => (t_forecast, fullR),
]

sol = solve(enprob; saveat = t_ensem);

@test ensemble_weights(sol, data_ensem) ≈ [0.2, 0.5, 0.3]

probs = (prob, prob2, prob3)
ps = Tuple([β => Uniform(0.01, 10.0), γ => Uniform(0.01, 10.0)] for i in 1:3)
datas = (data_train, data_train, data_train)
enprobs = bayesian_ensemble(probs, ps, datas, nchains = 2, niter = 200)

sol = solve(enprobs; saveat = t_ensem);
ensemble_weights(sol, data_ensem)

# only supports one datas
ensembleofweightedensembles = bayesian_datafit(probs,
    ps,
    data_train,
    nchains = 2,
    niter = 200)

@test length(ensembleofweightedensembles.prob[1].prob) ==
      length(ensembleofweightedensembles.prob[1].weights) == length(ps)
for prob in ensembleofweightedensembles.prob
    @test sum(prob.weights) ≈ 1.0
end
