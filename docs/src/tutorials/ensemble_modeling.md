# Ensemble Modeling

Ensemble modeling is the process of building predictors which are combinations of
predictive models. In this tutorial we will show how to use EMA.jl to build such
ensemble models.

## The Predictive Models

For this tutorial we will use a set of SIR-type models as the basis. In particular,
we will use a basic SIR model, an SIRHD, and an SIRHD model with vaccintation. The
construction of the models is as follows:

```@example ensemble
using EasyModelAnalysis, LinearAlgebra

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
```

## Representing Ensemble Models with the SciML EnsembleProblem

The SciML libraries allow for what's known as an `EnsembleProblem`, which is an object that
solves many simultainous problems and represents the aggregate solution. This object is
documented
[in the DifferentialEquations.jl documentation](https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/)
and has all kinds of features, such as automated GPU acceleration, though we will instead
focus just on the subset of features required for this demonstration. To build an
EnsembleProblem, the main object is the `prob_func`, which is a function of `(prob,i,repeat)`
which describes what the `i`th problem should be. The `prob` in this case is a
prototype problem, which we are effectively ignoring for our use case.

Thus a simple `EnsembleProblem` which ensembles the three models built above is as follows:

```@example ensemble
probs = [prob, prob2, prob3]
enprob = EnsembleProblem(probs)
```

Here, `prob_func` returns model `i` on the `i`th iteration, and thus if we solve with
3 trajectories we will get the solution to all three models. This looks like:

```@example ensemble
sol = solve(enprob; saveat = 1);
```

We can access the 3 solutions as `sol[i]` respectively. Let's get the time series
for `S` from each of the models:

```@example ensemble
sol[:,S]
```

## Building a Dataset

Now let's build a dataset from our ensemble model. We will make our dataset for `S`,
`I`, and `R` by taking a linear combination of our models and using the aforementioned
interface on the ensemble solution.

```@example ensemble
weights = [0.2, 0.5, 0.3]
data = [
    S => vec(sum(stack(weights .* sol[:,S]), dims = 2)),
    I => vec(sum(stack(weights .* sol[:,I]), dims = 2)),
    R => vec(sum(stack(weights .* sol[:,R]), dims = 2)),
]
```

```@example ensemble
plot(sol; idxs = S)
scatter!(data[1][2])
```

```@example ensemble
plot(sol; idxs = I)
scatter!(data[2][2])
```

```@example ensemble
plot(sol; idxs = R)
scatter!(data[3][2])
```

Now let's split that into training, ensembling, and forecast sections:

```@example ensemble
fullS = vec(sum(stack(weights .* sol[:,S]),dims=2))
fullI = vec(sum(stack(weights .* sol[:,I]),dims=2))
fullR = vec(sum(stack(weights .* sol[:,R]),dims=2))

t_train = 0:14
data_train = [
    S => (t_train,fullS[1:15]),
    I => (t_train,fullI[1:15]),
    R => (t_train,fullR[1:15]),
]
t_ensem = 0:21
data_train = [
    S => (t_ensem,fullS[1:22]),
    I => (t_ensem,fullI[1:22]),
    R => (t_ensem,fullR[1:22]),
]
t_forecast = 0:30
data_forecast = [
    S => (t_forecast,fullS),
    I => (t_forecast,fullI),
    R => (t_forecast,fullR),
]
```

## Bayesian Calibration

Now let's perform a Bayesian calibration on each of the models. This gives us multiple parameterizations for each model, which then gives an ensemble which is `parameterizations x models` in size.

```@example ensemble
probs = [prob, prob2, prob3]
ps = [[β => Uniform(0.01, 10.0), γ => Uniform(0.01, 10.0)] for i in 1:3]
datas = [data_train,data_train,data_train]
enprobs = bayesian_ensemble(probs, ps, datas)
```

Let's see how each of our models in the ensemble compare against the data when changed
to use the fit parameters:

```@example ensemble
sol = solve(enprobs);

plot(sol; idxs = S)
scatter!(t_train, data_train[1][2][2])
```

```@example ensemble
plot(sol; idxs = I)
scatter!(t_train, data_train[2][2][2])
```

```@example ensemble
plot(sol; idxs = R)
scatter!(t_train, data_train[3][2][2])
```

## Training the Ensemble Model

Now let's train the ensemble model. We will do that by solving a bit further than the
calibration step. Let's build that solution data:

```@example ensemble
plot(sol;idxs = S)
scatter!(t_ensem,data_ensem[1][2][2])
```

We can obtain the optimal weights for ensembling by solving a linear regression of
the solution's data against the wanted trajectory:

```@example ensemble
sol = solve(enprobs; saveat = t_ensem);
ensem_weights = ensemble_weights(sol, data_ensem)
```

Now we can extrapolate forward with these ensemble weights as follows:

```@example ensemble
sol = solve(enprobs; saveat = t_ensem);
ensem_prediction = sum(stack(ensem_weights .* sol[:,S]), dims = 2)
plot(sol; idxs = S, color = :blue)
plot!(t_ensem, ensem_prediction, lw = 5, color = :red)
scatter!(t_ensem, data_ensem[1][2][2])
```

```@example ensemble
ensem_prediction = sum(stack(ensem_weights .* sol[:,I]), dims = 2)
plot(sol; idxs = I, color = :blue)
plot!(t_ensem, ensem_prediction, lw = 3, color = :red)
scatter!(t_ensem, data_ensem[2][2][2])
```

## Forecasting the Trained Ensemble

Once we have obtained the ensemble model, we can forecast ahead with it:

```@example ensemble
forecast_probs = [remake(enprobs.prob[i]; tspan = (t_train[1],t_forecast[end])) for i in 1:length(enprobs.prob)]
fit_enprob = EnsembleProblem(forecast_probs)

sol = solve(fit_enprob; saveat = t_forecast);
ensem_prediction = sum(stack(ensem_weights .* sol[:,S]), dims = 2)
plot(sol; idxs = S, color = :blue)
plot!(t_forecast, ensem_prediction, lw = 3, color = :red)
scatter!(t_forecast, data_forecast[1][2][2])
```

```@example ensemble
ensem_prediction = sum(stack([ensem_weights[i] * sol[i][I] for i in 1:length(forecast_probs)]), dims = 2)
plot(sol; idxs = I, color = :blue)
plot!(t_forecast, ensem_prediction, lw = 3, color = :red)
scatter!(t_forecast, data_forecast[2][2][2])
```

```@example ensemble
ensem_prediction = sum(stack([ensem_weights[i] * sol[i][R] for i in 1:length(forecast_probs)]), dims = 2)
plot(sol; idxs = R, color = :blue)
plot!(t_forecast, ensem_prediction, lw = 3, color = :red)
scatter!(t_forecast, data_forecast[3][2][2])
```