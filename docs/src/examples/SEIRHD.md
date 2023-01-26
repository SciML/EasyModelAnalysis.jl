# Analysis of Interventions On The SEIRHD Epidemic Model

First, let's implement the classic SEIRHD epidemic model with ModelingToolkit:

```@examples seirhd
using EasyModelAnalysis
@variables t
Dₜ = Differential(t)
@variables S(t)=0.9 E(t)=0.05 I(t)=0.01 R(t)=0.2 H(t)=0.1 D(t)=0.01 T(t)=0.0 η(t)=0.0
@parameters β₁=0.6 β₂=0.143 β₃=0.055 α=0.003 γ₁=0.007 γ₂=0.011 δ=0.1 μ=0.14
eqs = [T ~ S + E + I + R + H + D
       η ~ (β₁ * E + β₂ * I + β₃ * H) / T
       Dₜ(S) ~ -η * S
       Dₜ(E) ~ η * S - α * E
       Dₜ(I) ~ α * E - (γ₁ + δ) * I
       Dₜ(R) ~ γ₁ * I + γ₂ * H
       Dₜ(H) ~ δ * I - (μ + γ₂) * H
       Dₜ(D) ~ μ * H];
@named seirhd = ODESystem(eqs)
seirhd = structural_simplify(seirhd)
prob = ODEProblem(seirhd, [], (0, 110.0))
```