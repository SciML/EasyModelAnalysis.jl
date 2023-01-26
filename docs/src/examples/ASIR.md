# Analysis of The Asymptomatic SIR Model

First, we implement the asymptomatic SIR model from BioMedInformatics 2022, 2(3),
398-404; https://doi.org/10.3390/biomedinformatics2030025.

```@example asir
using EasyModelAnalysis
@variables t
Dₜ = Differential(t)
@variables S(t)=0.9 Iₐ(t)=0.05 Iₛ(t)=0.01 Rₐ(t)=0.2 Rₛ(t)=0.1 D(t)=0.01
@parameters α=0.6 βₐ=0.143 βₛ=0.055 ρ=0.003 μₙ=0.007 μₘ=0.011 θ=0.1 ωₛ=0.14
eqs = [
       Dₜ(S) ~ μₙ*S - μₘ*S - θ*α*S*Iₛ - (1-θ)*α*S*Iₐ + ρ*(Rₐ + Rₛ)
       Dₜ(Iₐ) ~ (1 - θ)*α*S*Iₐ - βₐ*Iₐ
       Dₜ(Iₛ) ~ θ*α*S*Iₛ - βₛ*Iₛ
       Dₜ(Rₐ) ~ βₐ*Iₐ - ρ*Rₐ
       Dₜ(Rₛ) ~ (1 - ωₛ)*βₛ*Iₛ - ρ*Rₛ
       Dₜ(D) ~ ωₛ*βₛ*Iₛ
      ]
@named asir = ODESystem(eqs)
prob = ODEProblem(asir, [], (0, 110.0))
sol = solve(prob)
plot(sol)
```
