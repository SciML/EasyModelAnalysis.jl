# A SIR
using EasyModelAnalysis
@variables t
Dₜ = Differential(t)
@variables S(t)=0.9 Iₐ(t)=0.05 Iₛ(t)=0.01 Rₐ(t)=0.2 Rₛ(t)=0.1 D(t)=0.01
@parameters α=0.6 βₐ=0.143 βₛ=0.055 ρ=0.003 μₙ=0.007 μₘ=0.011 θ=0.1 ωₛ=0.14
eqs = [Dₜ(S) ~ μₙ * S - μₘ * S - θ * α * S * Iₛ - (1 - θ) * α * S * Iₐ + ρ * (Rₐ + Rₛ)
    Dₜ(Iₐ) ~ (1 - θ) * α * S * Iₐ - βₐ * Iₐ
    Dₜ(Iₛ) ~ θ * α * S * Iₛ - βₛ * Iₛ
    Dₜ(Rₐ) ~ βₐ * Iₐ - ρ * Rₐ
    Dₜ(Rₛ) ~ (1 - ωₛ) * βₛ * Iₛ - ρ * Rₛ
    Dₜ(D) ~ ωₛ * βₛ * Iₛ]
@named asir = ODESystem(eqs)
prob = ODEProblem(asir, [], (0, 110.0))
sol = solve(prob)
# plot(sol)
tmax, imax = get_max_t(prob, Iₐ)
using Test
@test tmax > 10
@test imax > 0.3

# SEIRHD
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
sol = solve(prob)
# plot(sol)
tmax, imax = get_max_t(prob, I)
@test 39 <= tmax <= 40
@test imax > 0.02
