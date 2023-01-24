using EasyModelAnalysis, Test

using DifferentialEquations, ModelingToolkit, Optimization, OptimizationBBO

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

function get_timeseries(prob, sym, t)
    remake(prob, tspan = (min(prob.tspan[1], t[1]), max(prob.tspan[2], t[end]))) 
    sol = solve(prob, saveat = t)
    sol[sym]    
end

generate_timeseries(prob, x, [0.0, 1.0, 2.0])

function get_min_t(prob, sym)
    sol = solve(prob)
    f(t,_) = sol(t[1];idxs = sym)
    oprob = OptimizationProblem(f, [(prob.tspan[2] - prob.tspan[1]) / 2],
                                lb = [prob.tspan[1]],
                                ub = [prob.tspan[end]])
    res = solve(oprob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters=10000)
    res.u[1]
end

function get_max_t(prob, sym)
    sol = solve(prob)
    f(t,_) = -sol(t[1];idxs = sym)
    oprob = OptimizationProblem(f, [(prob.tspan[2] - prob.tspan[1]) / 2],
                                lb = [prob.tspan[1]],
                                ub = [prob.tspan[end]])
    res = solve(oprob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters=10000)
    res.u[1]
end

xmin = get_min_t(prob, x)
xmax = get_max_t(prob, x)

using Plots
plot(sol, idxs = x)
scatter!([xmin sol(xmin;idxs=x)])
scatter!([xmax,sol(xmax;idxs=x)])