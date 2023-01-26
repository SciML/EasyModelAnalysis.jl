"""
    stop_at_threshold(prob, obs, threshold)

Simulates `prob` until `obs == threshold`.
"""
function stop_at_threshold(prob, obs, threshold; alg = nothing, kw...)
    sys = prob.f.sys
    sys isa ModelingToolkit.AbstractSystem ||
        error("The problem must be a ModelingToolkit model.")
    obsfun = prob.f.observed(obs)
    condition = let obsfun = obsfun, threshold = threshold
        (u, t, integrator) -> obsfun(u, integrator.p, t) - threshold
    end
    affect! = integrator -> terminate!(integrator)
    cb = ContinuousCallback(condition, affect!)
    if alg === nothing
        sol = solve(prob; callback = cb, kw...)
    else
        sol = solve(prob, alg; callback = cb, kw...)
    end
    sol
end

"""
    get_threshold(prob, obs, threshold)

Returns the value `t` for the time point where the solution of the model `prob` has the observation
`obs` hit the `threshold` value.
"""
function get_threshold(prob, obs, threshold; alg = nothing, kw...)
    sol = stop_at_threshold(prob, obs, threshold; alg = nothing, kw...)
    sol.t[end]
end

"""
    prob_violating_treshold(prob, p, tresholds)

Returns the probability of violating `tresholds` given distributions of parameters `p`.
"""
function prob_violating_treshold(prob, p, tresholds)
    pkeys = getfield.(p, :first)
    p_dist = getfield.(p, :second)
    gd = GenericDistribution(p_dist...)
    sol = solve(prob)
    sm = SystemMap(prob, sol.alg)
    h(x, u, p) = u, remake(prob, p = Pair.(pkeys, [x...])).p # remake does not work well with static arrays
    function g(sol, p)
        for treshold in tresholds
            if (treshold.val.f == >) || (treshold.val.f == >=)
                if maximum(sol[treshold.val.arguments[1]]) > treshold.val.arguments[2]
                    return 1.0
                end
            elseif (treshold.val.f == <) || (treshold.val.f == <=)
                if minimum(sol[treshold.val.arguments[1]]) < treshold.val.arguments[2]
                    return 1.0
                end
            else
                error()
            end
        end
        return 0.0
    end
    exprob = ExpectationProblem(sm, g, h, gd; nout = 1)
    exsol = solve(exprob, Koopman(), batch = 0, quadalg = HCubatureJL())
    exsol.u
end
