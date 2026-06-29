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
    return sol
end

"""
    get_threshold(prob, obs, threshold)

Returns the value `t` for the time point where the solution of the model `prob` has the observation
`obs` hit the `threshold` value.
"""
function get_threshold(prob, obs, threshold; alg = nothing, kw...)
    sol = stop_at_threshold(prob, obs, threshold; alg = nothing, kw...)
    return sol.t[end]
end

# Decompose a symbolic threshold inequality (e.g. `x > 10.0`) into the state it
# constrains, the numeric bound, and whether the violating side is the upper one
# (`maximum(state) > bound`) or the lower one (`minimum(state) < bound`).
# Symbolics canonicalizes comparisons, so `x > 10.0` is stored as `<(10.0, x)`:
# the constant can land on either side of the operator, which this normalizes.
function _threshold_violation(threshold)
    v = ModelingToolkit.value(threshold)
    op = operation(v)
    args = arguments(v)
    isconst(z) = ModelingToolkit.value(z) isa Number
    if isconst(args[1])
        bound = ModelingToolkit.value(args[1])
        state = args[2]
        # `bound op state`: `bound < state` ⟺ `state > bound` (upper violation).
        upper = (op === <) || (op === <=)
    else
        bound = ModelingToolkit.value(args[2])
        state = args[1]
        # `state op bound`: `state > bound`/`state >= bound` is the upper violation.
        upper = (op === >) || (op === >=)
    end
    return state, bound, upper
end

"""
    prob_violating_thresholdd(prob, p, thresholds)

Returns the probability of violating `thresholds` given distributions of parameters `p`.
"""
function prob_violating_threshold(prob, p, thresholds)
    pkeys = getfield.(p, :first)
    p_dist = getfield.(p, :second)
    gd = GenericDistribution(p_dist...)
    sol = solve(prob)
    sm = SystemMap(prob, sol.alg)
    h(x, u, p) = u, remake(prob, p = Pair.(pkeys, [x...])).p # remake does not work well with static arrays
    function g(sol, p)
        for threshold in thresholds
            state, bound, upper = _threshold_violation(threshold)
            if upper
                if maximum(sol[state]) > bound
                    return 1.0
                end
            else
                if minimum(sol[state]) < bound
                    return 1.0
                end
            end
        end
        return 0.0
    end
    exprob = ExpectationProblem(sm, g, h, gd)
    exsol = solve(exprob, Koopman(), quadalg = HCubatureJL())
    return exsol.u
end

"""
    optimal_parameter_threshold(prob, obs, threshold, cost, ps, lb, ub; ineq_cons = nothing, maxtime = 60, kw...)

## Arguments

  - `prob`: An ODEProblem.
  - `obs`: The observation symbolic expression.
  - `threshold`: The threshold for the observation.
  - `cost`: the cost function for minimization, e.g. `α + 20 * β`.
  - `ps`: the parameters that appear in the cost, e.g. `[α, β]`.
  - `lb`: the lower bounds of the parameters e.g. `[-10, -5]`.
  - `ub`: the upper bounds of the parameters e.g. `[5, 10]`.

## Keyword Arguments

  - `maxtime`: Maximum optimization time. Defaults to `60`.
  - `ineq_cons`: a vector of symbolic expressions in terms of symbolic
    parameters. The optimizer will enforce `ineq_cons .< 0`.

# Returns

  - `opt_p`: Optimal intervention parameters.
  - `sol`: Solution with the optimal intervention parameters.
  - `ret`: Return code from the optimization.
"""
function optimal_parameter_threshold(
        prob, obs, threshold, cost, ps, lb, ub;
        ineq_cons = nothing, maxtime = 60,
        kw...
    )
    opt, (s1, s2, s3),
        ret = optimal_parameter_intervention_for_threshold(
        prob, obs,
        threshold, cost,
        ps, lb, ub;
        ineq_cons,
        maxtime, kw...
    )
    return opt, s2, ret
end
