"""
    optimal_threshold_intervention(prob, [p1 = prob.p], p2, obs, threshold, duration; maxtime)

## Arguments

  - `p1`: parameters for the pre-intervention scenario. Defaults to `prob.p`.
  - `p2`: parameters for the pose-intervention scenario.
  - `obs`: The observation symbolic expression.
  - `threshold`: The threshold for the observation.
  - `duration`: Duration for the evaluation of intervention.

## Keyword Arguments

  - `maxtime`: Maximum optimzation time. Defaults to `60`.

# Returns

  - `opt_tspan`: Optimal intervention time span.
  - `(s1, s2, s3)`: Pre-intervention, intervention, post-intervention solutions.
  - `ret`: Return code from the optimization.
"""
function optimal_threshold_intervention(prob, p2, obs, threshold, duration; kw...)
    p1 = prob.p
    optimal_threshold_intervention(prob, p1, p2, obs, threshold, duration; kw...)
end
function optimal_threshold_intervention(prob, p1, p2, obs, threshold, duration;
                                        maxtime = 60, kw...)
    t0 = prob.tspan[1]
    prob1 = remake(prob, p = p1)
    prob2 = remake(prob, p = p2)

    function cost(x::Vector, grad::Vector)
        return x[2] - x[1]
    end

    function duration_constraint(x::Vector, grad::Vector, ::Val{p} = Val(false)) where {p}
        prob_preintervention = remake(prob1, tspan = (t0, x[1]))
        sol_preintervention = stop_at_threshold(prob_preintervention, obs, threshold; kw...)
        violation = x[1] - sol_preintervention.t[end]
        violation > 0 && return violation + (duration - (x[1] - t0))

        prob_intervention = remake(prob2, u0 = sol_preintervention.u[end],
                                   tspan = (x[1], x[2]))
        sol_intervention = stop_at_threshold(prob_intervention, obs, threshold; kw...)
        violation = x[2] - sol_intervention.t[end]
        violation > 0 && return violation + (duration - (x[2] - t0))

        prob_postintervention = remake(prob1, u0 = sol_intervention.u[end],
                                       tspan = (x[2], t0 + duration))
        sol_postintervention = stop_at_threshold(prob_postintervention, obs, threshold;
                                                 kw...)
        violation = t0 + duration - sol_postintervention.t[end]
        return p ?
               (violation, (sol_preintervention, sol_intervention, sol_postintervention)) :
               violation
    end
    function start_end_constraint(x::Vector, grad::Vector)
        x[1] - x[2]
    end

    opt = Opt(:GN_ISRES, 2)
    opt.lower_bounds = [t0, t0]
    opt.upper_bounds = [t0 + duration, t0 + duration]
    opt.xtol_rel = 1e-4

    opt.min_objective = cost
    inequality_constraint!(opt, duration_constraint, 1e-16)
    inequality_constraint!(opt, start_end_constraint, 1e-8)
    opt.maxtime = maxtime
    init_x = [t0, t0 + duration]
    (optf, optx, ret) = NLopt.optimize(opt, init_x)
    _, (s1, s2, s3) = duration_constraint(optx, [], Val(true))
    optx, (s1, s2, s3), ret
end

"""
    optimal_parameter_intervention_for_threshold(prob, obs, threshold, cost, ps,
        lb, ub, intervention_tspan, duration; ineq_cons = nothing, maxtime=60)

## Arguments

  - `prob`: An ODEProblem.
  - `obs`: The observation symbolic expression.
  - `threshold`: The threshold for the observation.
  - `cost`: the cost function for minimization, e.g. `α + 20 * β`.
  - `ps`: the parameters that appear in the cost, e.g. `[α, β]`.
  - `lb`: the lower bounds of the parameters e.g. `[-10, -5]`.
  - `ub`: the uppwer bounds of the parameters e.g. `[5, 10]`.
  - `intervention_tspan`: intervention time span, e.g. `(20.0, 30.0)`. Defaults to `prob.tspan`.
  - `duration`: Duration for the evaluation of intervention. Defaults to `prob.tspan[2] - prob.tspan[1]`.


## Keyword Arguments

  - `maxtime`: Maximum optimzation time. Defaults to `60`.
  - `ineq_cons`: a vector of symbolic expressions in terms of symbolic
      parameters. The optimizer will enforce `ineq_cons .< 0`.

# Returns

  - `opt_p`: Optimal intervention parameters.
  - `(s1, s2, s3)`: Pre-intervention, intervention, post-intervention solutions.
  - `ret`: Return code from the optimization.
"""
function optimal_parameter_intervention_for_threshold(prob, obs, threshold,
                                                      symbolic_cost, ps, lb, ub,
                                                      intervention_tspan = prob.tspan,
                                                      duration = abs(-(prob.tspan...));
                                                      maxtime = 60, ineq_cons = nothing,
                                                      kw...)
    t0 = prob.tspan[1]
    ti_start, ti_end = intervention_tspan
    symbolic_cost = Symbolics.unwrap(symbolic_cost)
    #ps = collect(ModelingToolkit.vars(symbolic_cost))
    _cost = Symbolics.build_function(symbolic_cost, ps, expression = Val{false})
    _cost(prob.p) # just throw when something is wrong during the setup.

    cost = let _cost = _cost
        (x, grad) -> _cost(x)
    end

    function duration_constraint(x::Vector, grad::Vector, ::Val{p} = Val(false)) where {p}
        prob_preintervention = remake(prob, tspan = (t0, ti_start))
        sol_preintervention = stop_at_threshold(prob_preintervention, obs, threshold; kw...)
        violation = ti_start - sol_preintervention.t[end]
        violation > 0 && return p ? (sol_preintervention, nothing, nothing) :
               (violation + (duration - (ti_start - t0)))

        prob_intervention = remake(prob, u0 = sol_preintervention.u[end], p = ps .=> x,
                                   tspan = (ti_start, ti_end))
        sol_intervention = stop_at_threshold(prob_intervention, obs, threshold; kw...)
        violation = ti_end - sol_intervention.t[end]
        violation > 0 && return p ? (sol_preintervention, sol_intervention, nothing) :
               (violation + (duration - (ti_end - ti_start)))

        prob_postintervention = remake(prob, u0 = sol_intervention.u[end],
                                       tspan = (ti_end, t0 + duration))
        sol_postintervention = stop_at_threshold(prob_postintervention, obs, threshold;
                                                 kw...)
        violation = t0 + duration - sol_postintervention.t[end]
        return p ?
               (sol_preintervention, sol_intervention, sol_postintervention) :
               violation
    end

    opt = Opt(:GN_ISRES, length(ps))
    opt.lower_bounds = lb
    opt.upper_bounds = ub
    opt.xtol_rel = 1e-4

    opt.min_objective = cost
    inequality_constraint!(opt, duration_constraint, 1e-16)
    init_x = @. lb + ub / 2
    if ineq_cons !== nothing
        for con in ineq_cons
            _con = Symbolics.build_function(Symbolics.unwrap(con), ps,
                                            expression = Val{false})
            _con(init_x)
            ineq_con = (x, _) -> _con(x)
            inequality_constraint!(opt, ineq_con, 1e-16)
        end
    end
    opt.maxtime = maxtime
    (optf, optx, ret) = NLopt.optimize(opt, init_x)
    ss = duration_constraint(optx, [], Val(true))
    Dict(ps .=> optx), ss, ret
end

"""
    optimal_parameter_intervention_for_reach(prob, obs, reach, cost, ps,
        lb, ub, intervention_tspan, duration; ineq_cons = nothing, maxtime=60)

## Arguments

  - `prob`: An ODEProblem.
  - `obs`: The observation symbolic expression.
  - `reach`: The reach for the observation, i.e., the constraint enforces that `obs` reaches `reach`.
  - `cost`: the cost function for minimization, e.g. `α + 20 * β`. It could be a
  tuple where the first argument is a symbol object in terms of parameters, and
  the second entry of the tuple could be an arbitrary function that takes a
  solution object and returns a real scalar.
  - `ps`: the parameters that appear in the cost, e.g. `[α, β]`.
  - `lb`: the lower bounds of the parameters e.g. `[-10, -5]`.
  - `ub`: the uppwer bounds of the parameters e.g. `[5, 10]`.
  - `intervention_tspan`: intervention time span, e.g. `(20.0, 30.0)`. Defaults to `prob.tspan`.
  - `duration`: Duration for the evaluation of intervention. Defaults to `prob.tspan[2] - prob.tspan[1]`.


## Keyword Arguments

  - `maxtime`: Maximum optimzation time. Defaults to `60`.
  - `ineq_cons`: a vector of symbolic expressions in terms of symbolic
      parameters. The optimizer will enforce `ineq_cons .< 0`.

# Returns

  - `opt_p`: Optimal intervention parameters.
  - `(s1, s2, s3)`: Pre-intervention, intervention, post-intervention solutions.
  - `ret`: Return code from the optimization.
"""
function optimal_parameter_intervention_for_reach(prob, obs, reach,
                                                  symbolic_cost, ps, lb, ub,
                                                  intervention_tspan = prob.tspan,
                                                  duration = abs(-(prob.tspan...));
                                                  maxtime = 60, ineq_cons = nothing,
                                                  kw...)
    t0 = prob.tspan[1]
    ti_start, ti_end = intervention_tspan
    if symbolic_cost isa Tuple
        symbolic_cost, cost_sol = symbolic_cost
    else
        cost_sol = nothing
    end
    symbolic_cost = Symbolics.unwrap(symbolic_cost)
    #ps = collect(ModelingToolkit.vars(symbolic_cost))
    _cost = Symbolics.build_function(symbolic_cost, ps, expression = Val{false})
    _cost(prob.p) # just throw when something is wrong during the setup.

    cost = let _cost = _cost, cost_sol = cost_sol
        (x, grad) -> begin
            p_cost = _cost(x)
            if cost_sol === nothing
                sol_cost = 0.0
            else
                tf = t0 + duration
                prob_preintervention = remake(prob, tspan = (t0, ti_start))
                sol_preintervention = solve(prob_preintervention; kw...)
                prob_intervention = remake(prob, u0 = sol_preintervention.u[end],
                                           p = ps .=> x,
                                           tspan = (ti_start, ti_end))
                sol_intervention = solve(prob_intervention; kw...)
                sol_cost = cost_sol(sol_intervention)
            end
            p_cost + sol_cost
        end
    end

    function duration_constraint(x::Vector, grad::Vector, ::Val{p} = Val(false)) where {p}
        tf = t0 + duration
        prob_preintervention = remake(prob, tspan = (t0, ti_start))
        if p
            sol_preintervention = solve(prob_preintervention; kw...)
        else
            sol_preintervention = stop_at_threshold(prob_preintervention, obs, reach; kw...)
            reach_time = ti_start - sol_preintervention.t[end]
            reach_time > 0 && return sol_preintervention.t[end] - tf
        end

        prob_intervention = remake(prob, u0 = sol_preintervention.u[end], p = ps .=> x,
                                   tspan = (ti_start, ti_end))
        if p
            sol_intervention = solve(prob_intervention; kw...)
        else
            sol_intervention = stop_at_threshold(prob_intervention, obs, reach; kw...)
            reach_time = ti_end - sol_intervention.t[end]
            reach_time > 0 && return sol_intervention.t[end] - tf
        end

        prob_postintervention = remake(prob, u0 = sol_intervention.u[end],
                                       tspan = (ti_end, t0 + duration))
        if p
            sol_postintervention = solve(prob_postintervention; kw...)
            sol_preintervention, sol_intervention, sol_postintervention
        else
            sol_postintervention = stop_at_threshold(prob_postintervention, obs, reach;
                                                     kw...)
            reach_time = tf - sol_postintervention.t[end]
            10.0
        end
    end

    opt = Opt(:GN_ISRES, length(ps))
    opt.lower_bounds = lb
    opt.upper_bounds = ub
    opt.xtol_rel = 1e-4

    opt.min_objective = cost
    init_x = @. (lb + ub) / 2
    duration_constraint(init_x, [])
    inequality_constraint!(opt, duration_constraint, 1e-16)
    if ineq_cons !== nothing
        for con in ineq_cons
            _con = Symbolics.build_function(Symbolics.unwrap(con), ps,
                                            expression = Val{false})
            _con(init_x)
            ineq_con = (x, _) -> _con(x)
            inequality_constraint!(opt, ineq_con, 1e-16)
        end
    end
    opt.maxtime = maxtime
    (optf, optx, ret) = NLopt.optimize(opt, init_x)
    ss = duration_constraint(optx, [], Val(true))
    Dict(ps .=> optx), ss, ret
end
