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
