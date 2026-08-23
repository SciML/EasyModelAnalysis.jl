module EasyModelAnalysis

# EasyModelAnalysis is a facade over the SciML modeling stack: `using EasyModelAnalysis`
# is meant to be enough on its own to build a model, solve it, sample a parameter
# distribution and plot the result. These four packages are brought into scope for that
# reexport surface, which is pinned by the explicit `export` blocks at the bottom of this
# file -- the documented modeling, solving, distribution and plotting API rather than a
# blanket `@reexport` of everything the four packages happen to export. Every reexported
# name stays owned and documented by its upstream package.
using DifferentialEquations
using ModelingToolkit
using Distributions
using Plots

using LinearAlgebra: LinearAlgebra, I, norm
using DifferentialEquations: DifferentialEquations, remake, solve
using ModelingToolkit: ModelingToolkit, Num, Symbolics
using Distributions: Distributions, InverseGamma, MvNormal, product_distribution
using Plots: Plots, @layout, bar, plot, plot!, scatter!
using PrecompileTools: @compile_workload, @setup_workload
using Optimization: Optimization, OptimizationProblem
using OptimizationBBO: OptimizationBBO, BBO_adaptive_de_rand_1_bin_radiuslimited
using OptimizationNLopt: OptimizationNLopt
using GlobalSensitivity: GlobalSensitivity, Sobol
using NLopt: NLopt, Opt, inequality_constraint!
using Turing: Turing, @varname
using Integrals: Integrals, HCubatureJL
using SciMLExpectations: SciMLExpectations, ExpectationProblem, GenericDistribution,
    Koopman, SystemMap
using SciMLBase: SciMLBase, ContinuousCallback, EnsembleProblem, EnsembleSerial,
    EnsembleSolution, EnsembleThreads, ODESolution, terminate!
using SymbolicUtils: SymbolicUtils

include("basics.jl")
include("datafit.jl")
include("sensitivity.jl")
include("threshold.jl")
include("intervention.jl")
include("ensemble.jl")

export get_timeseries, get_min_t, get_max_t, plot_extrema, phaseplot_extrema
export get_uncertainty_forecast, get_uncertainty_forecast_quantiles
export plot_uncertainty_forecast, plot_uncertainty_forecast_quantiles
export datafit, global_datafit, bayesian_datafit
export get_sensitivity, create_sensitivity_plot, get_sensitivity_of_maximum
export stop_at_threshold, get_threshold
export model_forecast_score
export optimal_threshold_intervention, prob_violating_threshold,
    optimal_parameter_intervention_for_threshold, optimal_parameter_threshold,
    optimal_parameter_intervention_for_reach
export bayesian_ensemble, ensemble_weights

# Reexported DifferentialEquations / SciML common interface: the problem, solution and
# ensemble types an EasyModelAnalysis query is built on, the `solve`/`init`/`remake`
# interface, the callbacks, and the default solver set. Approved via `reexports_allow`
# in test/qa/qa.jl; every name stays owned and documented upstream.
export add_saveat!, add_tstop!, AutoFiniteDiff, AutoForwardDiff, AutoTsit5, AutoVern6,
    AutoVern7, AutoVern8, AutoVern9, CallbackSet, ContinuousCallback, DAEFunction,
    DAEProblem, DDEFunction, DDEProblem, DefaultODEAlgorithm, DifferentialEquations,
    DiscreteCallback, DiscreteFunction, DiscreteProblem, DynamicalODEFunction,
    DynamicalODEProblem, EnsembleAnalysis, EnsembleDistributed, EnsembleProblem,
    EnsembleSerial, EnsembleSolution, EnsembleSplitThreads, EnsembleSummary,
    EnsembleThreads, FBDF, get_du, get_du!, get_proposed_dt, init, NonlinearFunction,
    NonlinearProblem, NonlinearSolution, ODEFunction, ODEProblem, ODESolution,
    OptimizationFunction, OptimizationProblem, OptimizationSolution, OrdinaryDiffEq,
    reinit!, remake, ReturnCode, Rodas5P, RODEProblem, RODESolution, Rosenbrock23,
    savevalues!, SciMLBase, SDEFunction, SDEProblem, SecondOrderODEProblem,
    set_proposed_dt!, solve, solve!, SplitFunction, SplitODEProblem, SteadyStateProblem,
    SteadyStateSolution, step!, successful_retcode, terminate!, Tsit5, u_modified!,
    VectorContinuousCallback, Vern6, Vern7, Vern8, Vern9

# Reexported ModelingToolkit modeling API: the symbolic DSL a model is written in and the
# accessors used to inspect and transform the resulting system.
export add_accumulations, alg_equations, arguments, @brownian, @brownians, brownians,
    build_function, Clock, complete, @component, compose, connect, Connection, @connector,
    @constants, constraints, continuous_events, diff_equations, Differential,
    discrete_events, @discretes, DiscreteSystem, Equation, equations, expand,
    expand_derivatives, extend, flatten, Flow, full_equations, get_variables, getbounds,
    getdescription, getguess, GlobalScope, guesses, hasbounds, hasdescription, hasguess,
    ImplicitDiscreteSystem, independent_variable, @independent_variables,
    independent_variables, Initial, initialization_equations, Integral, iscall, isinput,
    isoutput, jumps, JumpSystem, linearization_function, linearize, LocalScope,
    ModelingToolkit, modelingtoolkitize, @mtkbuild, @mtkcompile, mtkcompile, @mtkcomplete,
    @named, @nonamespace, NonlinearSystem, Num, observables, observed, ODESystem, operation,
    OptimizationSystem, @parameters, parameters, ParentScope, PDESystem, Pre,
    @register_symbolic, SampleTime, SDESystem, Shift, ShiftIndex, simplify, Stream,
    structural_simplify, substitute, Symbolics, System, Term, terms, toexpr, tosymbol,
    tunable_parameters, unknowns, @unpack, @variables

# Reexported Distributions API: the distribution types passed to the uncertainty,
# sensitivity and Bayesian-datafit queries, and the accessors used on their results.
export Arcsine, Bernoulli, Beta, Binomial, Categorical, Cauchy, ccdf, cdf, censored, Chisq,
    Continuous, ContinuousMultivariateDistribution, ContinuousUnivariateDistribution, cor,
    cov, cquantile, Dirac, Dirichlet, Discrete, DiscreteMultivariateDistribution,
    DiscreteNonParametric, DiscreteUniform, DiscreteUnivariateDistribution, Distribution,
    Distributions, entropy, Exponential, failprob, FDist, fit, fit_mle, Frechet, Gamma,
    Geometric, Gumbel, Hypergeometric, insupport, InverseGamma, InverseWishart,
    kldivergence, kurtosis, Laplace, LKJ, location, logccdf, logcdf, Logistic,
    loglikelihood, LogNormal, logpdf, LogUniform, MatrixDistribution, mean, median,
    MixtureModel, mode, modes, Multinomial, Multivariate, MultivariateDistribution,
    MvLogNormal, MvNormal, ncategories, NegativeBinomial, Normal, ntrials, params, Pareto,
    partype, pdf, Poisson, probs, Product, product_distribution, quantile, rate, Rayleigh,
    Sampleable, sampler, scale, shape, skewness, SkewNormal, std, succprob, support, TDist,
    Truncated, truncated, Uniform, Univariate, UnivariateDistribution, var, Weibull, Wishart

# Reexported Plots API: the plotting verbs and attribute helpers the analysis plots and
# the documented examples are built from.
export @animate, animate, Animation, annotate!, areaplot, areaplot!, arrow, backend,
    backends, bar, bar!, bbox, boxplot, boxplot!, brush, cgrad, closeall, Colorant,
    @colorant_str, colormap, contour, contour!, contourf, contourf!, current, default,
    density, density!, distinguishable_colors, font, frame, @gif, gif, Gray, grid, gui,
    heatmap, heatmap!, histogram, histogram!, histogram2d, histogram2d!, hline, hline!,
    hspan, hspan!, @layout, lens!, mesh3d, mesh3d!, mov, mp4, palette, path3d, path3d!, pie,
    pie!, plot, plot!, plot3d, plot3d!, plot_color, plotattr, Plots, png, quiver, quiver!,
    @recipe, RGB, RGBA, savefig, scatter, scatter!, scatter3d, scatter3d!, @series, Shape,
    showtheme, spy, spy!, stephist, stephist!, sticks, sticks!, stroke, Surface, surface,
    surface!, text, theme, theme_palette, title!, twinx, twiny, @userplot, violin, violin!,
    vline, vline!, vspan, vspan!, webm, wireframe, wireframe!, xaxis!, xerror, xerror!,
    xflip!, xgrid!, xlabel!, xlims, xlims!, xticks, xticks!, yaxis!, yerror, yerror!,
    yflip!, ygrid!, ylabel!, ylims, ylims!, yticks, yticks!, zaxis!, zflip!, zgrid!,
    zlabel!, zlims!, zticks, zticks!

@setup_workload begin
    @compile_workload begin
        ModelingToolkit.@independent_variables t
        ModelingToolkit.@variables x(t)
        D = ModelingToolkit.Differential(t)
        ModelingToolkit.@mtkcompile sys = ModelingToolkit.System([D(x) ~ -x], t)
        prob = DifferentialEquations.ODEProblem(sys, Dict(x => 1.0), (0.0, 1.0))
        get_timeseries(prob, x, [0.0, 0.5, 1.0])
    end
end

end
