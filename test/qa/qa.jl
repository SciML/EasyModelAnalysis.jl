using EasyModelAnalysis, SciMLTesting, Test

# The DifferentialEquations / ModelingToolkit / Distributions / Plots API that
# EasyModelAnalysis deliberately reexports, so that `using EasyModelAnalysis` on its own is
# enough to build a model, solve it, sample a parameter distribution and plot the result.
# Owned and documented upstream; kept in sync with the reexport `export` blocks in
# src/EasyModelAnalysis.jl.
const REEXPORTS = (
    :add_accumulations, :add_saveat!, :add_tstop!, :alg_equations, Symbol("@animate"),
    :animate, :Animation, :annotate!, :Arcsine, :areaplot, :areaplot!, :arguments, :arrow,
    :AutoFiniteDiff, :AutoForwardDiff, :AutoTsit5, :AutoVern6, :AutoVern7, :AutoVern8,
    :AutoVern9, :backend, :backends, :bar, :bar!, :bbox, :Bernoulli, :Beta, :Binomial,
    :boxplot, :boxplot!, Symbol("@brownian"), Symbol("@brownians"), :brownians, :brush,
    :build_function, :CallbackSet, :Categorical, :Cauchy, :ccdf, :cdf, :censored, :cgrad,
    :Chisq, :Clock, :closeall, :Colorant, Symbol("@colorant_str"), :colormap, :complete,
    Symbol("@component"), :compose, :connect, :Connection, Symbol("@connector"),
    Symbol("@constants"), :constraints, :Continuous, :continuous_events,
    :ContinuousCallback, :ContinuousMultivariateDistribution,
    :ContinuousUnivariateDistribution, :contour, :contour!, :contourf, :contourf!, :cor,
    :cov, :cquantile, :current, :DAEFunction, :DAEProblem, :DDEFunction, :DDEProblem,
    :default, :DefaultODEAlgorithm, :density, :density!, :diff_equations, :Differential,
    :DifferentialEquations, :Dirac, :Dirichlet, :Discrete, :discrete_events,
    :DiscreteCallback, :DiscreteFunction, :DiscreteMultivariateDistribution,
    :DiscreteNonParametric, :DiscreteProblem, Symbol("@discretes"), :DiscreteSystem,
    :DiscreteUniform, :DiscreteUnivariateDistribution, :distinguishable_colors,
    :Distribution, :Distributions, :DynamicalODEFunction, :DynamicalODEProblem,
    :EnsembleAnalysis, :EnsembleDistributed, :EnsembleProblem, :EnsembleSerial,
    :EnsembleSolution, :EnsembleSplitThreads, :EnsembleSummary, :EnsembleThreads, :entropy,
    :Equation, :equations, :expand, :expand_derivatives, :Exponential, :extend, :failprob,
    :FBDF, :FDist, :fit, :fit_mle, :flatten, :Flow, :font, :frame, :Frechet,
    :full_equations, :Gamma, :Geometric, :get_du, :get_du!, :get_proposed_dt,
    :get_variables, :getbounds, :getdescription, :getguess, Symbol("@gif"), :gif,
    :GlobalScope, :Gray, :grid, :guesses, :gui, :Gumbel, :hasbounds, :hasdescription,
    :hasguess, :heatmap, :heatmap!, :histogram, :histogram!, :histogram2d, :histogram2d!,
    :hline, :hline!, :hspan, :hspan!, :Hypergeometric, :ImplicitDiscreteSystem,
    :independent_variable, Symbol("@independent_variables"), :independent_variables, :init,
    :Initial, :initialization_equations, :insupport, :Integral, :InverseGamma,
    :InverseWishart, :iscall, :isinput, :isoutput, :jumps, :JumpSystem, :kldivergence,
    :kurtosis, :Laplace, Symbol("@layout"), :lens!, :linearization_function, :linearize,
    :LKJ, :LocalScope, :location, :logccdf, :logcdf, :Logistic, :loglikelihood, :LogNormal,
    :logpdf, :LogUniform, :MatrixDistribution, :mean, :median, :mesh3d, :mesh3d!,
    :MixtureModel, :mode, :ModelingToolkit, :modelingtoolkitize, :modes, :mov, :mp4,
    Symbol("@mtkbuild"), Symbol("@mtkcompile"), :mtkcompile, Symbol("@mtkcomplete"),
    :Multinomial, :Multivariate, :MultivariateDistribution, :MvLogNormal, :MvNormal,
    Symbol("@named"), :ncategories, :NegativeBinomial, Symbol("@nonamespace"),
    :NonlinearFunction, :NonlinearProblem, :NonlinearSolution, :NonlinearSystem, :Normal,
    :ntrials, :Num, :observables, :observed, :ODEFunction, :ODEProblem, :ODESolution,
    :ODESystem, :operation, :OptimizationFunction, :OptimizationProblem,
    :OptimizationSolution, :OptimizationSystem, :OrdinaryDiffEq, :palette,
    Symbol("@parameters"), :parameters, :params, :ParentScope, :Pareto, :partype, :path3d,
    :path3d!, :PDESystem, :pdf, :pie, :pie!, :plot, :plot!, :plot3d, :plot3d!, :plot_color,
    :plotattr, :Plots, :png, :Poisson, :Pre, :probs, :Product, :product_distribution,
    :quantile, :quiver, :quiver!, :rate, :Rayleigh, Symbol("@recipe"),
    Symbol("@register_symbolic"), :reinit!, :remake, :ReturnCode, :RGB, :RGBA, :Rodas5P,
    :RODEProblem, :RODESolution, :Rosenbrock23, :Sampleable, :sampler, :SampleTime,
    :savefig, :savevalues!, :scale, :scatter, :scatter!, :scatter3d, :scatter3d!,
    :SciMLBase, :SDEFunction, :SDEProblem, :SDESystem, :SecondOrderODEProblem,
    Symbol("@series"), :set_proposed_dt!, :Shape, :shape, :Shift, :ShiftIndex, :showtheme,
    :simplify, :skewness, :SkewNormal, :solve, :solve!, :SplitFunction, :SplitODEProblem,
    :spy, :spy!, :std, :SteadyStateProblem, :SteadyStateSolution, :step!, :stephist,
    :stephist!, :sticks, :sticks!, :Stream, :stroke, :structural_simplify, :substitute,
    :successful_retcode, :succprob, :support, :Surface, :surface, :surface!, :Symbolics,
    :System, :TDist, :Term, :terminate!, :terms, :text, :theme, :theme_palette, :title!,
    :toexpr, :tosymbol, :Truncated, :truncated, :Tsit5, :tunable_parameters, :twinx, :twiny,
    :u_modified!, :Uniform, :Univariate, :UnivariateDistribution, :unknowns,
    Symbol("@unpack"), Symbol("@userplot"), :var, Symbol("@variables"),
    :VectorContinuousCallback, :Vern6, :Vern7, :Vern8, :Vern9, :violin, :violin!, :vline,
    :vline!, :vspan, :vspan!, :webm, :Weibull, :wireframe, :wireframe!, :Wishart, :xaxis!,
    :xerror, :xerror!, :xflip!, :xgrid!, :xlabel!, :xlims, :xlims!, :xticks, :xticks!,
    :yaxis!, :yerror, :yerror!, :yflip!, :ygrid!, :ylabel!, :ylims, :ylims!, :yticks,
    :yticks!, :zaxis!, :zflip!, :zgrid!, :zlabel!, :zlims!, :zticks, :zticks!,
)

run_qa(
    EasyModelAnalysis;
    reexports_allow = REEXPORTS,
    api_docs_kwargs = (; ignore = REEXPORTS, rendered_ignore = REEXPORTS),
    # The reexported names are brought into scope by the bare `using` of each upstream
    # package in src/EasyModelAnalysis.jl; they are reexported, not used, so they are not
    # implicit imports that should be made explicit.
    ei_kwargs = (; no_implicit_imports = (; ignore = REEXPORTS)),
)

@testset "Reexport surface" begin
    # Every approved reexport must actually be reachable from `using EasyModelAnalysis`,
    # so the allow-list cannot drift into approving names the package no longer provides.
    unreachable = filter(
        name -> !(name in names(EasyModelAnalysis) && isdefined(@__MODULE__, name)),
        collect(REEXPORTS),
    )
    @test isempty(unreachable)
end
