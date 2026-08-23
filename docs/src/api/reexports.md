# Reexported API

EasyModelAnalysis is a facade over the SciML modeling stack: `using EasyModelAnalysis`
is meant to be enough on its own to build a model, solve it, sample a parameter
distribution and plot the result. To make that true it reexports the documented
modeling, solving, distribution and plotting API of four upstream packages.

**EasyModelAnalysis does not own or document any of the names on this page.** It only
puts them in scope. Each group below names the package that owns them and links to the
documentation you should actually read. The EasyModelAnalysis-owned API — the queries
themselves — is on the other API pages.

The four packages are:

| Package | Owns | Documentation |
|---|---|---|
| DifferentialEquations.jl | problems, solvers, solutions, ensembles, callbacks | [DiffEqDocs](https://docs.sciml.ai/DiffEqDocs/stable/) and [SciMLBase](https://docs.sciml.ai/SciMLBase/stable/) |
| ModelingToolkit.jl | the symbolic modeling DSL and system accessors | [ModelingToolkit](https://docs.sciml.ai/ModelingToolkit/stable/) and [Symbolics](https://docs.sciml.ai/Symbolics/stable/) |
| Distributions.jl | distribution types and their statistics | [Distributions](https://juliastats.org/Distributions.jl/stable/) |
| Plots.jl | the plotting verbs and attributes | [Plots](https://docs.juliaplots.org/stable/) |

## Modeling — owned by ModelingToolkit.jl

Documented by [ModelingToolkit](https://docs.sciml.ai/ModelingToolkit/stable/); the
symbolic-expression half is owned by
[Symbolics](https://docs.sciml.ai/Symbolics/stable/) and reexported through
ModelingToolkit.

  - Declaring symbols: `@variables`, `@parameters`, `@independent_variables`,
    `@constants`, `@brownian`, `@brownians`, `@discretes`, `@named`, `@unpack`,
    `@nonamespace`, `@register_symbolic`
  - Operators and symbolic types: `Differential`, `Integral`, `Equation`, `Num`,
    `Term`, `Initial`, `Pre`, `Shift`, `ShiftIndex`, `Clock`, `SampleTime`
  - Systems: `System`, `ODESystem`, `SDESystem`, `NonlinearSystem`, `JumpSystem`,
    `OptimizationSystem`, `PDESystem`, `DiscreteSystem`, `ImplicitDiscreteSystem`
  - Building and simplifying: `mtkcompile`, `@mtkcompile`, `@mtkbuild`,
    `@mtkcomplete`, `structural_simplify`, `complete`, `compose`, `extend`,
    `flatten`, `connect`, `modelingtoolkitize`, `add_accumulations`
  - Component modeling: `@component`, `@connector`, `Connection`, `Flow`, `Stream`,
    `GlobalScope`, `LocalScope`, `ParentScope`
  - Inspecting a system: `equations`, `full_equations`, `alg_equations`,
    `diff_equations`, `unknowns`, `parameters`, `tunable_parameters`, `observed`,
    `observables`, `brownians`, `jumps`, `constraints`, `guesses`,
    `initialization_equations`, `continuous_events`, `discrete_events`,
    `independent_variable`, `independent_variables`, `get_variables`
  - Symbol metadata: `getbounds`, `hasbounds`, `getguess`, `hasguess`,
    `getdescription`, `hasdescription`, `isinput`, `isoutput`
  - Symbolic manipulation: `substitute`, `simplify`, `expand`, `expand_derivatives`,
    `build_function`, `operation`, `arguments`, `iscall`, `terms`, `toexpr`,
    `tosymbol`
  - Linearization: `linearize`, `linearization_function`
  - The `ModelingToolkit` and `Symbolics` modules themselves, for qualified access

Anything else from ModelingToolkit — the structural-transformation internals
(`TearingState`, `BipartiteGraph`, `StructuralTransformations`, `alias_elimination`,
`dae_index_lowering`, …), the rewriting machinery (`Rewriters`, `RuleSet`, `@rule`,
`@acrule`), the dynamic-optimization collocations, and the entire `sympy_*` bridge —
must be imported from ModelingToolkit directly.

## Solving — owned by DifferentialEquations.jl

Documented by [DiffEqDocs](https://docs.sciml.ai/DiffEqDocs/stable/); the common
interface types are owned by
[SciMLBase](https://docs.sciml.ai/SciMLBase/stable/) and the optimization types by
[Optimization.jl](https://docs.sciml.ai/Optimization/stable/).

  - Problems: `ODEProblem`, `SDEProblem`, `DAEProblem`, `DDEProblem`, `RODEProblem`,
    `DiscreteProblem`, `NonlinearProblem`, `SteadyStateProblem`, `SplitODEProblem`,
    `SecondOrderODEProblem`, `DynamicalODEProblem`, `OptimizationProblem`
  - Functions: `ODEFunction`, `SDEFunction`, `DAEFunction`, `DDEFunction`,
    `DiscreteFunction`, `NonlinearFunction`, `SplitFunction`, `DynamicalODEFunction`,
    `OptimizationFunction`
  - Solutions: `ODESolution`, `RODESolution`, `NonlinearSolution`,
    `SteadyStateSolution`, `OptimizationSolution`
  - Ensembles: `EnsembleProblem`, `EnsembleSolution`, `EnsembleSummary`,
    `EnsembleSerial`, `EnsembleThreads`, `EnsembleDistributed`,
    `EnsembleSplitThreads`, and the `EnsembleAnalysis` module
  - Solving: `solve`, `solve!`, `init`, `step!`, `remake`
  - Integrator interface: `reinit!`, `terminate!`, `u_modified!`, `add_tstop!`,
    `add_saveat!`, `savevalues!`, `get_du`, `get_du!`, `get_proposed_dt`,
    `set_proposed_dt!`
  - Return status: `ReturnCode`, `successful_retcode`
  - Callbacks: `ContinuousCallback`, `DiscreteCallback`, `VectorContinuousCallback`,
    `CallbackSet`
  - Algorithms: `DefaultODEAlgorithm`, `Tsit5`, `Vern6`, `Vern7`, `Vern8`, `Vern9`,
    `Rosenbrock23`, `Rodas5P`, `FBDF`, `AutoTsit5`, `AutoVern6`, `AutoVern7`,
    `AutoVern8`, `AutoVern9`
  - Differentiation choices: `AutoForwardDiff`, `AutoFiniteDiff`
  - The `DifferentialEquations`, `OrdinaryDiffEq` and `SciMLBase` modules themselves,
    for qualified access such as `SciMLBase.successful_retcode(sol.retcode)`

Anything else from DifferentialEquations — SciMLBase's internal traits and function
wrappers, the SciMLOperators operator algebra, and the problem/function/solution types
for equation classes EasyModelAnalysis does not analyze — must be imported from
DifferentialEquations (or its owning package) directly.

## Distributions — owned by Distributions.jl

Documented by [Distributions](https://juliastats.org/Distributions.jl/stable/). These
are the distributions passed to [`get_uncertainty_forecast`](basic_queries.md),
[`bayesian_datafit`](data_fitting_calibration.md),
[`get_sensitivity`](sensitivity_analysis.md) and
[`prob_violating_threshold`](threshold_interventions.md), plus the accessors used on
their results.

  - Continuous univariate: `Normal`, `LogNormal`, `Uniform`, `LogUniform`, `Beta`,
    `Gamma`, `InverseGamma`, `Exponential`, `Laplace`, `Cauchy`, `Chisq`, `TDist`,
    `FDist`, `Weibull`, `Rayleigh`, `Pareto`, `Logistic`, `Gumbel`, `Frechet`,
    `Arcsine`, `SkewNormal`
  - Discrete univariate: `Poisson`, `Binomial`, `Bernoulli`, `Geometric`,
    `NegativeBinomial`, `Categorical`, `Hypergeometric`, `DiscreteUniform`,
    `DiscreteNonParametric`, `Dirac`
  - Multivariate and matrix-variate: `MvNormal`, `MvLogNormal`, `Dirichlet`,
    `Multinomial`, `Wishart`, `InverseWishart`, `LKJ`
  - Composing: `Truncated`, `truncated`, `censored`, `MixtureModel`, `Product`,
    `product_distribution`
  - Supertypes: `Distribution`, `Sampleable`, `UnivariateDistribution`,
    `MultivariateDistribution`, `MatrixDistribution`,
    `ContinuousUnivariateDistribution`, `DiscreteUnivariateDistribution`,
    `ContinuousMultivariateDistribution`, `DiscreteMultivariateDistribution`,
    `Univariate`, `Multivariate`, `Continuous`, `Discrete`
  - Evaluation: `pdf`, `logpdf`, `cdf`, `logcdf`, `ccdf`, `logccdf`, `quantile`,
    `cquantile`, `insupport`, `support`
  - Statistics: `mean`, `median`, `mode`, `modes`, `var`, `std`, `cov`, `cor`,
    `skewness`, `kurtosis`, `entropy`, `kldivergence`, `loglikelihood`
  - Parameters: `params`, `partype`, `location`, `scale`, `shape`, `rate`, `probs`,
    `ncategories`, `ntrials`, `succprob`, `failprob`
  - Fitting and sampling: `fit`, `fit_mle`, `sampler`
  - The `Distributions` module itself, for qualified access

Anything else from Distributions — the exotic tail of the type hierarchy
(`WalleniusNoncentralHypergeometric`, the `Edgeworth*` and `ZeroMean*Canon`
families, `KSDist`, …) and the canonical-form machinery — must be imported from
Distributions directly.

## Plotting — owned by Plots.jl

Documented by [Plots](https://docs.juliaplots.org/stable/). These back the
`plot_extrema`, `phaseplot_extrema`, `plot_uncertainty_forecast` and
`create_sensitivity_plot` queries, and the plotting done in the tutorials.

  - Core verbs: `plot`, `plot!`, `plot3d`, `plot3d!`, `scatter`, `scatter!`,
    `scatter3d`, `scatter3d!`, `bar`, `bar!`, `histogram`, `histogram!`,
    `histogram2d`, `histogram2d!`, `stephist`, `stephist!`, `density`, `density!`,
    `boxplot`, `boxplot!`, `violin`, `violin!`, `pie`, `pie!`, `areaplot`,
    `areaplot!`, `sticks`, `sticks!`, `path3d`, `path3d!`
  - Surfaces and fields: `heatmap`, `heatmap!`, `contour`, `contour!`, `contourf`,
    `contourf!`, `surface`, `surface!`, `wireframe`, `wireframe!`, `mesh3d`,
    `mesh3d!`, `quiver`, `quiver!`, `spy`, `spy!`
  - Annotation and guides: `hline`, `hline!`, `vline`, `vline!`, `hspan`, `hspan!`,
    `vspan`, `vspan!`, `annotate!`, `lens!`, `title!`, `xlabel!`, `ylabel!`,
    `zlabel!`, `xlims`, `xlims!`, `ylims`, `ylims!`, `zlims!`, `xticks`, `xticks!`,
    `yticks`, `yticks!`, `zticks`, `zticks!`, `xaxis!`, `yaxis!`, `zaxis!`,
    `xflip!`, `yflip!`, `zflip!`, `xgrid!`, `ygrid!`, `zgrid!`, `xerror`, `xerror!`,
    `yerror`, `yerror!`, `twinx`, `twiny`, `grid`, `bbox`
  - Attributes: `font`, `text`, `stroke`, `brush`, `arrow`, `Shape`, `Surface`,
    `palette`, `cgrad`, `colormap`, `plot_color`, `distinguishable_colors`, `RGB`,
    `RGBA`, `Gray`, `Colorant`, `@colorant_str`, `theme`, `theme_palette`,
    `showtheme`, `default`, `plotattr`
  - Layouts and recipes: `@layout`, `@recipe`, `@series`, `@userplot`
  - Animation: `@animate`, `@gif`, `Animation`, `animate`, `frame`, `gif`, `mp4`,
    `mov`, `webm`
  - Output and backends: `savefig`, `png`, `gui`, `current`, `closeall`, `backend`,
    `backends`
  - The `Plots` module itself, for qualified access

Anything else from Plots — the individual backend selectors (`gr`, `plotly`,
`pyplot`, `unicodeplots`, …) and the ColorTypes color-space types it reexports
(`DIN99d`, `LCHuvA`, `AOklch`, `CIE1931_CMF`, …) — must be imported from Plots
directly.

## Keeping this page in sync

This list, the `export` blocks at the bottom of `src/EasyModelAnalysis.jl`, and the
`REEXPORTS` tuple in `test/qa/qa.jl` are the same list in three places. `test/qa/qa.jl`
checks that every approved name is actually reachable from `using EasyModelAnalysis`.
