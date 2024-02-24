module EasyModelAnalysis

using LinearAlgebra
using Reexport
@reexport using DifferentialEquations
@reexport using ModelingToolkit
@reexport using Distributions
using Optimization, OptimizationBBO, OptimizationNLopt
using GlobalSensitivity, Turing
using SciMLExpectations
@reexport using Plots
using SciMLBase.EnsembleAnalysis

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

end
