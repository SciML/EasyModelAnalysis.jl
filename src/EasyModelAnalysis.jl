module EasyModelAnalysis

using LinearAlgebra
using Reexport
@reexport using DifferentialEquations
@reexport using ModelingToolkit
@reexport using Distributions
using Optimization, OptimizationBBO, OptimizationNLopt
using GlobalSensitivity, Plots, Turing

include("basics.jl")
include("datafit.jl")
include("sensitivity.jl")
include("threshold.jl")

export get_timeseries, get_min_t, get_max_t, plot_extrema, phaseplot_extrema
export datafit, bayesian_datafit
export get_sensitivity, create_sensitivity_plot
export stop_at_threshold
end
