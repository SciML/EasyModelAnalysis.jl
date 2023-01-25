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

export get_timeseries, get_min_t, get_max_t, datafit, get_sensitivity,
       create_sensitivity_plot
end
