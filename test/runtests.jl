using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

@time begin
    if GROUP == "All" || GROUP == "Core"
        @time @safetestset "Basic Tests" include("basics.jl")
        @time @safetestset "Sensitivity Tests" include("sensitivity.jl")
        VERSION >= v"1.9" && @time @safetestset "Ensemble Tests" include("ensemble.jl")
        @time @safetestset "Threshold Tests" include("threshold.jl")
        @time @safetestset "Example Tests" include("examples.jl")
    if GROUP == "All" || GROUP == "Datafit"
        @time @safetestset "Datafit Tests" include("datafit.jl")
    end
end
