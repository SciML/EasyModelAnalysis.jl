using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

@time begin if GROUP == "All" || GROUP == "Core"
    @time @safetestset "Basic Tests" begin include("basics.jl") end
    @time @safetestset "Sensitivity Tests" begin include("sensitivity.jl") end
    @time @safetestset "Threshold Tests" begin include("threshold.jl") end
    @time @safetestset "Example Tests" begin include("examples.jl") end
elseif GROUP == "All" || GROUP == "Datafit"
    @time @safetestset "Datafit Tests" begin include("datafit.jl") end
end end
