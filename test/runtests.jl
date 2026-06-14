using SafeTestsets, Test
using SciMLTesting

run_tests(;
    core = () -> begin
        @testset "Core" begin
            @time @safetestset "Quality Assurance" include("qa.jl")
            @time @safetestset "Basic Tests" include("basics.jl")
            @time @safetestset "Sensitivity Tests" include("sensitivity.jl")
            @time @safetestset "Ensemble Tests" include("ensemble.jl")
            @time @safetestset "Threshold Tests" include("threshold.jl")
            @time @safetestset "Example Tests" include("examples.jl")
        end
    end,
    groups = Dict(
        "Datafit" => () -> begin
            @time @safetestset "Datafit Tests" include("datafit.jl")
        end,
    ),
    all = ["Core", "Datafit"],
)
