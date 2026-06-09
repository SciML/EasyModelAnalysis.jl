using SafeTestsets, Test

const GROUP = get(ENV, "GROUP", "All")

@time begin
    if GROUP == "All" || GROUP == "Core"
        @testset "Core" begin
            @time @safetestset "Basic Tests" include("basics.jl")
            @time @safetestset "Sensitivity Tests" include("sensitivity.jl")
            @time @safetestset "Ensemble Tests" include("ensemble.jl")
            @time @safetestset "Threshold Tests" include("threshold.jl")
            @time @safetestset "Example Tests" include("examples.jl")
        end
    end

    if GROUP == "All" || GROUP == "Datafit"
        @time @safetestset "Datafit Tests" include("datafit.jl")
    end

    if GROUP == "QA"
        using Pkg
        Pkg.activate(joinpath(@__DIR__, "qa"))
        Pkg.develop(path = joinpath(@__DIR__, ".."))
        Pkg.instantiate()
        @time @safetestset "Quality Assurance" include("qa.jl")
    elseif GROUP == "All"
        @time @safetestset "Quality Assurance" include("qa.jl")
    end
end
