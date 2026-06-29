using EasyModelAnalysis, Aqua
@testset "Aqua" begin
    # find_persistent_tasks_deps Pkg.develop's each dependency. On Julia 1.12 a Pkg
    # regression (JuliaLang/Pkg.jl#4587) makes Pkg.develop honor a developed
    # dependency's relative [sources] and resolve it against the depot, so
    # OptimizationBBO (which pins its in-repo OptimizationBase via [sources], as
    # every Optimization.jl sublibrary does) errors with "expected package
    # OptimizationBase to exist at path .../OptimizationBBO/OptimizationBase". The
    # [sources] is correct and load-bearing for the monorepo; the bug is upstream.
    # Remove this version gate once Pkg.jl#4587 is fixed (see EasyModelAnalysis#303).
    if VERSION < v"1.12"
        Aqua.find_persistent_tasks_deps(EasyModelAnalysis)
    end
    Aqua.test_ambiguities(EasyModelAnalysis, recursive = false)
    Aqua.test_deps_compat(EasyModelAnalysis)
    Aqua.test_piracies(
        EasyModelAnalysis,
        treat_as_own = []
    )
    Aqua.test_project_extras(EasyModelAnalysis)
    Aqua.test_stale_deps(EasyModelAnalysis)
    Aqua.test_unbound_args(EasyModelAnalysis)
    Aqua.test_undefined_exports(EasyModelAnalysis, broken = true)
end
