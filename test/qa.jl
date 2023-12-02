using EasyModelAnalysis, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(EasyModelAnalysis)
    Aqua.test_ambiguities(EasyModelAnalysis, recursive = false)
    Aqua.test_deps_compat(EasyModelAnalysis)
    Aqua.test_piracies(EasyModelAnalysis,
        treat_as_own = [])
    Aqua.test_project_extras(EasyModelAnalysis)
    Aqua.test_stale_deps(EasyModelAnalysis)
    Aqua.test_unbound_args(EasyModelAnalysis)
    Aqua.test_undefined_exports(EasyModelAnalysis, broken = true)
end
