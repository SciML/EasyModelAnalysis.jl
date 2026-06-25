using SciMLTesting, EasyModelAnalysis, Test

run_qa(
    EasyModelAnalysis;
    explicit_imports = true,
    aqua_kwargs = (; ambiguities = (; recursive = false)),
    # undefined_exports: `Variable` and `rotate!` leak in (dead) via `@reexport`
    # (SciML/EasyModelAnalysis.jl#300)
    aqua_broken = (:undefined_exports,),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (;
            ignore = (
                :AbstractSystem,    # ModelingToolkitBase (accessed via ModelingToolkit)
                :DynamicPPL,        # DynamicPPL (accessed via Turing)
                :unwrap,            # SymbolicUtils (accessed via Symbolics)
            ),
        ),
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractMCMCEnsemble,  # AbstractMCMC (not public)
                :AbstractSystem,        # ModelingToolkit (not public)
                :DynamicPPL,            # Turing (not public)
                :LN_SBPLX,              # NLopt (not public)
                :successful_retcode,    # SciMLBase (not public)
                :unwrap,                # Symbolics (not public)
            ),
        ),
    ),
    # no_implicit_imports: heavy reexport / bulk-using of the SciML stack
    # (SciML/EasyModelAnalysis.jl#301)
    ei_broken = (:no_implicit_imports,),
)
