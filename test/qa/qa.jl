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
            # DynamicPPL.acclogp!! is reached through Turing's re-export; DynamicPPL
            # is not a direct dependency, so accessing it via Turing is intentional.
            ignore = (:DynamicPPL,),
        ),
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractMCMCEnsemble,  # AbstractMCMC: no public alias for the ensemble type
                :DynamicPPL,            # Turing: submodule re-export, no public alias
                :LN_SBPLX,              # NLopt: algorithm constant, not marked public
            ),
        ),
    ),
    # no_implicit_imports: heavy reexport / bulk-using of the SciML stack
    # (SciML/EasyModelAnalysis.jl#301)
    ei_broken = (:no_implicit_imports,),
)
