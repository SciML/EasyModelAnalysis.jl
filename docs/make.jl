using Documenter, EasyModelAnalysis

# Make sure that plots don't throw a bunch of warnings / errors!
ENV["GKSwstype"] = "100"
using Plots

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

mathengine = MathJax3(Dict(:loader => Dict("load" => ["[tex]/require", "[tex]/mathtools"]),
    :tex => Dict("inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
        "packages" => [
            "base",
            "ams",
            "autoload",
            "mathtools",
            "require",
        ])))

makedocs(sitename = "EasyModelAnalysis.jl",
    authors = "Chris Rackauckas",
    modules = Module[EasyModelAnalysis],
    clean = true, doctest = false,
    linkcheck = true,
    strict = [
        :doctest,
        :linkcheck,
        :parse_error,
        :example_block,
        # Other available options are
        # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
    ],
    format = Documenter.HTML(analytics = "UA-90474609-3",
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/EasyModelAnalysis/stable/",
        mathengine = mathengine),
    pages = [
        "EasyModelAnalysis.jl: Quick and Easy Queries to Simulation Results" => "index.md",
        "Getting Started with EasyModelAnalysis" => "getting_started.md",
        "Tutorials" => [
            "tutorials/sensitivity_analysis.md",
            "tutorials/datafitting.md",
            "tutorials/threshold_interventions.md",
            "tutorials/probabilistic_thresholds.md",
        ],
        "Examples" => [
            "examples/petri.md",
            "examples/ASIR.md",
            "examples/SEIRHD.md",
            "examples/Carcione2020.md",
        ],
        "Scenarios" => [
            "scenarios/scenario1.md",
            "scenarios/scenario2.md",
            "scenarios/scenario3.md",
            "scenarios/scenario4.md",
            "scenarios/scenario5.md",
        ],
        "API" => [
            "api/basic_queries.md",
            "api/data_fitting_calibration.md",
            "api/sensitivity_analysis.md",
            "api/threshold_interventions.md",
        ],
    ])

deploydocs(repo = "github.com/SciML/EasyModelAnalysis.jl")
