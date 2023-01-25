using Documenter, EasyModelAnalysis

# Make sure that plots don't throw a bunch of warnings / errors!
ENV["GKSwstype"] = "100"
using Plots

mathengine = MathJax3(Dict(:loader => Dict("load" => ["[tex]/require", "[tex]/mathtools"]),
                           :tex => Dict("inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                                        "packages" => [
                                            "base",
                                            "ams",
                                            "autoload",
                                            "mathtools",
                                            "require",
                                        ])))

makedocs(sitename = "Overview of Julia's SciML",
         authors = "Chris Rackauckas",
         modules = Module[EasyModelAnalysis],
         clean = true, doctest = false,
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
                                  canonical = "https://docs.sciml.ai/stable/",
                                  mathengine = mathengine),
         pages = [
             "EasyModelAnalysis.jl: Quick and Easy Queries to Simulation Results" => "index.md",
             "Getting Started with EasyModelAnalysis" => "getting_started.md",
             "API" => [
                 "basic_queries.md",
                 "data_fitting_calibration.md",
                 "sensitivity_analysis.md",
             ],
         ])

deploydocs(repo = "github.com/SciML/EasyModelAnalysis.jl")
