# Sensitivity Analysis of the Carcione2020 Epidemic Model

```@example carcione
cd(@__DIR__)
using SBMLToolkit, ModelingToolkit, EasyModelAnalysis, UnPack

xmlfile = "../assets/Carcione2020.xml"

SBMLToolkit.checksupport_file(xmlfile)
mdl = readSBML(xmlfile, doc -> begin
    set_level_and_version(3, 2)(doc)
    convert_simplify_math(doc)
end)

rs = ReactionSystem(mdl)  # If you want to create a reaction system
odesys = convert(ODESystem, rs)  # Alternatively: ODESystem(mdl)
```

```@example carcione
sys = structural_simplify(odesys)
```

```@example carcione
@unpack Infected, Exposed, Deceased, Recovered, Total_population, Susceptible = sys
@unpack alpha, epsilon, gamma, mu, beta, City = sys
tspan = (0.0, 1.0)
prob = ODEProblem(odesys, [], tspan, [])
sol = solve(prob, Rodas5())
plot(sol, idxs = Deceased)
```

```@example carcione
pbounds = [
    alpha => [0.003, 0.006],
    epsilon => [1 / 6, 1 / 2],
    gamma => [0.1, 0.2],
    mu => [0.01, 0.02],
    beta => [0.7, 0.9],
]
create_sensitivity_plot(prob, 100.0, Deceased, pbounds; samples = 2000)
```
