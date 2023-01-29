# Scenario 1: Vaccination

## Generate the Model and Dataset

### Setup

```
using Catlab, Catlab.CategoricalAlgebra, Catlab.Programs, Catlab.WiringDiagrams, Catlab.Graphics
using AlgebraicPetri
using AlgebraicPetri.BilayerNetworks
using AlgebraicDynamics.UWDDynam
using LabelledArrays
using OrdinaryDiffEq, DelayDiffEq
using Plots

using ASKEM.Dec2022Demo: formSIRD, formInfType, augLabelledPetriNet, sirdAugStates, typeSIRD,
  makeMultiAge, typeAge, typed_stratify, formVax, vaxAugStates, typeVax, writeMdlStrat,
  loadSVIIvR, sviivrAugStates, typeSVIIvR
using ASKEM.SubACSets: mca
using ASKEM.Stratify: stratify_typed

types′ = LabelledPetriNet([:Pop],
  :infect => ((:Pop, :Pop) => (:Pop, :Pop)),
  :disease => (:Pop => :Pop),
  :strata => (:Pop => :Pop),
  :natural => (:Pop => :Pop),
  )
types = map(types′, Name=name -> nothing)

# Parts of type system for ease of reference
s, = parts(types′, :S)
t_interact, t_disease, t_strata,t_natural  = parts(types′, :T)
i_interact1, i_interact2, i_disease, i_strata, i_natural = parts(types′, :I)
o_interact1, o_interact2, o_disease, o_strata, o_natural = parts(types′, :O);
```

### Original SEIRD model from the paper

```
seirdnat = LabelledPetriNet([:S, :E, :I, :R, :D],
  :inf => ((:S, :I) => (:E, :I)),
  :conv => (:E => :I),
  :rec => (:I => :R),
  :death => (:I => :D),
  :nat_d_s => (:S => ()),
  :nat_d_e => (:E => ()),
  :nat_d_i => (:I => ()),
  :nat_d_r => (:R => ()),
  :nat_birth => (() => :S),
)

# seirdnat_aug = augLabelledPetriNet(seirdnat, [:S, :E, :I, :R])
seirdnat_typed = ACSetTransformation(seirdnat, types,
  S=[s, s, s, s, s],
  T=[t_interact, t_disease, t_disease, t_disease, t_disease, t_disease, t_disease, t_disease, t_natural],
  I=[i_interact1, i_interact2, i_disease, i_disease, i_disease, i_disease, i_disease, i_disease, i_disease],
  O=[o_interact1, o_interact2, o_disease, o_disease, o_disease, o_natural],
  Name=name -> nothing
)
@assert is_natural(seirdnat_typed)
```

### Model of vaccination process

```
vax_lpn = LabelledPetriNet([:U, :V],
  :infuu => ((:U, :U) => (:U, :U)),
  :infvu => ((:V, :U) => (:V, :U)),
  :infuv => ((:U, :V) => (:U, :V)),
  :infvv => ((:V, :V) => (:V, :V)),
  :vax => (:U => :V),
)

Vax_aug_typed = ACSetTransformation(vax_lpn, types,
  S=[s, s],
  T=[t_interact, t_interact, t_interact, t_interact, t_strata],
  I=[i_interact1, i_interact2, i_interact1, i_interact2, i_interact1, i_interact2, i_interact1, i_interact2, i_strata],
  O=[o_interact1, o_interact2, o_interact1, o_interact2, o_interact1, o_interact2, o_interact1, o_interact2, o_strata],
  Name=name -> nothing
)
@assert is_natural(Vax_aug_typed)
```

### Original model stratified with vaccination

```
seirdnat_vax = stratify_typed(
  seirdnat_typed=>[[:strata],[:strata],[:strata],[:strata],[]],
  Vax_aug_typed=>[[:disease,:natural],[:disease,]],
  types′)
```

conv: exposed => infected

### Model 3.a.i for comparison

```
# SEIRDnat "stratified with vax"
function formSEIRDnatV()
  SEIRDnatV = LabelledPetriNet([:Sv, :Ev, :Iv, :Rv, :D],
    :inf => ((:Sv, :Iv) => (:Ev, :Iv)),
    :conv => (:Ev => :Iv),
    :rec => (:Iv => :Rv),
    :death => (:Iv => :D),
    :nat_d_s => (:Sv => ()),
    :nat_d_e => (:Ev => ()),
    :nat_d_i => (:Iv => ()),
    :nat_d_r => (:Rv => ()),
  )
  return SEIRDnatV
end

seirdnat_v = formSEIRDnatV()
```

### Model 3.a.ii for comparison

```
# CHIMESVIIvR
sviivr_lbn_pth = joinpath(@__DIR__, "CHIME_SVIIvR_dynamics_BiLayer.json")
sviivr_lbn = read_json_acset(LabelledBilayerNetwork, sviivr_lbn_pth)
sviivr = LabelledPetriNet()
migrate!(sviivr, sviivr_lbn)
```

## Model Analysis

### Question 3 Numerical Comparison

> Compare simulation outputs between the three models, for the following two scenarios. Assume initial values and parameter values are consistent (to the extent possible) with Table 1 in https://biomedres.us/pdfs/BJSTR.MS.ID.007413.pdf. For initial values that are not specified, choose reasonable values and ensure they are the same between the three models being compared.
> i.	Vaccine efficacy = 75%, population vaccinated = 10%
> ii.	Vaccine efficacy = 75%, population vaccinated = 80%

E(0) = 99500      # exposed
I(0) = 1          # infected
recovered, deceased = 0
N = 10000000
mu = 0.012048     # death rate
alpha = 0.00142   # fatality rate among unvaccinated
alpha_v = 0.00142 # fatality rate among vaccinated
beta_uu = 0.75    # probability of transmission per unvax contact * # of unvax contacts per time
gamma^-1 = 3.31   # reciprocal of recovery rate of unvax
gamma_v^-1 = 3.31 # " vax
eps^-1 = 5.7      # reciprocal of rate of exposed,unvax => infectious,unvax
eps_v^-1 = 5.79   # " vax
xi = 0.5          # vaccine efficacy
kappa             # fraction vaccinated

#### Run model 3ai

system = ODESystem(seirdnat_v)
prob = ODEProblem(system, [10000000-99500, 99500, 1, 0, 0.0], [0, 100],
                  [0.75, 1/5.7, 1/3.31, 0.012048, 1e-3, 1e-3, 1e-3, 1e-3])


### Question 4

> Create an equally weighted ensemble model using the three models in 3b, and replicate the scenarios in 3.c.i and 3.c.ii. How does the ensemble model output compare to the output from the individual component models?

### Question 5

> For any of the models in question 3, conduct a sensitivity analysis to determine which intervention parameters should be prioritized in the model, for having the greatest impact on deaths – NPIs, or vaccine-related interventions?

### Question 6

> With the age-stratified model, simulate the following situations. You may choose initial values that seem reasonable given the location and time, and you can reuse values from any of the publications referenced):
> i.	High vaccination rate among older populations 65 years and older (e.g. 80%+), and low vaccination rate among all other age groups (e.g. below 15%)
> ii.	High vaccination rate among all age groups
> iii.	Repeat d.i and d.ii, but now add a social distancing policy at schools, that decreases contact rates by 20% for school-aged children only.
> iv.	Compare and summarize simulation outputs for d.i-d.iii
