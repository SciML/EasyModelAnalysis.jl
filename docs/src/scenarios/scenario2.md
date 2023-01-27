# Scenario 2: Limiting Hospitalizations

## Generate the Model and Dataset

```@example scenario2
using EasyModelAnalysis, AlgebraicPetri
using Catlab, Catlab.CategoricalAlgebra, Catlab.Programs, Catlab.WiringDiagrams, Catlab.Graphics

types′ = LabelledPetriNet([:Pop],
    :infect=>((:Pop, :Pop)=>(:Pop, :Pop)),
    :disease=>(:Pop=>:Pop),
    :strata=>(:Pop=>:Pop))
types = map(types′, Name=name->nothing)
# Parts of type system for ease of reference
s, = parts(types′, :S)
t_interact, t_disease, t_strata = parts(types′, :T)
i_interact1, i_interact2, i_disease, i_strata = parts(types′, :I)
o_interact1, o_interact2, o_disease, o_strata = parts(types′, :O);

function formSEIRHD()
  SEIRHD = LabelledPetriNet([:S, :E, :I, :R, :H, :D],
  :inf => ((:S, :I)=>(:E, :I)),
  :conv => (:E=>:I),
  :rec => (:I=>:R),
  :hosp => (:I=>:H),
  :death => (:H=>:D)
)
  return SEIRHD
end

seirhd = formSEIRHD()

seirhd_typed = ACSetTransformation(seirhd, types,
  S=[s, s, s, s, s, s],
  T=[t_interact, t_disease, t_disease, t_disease, t_disease],
  I=[i_interact1, i_interact2, i_disease, i_disease, i_disease, i_disease],
  O=[o_interact1, o_interact2, o_disease, o_disease, o_disease, o_disease],
  Name=name -> nothing
)

@assert is_natural(seirhd_typed)

vax_lpn = LabelledPetriNet([:U, :V],
  :infuu => ((:U, :U) => (:U, :U)),
  :infvu => ((:V, :U) => (:V, :U)),
  :infuv => ((:U, :V) => (:U, :V)),
  :infvv => ((:V, :V) => (:V, :V)),
  :vax => (:U => :V),
)
# vax_aug = augLabelledPetriNet(vax_lpn, vax_aug_st)

Vax_aug_typed = ACSetTransformation(vax_lpn, types,
  S=[s, s],
  T=[t_interact, t_interact, t_interact, t_interact, t_strata],
  I=[i_interact1, i_interact2, i_interact1, i_interact2, i_interact1, i_interact2, i_interact1, i_interact2, i_strata],
  O=[o_interact1, o_interact2, o_interact1, o_interact2, o_interact1, o_interact2, o_interact1, o_interact2, o_strata],
  Name=name -> nothing
)

@assert is_natural(Vax_aug_typed)

function stratify_typed(pn1, pn2, type_system)
  pn1′, pn2′ = [add_cross_terms(pn, type_system) for pn in [pn1, pn2]]
  pb = pullback(pn1′, pn2′) 
  return first(legs(pb)) ⋅ pn1′
end

seirhd_vax = stratify_typed(
  seirhd_typed=>[[:strata],[:strata],[:strata],[:strata],[:strata],[]],
  Vax_aug_typed=>[[:disease,:infect],[:disease,:infect]],
  types′)

@assert is_natural(seirhd_vax)

sys = ODESystem(seirhd_vax)

# Things that need values

prob = nothing
p_init = nothing # or box constraints
tsave, data = nothing, nothing
```

## Model Analysis

> Parameterize model either using data from the previous two months (October 28th – December 28th, 2021), or with relevant parameter values from the literature.

```julia
fit = datafit(prob, p_init, tsave, data)
```

### Question 1

> Forecast Covid cases and hospitalizations over the next 3 months under no interventions.

### Question 2

> Based on the forecast, do we need interventions to keep total Covid hospitalizations under a threshold of 3000 on any given day? If there is uncertainty in the model parameters, express the answer probabilistically, i.e., what is the likelihood or probability that the number of Covid hospitalizations will stay under this threshold for the next 3 months without interventions?

### Question 3

> Assume a consistent policy of social distancing/masking will be implemented, resulting in a 50% decrease from baseline transmission. Assume that we want to minimize the time that the policy is in place, and once it has been put in place and then ended, it can't be re-implemented. Looking forward from “today’s” date of Dec. 28, 2021, what are the optimal start and end dates for this policy, to keep projections below the hospitalization threshold over the entire 3-month period? How many fewer hospitalizations and cases does this policy result in?

### Question 4

> Assume there is a protocol to kick in mitigation policies when hospitalizations rise above 80% of the hospitalization threshold (i.e. 80% of 3000). When hospitalizations fall back below 80% of the threshold, these policies expire.

> When do we expect these policies to first kick in?

> What is the minimum impact on transmission rate these mitigation policies need to have the first time they kick in, to (1) ensure that we don't reach the hospitalization threshold at any time during the 3-month period, and (2) ensure that the policies only need to be implemented once, and potentially expired later, but never reimplemented? Express this in terms of change in baseline transmission levels (e.g. 10% decrease, 50% decrease, etc.).

### Question 5

> Now assume that instead of NPIs, the Board wants to focus all their resources on an aggressive vaccination campaign to increase the fraction of the total population that is vaccinated. What is the minimum intervention with vaccinations required in order for this intervention to have the same impact on cases and hospitalizations, as your optimal answer from question 3? Depending on the model you use, this may be represented as an increase in total vaccinated population, or increase in daily vaccination rate (% of eligible people vaccinated each day), or some other representation.
