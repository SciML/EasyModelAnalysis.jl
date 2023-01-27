# Scenario 2: Limiting Hospitalizations

## Generate the Model and Dataset

```@example scenario2
using EasyModelAnalysis, AlgebraicPetri
using Catlab, Catlab.CategoricalAlgebra, Catlab.Programs, Catlab.WiringDiagrams, Catlab.Graphics

types = LabelledPetriNet([:Pop],
    :infect=>((:Pop, :Pop)=>(:Pop, :Pop)),
    :disease=>(:Pop=>:Pop),
    :strata=>(:Pop=>:Pop))

seirhd_uwd = @relation () where (S::Pop, E::Pop, I::Pop, R::Pop, H::Pop, D::Pop) begin
  infect(S,I,E,I)
  disease(E,I)
  disease(I,R)
  disease(I,H)
  disease(H,D)
end

seirhd_typed = oapply_typed(types, seirhd_uwd, [:inf, :conv, :rec, :hosp, :death])

@assert is_natural(seirhd_typed)

vax_lpn_uwd = @relation () where (U::Pop, V::Pop) begin
  infect(U,U,U,U)
  infect(V,U,V,U)
  infect(U,V,U,V)
  infect(V,V,V,V)
  strata(U,V)
end

vax_lpn = LabelledPetriNet([:U, :V],
  :infuu => ((:U, :U) => (:U, :U)),
  :infvu => ((:V, :U) => (:V, :U)),
  :infuv => ((:U, :V) => (:U, :V)),
  :infvv => ((:V, :V) => (:V, :V)),
  :vax => (:U => :V),
)

vax_lpn_typed = oapply_typed(types, vax_lpn_uwd, [:infuu, :infvu, :infuv, :infvv, :vax])

seirhd_vax_typed = typed_product(seirhd_typed, vax_lpn_typed)

# The names for the typed product are tuples; here we combine them with an underscore
seirhd_vax = map(
  seirhd_vax_typed.dom,
  Name=t -> Symbol(string(t[1]) * "_" * string(t[2]))
)

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
