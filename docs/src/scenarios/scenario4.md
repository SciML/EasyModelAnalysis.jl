# Scenario 4: Testing and Return to Campus

Load packages:

```@example scenario3
using EasyModelAnalysis
using AlgebraicPetri
using UnPack
using Dates
```

## Assumptions from the Scenario

> University of Michigan (Ann Arbor campus)
> QUESTION: TA1 reachback for total population?

```@example scenario4
#=
From: https://obp.umich.edu/wp-content/uploads/pubdata/almanac/Almanac_Ch1_June2022.pdf#:~:text=Based%20on%20the%20November%202021,All%20other%20staff%20total%2015%2C795.
Undergraduate Students ................................ 32,282
Graduate Students ......................................... 15,268
Professional Students...................................... 2,728
Tenured/Tenure-Track Faculty........................ 3,136
Lecturers.......................................................... 1,044
Clinical Faculty ............................................... 2,268
Research Faculty ................................................ 868
Other Academic ................................................. 403
Research Fellows/Post-Doctoral Fellows........ 1,105
Staff............................................................... 15,795
Ann Arbor Campus Total1............................. 74,897
=#

N_ugrad = 32_282
N_grad = 15_268 + 2_728
N_staff = 3_136 + 1_044 + 2_268 + 868 + 403 + 1_105 + 15_795
NN = N_ugrad + N_grad + N_staff
```

> Assume the following numbers of true infections at the onset of the term by population type:
> Undergraduate=750
> Graduate/professional=250
> Faculty/staff=100

```@example scenario4
I_ugrad = 750
I_grad = 250
I_staff = 100
```

> Time/Setting: It is late 2021 and you are planning for the Spring 2022 term at the University of Michigan (Ann Arbor campus) beginning in early January 2022. For the purpose of this scenario, consider a four-month period that begins > January 1st and ends May 1st.

```@example scenario4
tstart = Date(2022, 01, 01)
tend = Date(2022, 05, 01)
tdays = (tend - tstart).value
```

## Generate the Model and Dataset

```@example scenario4
using AlgebraicPetri
using ModelingToolkit
using Catlab
using Catlab.CategoricalAlgebra, Catlab.Programs.RelationalPrograms, Catlab.WiringDiagrams.WiringDiagramAlgebras
using ASKEM.Dec2022Demo: typed_stratify

#*****
# Q1 *
#*****
# Define test strategy, minimize total testing whil maintaining infections below isolation capacity
# Can have unique test type and frequency per cohort. 
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

# SEIR
function form_seir()
  SEIR = LabelledPetriNet([:S, :E, :I, :R],
  :inf => ((:S, :I)=>(:E, :I)),
  :conv => (:E=>:I),
  :rec => (:I=>:R),
)
  return SEIR
end

#=seiir = formSEIIR()
seiir_aug = augLabelledPetriNet(seiir,[:S, :E, :I, :R])
seiir_typed = ACSetTransformation(seiir_aug, types,
  S = [s, s, s, s, s],
  T = [t_interact, t_disease, t_disease, t_disease, t_disease, t_strata, t_strata, t_strata, t_strata],
  I = [i_interact1, i_interact2, i_disease, i_disease, i_disease, i_disease, i_strata, i_strata, i_strata, i_strata],
  O = [o_interact1, o_interact2, o_disease, o_disease, o_disease, o_disease, o_strata, o_strata, o_strata, o_strata],
  Name = name -> nothing 
  )
@assert is_natural(seiir_typed)
=#

function form_test_iso(st, ed)
    states = [Symbol(st), Symbol("Iso_"*st)]
    if st != ed
        states = push!(states,Symbol(ed))
    end
    # test_and_iso = LabelledPetriNet([Symbol(st), Symbol("Iso_"*st)],
    test_and_iso = LabelledPetriNet(states,
    Symbol("t_rapid_pos_"*st) => (Symbol(st)=>Symbol("Iso_"*st)),
    Symbol("t_rapid_neg_"*st) => (Symbol(st)=>Symbol(st)),
    Symbol("t_pcr_pos_"*st) => (Symbol(st)=>Symbol("Iso_"*st)),
    Symbol("t_pcr_neg_"*st) => (Symbol(st)=>Symbol(st)),
    Symbol("deiso_"*st) => (Symbol("Iso_"*st)=>Symbol(ed))
  )
    return test_and_iso
end

testing_composition_pattern = @relation (S, E, I, R, Iso_s, Iso_e, Iso_i, Iso_r) where (S, E, I, R, Iso_s, Iso_e, Iso_i, Iso_r) begin
  SEIR(S, E, I, R)
  TI_S(S, Iso_s, S)
  TI_E(E, Iso_e, R)
  TI_I(I, Iso_i, R)
  TI_R(R, Iso_r, R)
  # cross_exposure(S, E, I, Sv, Ev, Iv)
end

seir = form_seir()
test_iso_s = form_test_iso("S", "S")
test_iso_e = form_test_iso("E", "R")
test_iso_i = form_test_iso("I", "R")
test_iso_r = form_test_iso("R", "R")

#= cross_exposure = Open(LabelledPetriNet([:S, :E, :I, :Sv, :Ev, :Iv],
  :inf_uv => ((:S, :Iv) => (:E, :Iv)),
  :inf_vu => ((:Sv, :I) => (:Ev, :I))
))=#

seir_test_iso = oapply(testing_composition_pattern, Dict(
  :SEIR => Open(seir),
  :TI_S => Open(test_iso_s),
  :TI_E => Open(test_iso_e),
  :TI_I => Open(test_iso_i),
  :TI_R => Open(test_iso_r)
  # :cross_exposure => cross_exposure
)) |> apex

function form_cohort()
    cohort = LabelledPetriNet([:U, :G, :F],
    :inf_uu => ((:U, :U)=>(:U, :U)),
    :inf_ug => ((:U, :G)=>(:U, :G)),
    :inf_uf => ((:U, :F)=>(:U, :F)),
    :inf_gu => ((:G, :U)=>(:G, :U)),
    :inf_gg => ((:G, :G)=>(:G, :G)),
    :inf_gf => ((:G, :F)=>(:G, :F)),
    :inf_fu => ((:F, :U)=>(:F, :U)),
    :inf_fg => ((:F, :G)=>(:F, :G)),
    :inf_ff => ((:F, :F)=>(:F, :F)),
    :dis_u => (:U => :U),
    :dis_g => (:G => :G),
    :dis_f => (:F => :F),
    :strat_u => (:U => :U),
    :strat_g => (:G => :G),
    :strat_f => (:F => :F)
  )
    return cohort
  end

cohort = form_cohort()
cohort_typed = ACSetTransformation(cohort, types,
    S = [s, s, s],
    T = [t_interact, t_interact, t_interact, t_interact, t_interact, t_interact, t_interact, t_interact, t_interact, t_disease, t_disease, t_disease, t_strata, t_strata, t_strata],
    I = [i_interact1, i_interact2, i_interact1, i_interact2, i_interact1, i_interact2, i_interact1, i_interact2, i_interact1, i_interact2, i_interact1, i_interact2, i_interact1, i_interact2, i_interact1, i_interact2, i_interact1, i_interact2, i_disease, i_disease, i_disease, i_strata, i_strata, i_strata],
    O = [o_interact1, o_interact2, o_interact1, o_interact2, o_interact1, o_interact2, o_interact1, o_interact2, o_interact1, o_interact2, o_interact1, o_interact2, o_interact1, o_interact2, o_interact1, o_interact2, o_interact1, o_interact2, o_disease, o_disease, o_disease, o_strata, o_strata, o_strata],
    Name = name -> nothing 
    )
  @assert is_natural(cohort_typed)
  

seir_test_iso_typed = ACSetTransformation(seir_test_iso, types,
  S = [s, s, s, s, s, s, s, s],
  T = [t_interact, t_disease, t_disease, t_strata, t_strata, t_strata, t_strata, t_strata, t_strata, t_strata, t_strata, t_strata, t_strata, t_strata, t_strata, t_strata, t_strata, t_strata, t_strata, t_strata, t_strata, t_strata, t_strata],
  I = [i_interact1, i_interact2, i_disease, i_disease, i_strata, i_strata, i_strata, i_strata, i_strata, i_strata, i_strata, i_strata, i_strata, i_strata, i_strata, i_strata, i_strata, i_strata, i_strata, i_strata, i_strata, i_strata, i_strata, i_strata],
  O = [o_interact1, o_interact2, o_disease, o_disease, o_strata, o_strata, o_strata, o_strata, o_strata, o_strata, o_strata, o_strata, o_strata, o_strata, o_strata, o_strata, o_strata, o_strata, o_strata, o_strata, o_strata, o_strata, o_strata, o_strata],
  Name = name -> nothing 
  )
@assert is_natural(seir_test_iso_typed)

seir_test_iso_cohort = typed_stratify(seir_test_iso_typed, cohort_typed)

sys = ODESystem(map(
        dom(seir_test_iso_cohort),
        Name=t -> Symbol(string(t[1]) * "_" * string(t[2]))
    ))

# TODO: TA1 needs to hand off to TA2
ps = parameters(sys)
rates = ps .=> rand(length(ps)) * NN

@unpack S_U, S_G, S_F, I_U, I_G, I_F = sys

u0 = merge(Dict(states(sys) .=> 0),
    Dict(S_U => Float64(N_ugrad - I_ugrad), S_G => Float64(N_grad - I_grad), S_F => Float64(N_staff - I_staff),
    I_U => Float64(I_ugrad), I_G => Float64(I_grad), I_F => Float64(I_staff)))
```

```@example scenario4
prob = ODEProblem(sys, u0, (0.0, tdays))
sol = solve(prob)
plot(sol)
```

## Model Analysis

### Question 1

> Define a return-to-campus strategy that minimizes total testing while
> maintaining infections below the initial isolation bed capacity of 430. The
> testing scheme can include an arrival testing strategy in addition to unique
> testing approaches within time periods of the simulation. Cohorts can have
> unique testing strategies defined by test type and number per week.

```@example scenario4
# Minimize u_test subject to IS <= 430
p_opt, s2, ret = optimal_parameter_threshold(prob, IS, 430, u_test, [u_test], [0.0], [NN],
                                             maxtime = 10);
plot(s2, idxs = [IS])
```

```@example scenario4
p_opt, s2, ret = optimal_parameter_threshold(prob, D, 430, u_test, [u_test], [0.0], [

],
                                             maxtime = 10);
plot(s2, idxs = [I, IS, D])
```

### Question 4

> Challenge question: assume that antigen tests are one fifth the cost of PCR
> tests but also much less (~half) as sensitive. Incorporate the cost of the
> testing program into your recommendations.

```@example scenario4
# Minimize u_test subject to IS <= 430
p_opt, s2, ret = optimal_parameter_threshold(prob, IS, 430, 5 * u_test, [u_test], [0.0],
                                             [NN], maxtime = 10);
plot(s2, idxs = [IS])
```
