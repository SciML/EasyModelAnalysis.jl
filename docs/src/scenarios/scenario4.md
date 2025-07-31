# Scenario 4: Testing and Return to Campus

Load packages:

```@example scenario4
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
function formSEIISRD()
    SEIRHD = LabelledPetriNet([:S, :E, :I, :IS, :R, :D],
        :expo => ((:S, :I) => (:E, :I)),
        :conv => (:E => :I),
        :rec => (:I => :R),
        :test => (:I => :IS),
        :leave => (:IS => :R),
        :death => (:I => :D))
    return SEIRHD
end
sys1 = ODESystem(formSEIISRD())

@unpack S, E, I, IS, R, D = sys1
@unpack expo, conv, rec, test, leave, death = sys1

@parameters u_expo=0.1*NN u_conv=0.1*NN u_rec=0.8*NN u_death=0.1*NN u_test=0.9*NN u_leave=0.2*
NN N=NN
translate_params = [expo => u_expo / NN,
    conv => u_conv / NN,
    rec => u_rec / NN,
    death => u_death / NN,
    test => u_test / NN,
    leave => u_leave / NN
]
subed_sys = substitute(sys1, translate_params)
sys = add_accumulations(subed_sys, [I])
@unpack accumulation_I = sys
```

```@example scenario4
I_total = I_ugrad + I_grad + I_staff
u0init = [
    S => NN - I_total,
    E => 0,
    I => I_total,
    IS => 0,
    R => 0,
    D => 0
]

prob = ODEProblem(sys, u0init, (0.0, tdays))
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
p_opt, s2,
ret = optimal_parameter_threshold(prob, IS, 430, u_test, [u_test], [0.0], [NN],
    maxtime = 10);
plot(s2, idxs = [IS])
```

```@example scenario4
p_opt, s2,
ret = optimal_parameter_threshold(prob, D, 430, u_test, [u_test], [0.0], [NN],
    maxtime = 10);
plot(s2, idxs = [I, IS, D])
```

### Question 4

> Challenge question: assume that antigen tests are one fifth the cost of PCR
> tests but also much less (~half) as sensitive. Incorporate the cost of the
> testing program into your recommendations.

```@example scenario4
# Minimize u_test subject to IS <= 430
p_opt, s2,
ret = optimal_parameter_threshold(prob, IS, 430, 5 * u_test, [u_test], [0.0],
    [NN], maxtime = 10);
plot(s2, idxs = [IS])
```
