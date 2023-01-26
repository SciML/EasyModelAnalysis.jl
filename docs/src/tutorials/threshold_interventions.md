# Designing Threshold Interventions

In this tutorial we will demonstrate how to analyze "threshold interventions", i.e. how to do calculations of model results
to determine quantities around thresholds and the interventions to fix/avoid thresholds. Thresholds are meant to answer
queries like "what is the first time point where the number of COVID cases will cross 10% of the population?", or
"when will the ball reach the wall?". Intervention is then an assessment of policies around such thresholds. For example,
"when should we start and stop an enforced masking policy to keep the total number of COVID cases below 10% for the full
time period?". Thus these two analyses go hand in hand: threshold queries give us information about when thresholds will
be reached, while intervention functions give us optimal strategies for avoiding such thresholds.

To see this in action, let's create a model of the population of rabbits in a post-apocolyptic Earth where fauna and
humans have moved to Mars on SpaceX ships, but a few bunnies were accidentally left behind to use Earth as an approximately
infinite food source. As you may recall from this model in elementary ecology courses, this is best defined by a linear
ODE system which we define in the ModelingToolkit sense:

```@example threshold_intervention
using EasyModelAnalysis
@variables t 🐰(t)
@parameters p
D = Differential(t)
eqs = [D(🐰) ~ p * 🐰]
@named sys = ODESystem(eqs)
prob = ODEProblem(sys, [🐰 => 0.01], (0.0, 10.0), [p => 1.0])
```

In this model `x(t)` is the number of billions of bunnies (commmonly referred in the units bB) and `t` is given in units of months. 
Solving this ODE shows the number of bunnies grow exponentially over time as they are all fat and happy with plenty of grass to feed on:

```@example threshold_intervention
plot(solve(prob))
```

We can ask the question, how long does it take for this exponential growth to cause the continuous number of bunnies `x` cross
the value of 50bB? Let's ask:

```@example threshold_intervention
get_threshold(prob, 🐰, 50)
```

Which, eyeballing the plot looks correct: in about 8 and half months the population will reach 50 billion bunnies! 

But now let's create an intervention. Let's say a corporation has created a "Earth viewing
tourism" business where humans are watch the bunny population through a telescope. Market research has found that when there are more
than 3bB in the population, the reaction goes from "that's cute" to "eww too many rodants", and thus to maximize their profits
the corporation wants to fire a laser that indescrimitely kills bunnies at an exponential rate (which just happens to be the same
rate as twice the growth of the bunny population). But as you probably know from experience, operating death rays that cover the distance 
from Mars to Earth cost a lot to operate, so the corporation wants to run this for as short as possible. The company knows they will
get shut down after 50 months anyways once regulators catch up, so they only need to keep the planet looking cute for that short amount
of time.

How should the bunny murder operation commence in order to successfully maximize corporate profits? To get the desired result
we use the `optimal_threshold_intervention` function. Our intervention will be to decrease the growth rate ``p = 1.0`` to
``p = -1.0`` (i.e. decrease it by twice the birth rate). What we want to know is what is the minimal time intervention to keep
the population under 3bB up until a time 50. Thus the call looks as follows:

```@example threshold_intervention
opt_tspan, (s1, s2, s3), ret = optimal_threshold_intervention(prob, [p => -1.0], 🐰, 3, 50)
```

The `opt_tspan` gives us the optimal timespan of the intervention:

```@example threshold_intervention
opt_tspan
```

We should begin the decimation of bunny civilization when it reaches 5.15 months, and then we can turn the destruction device off at
27.8 months into our tourism operation. To see the effect of this we can plot the results of the three sub-intervals:

```@example threshold_intervention
plot(s1, lab = "pre-intervention")
plot!(s2, lab = "intervention")
plot!(s3, xlims = (0, s3.t[end]), ylims = (0, 5), lab = "post-intervention", dpi = 300)
```

Let's understand this result. Because of the exponential growth, we want to start the laser as late as possible as that will give
us the most "bang for the buck", or rather, "burn for the buck", killing the most bunnies for the least amount of time. Thus we want
to wait for the population to grow a bit and be easier for the laser to hit. Then we leave the laser on for as short of a time as
possible. By choosing 27.8 months as the off point, this will allow for the rest of the time to be tourism friendly. When regulators 
finally get in touch on the 50th month, the population will have exactly reached the 3bB tipping point, allowing us to fully use
that entire time for unburned bunny tourism while also making regulators more sympathetic to our burnning cause by the time they
reach out to us.

Of course, this could also be used to uncover optimal policies for handling pandemics, but whatever your desired use case is,
churn away.