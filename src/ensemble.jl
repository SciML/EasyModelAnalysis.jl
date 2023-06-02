function ModelingToolkit.ODESystem(p::PropertyLabelledReactionNet; name = :MiraNet, kws...)
    t = first(@variables t)
    D = Differential(t)
    tm = TransitionMatrices(p)
    coefficients = tm.output - tm.input

    sname′(i) =
        if has_subpart(p, :sname)
            sname(p, i)
        else
            Symbol("S", i)
        end
    tname′(i) =
        if has_subpart(p, :tname)
            tname(p, i)
        else
            Symbol("r", i)
        end

    S = [first(@variables $Si(t)) for Si in sname′.(1:ns(p))]
    S_ = [first(@variables $Si) for Si in sname′.(1:ns(p))] # MathML doesn't know whether a Num should be dependent on t, so we use this to substitute 
    st_sub_map = S_ .=> S

    t_ps, flux_eqs = parse_tprops(p)
    s_ps = union(parse_prop_parameters.(sprops(p))...)
    ps = union(t_ps, s_ps)
    ps_sub_vars = [only(@variables $x) for x in Symbolics.getname.(ps)]
    ps_sub_map = ps_sub_vars .=> ps

    subs = [ps_sub_map; st_sub_map]
    subd = Dict(subs)

    flux_eqs = map(x -> substitute(x, subd), flux_eqs)
    tvars = ModelingToolkit.lhss(flux_eqs)

    default_p = ps .=> ModelingToolkit.getdefault.(ps)
    default_u0 = S .=> p[:concentration]
    defaults = Dict([default_p; default_u0])

    observable_species_idxs = filter(i -> sprop(p, i)["is_observable"], 1:ns(p))
    observable_species_names = Symbolics.getname.(S[observable_species_idxs])

    # i don't understand where p[:rate] comes into play. it seems like rate is only needed if there aren't custom rate laws
    deqs = [D(S[s]) ~ tvars' * coefficients[:, s]
            for s in 1:ns(p) if Symbolics.getname(S[s]) ∉ observable_species_names]

    # there should be a mathml but there isnt
    obs_eqs = [substitute(S[i] ~ Symbolics.parse_expr_to_symbolic(Meta.parse(sprop(p, i)["expression"]),
                                                                  @__MODULE__),
                          Dict(subs))
               for i in observable_species_idxs]

    eqs = Equation[flux_eqs; deqs; obs_eqs]
    ODESystem(eqs, t; name, defaults, kws...)
end

"can give it a S or T and itll make the pars from the prop dict"
function parse_prop_parameters(prop)
    pars = []
    !haskey(prop, "mira_parameters") && (return pars)

    mps = JSON3.read(prop["mira_parameters"])
    dists = JSON3.read(prop["mira_parameter_distributions"])
    for (k, v) in mps
        d = dists[k]
        pname = Symbol(k)
        if !isnothing(d) && haskey(d, "parameters")
            d_ps = d["parameters"]
            @assert d["type"] == "StandardUniform1"
            b = (d_ps["minimum"], d_ps["maximum"])

            par = only(@parameters $pname=v [bounds = b])
        else
            par = only(@parameters $pname = v)
        end
        push!(pars, par)
    end
    pars
end

function parse_tprops(p)
    mira_ps = Set()
    tps = tprops(p)
    tp = first(tps)
    flux_eqs = Equation[]
    for i in 1:nt(p)
        eq, pars = parse_tprop(p, i)
        push!(flux_eqs, eq)

        [push!(mira_ps, x) for x in pars] # no append!(set, xs)??
    end
    collect(mira_ps), flux_eqs
end

# doesn't really take a tprop, but the index
function parse_tprop(p, i)
    tp = tprop(p, i)
    tn = tname(p, i)

    pars = parse_prop_parameters(tp)

    t = only(@parameters t)
    tvar = only(@variables $tn(t))

    rl = MathML.parse_str(tp["mira_rate_law_mathml"])
    eq = tvar ~ rl
    eq, pars
end
