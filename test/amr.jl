
using AlgebraicPetri
using Catlab.CategoricalAlgebra

import JSON

struct ASKEMPetriNet
    petri::PropertyLabelledPetriNet
    json::AbstractDict
end

function to_petri(file)
    original_json = JSON.parsefile(file)
    model = original_json["model"]
    state_props = Dict(Symbol(s["id"]) => s for s in model["states"])
    states = [Symbol(s["id"]) for s in model["states"]]
    transition_props = Dict(Symbol(t["id"]) => t for t in model["transitions"])
    transitions = [Symbol(t["id"]) => (Symbol.(t["input"]) => Symbol.(t["output"]))
                   for t in model["transitions"]]

    petri = LabelledPetriNet(states, transitions...)
    ASKEMPetriNet(PropertyLabelledPetriNet{Dict}(petri, state_props, transition_props),
                  original_json)
end

function update_properties!(pn)
    # TODO: Add support for stratified models with tuples of attributes -> arrays of properties
    map(parts(pn, :S)) do s
        pn[s, :sprop]["id"] = String(pn[s, :sname])
    end
    map(parts(pn, :T)) do t
        props = pn[t, :tprop]
        props["id"] = String(pn[t, :tname])
        new_inputs = String.(pn[pn[incident(pn, t, :it), :is], :sname])
        for (i, input) in enumerate(props["input"])
            props["input"][i] = new_inputs[i]
        end
        new_outputs = String.(pn[pn[incident(pn, t, :ot), :os], :sname])
        for (o, output) in enumerate(props["output"])
            props["output"][o] = new_outputs[o]
        end
    end
    pn
end

update_properties!(askem_net::ASKEMPetriNet) = update_properties!(askem_net.petri)

function update_json!(askem_net::ASKEMPetriNet)
    pn = askem_net.petri
    askem_net.json["model"]["states"] = map(s -> pn[s, :sprop], parts(pn, :S))
    askem_net.json["model"]["transitions"] = map(t -> pn[t, :tprop], parts(pn, :T))
    askem_net.json
end

function update!(askem_net::ASKEMPetriNet)
    update_properties!(askem_net)
    update_json!(askem_net)
end

# JSON Interoperability
#######################

JSON.json(askem_net::ASKEMPetriNet) = JSON.json(askem_net.json)
JSON.print(io::IO, askem_net::ASKEMPetriNet) = JSON.print(io, askem_net.json)
function JSON.print(io::IO, askem_net::ASKEMPetriNet, indent)
    JSON.print(io, askem_net.json, indent)
end
# Load a model
file = "/Users/anand/code/julia/tmp/EasyModelAnalysis.jl/data/amr_sir.json"
askemnet = to_petri(file)

pn = askemnet.petri
ODESystem(pn)

@parameters t r=-1
D = Differential(t)
@variables x(t) = r
@named foo = ODESystem([D(x) ~ x], t, [x], [r])
solve(ODEProblem(foo, [], tspan))

j = original_json
ode = j["semantics"]["ode"]
inits, params, rates = ode["initials"], ode["parameters"], ode["rates"]

init = inits[1]
s = init["target"]
su0 = MathML.parse_str(init["expression_mathml"])
@assert length(Symbolics.get_variables(su0)) == 1

params
param = params[1]
function parse_param(param)
    pid = Symbol(param["id"])
    only(@parameters $pid=param["value"] [description = param["description"]])
end
sym_ps = parse_param.(params)
pd = Dict(Symbol(p["id"]) => p["value"] for p in params)

# if the u0 expression is anything more than just a plain parameter there'll be some issues

s_sts_syms = [Symbol(s["id"]) for s in j["model"]["states"]]
all_sts 