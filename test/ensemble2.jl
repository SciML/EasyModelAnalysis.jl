using Catlab, AlgebraicPetri, ModelingToolkit, DifferentialEquations, UnPack, SciMLBase,
      Distributions, Symbolics, DiffEqBase, Plots, EasyModelAnalysis
using DifferentialEquations.EnsembleAnalysis
using Catlab.CategoricalAlgebra
using OpenAIReplMode
using JSON, JSONSchema

@info "usings"
include("Mira.jl")

dir = "/Users/anand/code/python/mira/notebooks/ensemble"
fns  =filter(endswith("json"), readdir(dir;join=true))
# all_fns = readdir("/Users/anand/.julia/dev/ASKEM_Evaluation_Staging/docs/src/Scenario3";
#                   join = true)
fns2 = filter(endswith("json"), all_fns)
sir_fn = fns2[1]

prop_rn_sch = read_json_acset_schema("/Users/anand/code/python/algj/py-acsets/src/acsets/schemas/catlab/PropertyReactionNet.json")
write_json_acset_schema(prop_rn_sch, "prop_rn_sch.json")
sch2 = read_json_acset_schema("prop_rn_sch.json")

sch_fn = "/Users/anand/code/python/algj/py-acsets/src/acsets/schemas/catlab/PropertyLabelledReactionNet.json"
sch_fn = "/Users/anand/code/python/algj/py-acsets/src/acsets/schemas/catlab/PropertyLabelledReactionNet.json"
# sch_fn = "/Users/anand/code/python/algj/py-acsets/src/acsets/schemas/catlab/PropertyReactionNet.json"
fn = "/Users/anand/code/python/mira/notebooks/ensemble/BIOMD0000000955_miranet.json"
SchPRN = read_json_acset_schema(sch_fn)
@acset_type LPRN2(SchPRN)
read_json_acset(LPRN2{Any,Any, Any, Any}, fn)


l = PLRN{Any, Any, Any, Any}()
l = LPRN{Any, Any, Any}()
read_json_acset(SchPRN, fn)

read_json_acset(LPRN, fn)

py_pn_sch_fn = "/Users/anand/code/python/py-acsets/tests/petri_schema.json"
py_pn_sch_fn = "/Users/anand/code/python/py-acsets/tests/petri_schema2.json"
# https://github.com/AlgebraicJulia/py-acsets/blob/main/tests/petri_schema.json
# this one with keys changed
jpn = JSON.parse(read(py_pn_sch_fn, String))
keys(jpn)

py_pn_sch_fn = "petri_schema.json"
py_mira_sch_fn = "petri_schema.json"

py_mira_sch_fn = "/Users/anand/code/python/py-acsets/src/acsets/schemas/miranet_schema.json"
py_mira_sch_fn = "/Users/anand/code/python/py-acsets/src/acsets/schemas/miranet_schema2.json"
pet_sch = read_json_acset_schema(py_mira_sch_fn)
# Debugger.@enter read_json_acset_schema(py_pn_sch_fn)
acset_schema_json_schema()
Mira.load_mira.(fns)
Mira.load_mira_curated.(fns)

read_json_acset_schema()


keys(generate_json_acset_schema(SchLabelledPetriNet))
sch2 = read_json_acset_schema("sch_lpn.json")
sch2
@acset_type MyLPN(sch2, index=[:label])

sir = read_json_acset(MyLPN{String}, sir_fn)

acset_schema_json_schema()
json_schema = JSONSchema.Schema()
JSON.print(json_schema, 2)

@present SchDDS(FreeSchema) begin
  X::Ob
  next::Hom(X,X)
end
@acset_type DDS(SchDDS, index=[:next])
DDS()
JSON.print(generate_json_acset_schema(SchDDS), 2)
@present SchLabeledDDS <: SchDDS begin
  Label::AttrType
  label::Attr(X, Label)
end

JSON.print(generate_json_acset_schema(SchLabeledDDS), 2)

@acset_type LabeledDDS(SchLabeledDDS, index=[:next, :label])
LabeledDDS}()
ldds = LabeledDDS{Int}()
add_parts!(ldds, :X, 4, next=[2,3,4,1], label=[100, 101, 102, 103])
JSON.print(generate_json_acset(ldds),2)