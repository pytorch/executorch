// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// The deserializer bridge: native_backend::Program (FlatBuffer) -> ptn
// in-memory IR (Method / Graph / Node / Argument / Value). In-graph references
// (SSA names) are resolved to list ValueIds; per-graph namespaces are
// resolved independently (each HOP subgraph rebuilds its own name -> id map).
//
// Everything here runs on a buffer Program::load() has already put through
// flatbuffers::Verifier, so accessors return non-null wherever the schema
// declares the field required and wherever a union discriminator matches; the
// walkers below dereference those results directly. Fields the schema leaves
// optional are still checked, because verification says nothing about whether
// they are present.

#include <executorch/backends/native/runtime/Program.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>

#include <executorch/backends/native/runtime/native_graph_generated.h>

namespace ptn {
namespace {

std::string str_of(const flatbuffers::String* s) {
  return s != nullptr ? s->str() : std::string();
}

bool nonempty(const flatbuffers::String* s) {
  return s != nullptr && s->size() > 0;
}

ScalarType map_scalar_type(fbs::ScalarType t) {
  // ptn::ScalarType ids are pinned to the schema's, so the byte maps straight.
  return static_cast<ScalarType>(static_cast<int8_t>(t));
}

OpKind map_op_kind(fbs::OpKind k) {
  switch (k) {
    case fbs::OpKind::PLACEHOLDER:
      return OpKind::Placeholder;
    case fbs::OpKind::OUTPUT:
      return OpKind::Output;
    default:
      return OpKind::CallFunction;
  }
}

OutputValueKind map_output_value_kind(fbs::OutputValueKind k) {
  switch (k) {
    case fbs::OutputValueKind::TENSOR_LIST:
      return OutputValueKind::TensorList;
    case fbs::OutputValueKind::INT:
      return OutputValueKind::Int;
    case fbs::OutputValueKind::BOOL:
      return OutputValueKind::Bool;
    case fbs::OutputValueKind::FLOAT:
      return OutputValueKind::Float;
    default:
      return OutputValueKind::Tensor;
  }
}

ValueRole map_input_kind(fbs::InputKind k) {
  switch (k) {
    case fbs::InputKind::PARAMETER:
      return ValueRole::Parameter;
    case fbs::InputKind::BUFFER:
      return ValueRole::Buffer;
    case fbs::InputKind::CONSTANT_TENSOR:
      return ValueRole::ConstantTensor;
    default:
      return ValueRole::UserInput;
  }
}

OutputKind map_output_kind(fbs::OutputKind k) {
  switch (k) {
    case fbs::OutputKind::BUFFER_MUTATION:
      return OutputKind::BufferMutation;
    case fbs::OutputKind::USER_INPUT_MUTATION:
      return OutputKind::UserInputMutation;
    default:
      return OutputKind::UserOutput;
  }
}

// The wire describes a dim as a min..max range, while the IR holds a concrete
// extent. Collapsing a range to its upper bound would run the graph at that
// bound and compute over elements the caller never supplied, so a dim that is
// not a single non-negative extent is refused where it enters the IR.
int64_t static_extent(
    const fbs::Dim* d,
    const std::string& value_name,
    flatbuffers::uoffset_t i) {
  if (d->min() == d->max() && d->min() >= 0) {
    return d->min();
  }
  throw std::runtime_error(
      "build_tensor_meta: " + value_name + " dim " + std::to_string(i) +
      " is not a static extent (" + std::to_string(d->min()) + ".." +
      (d->max() < 0 ? std::string("inf") : std::to_string(d->max())) +
      "); this runtime requires static shapes");
}

TensorMeta build_tensor_meta(
    const fbs::TensorMeta* m,
    const std::string& name) {
  TensorMeta out;
  if (m == nullptr) {
    return out;
  }
  out.dtype = map_scalar_type(m->dtype());
  if (const auto* sizes = m->sizes()) {
    out.sizes.reserve(sizes->size());
    for (flatbuffers::uoffset_t i = 0; i < sizes->size(); ++i) {
      out.sizes.push_back(static_extent(sizes->Get(i), name, i));
    }
  }
  if (const auto* dord = m->dim_order()) {
    out.dim_order_hint.reserve(dord->size());
    for (flatbuffers::uoffset_t i = 0; i < dord->size(); ++i) {
      out.dim_order_hint.push_back(static_cast<int32_t>(dord->Get(i)));
    }
  }
  return out;
}

// value name -> tensor metadata.
using MetaTable = std::unordered_map<std::string, const fbs::TensorMeta*>;

// One graph body plus the SSA-name -> value-id map used to build it.
//
// The wire format addresses values two ways: a graph body is positional, and
// the value list keeps that (a ValueId is an index), while the method-level
// side tables -- constants, mutable_buffers, output_specs -- name their targets
// by SSA string, since the serializer writes them independently of the body.
// Only the builder knows how one maps to the other, so it hands the map back
// for build_method to resolve those names against.
//
// Build-time scaffolding: nothing outside build_method sees it, and neither
// Method nor Graph stores it. Once the bindings are resolved to ids it is
// discarded, and the graph is index-addressed from then on.
struct BuiltGraph {
  Graph graph;
  std::unordered_map<std::string, ValueId> name_to_id;
};

// `extra_meta` supplies metadata for values the graph's own tensor_values side
// table omits: a constant placeholder's meta rides on the Method's
// NamedTensorRef binding instead, so build_method passes it in to type those
// values on creation. Subgraphs have no such bindings.
BuiltGraph build_graph(const fbs::Graph* g, const MetaTable& extra_meta = {});

// Builds one Graph body. The graph under construction, its SSA-name -> id map,
// and the name -> metadata table are shared by every step of the build, so they
// are members rather than threaded through each call. One builder per body: a
// subgraph gets a fresh one, which is what gives each body its own independent
// SSA namespace.
class GraphBuilder {
 public:
  BuiltGraph run(const fbs::Graph* g, const MetaTable& extra_meta);

 private:
  // Resolve a name to its ValueId, creating the Value on first mention.
  // A name in the meta side table becomes a Tensor value; otherwise a None
  // value (scalar / symbolic outputs, refined once the loader models them).
  ValueId id_of(const std::string& name);

  Argument convert_arg(const fbs::Argument* a);

  Graph graph_;
  std::unordered_map<std::string, ValueId> n2i_;
  MetaTable tm_;
};

ValueId GraphBuilder::id_of(const std::string& name) {
  if (name.empty()) {
    return kInvalid;
  }
  const auto it = n2i_.find(name);
  if (it != n2i_.end()) {
    return it->second;
  }
  const ValueId id = static_cast<ValueId>(graph_.values.size());
  const auto mit = tm_.find(name);
  if (mit != tm_.end() && mit->second != nullptr) {
    graph_.values.emplace_back(name, build_tensor_meta(mit->second, name));
  } else {
    graph_.values.emplace_back(name);
  }
  n2i_[name] = id;
  return id;
}

Argument GraphBuilder::convert_arg(const fbs::Argument* a) {
  using AV = fbs::ArgumentValue;
  switch (a->value_type()) {
    case AV::TensorArg: {
      TensorArg t;
      t.id = id_of(str_of(a->value_as_TensorArg()->name()));
      return t;
    }
    case AV::IntArg: {
      const auto* x = a->value_as_IntArg();
      IntArg r;
      r.value = x->value();
      r.id = nonempty(x->ref()) ? id_of(x->ref()->str()) : kInvalid;
      return r;
    }
    case AV::FloatArg: {
      const auto* x = a->value_as_FloatArg();
      FloatArg r;
      r.value = x->value();
      r.id = nonempty(x->ref()) ? id_of(x->ref()->str()) : kInvalid;
      return r;
    }
    case AV::BoolArg: {
      const auto* x = a->value_as_BoolArg();
      BoolArg r;
      r.value = x->value();
      r.id = nonempty(x->ref()) ? id_of(x->ref()->str()) : kInvalid;
      return r;
    }
    case AV::StringArg: {
      StringArg r;
      r.value = str_of(a->value_as_StringArg()->value());
      return r;
    }
    case AV::ScalarTypeArg: {
      ScalarTypeArg r;
      r.value = map_scalar_type(a->value_as_ScalarTypeArg()->value());
      return r;
    }
    case AV::IntListArg: {
      const auto* x = a->value_as_IntListArg();
      IntListArg r;
      if (const auto* vals = x->values()) {
        for (flatbuffers::uoffset_t i = 0; i < vals->size(); ++i) {
          r.values.push_back(vals->Get(i));
        }
      }
      if (const auto* refs = x->refs()) {
        for (flatbuffers::uoffset_t i = 0; i < refs->size(); ++i) {
          r.ids.push_back(
              nonempty(refs->Get(i)) ? id_of(refs->Get(i)->str()) : kInvalid);
        }
      }
      return r;
    }
    case AV::FloatListArg: {
      FloatListArg r;
      if (const auto* vals = a->value_as_FloatListArg()->values()) {
        for (flatbuffers::uoffset_t i = 0; i < vals->size(); ++i) {
          r.values.push_back(vals->Get(i));
        }
      }
      return r;
    }
    case AV::BoolListArg: {
      BoolListArg r;
      if (const auto* vals = a->value_as_BoolListArg()->values()) {
        for (flatbuffers::uoffset_t i = 0; i < vals->size(); ++i) {
          r.values.push_back(vals->Get(i));
        }
      }
      return r;
    }
    case AV::TensorListArg: {
      TensorListArg r;
      if (const auto* nm = a->value_as_TensorListArg()->names()) {
        for (flatbuffers::uoffset_t i = 0; i < nm->size(); ++i) {
          r.ids.push_back(id_of(nm->Get(i)->str()));
        }
      }
      return r;
    }
    case AV::OptionalTensorListArg: {
      const auto* oa = a->value_as_OptionalTensorListArg();
      const auto* nm = oa->names();
      const auto* hv = oa->has_value();
      OptionalTensorListArg r;
      if (nm != nullptr) {
        for (flatbuffers::uoffset_t i = 0; i < nm->size(); ++i) {
          const bool present = hv != nullptr && i < hv->size() && hv->Get(i);
          r.ids.push_back(present ? id_of(nm->Get(i)->str()) : kInvalid);
        }
      }
      return r;
    }
    case AV::GraphArg: {
      const fbs::GraphArg* ga = a->value_as_GraphArg();
      GraphArg r;
      r.name = str_of(ga->name());
      r.subgraph_id = static_cast<GraphId>(graph_.subgraphs.size());
      graph_.subgraphs.push_back(build_graph(ga->graph()).graph);
      return r;
    }
    default:
      // NoneArg, and any tag this build does not know.
      return NoneArg{};
  }
}

BuiltGraph GraphBuilder::run(const fbs::Graph* g, const MetaTable& extra_meta) {
  if (g == nullptr) {
    return {};
  }

  if (const auto* tvs = g->tensor_values()) {
    for (flatbuffers::uoffset_t i = 0; i < tvs->size(); ++i) {
      const fbs::TensorValue* tv = tvs->Get(i);
      tm_[str_of(tv->name())] = tv->meta();
    }
  }
  for (const auto& entry : extra_meta) {
    const fbs::TensorMeta*& slot = tm_[entry.first];
    if (slot == nullptr) {
      slot = entry.second;
    }
  }

  // Pre-create meta-carrying values in a deterministic order (nicer ids).
  if (const auto* tvs = g->tensor_values()) {
    for (flatbuffers::uoffset_t i = 0; i < tvs->size(); ++i) {
      id_of(str_of(tvs->Get(i)->name()));
    }
  }

  if (const auto* nodes = g->nodes()) {
    for (flatbuffers::uoffset_t i = 0; i < nodes->size(); ++i) {
      const fbs::Node* nd = nodes->Get(i);
      Node node;
      node.name = str_of(nd->name());
      node.op_kind = map_op_kind(nd->op_kind());
      node.target = str_of(nd->target());

      if (const auto* ins = nd->inputs()) {
        for (flatbuffers::uoffset_t j = 0; j < ins->size(); ++j) {
          const fbs::NamedArgument* na = ins->Get(j);
          NamedArgument narg;
          narg.name = str_of(na->name());
          narg.mutated = na->mutated();
          narg.arg = convert_arg(na->arg());
          node.inputs.push_back(std::move(narg));
        }
      }

      if (const auto* outs = nd->outputs()) {
        for (flatbuffers::uoffset_t j = 0; j < outs->size(); ++j) {
          const fbs::Output* o = outs->Get(j);
          Output out;
          out.kind = map_output_value_kind(o->kind());
          if (o->kind() == fbs::OutputValueKind::TENSOR_LIST) {
            if (const auto* nm = o->names()) {
              for (flatbuffers::uoffset_t k = 0; k < nm->size(); ++k) {
                out.elem_ids.push_back(id_of(nm->Get(k)->str()));
              }
            }
          } else {
            out.value_id = id_of(str_of(o->name()));
            if (nonempty(o->alias_of()) && valid(out.value_id)) {
              graph_.values[out.value_id].alias_id =
                  id_of(o->alias_of()->str());
            }
          }
          node.outputs.push_back(std::move(out));
        }
      }

      // A placeholder with no explicit Output still produces its named value;
      // synthesize one so def-use wiring records the placeholder as producer.
      if (node.op_kind == OpKind::Placeholder && node.outputs.empty() &&
          !node.name.empty()) {
        Output out;
        out.value_id = id_of(node.name);
        node.outputs.push_back(out);
      }

      graph_.nodes.push_back(std::move(node));
    }
  }

  if (const auto* gi = g->inputs()) {
    for (flatbuffers::uoffset_t i = 0; i < gi->size(); ++i) {
      graph_.input_ids.push_back(id_of(gi->Get(i)->str()));
    }
  }
  if (const auto* go = g->outputs()) {
    for (flatbuffers::uoffset_t i = 0; i < go->size(); ++i) {
      graph_.output_ids.push_back(id_of(go->Get(i)->str()));
    }
  }

  graph_.initialize_schedule();
  graph_.rebuild_def_use();
  return BuiltGraph{std::move(graph_), std::move(n2i_)};
}

BuiltGraph build_graph(const fbs::Graph* g, const MetaTable& extra_meta) {
  return GraphBuilder().run(g, extra_meta);
}

// Resolve a method-level binding name (namespace 2) against the top-level
// graph's SSA names, or kInvalid if the graph holds no such value.
ValueId id_of_name(
    const std::unordered_map<std::string, ValueId>& n2i,
    const std::string& name) {
  const auto it = n2i.find(name);
  return it != n2i.end() ? it->second : kInvalid;
}

// Record a binding's role on the value it binds. A binding naming a value the
// graph does not contain is ignored.
void stamp_role(Graph& graph, ValueId id, ValueRole role) {
  if (in_bounds(id, graph.values.size())) {
    graph.values[id].role = role;
  }
}

} // namespace

Method Program::build_method(size_t index) const {
  if (program_fb_ == nullptr) {
    throw std::runtime_error("build_method: program is not loaded");
  }
  const auto* methods = program_fb_->methods();
  if (methods == nullptr || index >= methods->size()) {
    throw std::runtime_error("build_method: method index out of range");
  }
  const fbs::Method* m =
      methods->Get(static_cast<flatbuffers::uoffset_t>(index));

  Method method;
  method.name = str_of(m->name());

  MetaTable constant_meta;
  if (const auto* cs = m->constants()) {
    for (flatbuffers::uoffset_t i = 0; i < cs->size(); ++i) {
      const fbs::NamedTensorRef* c = cs->Get(i);
      constant_meta[str_of(c->name())] = c->meta();
    }
  }

  BuiltGraph built = build_graph(m->graph(), constant_meta);
  const std::unordered_map<std::string, ValueId>& n2i = built.name_to_id;
  method.graph = std::move(built.graph);
  Graph& graph = method.graph;

  // external-constant / buffer identity (key) -> value, for BufferMutation
  // output targets (a namespace-3 fqn, not an SSA name).
  std::unordered_map<std::string, ValueId> key_to_id;

  if (const auto* cs = m->constants()) {
    for (flatbuffers::uoffset_t i = 0; i < cs->size(); ++i) {
      const fbs::NamedTensorRef* c = cs->Get(i);
      DataBinding b;
      b.value_id = id_of_name(n2i, str_of(c->name()));
      b.role = map_input_kind(c->kind());
      b.key = str_of(c->data_key());
      b.has_data = true;
      b.mutated = c->mutated();
      stamp_role(graph, b.value_id, b.role);
      if (!b.key.empty()) {
        key_to_id[b.key] = b.value_id;
      }
      method.data_bindings.push_back(std::move(b));
    }
  }

  if (const auto* mbs = m->mutable_buffers()) {
    for (flatbuffers::uoffset_t i = 0; i < mbs->size(); ++i) {
      const fbs::MutableBufferSpec* mb = mbs->Get(i);
      DataBinding b;
      b.value_id = id_of_name(n2i, str_of(mb->name()));
      b.role = ValueRole::Buffer;
      b.key = str_of(mb->fqn());
      b.has_data = false;
      b.mutated = true;
      stamp_role(graph, b.value_id, ValueRole::Buffer);
      if (!b.key.empty()) {
        key_to_id[b.key] = b.value_id;
      }
      method.data_bindings.push_back(std::move(b));
    }
  }

  // Top-level graph inputs not otherwise bound are user inputs.
  for (const ValueId id : graph.input_ids) {
    if (in_bounds(id, graph.values.size()) &&
        graph.values[id].role == ValueRole::Intermediate) {
      graph.values[id].role = ValueRole::UserInput;
    }
  }

  // output_specs are parallel to graph.outputs (same order); each classifies
  // graph.output_ids[i]. The mutation target resolves to a placeholder value:
  // an fqn (BufferMutation) via key_to_id, else an SSA name
  // (UserInputMutation).
  if (const auto* os = m->output_specs()) {
    for (flatbuffers::uoffset_t i = 0; i < os->size(); ++i) {
      const fbs::OutputSpec* o = os->Get(i);
      OutputSpec spec;
      spec.kind = map_output_kind(o->kind());
      const std::string target = str_of(o->target());
      if (!target.empty()) {
        if (spec.kind == OutputKind::BufferMutation) {
          auto it = key_to_id.find(target);
          spec.target_id = it != key_to_id.end() ? it->second : kInvalid;
        } else {
          spec.target_id = id_of_name(n2i, target);
        }
      }
      method.output_specs.push_back(spec);
    }
  }

  return method;
}

// Defined here (rather than in Program.cpp) so it sits next to build_method and
// the deserializer helpers it drives: get_method is the public lazy entry
// point, build_method the private materializer it calls on a cache miss.
const Method& Program::get_method(const std::string& name) const {
  const auto it = method_cache_.find(name);
  if (it != method_cache_.end()) {
    return it->second;
  }
  if (program_fb_ != nullptr) {
    if (const auto* methods = program_fb_->methods()) {
      for (flatbuffers::uoffset_t i = 0; i < methods->size(); ++i) {
        if (str_of(methods->Get(i)->name()) == name) {
          auto res = method_cache_.emplace(name, build_method(i));
          return res.first->second;
        }
      }
    }
  }
  throw std::runtime_error(
      "Program::get_method: no method named '" + name + "'");
}

} // namespace ptn
