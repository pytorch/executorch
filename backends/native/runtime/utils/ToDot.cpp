// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/utils/ToDot.h>

#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/graph/StringFormat.h>
#include <executorch/backends/native/runtime/native_graph_generated.h>

// Everything below runs on a buffer that Program::load() has already put
// through flatbuffers::Verifier, so accessors return non-null wherever the
// schema declares the field required and wherever a union discriminator
// matches; the helpers here dereference those results directly. Fields the
// schema leaves optional are still checked, because verification says nothing
// about whether they are present.

namespace ptn {
namespace {

// Escape a dynamic string for embedding inside a DOT double-quoted label:
// backslash and double-quote get escaped; real newlines collapse to spaces so
// they cannot break the label (we insert line breaks ourselves as the literal
// two-character sequence "\n").
std::string esc(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (const char c : s) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
      case '\r':
        out += ' ';
        break;
      default:
        out += c;
    }
  }
  return out;
}

std::string str_of(const flatbuffers::String* s) {
  return s != nullptr ? s->str() : std::string();
}

bool nonempty(const flatbuffers::String* s) {
  return s != nullptr && s->size() > 0;
}

std::string dim_str(const fbs::Dim* d) {
  if (d->min() == d->max()) {
    return std::to_string(d->min());
  }
  return std::to_string(d->min()) + ".." +
      (d->max() < 0 ? std::string("inf") : std::to_string(d->max()));
}

std::string quant_suffix(const fbs::QuantSpec* q) {
  if (q == nullptr) {
    return "";
  }
  switch (q->scheme_type()) {
    case fbs::QuantScheme::AffineGroup: {
      const auto* a = q->scheme_as_AffineGroup();
      const int gs = a != nullptr ? a->group_size() : 0;
      return std::string(" q:affine g=") +
          (gs == 0 ? "perchan" : std::to_string(gs));
    }
    case fbs::QuantScheme::PackedQuant: {
      const auto* p = q->scheme_as_PackedQuant();
      return std::string(" q:") + (p != nullptr ? str_of(p->codec()) : "");
    }
    default:
      return "";
  }
}

// e.g. "FLOAT[16,16]" or "BYTE[8,16] q:affine g=32"
std::string meta_label(const fbs::TensorMeta* m) {
  if (m == nullptr) {
    return "";
  }
  std::string s = fbs::EnumNameScalarType(m->dtype());
  s += "[";
  const auto* sizes = m->sizes();
  if (sizes != nullptr) {
    for (flatbuffers::uoffset_t i = 0; i < sizes->size(); ++i) {
      if (i != 0) {
        s += ",";
      }
      s += dim_str(sizes->Get(i));
    }
  }
  s += "]";
  s += quant_suffix(m->quant());
  return s;
}

// Compact rendering of a non-tensor argument for a call node's label. Returns
// raw text (the caller esc()s it). Tensor/tensor-list args are drawn as edges,
// not here.
std::string arg_str(const fbs::Argument* a) {
  using AV = fbs::ArgumentValue;
  switch (a->value_type()) {
    case AV::NoneArg:
      return "None";
    case AV::TensorArg:
      return "%" + str_of(a->value_as_TensorArg()->name());
    case AV::IntArg: {
      const auto* x = a->value_as_IntArg();
      return nonempty(x->ref()) ? "%" + str_of(x->ref())
                                : std::to_string(x->value());
    }
    case AV::FloatArg: {
      const auto* x = a->value_as_FloatArg();
      return nonempty(x->ref()) ? "%" + str_of(x->ref())
                                : format_double(x->value());
    }
    case AV::BoolArg: {
      const auto* x = a->value_as_BoolArg();
      return nonempty(x->ref()) ? "%" + str_of(x->ref())
                                : (x->value() ? "true" : "false");
    }
    case AV::StringArg:
      return "\"" + str_of(a->value_as_StringArg()->value()) + "\"";
    case AV::ScalarTypeArg:
      return fbs::EnumNameScalarType(a->value_as_ScalarTypeArg()->value());
    case AV::IntListArg: {
      const auto* x = a->value_as_IntListArg();
      const auto* vals = x->values();
      const auto* refs = x->refs();
      std::string s = "[";
      if (vals != nullptr) {
        for (flatbuffers::uoffset_t i = 0; i < vals->size(); ++i) {
          if (i != 0) {
            s += ",";
          }
          if (refs != nullptr && i < refs->size() && nonempty(refs->Get(i))) {
            s += "%" + refs->Get(i)->str();
          } else {
            s += std::to_string(vals->Get(i));
          }
        }
      }
      return s + "]";
    }
    case AV::FloatListArg: {
      const auto* vals = a->value_as_FloatListArg()->values();
      std::string s = "[";
      if (vals != nullptr) {
        for (flatbuffers::uoffset_t i = 0; i < vals->size(); ++i) {
          if (i != 0) {
            s += ",";
          }
          s += format_double(vals->Get(i));
        }
      }
      return s + "]";
    }
    case AV::BoolListArg: {
      const auto* vals = a->value_as_BoolListArg()->values();
      std::string s = "[";
      if (vals != nullptr) {
        for (flatbuffers::uoffset_t i = 0; i < vals->size(); ++i) {
          if (i != 0) {
            s += ",";
          }
          s += vals->Get(i) ? "true" : "false";
        }
      }
      return s + "]";
    }
    case AV::TensorListArg: {
      const auto* names = a->value_as_TensorListArg()->names();
      std::string s = "[";
      if (names != nullptr) {
        for (flatbuffers::uoffset_t i = 0; i < names->size(); ++i) {
          if (i != 0) {
            s += ",";
          }
          s += "%" + names->Get(i)->str();
        }
      }
      return s + "]";
    }
    case AV::GraphArg:
      return "graph(" + str_of(a->value_as_GraphArg()->name()) + ")";
    default:
      return fbs::EnumNameArgumentValue(a->value_type());
  }
}

bool is_tensor_like(fbs::ArgumentValue t) {
  return t == fbs::ArgumentValue::TensorArg ||
      t == fbs::ArgumentValue::TensorListArg ||
      t == fbs::ArgumentValue::OptionalTensorListArg ||
      t == fbs::ArgumentValue::GraphArg;
}

// Method-level side tables (empty for HOP subgraphs, which carry none).
struct Ctx {
  std::unordered_map<std::string, const fbs::NamedTensorRef*> consts;
  std::unordered_map<std::string, const fbs::MutableBufferSpec*> muts;
  std::unordered_map<std::string, const fbs::OutputSpec*> ospecs;
};

// Draws the dataflow edges of one graph. `dangling` numbers the synthesized
// sources for values with no producer in this graph, so it persists across
// draw() calls. One emitter per graph body.
struct EdgeEmitter {
  std::string& out;
  const std::string& prefix;
  const std::unordered_map<std::string, const fbs::TensorMeta*>& tm;
  const std::unordered_map<std::string, std::string>& def;
  const Ctx& ctx;
  int dangling = 0;

  void draw(
      const std::string& vn,
      const std::string& to_id,
      const std::string& suffix,
      bool mutated,
      bool to_output);
};

void EdgeEmitter::draw(
    const std::string& vn,
    const std::string& to_id,
    const std::string& suffix,
    bool mutated,
    bool to_output) {
  std::string label = esc(vn);
  const auto mt = tm.find(vn);
  if (mt != tm.end()) {
    label += " " + meta_label(mt->second);
  }
  label += suffix;
  std::string attrs;
  if (mutated) {
    label += " (a!)";
    attrs += ", color=red, style=bold";
  }
  if (to_output) {
    const auto os = ctx.ospecs.find(vn);
    if (os != ctx.ospecs.end() &&
        os->second->kind() != fbs::OutputKind::USER_OUTPUT) {
      label += " [" + std::string(fbs::EnumNameOutputKind(os->second->kind())) +
          "->" + esc(str_of(os->second->target())) + "]";
    }
  }
  const auto it = def.find(vn);
  std::string from;
  if (it != def.end()) {
    from = it->second;
  } else {
    // No producer in this graph (malformed or lifted operand): synthesize a
    // visible source so the edge is not silently dropped.
    from = prefix + "_ext" + std::to_string(dangling++);
    out += "    " + from + " [shape=point, color=red];\n";
  }
  out += "    " + from + " -> " + to_id + " [label=\"" + label + "\"" + attrs +
      "];\n";
}

// Forward decl: graph emission recurses through HOP subgraphs.
void emit_graph(
    std::string& out,
    const std::string& prefix,
    const std::string& title,
    const fbs::Graph* g,
    const Ctx& ctx);

void placeholder_label(
    const std::string& name,
    const fbs::TensorMeta* meta,
    const Ctx& ctx,
    std::string& label,
    std::string& extra) {
  const auto ci = ctx.consts.find(name);
  if (ci != ctx.consts.end()) {
    const fbs::NamedTensorRef* c = ci->second;
    label = esc(name) + "\\n" + fbs::EnumNameInputKind(c->kind()) +
        (c->mutated() ? " (mut)" : "") + "\\n" + esc(str_of(c->data_key()));
    if (c->meta() != nullptr) {
      label += "\\n" + meta_label(c->meta());
    }
    extra = ", style=filled, fillcolor=\"#e6e6e6\"";
    return;
  }
  const auto mi = ctx.muts.find(name);
  if (mi != ctx.muts.end()) {
    label = esc(name) + "\\nMUTABLE_BUFFER\\n" + esc(str_of(mi->second->fqn()));
    if (meta != nullptr) {
      label += "\\n" + meta_label(meta);
    }
    extra = ", style=filled, fillcolor=\"#fff2cc\"";
    return;
  }
  label = esc(name) + "\\nUSER_INPUT";
  if (meta != nullptr) {
    label += "\\n" + meta_label(meta);
  }
}

void emit_node(
    std::string& out,
    const std::string& id,
    const fbs::Node* nd,
    const std::unordered_map<std::string, const fbs::TensorMeta*>& tm,
    const Ctx& ctx) {
  const std::string name = str_of(nd->name());
  std::string shape;
  std::string label;
  std::string extra;

  switch (nd->op_kind()) {
    case fbs::OpKind::PLACEHOLDER: {
      shape = "oval";
      const auto it = tm.find(name);
      placeholder_label(
          name, it != tm.end() ? it->second : nullptr, ctx, label, extra);
      break;
    }
    case fbs::OpKind::OUTPUT:
      shape = "doubleoctagon";
      label = name.empty() ? "output" : esc(name);
      break;
    default: {
      shape = "box";
      label = esc(name) + "\\n" + esc(str_of(nd->target()));
      const auto* ins = nd->inputs();
      if (ins != nullptr) {
        for (const fbs::NamedArgument* na : *ins) {
          if (is_tensor_like(na->arg()->value_type())) {
            continue; // drawn as an edge
          }
          label +=
              "\\n" + esc(str_of(na->name())) + "=" + esc(arg_str(na->arg()));
        }
      }
    }
  }
  out += "    " + id + " [shape=" + shape + ", label=\"" + label + "\"" +
      extra + "];\n";
}

void emit_graph(
    std::string& out,
    const std::string& prefix,
    const std::string& title,
    const fbs::Graph* g,
    const Ctx& ctx) {
  if (g == nullptr) {
    return;
  }

  std::unordered_map<std::string, const fbs::TensorMeta*> tm;
  if (const auto* tvs = g->tensor_values()) {
    for (const fbs::TensorValue* tv : *tvs) {
      tm[str_of(tv->name())] = tv->meta();
    }
  }

  // per-node ids + value name -> producing node id (def-use inversion)
  const auto* nodes = g->nodes();
  const flatbuffers::uoffset_t n = nodes != nullptr ? nodes->size() : 0;
  std::vector<std::string> ids(n);
  std::unordered_map<std::string, std::string> def;
  for (flatbuffers::uoffset_t j = 0; j < n; ++j) {
    ids[j] = prefix + "_" + std::to_string(j);
    const fbs::Node* nd = nodes->Get(j);
    if (const auto* outs = nd->outputs()) {
      for (const fbs::Output* o : *outs) {
        if (nonempty(o->name())) {
          def[o->name()->str()] = ids[j];
        }
        if (o->kind() == fbs::OutputValueKind::TENSOR_LIST) {
          if (const auto* nm = o->names()) {
            for (const flatbuffers::String* s : *nm) {
              def[s->str()] = ids[j];
            }
          }
        }
      }
    }
  }

  out += "  subgraph cluster_" + prefix + " {\n";
  out += "    label=\"" + esc(title) + "\";\n";
  out += "    style=rounded; color=gray;\n";

  for (flatbuffers::uoffset_t j = 0; j < n; ++j) {
    emit_node(out, ids[j], nodes->Get(j), tm, ctx);
  }

  EdgeEmitter edges{out, prefix, tm, def, ctx};

  for (flatbuffers::uoffset_t j = 0; j < n; ++j) {
    const fbs::Node* nd = nodes->Get(j);
    const bool to_output = nd->op_kind() == fbs::OpKind::OUTPUT;
    if (const auto* ins = nd->inputs()) {
      // Distinct cluster per GraphArg. Must stay outside the argument loop: a
      // node may carry several GraphArgs and each needs its own suffix.
      int sg_index = 0;
      for (const fbs::NamedArgument* na : *ins) {
        const fbs::Argument* a = na->arg();
        switch (a->value_type()) {
          case fbs::ArgumentValue::TensorArg:
            edges.draw(
                str_of(a->value_as_TensorArg()->name()),
                ids[j],
                "",
                na->mutated(),
                to_output);
            break;
          case fbs::ArgumentValue::TensorListArg: {
            const auto* nm = a->value_as_TensorListArg()->names();
            if (nm != nullptr) {
              for (flatbuffers::uoffset_t x = 0; x < nm->size(); ++x) {
                edges.draw(
                    nm->Get(x)->str(),
                    ids[j],
                    "[" + std::to_string(x) + "]",
                    na->mutated(),
                    to_output);
              }
            }
            break;
          }
          case fbs::ArgumentValue::OptionalTensorListArg: {
            const auto* oa = a->value_as_OptionalTensorListArg();
            const auto* nm = oa->names();
            const auto* hv = oa->has_value();
            if (nm != nullptr) {
              for (flatbuffers::uoffset_t x = 0; x < nm->size(); ++x) {
                if (hv != nullptr && x < hv->size() && hv->Get(x)) {
                  edges.draw(
                      nm->Get(x)->str(),
                      ids[j],
                      "[" + std::to_string(x) + "]?",
                      na->mutated(),
                      to_output);
                }
              }
            }
            break;
          }
          case fbs::ArgumentValue::GraphArg: {
            const fbs::GraphArg* ga = a->value_as_GraphArg();
            const std::string name = str_of(ga->name());
            const std::string sg = ids[j] + "_sg" + std::to_string(sg_index++);
            emit_graph(out, sg, "subgraph: " + name, ga->graph(), Ctx{});
            const auto* sgn =
                ga->graph() != nullptr ? ga->graph()->nodes() : nullptr;
            if (sgn != nullptr && sgn->size() > 0) {
              out += "    ";
              out += ids[j];
              out += " -> ";
              out += sg;
              out += "_0 [style=dashed, color=blue, label=\"";
              out += esc(name);
              out += "\", lhead=cluster_";
              out += sg;
              out += "];\n";
            }
            break;
          }
          default:
            break; // scalars shown in the node label
        }
      }
    }
    if (const auto* outs = nd->outputs()) {
      for (const fbs::Output* o : *outs) {
        if (nonempty(o->alias_of())) {
          const auto it = def.find(o->alias_of()->str());
          if (it != def.end()) {
            out += "    " + it->second + " -> " + ids[j] +
                " [style=dashed, color=\"#888888\", label=\"alias\"];\n";
          }
        }
      }
    }
  }

  out += "  }\n";
}

std::string render_program(const fbs::Program& program_fb) {
  std::string out;
  out += "digraph program {\n";
  out += "  compound=true;\n";
  out += "  rankdir=TB;\n";
  out += "  labelloc=\"t\";\n";
  out += "  node [fontname=\"monospace\", fontsize=10];\n";
  out += "  edge [fontname=\"monospace\", fontsize=9];\n";
  const std::string ver = str_of(program_fb.version());
  out += "  label=\"native_backend::Program" +
      (ver.empty() ? std::string() : "  version=" + esc(ver)) + "\";\n";

  if (const auto* methods = program_fb.methods()) {
    for (flatbuffers::uoffset_t i = 0; i < methods->size(); ++i) {
      const fbs::Method* m = methods->Get(i);
      Ctx ctx;
      if (const auto* cs = m->constants()) {
        for (const fbs::NamedTensorRef* c : *cs) {
          ctx.consts[str_of(c->name())] = c;
        }
      }
      if (const auto* mbs = m->mutable_buffers()) {
        for (const fbs::MutableBufferSpec* mb : *mbs) {
          ctx.muts[str_of(mb->name())] = mb;
        }
      }
      if (const auto* os = m->output_specs()) {
        for (const fbs::OutputSpec* o : *os) {
          ctx.ospecs[str_of(o->name())] = o;
        }
      }
      emit_graph(
          out,
          "m" + std::to_string(i),
          "method: " + str_of(m->name()),
          m->graph(),
          ctx);
    }
  }

  out += "}\n";
  return out;
}

} // namespace

// A debugging entry point, so it may have no caller in any checked-in code.
// cppcheck-suppress unusedFunction
std::string to_dot(const Program& program) {
  return render_program(*program.flatbuffer());
}

} // namespace ptn
