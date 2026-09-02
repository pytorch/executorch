// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/graph/utils/Print.h>

#include <cstddef>
#include <string>
#include <vector>

#include <executorch/backends/native/runtime/graph/StringFormat.h>

namespace ptn {

namespace {

std::string id_str(ValueId id) {
  return valid(id) ? "%" + std::to_string(id) : "None";
}

std::string id_list_str(const std::vector<ValueId>& ids) {
  std::string s = "[";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i) {
      s += ", ";
    }
    s += id_str(ids[i]);
  }
  return s + "]";
}

// Like id_list_str, but every entry is expected to resolve: a graph's I/O list
// has no absent slots, so kInvalid there is corruption worth showing as "%-1".
std::string join_ids(const std::vector<ValueId>& ids) {
  std::string s = "[";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i) {
      s += ", ";
    }
    s += "%" + std::to_string(ids[i]);
  }
  return s + "]";
}

std::string output_str(const Output& out) {
  if (out.kind == OutputValueKind::TensorList) {
    return id_list_str(out.elem_ids);
  }
  return id_str(out.value_id);
}

} // namespace

std::string to_string(const TensorMeta& meta) {
  std::string s = scalar_type_name(meta.dtype);
  s += "[";
  for (size_t i = 0; i < meta.sizes.size(); ++i) {
    if (i != 0) {
      s += ",";
    }
    s += std::to_string(meta.sizes[i]);
  }
  s += "]";
  return s;
}

std::string to_string(const Scalar& scalar) {
  if (scalar.is_bool()) {
    return scalar.to_bool() ? "true" : "false";
  }
  if (scalar.is_int()) {
    return std::to_string(scalar.to_int());
  }
  return format_double(scalar.to_double());
}

std::string to_string(const Argument& arg) {
  switch (arg.kind()) {
    case ArgKind::None:
      return "None";
    case ArgKind::Tensor:
      return id_str(arg.as_tensor().id);
    case ArgKind::Int: {
      const IntArg& a = arg.as_int();
      return valid(a.id) ? id_str(a.id) : std::to_string(a.value);
    }
    case ArgKind::Float: {
      const FloatArg& a = arg.as_float();
      return valid(a.id) ? id_str(a.id) : format_double(a.value);
    }
    case ArgKind::Bool: {
      const BoolArg& a = arg.as_bool();
      return valid(a.id) ? id_str(a.id) : (a.value ? "true" : "false");
    }
    case ArgKind::String:
      return "\"" + arg.as_string().value + "\"";
    case ArgKind::ScalarType:
      return scalar_type_name(arg.as_scalar_type().value);
    case ArgKind::IntList: {
      const IntListArg& a = arg.as_int_list();
      std::string s = "[";
      for (size_t i = 0; i < a.values.size(); ++i) {
        if (i) {
          s += ", ";
        }
        const bool sym = i < a.ids.size() && valid(a.ids[i]);
        s += sym ? id_str(a.ids[i]) : std::to_string(a.values[i]);
      }
      return s + "]";
    }
    case ArgKind::FloatList: {
      const FloatListArg& a = arg.as_float_list();
      std::string s = "[";
      for (size_t i = 0; i < a.values.size(); ++i) {
        if (i) {
          s += ", ";
        }
        s += format_double(a.values[i]);
      }
      return s + "]";
    }
    case ArgKind::BoolList: {
      const BoolListArg& a = arg.as_bool_list();
      std::string s = "[";
      for (size_t i = 0; i < a.values.size(); ++i) {
        if (i) {
          s += ", ";
        }
        s += a.values[i] ? "true" : "false";
      }
      return s + "]";
    }
    case ArgKind::TensorList:
      return id_list_str(arg.as_tensor_list().ids);
    case ArgKind::OptionalTensorList:
      return id_list_str(arg.as_optional_tensor_list().ids);
    case ArgKind::Graph:
      return "graph(" + arg.as_graph().name + ")";
  }
  return "?";
}

std::string to_string(const Node& node) {
  std::string s = node.name.empty() ? "_" : node.name;
  s += " = ";
  switch (node.op_kind) {
    case OpKind::CallFunction:
      s += node.target;
      break;
    case OpKind::Placeholder:
      s += "<placeholder>";
      break;
    case OpKind::Output:
      s += "<output>";
      break;
  }
  s += "(";
  for (size_t i = 0; i < node.inputs.size(); ++i) {
    if (i) {
      s += ", ";
    }
    const NamedArgument& named = node.inputs[i];
    if (!named.name.empty()) {
      s += named.name + "=";
    }
    s += to_string(named.arg);
    if (named.mutated) {
      s += "!";
    }
  }
  s += ")";
  if (!node.outputs.empty()) {
    s += " -> ";
    for (size_t i = 0; i < node.outputs.size(); ++i) {
      if (i) {
        s += ", ";
      }
      s += output_str(node.outputs[i]);
    }
  }
  return s;
}

std::string to_string(const Graph& graph) {
  std::string s = "inputs: " + join_ids(graph.input_ids) + "\n";
  if (!graph.schedule.empty()) {
    for (NodeId id : graph.schedule) {
      s += "  " + to_string(graph.node(id)) + "\n";
    }
  } else {
    for (const Node& n : graph.nodes) {
      s += "  " + to_string(n) + "\n";
    }
  }
  s += "outputs: " + join_ids(graph.output_ids) + "\n";
  if (!graph.subgraphs.empty()) {
    s += "(" + std::to_string(graph.subgraphs.size()) + " subgraphs)\n";
  }
  return s;
}

} // namespace ptn
