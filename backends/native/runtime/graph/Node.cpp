// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/graph/Node.h>

#include <string>

#include <executorch/backends/native/runtime/graph/Format.h>

namespace ptn {

namespace {

std::string ref_str(ValueRef ref) {
  return valid(ref) ? "%" + std::to_string(ref) : "None";
}

// Compact one-line rendering of an argument for Node::to_string. Symbolic
// scalars (a valid ref) render as the ref; literals render as their value.
std::string arg_str(const Argument& arg) {
  switch (arg.kind()) {
    case ArgKind::None:
      return "None";
    case ArgKind::Tensor:
      return ref_str(arg.as_tensor().ref);
    case ArgKind::Int: {
      const IntArg& a = arg.as_int();
      return valid(a.ref) ? ref_str(a.ref) : std::to_string(a.value);
    }
    case ArgKind::Float: {
      const FloatArg& a = arg.as_float();
      return valid(a.ref) ? ref_str(a.ref) : format_double(a.value);
    }
    case ArgKind::Bool: {
      const BoolArg& a = arg.as_bool();
      return valid(a.ref) ? ref_str(a.ref) : (a.value ? "true" : "false");
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
        const bool sym = i < a.refs.size() && valid(a.refs[i]);
        s += sym ? ref_str(a.refs[i]) : std::to_string(a.values[i]);
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
    case ArgKind::OptionalTensorList: {
      const std::vector<ValueRef>& refs = arg.kind() == ArgKind::TensorList
          ? arg.as_tensor_list().refs
          : arg.as_optional_tensor_list().refs;
      std::string s = "[";
      for (size_t i = 0; i < refs.size(); ++i) {
        if (i) {
          s += ", ";
        }
        s += ref_str(refs[i]);
      }
      return s + "]";
    }
    case ArgKind::Graph:
      return "graph(" + arg.as_graph().name + ")";
  }
  return "?";
}

std::string output_str(const Output& out) {
  if (out.kind == OutputValueKind::TensorList) {
    std::string s = "[";
    for (size_t i = 0; i < out.elem_refs.size(); ++i) {
      if (i) {
        s += ", ";
      }
      s += ref_str(out.elem_refs[i]);
    }
    return s + "]";
  }
  return ref_str(out.value_ref);
}

// Append `ref` to `refs` unless it is kInvalid (an unwired / literal operand).
void push_ref(std::vector<ValueRef>& refs, ValueRef ref) {
  if (valid(ref)) {
    refs.push_back(ref);
  }
}

} // namespace

std::vector<ValueRef> Node::input_value_refs() const {
  std::vector<ValueRef> refs;
  for (const NamedArgument& named : inputs) {
    const Argument& arg = named.arg;
    switch (arg.kind()) {
      case ArgKind::Tensor:
        push_ref(refs, arg.as_tensor().ref);
        break;
      case ArgKind::Int:
        push_ref(refs, arg.as_int().ref);
        break;
      case ArgKind::Float:
        push_ref(refs, arg.as_float().ref);
        break;
      case ArgKind::Bool:
        push_ref(refs, arg.as_bool().ref);
        break;
      case ArgKind::IntList:
        for (ValueRef r : arg.as_int_list().refs) {
          push_ref(refs, r);
        }
        break;
      case ArgKind::TensorList:
        for (ValueRef r : arg.as_tensor_list().refs) {
          push_ref(refs, r);
        }
        break;
      case ArgKind::OptionalTensorList:
        for (ValueRef r : arg.as_optional_tensor_list().refs) {
          push_ref(refs, r);
        }
        break;
      default:
        break; // None / String / ScalarType / Float|BoolList / Graph: no refs
    }
  }
  return refs;
}

std::string Node::to_string() const {
  std::string s = name.empty() ? "_" : name;
  s += " = ";
  switch (op_kind) {
    case OpKind::CallFunction:
      s += target;
      break;
    case OpKind::Placeholder:
      s += "<placeholder>";
      break;
    case OpKind::Output:
      s += "<output>";
      break;
  }
  s += "(";
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (i) {
      s += ", ";
    }
    const NamedArgument& named = inputs[i];
    if (!named.name.empty()) {
      s += named.name + "=";
    }
    s += arg_str(named.arg);
    if (named.mutated) {
      s += "!";
    }
  }
  s += ")";
  if (!outputs.empty()) {
    s += " -> ";
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (i) {
        s += ", ";
      }
      s += output_str(outputs[i]);
    }
  }
  return s;
}

} // namespace ptn
