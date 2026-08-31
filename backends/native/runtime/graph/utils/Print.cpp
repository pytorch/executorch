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

} // namespace ptn
