// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/utils/Print.h>

#include <cstddef>
#include <string>

namespace ptn {

namespace {

const char* output_kind_name(OutputKind kind) {
  switch (kind) {
    case OutputKind::UserOutput:
      return "UserOutput";
    case OutputKind::BufferMutation:
      return "BufferMutation";
    case OutputKind::UserInputMutation:
      return "UserInputMutation";
  }
  return "?";
}

const char* value_role_name(ValueRole role) {
  switch (role) {
    case ValueRole::Intermediate:
      return "Intermediate";
    case ValueRole::UserInput:
      return "UserInput";
    case ValueRole::Parameter:
      return "Parameter";
    case ValueRole::Buffer:
      return "Buffer";
    case ValueRole::ConstantTensor:
      return "ConstantTensor";
  }
  return "?";
}

} // namespace

std::string to_string(const Method& method) {
  std::string s = "method " + method.name + "\n";
  s += to_string(method.graph);

  s += "data_bindings: [";
  for (size_t i = 0; i < method.data_bindings.size(); ++i) {
    if (i) {
      s += ", ";
    }
    const DataBinding& b = method.data_bindings[i];
    s += "%" + std::to_string(b.value_id) + "=" + b.key + "(" +
        value_role_name(b.role) + (b.has_data ? "" : ",zero_init") +
        (b.mutated ? ",mutated" : "") + ")";
  }
  s += "]\n";

  s += "output_specs: [";
  for (size_t i = 0; i < method.output_specs.size(); ++i) {
    if (i) {
      s += ", ";
    }
    const OutputSpec& o = method.output_specs[i];
    s += output_kind_name(o.kind);
    if (valid(o.target_id)) {
      s += "(%" + std::to_string(o.target_id) + ")";
    }
  }
  s += "]\n";

  return s;
}

} // namespace ptn
