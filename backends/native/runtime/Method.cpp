// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/Method.h>

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

} // namespace

std::string Method::to_string() const {
  std::string s = "method " + name + "\n";
  s += graph.to_string();

  s += "data_bindings: [";
  for (size_t i = 0; i < data_bindings.size(); ++i) {
    if (i) {
      s += ", ";
    }
    const DataBinding& b = data_bindings[i];
    s += "%" + std::to_string(b.value_ref) + "=" + b.key + "(" +
        value_role_name(b.role) + (b.has_data ? "" : ",zero_init") +
        (b.mutated ? ",mutated" : "") + ")";
  }
  s += "]\n";

  s += "output_specs: [";
  for (size_t i = 0; i < output_specs.size(); ++i) {
    if (i) {
      s += ", ";
    }
    const OutputSpec& o = output_specs[i];
    s += output_kind_name(o.kind);
    if (valid(o.target_ref)) {
      s += "(%" + std::to_string(o.target_ref) + ")";
    }
  }
  s += "]\n";

  return s;
}

} // namespace ptn
