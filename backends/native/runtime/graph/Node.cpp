// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/graph/Node.h>

#include <vector>

namespace ptn {

namespace {

void push_id(std::vector<ValueId>& ids, ValueId id) {
  if (valid(id)) {
    ids.push_back(id);
  }
}

} // namespace

std::vector<ValueId> Node::input_value_ids() const {
  std::vector<ValueId> ids;
  for (const NamedArgument& named : inputs) {
    const Argument& arg = named.arg;
    switch (arg.kind()) {
      case ArgKind::Tensor:
        push_id(ids, arg.as_tensor().id);
        break;
      case ArgKind::Int:
        push_id(ids, arg.as_int().id);
        break;
      case ArgKind::Float:
        push_id(ids, arg.as_float().id);
        break;
      case ArgKind::Bool:
        push_id(ids, arg.as_bool().id);
        break;
      case ArgKind::IntList:
        for (ValueId i : arg.as_int_list().ids) {
          push_id(ids, i);
        }
        break;
      case ArgKind::TensorList:
        for (ValueId i : arg.as_tensor_list().ids) {
          push_id(ids, i);
        }
        break;
      case ArgKind::OptionalTensorList:
        for (ValueId i : arg.as_optional_tensor_list().ids) {
          push_id(ids, i);
        }
        break;
      // Carry no ids. Listed rather than defaulted so a new ArgKind that does
      // carry one is a compiler warning here, not a silently unwired operand.
      case ArgKind::None:
      case ArgKind::String:
      case ArgKind::ScalarType:
      case ArgKind::FloatList:
      case ArgKind::BoolList:
      case ArgKind::Graph:
        break;
    }
  }
  return ids;
}

} // namespace ptn
