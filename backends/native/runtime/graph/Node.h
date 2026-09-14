// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <any>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Ids.h>

namespace ptn {

// fx node kind (node.op). Pinned to the schema OpKind ids.
enum class OpKind : int8_t {
  CallFunction = 0,
  Placeholder = 1,
  Output = 2,
};

// What a single node Output produces. Pinned to the schema OutputValueKind ids;
// named to match the schema (distinct from the graph-level OutputKind — user
// output vs buffer mutation).
enum class OutputValueKind : int8_t {
  Tensor = 0,
  TensorList = 1,
  Int = 2,
  Bool = 3,
  Float = 4,
};

// One value produced by a node. Tensor / Int / Bool / Float use `value_id`;
// TensorList (e.g. split) uses `elem_ids`. The return-ABI grouping is
// preserved so engine translation can tell a single result from a tuple / list
// (topk emits two Tensor outputs; split emits one TensorList output). The
// storage-alias fact lives on the produced Value, not here.
struct Output {
  OutputValueKind kind = OutputValueKind::Tensor;
  ValueId value_id = kInvalid;
  std::vector<ValueId> elem_ids;
};

// One fx node: an op invocation (CallFunction) or a graph-boundary marker
// (Placeholder / Output). For an Output node, `inputs` is the ordered return
// list (tensors and literals alike) and `outputs` is empty; for a Placeholder,
// `target` is empty and it produces a single output. `attrs` is a transient
// scratch map (the fx node.meta analog); it is not serialized.
struct Node {
  std::string name;
  OpKind op_kind = OpKind::CallFunction;
  // fqn, e.g. "torch.ops.aten.addmm.default"; empty for placeholder / output.
  std::string target;
  std::vector<NamedArgument> inputs;
  std::vector<Output> outputs;
  std::unordered_map<std::string, std::any> attrs;

  bool is_call() const {
    return op_kind == OpKind::CallFunction;
  }
  bool is_placeholder() const {
    return op_kind == OpKind::Placeholder;
  }
  bool is_output() const {
    return op_kind == OpKind::Output;
  }

  // Every ValueId this node consumes: tensor args, tensor-list / optional-list
  // elements, and symbolic scalar ids (kInvalid entries skipped). Used to
  // (re)build def-use wiring.
  std::vector<ValueId> input_value_ids() const;
};

} // namespace ptn
