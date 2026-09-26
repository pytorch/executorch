// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>
#include <vector>

#include <executorch/backends/native/runtime/graph/Graph.h>
#include <executorch/backends/native/runtime/graph/Ids.h>
#include <executorch/backends/native/runtime/graph/Value.h>

namespace ptn {

// Classifies a graph-level output position; pinned to the schema OutputKind
// ids. Orthogonal to ValueRole, which classifies a value's storage: the value
// produced at a mutation output is an ordinary Intermediate, and the Buffer or
// UserInput role sits on its target.
enum class OutputKind : int8_t {
  UserOutput = 0,
  BufferMutation = 1,
  UserInputMutation = 2,
};

// Binds a method-graph placeholder to its external storage, unifying the
// schema's NamedTensorRef (data shipped under `key`) and MutableBufferSpec (no
// data, zero-initialized at load, e.g. a KV cache). Tensor metadata is not
// duplicated — it lives on the bound Value, which also mirrors `role`.
struct DataBinding {
  ValueId value_id = kInvalid;
  ValueRole role = ValueRole::Parameter; // Parameter / Buffer / ConstantTensor
  // Fully-qualified name: the constant-file key if has_data, else the buffer's
  // cross-method identity.
  std::string key;
  bool has_data = true; // false only for a non-persistent Buffer
  bool mutated = false; // written in place; state persists across executions
};

// Classifies one graph output. `target_id` is the placeholder value the output
// writes back into — a user input or a buffer, both lifted to placeholders
// here; kInvalid for UserOutput. It replaces the schema's dual-namespace target
// string, whose contents are recoverable from that Value.
struct OutputSpec {
  OutputKind kind = OutputKind::UserOutput;
  ValueId target_id = kInvalid;
};

// A named method: one top-level pure Graph plus the stateful signature bindings
// that Graph deliberately lacks. HOP subgraphs (inside graph.subgraphs) carry
// no bindings of their own — their params are lifted here and passed as
// operands, which is why Method wraps Graph rather than folding into it.
struct Method {
  std::string name;
  Graph graph;
  std::vector<DataBinding> data_bindings;
  std::vector<OutputSpec> output_specs; // aligned to graph.output_ids by index
};

} // namespace ptn
