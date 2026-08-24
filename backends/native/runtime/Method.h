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

// Classifies a graph-level output value (schema OutputKind). USER_OUTPUT is a
// real result; the mutation kinds mark a value that writes back into a
// persistent buffer or a user input. Distinct from the node-level
// OutputValueKind (tensor / list / scalar).
enum class OutputKind : int8_t {
  UserOutput = 0,
  BufferMutation = 1,
  UserInputMutation = 2,
};

// Binds a method-graph placeholder value to its external storage. Unifies the
// schema's NamedTensorRef (constant-backed: parameter / frozen constant /
// persistent buffer — data shipped under `key`) and MutableBufferSpec
// (non-persistent buffer — no data, zero-initialized at load, e.g. a KV cache).
// `key` is the namespace-3 fully-qualified name: an external-constant-file key
// when `has_data`, else the buffer's cross-method identity. `has_data` selects
// the load path (fetch bytes vs zero-init). `mutated` (schema
// NamedTensorRef.mutated) marks a Buffer written in place, whose state persists
// across executions — parameters / frozen constants / read-only buffers are
// false. `role` mirrors the bound Value's role (kept here too for direct access
// when iterating bindings). Tensor metadata is not duplicated — it lives on the
// bound Value.
struct DataBinding {
  ValueRef value_ref = kInvalid;
  ValueRole role = ValueRole::Parameter; // Parameter / Buffer / ConstantTensor
  // Namespace-3 fqn: the constant-file key if has_data, else the buffer's
  // cross-method identity.
  std::string key;
  bool has_data = true; // bytes shipped under key, else zero-init at load
  bool mutated = false; // written in place; state persists across executions
};

// Per graph-output classification (schema OutputSpec), aligned to
// graph.output_refs by index. `target_ref` references the mutated placeholder
// value in this method graph — the user input (UserInputMutation) or the buffer
// (BufferMutation), both of which are lifted placeholders here; kInvalid for
// UserOutput. The wire strings are recovered from that Value: its `name` (SSA,
// namespace 2) and, for a buffer, its DataBinding's `key` (fqn, namespace 3).
struct OutputSpec {
  OutputKind kind = OutputKind::UserOutput;
  ValueRef target_ref = kInvalid;
};

// A named method: one top-level pure Graph plus its stateful signature
// bindings. `data_bindings` is the authoritative placeholder→storage table
// (merging the schema's constants + mutable_buffers); the deserializer also
// stamps each bound placeholder Value's role / data_key for O(1) per-value
// queries. HOP subgraphs (inside graph.subgraphs) carry no bindings — their
// params are lifted here and passed as operands.
struct Method {
  std::string name;
  Graph graph;
  std::vector<DataBinding> data_bindings;
  std::vector<OutputSpec> output_specs; // aligned to graph.output_refs by index

  // Multi-line debug dump: name, the graph, and the binding tables.
  std::string to_string() const;
};

} // namespace ptn
