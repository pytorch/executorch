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
#include <utility>
#include <variant>
#include <vector>

#include <executorch/backends/native/runtime/graph/Ids.h>
#include <executorch/backends/native/runtime/graph/Scalar.h>
#include <executorch/backends/native/runtime/graph/TensorMeta.h>

namespace ptn {

enum class ValueKind : int8_t {
  None = 0,
  Tensor = 1,
  Scalar = 2,
  List = 3,
};

// A single SSA value (dataflow edge) in a Graph: its contents plus def-use
// wiring, a storage alias and an open annotation map. The id fields are plain
// handles; whether one is in range is a property of the owning arena, so
// nothing here validates them.
//
// The variant's alternatives are listed in ValueKind order, so kind() is its
// index. A Tensor carries metadata only, so a weight is an ordinary arena
// value like any other, with its bytes held outside the graph. A List holds
// ValueIds to its elements, so nesting goes through the arena; nothing
// deserialized is a List, it exists for in-memory rewrites such as grouping a
// tuple.
class Value {
 private:
  std::variant<std::monostate, TensorMeta, Scalar, std::vector<ValueId>> value_;

 public:
  // SSA name, scoped to the enclosing Graph.
  std::string name;
  // Defining node (a placeholder node for a graph input); invalid => unwired.
  NodeId producer_id = kInvalid;
  // Def-use, built by inverting node inputs. The consuming nodes, each listed
  // once: a node that reads this value twice (`add(x, x)`) appears once, so
  // size() counts consumers rather than uses.
  std::vector<NodeId> consumer_ids;
  // Shares storage with this value (a view); fresh if invalid.
  ValueId alias_id = kInvalid;
  // Open annotations for graph passes and engines, like node.meta in FX.
  std::unordered_map<std::string, std::any> attrs;

  Value() = default; // a None value with an empty name

  explicit Value(std::string name) // a named None value
      : name(std::move(name)) {}

  Value(std::string name, TensorMeta meta)
      : value_(std::move(meta)), name(std::move(name)) {}

  // The empty dim_order_hint is what makes the tensor contiguous.
  Value(std::string name, ScalarType dtype, std::vector<Dim> sizes)
      : value_(TensorMeta{dtype, std::move(sizes), {}}),
        name(std::move(name)) {}

  Value(std::string name, Scalar value)
      : value_(value), name(std::move(name)) {}

  Value(std::string name, std::vector<ValueId> elem_ids)
      : value_(std::move(elem_ids)), name(std::move(name)) {}

  ValueKind kind() const {
    return static_cast<ValueKind>(value_.index());
  }
  bool is_tensor() const {
    return std::holds_alternative<TensorMeta>(value_);
  }
  bool is_scalar() const {
    return std::holds_alternative<Scalar>(value_);
  }
  bool is_list() const {
    return std::holds_alternative<std::vector<ValueId>>(value_);
  }
  bool is_none() const {
    return std::holds_alternative<std::monostate>(value_);
  }

  // Typed payload accessors: throw std::runtime_error unless the kind matches.
  const TensorMeta& tensor_meta() const;
  const Scalar& scalar() const;
  const std::vector<ValueId>& content_ids() const;
};

} // namespace ptn
