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
// wiring, a storage alias and an open annotation map. The ref fields are plain
// handles; whether one is in range is a property of the owning arena, so
// nothing here validates them.
//
// The variant's alternatives are listed in ValueKind order, so kind() is its
// index. A List holds ValueRefs to its elements rather than nested Values, so
// nesting goes through the arena; nothing deserialized is a List, it exists for
// in-memory rewrites such as grouping a tuple.
class Value {
 private:
  std::variant<std::monostate, TensorMeta, Scalar, std::vector<ValueRef>>
      value_;

 public:
  // SSA name, scoped to the enclosing Graph.
  std::string name;
  // Defining node; invalid => graph input.
  NodeRef producer_ref = kInvalid;
  // Def-use, built by inverting node inputs.
  std::vector<NodeRef> consumer_refs;
  // Shares storage with this value (a view); fresh if invalid.
  ValueRef alias_ref = kInvalid;
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

  Value(std::string name, std::vector<ValueRef> elem_refs)
      : value_(std::move(elem_refs)), name(std::move(name)) {}

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
    return std::holds_alternative<std::vector<ValueRef>>(value_);
  }
  bool is_none() const {
    return std::holds_alternative<std::monostate>(value_);
  }

  // Typed payload accessors: throw std::runtime_error unless the kind matches.
  const TensorMeta& tensor_meta() const;
  const Scalar& scalar() const;
  const std::vector<ValueRef>& content_refs() const;
};

} // namespace ptn
