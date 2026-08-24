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

// What a Value holds.
enum class ValueKind : int8_t {
  None = 0,
  Tensor = 1,
  Scalar = 2,
  List = 3,
};

// A single SSA value (dataflow edge) in a Graph: its contents plus def-use
// wiring, a storage-alias fact, and an open attrs map.
//
// The contents are a std::variant whose alternatives are listed in ValueKind
// order, so kind() is the variant's index. Construct via the constructors (one
// per kind); read the payload via the typed accessors after checking kind()
// (each accessor throws std::runtime_error on a kind mismatch). A List holds
// ValueRefs to its element values (a grouping over arena values, e.g. a tuple
// produced by an in-memory rewrite) — nesting is via the arena, so there is no
// recursive value type. The AOT deserializer builds only Tensor / Scalar /
// None; List is reserved for in-memory construction.
class Value {
 private:
  std::variant<std::monostate, TensorMeta, Scalar, std::vector<ValueRef>>
      value_;

 public:
  // SSA name, scoped to the enclosing Graph.
  std::string name;
  // Defining node (a placeholder node for a graph input); invalid => unwired.
  NodeRef producer_ref = kInvalid;
  // Def-use, built by inverting node inputs. The consuming nodes, each listed
  // once: a node that reads this value twice (`add(x, x)`) appears once, so
  // size() counts consumers rather than uses.
  std::vector<NodeRef> consumer_refs;
  // Shares storage with this value (a view); fresh if invalid.
  ValueRef alias_ref = kInvalid;
  // Scratch + planner annotations.
  std::unordered_map<std::string, std::any> attrs;

  Value() = default; // a None value with an empty name

  explicit Value(std::string name) // a named None value
      : name(std::move(name)) {}

  Value(std::string name, TensorMeta meta) // Tensor
      : value_(std::move(meta)), name(std::move(name)) {}

  // Tensor from dtype + sizes: a contiguous, unquantized TensorMeta. The empty
  // dim_order_hint is what makes it contiguous, so it is spelled out.
  Value(std::string name, ScalarType dtype, std::vector<Dim> sizes)
      : value_(TensorMeta{dtype, std::move(sizes), {}}),
        name(std::move(name)) {}

  Value(std::string name, Scalar value) // Scalar
      : value_(value), name(std::move(name)) {}

  Value(std::string name, std::vector<ValueRef> elem_refs) // List
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

  // Typed payload accessors. Each throws std::runtime_error unless kind()
  // matches; guard with the is_*() / kind() predicates.
  const TensorMeta& tensor_meta() const;
  const Scalar& scalar() const;
  const std::vector<ValueRef>& content_refs() const;
};

} // namespace ptn
