// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <executorch/backends/native/runtime/graph/Ids.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>

namespace ptn {

// One payload struct per kind of value an fx arg/kwarg can hold; the same set
// of kinds as the schema ArgumentValue union, but in-graph references are
// resolved to ValueIds (the deserializer turns SSA names into arena indices).

struct NoneArg {};

struct TensorArg {
  ValueId id = kInvalid;
};

// A scalar int operand: a literal `value`, or (when `id` is valid) a reference
// to an in-graph int value — a sym_size / arith node — with `value` ignored.
struct IntArg {
  int64_t value = 0;
  ValueId id = kInvalid;
};

struct FloatArg {
  double value = 0.0;
  ValueId id = kInvalid;
};

struct BoolArg {
  bool value = false;
  ValueId id = kInvalid;
};

struct StringArg {
  std::string value;
};

struct ScalarTypeArg {
  ScalarType value = ScalarType::Float;
};

// A list of ints. Element i is a literal (values[i]), or a symbolic reference
// when `ids` is non-empty and ids[i] is valid (values[i] ignored). Empty `ids`
// means all literal; otherwise ids.size() == values.size(). E.g. a dynamic view
// size [s0, -1] is values={0, -1}, ids={<sym>, kInvalid}.
struct IntListArg {
  std::vector<int64_t> values;
  std::vector<ValueId> ids;
};

struct FloatListArg {
  std::vector<double> values;
};

struct BoolListArg {
  std::vector<bool> values;
};

struct TensorListArg {
  std::vector<ValueId> ids;
};

// A list of optional tensor references (Tensor?[]); a kInvalid entry is None.
struct OptionalTensorListArg {
  std::vector<ValueId> ids;
};

// A subgraph passed to a higher-order op (torch.cond / while_loop / map).
// `name` is the original submodule attr label, for debug output only.
struct GraphArg {
  std::string name;
  GraphId subgraph_id = kInvalid;
};

// Same order as the schema ArgumentValue union, with every id one lower:
// flatbuffers reserves 0 for an absent union, so its tags start at 1. Nothing
// casts between the two; the deserializer switches on the schema tag by name.
enum class ArgKind : int8_t {
  Tensor = 0,
  None = 1,
  Int = 2,
  Float = 3,
  Bool = 4,
  String = 5,
  ScalarType = 6,
  IntList = 7,
  FloatList = 8,
  BoolList = 9,
  TensorList = 10,
  OptionalTensorList = 11,
  Graph = 12,
};

// A single fx argument value: a std::variant over the payload structs, with the
// alternatives listed in ArgKind order so kind() is the variant's index. The
// static_asserts below hold the two in step.
//
// Construct from a payload directly: `Argument a = IntArg{5};`.
class Argument {
 private:
  using Storage = std::variant<
      TensorArg,
      NoneArg,
      IntArg,
      FloatArg,
      BoolArg,
      StringArg,
      ScalarTypeArg,
      IntListArg,
      FloatListArg,
      BoolListArg,
      TensorListArg,
      OptionalTensorListArg,
      GraphArg>;

  // kind() casts the variant index straight to ArgKind, so inserting an
  // alternative or an enumerator without the other would silently mis-tag every
  // switch on kind(). These make that a compile error instead.
  template <ArgKind K, typename T>
  static constexpr bool alt_is = std::
      is_same_v<std::variant_alternative_t<static_cast<size_t>(K), Storage>, T>;

  static_assert(alt_is<ArgKind::Tensor, TensorArg>);
  static_assert(alt_is<ArgKind::None, NoneArg>);
  static_assert(alt_is<ArgKind::Int, IntArg>);
  static_assert(alt_is<ArgKind::Float, FloatArg>);
  static_assert(alt_is<ArgKind::Bool, BoolArg>);
  static_assert(alt_is<ArgKind::String, StringArg>);
  static_assert(alt_is<ArgKind::ScalarType, ScalarTypeArg>);
  static_assert(alt_is<ArgKind::IntList, IntListArg>);
  static_assert(alt_is<ArgKind::FloatList, FloatListArg>);
  static_assert(alt_is<ArgKind::BoolList, BoolListArg>);
  static_assert(alt_is<ArgKind::TensorList, TensorListArg>);
  static_assert(alt_is<ArgKind::OptionalTensorList, OptionalTensorListArg>);
  static_assert(alt_is<ArgKind::Graph, GraphArg>);
  static_assert(
      std::variant_size_v<Storage> == static_cast<size_t>(ArgKind::Graph) + 1);

  Storage value_;

  // The live payload, or a std::runtime_error reading "Argument::<what>".
  template <typename T>
  const T& payload(const char* what) const {
    const T* p = std::get_if<T>(&value_);
    if (p == nullptr) {
      throw_bad_kind(what);
    }
    return *p;
  }

  [[noreturn]] static void throw_bad_kind(const char* what);

 public:
  // Pinned to NoneArg rather than defaulted, so the default stays None however
  // the alternatives are ordered.
  Argument() : value_(NoneArg{}) {}

  // Implicit by design: a payload struct is the natural spelling of an
  // Argument. Spelled out per alternative rather than as one constrained
  // template, because several payloads are aggregates that would make a
  // template's overload resolution depend on which of them happens to accept
  // the argument. The move is omitted on the trivially copyable payloads,
  // where it buys nothing.
  // cppcheck-suppress-begin noExplicitConstructor
  /* implicit */ Argument(TensorArg a) : value_(a) {}
  /* implicit */ Argument(NoneArg a) : value_(a) {}
  /* implicit */ Argument(IntArg a) : value_(a) {}
  /* implicit */ Argument(FloatArg a) : value_(a) {}
  /* implicit */ Argument(BoolArg a) : value_(a) {}
  /* implicit */ Argument(StringArg a) : value_(std::move(a)) {}
  /* implicit */ Argument(ScalarTypeArg a) : value_(a) {}
  /* implicit */ Argument(IntListArg a) : value_(std::move(a)) {}
  /* implicit */ Argument(FloatListArg a) : value_(std::move(a)) {}
  /* implicit */ Argument(BoolListArg a) : value_(std::move(a)) {}
  /* implicit */ Argument(TensorListArg a) : value_(std::move(a)) {}
  /* implicit */ Argument(OptionalTensorListArg a) : value_(std::move(a)) {}
  /* implicit */ Argument(GraphArg a) : value_(std::move(a)) {}
  // cppcheck-suppress-end noExplicitConstructor

  ArgKind kind() const {
    return static_cast<ArgKind>(value_.index());
  }

  // Typed payload accessors: throw std::runtime_error unless the kind matches.
  const TensorArg& as_tensor() const {
    return payload<TensorArg>("as_tensor: argument is not a Tensor");
  }
  const IntArg& as_int() const {
    return payload<IntArg>("as_int: argument is not an Int");
  }
  const FloatArg& as_float() const {
    return payload<FloatArg>("as_float: argument is not a Float");
  }
  const BoolArg& as_bool() const {
    return payload<BoolArg>("as_bool: argument is not a Bool");
  }
  const StringArg& as_string() const {
    return payload<StringArg>("as_string: argument is not a String");
  }
  const ScalarTypeArg& as_scalar_type() const {
    return payload<ScalarTypeArg>(
        "as_scalar_type: argument is not a ScalarType");
  }
  const IntListArg& as_int_list() const {
    return payload<IntListArg>("as_int_list: argument is not an IntList");
  }
  const FloatListArg& as_float_list() const {
    return payload<FloatListArg>("as_float_list: argument is not a FloatList");
  }
  const BoolListArg& as_bool_list() const {
    return payload<BoolListArg>("as_bool_list: argument is not a BoolList");
  }
  const TensorListArg& as_tensor_list() const {
    return payload<TensorListArg>(
        "as_tensor_list: argument is not a TensorList");
  }
  const OptionalTensorListArg& as_optional_tensor_list() const {
    return payload<OptionalTensorListArg>(
        "as_optional_tensor_list: argument is not an OptionalTensorList");
  }
  const GraphArg& as_graph() const {
    return payload<GraphArg>("as_graph: argument is not a Graph");
  }
};

// A positional or keyword argument. `name` is the operator-schema parameter
// name (NOT a value reference; empty for positional-only). `mutated` is true
// when the op writes this input in place (op-schema Tensor(a!), e.g. counter /
// kv_cache).
struct NamedArgument {
  std::string name;
  Argument arg;
  bool mutated = false;
};

} // namespace ptn
