// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <executorch/backends/native/runtime/graph/Ids.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>

namespace ptn {

// One payload struct per kind of value an fx arg/kwarg can hold; mirrors the
// schema ArgumentValue union, but in-graph references are resolved to ValueRefs
// (the deserializer turns SSA names into arena indices).

struct NoneArg {};

struct TensorArg {
  ValueRef ref = kInvalid;
};

// A scalar int operand: a literal `value`, or (when `ref` is valid) a reference
// to an in-graph int value — a sym_size / arith node — with `value` ignored.
struct IntArg {
  int64_t value = 0;
  ValueRef ref = kInvalid;
};

struct FloatArg {
  double value = 0.0;
  ValueRef ref = kInvalid;
};

struct BoolArg {
  bool value = false;
  ValueRef ref = kInvalid;
};

struct StringArg {
  std::string value;
};

struct ScalarTypeArg {
  ScalarType value = ScalarType::Float;
};

// A list of ints. Element i is a literal (values[i]), or a symbolic reference
// when `refs` is non-empty and refs[i] is valid (values[i] ignored). Empty
// `refs` means all literal; otherwise refs.size() == values.size(). E.g. a
// dynamic view size [s0, -1] is values={0, -1}, refs={<sym>, kInvalid}.
struct IntListArg {
  std::vector<int64_t> values;
  std::vector<ValueRef> refs;
};

struct FloatListArg {
  std::vector<double> values;
};

struct BoolListArg {
  std::vector<bool> values;
};

// A list of tensor references (e.g. cat's input list).
struct TensorListArg {
  std::vector<ValueRef> refs;
};

// A list of optional tensor references (Tensor?[]); a kInvalid entry is None.
struct OptionalTensorListArg {
  std::vector<ValueRef> refs;
};

// A subgraph passed to a higher-order op (torch.cond / while_loop / map).
// `name` is the original submodule attr label (debug only); `subgraph_ref`
// indexes the subgraph arena (populated once Graph/Model land).
struct GraphArg {
  std::string name;
  GraphRef subgraph_ref = kInvalid;
};

enum class ArgKind : int8_t {
  None = 0,
  Tensor = 1,
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
// alternatives listed in ArgKind order so kind() is the variant's index and the
// two can never drift apart.
//
// Construct from a payload directly (`Argument a = IntArg{5};`); read the
// payload via the as_*() accessors after checking kind() (each throws
// std::runtime_error on a kind mismatch).
class Argument {
 private:
  using Storage = std::variant<
      NoneArg,
      TensorArg,
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
  Argument() = default; // None

  // Implicit by design: a payload struct is the natural spelling of an
  // Argument. Spelled out per alternative rather than as one constrained
  // template, because several payloads are aggregates that would make a
  // template's overload resolution depend on which of them happens to accept
  // the argument. The move is omitted on the trivially copyable payloads,
  // where it buys nothing.
  //
  // cppcheck reads every one of these as an unintended converting constructor;
  // the suppression is scoped to this block so a genuinely accidental implicit
  // constructor added elsewhere in the class is still reported.
  // cppcheck-suppress-begin noExplicitConstructor
  /* implicit */ Argument(TensorArg a) : value_(a) {}
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

  // Typed payload accessors. Each throws std::runtime_error unless kind()
  // matches; guard with kind().
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
