/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * Graph: backend-facing IR adapter.
 *
 * Wraps ExecuTorch's flatbuffer (ExecutionPlan / KernelCall) behind a
 * backing-agnostic API. A different serialization could replace the
 * underlying storage without changing this header.
 *
 * IR schema (what Graph presents):
 *
 *   table Graph {
 *     version:         string;
 *     values:          [Value];        // dense pool, indexed by value_id
 *     inputs:          [uint];
 *     outputs:         [uint];
 *     mutable_buffers: [uint];         // values that persist across executes
 *     operators:       [OperatorDef];  // deduped op-name registry
 *     instructions:    [KernelCall];   // flat list of operator calls
 *   }
 *
 *   table KernelCall {
 *     op_index: uint;                  // -> operators[op_index]
 *     args:     [uint];                // value_ids; last = output
 *   }
 *
 * Derived views (computed at construction):
 *   mem_obj_id(vid)    — dense int from (pool_id, offset) sort-rank
 *   value_kind(vid)    — INPUT / OUTPUT / CONSTANT / INTERMEDIATE /
 * MUTABLE_BUFFER
 */

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/core/tensor_shape_dynamism.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/schema/program_generated.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include <cstdint>

namespace executorch {
namespace backends {
namespace portable {

class Graph;

enum class ValueKind : uint8_t {
  INPUT = 0,
  OUTPUT,
  CONSTANT,
  MUTABLE_BUFFER,
  INTERMEDIATE,
};

enum class ValueType : uint8_t {
  None = 0,
  Int,
  Double,
  Bool,
  Tensor,
  IntList,
  TensorList,
  OptionalTensorList,
  Other,
};

/**
 * Wraps a flatbuffer KernelCall. Provides access to op name and
 * input/output value_ids. Last arg is the (single) output.
 */
class OperatorCall {
 public:
  explicit OperatorCall(
      const executorch_flatbuffer::KernelCall* call,
      const Graph* graph)
      : call_(call), graph_(graph) {}

  uint32_t node_id() const {
    return node_id_;
  }
  void set_node_id(uint32_t id) {
    node_id_ = id;
  }

  const char* name() const;
  const char* overload() const;
  std::string full_name() const;

  runtime::Span<const int32_t> args() const {
    auto* a = call_->args();
    return a ? runtime::Span<const int32_t>(a->data(), a->size())
             : runtime::Span<const int32_t>{};
  }

  runtime::Span<const int32_t> inputs() const {
    auto a = args();
    return a.empty() ? a : runtime::Span<const int32_t>(a.data(), a.size() - 1);
  }

  size_t num_inputs() const {
    auto a = args();
    return a.empty() ? 0 : a.size() - 1;
  }

  uint32_t input(size_t i) const {
    ET_CHECK_MSG(
        i < num_inputs(),
        "OperatorCall::input: index %zu >= num_inputs()=%zu",
        i,
        num_inputs());
    return static_cast<uint32_t>(args()[i]);
  }

  size_t num_outputs() const {
    return args().empty() ? 0 : 1;
  }

  uint32_t output(size_t i) const {
    ET_CHECK_MSG(
        i < num_outputs(),
        "OperatorCall::output: index %zu >= num_outputs()=%zu",
        i,
        num_outputs());
    auto a = args();
    return static_cast<uint32_t>(a[a.size() - 1]);
  }

 private:
  const executorch_flatbuffer::KernelCall* call_;
  const Graph* graph_;
  uint32_t node_id_ = 0;
};

/**
 * IR view of a program. Adapts executorch_flatbuffer::ExecutionPlan.
 */
class Graph {
 public:
  explicit Graph(
      const executorch_flatbuffer::ExecutionPlan* plan,
      const executorch_flatbuffer::Program* program = nullptr)
      : plan_(plan), program_(program) {
    if (auto* in = plan_->inputs()) {
      input_ids_.reserve(in->size());
      for (size_t i = 0; i < in->size(); ++i) {
        input_ids_.insert(static_cast<uint32_t>(in->Get(i)));
      }
    }
    if (auto* out = plan_->outputs()) {
      output_ids_.reserve(out->size());
      for (size_t i = 0; i < out->size(); ++i) {
        output_ids_.insert(static_cast<uint32_t>(out->Get(i)));
      }
    }

    // Compute mem_obj_id: sort (pool_id, offset) pairs, assign dense rank.
    // Same (pool_id, offset) → same id (shared storage).
    size_t n_vals = num_values();
    mem_obj_ids_.assign(n_vals, -1);
    if (n_vals == 0)
      return;

    struct Entry {
      uint64_t key; // (pool_id << 32) | offset
      uint32_t value_id;
    };
    std::vector<Entry> entries;
    entries.reserve(n_vals);
    for (uint32_t i = 0; i < n_vals; ++i) {
      auto* val = value_meta(i);
      if (!val ||
          val->val_type() != executorch_flatbuffer::KernelTypes::Tensor) {
        continue;
      }
      auto* t = val->val_as_Tensor();
      if (!t)
        continue;
      auto* alloc = t->allocation_info();
      if (!alloc)
        continue;
      uint64_t pool = static_cast<uint32_t>(alloc->memory_id());
      uint64_t off = alloc->memory_offset_low();
      entries.push_back({(pool << 32) | off, i});
    }
    if (entries.empty())
      return;

    std::sort(
        entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
          return a.key < b.key;
        });

    int32_t next_id = -1;
    uint64_t prev_key = ~0ULL;
    for (const auto& e : entries) {
      if (e.key != prev_key) {
        ++next_id;
        prev_key = e.key;
      }
      mem_obj_ids_[e.value_id] = next_id;
    }

    // Mutable buffers: allocated tensors that aren't IO, constants, or
    // produced by any op. These are buffer placeholders (e.g., KV cache)
    // that persist across execute() calls.
    {
      std::unordered_set<uint32_t> produced_vids;
      auto chains = plan_->chains();
      auto instrs = chains->Get(0)->instructions();
      size_t n_instr = instrs ? instrs->size() : 0;
      for (size_t oi = 0; oi < n_instr; ++oi) {
        OperatorCall op = get_op(oi);
        for (size_t k = 0; k < op.num_outputs(); ++k) {
          produced_vids.insert(op.output(k));
        }
      }
      for (uint32_t i = 0; i < n_vals; ++i) {
        if (mem_obj_ids_[i] < 0)
          continue;
        if (input_ids_.count(i) > 0)
          continue;
        if (output_ids_.count(i) > 0)
          continue;
        if (tensor_constant_data_key(i) != nullptr)
          continue;
        if (produced_vids.count(i) > 0)
          continue;
        mutable_buffer_ids_.push_back(i);
      }
    }
  }

  //===------------------------------------------------------------------===//
  // Version
  //===------------------------------------------------------------------===//

  static const char* version() {
    return "1.0";
  }

  //===------------------------------------------------------------------===//
  // Values
  //===------------------------------------------------------------------===//

  size_t num_values() const {
    auto v = plan_->values();
    return v ? v->size() : 0;
  }

  const executorch_flatbuffer::EValue* value_meta(uint32_t value_id) const {
    auto values = plan_->values();
    if (!values || value_id >= values->size())
      return nullptr;
    return values->Get(value_id);
  }

  ValueKind value_kind(uint32_t value_id) const;
  int32_t mem_obj_id(uint32_t value_id) const;

  //===------------------------------------------------------------------===//
  // Typed value accessors
  //===------------------------------------------------------------------===//

  ValueType value_type(uint32_t value_id) const;

  int64_t int_value(uint32_t value_id) const;
  double double_value(uint32_t value_id) const;
  bool bool_value(uint32_t value_id) const;

  ::executorch::aten::ScalarType tensor_dtype(uint32_t value_id) const;
  ::executorch::runtime::Span<const int32_t> tensor_sizes(
      uint32_t value_id) const;
  ::executorch::runtime::Span<const uint8_t> tensor_dim_order(
      uint32_t value_id) const;
  ::executorch::aten::TensorShapeDynamism tensor_shape_dynamism(
      uint32_t value_id) const;

  // NDM key (FQN) for external constants, or nullptr.
  const char* tensor_constant_data_key(uint32_t value_id) const;

  // Raw bytes for inline constants, or empty span.
  ::executorch::runtime::Span<const uint8_t> tensor_inline_data(
      uint32_t value_id) const;

  bool is_constant(uint32_t value_id) const {
    if (tensor_constant_data_key(value_id) != nullptr)
      return true;
    return !tensor_inline_data(value_id).empty();
  }

  size_t tensor_nbytes_max(uint32_t value_id) const;

  ::executorch::runtime::Span<const int64_t> int_list_member_ids(
      uint32_t value_id) const;

  ::executorch::runtime::Span<const int32_t> tensor_list_member_ids(
      uint32_t value_id) const;

  //===------------------------------------------------------------------===//
  // Input/Output IDs
  //===------------------------------------------------------------------===//

  size_t num_input_ids() const {
    auto in = plan_->inputs();
    return in ? in->size() : 0;
  }

  uint32_t input_id(size_t i) const {
    auto in = plan_->inputs();
    ET_CHECK_MSG(
        in && i < in->size(),
        "Graph::input_id(%zu) out of range (have %zu inputs)",
        i,
        in ? in->size() : 0);
    return static_cast<uint32_t>(in->Get(i));
  }

  size_t num_output_ids() const {
    auto out = plan_->outputs();
    return out ? out->size() : 0;
  }

  uint32_t output_id(size_t i) const {
    auto out = plan_->outputs();
    ET_CHECK_MSG(
        out && i < out->size(),
        "Graph::output_id(%zu) out of range (have %zu outputs)",
        i,
        out ? out->size() : 0);
    return static_cast<uint32_t>(out->Get(i));
  }

  //===------------------------------------------------------------------===//
  // Mutable Buffers
  //===------------------------------------------------------------------===//

  size_t num_mutable_buffer_ids() const {
    return mutable_buffer_ids_.size();
  }

  uint32_t mutable_buffer_id(size_t i) const {
    ET_CHECK_MSG(
        i < mutable_buffer_ids_.size(),
        "Graph::mutable_buffer_id: index %zu out of range "
        "(have %zu mutable buffers)",
        i,
        mutable_buffer_ids_.size());
    return mutable_buffer_ids_[i];
  }

  //===------------------------------------------------------------------===//
  // Operators
  //===------------------------------------------------------------------===//

  size_t num_operators() const {
    auto ops = plan_->operators();
    return ops ? ops->size() : 0;
  }

  const char* operator_name(size_t idx) const {
    auto ops = plan_->operators();
    if (!ops || idx >= ops->size())
      return nullptr;
    auto op = ops->Get(idx);
    return op && op->name() ? op->name()->c_str() : nullptr;
  }

  const char* operator_overload(size_t idx) const {
    auto ops = plan_->operators();
    if (!ops || idx >= ops->size())
      return nullptr;
    auto op = ops->Get(idx);
    return op && op->overload() ? op->overload()->c_str() : nullptr;
  }

  //===------------------------------------------------------------------===//
  // Instructions
  //===------------------------------------------------------------------===//

  size_t num_instructions() const {
    auto chains = plan_->chains();
    ET_CHECK_MSG(chains && chains->size() > 0, "Graph has no chains");
    auto instrs = chains->Get(0)->instructions();
    return instrs ? instrs->size() : 0;
  }

  OperatorCall get_op(size_t op_idx) const {
    auto chains = plan_->chains();
    ET_CHECK_MSG(chains && chains->size() > 0, "Graph has no chains");
    auto instrs = chains->Get(0)->instructions();
    ET_CHECK_MSG(
        instrs && op_idx < instrs->size(),
        "Graph::get_op: op_idx=%zu out of range (have %zu instructions)",
        op_idx,
        instrs ? instrs->size() : 0);
    auto instr = instrs->Get(op_idx);
    ET_CHECK_MSG(
        instr->instr_args_type() ==
            executorch_flatbuffer::InstructionArguments::KernelCall,
        "Graph::get_op: instruction at op_idx=%zu is not a KernelCall "
        "(type=%u)",
        op_idx,
        static_cast<unsigned>(instr->instr_args_type()));
    auto kernel = static_cast<const executorch_flatbuffer::KernelCall*>(
        instr->instr_args());
    return OperatorCall(kernel, this);
  }

  OperatorCall get_instruction(size_t idx) const {
    return get_op(idx);
  }

 private:
  const executorch_flatbuffer::ExecutionPlan* plan_;
  const executorch_flatbuffer::Program* program_;
  std::unordered_set<uint32_t> input_ids_;
  std::unordered_set<uint32_t> output_ids_;
  std::vector<int32_t> mem_obj_ids_;
  std::vector<uint32_t> mutable_buffer_ids_;
};

inline const char* OperatorCall::name() const {
  return graph_->operator_name(call_->op_index());
}

inline const char* OperatorCall::overload() const {
  return graph_->operator_overload(call_->op_index());
}

inline std::string OperatorCall::full_name() const {
  const char* base = name();
  const char* ovl = overload();
  if (!base)
    return {};
  if (!ovl || *ovl == '\0')
    return std::string(base);
  std::string s;
  s.reserve(std::strlen(base) + 1 + std::strlen(ovl));
  s.append(base);
  s.push_back('.');
  s.append(ovl);
  return s;
}

inline ValueKind Graph::value_kind(uint32_t value_id) const {
  if (input_ids_.count(value_id))
    return ValueKind::INPUT;
  if (output_ids_.count(value_id))
    return ValueKind::OUTPUT;

  auto val = value_meta(value_id);
  if (val && val->val_type() == executorch_flatbuffer::KernelTypes::Tensor) {
    auto* tensor = val->val_as_Tensor();
    if (tensor && tensor->data_buffer_idx() > 0) {
      return ValueKind::CONSTANT;
    }
  }
  return ValueKind::INTERMEDIATE;
}

inline int32_t Graph::mem_obj_id(uint32_t value_id) const {
  return value_id < mem_obj_ids_.size() ? mem_obj_ids_[value_id] : -1;
}

inline ValueType Graph::value_type(uint32_t value_id) const {
  auto* val = value_meta(value_id);
  if (!val)
    return ValueType::None;
  using KT = executorch_flatbuffer::KernelTypes;
  switch (val->val_type()) {
    case KT::Null:
      return ValueType::None;
    case KT::Int:
      return ValueType::Int;
    case KT::Double:
      return ValueType::Double;
    case KT::Bool:
      return ValueType::Bool;
    case KT::Tensor:
      return ValueType::Tensor;
    case KT::IntList:
      return ValueType::IntList;
    case KT::TensorList:
      return ValueType::TensorList;
    case KT::OptionalTensorList:
      return ValueType::OptionalTensorList;
    default:
      return ValueType::Other;
  }
}

inline int64_t Graph::int_value(uint32_t value_id) const {
  auto* val = value_meta(value_id);
  ET_CHECK_MSG(
      val && val->val_type() == executorch_flatbuffer::KernelTypes::Int,
      "Graph::int_value(%u): not an Int",
      value_id);
  return static_cast<const executorch_flatbuffer::Int*>(val->val())->int_val();
}

inline double Graph::double_value(uint32_t value_id) const {
  auto* val = value_meta(value_id);
  ET_CHECK_MSG(
      val && val->val_type() == executorch_flatbuffer::KernelTypes::Double,
      "Graph::double_value(%u): not a Double",
      value_id);
  return static_cast<const executorch_flatbuffer::Double*>(val->val())
      ->double_val();
}

inline bool Graph::bool_value(uint32_t value_id) const {
  auto* val = value_meta(value_id);
  ET_CHECK_MSG(
      val && val->val_type() == executorch_flatbuffer::KernelTypes::Bool,
      "Graph::bool_value(%u): not a Bool",
      value_id);
  return static_cast<const executorch_flatbuffer::Bool*>(val->val())
      ->bool_val();
}

namespace detail {
inline const executorch_flatbuffer::Tensor* tensor_or_null(
    const executorch_flatbuffer::EValue* val) {
  if (!val || val->val_type() != executorch_flatbuffer::KernelTypes::Tensor) {
    return nullptr;
  }
  return val->val_as_Tensor();
}
} // namespace detail

inline ::executorch::aten::ScalarType Graph::tensor_dtype(
    uint32_t value_id) const {
  auto* t = detail::tensor_or_null(value_meta(value_id));
  ET_CHECK_MSG(t, "Graph::tensor_dtype(%u): not a Tensor", value_id);
  return static_cast<::executorch::aten::ScalarType>(t->scalar_type());
}

inline ::executorch::runtime::Span<const int32_t> Graph::tensor_sizes(
    uint32_t value_id) const {
  auto* t = detail::tensor_or_null(value_meta(value_id));
  ET_CHECK_MSG(t, "Graph::tensor_sizes(%u): not a Tensor", value_id);
  auto* s = t->sizes();
  return s ? ::executorch::runtime::Span<const int32_t>(s->data(), s->size())
           : ::executorch::runtime::Span<const int32_t>{};
}

inline ::executorch::runtime::Span<const uint8_t> Graph::tensor_dim_order(
    uint32_t value_id) const {
  auto* t = detail::tensor_or_null(value_meta(value_id));
  ET_CHECK_MSG(t, "Graph::tensor_dim_order(%u): not a Tensor", value_id);
  auto* d = t->dim_order();
  return d ? ::executorch::runtime::Span<const uint8_t>(d->data(), d->size())
           : ::executorch::runtime::Span<const uint8_t>{};
}

inline ::executorch::aten::TensorShapeDynamism Graph::tensor_shape_dynamism(
    uint32_t value_id) const {
  auto* t = detail::tensor_or_null(value_meta(value_id));
  ET_CHECK_MSG(t, "Graph::tensor_shape_dynamism(%u): not a Tensor", value_id);
  return static_cast<::executorch::aten::TensorShapeDynamism>(
      t->shape_dynamism());
}

inline const char* Graph::tensor_constant_data_key(uint32_t value_id) const {
  auto* t = detail::tensor_or_null(value_meta(value_id));
  if (!t)
    return nullptr;
  auto* eti = t->extra_tensor_info();
  if (!eti)
    return nullptr;
  if (eti->location() != executorch_flatbuffer::TensorDataLocation::EXTERNAL) {
    return nullptr;
  }
  auto* fqn = eti->fully_qualified_name();
  return (fqn && fqn->size() > 0) ? fqn->c_str() : nullptr;
}

inline ::executorch::runtime::Span<const uint8_t> Graph::tensor_inline_data(
    uint32_t value_id) const {
  if (!program_)
    return {};
  auto* t = detail::tensor_or_null(value_meta(value_id));
  if (!t)
    return {};
  uint32_t idx = static_cast<uint32_t>(t->data_buffer_idx());
  // Index 0 is reserved (no inline data).
  if (idx == 0)
    return {};
  auto* buffers = program_->constant_buffer();
  if (!buffers || idx >= buffers->size())
    return {};
  auto* buf = buffers->Get(idx);
  if (!buf)
    return {};
  auto* storage = buf->storage();
  if (!storage || storage->size() == 0)
    return {};
  return ::executorch::runtime::Span<const uint8_t>(
      storage->data(), storage->size());
}

inline size_t Graph::tensor_nbytes_max(uint32_t value_id) const {
  auto* t = detail::tensor_or_null(value_meta(value_id));
  if (!t || !t->sizes())
    return 0;
  size_t numel = 1;
  for (size_t i = 0; i < t->sizes()->size(); ++i) {
    int dim = t->sizes()->Get(i);
    if (dim < 0)
      return 0;
    numel *= static_cast<size_t>(dim);
  }
  auto stype = static_cast<::executorch::aten::ScalarType>(t->scalar_type());
  return numel * ::executorch::runtime::elementSize(stype);
}

inline ::executorch::runtime::Span<const int64_t> Graph::int_list_member_ids(
    uint32_t value_id) const {
  auto* val = value_meta(value_id);
  ET_CHECK_MSG(
      val && val->val_type() == executorch_flatbuffer::KernelTypes::IntList,
      "Graph::int_list_member_ids(%u): not an IntList",
      value_id);
  auto* items =
      static_cast<const executorch_flatbuffer::IntList*>(val->val())->items();
  return items
      ? ::executorch::runtime::Span<const int64_t>(items->data(), items->size())
      : ::executorch::runtime::Span<const int64_t>{};
}

inline ::executorch::runtime::Span<const int32_t> Graph::tensor_list_member_ids(
    uint32_t value_id) const {
  auto* val = value_meta(value_id);
  ET_CHECK_MSG(
      val &&
          (val->val_type() == executorch_flatbuffer::KernelTypes::TensorList ||
           val->val_type() ==
               executorch_flatbuffer::KernelTypes::OptionalTensorList),
      "Graph::tensor_list_member_ids(%u): not a TensorList or OptionalTensorList",
      value_id);
  const flatbuffers::Vector<int32_t>* items = nullptr;
  if (val->val_type() == executorch_flatbuffer::KernelTypes::TensorList) {
    items = static_cast<const executorch_flatbuffer::TensorList*>(val->val())
                ->items();
  } else {
    items = static_cast<const executorch_flatbuffer::OptionalTensorList*>(
                val->val())
                ->items();
  }
  return items
      ? ::executorch::runtime::Span<const int32_t>(items->data(), items->size())
      : ::executorch::runtime::Span<const int32_t>{};
}

} // namespace portable
} // namespace backends
} // namespace executorch
