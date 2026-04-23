/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * CpuOpRegistry.h - CPU backend op registration macros
 *
 * Provides CPU-specific registration macros built on the generic OpRegistry.
 * CPU ops dispatch to ExecuTorch's existing portable kernel library.
 */

#include <executorch/backends/portable/runtime/OpRegistry.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <unordered_map>
#include <unordered_set>

namespace executorch {
namespace backends {
namespace portable {

// Forward declaration
class CpuRuntime;

/**
 * CpuGraph - Wrapper providing the Graph interface for CPU execution.
 *
 * Routes tensor access: boundary values use shared values_ array,
 * CPU-internal intermediates use shadow EVaules from CpuRuntime.
 */
class CpuGraph {
 public:
  CpuGraph(
      runtime::KernelRuntimeContext& ctx,
      runtime::Span<runtime::EValue> values,
      const std::unordered_set<uint32_t>* intermediate_indices = nullptr,
      std::unordered_map<uint32_t, runtime::EValue>* cpu_shadow = nullptr)
      : ctx_(ctx), values_(values), 
        intermediate_indices_(intermediate_indices), cpu_shadow_(cpu_shadow) {}

  //===--------------------------------------------------------------------===//
  // Graph Interface (required by OpView)
  //===--------------------------------------------------------------------===//

  bool val_is_tensor(ValueRef idx) const {
    return idx < values_.size() && get_value(idx).isTensor();
  }

  bool val_is_none(ValueRef idx) const {
    return idx >= values_.size() || get_value(idx).isNone();
  }

  template <typename T>
  T extract_scalar(ValueRef idx) const {
    const auto& ev = get_value(idx);
    if (ev.isInt()) {
      return static_cast<T>(ev.toInt());
    } else if (ev.isDouble()) {
      return static_cast<T>(ev.toDouble());
    } else if (ev.isBool()) {
      return static_cast<T>(ev.toBool());
    }
    return T{};
  }

  executorch::aten::ScalarType dtype_of(ValueRef idx) const {
    if (idx < values_.size() && get_value(idx).isTensor()) {
      return get_value(idx).toTensor().scalar_type();
    }
    return executorch::aten::ScalarType::Float;
  }

  std::vector<int64_t> sizes_of(ValueRef idx) const {
    if (idx < values_.size() && get_value(idx).isTensor()) {
      auto sizes = get_value(idx).toTensor().sizes();
      return std::vector<int64_t>(sizes.begin(), sizes.end());
    }
    return {};
  }

  //===--------------------------------------------------------------------===//
  // CPU-Specific Accessors
  //===--------------------------------------------------------------------===//

  runtime::KernelRuntimeContext& context() {
    return ctx_;
  }

  /// Get EValue - routes to shadow for intermediates, values_ for boundaries
  runtime::EValue& value(ValueRef idx) {
    return get_value_mut(idx);
  }

  runtime::EValue* value_ptr(ValueRef idx) {
    return &get_value_mut(idx);
  }

  size_t num_values() const {
    return values_.size();
  }

 private:
  runtime::KernelRuntimeContext& ctx_;
  runtime::Span<runtime::EValue> values_;
  const std::unordered_set<uint32_t>* intermediate_indices_;
  std::unordered_map<uint32_t, runtime::EValue>* cpu_shadow_;
  
  /// Route access: intermediate → shadow, boundary → values_
  const runtime::EValue& get_value(ValueRef idx) const {
    if (intermediate_indices_ && cpu_shadow_ && 
        intermediate_indices_->count(idx) > 0) {
      auto it = cpu_shadow_->find(idx);
      if (it != cpu_shadow_->end()) {
        return it->second;
      }
    }
    return values_[idx];
  }
  
  runtime::EValue& get_value_mut(ValueRef idx) {
    if (intermediate_indices_ && cpu_shadow_ && 
        intermediate_indices_->count(idx) > 0) {
      auto it = cpu_shadow_->find(idx);
      if (it != cpu_shadow_->end()) {
        return it->second;
      }
    }
    return values_[idx];
  }
};

/// Global CPU op registry accessor.
OperatorRegistry<CpuGraph>& cpu_op_registry();

//===----------------------------------------------------------------------===//
// CPU Registration Macros
//===----------------------------------------------------------------------===//

/// Check if op is registered.
#define CPU_HAS_OP(name) \
  ::executorch::backends::portable::cpu_op_registry().has_op(name)

/// Check if op supports dtype.
#define CPU_HAS_OP_DTYPE(name, dtype) \
  ::executorch::backends::portable::cpu_op_registry().has_op(name, dtype)

/// Get op function.
#define CPU_GET_OP_FN(name) \
  ::executorch::backends::portable::cpu_op_registry().get_op_fn(name)

/// Register an op (all dtypes).
#define CPU_REGISTER_OP(name, fn) \
  ::executorch::backends::portable::cpu_op_registry().register_op(#name, fn)

/// Register an op with specific dtypes.
#define CPU_REGISTER_OP_DTYPES(name, fn, ...) \
  ::executorch::backends::portable::cpu_op_registry().register_op( \
      #name, fn, {__VA_ARGS__})

/// Static registration block.
#define REGISTER_CPU_OPERATORS \
  REGISTER_OPERATORS(cpu)

}  // namespace portable
}  // namespace backends
}  // namespace executorch
