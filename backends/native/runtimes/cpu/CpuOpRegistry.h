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
#include <executorch/runtime/kernel/operator_registry.h>
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
      : ctx_(ctx),
        values_(values),
        intermediate_indices_(intermediate_indices),
        cpu_shadow_(cpu_shadow) {}

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

/// Wrapper around OperatorRegistry<CpuGraph> that adds default-handler
/// semantics for the catch-all dispatch path.
///
/// CpuOps.cpp registers explicit handlers for ops that need bespoke
/// dispatch (e.g., aten::copy_ in-place semantics, aten::X_ in-place
/// → .out kernel rewrite). Everything else falls through to the
/// default handler — typically a generic dispatcher that computes the
/// ET kernel name from the op name (e.g., aten::X → aten::X.out) and
/// invokes ET's portable kernel registry directly.
///
/// has_op() is truthful: for unregistered names, it consults ET's
/// kernel registry via the supplied name_resolver to verify the kernel
/// the default dispatcher would call actually exists. Selective-build
/// trims propagate cleanly to routing decisions.
class CpuOpDispatcher final {
 public:
  using OpFn = OpFunction<CpuGraph>;
  using DefaultFn = std::function<void(
      CpuGraph&,
      const std::vector<ValueRef>&,
      const std::string& op_name)>;
  using NameResolver =
      std::function<std::string(const std::string& op_name)>;

  /// Register an explicit handler. Overrides the default for `name`.
  void register_op(
      const std::string& name,
      OpFn fn,
      std::initializer_list<executorch::aten::ScalarType> dtypes = {}) {
    underlying_.register_op(name, std::move(fn), dtypes);
  }

  /// Install a default dispatcher + name resolver. The dispatcher is
  /// called for any op not explicitly registered. The resolver maps
  /// op name → ET kernel name and is consulted by has_op() to verify
  /// kernel availability.
  void set_default_handler(DefaultFn dispatcher, NameResolver resolver) {
    default_dispatcher_ = std::move(dispatcher);
    default_resolver_ = std::move(resolver);
  }

  /// Returns true if the op is explicitly registered, OR a default
  /// handler is installed AND ET has the resolved kernel, OR ANY
  /// explicitly-registered overload's full_name starts with `name + "."`
  /// (so router's base-name can_run probe finds explicit overload-keyed
  /// registrations).
  bool has_op(const std::string& name) const {
    if (underlying_.has_op(name)) {
      return true;
    }
    // Check for any overload-prefixed registration (e.g., name="aten::add_"
    // matches registered "aten::add_.Tensor"). Cheap linear scan.
    const std::string prefix = name + ".";
    for (const auto& kv : underlying_.table()) {
      if (kv.first.compare(0, prefix.size(), prefix) == 0) return true;
    }
    if (!default_dispatcher_ || !default_resolver_) {
      return false;
    }
    std::string kernel_name = default_resolver_(name);
    return torch::executor::hasOpsFn(
        kernel_name.c_str(),
        ::executorch::runtime::ArrayRef<
            ::executorch::runtime::TensorMeta>());
  }

  /// Dtype-aware overload; for now defers to underlying for explicitly
  /// registered ops and ignores dtype for default-handled ops (ET's
  /// kernel registry does its own dtype matching).
  bool has_op(const std::string& name,
              executorch::aten::ScalarType dtype) const {
    if (underlying_.has_op(name, dtype)) {
      return true;
    }
    return has_op(name);
  }

  /// Returns the dispatch function. Empty std::function (operator bool
  /// is false) when no explicit registration AND no default handler.
  /// For default-handled ops, returns a closure that binds the op_name
  /// into the default dispatcher.
  OpFn try_get_op_fn(const std::string& name) const {
    OpFn* registered = const_cast<OperatorRegistry<CpuGraph>&>(underlying_)
                           .try_get_op_fn(name);
    if (registered) {
      return *registered;
    }
    if (!default_dispatcher_) {
      return OpFn{};
    }
    auto dispatcher = default_dispatcher_;
    std::string captured = name;
    return [dispatcher, captured](
               CpuGraph& g, const std::vector<ValueRef>& args) {
      dispatcher(g, args, captured);
    };
  }

 private:
  OperatorRegistry<CpuGraph> underlying_;
  DefaultFn default_dispatcher_;
  NameResolver default_resolver_;
};

/// Global CPU op dispatcher accessor.
CpuOpDispatcher& cpu_op_registry();

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
#define CPU_REGISTER_OP_DTYPES(name, fn, ...)                      \
  ::executorch::backends::portable::cpu_op_registry().register_op( \
      #name, fn, {__VA_ARGS__})

/// Static registration block.
#define REGISTER_CPU_OPERATORS REGISTER_OPERATORS(cpu)

} // namespace portable
} // namespace backends
} // namespace executorch
