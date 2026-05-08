/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * OpRegistry.h - Generic op registration machinery for portable backends
 *
 * This provides a Vulkan-style registration pattern that any backend
 * (CPU, Metal, Vulkan) can use. Each backend:
 * 1. Defines its own Graph type (e.g., MetalGraph, VulkanComputeGraph)
 * 2. Uses OperatorRegistry<Graph> for registration
 * 3. Defines macros like MTL_REGISTER_OP using OperatorRegisterInit
 *
 * Example usage for Metal:
 *
 *   // In MetalOpRegistry.h
 *   namespace metal {
 *   OperatorRegistry<MetalGraph>& operator_registry();
 *   }
 *
 *   #define MTL_REGISTER_OP(name, fn) \
 *       ::metal::operator_registry().register_op(#name, fn)
 *
 *   // In BinaryOp.mm
 *   REGISTER_MTL_OPERATORS {
 *       MTL_REGISTER_OP(aten.add.Tensor, add);
 *       MTL_REGISTER_OP(aten.mul.Tensor, mul);
 *   }
 */

#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <functional>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>

namespace executorch {
namespace backends {
namespace portable {

/// ValueRef is an index into the graph's values array.
using ValueRef = uint32_t;
constexpr ValueRef kInvalidValueRef = UINT32_MAX;

/**
 * Op function signature - generic for all backends.
 *
 * @tparam Graph  The backend's graph type (e.g., MetalGraph, CpuGraph)
 *
 * The function receives:
 * - graph: Reference to the backend's graph object for encoding work
 * - args:  Vector of value indices in schema order
 *
 * The function should:
 * - Extract input/output tensors using args indices
 * - Encode the operation into the graph's command buffer/pipeline
 */
template <typename Graph>
using OpFunction = std::function<void(Graph&, const std::vector<ValueRef>&)>;

/**
 * Registration entry with optional dtype constraints.
 */
template <typename Graph>
struct OpRegistration {
  OpFunction<Graph> fn;
  std::vector<executorch::aten::ScalarType> supported_dtypes;  // Empty = all
};

/**
 * Global registry for a backend's ops.
 *
 * @tparam Graph  The backend's graph type
 *
 * Thread-safety: Registration happens at static init time (single-threaded).
 * Lookup happens at runtime (read-only, thread-safe).
 */
template <typename Graph>
class OperatorRegistry final {
 public:
  /**
   * Register an op with optional dtype constraints.
   *
   * @param name    Op name (e.g., "aten::add.Tensor")
   * @param fn      Implementation function
   * @param dtypes  Supported dtypes (empty = all supported)
   */
  void register_op(
      const std::string& name,
      OpFunction<Graph> fn,
      std::initializer_list<executorch::aten::ScalarType> dtypes = {}) {
    table_[name] = {std::move(fn), std::vector<executorch::aten::ScalarType>(dtypes)};
  }

  /**
   * Check if an op is registered (any dtype).
   */
  bool has_op(const std::string& name) const {
    return table_.find(name) != table_.end();
  }

  /**
   * Check if an op supports a specific dtype.
   */
  bool has_op(const std::string& name, executorch::aten::ScalarType dtype) const {
    auto it = table_.find(name);
    if (it == table_.end()) {
      return false;
    }

    const auto& dtypes = it->second.supported_dtypes;
    if (dtypes.empty()) {
      return true;  // No constraints = all supported
    }

    for (auto d : dtypes) {
      if (d == dtype) {
        return true;
      }
    }
    return false;
  }

  /**
   * Get the implementation function for an op.
   *
   * @throws std::out_of_range if op not registered
   */
  OpFunction<Graph>& get_op_fn(const std::string& name) {
    return table_.at(name).fn;
  }

  /**
   * Get op function if registered, nullptr otherwise.
   */
  OpFunction<Graph>* try_get_op_fn(const std::string& name) {
    auto it = table_.find(name);
    return it != table_.end() ? &it->second.fn : nullptr;
  }

  /**
   * Read-only access to the registration table (used by wrappers that
   * want to scan for prefix matches, e.g., CpuOpDispatcher's base-name
   * has_op probe).
   */
  const std::unordered_map<std::string, OpRegistration<Graph>>& table() const {
    return table_;
  }

 private:
  std::unordered_map<std::string, OpRegistration<Graph>> table_;
};

/**
 * Helper for static initialization of op registrations.
 *
 * Usage:
 *   static void register_ops() {
 *       REGISTER_OP(...);
 *   }
 *   static const OperatorRegisterInit _init(&register_ops);
 */
class OperatorRegisterInit final {
 public:
  explicit OperatorRegisterInit(void (*init_fn)()) {
    init_fn();
  }
};

/**
 * Macro for defining a static registration block.
 *
 * Usage:
 *   REGISTER_OPERATORS(my_backend) {
 *       MY_BACKEND_REGISTER_OP(aten.add.Tensor, add_impl);
 *   }
 */
#define REGISTER_OPERATORS(backend_name) \
  static void _register_##backend_name##_ops(); \
  static const ::executorch::backends::portable::OperatorRegisterInit \
      _##backend_name##_reg(&_register_##backend_name##_ops); \
  static void _register_##backend_name##_ops()

}  // namespace portable
}  // namespace backends
}  // namespace executorch
