/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef USE_VULKAN_API

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <functional>
#include <unordered_map>

#define VK_HAS_OP(name) ::at::native::vulkan::operator_registry().has_op(name)

#define VK_GET_OP_FN(name) \
  ::at::native::vulkan::operator_registry().get_op_fn(name)

namespace at {
namespace native {
namespace vulkan {

/*
 * The Vulkan operator registry maps ATen operator names to their Vulkan
 * delegate function implementation. It is a simplified version of
 * executorch/runtime/kernel/operator_registry.h that uses the C++ Standard
 * Library.
 */
class OperatorRegistry final {
  using OpFunction =
      const std::function<void(ComputeGraph&, const std::vector<ValueRef>&)>;
  using OpTable = std::unordered_map<std::string, OpFunction>;

  static const OpTable kTable;

 public:
  /*
   * Check if the registry has an operator registered under the given name
   */
  bool has_op(const std::string& name);

  /*
   * Given an operator name, return the Vulkan delegate function
   */
  OpFunction& get_op_fn(const std::string& name);
};

// The Vulkan operator registry is global. It is retrieved using this function,
// where it is declared as a static local variable.
OperatorRegistry& operator_registry();

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
