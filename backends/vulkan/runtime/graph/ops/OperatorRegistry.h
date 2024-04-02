/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <functional>
#include <unordered_map>

#define VK_HAS_OP(name) ::vkcompute::operator_registry().has_op(name)

#define VK_GET_OP_FN(name) ::vkcompute::operator_registry().get_op_fn(name)

#define VK_REGISTER_OP(name, function)          \
  ::vkcompute::operator_registry().register_op( \
      #name,                                    \
      std::bind(&function, std::placeholders::_1, std::placeholders::_2))

#define REGISTER_OPERATORS                              \
  static void register_ops();                           \
  static const OperatorRegisterInit reg(&register_ops); \
  static void register_ops()

namespace vkcompute {

/*
 * The Vulkan operator registry maps ATen operator names
 * to their Vulkan delegate function implementation. It is
 * a simplified version of
 * executorch/runtime/kernel/operator_registry.h that uses
 * the C++ Standard Library.
 */
class OperatorRegistry final {
  using OpFunction =
      const std::function<void(ComputeGraph&, const std::vector<ValueRef>&)>;
  using OpTable = std::unordered_map<std::string, OpFunction>;

  OpTable table_;

 public:
  /*
   * Check if the registry has an operator registered under the given name
   */
  bool has_op(const std::string& name);

  /*
   * Given an operator name, return the Vulkan delegate function
   */
  OpFunction& get_op_fn(const std::string& name);

  /*
   * Register a function to a given operator name
   */
  void register_op(const std::string& name, OpFunction& fn);
};

class OperatorRegisterInit final {
  using InitFn = void();

 public:
  explicit OperatorRegisterInit(InitFn* init_fn) {
    init_fn();
  }
};

// The Vulkan operator registry is global. It is retrieved using this function,
// where it is declared as a static local variable.
OperatorRegistry& operator_registry();

} // namespace vkcompute
