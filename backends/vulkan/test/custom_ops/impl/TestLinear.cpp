/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/MatMul.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

// Implementation selector values:
// 0 = default (use standard aten.linear.default dispatch)
// 1 = experimental tiled linear implementation

void test_fp_linear(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef input = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef impl_selector_ref = args.at(idx++);
  const ValueRef output = args.at(idx++);

  // Extract the impl_selector flag
  int32_t impl_selector = graph.extract_scalar<int32_t>(impl_selector_ref);

  if (impl_selector == 0) {
    // Use standard linear operator dispatch
    std::vector<ValueRef> linear_args = {input, weight_data, bias_data, output};
    VK_GET_OP_FN("aten.linear.default")(graph, linear_args);
  } else if (impl_selector == 1) {
    // Use experimental tiled linear implementation
    std::vector<ValueRef> linear_args = {input, weight_data, bias_data, output};
    VK_GET_OP_FN("etvk.linear_nv_cm2.default")(graph, linear_args);
  } else {
    VK_THROW("Invalid impl_selector value: ", impl_selector);
  }
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.test_fp_linear.default, test_fp_linear);
}

} // namespace vkcompute
