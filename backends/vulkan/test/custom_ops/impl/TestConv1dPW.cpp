/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>

namespace vkcompute {

void test_conv1d_pw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // args: in, weight, bias, stride, padding, dilation, groups, out
  VK_GET_OP_FN("et_vk.conv1d_pw.default")(graph, args);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.test_conv1d_pw.default, test_conv1d_pw);
}

} // namespace vkcompute
