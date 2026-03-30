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
  const ValueRef input = args.at(0);
  const ValueRef weight = args.at(1);
  const ValueRef bias = args.at(2);
  const ValueRef stride = args.at(3);
  const ValueRef padding = args.at(4);
  const ValueRef dilation = args.at(5);
  const ValueRef groups = args.at(6);
  const ValueRef out = args.at(7);

  // conv1d_pw expects: in, weight, bias, stride, padding, dilation, groups,
  //                    output_min, output_max, out
  VK_GET_OP_FN("et_vk.conv1d_pw.default")
  (graph,
   {input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    kDummyValueRef,
    kDummyValueRef,
    out});
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.test_conv1d_pw.default, test_conv1d_pw);
}

} // namespace vkcompute
