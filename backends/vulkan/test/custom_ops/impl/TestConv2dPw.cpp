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

void test_conv2d_pw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // args[0] = input [N, C_in, H, W]
  // args[1] = weight [C_out, C_in, 1, 1] (constant)
  // args[2] = bias (constant, or none)
  // args[3] = impl_selector (string)
  // args[4] = output [N, C_out, H, W]
  const ValueRef input = args.at(0);
  const ValueRef weight = args.at(1);
  const ValueRef bias = args.at(2);
  const ValueRef impl_selector_str = args.at(3);
  const ValueRef out = args.at(4);

  std::string impl_selector = graph.extract_string(impl_selector_str);
  (void)impl_selector; // Reserved for future use

  // Create fixed pointwise conv parameters
  ValueRef stride = graph.add_scalar_list<int64_t>(std::vector<int64_t>{1, 1});
  ValueRef padding = graph.add_scalar_list<int64_t>(std::vector<int64_t>{0, 0});
  ValueRef dilation =
      graph.add_scalar_list<int64_t>(std::vector<int64_t>{1, 1});
  ValueRef transposed = graph.add_scalar<bool>(false);
  ValueRef output_padding =
      graph.add_scalar_list<int64_t>(std::vector<int64_t>{0, 0});
  ValueRef groups = graph.add_scalar<int64_t>(1);

  // Call aten.convolution.default with all 10 args:
  // input, weight, bias, stride, padding, dilation, transposed,
  // output_padding, groups, output
  VK_GET_OP_FN("aten.convolution.default")
  (graph,
   {input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    out});
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.test_conv2d_pw.default, test_conv2d_pw);
}

} // namespace vkcompute
