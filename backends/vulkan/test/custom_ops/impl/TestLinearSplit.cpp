/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

namespace vkcompute {

// et_vk.linear_dq8ca_q4gsw_split with its three outputs passed as separate
// trailing args instead of a value list.
void linear_dq8ca_q4gsw_split3_test(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  std::vector<ValueRef> op_args(args.begin(), args.end() - 3);
  op_args.push_back(
      graph.add_value_list(std::vector<ValueRef>(args.end() - 3, args.end())));
  VK_GET_OP_FN("et_vk.linear_dq8ca_q4gsw_split.default")(graph, op_args);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(
      test_etvk.linear_dq8ca_q4gsw_split3.default,
      linear_dq8ca_q4gsw_split3_test);
}

} // namespace vkcompute
