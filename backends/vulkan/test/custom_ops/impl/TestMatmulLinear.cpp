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

void test_mm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef mat1 = args.at(0);
  const ValueRef mat2 = args.at(1);
  const ValueRef impl_selector_str = args.at(2);
  const ValueRef out = args.at(3);

  std::string impl_selector = graph.extract_string(impl_selector_str);
  std::string op_name = "aten.mm." + impl_selector;

  VK_GET_OP_FN(op_name.c_str())(graph, {mat1, mat2, out});
}

void test_bmm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef mat1 = args.at(0);
  const ValueRef mat2 = args.at(1);
  const ValueRef impl_selector_str = args.at(2);
  const ValueRef out = args.at(3);

  std::string impl_selector = graph.extract_string(impl_selector_str);
  std::string op_name = "aten.bmm." + impl_selector;

  VK_GET_OP_FN(op_name.c_str())(graph, {mat1, mat2, out});
}

void test_addmm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef self = args.at(0);
  const ValueRef mat1 = args.at(1);
  const ValueRef mat2 = args.at(2);
  const ValueRef beta = args.at(3);
  const ValueRef alpha = args.at(4);
  const ValueRef impl_selector_str = args.at(5);
  const ValueRef out = args.at(6);

  std::string impl_selector = graph.extract_string(impl_selector_str);
  std::string op_name = "aten.addmm." + impl_selector;

  VK_GET_OP_FN(op_name.c_str())(graph, {self, mat1, mat2, beta, alpha, out});
}

void test_linear(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef input = args.at(0);
  const ValueRef weight = args.at(1);
  const ValueRef bias = args.at(2);
  const ValueRef impl_selector_str = args.at(3);
  const ValueRef out = args.at(4);

  std::string impl_selector = graph.extract_string(impl_selector_str);
  std::string op_name = "aten.linear." + impl_selector;

  VK_GET_OP_FN(op_name.c_str())(graph, {input, weight, bias, out});
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.test_mm.default, test_mm);
  VK_REGISTER_OP(test_etvk.test_bmm.default, test_bmm);
  VK_REGISTER_OP(test_etvk.test_addmm.default, test_addmm);
  VK_REGISTER_OP(test_etvk.test_linear.default, test_linear);
}

} // namespace vkcompute
