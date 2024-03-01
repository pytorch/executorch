/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/impl/Arithmetic.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace at {
namespace native {
namespace vulkan {

#define DECLARE_ARITHMETIC_FN(function) \
  ValueRef function(ComputeGraph& graph, const std::vector<ValueRef>& args);

DECLARE_ARITHMETIC_FN(add);
DECLARE_ARITHMETIC_FN(sub);
DECLARE_ARITHMETIC_FN(mul);
DECLARE_ARITHMETIC_FN(div);
DECLARE_ARITHMETIC_FN(floor_div);
DECLARE_ARITHMETIC_FN(pow);

ValueRef add_arithmetic_node(
    ComputeGraph& graph,
    const ValueRef t1,
    const ValueRef t2,
    const float alpha,
    const arithmetic::OpType optype,
    const int64_t shared_object_idx = -1);

void add_arithmetic_node(
    ComputeGraph& graph,
    const ValueRef t1,
    const ValueRef t2,
    const ValueRef out,
    const float alpha,
    const arithmetic::OpType optype);

class ArithmeticPrepack : public virtual PrepackNode {
 public:
  explicit ArithmeticPrepack(const ValueRef tref, const ValueRef packed);

  void encode(ComputeGraph* graph) const override;
};

class ArithmeticNode : public virtual ExecuteNode {
 public:
  explicit ArithmeticNode(
      const ValueRef t1,
      const ValueRef t2,
      const ValueRef out,
      const float alpha,
      const arithmetic::OpType optype);

  void encode(ComputeGraph* graph) const override;

 private:
  float alpha_;
  arithmetic::OpType optype_;
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
