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

#include <executorch/backends/vulkan/runtime/graph/Graph.h>

namespace at {
namespace native {
namespace vulkan {

void add_arithmetic_node(
    ComputeGraph& graph,
    const ValueRef t1,
    const ValueRef t2,
    const ValueRef out,
    const float alpha,
    const arithmetic::OpType optype);

ValueRef add_arithmetic_node(
    ComputeGraph& graph,
    const ValueRef t1,
    const ValueRef t2,
    const float alpha,
    const arithmetic::OpType optype,
    const int64_t shared_object_idx = -1);

class ArithmeticPrepack : public virtual OpNode {
 public:
  explicit ArithmeticPrepack(const ValueRef tref, const ValueRef packed);

  void encode_prepack(ComputeGraph* graph) const override;
};

class ArithmeticNode : public virtual OpNode {
 public:
  explicit ArithmeticNode(
      const ValueRef t1,
      const ValueRef t2,
      const ValueRef out,
      const float alpha,
      const arithmetic::OpType optype);

  void encode_execute(ComputeGraph* graph) const override;

 private:
  float alpha_;
  arithmetic::OpType optype_;
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
