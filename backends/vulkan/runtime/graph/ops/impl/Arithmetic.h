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

#include <executorch/backends/vulkan/runtime/graph/ops/Utils.h>

namespace at {
namespace native {
namespace vulkan {

DECLARE_OP_FN(add);
DECLARE_OP_FN(sub);
DECLARE_OP_FN(mul);
DECLARE_OP_FN(div);
DECLARE_OP_FN(floor_div);
DECLARE_OP_FN(pow);

ValueRef add_arithmetic_node(
    ComputeGraph& graph,
    const ValueRef in1,
    const ValueRef in2,
    const float alpha,
    const api::ShaderInfo& shader,
    const int64_t shared_object_idx = -1);

void add_arithmetic_node(
    ComputeGraph& graph,
    const ValueRef in1,
    const ValueRef in2,
    const ValueRef out,
    const float alpha,
    const api::ShaderInfo& shader);

struct ArithmeticParams final {
  api::utils::ivec4 outputSizes;
  api::utils::ivec4 input1Sizes;
  api::utils::ivec4 input2Sizes;
  float alpha;
};

class ArithmeticPrepack : public virtual PrepackNode {
 public:
  explicit ArithmeticPrepack(const ValueRef tref, const ValueRef packed);

  void encode(ComputeGraph* graph) const override;
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
