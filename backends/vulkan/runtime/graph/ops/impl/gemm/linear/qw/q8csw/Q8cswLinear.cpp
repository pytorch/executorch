/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/QuantizedLinear.h>

namespace vkcompute {

void linear_q8csw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef output = args.at(idx++);

  const int64_t K = graph.size_at<int64_t>(-1, fp_input);

  QuantizationConfig input_quant_config(32, kNoQuantization, {});
  QuantizationConfig weight_quant_config(8, kPerChannel, {K});

  quantized_linear_impl(
      graph,
      input_quant_config,
      weight_quant_config,
      fp_input,
      kDummyValueRef, // input scale
      kDummyValueRef, // input zp
      weight_data,
      kDummyValueRef, // weight sums
      weight_scales_data,
      kDummyValueRef, // weight zeros
      kDummyValueRef, // group size
      bias_data,
      output);
}

// aten._weight_int8pack_mm is what the AOT weight-only int8 fusion
// (FuseQuantizedOpsTransform) emits. It carries the same operands as
// et_vk.linear_q8csw minus the bias, so it runs through the same
// implementation.
void weight_int8pack_mm(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef output = args.at(idx++);

  const int64_t K = graph.size_at<int64_t>(-1, fp_input);

  QuantizationConfig input_quant_config(32, kNoQuantization, {});
  QuantizationConfig weight_quant_config(8, kPerChannel, {K});

  quantized_linear_impl(
      graph,
      input_quant_config,
      weight_quant_config,
      fp_input,
      kDummyValueRef, // input scale
      kDummyValueRef, // input zp
      weight_data,
      kDummyValueRef, // weight sums
      weight_scales_data,
      kDummyValueRef, // weight zeros
      kDummyValueRef, // group size
      kDummyValueRef, // bias
      output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.linear_q8csw.default, linear_q8csw);
  VK_REGISTER_OP(aten._weight_int8pack_mm.default, weight_int8pack_mm);
}

} // namespace vkcompute
