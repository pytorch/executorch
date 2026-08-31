/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taPixelShuffle.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taQuantizeDequantize.h>

namespace vkcompute {

namespace {

// Map a layout name string from the test driver to the corresponding
// channels-packed int8x4 GPUMemoryLayout enum.
//
// Note: PACKED_INT8_CONV2D is a Python/serialization-level alias that the
// runtime resolves to kPackedInt8_4C1W (see VulkanBackend.cpp). At the C++
// runtime layer there is no distinct kPackedInt8_CONV2D enum, so testing
// "CONV2D" here would just be duplicate runtime work over "4C1W". We
// therefore only expose the two real enum values to the test driver.
utils::GPUMemoryLayout layout_from_string(const std::string& s) {
  if (s == "4W4C") {
    return utils::kPackedInt8_4W4C;
  } else if (s == "4C1W") {
    return utils::kPackedInt8_4C1W;
  }
  VK_THROW("Unknown q8ta layout name: " + s);
}

} // namespace

//
// Test op: takes a float input and float (re)quantization params, performs
// quantize -> fused pixel_shuffle -> dequantize, returns float output.
//
// The test op signature is:
//   test_q8ta_pixel_shuffle(
//       Tensor fp_input,
//       float input_scale,
//       int   input_zp,
//       float output_scale,
//       int   output_zp,
//       int   upscale_factor,
//       str   in_layout,    // "4W4C" | "4C1W"
//       str   out_layout,   // "4W4C" | "4C1W"
//   ) -> Tensor (float, output shape = pixel_shuffle(in, r))
//

void test_q8ta_pixel_shuffle(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef output_scale = args.at(idx++);
  const ValueRef output_zp = args.at(idx++);
  const ValueRef upscale_factor = args.at(idx++);
  const ValueRef in_layout_ref = args.at(idx++);
  const ValueRef out_layout_ref = args.at(idx++);
  const ValueRef fp_output = args.at(idx++);

  const int32_t r = graph.extract_scalar<int32_t>(upscale_factor);

  const std::string in_layout_str = graph.extract_string(in_layout_ref);
  const std::string out_layout_str = graph.extract_string(out_layout_ref);
  const utils::GPUMemoryLayout in_layout = layout_from_string(in_layout_str);
  const utils::GPUMemoryLayout out_layout = layout_from_string(out_layout_str);

  const std::vector<int64_t> in_sizes = graph.sizes_of(fp_input);
  VK_CHECK_COND(in_sizes.size() == 4);
  const int64_t N = in_sizes[0];
  const int64_t C = in_sizes[1];
  const int64_t H = in_sizes[2];
  const int64_t W = in_sizes[3];
  std::vector<int64_t> out_sizes = {N, C / (r * r), H * r, W * r};

  // Quantize fp_input to int8x4 with the channels-packed input layout.
  TmpTensor q_in(&graph, in_sizes, vkapi::kInt8x4, utils::kBuffer, in_layout);
  add_q8ta_quantize_node(graph, fp_input, input_scale, input_zp, q_in);

  // int8x4 output tensor with the channels-packed output layout.
  TmpTensor q_out(
      &graph, out_sizes, vkapi::kInt8x4, utils::kBuffer, out_layout);

  // Fused fast path. The fused kernel takes inv_scale, so compute it from
  // output_scale here.
  float output_scale_val = graph.extract_scalar<float>(output_scale);
  float output_inv_scale_val = 1.0f / output_scale_val;
  ValueRef inv_scale_ref = graph.add_scalar<double>(output_inv_scale_val);
  add_q8ta_pixel_shuffle_node(
      graph,
      q_in,
      input_scale,
      input_zp,
      inv_scale_ref,
      output_zp,
      upscale_factor,
      q_out);

  // Dequantize back to fp for correctness comparison.
  add_q8ta_dequantize_node(graph, q_out, output_scale, output_zp, fp_output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(
      test_etvk.test_q8ta_pixel_shuffle.default, test_q8ta_pixel_shuffle);
}

} // namespace vkcompute
