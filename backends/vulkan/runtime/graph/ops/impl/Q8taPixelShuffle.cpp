/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <executorch/backends/vulkan/runtime/graph/Logging.h>

#include <cmath>
#include <cstdint>

namespace vkcompute {

namespace {

void resize_q8ta_pixel_shuffle_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  // resize_args[0] is the upscale factor packed into a ValueRef (an int).
  const int32_t r = static_cast<int32_t>(resize_args.at(0));

  const std::vector<int64_t> in_sizes = graph->sizes_of(in);
  VK_CHECK_COND(in_sizes.size() == 4);
  // Input is [N, C, H, W]; output is [N, C/r/r, H*r, W*r].
  std::vector<int64_t> out_sizes = in_sizes;
  out_sizes[1] = in_sizes[1] / (r * r);
  out_sizes[2] = in_sizes[2] * r;
  out_sizes[3] = in_sizes[3] * r;
  graph->virtual_resize(out, out_sizes);
}

// Global wg picker: one thread per output int32 word. For a channels-packed
// int8x4 output with channel block size 4, the number of output int words is
// N * div_up_4(C_out) * H_out * W_out.
utils::uvec3 pick_q8ta_pixel_shuffle_global_wg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const std::vector<int64_t>& sizes = graph->sizes_of(out);
  const int64_t N = utils::val_at(-4, sizes);
  const int64_t C = utils::val_at(-3, sizes);
  const int64_t H = utils::val_at(-2, sizes);
  const int64_t W = utils::val_at(-1, sizes);
  const int64_t c_words = utils::div_up(C, int64_t(4));
  const uint32_t total_words =
      utils::safe_downcast<uint32_t>(N * c_words * H * W);
  return {total_words, 1u, 1u};
}

utils::uvec3 pick_q8ta_pixel_shuffle_local_wg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)global_workgroup_size;
  (void)args;
  (void)resize_args;
  // Linear (1D) dispatch: a flat 64-wide workgroup matches the pattern used
  // by pick_square_local_wg_with_block_config in the linear case.
  return {64u, 1u, 1u};
}

} // namespace

//
// Dispatch
//

void add_q8ta_pixel_shuffle_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef output_inv_scale,
    const ValueRef output_zp,
    const ValueRef upscale_factor,
    const ValueRef packed_int8_output) {
  // dtype must be the int8x4 packed type
  VK_CHECK_COND(graph.dtype_of(packed_int8_input) == vkapi::kInt8x4);
  VK_CHECK_COND(graph.dtype_of(packed_int8_output) == vkapi::kInt8x4);

  // Both tensors must be buffer-backed
  VK_CHECK_COND(graph.is_buffer_storage(packed_int8_input));
  VK_CHECK_COND(graph.is_buffer_storage(packed_int8_output));

  // Tensors must be 4D (N, C, H, W).
  VK_CHECK_COND(graph.dim_of(packed_int8_input) == 4);
  VK_CHECK_COND(graph.dim_of(packed_int8_output) == 4);

  // Both tensors must use a channels-packed int8x4 layout (packed_dim=C with
  // packed_dim_block_size=4). The supported layouts are PACKED_INT8_4W4C
  // (outer block on W), PACKED_INT8_4C1W (no outer block), and
  // PACKED_INT8_CONV2D. Each output thread writes one int word covering 4
  // consecutive channels at one (n, oh, ow) position.
  const api::PackedDimInfo& in_info =
      graph.packed_dim_info_of(packed_int8_input);
  const api::PackedDimInfo& out_info =
      graph.packed_dim_info_of(packed_int8_output);
  VK_CHECK_COND(in_info.packed_dim_block_size == 4);
  VK_CHECK_COND(out_info.packed_dim_block_size == 4);
  // Channels-packed only: packed_dim must be the channels axis (WHCN dim 2).
  VK_CHECK_COND(in_info.packed_dim == WHCN::kChannelsDim);
  VK_CHECK_COND(out_info.packed_dim == WHCN::kChannelsDim);

  // Upscale factor: only r=2 is exercised by the model and tests.
  const int32_t r = graph.extract_scalar<int32_t>(upscale_factor);
  VK_CHECK_COND(r == 2);

  // Validate shape relationship: out = [N, C/r/r, H*r, W*r] given in =
  // [N, C, H, W].
  const std::vector<int64_t> in_sizes = graph.sizes_of(packed_int8_input);
  const std::vector<int64_t> out_sizes = graph.sizes_of(packed_int8_output);
  VK_CHECK_COND(in_sizes[0] == out_sizes[0]);
  VK_CHECK_COND(in_sizes[1] == out_sizes[1] * r * r);
  VK_CHECK_COND(in_sizes[2] * r == out_sizes[2]);
  VK_CHECK_COND(in_sizes[3] * r == out_sizes[3]);

  // Push constants
  float scale_in = graph.extract_scalar<float>(input_scale);
  float scale_out_actual = 1.0f / graph.extract_scalar<float>(output_inv_scale);
  float inv_scale_out = graph.extract_scalar<float>(output_inv_scale);
  int32_t zp_in = graph.extract_scalar<int32_t>(input_zp);
  int32_t zp_out = graph.extract_scalar<int32_t>(output_zp);

  // Detect the pure-byte-shuffle case: same scale & same zero-point. In that
  // case the shader can skip the requantize math entirely.
  // Use a small relative tolerance on the scales.
  const float scale_diff = std::abs(scale_in - scale_out_actual);
  const float scale_thresh = 1e-7f * std::max(std::abs(scale_in), 1e-7f);
  int32_t passthrough = 0;
  if (scale_diff <= scale_thresh && zp_in == zp_out) {
    passthrough = 1;
  }
  int32_t r_val = r;

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&scale_in, sizeof(scale_in)),
      PushConstantDataInfo(&inv_scale_out, sizeof(inv_scale_out)),
      PushConstantDataInfo(&zp_in, sizeof(zp_in)),
      PushConstantDataInfo(&zp_out, sizeof(zp_out)),
      PushConstantDataInfo(&r_val, sizeof(r_val)),
      PushConstantDataInfo(&passthrough, sizeof(passthrough)),
  };

  // UBOs
  vkapi::ParamsBindList ubos;
  ubos.append(graph.buffer_meta_ubo(packed_int8_output));
  ubos.append(graph.buffer_meta_ubo(packed_int8_input));

  // Each thread writes one int32 output word (= 4 consecutive output channels
  // at one (n, oh, ow) spatial position). Total threads =
  //   N * div_up_4(C_out) * H_out * W_out
  // The custom global wg picker computes this from the output sizes; the
  // shader internally re-derives the same decomposition from
  // gl_GlobalInvocationID. The resize-args list still carries the upscale
  // factor so the resize callback can stamp the output size.
  const ValueRef r_resize_arg = static_cast<ValueRef>(r);

  std::string kernel_name = "q8ta_pixel_shuffle";

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_q8ta_pixel_shuffle_global_wg,
      pick_q8ta_pixel_shuffle_local_wg,
      // Inputs and Outputs
      {{packed_int8_output, vkapi::kWrite}, {packed_int8_input, vkapi::kRead}},
      // Shader params buffers
      ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      {graph.hashed_layout_of(packed_int8_input),
       graph.hashed_layout_of(packed_int8_output)},
      // Resize args: [upscale_factor]
      {r_resize_arg},
      // Resizing Logic
      resize_q8ta_pixel_shuffle_node));
}

//
// High level operator impl
//

void q8ta_pixel_shuffle(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef packed_int8_input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef output_inv_scale = args.at(idx++);
  const ValueRef output_zp = args.at(idx++);
  const ValueRef upscale_factor = args.at(idx++);
  const ValueRef packed_int8_output = args.at(idx);

  add_q8ta_pixel_shuffle_node(
      graph,
      packed_int8_input,
      input_scale,
      input_zp,
      output_inv_scale,
      output_zp,
      upscale_factor,
      packed_int8_output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.q8ta_pixel_shuffle.default, q8ta_pixel_shuffle);
}

} // namespace vkcompute
