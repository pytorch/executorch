/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Convolution.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

//
// Shader dispatch utilities
//

void resize_conv2d_pw_tiled_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);

  std::vector<int64_t> self_sizes = graph->sizes_of(self);
  TensorRefPtr weight_ref = graph->get_tref(extra_args.at(0));
  const auto& weight_sizes = weight_ref->sizes;

  const auto stride_list = graph->get_int_list(extra_args.at(1));
  const auto padding_list = graph->get_int_list(extra_args.at(2));

  const int64_t stride_h = stride_list->at(0);
  const int64_t stride_w = stride_list->at(1);
  const int64_t padding_h = padding_list->at(0);
  const int64_t padding_w = padding_list->at(1);

  const int64_t in_h = self_sizes.at(self_sizes.size() - 2);
  const int64_t in_w = self_sizes.at(self_sizes.size() - 1);

  // For 1x1 kernel with dilation=1: out = (in + 2*padding - 1) / stride + 1
  const int64_t out_h = (in_h + 2 * padding_h - 1) / stride_h + 1;
  const int64_t out_w = (in_w + 2 * padding_w - 1) / stride_w + 1;

  std::vector<int64_t> new_out_sizes = self_sizes;
  new_out_sizes.at(self_sizes.size() - 3) = weight_sizes.at(0);
  new_out_sizes.at(self_sizes.size() - 2) = out_h;
  new_out_sizes.at(self_sizes.size() - 1) = out_w;

  graph->virtual_resize(out, new_out_sizes);
}

vkapi::ShaderInfo pick_conv2d_pw_tiled_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);

  std::string kernel_name = "conv2d_pw_tiled";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  return VK_KERNEL_FROM_STR(kernel_name);
}

utils::uvec3 pick_conv2d_pw_tiled_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  uint32_t W = graph->size_at<uint32_t>(-1, out);
  uint32_t H = graph->size_at<uint32_t>(-2, out);
  uint32_t C_out = graph->size_at<uint32_t>(-3, out);
  uint32_t M = H * W;
  uint32_t N4 = utils::div_up_4(C_out);
  // TILE_N4=1, TILE_M=4
  return {N4, utils::div_up(M, 4u), 1};
}

//
// Prepack nodes
//

struct PackParams {
  int32_t N;
  int32_t K;
  int32_t B;
  int32_t is_transposed;
};

ValueRef prepack_conv2d_pw_weight(
    ComputeGraph& graph,
    const ValueRef weight_data) {
  const std::vector<int64_t> weight_sizes = graph.sizes_of(weight_data);
  const int64_t N = weight_sizes.at(0); // C_out
  const int64_t K = weight_sizes.at(1); // C_in
  const int64_t N4 = utils::div_up(N, int64_t(4));
  const int64_t K4 = utils::div_up(K, int64_t(4));

  const int64_t output_height = K4;
  const int64_t output_width = N4 * 4 * 4;

  utils::StorageType weight_storage = utils::kTexture2D;
  uint32_t max_extent = graph.context()->adapter_ptr()->max_texture2d_dim();
  if (output_width / 4 > max_extent ||
      static_cast<uint32_t>(output_height) > max_extent) {
    weight_storage = utils::kBuffer;
  }

  ValueRef packed_weight = graph.add_tensor(
      {output_height, output_width},
      graph.dtype_of(weight_data),
      weight_storage,
      utils::kWidthPacked);

  utils::uvec3 global_wg_size = {
      utils::safe_downcast<uint32_t>(N4),
      utils::safe_downcast<uint32_t>(K4),
      1u};

  PackParams pack_params{
      utils::safe_downcast<int32_t>(N), utils::safe_downcast<int32_t>(K), 1, 1};

  std::string pack_kernel_name = "pack_fp_linear_weight";
  add_storage_type_suffix(pack_kernel_name, weight_storage);
  add_dtype_suffix(pack_kernel_name, graph.dtype_of(weight_data));

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(pack_kernel_name),
      global_wg_size,
      graph.create_local_wg_size(global_wg_size),
      weight_data,
      packed_weight,
      {},
      {},
      {PushConstantDataInfo(&pack_params, sizeof(PackParams))}));

  return packed_weight;
}

//
// Dispatch nodes
//

void add_conv2d_pw_tiled_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef packed_weight,
    const ValueRef packed_bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef out,
    const ValueRef weight_data,
    const bool clamp_out,
    const float out_min_val,
    const float out_max_val) {
  int32_t stride_h, stride_w, padding_h, padding_w;
  {
    const auto stride_list = graph.get_int_list(stride);
    const auto padding_list = graph.get_int_list(padding);
    stride_h = utils::safe_downcast<int32_t>(stride_list->at(0));
    stride_w = utils::safe_downcast<int32_t>(stride_list->at(1));
    padding_h = utils::safe_downcast<int32_t>(padding_list->at(0));
    padding_w = utils::safe_downcast<int32_t>(padding_list->at(1));
  }

  bool s1p0 =
      stride_h == 1 && stride_w == 1 && padding_h == 0 && padding_w == 0;

  utils::ivec4 stride_padding{stride_h, stride_w, padding_h, padding_w};

  struct ClampParams final {
    float out_min;
    float out_max;
  };
  ClampParams clamp_params{out_min_val, out_max_val};

  ValueRef clamp_out_ref = graph.add_scalar(clamp_out);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_conv2d_pw_tiled_shader,
      pick_conv2d_pw_tiled_global_wg_size,
      pick_hw_square_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{in, packed_weight, packed_bias}, vkapi::kRead}},
      // Shader params buffers
      {graph.sizes_ubo(in), graph.sizes_ubo(out)},
      // Push Constants
      {PushConstantDataInfo(&stride_padding, sizeof(stride_padding)),
       PushConstantDataInfo(&clamp_params, sizeof(clamp_params))},
      // Specialization Constants
      // activation_type: 0=none, 1=relu, 2=clamp
      {s1p0 ? 1 : 0, clamp_out ? 2 : 0},
      // Resize Args
      {weight_data, stride, padding, clamp_out_ref},
      // Resizing Logic
      resize_conv2d_pw_tiled_node));
}

//
// High level operator impl
//

void conv2d_pw_impl(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef out,
    const bool transposed_val,
    const bool clamp_out,
    const float out_min_val,
    const float out_max_val) {
  ValueRef packed_weight = prepack_conv2d_pw_weight(graph, weight_data);

  ValueRef packed_bias = prepack_biases(
      graph,
      bias,
      weight_data,
      transposed_val,
      utils::kTexture2D,
      utils::kWidthPacked);

  check_conv_args(graph, in, out);

  const std::vector<int64_t> in_sizes = graph.sizes_of(in);
  if (in_sizes.at(0) > 1) {
    VK_THROW("conv2d: input batch size > 1 is not supported yet!");
  }

  add_conv2d_pw_tiled_node(
      graph,
      in,
      packed_weight,
      packed_bias,
      stride,
      padding,
      out,
      weight_data,
      clamp_out,
      out_min_val,
      out_max_val);
}

} // namespace vkcompute
