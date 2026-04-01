/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Linear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/MatMul.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

ValueRef prepack_fp_linear_weight(
    ComputeGraph& graph,
    const ValueRef weight_data,
    bool is_transposed,
    int64_t B) {
  std::vector<int64_t> weight_sizes = graph.sizes_of(weight_data);

  int64_t N, K;
  if (is_transposed) {
    // Source is [N, K] or [B, N, K]
    N = weight_sizes.at(weight_sizes.size() - 2);
    K = weight_sizes.at(weight_sizes.size() - 1);
  } else {
    // Source is [K, N] or [B, K, N]
    K = weight_sizes.at(weight_sizes.size() - 2);
    N = weight_sizes.at(weight_sizes.size() - 1);
  }

  int64_t K4 = utils::div_up(K, int64_t(4));
  int64_t N4 = utils::div_up(N, int64_t(4));

  // Packed tensor: B*K4 rows, N4*4 vec4 elements per row (batch-stacked).
  // Since the tensor size is in scalars and kWidthPacked packs 4 scalars per
  // texel, we need width = N4*4*4 scalars to get N4*4 texels.
  int64_t output_height = B * K4;
  int64_t output_width = N4 * 4 * 4;

  utils::StorageType weight_storage = utils::kTexture2D;
  uint32_t max_extent = graph.context()->adapter_ptr()->max_texture2d_dim();
  // output_width is in scalars; texture width in texels = output_width / 4
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
      utils::safe_downcast<uint32_t>(B)};

  struct PackParams {
    int32_t N;
    int32_t K;
    int32_t B;
    int32_t is_transposed;
  };
  PackParams pack_params{
      utils::safe_downcast<int32_t>(N),
      utils::safe_downcast<int32_t>(K),
      utils::safe_downcast<int32_t>(B),
      is_transposed ? 1 : 0};

  std::string kernel_name = "pack_fp_linear_weight";
  add_storage_type_suffix(kernel_name, weight_storage);
  add_dtype_suffix(kernel_name, graph.dtype_of(weight_data));
  add_dtype_suffix(kernel_name, graph.get_staging_dtype_for(weight_data));

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      graph.create_local_wg_size(global_wg_size),
      weight_data,
      packed_weight,
      {},
      {},
      {PushConstantDataInfo(&pack_params, sizeof(PackParams))}));

  return packed_weight;
}

void resize_linear_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef mat1 = args.at(1).refs.at(0);

  const std::vector<int64_t> mat1_sizes = graph->sizes_of(mat1);

  int64_t M = mat1_sizes.at(mat1_sizes.size() - 2);
  int64_t N = graph->get_int(resize_args.at(0));

  if (mat1_sizes.size() >= 3) {
    int64_t B = mat1_sizes.at(0);
    graph->virtual_resize(out, {B, M, N});
  } else {
    graph->virtual_resize(out, {M, N});
  }
}

struct LinearIntParams final {
  int32_t weight_B;
};

struct LinearBiasParams final {
  float alpha;
  float beta;
};

vkapi::ShaderInfo pick_linear_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef input = args.at(1).refs.at(0);
  const ValueRef packed_weight = args.at(1).refs.at(1);
  bool has_bias = graph->get_bool(resize_args.at(1));
  uint32_t tile_m = pick_matmul_tile_m(graph, out);

  bool is_buffer = graph->storage_type_of(out) == utils::kBuffer;
  // Use vec4 shader when all tensor widths are aligned to 4, even for buffers
  uint32_t K = graph->size_at<uint32_t>(-1, input);
  uint32_t N = graph->size_at<uint32_t>(-1, out);
  bool use_scalar = is_buffer && (K % 4 != 0 || N % 4 != 0);
  std::string base = use_scalar ? "linear" : "linear_vec";

  std::string kernel_name;
  if (has_bias) {
    kernel_name = tile_m <= 1 ? base + "_bias_tile_row_1"
        : tile_m <= 2         ? base + "_bias_tile_row_2"
                              : base + "_bias";
  } else {
    kernel_name = tile_m <= 1 ? base + "_tile_row_1"
        : tile_m <= 2         ? base + "_tile_row_2"
                              : base;
  }
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
  add_storage_type_suffix(kernel_name, graph->storage_type_of(packed_weight));
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  return VK_KERNEL_FROM_STR(kernel_name);
}

utils::uvec3 pick_linear_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  uint32_t N = graph->size_at<uint32_t>(-1, out);
  uint32_t M = graph->size_at<uint32_t>(-2, out);
  uint32_t B = graph->dim_of(out) >= 3 ? graph->size_at<uint32_t>(-3, out) : 1;
  uint32_t tile_m = pick_matmul_tile_m(graph, out);
  return {utils::div_up_4(N), utils::div_up(M, tile_m), B};
}

void add_linear_tiled_node(
    ComputeGraph& graph,
    const ValueRef input,
    const ValueRef packed_weight,
    const ValueRef packed_bias,
    bool has_bias,
    const ValueRef out,
    int32_t weight_B,
    float alpha,
    float beta) {
  VK_CHECK_COND(graph.packed_dim_of(input) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kWidthDim);
  std::vector<int64_t> out_sizes = graph.sizes_of(out);
  int32_t orig_N = utils::safe_downcast<int32_t>(out_sizes.back());

  LinearIntParams int_params{weight_B};
  LinearBiasParams bias_params{alpha, beta};
  ValueRef has_bias_ref = graph.add_scalar(has_bias);
  ValueRef orig_N_ref = graph.add_scalar(static_cast<int64_t>(orig_N));

  std::vector<ValueRef> read_inputs = {input, packed_weight};
  if (has_bias) {
    read_inputs.push_back(packed_bias);
  }

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&int_params, sizeof(LinearIntParams)),
  };
  if (has_bias) {
    push_constants.push_back(
        PushConstantDataInfo(&bias_params, sizeof(LinearBiasParams)));
  }

  vkapi::ParamsBindList shader_params = {
      graph.sizes_ubo(input), graph.sizes_ubo(out)};
  if (has_bias) {
    shader_params.append(graph.sizes_ubo(packed_bias));
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_linear_shader,
      pick_linear_global_wg_size,
      pick_hw_square_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {read_inputs, vkapi::kRead}},
      // Shader params buffers
      shader_params,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize Args
      {orig_N_ref, has_bias_ref},
      // Resizing Logic
      resize_linear_node));
}

void linear_packed_weight(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  ValueRef input = args.at(0);
  ValueRef weight_data = args.at(1);
  ValueRef bias = args.at(2);
  ValueRef out = args.at(3);

  ValueRef packed_weight = prepack_fp_linear_weight(
      graph, weight_data, /*is_transposed=*/true, /*B=*/1);

  ValueRef packed_bias = kDummyValueRef;
  bool has_bias = graph.val_is_not_none(bias);
  if (has_bias) {
    packed_bias = prepack_standard(
        graph, bias, graph.storage_type_of(out), utils::kWidthPacked);
  }

  add_linear_tiled_node(
      graph, input, packed_weight, packed_bias, has_bias, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.linear.default, linear_packed_weight);
}

} // namespace vkcompute
