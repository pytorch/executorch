/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/GemmCommon.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

ValueRef prepack_fp_linear_weight(
    ComputeGraph& graph,
    const ValueRef weight_data,
    bool is_transposed,
    int64_t B,
    bool force_buffer) {
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

  utils::StorageType weight_storage;
  if (force_buffer) {
    weight_storage = utils::kBuffer;
  } else {
    weight_storage = utils::kTexture2D;
    uint32_t max_extent = graph.context()->adapter_ptr()->max_texture2d_dim();
    // output_width is in scalars; texture width in texels = output_width / 4
    if (output_width / 4 > max_extent ||
        static_cast<uint32_t>(output_height) > max_extent) {
      weight_storage = utils::kBuffer;
    }
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

void resize_matmul_tiled_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef mat1 = args.at(1).refs.at(0);
  const ValueRef mat2 = args.at(1).refs.at(1);

  const std::vector<int64_t> mat1_sizes = graph->sizes_of(mat1);
  const std::vector<int64_t> mat2_sizes = graph->sizes_of(mat2);

  std::vector<int64_t> new_out_sizes(mat1_sizes);
  new_out_sizes.at(new_out_sizes.size() - 1) = mat2_sizes.back();
  new_out_sizes.at(new_out_sizes.size() - 2) =
      mat1_sizes.at(mat1_sizes.size() - 2);

  graph->virtual_resize(out, new_out_sizes);
}

} // namespace vkcompute
