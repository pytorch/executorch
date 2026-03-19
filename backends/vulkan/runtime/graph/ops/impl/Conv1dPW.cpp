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
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <limits>

namespace vkcompute {

// Minimum number of thread groups to target for good GPU occupancy.
static constexpr uint32_t kMinOccupancyThreads = 4096;

// Returns the tile_m (1, 2, or 4) for the conv1d_pw shader. tile_m tiles the
// L (spatial) dimension. The largest tile that produces at least
// kMinOccupancyThreads thread groups is chosen.
static uint32_t
pick_conv1d_pw_tile_m(uint32_t C_out, uint32_t L, uint32_t N_batch) {
  uint32_t n_groups = utils::div_up_4(C_out);
  for (uint32_t tile_m : {4u, 2u, 1u}) {
    uint32_t total = n_groups * utils::div_up(L, tile_m) * N_batch;
    if (total >= kMinOccupancyThreads) {
      return tile_m;
    }
  }
  return 1u;
}

// Prepack conv1d_pw weight [C_out, C_in, 1] into 4OC x 4IC blocked layout.
// This is equivalent to prepack_fp_linear_weight with N=C_out, K=C_in,
// is_transposed=true, but extracts dimensions from the conv weight shape.
static ValueRef prepack_conv1d_pw_weight(
    ComputeGraph& graph,
    const ValueRef weight_data) {
  std::vector<int64_t> weight_sizes = graph.sizes_of(weight_data);
  // weight is [C_out, C_in, 1]
  int64_t N = weight_sizes.at(0); // C_out
  int64_t K = weight_sizes.at(1); // C_in

  int64_t K4 = utils::div_up(K, int64_t(4));
  int64_t N4 = utils::div_up(N, int64_t(4));

  // Packed tensor: K4 rows, N4*4 vec4 elements per row.
  int64_t output_height = K4;
  int64_t output_width = N4 * 4 * 4;

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

  struct PackParams {
    int32_t N;
    int32_t K;
    int32_t B;
    int32_t is_transposed;
  };
  PackParams pack_params{
      utils::safe_downcast<int32_t>(N), utils::safe_downcast<int32_t>(K), 1, 1};

  std::string kernel_name = "pack_fp_linear_weight";
  add_storage_type_suffix(kernel_name, weight_storage);
  add_dtype_suffix(kernel_name, graph.dtype_of(weight_data));

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

void resize_conv1d_pw_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);

  const int64_t C_out = graph->get_int(extra_args.at(0));

  const std::vector<int64_t> in_sizes = graph->sizes_of(self);
  const int64_t N_batch = in_sizes.at(0);
  const int64_t L = in_sizes.at(2);

  graph->virtual_resize(out, {N_batch, C_out, L});
}

struct Conv1dPWIntParams final {
  int32_t weight_B;
  float output_min;
  float output_max;
};

struct Conv1dPWBiasParams final {
  float alpha;
  float beta;
  float output_min;
  float output_max;
};

vkapi::ShaderInfo pick_conv1d_pw_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef packed_weight = args.at(1).refs.at(1);
  bool has_bias = graph->get_bool(resize_args.at(1));

  // out is [N_batch, C_out, L]; in WHCN: {L, C_out, N_batch, 1}
  uint32_t C_out = graph->size_at<uint32_t>(-2, out);
  uint32_t L = graph->size_at<uint32_t>(-1, out);
  uint32_t N_batch =
      graph->dim_of(out) >= 3 ? graph->size_at<uint32_t>(-3, out) : 1;
  uint32_t tile_m = pick_conv1d_pw_tile_m(C_out, L, N_batch);

  std::string kernel_name;
  if (has_bias) {
    kernel_name = tile_m <= 1 ? "conv1d_pw_bias_tile_row_1"
        : tile_m <= 2         ? "conv1d_pw_bias_tile_row_2"
                              : "conv1d_pw_bias";
  } else {
    kernel_name = tile_m <= 1 ? "conv1d_pw_tile_row_1"
        : tile_m <= 2         ? "conv1d_pw_tile_row_2"
                              : "conv1d_pw";
  }
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
  add_storage_type_suffix(kernel_name, graph->storage_type_of(packed_weight));
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  return VK_KERNEL_FROM_STR(kernel_name);
}

utils::uvec3 pick_conv1d_pw_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);

  // out is [N_batch, C_out, L]; in WHCN: {L, C_out, N_batch, 1}
  uint32_t C_out = graph->size_at<uint32_t>(-2, out);
  uint32_t L = graph->size_at<uint32_t>(-1, out);
  uint32_t N_batch =
      graph->dim_of(out) >= 3 ? graph->size_at<uint32_t>(-3, out) : 1;
  uint32_t tile_m = pick_conv1d_pw_tile_m(C_out, L, N_batch);

  // X=OC4 (div_up_4(C_out)), Y=L/tile_m, Z=N_batch
  return {utils::div_up_4(C_out), utils::div_up(L, tile_m), N_batch};
}

void add_conv1d_pw_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef out,
    const float output_min = std::numeric_limits<float>::lowest(),
    const float output_max = std::numeric_limits<float>::max()) {
  VK_CHECK_COND(graph.packed_dim_of(in) == WHCN::kHeightDim);
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kHeightDim);

  ValueRef packed_weight = prepack_conv1d_pw_weight(graph, weight_data);

  bool has_bias = graph.val_is_not_none(bias);
  ValueRef packed_bias = kDummyValueRef;
  if (has_bias) {
    packed_bias = prepack_standard(
        graph, bias, graph.storage_type_of(out), utils::kWidthPacked);
  }

  std::vector<int64_t> out_sizes = graph.sizes_of(out);
  int64_t C_out = out_sizes.at(1);
  ValueRef C_out_ref = graph.add_scalar(C_out);
  ValueRef has_bias_ref = graph.add_scalar(has_bias);

  Conv1dPWIntParams int_params{1, output_min, output_max};
  Conv1dPWBiasParams bias_params{1.0f, 1.0f, output_min, output_max};

  std::vector<ValueRef> read_inputs = {in, packed_weight};
  if (has_bias) {
    read_inputs.push_back(packed_bias);
  }

  std::vector<PushConstantDataInfo> push_constants;
  if (has_bias) {
    push_constants.push_back(
        PushConstantDataInfo(&bias_params, sizeof(Conv1dPWBiasParams)));
  } else {
    push_constants.push_back(
        PushConstantDataInfo(&int_params, sizeof(Conv1dPWIntParams)));
  }

  vkapi::ParamsBindList shader_params = {
      graph.sizes_ubo(in), graph.sizes_ubo(out)};
  if (has_bias) {
    shader_params.append(graph.sizes_ubo(packed_bias));
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_conv1d_pw_shader,
      pick_conv1d_pw_global_wg_size,
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
      {C_out_ref, has_bias_ref},
      // Resizing Logic
      resize_conv1d_pw_node));
}

// Args: in, weight, bias, stride, padding, dilation, groups,
//       output_min, output_max, out
// output_min and output_max may be kDummyValueRef (no clamp).
void conv1d_pw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  ValueRef in = args[0];
  ValueRef weight = args[1];
  ValueRef bias = args[2];
  ValueRef out = args[9];

  const std::vector<int64_t> weight_sizes = graph.sizes_of(weight);
  VK_CHECK_COND(
      weight_sizes.at(2) == 1, "conv1d_pw only supports kernel_size=1");
  VK_CHECK_COND(
      graph.get_int(args[6]) == 1, "conv1d_pw only supports groups=1");

  float output_min = std::numeric_limits<float>::lowest();
  float output_max = std::numeric_limits<float>::max();
  if (is_valid(args[7])) {
    output_min = graph.extract_scalar<float>(args[7]);
  }
  if (is_valid(args[8])) {
    output_max = graph.extract_scalar<float>(args[8]);
  }

  add_conv1d_pw_node(graph, in, weight, bias, out, output_min, output_max);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.conv1d_pw.default, conv1d_pw);
}

} // namespace vkcompute
