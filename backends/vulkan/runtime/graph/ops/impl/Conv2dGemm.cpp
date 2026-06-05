/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Conv2dGemm.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Conv2dIm2Col.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Convolution.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <optional>

namespace vkcompute {

namespace {

//
// Weight handling
//

// Prepack the ORIGINAL serialized conv2d weight [C_out, C_in, K_h, K_w]
// directly on the GPU into the 4OC x 4IC blocked layout that conv2d_gemm.glsl
// loads via load_packed_weight_tile_with_checks. The serialized weight data is
// read as-is (never CPU-repacked); pack_conv2d_gemm_weight.glsl performs the
// im2col K-axis reorder (k = (ki * K_w + kj) * Cin_padded + ci, ci-padding
// lanes zeroed) and the 4x4 transpose in one pass.
//
// The packed output is byte-identical to the layout the generic
// prepack_fp_linear_weight (is_transposed=1) produced over a CPU-flattened
// [C_out, K_total] weight, so conv2d_gemm.glsl is unchanged.
ValueRef prepack_conv2d_gemm_weight(
    ComputeGraph& graph,
    const ValueRef weight_data) {
  const std::vector<int64_t> w_sizes = graph.sizes_of(weight_data);
  VK_CHECK_COND(w_sizes.size() == 4);
  const int64_t C_out = w_sizes[0];
  const int64_t C_in = w_sizes[1];
  const int64_t K_h = w_sizes[2];
  const int64_t K_w = w_sizes[3];

  const int64_t Cin_padded = utils::align_up_4(C_in);
  const int64_t K_total = K_h * K_w * Cin_padded;

  const int64_t N = C_out;
  const int64_t K = K_total;
  const int64_t N4 = utils::div_up(N, int64_t(4));
  const int64_t K4 = utils::div_up(K, int64_t(4));

  // Packed tensor: K4 rows, N4*4 vec4 elements per row (4OC x 4IC blocks).
  // kWidthPacked packs 4 scalars per texel, so width = N4*4*4 scalars.
  const int64_t output_height = K4;
  const int64_t output_width = N4 * 4 * 4;

  utils::StorageType weight_storage = utils::kTexture2D;
  const uint32_t max_extent =
      graph.context()->adapter_ptr()->max_texture2d_dim();
  if (output_width / 4 > max_extent ||
      utils::safe_downcast<uint32_t>(output_height) > max_extent) {
    weight_storage = utils::kBuffer;
  }

  ValueRef packed_weight = graph.add_tensor(
      {output_height, output_width},
      graph.dtype_of(weight_data),
      weight_storage,
      utils::kWidthPacked);

  const utils::uvec3 global_wg_size = {
      utils::safe_downcast<uint32_t>(N4),
      utils::safe_downcast<uint32_t>(K4),
      1u};

  // Push constants must be uploaded in <= 16-byte (one ivec4) chunks; the
  // shader's Block reads them back as dims0 / dims1. Layout must match
  // pack_conv2d_gemm_weight.glsl.
  const utils::ivec4 dims0{
      utils::safe_downcast<int32_t>(N),
      utils::safe_downcast<int32_t>(K),
      utils::safe_downcast<int32_t>(C_in),
      utils::safe_downcast<int32_t>(Cin_padded)};
  const utils::ivec4 dims1{
      utils::safe_downcast<int32_t>(K_h),
      utils::safe_downcast<int32_t>(K_w),
      0,
      0};

  std::string kernel_name = "pack_conv2d_gemm_weight";
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
      {PushConstantDataInfo(&dims0, sizeof(dims0)),
       PushConstantDataInfo(&dims1, sizeof(dims1))}));

  return packed_weight;
}

//
// GEMM dispatch
//

vkapi::ShaderInfo pick_conv2d_gemm_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  // The im2col tensor's storage selects the input-load codegen variant of
  // conv2d_gemm: texture2d vs buffer.
  const ValueRef im2col_in = args.at(1).refs.at(0);

  std::string kernel_name = "conv2d_gemm";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph->storage_type_of(im2col_in));
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  return VK_KERNEL_FROM_STR(kernel_name);
}

utils::uvec3 pick_conv2d_gemm_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const uint32_t W = graph->size_at<uint32_t>(-1, out);
  const uint32_t H = graph->size_at<uint32_t>(-2, out);
  const uint32_t C_out = graph->size_at<uint32_t>(-3, out);
  const uint32_t M = H * W;
  const uint32_t N4 = utils::div_up_4(C_out);
  // TILE_N4=1, TILE_M=4
  return {N4, utils::div_up(M, 4u), 1};
}

// Output sizes are determined by the conv shape (im2col tensor's spatial
// extents match the conv output), so the GEMM shader doesn't need to resize
// the output tensor — it's already set by the caller.  We still need a noop
// resize because the dispatch infra expects one.
void resize_conv2d_gemm_node(
    ComputeGraph* /*graph*/,
    const std::vector<ArgGroup>& /*args*/,
    const std::vector<ValueRef>& /*extra_args*/) {
  // no-op
}

void add_conv2d_gemm_node(
    ComputeGraph& graph,
    const ValueRef im2col_in,
    const ValueRef packed_weight,
    const ValueRef packed_bias,
    const ValueRef out,
    const int32_t K_total,
    const int32_t M_total,
    const bool clamp_out,
    const float out_min_val,
    const float out_max_val) {
  const int32_t K4_total = K_total / 4;

  const utils::ivec4 gemm_dims{K_total, K4_total, M_total, 0};
  const utils::vec4 clamp_vals{out_min_val, out_max_val, 0.0f, 0.0f};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_conv2d_gemm_shader,
      pick_conv2d_gemm_global_wg_size,
      pick_hw_square_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite},
       {{im2col_in, packed_weight, packed_bias}, vkapi::kRead}},
      // Shader params buffers
      {graph.sizes_ubo(out)},
      // Push constants (2 × 16 bytes)
      {PushConstantDataInfo(&gemm_dims, sizeof(gemm_dims)),
       PushConstantDataInfo(&clamp_vals, sizeof(clamp_vals))},
      // Specialization constants
      // activation_type: 0=none, 1=relu, 2=clamp
      {clamp_out ? 2 : 0},
      // Resize args
      {},
      // Resizing logic
      resize_conv2d_gemm_node));
}

} // namespace

//
// Orchestration
//

void conv2d_gemm_impl(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef out,
    const bool clamp_out,
    const float out_min_val,
    const float out_max_val,
    const std::optional<utils::StorageType> im2col_storage_override) {
  const std::vector<int64_t> in_sizes = graph.sizes_of(in);
  const std::vector<int64_t> w_sizes = graph.sizes_of(weight_data);
  const std::vector<int64_t> out_sizes = graph.sizes_of(out);
  VK_CHECK_COND(in_sizes.size() == 4 && in_sizes[0] == 1);
  VK_CHECK_COND(w_sizes.size() == 4);

  const int64_t C_in = w_sizes[1];
  const int64_t K_h = w_sizes[2];
  const int64_t K_w = w_sizes[3];
  const int64_t H_out = out_sizes[2];
  const int64_t W_out = out_sizes[3];

  const int64_t Cin_padded = utils::align_up_4(C_in);
  const int64_t K_total = K_h * K_w * Cin_padded;
  // Cin_padded is align_up_4(C_in), so K_total is a multiple of 4 and the
  // K4_total = K_total / 4 division below is exact.
  VK_CHECK_COND(K_total % 4 == 0);

  // Extract scalar conv params, scoping the IntListPtrs so they don't keep
  // active value pointers around while we mutate the graph below.
  int32_t stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w;
  {
    const auto stride_list = graph.get_int_list(stride);
    const auto padding_list = graph.get_int_list(padding);
    const auto dilation_list = graph.get_int_list(dilation);
    stride_h = utils::safe_downcast<int32_t>(stride_list->at(0));
    stride_w = utils::safe_downcast<int32_t>(stride_list->at(1));
    padding_h = utils::safe_downcast<int32_t>(padding_list->at(0));
    padding_w = utils::safe_downcast<int32_t>(padding_list->at(1));
    dilation_h = utils::safe_downcast<int32_t>(dilation_list->at(0));
    dilation_w = utils::safe_downcast<int32_t>(dilation_list->at(1));
  }

  const int64_t M = H_out * W_out;
  const int64_t K4_total = K_total / 4;

  // Pick im2col storage. When an explicit override is provided (test-only),
  // honor it and skip auto-selection. Otherwise run the production
  // auto-selection per device:
  //   - Mali: always buffer (texture sampling on Mali is comparatively slow).
  //   - Others: prefer texture2d (M × K4_total). If that doesn't fit the
  //     device's max texture2d dim, fall back to texture3d laid out as
  //     (W_out, H_out, K4_total). Buffer is the last-resort fallback.
  utils::StorageType im2col_storage;
  if (im2col_storage_override.has_value()) {
    im2col_storage = im2col_storage_override.value();
    VK_CHECK_COND(
        im2col_storage == utils::kBuffer ||
        im2col_storage == utils::kTexture2D ||
        im2col_storage == utils::kTexture3D);
  } else if (graph.device_is_mali()) {
    im2col_storage = utils::kBuffer;
  } else {
    const uint32_t max_2d = graph.context()->adapter_ptr()->max_texture2d_dim();
    const uint32_t max_3d = graph.context()->adapter_ptr()->max_texture3d_dim();
    const bool fits_2d = utils::safe_downcast<uint32_t>(K4_total) <= max_2d &&
        utils::safe_downcast<uint32_t>(M) <= max_2d;
    const bool fits_3d = utils::safe_downcast<uint32_t>(W_out) <= max_3d &&
        utils::safe_downcast<uint32_t>(H_out) <= max_3d &&
        utils::safe_downcast<uint32_t>(K4_total) <= max_3d;
    if (fits_2d) {
      im2col_storage = utils::kTexture2D;
    } else if (fits_3d) {
      im2col_storage = utils::kTexture3D;
    } else {
      im2col_storage = utils::kBuffer;
    }
  }

  // Allocate the im2col intermediate as a scoped scratch tensor. The im2col
  // value is produced by the im2col node and consumed immediately by the GEMM
  // node, both below, and is dead afterwards. Using a TmpTensor lets the memory
  // planner alias one backing buffer across the (non-overlapping) im2col
  // lifetimes of every conv2d layer, so peak memory tracks the largest single
  // im2col rather than the sum of all of them. The TmpTensor must outlive
  // add_conv2d_gemm_node (its last consumer), so it lives to the end of this
  // function.
  //
  // The 2D and buffer variants use a flat [M, K_total] kWidthPacked shape; the
  // texture3d variant uses the natural [1, K_total, H_out, W_out]
  // kChannelsPacked shape so K4 lays along Z. Hoist the per-storage differences
  // into locals so the TmpTensor is constructed exactly once and never needs to
  // be copied or moved.
  std::vector<int64_t> im2col_sizes;
  utils::StorageType im2col_tmp_storage;
  utils::GPUMemoryLayout im2col_layout;
  if (im2col_storage == utils::kTexture3D) {
    im2col_sizes = {1, K_total, H_out, W_out};
    im2col_tmp_storage = utils::kTexture3D;
    im2col_layout = utils::kChannelsPacked;
  } else {
    im2col_sizes = {M, K_total};
    im2col_tmp_storage = im2col_storage;
    im2col_layout = utils::kWidthPacked;
  }
  TmpTensor im2col_tmp(
      &graph,
      im2col_sizes,
      graph.dtype_of(in),
      im2col_tmp_storage,
      im2col_layout);
  const ValueRef im2col_tensor = im2col_tmp.vref;

  // Step 1: im2col
  add_conv2d_im2col_node(
      graph,
      in,
      im2col_tensor,
      utils::safe_downcast<int32_t>(K_h),
      utils::safe_downcast<int32_t>(K_w),
      stride_h,
      stride_w,
      padding_h,
      padding_w,
      dilation_h,
      dilation_w,
      utils::safe_downcast<int32_t>(Cin_padded),
      utils::safe_downcast<int32_t>(H_out),
      utils::safe_downcast<int32_t>(W_out));

  // Step 2: prepack weight for the GEMM directly from the serialized
  // [C_out, C_in, K_h, K_w] weight on the GPU. The serialized data is read
  // as-is (never CPU-repacked); the prepack shader does the im2col K-axis
  // reorder + 4x4 transpose into the layout conv2d_gemm.glsl loads via
  // load_packed_weight_tile_with_checks.
  ValueRef packed_weight = prepack_conv2d_gemm_weight(graph, weight_data);

  // Bias prepack: matches the bias format conv2d_gemm expects. prepack_biases
  // only reads dim 0 (= C_out) of the weight, so the original 4D weight works
  // directly.
  ValueRef packed_bias = prepack_biases(
      graph,
      bias,
      weight_data,
      /*transposed=*/false,
      utils::kTexture2D,
      utils::kWidthPacked);

  check_conv_args(graph, in, out);

  // Step 3: GEMM
  add_conv2d_gemm_node(
      graph,
      im2col_tensor,
      packed_weight,
      packed_bias,
      out,
      utils::safe_downcast<int32_t>(K_total),
      utils::safe_downcast<int32_t>(M),
      clamp_out,
      out_min_val,
      out_max_val);
}

//
// Op registration — matches aten.convolution.default's 10-arg signature:
//   in, weight, bias, stride, padding, dilation, transposed,
//   output_padding, groups, out
//
// Only the conv2d non-transposed, groups=1 case is supported.

void conv2d_gemm_op(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  VK_CHECK_COND(args.size() == 10);
  const ValueRef in = args[0];
  const ValueRef weight = args[1];
  const ValueRef bias = args[2];
  const ValueRef stride = args[3];
  const ValueRef padding = args[4];
  const ValueRef dilation = args[5];
  const ValueRef transposed = args[6];
  const ValueRef /*output_padding*/ _output_padding = args[7];
  (void)_output_padding;
  const ValueRef groups = args[8];
  const ValueRef out = args[9];

  VK_CHECK_COND(graph.get_bool(transposed) == false);
  VK_CHECK_COND(graph.get_int(groups) == 1);

  conv2d_gemm_impl(
      graph,
      in,
      weight,
      bias,
      stride,
      padding,
      dilation,
      out,
      /*clamp_out=*/false,
      /*out_min_val=*/0.0f,
      /*out_max_val=*/0.0f);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.conv2d_gemm.default, conv2d_gemm_op);
}

} // namespace vkcompute
