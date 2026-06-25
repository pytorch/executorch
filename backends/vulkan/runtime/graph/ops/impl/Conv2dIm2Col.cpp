/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Conv2dIm2Col.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

namespace {

// Compute the im2col output extents (M = H_out * W_out, K4_total) from the
// im2col_out tensor's current sizes. The tensor is virtually resized on
// trigger_resize (see resize_conv2d_im2col_node), so reading from it tracks
// dynamic shapes.
//
// Two layouts are possible:
//   - flat [M, K_total]                  (buffer / texture2d)
//   - [1, K_total, H_out, W_out]         (texture3d)
struct Im2colExtents {
  uint32_t m;
  uint32_t k4_total;
};

Im2colExtents im2col_extents_of(ComputeGraph* graph, const ValueRef im2col) {
  const std::vector<int64_t> sizes = graph->sizes_of(im2col);
  uint32_t m;
  uint32_t k_total;
  if (sizes.size() == 4) {
    // texture3d [1, K_total, H_out, W_out]
    const int64_t h_out = sizes.at(2);
    const int64_t w_out = sizes.at(3);
    m = utils::safe_downcast<uint32_t>(h_out * w_out);
    k_total = utils::safe_downcast<uint32_t>(sizes.at(1));
  } else {
    // flat [M, K_total]
    m = utils::safe_downcast<uint32_t>(sizes.at(0));
    k_total = utils::safe_downcast<uint32_t>(sizes.at(1));
  }
  return {m, k_total / 4u};
}

utils::uvec3 pick_conv2d_im2col_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef im2col_out = args.at(0).refs.at(0);
  const Im2colExtents ext = im2col_extents_of(graph, im2col_out);
  // Global wg: one thread per (k4, m) vec4 in the output.
  return {ext.k4_total, ext.m, 1u};
}

utils::uvec3 pick_conv2d_im2col_local_wg_size(
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
  // Fixed {16, 4, 1} mirrors the original static dispatch — one thread per
  // (k4, m) vec4 with 16 K-tiles × 4 M positions per workgroup.
  return {16u, 4u, 1u};
}

// Recompute the im2col output spatial extents from the current input shape and
// virtually resize the im2col tensor. Both possible layouts must be handled:
//   - flat [M, K_total]            -> resize dim 0 (M = H_out * W_out)
//   - [1, K_total, H_out, W_out]   -> resize dims 2/3 (H_out, W_out)
// K_total / Cin_padded are shape-independent, so the K dimension is preserved
// from the current tensor sizes.
//
// resize_args = { in, weight_data, stride, padding, dilation }
void resize_conv2d_im2col_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef im2col_out = args.at(0).refs.at(0);
  const ValueRef in = resize_args.at(0);
  const ValueRef weight_data = resize_args.at(1);
  const ValueRef stride = resize_args.at(2);
  const ValueRef padding = resize_args.at(3);
  const ValueRef dilation = resize_args.at(4);

  const std::vector<int64_t> in_sizes = graph->sizes_of(in);

  // Height / Width from the current input, via the shared conv-output helper
  // (same H/W split + formula the direct-conv resize uses). kernel_size is read
  // from the weight dims; stride/padding/dilation from the original IntList
  // ValueRefs. All are shape-independent — only H_in / W_in change at runtime.
  // transposed=false, and the args[3] slot (consulted only as an optional
  // ceil_mode) is a non-bool ValueRef, so ceil_mode resolves to false.
  const std::vector<int64_t> out_hw = calc_out_sizes_hw(
      *graph,
      in_sizes,
      weight_data,
      /*kernel_size_only=*/false,
      {stride, padding, dilation, dilation},
      /*transposed=*/false);
  const int64_t H_out = out_hw.at(0);
  const int64_t W_out = out_hw.at(1);

  const std::vector<int64_t> cur_sizes = graph->sizes_of(im2col_out);
  std::vector<int64_t> new_sizes = cur_sizes;
  if (cur_sizes.size() == 4) {
    // texture3d [1, K_total, H_out, W_out]: K_total (dim 1) is preserved.
    new_sizes.at(2) = H_out;
    new_sizes.at(3) = W_out;
  } else {
    // flat [M, K_total]: K_total (dim 1) is preserved.
    new_sizes.at(0) = H_out * W_out;
  }
  graph->virtual_resize(im2col_out, new_sizes);
}

} // namespace

// Push constants are uploaded in 16-byte chunks (one ivec4 each) to comply
// with the per-entry size limit. Layout matches conv2d_im2col.glsl:
//   { ivec4 kernel_stride, ivec4 padding_dil, ivec4 dims }
// All fields are shape-independent; W_out / H_out / M are derived in the shader
// from the (resize-refreshed) in_sizes UBO.

void add_conv2d_im2col_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef im2col_out,
    const ValueRef weight_data,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t padding_h,
    const int32_t padding_w,
    const int32_t dilation_h,
    const int32_t dilation_w,
    const int32_t Cin_padded) {
  const utils::StorageType out_storage = graph.storage_type_of(im2col_out);
  VK_CHECK_COND(
      out_storage == utils::kBuffer || out_storage == utils::kTexture2D ||
      out_storage == utils::kTexture3D);

  std::string kernel_name = "conv2d_im2col";
  add_storage_type_suffix(kernel_name, out_storage);
  add_dtype_suffix(kernel_name, graph.dtype_of(im2col_out));
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  // K_total is laid out so that 4-tiles share a kernel position; since
  // Cin_padded is a multiple of 4, K_total is also a multiple of 4.
  const int32_t K_total = kernel_h * kernel_w * Cin_padded;
  VK_CHECK_COND(K_total % 4 == 0);
  const int32_t K4_total = K_total / 4;

  const utils::ivec4 kernel_stride{kernel_h, kernel_w, stride_h, stride_w};
  const utils::ivec4 padding_dil{padding_h, padding_w, dilation_h, dilation_w};
  // dims.y / dims.z (formerly W_out / H_out) are unused by the shader now —
  // the spatial extents are derived at runtime from in_sizes. Only Cin_padded
  // and K4_total (both shape-independent) are consumed.
  const utils::ivec4 dims{Cin_padded, 0, 0, K4_total};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      shader,
      pick_conv2d_im2col_global_wg_size,
      pick_conv2d_im2col_local_wg_size,
      // Inputs and Outputs
      {{im2col_out, vkapi::kWrite}, {in, vkapi::kRead}},
      // UBOs
      {graph.sizes_ubo(in)},
      // Push constants (3 × ivec4 = 48 bytes, split per 16-byte limit)
      {PushConstantDataInfo(&kernel_stride, sizeof(kernel_stride)),
       PushConstantDataInfo(&padding_dil, sizeof(padding_dil)),
       PushConstantDataInfo(&dims, sizeof(dims))},
      // Specialization constants
      {},
      // Resize args
      {in, weight_data, stride, padding, dilation},
      // Resizing logic
      resize_conv2d_im2col_node));
}

} // namespace vkcompute
