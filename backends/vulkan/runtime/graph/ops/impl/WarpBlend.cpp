/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_warp_blend_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef img0 = args.at(1).refs.at(0);
  // out has the same shape as img0: [N, 3, H, W].
  graph->virtual_resize(out, graph->sizes_of(img0));
}

void add_warp_blend_node(
    ComputeGraph& graph,
    const ValueRef img0,
    const ValueRef img1,
    const ValueRef flow,
    const ValueRef mask,
    const ValueRef out) {
  // The shader hard-codes channels-packed texture indexing for every tensor and
  // assumes each has <= 4 channels (one z-slice per batch), so the output texel
  // position maps to the matching slice in every input. The partitioner gates
  // this via `inputs_storage` in op_registry.py::register_warp_blend, but assert
  // here too to avoid a silent miscompute on a bypass path.
  VK_CHECK_COND(
      graph.is_standard_channels_packed_texture_tensor(img0) &&
          graph.is_standard_channels_packed_texture_tensor(img1) &&
          graph.is_standard_channels_packed_texture_tensor(flow) &&
          graph.is_standard_channels_packed_texture_tensor(mask) &&
          graph.is_standard_channels_packed_texture_tensor(out),
      "Vulkan warp_blend requires all tensors to be channels-packed textures");

  // The shader binds all tensors with a single DTYPE selected via dtype_of(out),
  // so a mixed-precision graph would bind a buffer as the wrong type and
  // silently miscompute.
  VK_CHECK_COND(
      graph.dtype_of(img0) == graph.dtype_of(out) &&
          graph.dtype_of(img1) == graph.dtype_of(out) &&
          graph.dtype_of(flow) == graph.dtype_of(out) &&
          graph.dtype_of(mask) == graph.dtype_of(out),
      "Vulkan warp_blend requires all tensors to share dtype");

  // flow/mask are sampled at their own (internal) spatial size and bilinearly
  // upsampled to the output size in-shader, so they may be smaller than out but
  // must match each other (the shader reads both from `flowp` metadata).
  VK_CHECK_COND(
      graph.sizes_of(flow).at(2) == graph.sizes_of(mask).at(2) &&
          graph.sizes_of(flow).at(3) == graph.sizes_of(mask).at(3),
      "Vulkan warp_blend requires flow and mask to share spatial size");

  // Unlike flow/mask, both images are sampled at the OUTPUT extent: the shader
  // derives a single `max_xy` from `outp.sizes.xy` and clamps the sampling
  // coords of t_img0 and t_img1 with it. An img1 (or img0) smaller than out
  // would therefore let texelFetch read past that image's own extent, so
  // require all three to match rather than only the shape out is resized from.
  VK_CHECK_COND(
      graph.sizes_of(img0) == graph.sizes_of(out) &&
          graph.sizes_of(img1) == graph.sizes_of(out),
      "Vulkan warp_blend requires img0, img1 and out to share sizes");

  std::string kernel_name("warp_blend");
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_gwg,
      default_pick_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{img0, img1, flow, mask}, vkapi::kRead}},
      // Shader params buffers: output metadata (warp/blend at original size) and
      // flow metadata (internal size used for the in-shader bilinear upsample;
      // mask shares it).
      {graph.meta_ubo(out), graph.meta_ubo(flow)},
      // Push Constants
      {},
      // Specialization Constants — output hashed layout for packed_dim.
      {graph.hashed_layout_of(out)},
      // Resize Args
      {},
      // Resizing Logic
      resize_warp_blend_node));
}

void warp_blend(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // Argument order matches custom_ops_lib.py::warp_blend:
  //   (img0, img1, flow, mask, out)
  return add_warp_blend_node(
      graph, args[0], args[1], args[2], args[3], args[4]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.warp_blend.default, warp_blend);
}

} // namespace vkcompute
