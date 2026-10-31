/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/Convolution.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv1d/Conv1d.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2d.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

namespace vkcompute {

ValueRef prepack_biases(
    ComputeGraph& graph,
    const ValueRef vref,
    const ValueRef weight,
    const bool transposed,
    const utils::StorageType storage_type,
    const utils::GPUMemoryLayout memory_layout) {
  auto sizes = graph.sizes_of(weight);
  const int64_t out_channels = transposed ? sizes.at(1) : sizes.at(0);

  // Texture biases are padded to an even number of texels so shaders that
  // compute two output texels per invocation can read both without a check.
  const int64_t packed_out_channels = storage_type == utils::kBuffer
      ? out_channels
      : utils::align_up(out_channels, int64_t(8));
  ValueRef v = graph.add_tensor(
      {packed_out_channels},
      graph.dtype_of(weight),
      storage_type,
      memory_layout);

  vkapi::ShaderInfo shader =
      get_nchw_to_tensor_shader(graph, v, graph.get_staging_dtype_for(weight));

  vkapi::ParamsBindList param_buffers = {};
  if (graph.is_buffer_storage(v)) {
    param_buffers.append(graph.buffer_meta_ubo(v));
  } else if (packed_out_channels != out_channels) {
    vTensorPtr bias(&graph, v);
    const api::vTensor::TextureMetadata metadata(
        {out_channels},
        bias->image_extents(),
        bias->axis_map(),
        bias->packed_dim_info());
    param_buffers.append(graph.create_params_buffer(metadata));
  } else {
    param_buffers.append(graph.texture_meta_ubo(v));
  }

  std::vector<PushConstantDataInfo> pcs;
  if (graph.is_buffer_storage(v)) {
    pcs = {graph.sizes_pc_of(v), graph.strides_pc_of(v), graph.numel_pc_of(v)};
  }

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      shader,
      graph.create_gwg(v),
      graph.create_lwg(v),
      vref,
      v,
      param_buffers,
      // Specialization constants
      {graph.hashed_layout_of(v)},
      pcs));

  return v;
}

void check_conv_args(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef out) {
  VK_CHECK_COND(graph.packed_dim_of(in) == WHCN::kChannelsDim);
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kChannelsDim);
}

void conv(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  if (graph.dim_of(args[0]) == 4) {
    return conv2d_impl(graph, args);
  }
  return conv1d_impl(graph, args);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.convolution.default, conv);
  VK_REGISTER_OP(conv_with_clamp.default, conv);
  VK_REGISTER_OP(et_vk.conv_with_clamp.default, conv);
}

} // namespace vkcompute
