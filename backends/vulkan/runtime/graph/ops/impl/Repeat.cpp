/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/Logging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Copy.h>

#include <iostream>

namespace vkcompute {

namespace {

void check_args(
    const vTensor& in,
    const std::vector<int64_t>& repeats,
    const vTensor& out) {
  VK_CHECK_COND(check_memory_layout_is(in, api::kChannelsPacked));
  VK_CHECK_COND(check_memory_layout_is(out, api::kChannelsPacked));

  int64_t in_dim = in.dim();
  VK_CHECK_COND(
      in_dim == repeats.size(), "Input tensor dim size must match argument");

  VK_CHECK_COND(
      dim_at<Dim4D::Width>(in.sizes()) * dim_at<Dim4D::Width>(repeats) ==
          dim_at<Dim4D::Width>(out.sizes()),
      "Output's width doesn't match input's width * repeat count");

  VK_CHECK_COND(
      dim_at<Dim4D::Height>(in.sizes()) * dim_at<Dim4D::Height>(repeats) ==
          dim_at<Dim4D::Height>(out.sizes()),
      "Output's height doesn't match input's height * repeat count");

  VK_CHECK_COND(
      dim_at<Dim4D::Channel>(in.sizes()) * dim_at<Dim4D::Channel>(repeats) ==
          dim_at<Dim4D::Channel>(out.sizes()),
      "Output's channel doesn't match input's channel * repeat count");

  VK_CHECK_COND(
      dim_at<Dim4D::Batch>(in.sizes()) * dim_at<Dim4D::Batch>(repeats) ==
          dim_at<Dim4D::Batch>(out.sizes()),
      "Output's batch doesn't match input's batch * repeat count");
}

} // namespace

void add_repeat_channel_node(
    ComputeGraph& graph,
    ValueRef in,
    int64_t repeat_channel,
    ValueRef out,
    api::utils::ivec3& running_range) {
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);

  std::string kernel_name = "repeat_channel";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);

  const std::vector<int64_t>& in_sizes = t_in->sizes();

  int32_t in_width =
      api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Width>(in_sizes));
  int32_t in_height =
      api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Height>(in_sizes));
  int32_t in_channel =
      api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Channel>(in_sizes));
  int32_t in_batch =
      api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Batch>(in_sizes));

  int32_t out_channel = repeat_channel * in_channel;

  api::utils::ivec4 out_whcn_sizes{in_width, in_height, out_channel, in_batch};

  api::utils::ivec4 in_whcn_sizes{in_width, in_height, in_channel, in_batch};

  // Channel packed global work ids
  running_range.data[2] =
      out_whcn_sizes.data[3] * api::utils::div_up(out_whcn_sizes.data[2], 4);
  api::utils::uvec3 global_size = api::utils::make_uvec3(running_range);
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  const struct Block final {
    api::utils::ivec4 out_sizes;
    api::utils::ivec4 in_size;
  } repeat_channel_args{
      out_whcn_sizes,
      in_whcn_sizes,
  };

  auto shader = VK_KERNEL_FROM_STR(kernel_name);
  // std::cout << "out tile size: " << shader.out_tile_size << std::endl;

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{out, api::MemoryAccessType::WRITE}, {in, api::MemoryAccessType::READ}},
      // Parameter buffers
      {graph.create_params_buffer(repeat_channel_args)},
      // Specialization Constants
      {}));
}

void add_repeat_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef repeats_ref,
    ValueRef out) {
  std::vector<int64_t> repeats = *(graph.get_int_list(repeats_ref));

  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);
  check_args(*t_in, repeats, *t_out);

  // In this function, we expand the dimensions in the following order:
  // 1. Channel
  // 2. Width
  // 3. Height
  // 4. Batch
  // After expanding a dimension, we will update the "running_range" since we
  // will need to copy the "expanded" area.

  api::utils::ivec3 running_range = t_in->texture_limits().limits;

  const std::vector<int64_t>& in_sizes = t_in->sizes();

  // We use channel packing, repeating the channel dimension is the most
  // complicated and time-consuming, since we need to reason over misaligned
  // channels. Hence we expand it first to minimize cost. Also, in this first
  // dimension, we copy over the input texure to the output. In subsequent
  // dimensions, we read and write from the same tensor.

  if (int64_t channel_repeat = dim_at<Dim4D::Channel>(repeats);
      channel_repeat == 1) {
    // If no repeat, short-cut to a direct copy
    api::utils::ivec3 src_offset = api::utils::make_ivec3({0, 0, 0}, false);
    api::utils::ivec3 dst_offset = api::utils::make_ivec3({0, 0, 0}, false);

    add_copy_offset_node(graph, in, running_range, src_offset, dst_offset, out);

  } else {
    add_repeat_channel_node(graph, in, channel_repeat, out, running_range);
  }

  // Width
  if (int64_t width_repeat = dim_at<Dim4D::Width>(repeats); width_repeat > 1) {
    api::utils::ivec3 src_offset = api::utils::make_ivec3({0, 0, 0}, false);
    // api::utils::ivec3 range = t_in->texture_limits().limits;

    for (int i = 1; i < width_repeat; i++) {
      api::utils::ivec3 dst_offset = api::utils::make_ivec3(
          {i * dim_at<Dim4D::Width>(in_sizes), 0, 0}, false);

      add_copy_offset_node(
          graph, out, running_range, src_offset, dst_offset, out);
    }

    running_range.data[0] = running_range.data[0] * width_repeat;
  }

  // Height
  if (int64_t height_repeat = dim_at<Dim4D::Height>(repeats);
      height_repeat > 1) {
    api::utils::ivec3 src_offset = api::utils::make_ivec3({0, 0, 0}, false);

    for (int i = 1; i < height_repeat; i++) {
      api::utils::ivec3 dst_offset = api::utils::make_ivec3(
          {0, i * dim_at<Dim4D::Height>(in_sizes), 0}, false);

      add_copy_offset_node(
          graph, out, running_range, src_offset, dst_offset, out);
    }

    running_range.data[1] = running_range.data[1] * height_repeat;
  }

  // Batch
  if (int64_t batch_repeat = dim_at<Dim4D::Batch>(repeats); batch_repeat > 1) {
    api::utils::ivec3 src_offset = api::utils::make_ivec3({0, 0, 0}, false);

    for (int i = 1; i < batch_repeat; i++) {
      api::utils::ivec3 dst_offset =
          api::utils::make_ivec3({0, 0, i * running_range.data[2]}, false);

      add_copy_offset_node(
          graph, out, running_range, src_offset, dst_offset, out);
    }

    running_range.data[2] = running_range.data[2] * batch_repeat;
  }
}

void repeat(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  add_repeat_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.repeat.default, repeat);
}

} // namespace vkcompute
