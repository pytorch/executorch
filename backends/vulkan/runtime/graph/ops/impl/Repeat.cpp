/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Copy.h>

namespace vkcompute {

namespace {

void check_args(
    const api::vTensor& in,
    const std::vector<int64_t>& repeats,
    const api::vTensor& out) {
  VK_CHECK_COND(check_memory_layout_is(in, utils::kChannelsPacked));
  VK_CHECK_COND(check_memory_layout_is(out, utils::kChannelsPacked));

  int64_t in_dim = in.dim();
  VK_CHECK_COND(
      in_dim <= repeats.size(),
      "Input tensor dim size must be not greater than the repeat argument's size");

  VK_CHECK_COND(
      dim_at<kWidth4D>(in.sizes()) * dim_at<kWidth4D>(repeats) ==
          dim_at<kWidth4D>(out.sizes()),
      "Output's width doesn't match input's width * repeat count");

  VK_CHECK_COND(
      dim_at<kHeight4D>(in.sizes()) * dim_at<kHeight4D>(repeats) ==
          dim_at<kHeight4D>(out.sizes()),
      "Output's height doesn't match input's height * repeat count");

  VK_CHECK_COND(
      dim_at<kChannel4D>(in.sizes()) * dim_at<kChannel4D>(repeats) ==
          dim_at<kChannel4D>(out.sizes()),
      "Output's channel doesn't match input's channel * repeat count");

  VK_CHECK_COND(
      dim_at<kBatch4D>(in.sizes()) * dim_at<kBatch4D>(repeats) ==
          dim_at<kBatch4D>(out.sizes()),
      "Output's batch doesn't match input's batch * repeat count");
}

} // namespace

void add_repeat_channel_node(
    ComputeGraph& graph,
    ValueRef in,
    int64_t repeat_channel,
    ValueRef out,
    utils::ivec3& running_range) {
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);

  std::string kernel_name = "repeat_channel";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);

  const std::vector<int64_t>& in_sizes = t_in->sizes();

  int32_t in_width = utils::safe_downcast<int32_t>(dim_at<kWidth4D>(in_sizes));
  int32_t in_height =
      utils::safe_downcast<int32_t>(dim_at<kHeight4D>(in_sizes));
  int32_t in_channel =
      utils::safe_downcast<int32_t>(dim_at<kChannel4D>(in_sizes));
  int32_t in_batch = utils::safe_downcast<int32_t>(dim_at<kBatch4D>(in_sizes));

  int32_t out_channel = repeat_channel * in_channel;

  utils::ivec4 out_whcn_sizes{in_width, in_height, out_channel, in_batch};

  utils::ivec4 in_whcn_sizes{in_width, in_height, in_channel, in_batch};

  // Channel packed global work ids
  running_range[2] = out_whcn_sizes[3] * utils::div_up_4(out_whcn_sizes[2]);
  utils::uvec3 global_size = utils::make_uvec3(running_range);
  utils::uvec3 local_size = adaptive_work_group_size(global_size);

  const struct Block final {
    utils::ivec4 out_sizes;
    utils::ivec4 in_size;
  } repeat_channel_args{
      out_whcn_sizes,
      in_whcn_sizes,
  };

  auto shader = VK_KERNEL_FROM_STR(kernel_name);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{out, vkapi::MemoryAccessType::WRITE},
       {in, vkapi::MemoryAccessType::READ}},
      // Parameter buffers
      {graph.create_params_buffer(repeat_channel_args)},
      // Specialization Constants
      {SV(t_out->packed_dim_whcn_idx())}));
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

  utils::ivec3 running_range = t_in->texture_limits();

  const std::vector<int64_t>& in_sizes = t_in->sizes();

  // Since we use channel packing, repeating the channel dimension is the most
  // complicated and time-consuming, as we need to reason over misaligned
  // channels. Hence we expand it first to minimize cost. Also, in this first
  // dimension, we copy over the input texure to the output. In subsequent
  // dimensions, we read and write from the same tensor.

  if (int64_t channel_repeat = dim_at<kChannel4D>(repeats);
      channel_repeat == 1) {
    // If no repeat, short-cut to a direct copy
    utils::ivec3 src_offset{0, 0, 0};
    utils::ivec3 dst_offset{0, 0, 0};

    add_copy_offset_node(graph, in, running_range, src_offset, dst_offset, out);

  } else {
    add_repeat_channel_node(graph, in, channel_repeat, out, running_range);
  }

  // TODO: refactor width, height, and batch into a common helper function.
  // Width
  if (int64_t width_repeat = dim_at<kWidth4D>(repeats); width_repeat > 1) {
    utils::ivec3 src_offset{0, 0, 0};

    for (int i = 1; i < width_repeat; ++i) {
      utils::ivec3 dst_offset{i * dim_at<kWidth4D>(in_sizes), 0, 0};

      add_copy_offset_node(
          graph, out, running_range, src_offset, dst_offset, out);
    }

    running_range[0] = running_range[0] * width_repeat;
  }

  // Height
  if (int64_t height_repeat = dim_at<kHeight4D>(repeats); height_repeat > 1) {
    utils::ivec3 src_offset{0, 0, 0};

    for (int i = 1; i < height_repeat; ++i) {
      utils::ivec3 dst_offset = {0, i * dim_at<kHeight4D>(in_sizes), 0};

      add_copy_offset_node(
          graph, out, running_range, src_offset, dst_offset, out);
    }

    running_range[1] = running_range[1] * height_repeat;
  }

  // Batch
  if (int64_t batch_repeat = dim_at<kBatch4D>(repeats); batch_repeat > 1) {
    utils::ivec3 src_offset{0, 0, 0};

    for (int i = 1; i < batch_repeat; ++i) {
      utils::ivec3 dst_offset = {0, 0, i * running_range[2]};

      add_copy_offset_node(
          graph, out, running_range, src_offset, dst_offset, out);
    }

    running_range[2] = running_range[2] * batch_repeat;
  }
}

void repeat(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  add_repeat_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.repeat.default, repeat);
}

} // namespace vkcompute
