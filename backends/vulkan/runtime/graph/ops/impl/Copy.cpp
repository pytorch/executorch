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

namespace vkcompute {

using utils::ivec3;
using utils::uvec3;

void add_copy_offset_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ivec3& range,
    const ivec3& src_offset,
    const ivec3& dst_offset,
    const ValueRef out) {
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);

  std::string kernel_name = "copy_offset";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);

  const struct Block final {
    ivec3 range;
    int32_t unused0;
    ivec3 src_offset;
    int32_t unused1;
    ivec3 dst_offset;
    int32_t unused2;
  } offset_params{
      range,
      0,
      src_offset,
      0,
      dst_offset,
      0,
  };

  auto shader = VK_KERNEL_FROM_STR(kernel_name);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
      // Inputs and Outputs
      {
          {out, vkapi::MemoryAccessType::WRITE},
          {in, vkapi::MemoryAccessType::READ},
      },
      // Parameter buffers
      {graph.create_params_buffer(offset_params)},
      // Specialization Constants
      {}));
}

void add_copy_channel_offset_node(
    ComputeGraph& graph,
    const ValueRef in,
    int32_t channel_range,
    int32_t src_channel_offset,
    int32_t dst_channel_offset,
    const ValueRef out) {
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);

  // Likely need to prepad these numbers.
  std::vector<int64_t> in_sizes = t_in->sizes();
  std::vector<int64_t> out_sizes = t_out->sizes();

  VK_CHECK_COND(check_memory_layout_is(*t_in, utils::kChannelsPacked));
  VK_CHECK_COND(check_memory_layout_is(*t_out, utils::kChannelsPacked));

  // NOTE: This function should be able to support 1d and 2d tensors when
  // range=1, src_offset=dst_offset=1.
  VK_CHECK_COND(t_in->dim() >= 3, "Src dim should be at least 3");
  VK_CHECK_COND(t_out->dim() >= 3, "Dst dim should be at least 3");

  VK_CHECK_COND(
      dim_at<kChannel4D>(in_sizes) >= src_channel_offset + channel_range,
      "Src channel (",
      src_channel_offset,
      ") and range (",
      channel_range,
      ") should be less than or equal to input tensor's channel size (",
      dim_at<kChannel4D>(in_sizes),
      ")");

  VK_CHECK_COND(
      dim_at<kChannel4D>(out_sizes) >= dst_channel_offset + channel_range,
      "Dst channel (",
      dst_channel_offset,
      ") and range (",
      channel_range,
      ") should be less than or equal to input tensor's channel size (",
      dim_at<kChannel4D>(out_sizes),
      ")");

  VK_CHECK_COND(channel_range >= 0, "Channel range must be non-negative");
  VK_CHECK_COND(
      src_channel_offset >= 0, "Src channel offset must be non-negative");
  VK_CHECK_COND(
      dst_channel_offset >= 0, "Dst channel offset must be non-negative");

  std::string kernel_name = "copy_channel_offset";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);

  int32_t out_channels = dim_at<kChannel4D>(out_sizes);

  // Copy one batch at a time.
  for (int batch_idx = 0; batch_idx < dim_at<kBatch4D>(in_sizes); batch_idx++) {
    // Mapping the tensor NCHW coordinates into texture XYZ coordinates
    int32_t dst_first_z = dst_channel_offset / 4;
    int32_t dst_last_z = (dst_channel_offset + channel_range - 1) / 4;

    // We copy the entire width and height dimension. For the channel dimension,
    // we use the z-dimension of the global_size to specify the texture range.
    // The shader combines the global invocation id and the dst_offset to get
    // the actual coordinate.

    ivec3 dst_offset{
        0, 0, dst_first_z + batch_idx * utils::div_up_4(out_channels)};

    uvec3 global_size{
        utils::safe_downcast<uint32_t>(dim_at<kWidth4D>(in_sizes)),
        utils::safe_downcast<uint32_t>(dim_at<kHeight4D>(in_sizes)),
        utils::safe_downcast<uint32_t>(dst_last_z - dst_first_z + 1)};
    uvec3 local_size = adaptive_work_group_size(global_size);

    const struct Block final {
      utils::ivec4 out_sizes;
      utils::ivec4 in_sizes;
      int32_t channel_range;
      int32_t src_channel_offset;
      int32_t dst_channel_offset;
      int32_t unused;
      ivec3 range;
      int32_t unused1;
      ivec3 dst_offset;
      int32_t unused2;

    } channel_offset_params{
        utils::make_whcn_ivec4(out_sizes),
        utils::make_whcn_ivec4(in_sizes),
        channel_range,
        src_channel_offset,
        dst_channel_offset,
        0,
        utils::make_ivec3(global_size),
        0,
        dst_offset,
        0,
    };

    auto shader = VK_KERNEL_FROM_STR(kernel_name);

    graph.execute_nodes().emplace_back(new ExecuteNode(
        graph,
        VK_KERNEL_FROM_STR(kernel_name),
        global_size,
        local_size,
        // Inputs and Outputs
        {
            {out, vkapi::MemoryAccessType::WRITE},
            {out, vkapi::MemoryAccessType::READ},
            {in, vkapi::MemoryAccessType::READ},
        },
        // Parameter buffers
        {graph.create_params_buffer(channel_offset_params)},
        // Specialization Constants
        {}));
  }
}

void add_copy_offset_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef range_ref,
    ValueRef src_offset_ref,
    ValueRef dst_offset_ref,
    ValueRef out) {
  ivec3 range = utils::make_ivec3(*graph.get_int_list(range_ref));
  ivec3 src_offset = utils::make_ivec3(*graph.get_int_list(src_offset_ref));
  ivec3 dst_offset = utils::make_ivec3(*graph.get_int_list(dst_offset_ref));

  add_copy_offset_node(graph, in, range, src_offset, dst_offset, out);
}

void copy_offset(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  add_copy_offset_node(graph, args[0], args[1], args[2], args[3], args[4]);
}

void copy_channel_offset(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  ValueRef in = args[0];
  ValueRef channel_range_ref = args[1];
  ValueRef src_channel_offset_ref = args[2];
  ValueRef dst_channel_offset_ref = args[3];
  ValueRef out = args[4];

  auto channel_range = graph.extract_scalar<int64_t>(channel_range_ref);
  auto src_channel_offset =
      graph.extract_scalar<int64_t>(src_channel_offset_ref);
  auto dst_channel_offset =
      graph.extract_scalar<int64_t>(dst_channel_offset_ref);

  add_copy_channel_offset_node(
      graph, in, channel_range, src_channel_offset, dst_channel_offset, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(etvk.copy_offset, copy_offset);
  VK_REGISTER_OP(etvk.copy_channel_offset, copy_channel_offset);
}

} // namespace vkcompute
