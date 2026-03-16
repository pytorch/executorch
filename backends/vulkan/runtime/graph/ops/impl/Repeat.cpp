/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

namespace {

void check_args(
    ComputeGraph& graph,
    const ValueRef in,
    const std::vector<int64_t>& repeats,
    const ValueRef out) {
  VK_CHECK_COND(graph.packed_dim_of(in) == graph.packed_dim_of(out));

  VK_CHECK_COND(graph.storage_type_of(in) == graph.storage_type_of(out));
  if (graph.storage_type_of(in) == utils::kTexture2D) {
    VK_CHECK_COND(graph.dim_of(in) <= 2);
  }

  const int64_t in_dim = graph.dim_of(in);
  VK_CHECK_COND(
      in_dim <= repeats.size(),
      "Input tensor dim size must be not greater than the repeat argument's size");

  const std::vector<int64_t> in_sizes = graph.sizes_of(in);
  const std::vector<int64_t> out_sizes = graph.sizes_of(out);

  VK_CHECK_COND(
      dim_at<kWidth4D>(in_sizes) * dim_at<kWidth4D>(repeats) ==
          dim_at<kWidth4D>(out_sizes),
      "Output's width doesn't match input's width * repeat count");

  VK_CHECK_COND(
      dim_at<kHeight4D>(in_sizes) * dim_at<kHeight4D>(repeats) ==
          dim_at<kHeight4D>(out_sizes),
      "Output's height doesn't match input's height * repeat count");

  VK_CHECK_COND(
      dim_at<kChannel4D>(in_sizes) * dim_at<kChannel4D>(repeats) ==
          dim_at<kChannel4D>(out_sizes),
      "Output's channel doesn't match input's channel * repeat count");

  VK_CHECK_COND(
      dim_at<kBatch4D>(in_sizes) * dim_at<kBatch4D>(repeats) ==
          dim_at<kBatch4D>(out_sizes),
      "Output's batch doesn't match input's batch * repeat count");
}

} // namespace

void add_repeat_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef repeats_ref,
    ValueRef out) {
  const std::vector<int64_t> repeats = *(graph.get_int_list(repeats_ref));

  check_args(graph, in, repeats, out);

  const std::vector<int64_t> in_sizes = graph.sizes_of(in);
  const utils::ivec4 src_dims{
      dim_at<kWidth4D>(in_sizes),
      dim_at<kHeight4D>(in_sizes),
      dim_at<kChannel4D>(in_sizes),
      dim_at<kBatch4D>(in_sizes)};
  const utils::ivec4 dst_repeats{
      dim_at<kWidth4D>(repeats),
      dim_at<kHeight4D>(repeats),
      dim_at<kChannel4D>(repeats),
      dim_at<kBatch4D>(repeats)};

  std::string kernel_name = "repeat";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  // A copy of range with the last element set to batch size of the input tensor
  const utils::ivec3 wg_size = graph.logical_limits_of(out);

  const auto shader = VK_KERNEL_FROM_STR(kernel_name);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {
          {out, vkapi::kWrite},
          {in, vkapi::kRead},
      },
      // Parameter buffers
      {},
      // Push Constants
      {
          PushConstantDataInfo(&wg_size, sizeof(wg_size), sizeof(utils::ivec4)),
          PushConstantDataInfo(
              &src_dims, sizeof(src_dims), sizeof(utils::ivec4)),
          PushConstantDataInfo(
              &dst_repeats, sizeof(dst_repeats), sizeof(utils::ivec4)),
      },
      // Specialization Constants
      {graph.hashed_layout_of(out), graph.hashed_layout_of(in)},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

void repeat(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  add_repeat_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.repeat.default, repeat);
}

} // namespace vkcompute
