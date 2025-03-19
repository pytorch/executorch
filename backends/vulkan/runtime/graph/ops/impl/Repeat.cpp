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
  VK_CHECK_COND(check_same_packed_dim(in, out));

  VK_CHECK_COND(in.storage_type() == out.storage_type());
  if (in.storage_type() == utils::kTexture2D) {
    VK_CHECK_COND(in.dim() <= 2);
  }

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

void add_repeat_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef repeats_ref,
    ValueRef out) {
  const std::vector<int64_t> repeats = *(graph.get_int_list(repeats_ref));

  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);
  check_args(*t_in, repeats, *t_out);

  const utils::ivec4 src_offset{
      dim_at<kWidth4D>(t_in->sizes()),
      dim_at<kHeight4D>(t_in->sizes()),
      dim_at<kChannel4D>(t_in->sizes()),
      dim_at<kBatch4D>(t_in->sizes())};
  const utils::ivec4 dst_offset{
      dim_at<kWidth4D>(repeats),
      dim_at<kHeight4D>(repeats),
      dim_at<kChannel4D>(repeats),
      dim_at<kBatch4D>(repeats)};
  add_copy_packed_dim_offset_node(
      graph, in, t_out->logical_limits(), src_offset, dst_offset, out, true);
}

void repeat(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  add_repeat_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.repeat.default, repeat);
}

} // namespace vkcompute
