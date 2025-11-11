/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/BlitNode.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {

BlitNode::BlitNode(
    ComputeGraph& graph,
    ValueRef src,
    ValueRef dst,
    // const vkapi::ScalarType& dtype,
    const ResizeFunction& resize_fn,
    const std::vector<ValueRef>& resize_args)
    : ExecuteNode(resize_fn, resize_args, {}, "Blit Node"),
      src_(src),
      dst_(dst) {
  (void)graph;
}

void BlitNode::encode(ComputeGraph* graph) {
  VK_CHECK_COND(
      graph->storage_type_of(src_) != utils::kBuffer &&
          graph->storage_type_of(dst_) != utils::kBuffer,
      "BlitNode: Only texture backed tensors are supported.");

  api::Context* const context = graph->context();
  vkapi::PipelineBarrier pipeline_barrier{};

  std::unique_lock<std::mutex> cmd_lock = context->dispatch_lock();

  // Hack to get timing data for non shader op
  std::string kernel_name("Blit_");
  kernel_name.reserve(32);
  kernel_name += vkapi::to_string(graph->dtype_of(src_));
  kernel_name += "_to_";
  kernel_name += vkapi::to_string(graph->dtype_of(dst_));

  context->report_shader_dispatch_start(
      kernel_name, utils::uvec3(), utils::WorkgroupSize(), node_id_);

  context->register_blit(
      pipeline_barrier,
      graph->get_tensor(src_)->image(
          pipeline_barrier, vkapi::PipelineStage::TRANSFER, vkapi::kRead),
      graph->get_tensor(dst_)->image(
          pipeline_barrier, vkapi::PipelineStage::TRANSFER, vkapi::kWrite));

  context->report_shader_dispatch_end();
}

} // namespace vkcompute
