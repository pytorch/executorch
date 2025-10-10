/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/Logging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/View.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_clone_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  // TODO: support for when dimensionality doesn't match, i.e. clone is used to
  // implement squeeze.
  if (graph->dim_of(out) == graph->dim_of(in)) {
    graph->virtual_resize(out, graph->sizes_of(in));
  }
}

void add_clone_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef out) {
  std::string kernel_name = "clone";
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Parameter Buffers
      {graph.logical_limits_ubo(out)},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_clone_node));
}

utils::uvec3 clone_image_to_buffer_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef image = args.at(1).refs.at(0);
  return graph->create_global_wg_size(image);
}

void add_image_to_buffer_node(
    ComputeGraph& graph,
    const ValueRef image,
    const ValueRef buffer) {
  std::string kernel_name = "clone_image_to_buffer";
  add_dtype_suffix(kernel_name, graph.dtype_of(image));
  add_dtype_suffix(kernel_name, graph.dtype_of(buffer));
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      shader,
      clone_image_to_buffer_global_wg_size,
      default_pick_local_wg_size,
      // Input and Outputs
      {{buffer, vkapi::kWrite}, {image, vkapi::kRead}},
      // Parameter Buffers
      {},
      // Push Constants
      {graph.sizes_pc_of(image), graph.strides_pc_of(buffer)},
      // Specialization Constants
      {graph.hashed_layout_of(image)},
      // Resize Args
      {},
      // Resizing Logic
      resize_clone_node));
}

void add_buffer_to_image_node(
    ComputeGraph& graph,
    const ValueRef buffer,
    const ValueRef image) {
  std::string kernel_name = "clone_buffer_to_image";
  add_dtype_suffix(kernel_name, graph.dtype_of(image));
  add_dtype_suffix(kernel_name, graph.dtype_of(buffer));
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      shader,
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Input and Outputs
      {{image, vkapi::kWrite}, {buffer, vkapi::kRead}},
      // Parameter Buffers
      {},
      // Push Constants
      {graph.sizes_pc_of(image), graph.strides_pc_of(buffer)},
      // Specialization Constants
      {graph.hashed_layout_of(image)},
      // Resize Args
      {},
      // Resizing Logic
      resize_clone_node));
}

void clone(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef src = args[0];
  const ValueRef dst = args[2];

  const utils::StorageType src_storage = graph.storage_type_of(src);
  const utils::StorageType dst_storage = graph.storage_type_of(dst);
  if (src_storage == utils::kTexture3D && dst_storage == utils::kTexture3D) {
    if (graph.hashed_layout_of(src) == graph.hashed_layout_of(dst)) {
      return add_clone_node(graph, src, dst);
    } else {
      return add_view_node(graph, src, kDummyValueRef, dst);
    }
  }
  if (src_storage == utils::kTexture3D && dst_storage == utils::kBuffer) {
    return add_image_to_buffer_node(graph, src, dst);
  }
  if (src_storage == utils::kBuffer && dst_storage == utils::kTexture3D) {
    return add_buffer_to_image_node(graph, src, dst);
  }

  std::vector<ValueRef> extra_args = {};
  // Buffer to buffer copy
  return add_view_copy_buffer_node(
      graph, src, dst, extra_args, resize_clone_node);
}

// Clone node is not the most efficient implementation for the aten.clone
// operation. A more efficient implementation can be achieved during vulkan
// export with the use of shared object. This clone node is introduced to enable
// a "copy" mechanism if there is no alternative (e.g. during direct
// ComputeGraph manipulation, we need to make a copy of a Tensor).

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.clone.default, clone);
}

} // namespace vkcompute
