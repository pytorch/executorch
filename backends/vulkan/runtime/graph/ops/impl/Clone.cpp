/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/Logging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/View.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_clone_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr in = graph->get_tensor(args[1].refs[0]);
  // TODO: support for when dimensionality doesn't match, i.e. clone is used to
  // implement squeeze.
  if (out->dim() == in->dim()) {
    out->virtual_resize(in->sizes());
  }
}

void add_clone_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef out) {
  vTensorPtr t_out = graph.get_tensor(out);

  std::string kernel_name = "clone";
  add_dtype_suffix(kernel_name, *t_out);

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Parameter Buffers
      {t_out->logical_limits_ubo()},
      // Specialization Constants
      {},
      // Resizing Logic
      resize_clone_node));
}

void add_image_to_buffer_node(
    ComputeGraph& graph,
    const ValueRef image,
    const ValueRef buffer) {
  std::string kernel_name = "clone_image_to_buffer";
  add_dtype_suffix(kernel_name, graph.dtype_of(image));
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  utils::uvec3 global_wg_size = graph.create_global_wg_size(image);
  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      shader,
      global_wg_size,
      graph.create_local_wg_size(global_wg_size),
      // Input and Outputs
      {{buffer, vkapi::kWrite}, {image, vkapi::kRead}},
      // Parameter Buffers
      {graph.sizes_ubo(image), graph.strides_ubo(buffer)},
      // Specialization Constants
      {graph.hashed_layout_of(image)},
      // Resizing Logic
      resize_clone_node));
}

void add_buffer_to_image_node(
    ComputeGraph& graph,
    const ValueRef buffer,
    const ValueRef image) {
  std::string kernel_name = "clone_buffer_to_image";
  add_dtype_suffix(kernel_name, graph.dtype_of(image));
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  utils::uvec3 global_wg_size = graph.create_global_wg_size(image);
  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      shader,
      global_wg_size,
      graph.create_local_wg_size(global_wg_size),
      // Input and Outputs
      {{image, vkapi::kWrite}, {buffer, vkapi::kRead}},
      // Parameter Buffers
      {graph.sizes_ubo(image), graph.strides_ubo(buffer)},
      // Specialization Constants
      {graph.hashed_layout_of(image)},
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
  VK_THROW("Buffer to buffer memory layout transition not supported yet!");
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
