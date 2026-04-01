/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

utils::ivec4 create_pad_per_dim(const std::vector<int64_t>& pad) {
  // pad contains pairs of (before, after) values for each dimension, starting
  // from the innermost (W). BufferMetadata/TextureMetadata use WHCN order, so
  // map pad[0]->W, pad[2]->H, pad[4]->C, pad[6]->N.
  utils::ivec4 pad_per_dim{0, 0, 0, 0};
  if (pad.size() >= 2) {
    pad_per_dim[0] = static_cast<int32_t>(pad[0]);
  }
  if (pad.size() >= 4) {
    pad_per_dim[1] = static_cast<int32_t>(pad[2]);
  }
  if (pad.size() >= 6) {
    pad_per_dim[2] = static_cast<int32_t>(pad[4]);
  }
  if (pad.size() >= 8) {
    pad_per_dim[3] = static_cast<int32_t>(pad[6]);
  }
  return pad_per_dim;
}

void resize_constant_pad_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);
  const IntListPtr pad_vec = graph->get_int_list(extra_args.at(0));
  std::vector<int64_t> in_size = graph->sizes_of(self);
  int dim = in_size.size() - 1;
  for (int i = 0; i < pad_vec->size(); i += 2) {
    in_size.at(dim) += pad_vec->at(i) + pad_vec->at(i + 1);
    dim--;
  }

  graph->virtual_resize(out, in_size);
}

void add_constant_pad_nd_node(
    ComputeGraph& graph,
    const ValueRef& in,
    const ValueRef& pad,
    const ValueRef& fill_value_ref,
    const ValueRef& out) {
  const float fill_value_val = graph.extract_scalar<float>(fill_value_ref);
  const IntListPtr pad_vec = graph.get_int_list(pad);
  const utils::ivec4 pad_per_dim = create_pad_per_dim(*pad_vec);

  std::string kernel_name = "pad";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Parameter buffers
      {graph.meta_ubo(out),
       graph.meta_ubo(in),
       graph.create_params_buffer(pad_per_dim),
       graph.create_params_buffer(fill_value_val)},
      // Push Constants
      {},
      // Specialization Constants
      {graph.hashed_layout_of(out)},
      // Resize Args
      {pad},
      // Resizing Logic
      resize_constant_pad_node));
}

void constant_pad_nd(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  add_constant_pad_nd_node(graph, args[0], args[1], args[2], args[3]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.constant_pad_nd.default, constant_pad_nd);
}

} // namespace vkcompute
