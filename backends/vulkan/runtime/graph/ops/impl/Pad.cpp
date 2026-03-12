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

struct PadParam final {
  int32_t left;
  int32_t top;
  int32_t front;
};

PadParam creat_pad_param(const std::vector<int64_t>& pad) {
  if (pad.size() == 2) {
    return PadParam{static_cast<int32_t>(pad[0]), 0, 0};
  } else if (pad.size() == 4) {
    return PadParam{
        static_cast<int32_t>(pad[0]), static_cast<int32_t>(pad[2]), 0};
  } else if (pad.size() == 6) {
    return PadParam{
        static_cast<int32_t>(pad[0]),
        static_cast<int32_t>(pad[2]),
        static_cast<int32_t>(pad[4])};
  } else {
    VK_THROW("invalid pad form");
  }
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
  const PadParam pad_param = creat_pad_param(*pad_vec);

  std::string kernel_name = "pad";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  vkapi::ParamsBindList param_ubos;
  if (graph.is_buffer_storage(out)) {
    // BufferMetadata stores sizes/strides in WHCN order (flip_and_unsqueeze
    // reverses from NCHW). Map pad offsets to match: W=0, H=1, C=2.
    utils::ivec4 pad_per_dim{pad_param.left, pad_param.top, pad_param.front, 0};
    param_ubos = {
        graph.buffer_meta_ubo(out),
        graph.buffer_meta_ubo(in),
        graph.create_params_buffer(pad_per_dim),
        graph.create_params_buffer(fill_value_val)};
  } else {
    param_ubos = {
        graph.meta_ubo(out),
        graph.meta_ubo(in),
        graph.create_params_buffer(pad_param),
        graph.create_params_buffer(fill_value_val)};
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Parameter buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
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
