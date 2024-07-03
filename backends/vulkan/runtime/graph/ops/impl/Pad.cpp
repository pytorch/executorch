/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

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
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr self = graph->get_tensor(args[1].refs[0]);
  IntListPtr pad_vec = graph->get_int_list(extra_args[0]);
  std::vector<int64_t> in_size = self->sizes();
  int dim = in_size.size() - 1;
  for (int i = 0; i < pad_vec->size(); i += 2) {
    in_size.at(dim) += pad_vec->at(i) + pad_vec->at(i + 1);
    dim--;
  }

  out->virtual_resize(in_size);
}

void add_constant_pad_nd_node(
    ComputeGraph& graph,
    const ValueRef& in,
    const ValueRef& pad,
    const ValueRef& fill_value,
    const ValueRef& out) {
  float fill_value_val = graph.extract_scalar<float>(fill_value);
  IntListPtr pad_vec = graph.get_int_list(pad);
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);

  std::string kernel_name = "";
  PadParam pad_param = creat_pad_param(*pad_vec);

  if (pad_vec->size() <= 4) {
    kernel_name = "pad_height_width";
    kernel_name.reserve(kShaderNameReserve);
    add_dtype_suffix(kernel_name, *t_out);
  } else {
    kernel_name = "pad_channel";
    kernel_name.reserve(kShaderNameReserve);
    add_dtype_suffix(kernel_name, *t_out);
  }

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
      // Inputs and Outputs
      {{out, vkapi::MemoryAccessType::WRITE},
       {in, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      {t_out->sizes_ubo(),
       t_in->sizes_ubo(),
       graph.create_params_buffer(pad_param),
       graph.create_params_buffer(fill_value_val)},
      // Specialization Constants
      {},
      resize_constant_pad_node,
      {pad}));
}

void constant_pad_nd(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_constant_pad_nd_node(graph, args[0], args[1], args[2], args[3]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.constant_pad_nd.default, constant_pad_nd);
}

} // namespace vkcompute
