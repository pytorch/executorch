/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

std::vector<int64_t> calc_out_mean_sizes(
    api::vTensor& self,
    int64_t normalized_shape_dim) {
  std::vector<int64_t> output_size = self.sizes();
  int64_t self_dim = self.sizes().size();
  for (int64_t i = 0; i < normalized_shape_dim; ++i) {
    output_size.at(self_dim - i - 1) = 1;
  }
  return output_size;
}

void resize_native_layer_norm_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr mean = graph->get_tensor(args[0].refs[1]);
  vTensorPtr rstd = graph->get_tensor(args[0].refs[2]);
  vTensorPtr in = graph->get_tensor(args[1].refs[0]);
  std::vector<int64_t> in_sizes = in->sizes();

  const auto normalized_shape_dim = graph->get_int_list(extra_args[0])->size();

  std::vector<int64_t> mean_size =
      calc_out_mean_sizes(*in, normalized_shape_dim);

  out->virtual_resize(in_sizes);
  mean->virtual_resize(mean_size);
  rstd->virtual_resize(mean_size);
}

void add_native_layer_norm_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef normalized_shape,
    const ValueRef weight_data,
    const ValueRef bias_data,
    const ValueRef eps,
    const ValueRef out) {
  const auto normalized_shape_dim =
      graph.get_int_list(normalized_shape)->size();
  if (normalized_shape_dim > 1) {
    VK_THROW("native_layer_norm only supports normalized_shape with dim == 1");
  }

  if (graph.val_is_none(weight_data)) {
    VK_THROW("native_layer_norm requires weight to be non-None");
  }

  if (graph.val_is_none(bias_data)) {
    VK_THROW("native_layer_norm requires bias to be non-None");
  }

  ValueRef arg_weight = prepack_standard_like(graph, weight_data, in);
  ValueRef arg_bias = prepack_standard_like(graph, bias_data, in);

  const auto out_val = graph.get_value_list(out);
  vTensorPtr t_out = graph.get_tensor(out_val->at(0));
  vTensorPtr t_mean = graph.get_tensor(out_val->at(1));
  vTensorPtr t_input = graph.get_tensor(in);
  float epsilon = graph.extract_scalar<float>(eps);

  VK_CHECK_COND(check_same_packed_dim(*t_input, *t_out));

  std::vector<int64_t> in_sizes = t_input->sizes();

  utils::uvec3 global_size = t_out->logical_limits();
  utils::uvec3 local_size;

  // Since the shader sets shared memory scale factor > 1, if dispatch is
  // greater than maximum WG size. Setting WG size in X axis to max WG size,
  // would allow best thread utilization.
  if (global_size[0] > 64) {
    local_size = {64, 1, 1};
  } else {
    // If thread size in X axis is smaller or equal to maximum WG size, we can
    // let the function decide the best WG size.
    local_size = graph.create_local_wg_size(global_size);
  }

  std::string kernel_name("native_layer_norm");
  kernel_name.reserve(kShaderNameReserve);

  add_dtype_suffix(kernel_name, *t_out);

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{{out_val->at(0), out_val->at(1), out_val->at(2)},
        vkapi::MemoryAccessType::WRITE},
       {{in, arg_weight, arg_bias}, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      {},
      // Specialization Constants
      {
          t_input->hashed_layout(),
          t_out->hashed_layout(),
      },
      // Resizing Logic
      resize_native_layer_norm_node,
      {normalized_shape},
      {
          graph.logical_limits_pc_of(out_val->at(0)),
          graph.sizes_pc_of(out_val->at(0)),
          PushConstantDataInfo(&epsilon, sizeof(epsilon)),
      }));
}

void native_layer_norm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_native_layer_norm_node(
      graph, args[0], args[1], args[2], args[3], args[4], args[5]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.native_layer_norm.default, native_layer_norm);
}

} // namespace vkcompute
