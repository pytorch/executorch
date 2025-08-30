/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

std::vector<int64_t> calc_out_mean_sizes(
    const std::vector<int64_t>& self_sizes,
    int64_t normalized_shape_dim) {
  std::vector<int64_t> output_size = self_sizes;
  int64_t self_dim = self_sizes.size();
  for (int64_t i = 0; i < normalized_shape_dim; ++i) {
    output_size.at(self_dim - i - 1) = 1;
  }
  return output_size;
}

void resize_native_layer_norm_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef mean = args.at(0).refs.at(1);
  const ValueRef rstd = args.at(0).refs.at(2);
  const ValueRef in = args.at(1).refs.at(0);
  const std::vector<int64_t> in_sizes = graph->sizes_of(in);

  const auto normalized_shape_dim =
      graph->get_int_list(extra_args.at(0))->size();

  const std::vector<int64_t> mean_size =
      calc_out_mean_sizes(in_sizes, normalized_shape_dim);

  graph->virtual_resize(out, in_sizes);
  graph->virtual_resize(mean, mean_size);
  graph->virtual_resize(rstd, mean_size);
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
  const ValueRef out_tensor = out_val->at(0);
  const ValueRef mean_tensor = out_val->at(1);
  const ValueRef rstd_tensor = out_val->at(2);

  float epsilon = graph.extract_scalar<float>(eps);

  VK_CHECK_COND(check_same_packed_dim(graph, in, out_tensor));

  const std::vector<int64_t> in_sizes = graph.sizes_of(in);

  utils::uvec3 global_size = graph.logical_limits_of(out_tensor);
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

  add_dtype_suffix(kernel_name, graph.dtype_of(out_tensor));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{{out_tensor, mean_tensor, rstd_tensor}, vkapi::kWrite},
       {{in, arg_weight, arg_bias}, vkapi::kRead}},
      // Shader params buffers
      {},
      // Push Constants
      {
          graph.logical_limits_pc_of(out_tensor),
          graph.sizes_pc_of(out_tensor),
          PushConstantDataInfo(&epsilon, sizeof(epsilon)),
      },
      // Specialization Constants
      {
          graph.hashed_layout_of(in),
          graph.hashed_layout_of(out_tensor),
      },
      // Resize Args
      {normalized_shape},
      // Resizing Logic
      resize_native_layer_norm_node));
}

void native_layer_norm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_native_layer_norm_node(
      graph, args[0], args[1], args[2], args[3], args[4], args[5]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.native_layer_norm.default, native_layer_norm);
}

} // namespace vkcompute
