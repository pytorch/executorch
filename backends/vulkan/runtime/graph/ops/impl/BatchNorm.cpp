/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

ValueRef prepack_arg(
    ComputeGraph& graph,
    ValueRef arg_ref,
    int64_t num_channels,
    const std::string& debug_name) {
  VK_CHECK_COND(
      graph.val_is_tref(arg_ref),
      "native_batch_norm requires ",
      debug_name,
      " to be a constant tensorref");
  VK_CHECK_COND(graph.get_tref(arg_ref)->sizes[0] == num_channels);

  // batch_norm's param are broadcasted on the channel dimension.
  // In this implementation, we pack the weights along the x dimension, and
  // in the shader, we lookup using the along the x.
  return prepack_if_tensor_ref(graph, arg_ref, utils::kWidthPacked);
}

void add_native_batch_norm_node(
    ComputeGraph& graph,
    ValueRef in_ref,
    ValueRef weight_ref,
    ValueRef bias_ref,
    ValueRef mean_ref,
    ValueRef var_ref,
    ValueRef eps_ref,
    ValueRef out_tuple_ref) {
  std::vector<int64_t> in_sizes = graph.get_tensor(in_ref)->sizes();
  std::vector<int64_t> out_sizes = graph.get_tensor(in_ref)->sizes();

  VK_CHECK_COND(in_sizes.size() == 4, "BatchNorm only support 4d tensor");
  VK_CHECK_COND(out_sizes.size() == 4, "BatchNorm only support 4d tensor");

  int64_t num_channels = dim_at<kChannel4D>(in_sizes);

  ValueRef arg_weight = prepack_arg(graph, weight_ref, num_channels, "weight");
  ValueRef arg_bias = prepack_arg(graph, bias_ref, num_channels, "bias");
  ValueRef arg_mean = prepack_arg(graph, mean_ref, num_channels, "mean");
  ValueRef arg_var = prepack_arg(graph, var_ref, num_channels, "var");
  float epsilon = graph.extract_scalar<float>(eps_ref);

  vTensorPtr t_in = graph.get_tensor(in_ref);

  // Only the first element of the return value is propagated. The remaining 2
  // elements are zero-size dummy tensor.
  const auto out_tuple_val = graph.get_value_list(out_tuple_ref);

  ValueRef out_ref = out_tuple_val->at(0);

  VK_CHECK_COND(!graph.val_is_tref(out_ref), "Output should not be tref");
  vTensorPtr t_out = graph.get_tensor(out_ref);

  VK_CHECK_COND(
      dim_at<kChannel4D>(t_out->sizes()) == num_channels,
      "out channel must match in channel");

  std::string kernel_name = "batchnorm";
  add_dtype_suffix(kernel_name, *t_out);

  int32_t num_texel_per_batch =
      utils::div_up_4((dim_at<kChannel4D>(t_in->sizes())));

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out_ref),
      graph.create_local_wg_size(out_ref),
      {{out_ref, vkapi::MemoryAccessType::WRITE},
       {{in_ref, arg_weight, arg_bias, arg_mean, arg_var},
        vkapi::MemoryAccessType::READ}},
      {t_out->texture_limits_ubo(),
       graph.create_params_buffer(epsilon),
       graph.create_params_buffer(num_texel_per_batch)}));
}

void native_batch_norm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // args[5] is momentum. It is not used in the calculation.
  return add_native_batch_norm_node(
      graph, args[0], args[1], args[2], args[3], args[4], args[6], args[7]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(
      aten._native_batch_norm_legit_no_training.default, native_batch_norm);
}

} // namespace vkcompute
