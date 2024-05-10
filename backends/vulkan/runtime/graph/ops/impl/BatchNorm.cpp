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

#include <iostream>

namespace vkcompute {

ValueRef enforce_width_packing(ComputeGraph& graph, ValueRef arg) {
  if (graph.memory_layout_of(arg) == api::kWidthPacked) {
    return arg;
  }

  std::vector<int64_t> sizes = graph.get_tensor(arg)->sizes();

  ValueRef w_packed = graph.add_tensor(sizes, api::kFloat, api::kWidthPacked);

  auto viewFn = VK_GET_OP_FN("aten.view_copy.default");
  viewFn(graph, {arg, graph.add_none(), w_packed});

  return w_packed;
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
  std::cout << "AAAA" << std::endl;

  if (graph.val_is_none(weight_ref)) {
    VK_THROW("native_batch_norm requires weight to be non-None");
  }

  VK_CHECK_COND(
      !graph.val_is_none(weight_ref),
      "native_batch_norm requires weight to be non-None");
  VK_CHECK_COND(
      !graph.val_is_none(bias_ref),
      "native_batch_norm requires bias to be non-None");
  VK_CHECK_COND(
      !graph.val_is_none(mean_ref),
      "native_batch_norm requires mean to be non-None");
  VK_CHECK_COND(
      !graph.val_is_none(var_ref),
      "native_batch_norm requires var to be non-None");

  // batch_norm's param are broadcasted on the channel dimension.
  // In this implementation, we pack the weights along the x dimension, and
  // in the shader, we lookup using the along the x.
  ValueRef arg_weight =
      prepack_if_tensor_ref(graph, weight_ref, api::kWidthPacked);
  ValueRef arg_bias = prepack_if_tensor_ref(graph, bias_ref, api::kWidthPacked);
  ValueRef arg_mean = prepack_if_tensor_ref(graph, mean_ref, api::kWidthPacked);
  ValueRef arg_var = prepack_if_tensor_ref(graph, var_ref, api::kWidthPacked);
  float epsilon = graph.extract_scalar<float>(eps_ref);

  arg_weight = enforce_width_packing(graph, arg_weight);
  arg_bias = enforce_width_packing(graph, arg_bias);

  arg_mean = enforce_width_packing(graph, arg_mean);
  arg_var = enforce_width_packing(graph, arg_var);

  vTensorPtr t_in = graph.get_tensor(in_ref);

  const auto out_tuple_val = graph.get_value_list(out_tuple_ref);

  ValueRef out_ref = out_tuple_val->at(0);

  VK_CHECK_COND(!graph.val_is_tref(out_ref), "Output should not be tref");
  vTensorPtr t_out = graph.get_tensor(out_ref);

  VK_CHECK_COND(t_in->dim() == 4, "BatchNorm only support 4d tensor");
  VK_CHECK_COND(t_out->dim() == 4, "BatchNorm only support 4d tensor");

  int64_t num_channels = dim_at<kChannel4D>(t_in->sizes());
  VK_CHECK_COND(
      dim_at<kChannel4D>(t_out->sizes()) == num_channels,
      "out channel must match in channel");
  VK_CHECK_COND(graph.get_tensor(arg_weight)->size(0) == num_channels);
  VK_CHECK_COND(graph.get_tensor(arg_bias)->size(0) == num_channels);
  VK_CHECK_COND(graph.get_tensor(arg_mean)->size(0) == num_channels);
  VK_CHECK_COND(graph.get_tensor(arg_var)->size(0) == num_channels);

  std::string kernel_name = "batchnorm";
  add_dtype_suffix(kernel_name, *t_out);

  api::utils::uvec3 global_size = t_out->extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  int32_t num_texel_per_batch = static_cast<int32_t>(
      std::ceil(static_cast<float>(dim_at<kChannel4D>(t_in->sizes())) / 4));

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      {{out_ref, api::MemoryAccessType::WRITE},
       {{in_ref, arg_weight, arg_bias, arg_mean, arg_var},
        api::MemoryAccessType::READ}},
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
