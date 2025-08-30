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

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

ValueRef check_and_prepack_arg(
    ComputeGraph& graph,
    ValueRef arg_ref,
    const utils::StorageType stype,
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
  return prepack_standard(graph, arg_ref, stype, utils::kWidthPacked);
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
  const std::vector<int64_t> in_sizes = graph.sizes_of(in_ref);
  const std::vector<int64_t> out_sizes = graph.sizes_of(in_ref);

  VK_CHECK_COND(in_sizes.size() == 4, "BatchNorm only support 4d tensor");
  VK_CHECK_COND(out_sizes.size() == 4, "BatchNorm only support 4d tensor");

  // Only the first element of the return value is propagated. The remaining 2
  // elements are zero-size dummy tensor.
  const ValueRef out_ref = graph.get_value_list(out_tuple_ref)->at(0);

  const utils::StorageType stype = graph.storage_type_of(out_ref);

  const int64_t num_channels = dim_at<kChannel4D>(in_sizes);

  const ValueRef arg_weight =
      check_and_prepack_arg(graph, weight_ref, stype, num_channels, "weight");
  const ValueRef arg_bias =
      check_and_prepack_arg(graph, bias_ref, stype, num_channels, "bias");
  const ValueRef arg_mean =
      check_and_prepack_arg(graph, mean_ref, stype, num_channels, "mean");
  const ValueRef arg_var =
      check_and_prepack_arg(graph, var_ref, stype, num_channels, "var");
  const float epsilon = graph.extract_scalar<float>(eps_ref);

  VK_CHECK_COND(!graph.val_is_tref(out_ref), "Output should not be tref");

  const std::vector<int64_t> out_tensor_sizes = graph.sizes_of(out_ref);
  VK_CHECK_COND(
      dim_at<kChannel4D>(out_tensor_sizes) == num_channels,
      "out channel must match in channel");

  std::string kernel_name = "batchnorm";
  add_dtype_suffix(kernel_name, graph.dtype_of(out_ref));

  const int32_t num_texel_per_batch =
      utils::div_up_4((dim_at<kChannel4D>(in_sizes)));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      {{out_ref, vkapi::kWrite},
       {{in_ref, arg_weight, arg_bias, arg_mean, arg_var}, vkapi::kRead}},
      {graph.logical_limits_ubo(out_ref),
       graph.create_params_buffer(epsilon),
       graph.create_params_buffer(num_texel_per_batch)},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
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
