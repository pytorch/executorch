/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_upsample_nearest2d_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr self = graph->get_tensor(args[1].refs[0]);
  std::vector<int64_t> out_sizes = self->sizes(); // NCHW

  const ValueRef output_sizes = extra_args[0]; // HW
  const ValueRef scale_factors = extra_args[1]; // HW
  if (!graph->val_is_none(output_sizes)) {
    IntListPtr output_size_ref = graph->get_int_list(output_sizes);
    out_sizes.at(2) = output_size_ref->at(0);
    out_sizes.at(3) = output_size_ref->at(1);
  } else {
    DoubleListPtr scales = graph->get_double_list(scale_factors);
    out_sizes.at(2) *= scales->at(0);
    out_sizes.at(3) *= scales->at(1);
  }

  out->virtual_resize(out_sizes);
}

// ExecuTorch-Vulkan framework to add node
// Args:
//   in: will be converted from NCHW input tensor to 3D ARGB representation in
//   openGL (via ExecuTorch) output_sizes: optional 2D array of targetting
//   output size of H and W dimensions. >= input sizes;

//      will be computed if only given the scale_factors.
//   scale_factors: optional 2D array of scale factors for H and W dimensions.
//      Will be computed if only given the output_sizes.
void add_upsample_nearest2d_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef output_sizes,
    const ValueRef scale_factors,
    const ValueRef out) {
  if (graph.val_is_none(output_sizes) && graph.val_is_none(scale_factors)) {
    VK_THROW(
        "Invalid input, must provide either output_sizes or scale_factors");
  }
  if (!graph.val_is_none(output_sizes) && !graph.val_is_none(scale_factors)) {
    VK_THROW(
        "Invalid input, must provide ONLY one of output_sizes or scale_factors");
  }

  ValueRef arg_in = prepack_if_tensor_ref(graph, in);

  vTensorPtr t_in = graph.get_tensor(in);
  utils::uvec3 input_sizes = t_in->image_extents();

  utils::ivec2 input_size = {
      utils::safe_downcast<int32_t>(input_sizes[0]),
      utils::safe_downcast<int32_t>(input_sizes[1])};
  utils::vec2 rev_scales = {
      utils::safe_downcast<float>(1.0), utils::safe_downcast<float>(1.0)};

  // Reverse scale factors that pre-computed before GLSL.
  if (!graph.val_is_none(output_sizes)) {
    auto output_size_ref = graph.get_int_list(output_sizes);
    rev_scales = {
        utils::safe_downcast<float>(
            (float)input_size[0] / output_size_ref->at(1)),
        utils::safe_downcast<float>(
            (float)input_size[1] / output_size_ref->at(0))};

  } else {
    auto scales = graph.get_double_list(scale_factors);
    rev_scales = {
        utils::safe_downcast<float>(1.0 / scales->at(1)),
        utils::safe_downcast<float>(1.0 / scales->at(0))};
  }

  vTensorPtr t_out = graph.get_tensor(out);

  std::string kernel_name("upsample_nearest2d");
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
      // Inputs and Outputs
      {{out, vkapi::MemoryAccessType::WRITE},
       {arg_in, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      {t_out->texture_limits_ubo(),
       graph.create_params_buffer(input_size),
       graph.create_params_buffer(rev_scales)},
      // Specialization Constants
      {},
      resize_upsample_nearest2d_node,
      {output_sizes, scale_factors}));
}

void upsample(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_upsample_nearest2d_node(graph, args[0], args[1], args[2], args[3]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.upsample_nearest2d.vec, upsample);
}

} // namespace vkcompute
