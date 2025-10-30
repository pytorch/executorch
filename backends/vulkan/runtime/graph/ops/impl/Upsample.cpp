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

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

enum class UpsampleMode : int { NEAREST, BILINEAR };

void resize_upsample_nearest2d_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);
  std::vector<int64_t> out_sizes = graph->sizes_of(self); // NCHW

  const ValueRef output_sizes = extra_args.at(0); // HW
  const ValueRef scale_factors = extra_args.at(1); // HW
  if (!graph->val_is_none(output_sizes)) {
    IntListPtr output_size_ref = graph->get_int_list(output_sizes);
    out_sizes.at(2) = output_size_ref->at(0);
    out_sizes.at(3) = output_size_ref->at(1);
  } else {
    DoubleListPtr scales = graph->get_double_list(scale_factors);
    out_sizes.at(2) *= scales->at(0);
    out_sizes.at(3) *= scales->at(1);
  }

  graph->virtual_resize(out, out_sizes);
}

void add_upsample_nearest2d_node(
    ComputeGraph& graph,
    const UpsampleMode mode,
    const ValueRef in,
    const ValueRef output_sizes,
    const ValueRef align_corners,
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

  int align_corners_val = 0;
  if (is_valid(align_corners) && graph.get_bool(align_corners)) {
    align_corners_val = 1;
  }

  utils::uvec3 in_limits = graph.logical_limits_of(in);
  utils::uvec3 out_limits = graph.logical_limits_of(out);

  uint32_t out_width = out_limits[0u];
  uint32_t out_height = out_limits[1u];

  float scale_factor_x = float(in_limits[0u]) / float(out_width);
  float scale_factor_y = float(in_limits[1u]) / float(out_height);

  float recip_scale_factor_x = 1.0f / scale_factor_x;
  float recip_scale_factor_y = 1.0f / scale_factor_y;

  if (!graph.val_is_none(output_sizes)) {
    IntListPtr output_size_ref = graph.get_int_list(output_sizes);
    out_width = output_size_ref->at(1);
    out_height = output_size_ref->at(0);

    VK_CHECK_COND(out_width == out_limits[0u]);
    VK_CHECK_COND(out_height == out_limits[1u]);

  } else {
    DoubleListPtr scales = graph.get_double_list(scale_factors);
    scale_factor_x = scales->at(1);
    scale_factor_y = scales->at(0);

    VK_CHECK_COND(in_limits[0u] * scale_factor_x == out_width);
    VK_CHECK_COND(in_limits[1u] * scale_factor_y == out_height);
  }

  if (align_corners_val == 1) {
    recip_scale_factor_x = float(in_limits[0u] - 1) / float(out_width - 1);
    recip_scale_factor_y = float(in_limits[1u] - 1) / float(out_height - 1);
  } else {
    recip_scale_factor_x = float(in_limits[0u]) / float(out_width);
    recip_scale_factor_y = float(in_limits[1u]) / float(out_height);
  }

  utils::vec2 recip_scales = {recip_scale_factor_x, recip_scale_factor_y};

  std::string kernel_name;
  kernel_name.reserve(kShaderNameReserve);
  switch (mode) {
    case UpsampleMode::NEAREST:
      kernel_name = "upsample_nearest2d";
      break;
    case UpsampleMode::BILINEAR:
      kernel_name = "upsample_bilinear2d";
      break;
  }
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::MemoryAccessType::WRITE},
       {in, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      {graph.logical_limits_ubo(out),
       graph.logical_limits_ubo(in),
       graph.create_params_buffer(recip_scales)},
      // Push Constants
      {},
      // Specialization Constants
      {align_corners_val},
      // Resize Args
      {output_sizes, scale_factors},
      // Resizing Logic
      resize_upsample_nearest2d_node));
}

void upsample_nearest2d(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  return add_upsample_nearest2d_node(
      graph,
      UpsampleMode::NEAREST,
      args[0],
      args[1],
      kDummyValueRef,
      args[2],
      args[3]);
}

void upsample_bilinear2d(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  return add_upsample_nearest2d_node(
      graph,
      UpsampleMode::BILINEAR,
      args[0],
      args[1],
      args[2],
      args[3],
      args[4]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.upsample_nearest2d.vec, upsample_nearest2d);
  VK_REGISTER_OP(aten.upsample_bilinear2d.vec, upsample_bilinear2d);
}

} // namespace vkcompute
