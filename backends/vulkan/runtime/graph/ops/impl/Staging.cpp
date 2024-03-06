/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OpUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/StagingUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/Utils.h>

namespace at {
namespace native {
namespace vulkan {

StagingParams create_staging_params(const vTensor& t) {
  int32_t height = api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Height>(t));
  int32_t width = api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Width>(t));
  int32_t channels =
      api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Channel>(t));

  int32_t plane_size = height * width;
  int32_t c_depth = api::utils::div_up(channels, 4);

  return {
      api::utils::make_ivec3(t.extents()),
      plane_size,
      {c_depth, channels},
  };
}

void add_staging_to_tensor_node(
    ComputeGraph& graph,
    const ValueRef in_staging,
    const ValueRef out_tensor) {
  vTensor& t_out = graph.get_val(out_tensor).toTensor();
  VK_CHECK_COND(graph.get_val(in_staging).isStaging());

  api::ShaderInfo shader = get_nchw_to_image_shader(t_out);

  api::utils::uvec3 global_size = t_out.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  api::UniformParamsBuffer params(
      graph.context(), create_staging_params(t_out));

  graph.execute_nodes().emplace_back(new ExecuteNode(
      shader,
      global_size,
      local_size,
      {{out_tensor, api::MemoryAccessType::WRITE},
       {in_staging, api::MemoryAccessType::READ}},
      std::move(params)));
}

void add_tensor_to_staging_node(
    ComputeGraph& graph,
    const ValueRef in_tensor,
    const ValueRef out_staging) {
  vTensor& t_in = graph.get_val(in_tensor).toTensor();
  VK_CHECK_COND(graph.get_val(out_staging).isStaging());

  api::ShaderInfo shader = get_image_to_nchw_shader(t_in);

  api::utils::uvec3 global_size = t_in.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  StagingParams sp = create_staging_params(t_in);
  api::UniformParamsBuffer params(graph.context(), sp);

  // TODO(T181194784): These are workgroup sizes for special cases. Refactor the
  // calculation of workgroup sizes to a standalone function. We should use
  // scalar type to get the shader name, and use the shader name to get the
  // workgroup size.
  if (t_in.dtype() == api::ScalarType::QUInt8 ||
      t_in.dtype() == api::ScalarType::QInt8 || t_in.dtype() == api::kBool) {
    if (sp.plane_size % 4 == 0) {
      global_size.data[0u] = sp.plane_size / 4;
      global_size.data[1u] = 1;
      local_size.data[0u] *= local_size.data[1u];
      local_size.data[1u] = 1;
    } else {
      uint32_t numel = t_in.numel();
      global_size = {api::utils::div_up(numel, uint32_t(4)), 1u, 1u};
      local_size = {64u, 1u, 1u};
    }
  }

  graph.execute_nodes().emplace_back(new ExecuteNode(
      shader,
      global_size,
      local_size,
      {{in_tensor, api::MemoryAccessType::READ},
       {out_staging, api::MemoryAccessType::WRITE}},
      std::move(params)));
}

ValueRef prepack(ComputeGraph& graph, const ValueRef vref) {
  TensorRef& tref = graph.get_val(vref).toTensorRef();
  ValueRef v = graph.add_tensor(tref.sizes, tref.dtype);
  vTensor t = graph.get_val(v).toTensor();

  api::ShaderInfo shader = get_nchw_to_image_shader(t);

  api::utils::uvec3 global_size = t.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  StagingParams sp = create_staging_params(t);
  api::UniformParamsBuffer params(graph.context(), sp);

  graph.prepack_nodes().emplace_back(new PrepackNode(
      shader, global_size, local_size, vref, v, std::move(params)));

  return v;
}

ValueRef prepack_if_tensor_ref(ComputeGraph& graph, const ValueRef v) {
  if (graph.get_val(v).isTensorRef()) {
    return prepack(graph, v);
  } else {
    return v;
  }
}

} // namespace vulkan
} // namespace native
} // namespace at
