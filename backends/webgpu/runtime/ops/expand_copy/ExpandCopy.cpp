/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/TensorMeta.h>
#include <executorch/backends/webgpu/runtime/ops/expand_copy/expand_copy_wgsl.h>

#include <webgpu/webgpu.h>

#include <limits>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

// out coord -> in coord; size-1 in-dims broadcast via clamp (mirrors mul).
void expand_copy_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int in_id = args.at(0);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("expand_copy: in/out arg is not a tensor");
  }

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);

  if (graph.get_value_type(args.at(1)) != WebGPUGraph::ValueType::IntList) {
    throw std::runtime_error(
        "WebGPU expand_copy: dynamic target sizes are unsupported");
  }
  for (int64_t target_size : graph.get_int_list(args.at(1))) {
    if (target_size == -1) {
      throw std::runtime_error(
          "WebGPU expand_copy: inferred target sizes are unsupported");
    }
  }
  if (graph.tensor_has_dynamic_dims(in_id) ||
      graph.tensor_has_dynamic_dims(out_id)) {
    throw std::runtime_error(
        "WebGPU expand_copy: dynamic shapes are unsupported");
  }

  TensorMeta out_meta;
  TensorMeta in_meta;
  fill_tensor_meta(out_tensor, &out_meta);
  fill_tensor_meta_broadcast(in_tensor, out_meta.ndim, &in_meta);
  if (out_tensor.nbytes !=
          static_cast<size_t>(out_meta.numel) * sizeof(float) ||
      in_tensor.nbytes != static_cast<size_t>(in_meta.numel) * sizeof(float)) {
    throw std::runtime_error(
        "expand_copy: non-fp32 operand (nbytes != numel*4)");
  }
  if (out_meta.numel >
      static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error(
        "WebGPU expand_copy: element count exceeds the flattened 2D dispatch "
        "limit");
  }

  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kExpandCopyWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, out_meta.numel, wg_size, "expand_copy");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer out_meta_buf = graph.create_params_buffer(out_meta);
  WGPUBuffer in_meta_buf = graph.create_params_buffer(in_meta);

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kExpandCopyWGSL,
      {
          {0,
           WGPUBufferBindingType_ReadOnlyStorage,
           in_tensor.buffer,
           in_tensor.nbytes},
          {1,
           WGPUBufferBindingType_Storage,
           out_tensor.buffer,
           out_tensor.nbytes},
          {2, WGPUBufferBindingType_Uniform, out_meta_buf, sizeof(TensorMeta)},
          {3, WGPUBufferBindingType_Uniform, in_meta_buf, sizeof(TensorMeta)},
      },
      &wg_size_constant,
      1);

  graph.add_dispatch_2d(
      bundle.pipeline, bundle.bind_group, workgroup_count.x, workgroup_count.y);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.expand_copy.default, expand_copy_impl);
}

} // namespace executorch::backends::webgpu
