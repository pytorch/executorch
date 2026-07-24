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
#include <executorch/backends/webgpu/runtime/ops/gather/gather_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

struct GatherParams {
  uint32_t dim;
  uint32_t _pad[3];
};

// gather: out[c] = self[c] with c[dim] replaced by index[c].
void gather_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int self_id = args.at(0);
  const int dim_id = args.at(1);
  const int index_id = args.at(2);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(dim_id) != WebGPUGraph::ValueType::Int) {
    throw std::runtime_error("gather: dim is not an int");
  }
  WGPUDevice device = graph.device();
  const auto& self_tensor = graph.get_tensor(self_id);
  const auto& index_tensor = graph.get_tensor(index_id);
  const auto& out_tensor = graph.get_tensor(out_id);

  if (self_tensor.buffer == nullptr || index_tensor.buffer == nullptr ||
      out_tensor.buffer == nullptr) {
    throw std::runtime_error("gather: null buffer binding");
  }

  const int64_t ndim = static_cast<int64_t>(out_tensor.dims.size());
  int64_t dim = graph.get_int(dim_id);
  if (dim < 0) {
    dim += ndim;
  }
  if (ndim == 0 || dim < 0 || dim >= ndim) {
    throw std::runtime_error("gather: dim out of range");
  }

  TensorMeta out_meta;
  TensorMeta self_meta;
  fill_tensor_meta(out_tensor, &out_meta);
  fill_tensor_meta(self_tensor, &self_meta);
  if (out_meta.ndim != self_meta.ndim) {
    throw std::runtime_error("gather: self/out rank mismatch");
  }

  const size_t out_numel = out_tensor.nbytes / sizeof(float);
  const size_t index_numel = index_tensor.nbytes / sizeof(int32_t);
  if (out_tensor.nbytes != out_numel * sizeof(float) ||
      self_tensor.nbytes % sizeof(float) != 0 ||
      index_tensor.nbytes != index_numel * sizeof(int32_t)) {
    throw std::runtime_error("gather: fp32 self/out + i32 index required");
  }
  if (out_numel != index_numel) {
    throw std::runtime_error("gather: out numel != index numel");
  }

  uint32_t wg_size = utils::clamp_workgroup_size(device, kGatherWorkgroupSizeX);
  uint32_t workgroup_count = utils::compute_1d_workgroup_count(
      device, out_meta.numel, wg_size, "gather");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  GatherParams params = {};
  params.dim = static_cast<uint32_t>(dim);

  WGPUBuffer out_meta_buf =
      utils::make_uniform(device, &out_meta, sizeof(TensorMeta));
  WGPUBuffer self_meta_buf =
      utils::make_uniform(device, &self_meta, sizeof(TensorMeta));
  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(GatherParams));
  graph.add_uniform_buffer_bytes(2 * sizeof(TensorMeta) + sizeof(GatherParams));

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kGatherWGSL,
      {
          {0,
           WGPUBufferBindingType_ReadOnlyStorage,
           self_tensor.buffer,
           self_tensor.nbytes},
          {1,
           WGPUBufferBindingType_ReadOnlyStorage,
           index_tensor.buffer,
           index_tensor.nbytes},
          {2,
           WGPUBufferBindingType_Storage,
           out_tensor.buffer,
           out_tensor.nbytes},
          {3, WGPUBufferBindingType_Uniform, out_meta_buf, sizeof(TensorMeta)},
          {4, WGPUBufferBindingType_Uniform, self_meta_buf, sizeof(TensorMeta)},
          {5, WGPUBufferBindingType_Uniform, params_buf, sizeof(GatherParams)},
      },
      &wg_size_constant,
      1);

  graph.add_dispatch({bundle.pipeline, bundle.bind_group, workgroup_count});

  wgpuBufferRelease(out_meta_buf);
  wgpuBufferRelease(self_meta_buf);
  wgpuBufferRelease(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.gather.default, gather_impl);
}

} // namespace executorch::backends::webgpu
