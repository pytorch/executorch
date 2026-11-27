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
#include <executorch/backends/webgpu/runtime/ops/index_select/index_select_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct IndexSelectParams {
  uint32_t info[4]; // info[0] = dim
};
static_assert(
    sizeof(IndexSelectParams) == 16,
    "IndexSelectParams must match the WGSL Params vec4<u32> (16 bytes)");

// index_select: gather rows along dim via an int index (Vulkan IndexSelect).
void index_select_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [self, dim, index, out]; index is an int32 tensor (downcast_64_bit).
  const int self_id = args.at(0);
  const int dim_id = args.at(1);
  const int index_id = args.at(2);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(self_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(index_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("index_select: self/index/out is not a tensor");
  }

  WGPUDevice device = graph.device();
  const auto& self_tensor = graph.get_tensor(self_id);
  const auto& index_tensor = graph.get_tensor(index_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (self_tensor.buffer == nullptr || index_tensor.buffer == nullptr ||
      out_tensor.buffer == nullptr) {
    throw std::runtime_error("index_select: null buffer binding");
  }

  const int64_t ndim = static_cast<int64_t>(self_tensor.dims.size());
  if (ndim > static_cast<int64_t>(kTensorMetaMaxNdim)) {
    throw std::runtime_error("index_select: tensor rank exceeds 8 (MAX_NDIM)");
  }

  if (graph.get_value_type(dim_id) != WebGPUGraph::ValueType::Int) {
    throw std::runtime_error("index_select: dim arg is not a static Int");
  }
  int64_t dim = graph.get_int(dim_id);
  if (dim < 0) {
    dim += ndim;
  }
  if (dim < 0 || dim >= ndim) {
    throw std::runtime_error("index_select: dim out of range");
  }

  TensorMeta out_meta;
  TensorMeta in_meta;
  fill_tensor_meta(out_tensor, &out_meta);
  fill_tensor_meta(self_tensor, &in_meta);
  if (out_tensor.nbytes !=
          static_cast<size_t>(out_meta.numel) * sizeof(float) ||
      self_tensor.nbytes % sizeof(float) != 0) {
    throw std::runtime_error("index_select: non-fp32 self/out (nbytes % 4)");
  }
  // Index is the int32 downcast of the int64 index (downcast_64_bit).
  if (index_tensor.nbytes % sizeof(int32_t) != 0) {
    throw std::runtime_error("index_select: index buffer is not int32");
  }
  // The index gathers out.dims[dim] rows (one per index element), so it must
  // hold at least that many entries.
  uint64_t index_numel = 1;
  for (int64_t d : index_tensor.dims) {
    index_numel *= static_cast<uint64_t>(d);
  }
  if (index_numel < static_cast<uint64_t>(out_tensor.dims.at(dim))) {
    throw std::runtime_error("index_select: index numel < out.dims[dim]");
  }

  IndexSelectParams params = {};
  params.info[0] = static_cast<uint32_t>(dim);

  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kIndexSelectWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, out_meta.numel, wg_size, "index_select");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer out_meta_buf =
      utils::make_uniform(device, &out_meta, sizeof(TensorMeta));
  WGPUBuffer in_meta_buf =
      utils::make_uniform(device, &in_meta, sizeof(TensorMeta));
  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(IndexSelectParams));
  graph.add_uniform_buffer_bytes(
      2 * sizeof(TensorMeta) + sizeof(IndexSelectParams));

  // in, out (rw), index (read i32), out_meta, in_meta, params (3 uniforms).
  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kIndexSelectWGSL,
      {
          {0,
           WGPUBufferBindingType_ReadOnlyStorage,
           self_tensor.buffer,
           self_tensor.nbytes},
          {1,
           WGPUBufferBindingType_Storage,
           out_tensor.buffer,
           out_tensor.nbytes},
          {2,
           WGPUBufferBindingType_ReadOnlyStorage,
           index_tensor.buffer,
           index_tensor.nbytes},
          {3, WGPUBufferBindingType_Uniform, out_meta_buf, sizeof(TensorMeta)},
          {4, WGPUBufferBindingType_Uniform, in_meta_buf, sizeof(TensorMeta)},
          {5,
           WGPUBufferBindingType_Uniform,
           params_buf,
           sizeof(IndexSelectParams)},
      },
      &wg_size_constant,
      1);

  // Static shapes only: index_select registers no resize hook, so the output
  // extent (out.dims[dim] == index numel) is fixed at build time.
  graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       "index_select",
       workgroup_count.y});

  // Drop our refs; the bind group keeps the uniforms alive until release.
  wgpuBufferRelease(out_meta_buf);
  wgpuBufferRelease(in_meta_buf);
  wgpuBufferRelease(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.index_select.default, index_select_impl);
}

} // namespace executorch::backends::webgpu
