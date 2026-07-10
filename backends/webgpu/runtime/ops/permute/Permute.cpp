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
#include <executorch/backends/webgpu/runtime/ops/permute/permute_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct PermuteParams {
  uint32_t perm[kTensorMetaMaxNdim];
};
static_assert(
    sizeof(PermuteParams) == 16,
    "PermuteParams must match the WGSL Params vec4<u32> (16 bytes)");

// permute: out coord d -> in coord perm[d] (Vulkan permute_buffer.glsl, NCHW).
void permute_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [self, dims, out]; out is the last value-id.
  const int in_id = args.at(0);
  const int dims_id = args.at(1);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("permute: in/out arg is not a tensor");
  }
  if (graph.get_value_type(dims_id) != WebGPUGraph::ValueType::IntList) {
    throw std::runtime_error("permute: dims arg is not an IntList");
  }

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  const int ndim = static_cast<int>(in_tensor.dims.size());

  const std::vector<int64_t>& dims = graph.get_int_list(dims_id);
  if (static_cast<int>(dims.size()) != ndim ||
      static_cast<int>(out_tensor.dims.size()) != ndim) {
    throw std::runtime_error("permute: perm length != input/output rank");
  }

  // Normalize negative dims and verify perm is a permutation of [0, ndim).
  uint32_t perm[kTensorMetaMaxNdim];
  bool seen[kTensorMetaMaxNdim] = {};
  if (ndim > static_cast<int>(kTensorMetaMaxNdim)) {
    throw std::runtime_error("permute: tensor rank exceeds 4 (MAX_NDIM)");
  }
  for (int d = 0; d < ndim; d++) {
    int64_t p = dims[d];
    if (p < 0) {
      p += ndim;
    }
    if (p < 0 || p >= ndim || seen[p]) {
      throw std::runtime_error("permute: dims is not a valid permutation");
    }
    seen[p] = true;
    perm[d] = static_cast<uint32_t>(p);
  }
  for (int d = ndim; d < static_cast<int>(kTensorMetaMaxNdim); d++) {
    perm[d] = static_cast<uint32_t>(d);
  }

  TensorMeta out_meta;
  TensorMeta in_meta;
  fill_tensor_meta(out_tensor, &out_meta);
  fill_tensor_meta(in_tensor, &in_meta);
  if (out_tensor.nbytes !=
          static_cast<size_t>(out_meta.numel) * sizeof(float) ||
      in_tensor.nbytes != static_cast<size_t>(in_meta.numel) * sizeof(float)) {
    throw std::runtime_error("permute: non-fp32 operand (nbytes != numel * 4)");
  }

  PermuteParams params = {};
  std::memcpy(params.perm, perm, sizeof(perm));

  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kPermuteWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, out_meta.numel, wg_size, "permute");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer out_meta_buf =
      utils::make_uniform(device, &out_meta, sizeof(TensorMeta));
  WGPUBuffer in_meta_buf =
      utils::make_uniform(device, &in_meta, sizeof(TensorMeta));
  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(PermuteParams));
  graph.add_uniform_buffer_bytes(
      2 * sizeof(TensorMeta) + sizeof(PermuteParams));

  // Bind group: in, out (rw), out_meta, in_meta, params (3 uniforms).
  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kPermuteWGSL,
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
          {4, WGPUBufferBindingType_Uniform, params_buf, sizeof(PermuteParams)},
      },
      &wg_size_constant,
      1);

  graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       "permute",
       workgroup_count.y});

  // Drop our refs; the bind group keeps the uniforms alive until release.
  wgpuBufferRelease(out_meta_buf);
  wgpuBufferRelease(in_meta_buf);
  wgpuBufferRelease(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.permute_copy.default, permute_impl);
  WEBGPU_REGISTER_OP(aten.permute.default, permute_impl);
}

} // namespace executorch::backends::webgpu
