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
#include <executorch/backends/webgpu/runtime/ops/flip/flip_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct FlipParams {
  uint32_t flip[kTensorMetaMaxNdim];
};
static_assert(
    sizeof(FlipParams) == 32,
    "FlipParams must match the WGSL Params array<vec4<u32>, 2> (32 bytes)");

// flip: reverse coords along dims (Vulkan Flip.cpp, NCHW; any 4-byte dtype).
void flip_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [self, dims, out]; dims = dims to reverse; out is the last value-id.
  const int in_id = args.at(0);
  const int dims_id = args.at(1);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("flip: in/out arg is not a tensor");
  }
  if (graph.get_value_type(dims_id) != WebGPUGraph::ValueType::IntList) {
    throw std::runtime_error("flip: dims arg is not an IntList");
  }

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  const int ndim = static_cast<int>(in_tensor.dims.size());
  if (ndim > static_cast<int>(kTensorMetaMaxNdim)) {
    throw std::runtime_error("flip: tensor rank exceeds 8 (MAX_NDIM)");
  }
  // flip preserves shape: the output rank + dims must equal the input's, else
  // the kernel unravels out coords against a shape it wasn't reversed for.
  if (static_cast<int>(out_tensor.dims.size()) != ndim) {
    throw std::runtime_error("flip: output rank != input rank");
  }
  for (int i = 0; i < ndim; i++) {
    if (out_tensor.dims[i] != in_tensor.dims[i]) {
      throw std::runtime_error("flip: output shape != input shape");
    }
  }

  // Build the flip bitmap: 1 for each (normalized) dim to reverse.
  const std::vector<int64_t>& dims = graph.get_int_list(dims_id);
  uint32_t flip[kTensorMetaMaxNdim] = {};
  bool seen[kTensorMetaMaxNdim] = {};
  for (int64_t dv : dims) {
    int64_t d = dv < 0 ? dv + ndim : dv;
    if (d < 0 || d >= ndim) {
      throw std::runtime_error("flip: dim out of range");
    }
    if (seen[static_cast<size_t>(d)]) {
      throw std::runtime_error("flip: duplicate dim");
    }
    seen[static_cast<size_t>(d)] = true;
    flip[static_cast<size_t>(d)] = 1u;
  }

  TensorMeta out_meta;
  TensorMeta in_meta;
  fill_tensor_meta(out_tensor, &out_meta);
  fill_tensor_meta(in_tensor, &in_meta);
  if (out_tensor.nbytes !=
          static_cast<size_t>(out_meta.numel) * sizeof(float) ||
      in_tensor.nbytes != static_cast<size_t>(in_meta.numel) * sizeof(float)) {
    throw std::runtime_error("flip: non-4-byte operand (nbytes != numel * 4)");
  }

  FlipParams params = {};
  std::memcpy(params.flip, flip, sizeof(flip));

  uint32_t wg_size = utils::clamp_workgroup_size(device, kFlipWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, out_meta.numel, wg_size, "flip");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer out_meta_buf =
      utils::make_uniform(device, &out_meta, sizeof(TensorMeta));
  WGPUBuffer in_meta_buf =
      utils::make_uniform(device, &in_meta, sizeof(TensorMeta));
  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(FlipParams));
  graph.add_uniform_buffer_bytes(2 * sizeof(TensorMeta) + sizeof(FlipParams));

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kFlipWGSL,
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
          {4, WGPUBufferBindingType_Uniform, params_buf, sizeof(FlipParams)},
      },
      &wg_size_constant,
      1);

  graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       "flip",
       workgroup_count.y});

  // Drop our refs; the bind group keeps the uniforms alive until release.
  wgpuBufferRelease(out_meta_buf);
  wgpuBufferRelease(in_meta_buf);
  wgpuBufferRelease(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.flip.default, flip_impl);
}

} // namespace executorch::backends::webgpu
