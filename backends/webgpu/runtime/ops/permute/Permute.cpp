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
  uint32_t workgroup_count = utils::compute_1d_workgroup_count(
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

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kPermuteWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  // Bind group: in, out (rw), out_meta, in_meta, params (3 uniforms).
  WGPUBindGroupLayoutEntry entries[5] = {};
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[1].binding = 1;
  entries[1].visibility = WGPUShaderStage_Compute;
  entries[1].buffer.type = WGPUBufferBindingType_Storage;
  entries[2].binding = 2;
  entries[2].visibility = WGPUShaderStage_Compute;
  entries[2].buffer.type = WGPUBufferBindingType_Uniform;
  entries[3].binding = 3;
  entries[3].visibility = WGPUShaderStage_Compute;
  entries[3].buffer.type = WGPUBufferBindingType_Uniform;
  entries[4].binding = 4;
  entries[4].visibility = WGPUShaderStage_Compute;
  entries[4].buffer.type = WGPUBufferBindingType_Uniform;

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 5;
  bgl_desc.entries = entries;
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);

  WGPUPipelineLayoutDescriptor pl_desc = {};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(device, &pl_desc);

  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.layout = pipeline_layout;
  pipeline_desc.compute.module = shader;
  pipeline_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  pipeline_desc.compute.constantCount = 1;
  pipeline_desc.compute.constants = &wg_size_constant;
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device, &pipeline_desc);

  WGPUBindGroupEntry bg_entries[5] = {};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer = in_tensor.buffer;
  bg_entries[0].size = in_tensor.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = out_tensor.buffer;
  bg_entries[1].size = out_tensor.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = out_meta_buf;
  bg_entries[2].size = sizeof(TensorMeta);
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = in_meta_buf;
  bg_entries[3].size = sizeof(TensorMeta);
  bg_entries[4].binding = 4;
  bg_entries[4].buffer = params_buf;
  bg_entries[4].size = sizeof(PermuteParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 5;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch({pipeline, bind_group, workgroup_count});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
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
