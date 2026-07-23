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
#include <executorch/backends/webgpu/runtime/ops/softmax/log_softmax_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching the WGSL Params struct (16-byte aligned).
struct LogSoftmaxParams {
  uint32_t outer;
  uint32_t r;
  uint32_t inner;
  uint32_t pad_;
};
static_assert(
    sizeof(LogSoftmaxParams) == 16,
    "LogSoftmaxParams must be 16 bytes");

// Decompose dims into [outer, R, inner] for a reduction along `dim`.
void decompose(
    const std::vector<int64_t>& dims,
    int64_t dim,
    uint32_t& outer,
    uint32_t& r,
    uint32_t& inner) {
  const int64_t ndim = static_cast<int64_t>(dims.size());
  if (dim < 0) {
    dim += ndim;
  }
  if (ndim == 0 || dim < 0 || dim >= ndim) {
    throw std::runtime_error("WebGPU log_softmax: dim out of range");
  }
  uint64_t o = 1, in = 1;
  for (int64_t d = 0; d < dim; ++d) {
    o *= static_cast<uint64_t>(dims[d]);
  }
  for (int64_t d = dim + 1; d < ndim; ++d) {
    in *= static_cast<uint64_t>(dims[d]);
  }
  outer = static_cast<uint32_t>(o);
  r = static_cast<uint32_t>(dims[dim]);
  inner = static_cast<uint32_t>(in);
}

void log_softmax_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int in_id = args.at(0);
  const int dim_id = args.at(1);
  const int out_id = args.at(3);

  WGPUDevice device = graph.device();

  const auto& in = graph.get_tensor(in_id);
  const auto& out = graph.get_tensor(out_id);

  if (in.dims.empty()) {
    throw std::runtime_error("WebGPU log_softmax: scalar input unsupported");
  }
  if (graph.get_value_type(dim_id) != WebGPUGraph::ValueType::Int) {
    throw std::runtime_error("WebGPU log_softmax: dim must be an int");
  }
  const int64_t dim = graph.get_int(dim_id);

  uint32_t outer = 0, r = 0, inner = 0;
  decompose(in.dims, dim, outer, r, inner);
  if (outer == 0 || r == 0 || inner == 0) {
    throw std::runtime_error("WebGPU log_softmax: zero-sized reduction");
  }

  uint64_t numel = 1;
  for (int64_t d : in.dims) {
    numel *= static_cast<uint64_t>(d);
  }
  if (in.nbytes != numel * sizeof(float) ||
      out.nbytes != numel * sizeof(float)) {
    throw std::runtime_error(
        "WebGPU log_softmax: fp32-only (byte-size mismatch)");
  }

  const uint64_t lines = static_cast<uint64_t>(outer) * inner;
  if (lines > UINT32_MAX) {
    throw std::runtime_error(
        "WebGPU log_softmax: line count exceeds dispatch limit");
  }

  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kLogSoftmaxWorkgroupSizeX);
  const uint32_t workgroup_count = utils::compute_1d_workgroup_count(
      device, static_cast<uint32_t>(lines), wg_size, "log_softmax");

  LogSoftmaxParams params = {};
  params.outer = outer;
  params.r = r;
  params.inner = inner;

  WGPUBufferDescriptor uniform_desc = {};
  uniform_desc.size = sizeof(LogSoftmaxParams);
  uniform_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  uniform_desc.mappedAtCreation = true;
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(device, &uniform_desc);
  void* mapped =
      wgpuBufferGetMappedRange(uniform_buffer, 0, sizeof(LogSoftmaxParams));
  std::memcpy(mapped, &params, sizeof(LogSoftmaxParams));
  wgpuBufferUnmap(uniform_buffer);
  graph.add_uniform_buffer_bytes(sizeof(LogSoftmaxParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kLogSoftmaxWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry entries[3] = {};
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[1].binding = 1;
  entries[1].visibility = WGPUShaderStage_Compute;
  entries[1].buffer.type = WGPUBufferBindingType_Storage;
  entries[2].binding = 2;
  entries[2].visibility = WGPUShaderStage_Compute;
  entries[2].buffer.type = WGPUBufferBindingType_Uniform;

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 3;
  bgl_desc.entries = entries;
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);

  WGPUPipelineLayoutDescriptor pl_desc = {};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(device, &pl_desc);

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.layout = pipeline_layout;
  pipeline_desc.compute.module = shader;
  pipeline_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  pipeline_desc.compute.constantCount = 1;
  pipeline_desc.compute.constants = &wg_size_constant;
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device, &pipeline_desc);

  WGPUBindGroupEntry bg_entries[3] = {};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer = in.buffer;
  bg_entries[0].size = in.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = out.buffer;
  bg_entries[1].size = out.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = uniform_buffer;
  bg_entries[2].size = sizeof(LogSoftmaxParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 3;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  const size_t dispatch_idx = graph.add_dispatch(
      {pipeline, bind_group, workgroup_count, "log_softmax"});

  // Dynamic shapes: recompute the decomposition for the reduced dim + dispatch.
  WGPUBuffer params_buf = uniform_buffer;
  const uint64_t build_numel = static_cast<uint64_t>(outer) * r * inner;
  graph.add_tensor_resize_hook(
      in_id,
      [in_id, out_id, dim, build_numel, wg_size, dispatch_idx, params_buf](
          WebGPUGraph& g) {
        const auto& d = g.cur_dims(in_id);
        uint32_t o = 0, rr = 0, n = 0;
        decompose(std::vector<int64_t>(d.begin(), d.end()), dim, o, rr, n);
        if (o == 0u || rr == 0u || n == 0u) {
          throw std::runtime_error(
              "WebGPU log_softmax: live zero-sized reduction");
        }
        if (static_cast<uint64_t>(o) * rr * n > build_numel) {
          throw std::runtime_error(
              "WebGPU log_softmax: live numel exceeds build max");
        }
        const uint64_t live_lines = static_cast<uint64_t>(o) * n;
        LogSoftmaxParams p = {};
        p.outer = o;
        p.r = rr;
        p.inner = n;
        wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
        g.dispatch_at(dispatch_idx).workgroup_count_x =
            utils::compute_1d_workgroup_count(
                g.device(),
                static_cast<uint32_t>(live_lines),
                wg_size,
                "log_softmax(resize)");
        g.set_cur_dims(out_id, std::vector<int64_t>(d.begin(), d.end()));
      });

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // Graph owns it so the resize hook can rewrite it; freed in the dtor.
  graph.own_uniform_buffer(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten._log_softmax.default, log_softmax_impl);
}

} // namespace executorch::backends::webgpu
