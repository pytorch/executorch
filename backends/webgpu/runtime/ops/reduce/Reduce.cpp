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
#include <executorch/backends/webgpu/runtime/ops/reduce/reduce_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching the WGSL Params struct (16-byte aligned).
struct ReduceParams {
  uint32_t outer;
  uint32_t r;
  uint32_t inner;
  uint32_t is_mean;
};
static_assert(sizeof(ReduceParams) == 16, "ReduceParams must be 16 bytes");

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
    throw std::runtime_error("WebGPU reduce: dim out of range");
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

void reduce_impl(
    WebGPUGraph& graph,
    const std::vector<int>& args,
    bool is_mean,
    const char* op_name) {
  const int in_id = args.at(0);
  const int dim_id = args.at(1);
  const int keepdim_id = args.at(2);
  const int out_id = args.at(args.size() - 1);

  WGPUDevice device = graph.device();
  const auto& in = graph.get_tensor(in_id);
  const auto& out = graph.get_tensor(out_id);

  bool keepdim = false;
  if (graph.get_value_type(keepdim_id) == WebGPUGraph::ValueType::Int) {
    keepdim = graph.get_int(keepdim_id) != 0;
  }

  if (in.dims.empty()) {
    throw std::runtime_error("WebGPU reduce: scalar input unsupported");
  }
  if (graph.get_value_type(dim_id) != WebGPUGraph::ValueType::IntList) {
    throw std::runtime_error("WebGPU reduce: dim arg is not an IntList");
  }
  const std::vector<int64_t>& reduce_dims = graph.get_int_list(dim_id);
  // Single-dim reduction only for now; multi-dim is a tracked extension.
  if (reduce_dims.size() != 1) {
    throw std::runtime_error(
        "WebGPU reduce: only single-dim reduction is supported");
  }
  const int64_t dim = reduce_dims[0];

  uint32_t outer = 0, r = 0, inner = 0;
  decompose(in.dims, dim, outer, r, inner);
  if (outer == 0 || r == 0 || inner == 0) {
    throw std::runtime_error("WebGPU reduce: zero-sized reduction");
  }

  uint64_t in_numel = 1;
  for (int64_t d : in.dims) {
    in_numel *= static_cast<uint64_t>(d);
  }
  const uint64_t outputs = static_cast<uint64_t>(outer) * inner;
  if (in.nbytes != in_numel * sizeof(float) ||
      out.nbytes != outputs * sizeof(float)) {
    throw std::runtime_error("WebGPU reduce: fp32-only (byte-size mismatch)");
  }
  if (outputs > UINT32_MAX) {
    throw std::runtime_error(
        "WebGPU reduce: output count exceeds dispatch limit");
  }

  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kReduceWorkgroupSizeX);
  const uint32_t workgroup_count = utils::compute_1d_workgroup_count(
      device, static_cast<uint32_t>(outputs), wg_size, op_name);

  ReduceParams params = {};
  params.outer = outer;
  params.r = r;
  params.inner = inner;
  params.is_mean = is_mean ? 1u : 0u;

  WGPUBufferDescriptor uniform_desc = {};
  uniform_desc.size = sizeof(ReduceParams);
  uniform_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  uniform_desc.mappedAtCreation = true;
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(device, &uniform_desc);
  void* mapped =
      wgpuBufferGetMappedRange(uniform_buffer, 0, sizeof(ReduceParams));
  std::memcpy(mapped, &params, sizeof(ReduceParams));
  wgpuBufferUnmap(uniform_buffer);
  graph.add_uniform_buffer_bytes(sizeof(ReduceParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kReduceWGSL, WGPU_STRLEN};
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
  bg_entries[2].size = sizeof(ReduceParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 3;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  const size_t dispatch_idx =
      graph.add_dispatch({pipeline, bind_group, workgroup_count, op_name});

  // Dynamic shapes: recompute the decomposition for the reduced dim + dispatch.
  WGPUBuffer params_buf = uniform_buffer;
  const uint32_t is_mean_u = is_mean ? 1u : 0u;
  const uint64_t build_outputs = outputs;
  graph.add_tensor_resize_hook(
      in_id,
      [in_id,
       out_id,
       dim,
       keepdim,
       is_mean_u,
       build_outputs,
       wg_size,
       dispatch_idx,
       params_buf](WebGPUGraph& g) {
        const auto& d = g.cur_dims(in_id);
        uint32_t o = 0, rr = 0, n = 0;
        decompose(std::vector<int64_t>(d.begin(), d.end()), dim, o, rr, n);
        if (o == 0u || rr == 0u || n == 0u) {
          throw std::runtime_error("WebGPU reduce: live zero-sized reduction");
        }
        const uint64_t live_outputs = static_cast<uint64_t>(o) * n;
        if (live_outputs > build_outputs) {
          throw std::runtime_error(
              "WebGPU reduce: live output count exceeds build max");
        }
        ReduceParams p = {};
        p.outer = o;
        p.r = rr;
        p.inner = n;
        p.is_mean = is_mean_u;
        wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
        g.dispatch_at(dispatch_idx).workgroup_count_x =
            utils::compute_1d_workgroup_count(
                g.device(), static_cast<uint32_t>(live_outputs), wg_size,
                "reduce(resize)");
        // Propagate reduced output dims for downstream resize hooks.
        int64_t nd = static_cast<int64_t>(d.size());
        int64_t rd = dim < 0 ? dim + nd : dim;
        std::vector<int64_t> od;
        for (int64_t i = 0; i < nd; ++i) {
          if (i == rd) {
            if (keepdim) {
              od.push_back(1);
            }
          } else {
            od.push_back(d[i]);
          }
        }
        g.set_cur_dims(out_id, od);
      });

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // Graph owns it so the resize hook can rewrite it; freed in the dtor.
  graph.own_uniform_buffer(uniform_buffer);
}

void sum_dim_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  reduce_impl(graph, args, /*is_mean=*/false, "sum.dim_IntList");
}

void mean_dim_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  reduce_impl(graph, args, /*is_mean=*/true, "mean.dim");
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.sum.dim_IntList, sum_dim_impl);
  WEBGPU_REGISTER_OP(aten.mean.dim, mean_dim_impl);
}

} // namespace executorch::backends::webgpu
