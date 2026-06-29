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
#include <executorch/backends/webgpu/runtime/ops/select/select_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct SelectParams {
  uint32_t dim;
  uint32_t index;
  uint32_t _pad[2];
};

// dim/index are required Ints (SymInt throws); no Null default unlike slice.
int64_t read_scalar(WebGPUGraph& graph, int id, const char* what) {
  if (graph.get_value_type(id) == WebGPUGraph::ValueType::Int) {
    return graph.get_int(id);
  }
  throw std::runtime_error(std::string("select: dynamic/unsupported ") + what);
}

// Build a TensorMeta from live dims, write it to buf, return numel.
uint32_t write_meta_from_dims(
    WebGPUGraph& g,
    WGPUBuffer buf,
    const std::vector<int64_t>& dims) {
  WebGPUTensor t;
  t.dims = dims;
  TensorMeta m;
  fill_tensor_meta(t, &m);
  wgpuQueueWriteBuffer(g.queue(), buf, 0, &m, sizeof(m));
  return m.numel;
}

void select_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [self, dim, index, out]; output rank = in rank - 1.
  const int in_id = args.at(0);
  const int out_id = args.at(3);

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (in_tensor.buffer == nullptr || out_tensor.buffer == nullptr) {
    throw std::runtime_error("select: null buffer binding");
  }

  const int in_ndim = static_cast<int>(in_tensor.dims.size());
  int64_t dim = read_scalar(graph, args.at(1), "dim");
  if (dim < 0) {
    dim += in_ndim;
  }
  if (dim < 0 || dim >= in_ndim) {
    throw std::runtime_error("select: dim out of range");
  }
  const int64_t in_size = in_tensor.dims[dim];
  // Keep the RAW index: -1 normalizes against the LIVE dim (the resize hook).
  const int64_t raw_index = read_scalar(graph, args.at(2), "index");
  int64_t index = raw_index < 0 ? raw_index + in_size : raw_index;
  if (index < 0 || index >= in_size) {
    throw std::runtime_error("select: index out of range");
  }

  TensorMeta out_meta;
  TensorMeta in_meta;
  fill_tensor_meta(out_tensor, &out_meta);
  fill_tensor_meta(in_tensor, &in_meta);
  if (out_tensor.nbytes !=
          static_cast<size_t>(out_meta.numel) * sizeof(float) ||
      in_tensor.nbytes != static_cast<size_t>(in_meta.numel) * sizeof(float)) {
    throw std::runtime_error("select: non-fp32 operand (nbytes != numel * 4)");
  }

  SelectParams params = {};
  params.dim = static_cast<uint32_t>(dim);
  params.index = static_cast<uint32_t>(index);

  uint32_t wg_size = utils::clamp_workgroup_size(device, kSelectWorkgroupSizeX);
  uint32_t workgroup_count = utils::compute_1d_workgroup_count(
      device, out_meta.numel, wg_size, "select");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer out_meta_buf =
      utils::make_uniform(device, &out_meta, sizeof(TensorMeta));
  WGPUBuffer in_meta_buf =
      utils::make_uniform(device, &in_meta, sizeof(TensorMeta));
  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(SelectParams));
  graph.add_uniform_buffer_bytes(2 * sizeof(TensorMeta) + sizeof(SelectParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kSelectWGSL, WGPU_STRLEN};
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
  bg_entries[4].size = sizeof(SelectParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 5;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  const size_t dispatch_idx =
      graph.add_dispatch({pipeline, bind_group, workgroup_count});

  // Dynamic shapes: out = in minus `dim`; re-resolve index, meta, dispatch.
  graph.add_tensor_resize_hook(
      in_id,
      [in_id,
       out_id,
       dim,
       raw_index,
       out_meta_buf,
       in_meta_buf,
       params_buf,
       wg_size,
       dispatch_idx](WebGPUGraph& g) {
        const auto& ind = g.cur_dims(in_id);
        if (dim < 0 || dim >= static_cast<int>(ind.size())) {
          throw std::runtime_error("select(resize): dim out of range");
        }
        const int64_t live_in_size = ind[dim];
        int64_t idx = raw_index < 0 ? raw_index + live_in_size : raw_index;
        if (idx < 0 || idx >= live_in_size) {
          throw std::runtime_error("select(resize): index out of range");
        }
        std::vector<int64_t> od;
        for (size_t k = 0; k < ind.size(); k++) {
          if (static_cast<int>(k) != dim) {
            od.push_back(ind[k]);
          }
        }
        g.set_cur_dims(out_id, od);
        const uint32_t out_numel = write_meta_from_dims(g, out_meta_buf, od);
        write_meta_from_dims(g, in_meta_buf, ind);
        SelectParams p = {};
        p.dim = static_cast<uint32_t>(dim);
        p.index = static_cast<uint32_t>(idx);
        wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
        g.dispatch_at(dispatch_idx).workgroup_count_x =
            utils::compute_1d_workgroup_count(
                g.device(), out_numel, wg_size, "select(resize)");
      });

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // Graph owns them so the resize hook can rewrite them; freed in the dtor.
  graph.own_uniform_buffer(out_meta_buf);
  graph.own_uniform_buffer(in_meta_buf);
  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.select_copy.int, select_impl);
}

} // namespace executorch::backends::webgpu
