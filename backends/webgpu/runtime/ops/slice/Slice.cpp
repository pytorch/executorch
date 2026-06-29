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
#include <executorch/backends/webgpu/runtime/ops/slice/slice_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct SliceParams {
  uint32_t dim;
  uint32_t start;
  uint32_t step;
  uint32_t _pad;
};

// Read scalar arg: Int->value (INT64_MAX->default), Null->default, else throw.
int64_t
read_scalar(WebGPUGraph& graph, int id, int64_t dflt, const char* what) {
  switch (graph.get_value_type(id)) {
    case WebGPUGraph::ValueType::Int: {
      const int64_t v = graph.get_int(id);
      return v == INT64_MAX ? dflt : v;
    }
    case WebGPUGraph::ValueType::Null:
      return dflt;
    default:
      throw std::runtime_error(
          std::string("slice: dynamic/unsupported ") + what);
  }
}

// Read a slice index (start/end) that MAY be a dynamic SymInt; else Int/Null.
int64_t read_index(WebGPUGraph& graph, int id, int64_t dflt) {
  switch (graph.get_value_type(id)) {
    case WebGPUGraph::ValueType::SymInt:
      return graph.read_symint(id);
    case WebGPUGraph::ValueType::Int: {
      const int64_t v = graph.get_int(id);
      return v == INT64_MAX ? dflt : v;
    }
    default:
      return dflt;
  }
}

bool is_symint(WebGPUGraph& graph, int id) {
  return graph.get_value_type(id) == WebGPUGraph::ValueType::SymInt;
}

// Clamp + normalize a (possibly negative) index into [0, size].
int64_t norm_clamp(int64_t idx, int64_t size) {
  if (idx < 0) {
    idx += size;
  }
  return idx < 0 ? 0 : (idx > size ? size : idx);
}

void slice_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [self, dim, start, end, step, out]. start/end may be dynamic SymInts;
  // a resize hook recomputes the live extent on `dim` (out[dim] / cur_dims).
  const int in_id = args.at(0);
  const int start_id = args.at(2);
  const int end_id = args.at(3);
  const int out_id = args.at(5);

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);

  const int in_ndim = static_cast<int>(in_tensor.dims.size());
  int64_t dim = read_scalar(graph, args.at(1), 0, "dim");
  if (dim < 0) {
    dim += in_ndim;
  }
  if (dim < 0 || dim >= in_ndim) {
    throw std::runtime_error("slice: dim out of range");
  }
  const int64_t step = read_scalar(graph, args.at(4), 1, "step");
  if (step < 1) {
    throw std::runtime_error("slice: step must be >= 1");
  }
  // start/end may be dynamic SymInts; seed from current (max) dims, the resize
  // hook recomputes live. Clamp guards the gather offset.
  const int64_t in_size = in_tensor.dims[dim];
  const int64_t start = norm_clamp(read_index(graph, start_id, 0), in_size);

  TensorMeta out_meta;
  TensorMeta in_meta;
  fill_tensor_meta(out_tensor, &out_meta);
  fill_tensor_meta(in_tensor, &in_meta);
  if (out_tensor.nbytes !=
          static_cast<size_t>(out_meta.numel) * sizeof(float) ||
      in_tensor.nbytes != static_cast<size_t>(in_meta.numel) * sizeof(float)) {
    throw std::runtime_error("slice: non-fp32 operand (nbytes != numel * 4)");
  }

  SliceParams params = {};
  params.dim = static_cast<uint32_t>(dim);
  params.start = static_cast<uint32_t>(start);
  params.step = static_cast<uint32_t>(step);

  uint32_t wg_size = utils::clamp_workgroup_size(device, kSliceWorkgroupSizeX);
  uint32_t workgroup_count = utils::compute_1d_workgroup_count(
      device, out_meta.numel, wg_size, "slice");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer out_meta_buf =
      utils::make_uniform(device, &out_meta, sizeof(TensorMeta));
  WGPUBuffer in_meta_buf =
      utils::make_uniform(device, &in_meta, sizeof(TensorMeta));
  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(SliceParams));
  graph.add_uniform_buffer_bytes(2 * sizeof(TensorMeta) + sizeof(SliceParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kSliceWGSL, WGPU_STRLEN};
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
  bg_entries[4].size = sizeof(SliceParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 5;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch({pipeline, bind_group, workgroup_count});
  const size_t dispatch_idx = graph.num_dispatches() - 1;

  // Dynamic shapes: live start/end -> out[dim] len + meta/params/dispatch.
  auto recompute = [in_id,
                    out_id,
                    start_id,
                    end_id,
                    dim,
                    step,
                    wg_size,
                    out_meta_buf,
                    in_meta_buf,
                    params_buf,
                    dispatch_idx](WebGPUGraph& g) {
    const auto& in_dims = g.cur_dims(in_id);
    const int64_t live_in_size = in_dims[dim];
    const int64_t start = norm_clamp(read_index(g, start_id, 0), live_in_size);
    const int64_t end =
        norm_clamp(read_index(g, end_id, live_in_size), live_in_size);
    const int64_t len = end > start ? (end - start + step - 1) / step : 0;

    // Out dims = live input dims (mirror Vulkan resize_slice_copy_node).
    std::vector<int64_t> od = in_dims;
    od[dim] = len;
    g.set_cur_dims(out_id, od);

    WebGPUTensor t_out;
    t_out.dims = od;
    WebGPUTensor t_in;
    t_in.dims = in_dims;
    TensorMeta om;
    TensorMeta im;
    fill_tensor_meta(t_out, &om);
    fill_tensor_meta(t_in, &im);
    wgpuQueueWriteBuffer(g.queue(), out_meta_buf, 0, &om, sizeof(om));
    wgpuQueueWriteBuffer(g.queue(), in_meta_buf, 0, &im, sizeof(im));
    SliceParams p = {};
    p.dim = static_cast<uint32_t>(dim);
    p.start = static_cast<uint32_t>(start);
    p.step = static_cast<uint32_t>(step);
    wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
    g.dispatch_at(dispatch_idx).workgroup_count_x =
        utils::compute_1d_workgroup_count(
            g.device(), om.numel, wg_size, "slice(resize)");
  };
  if (is_symint(graph, start_id)) {
    graph.add_resize_hook(start_id, recompute);
  }
  if (is_symint(graph, end_id)) {
    graph.add_resize_hook(end_id, recompute);
  }
  graph.add_tensor_resize_hook(in_id, recompute);

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // Graph owns the uniforms so the resize hook can rewrite them; freed in dtor.
  graph.own_uniform_buffer(out_meta_buf);
  graph.own_uniform_buffer(in_meta_buf);
  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.slice_copy.Tensor, slice_impl);
}

} // namespace executorch::backends::webgpu
