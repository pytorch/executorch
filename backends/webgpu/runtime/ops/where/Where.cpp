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
#include <executorch/backends/webgpu/runtime/ops/where/where_wgsl.h>

#include <webgpu/webgpu.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// where.self selects self/other by cond; broadcasts all three.
void where_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [condition, self, other, out]; out is the last value id.
  const int cond_id = args.at(0);
  const int a_id = args.at(1);
  const int b_id = args.at(2);
  const int out_id = args.at(args.size() - 1);

  WGPUDevice device = graph.device();

  const auto& cond_tensor = graph.get_tensor(cond_id);
  const auto& a_tensor = graph.get_tensor(a_id);
  const auto& b_tensor = graph.get_tensor(b_id);
  const auto& out_tensor = graph.get_tensor(out_id);

  if (cond_tensor.buffer == nullptr || a_tensor.buffer == nullptr ||
      b_tensor.buffer == nullptr || out_tensor.buffer == nullptr) {
    throw std::runtime_error("where: null buffer binding");
  }
  if (out_tensor.dims.size() > kTensorMetaMaxNdim ||
      cond_tensor.dims.size() > kTensorMetaMaxNdim ||
      a_tensor.dims.size() > kTensorMetaMaxNdim ||
      b_tensor.dims.size() > kTensorMetaMaxNdim) {
    throw std::runtime_error("where: tensor rank exceeds 4 (MAX_NDIM)");
  }

  const uint32_t out_ndim = static_cast<uint32_t>(out_tensor.dims.size());

  TensorMeta out_meta;
  TensorMeta cond_meta;
  TensorMeta a_meta;
  TensorMeta b_meta;
  fill_tensor_meta_broadcast(out_tensor, out_ndim, &out_meta);
  fill_tensor_meta_broadcast(cond_tensor, out_ndim, &cond_meta);
  fill_tensor_meta_broadcast(a_tensor, out_ndim, &a_meta);
  fill_tensor_meta_broadcast(b_tensor, out_ndim, &b_meta);

  // a/b/out are fp32; cond is 1-byte bool (read byte-packed as array<u32>).
  if (out_tensor.nbytes !=
          static_cast<size_t>(out_meta.numel) * sizeof(float) ||
      a_tensor.nbytes != static_cast<size_t>(a_meta.numel) * sizeof(float) ||
      b_tensor.nbytes != static_cast<size_t>(b_meta.numel) * sizeof(float)) {
    throw std::runtime_error(
        "where: non-fp32 self/other (nbytes != numel * 4)");
  }
  if (cond_tensor.nbytes != static_cast<size_t>(cond_meta.numel)) {
    throw std::runtime_error("where: condition is not a 1-byte (bool) tensor");
  }

  // Buffer is 4-byte-rounded at alloc; bind the padded span for the u32 read.
  const size_t cond_bind_size = (cond_tensor.nbytes + 3) & ~size_t(3);

  uint32_t wg_size = utils::clamp_workgroup_size(device, kWhereWorkgroupSizeX);
  uint32_t workgroup_count = utils::compute_1d_workgroup_count(
      device, out_meta.numel, wg_size, "where");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer out_meta_buf =
      utils::make_uniform(device, &out_meta, sizeof(TensorMeta));
  WGPUBuffer cond_meta_buf =
      utils::make_uniform(device, &cond_meta, sizeof(TensorMeta));
  WGPUBuffer a_meta_buf =
      utils::make_uniform(device, &a_meta, sizeof(TensorMeta));
  WGPUBuffer b_meta_buf =
      utils::make_uniform(device, &b_meta, sizeof(TensorMeta));
  graph.add_uniform_buffer_bytes(4 * sizeof(TensorMeta));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kWhereWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry entries[8] = {};
  entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[3].buffer.type = WGPUBufferBindingType_Storage;
  entries[4].buffer.type = WGPUBufferBindingType_Uniform;
  entries[5].buffer.type = WGPUBufferBindingType_Uniform;
  entries[6].buffer.type = WGPUBufferBindingType_Uniform;
  entries[7].buffer.type = WGPUBufferBindingType_Uniform;
  for (uint32_t i = 0; i < 8; i++) {
    entries[i].binding = i;
    entries[i].visibility = WGPUShaderStage_Compute;
  }

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 8;
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

  WGPUBindGroupEntry bg_entries[8] = {};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer = cond_tensor.buffer;
  bg_entries[0].size = cond_bind_size;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = a_tensor.buffer;
  bg_entries[1].size = a_tensor.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = b_tensor.buffer;
  bg_entries[2].size = b_tensor.nbytes;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = out_tensor.buffer;
  bg_entries[3].size = out_tensor.nbytes;
  bg_entries[4].binding = 4;
  bg_entries[4].buffer = out_meta_buf;
  bg_entries[4].size = sizeof(TensorMeta);
  bg_entries[5].binding = 5;
  bg_entries[5].buffer = cond_meta_buf;
  bg_entries[5].size = sizeof(TensorMeta);
  bg_entries[6].binding = 6;
  bg_entries[6].buffer = a_meta_buf;
  bg_entries[6].size = sizeof(TensorMeta);
  bg_entries[7].binding = 7;
  bg_entries[7].buffer = b_meta_buf;
  bg_entries[7].size = sizeof(TensorMeta);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 8;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  const size_t dispatch_idx =
      graph.add_dispatch({pipeline, bind_group, workgroup_count});

  // Dynamic shapes: rebuild the 4 broadcast TensorMeta UBOs + dispatch count.
  WGPUBuffer o_buf = out_meta_buf, c_buf = cond_meta_buf, a_buf = a_meta_buf,
             bb_buf = b_meta_buf;
  auto where_resize = [cond_id,
                       a_id,
                       b_id,
                       out_id,
                       wg_size,
                       dispatch_idx,
                       o_buf,
                       c_buf,
                       a_buf,
                       bb_buf](WebGPUGraph& g) {
    const auto& c = g.cur_dims(cond_id);
    const auto& a = g.cur_dims(a_id);
    const auto& b = g.cur_dims(b_id);
    const size_t r = std::max({c.size(), a.size(), b.size()});
    auto dim_at = [r](const std::vector<int64_t>& d, size_t i) -> int64_t {
      return (i + d.size() < r) ? 1 : d[i - (r - d.size())];
    };
    std::vector<int64_t> out_d(r, 1);
    for (size_t i = 0; i < r; i++) {
      const int64_t cv = dim_at(c, i), av = dim_at(a, i), bv = dim_at(b, i);
      int64_t m = std::max({cv, av, bv});
      if ((cv != m && cv != 1) || (av != m && av != 1) ||
          (bv != m && bv != 1)) {
        throw std::runtime_error(
            "where(resize): operands not broadcast-compatible");
      }
      out_d[i] = m;
    }
    g.set_cur_dims(out_id, out_d);
    const uint32_t out_ndim = static_cast<uint32_t>(r);
    WebGPUTensor tc, ta, tb, to;
    tc.dims = c;
    ta.dims = a;
    tb.dims = b;
    to.dims = out_d;
    TensorMeta om, cm, am, bm;
    fill_tensor_meta_broadcast(to, out_ndim, &om);
    fill_tensor_meta_broadcast(tc, out_ndim, &cm);
    fill_tensor_meta_broadcast(ta, out_ndim, &am);
    fill_tensor_meta_broadcast(tb, out_ndim, &bm);
    wgpuQueueWriteBuffer(g.queue(), o_buf, 0, &om, sizeof(om));
    wgpuQueueWriteBuffer(g.queue(), c_buf, 0, &cm, sizeof(cm));
    wgpuQueueWriteBuffer(g.queue(), a_buf, 0, &am, sizeof(am));
    wgpuQueueWriteBuffer(g.queue(), bb_buf, 0, &bm, sizeof(bm));
    g.dispatch_at(dispatch_idx).workgroup_count_x =
        utils::compute_1d_workgroup_count(
            g.device(), om.numel, wg_size, "where(resize)");
  };
  graph.add_tensor_resize_hook(cond_id, where_resize);
  graph.add_tensor_resize_hook(a_id, where_resize);
  graph.add_tensor_resize_hook(b_id, where_resize);

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  graph.own_uniform_buffer(out_meta_buf);
  graph.own_uniform_buffer(cond_meta_buf);
  graph.own_uniform_buffer(a_meta_buf);
  graph.own_uniform_buffer(b_meta_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.where.self, where_impl);
}

} // namespace executorch::backends::webgpu
