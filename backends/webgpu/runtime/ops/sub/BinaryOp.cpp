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
#include <executorch/backends/webgpu/runtime/ops/sub/binary_sub_wgsl.h>

#include <webgpu/webgpu.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

void sub_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int in1_id = args.at(0);
  const int in2_id = args.at(1);
  const int alpha_id = args.at(2);
  const int out_id = args.at(3);

  WGPUDevice device = graph.device();

  float alpha = 1.0f;
  if (graph.get_value_type(alpha_id) == WebGPUGraph::ValueType::Int) {
    alpha = static_cast<float>(graph.get_int(alpha_id));
  } else if (graph.get_value_type(alpha_id) == WebGPUGraph::ValueType::Double) {
    alpha = static_cast<float>(graph.get_double(alpha_id));
  }

  const auto& in1_tensor = graph.get_tensor(in1_id);
  const auto& in2_tensor = graph.get_tensor(in2_id);
  const auto& out_tensor = graph.get_tensor(out_id);

  // Rank guard (NCHW backend is <= 4 dims; 1D dispatch only).
  if (out_tensor.dims.size() > kTensorMetaMaxNdim ||
      in1_tensor.dims.size() > kTensorMetaMaxNdim ||
      in2_tensor.dims.size() > kTensorMetaMaxNdim) {
    throw std::runtime_error("sub: tensor rank exceeds 4 (MAX_NDIM)");
  }

  const uint32_t out_ndim = static_cast<uint32_t>(out_tensor.dims.size());

  // 3 per-tensor meta uniforms (mirror Vulkan); inputs broadcast-aligned.
  TensorMeta out_meta;
  TensorMeta in1_meta;
  TensorMeta in2_meta;
  fill_tensor_meta_broadcast(out_tensor, out_ndim, &out_meta);
  fill_tensor_meta_broadcast(in1_tensor, out_ndim, &in1_meta);
  fill_tensor_meta_broadcast(in2_tensor, out_ndim, &in2_meta);

  // fp32-only: nbytes must equal numel * 4 for every operand.
  if (out_tensor.nbytes !=
          static_cast<size_t>(out_meta.numel) * sizeof(float) ||
      in1_tensor.nbytes !=
          static_cast<size_t>(in1_meta.numel) * sizeof(float) ||
      in2_tensor.nbytes !=
          static_cast<size_t>(in2_meta.numel) * sizeof(float)) {
    throw std::runtime_error("sub: non-fp32 operand (nbytes != numel * 4)");
  }

  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kBinarySubWorkgroupSizeX);
  utils::WgCount workgroup_count =
      utils::compute_2d_workgroup_count(device, out_meta.numel, wg_size, "sub");

  WGPUConstantEntry constants[2] = {};
  constants[0].key = {"wg_size", WGPU_STRLEN};
  constants[0].value = static_cast<double>(wg_size);
  constants[1].key = {"alpha", WGPU_STRLEN};
  constants[1].value = static_cast<double>(alpha);

  WGPUBuffer out_meta_buf =
      utils::make_uniform(device, &out_meta, sizeof(TensorMeta));
  WGPUBuffer in1_meta_buf =
      utils::make_uniform(device, &in1_meta, sizeof(TensorMeta));
  WGPUBuffer in2_meta_buf =
      utils::make_uniform(device, &in2_meta, sizeof(TensorMeta));
  graph.add_uniform_buffer_bytes(3 * sizeof(TensorMeta));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kBinarySubWGSL, WGPU_STRLEN};

  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry entries[6] = {};

  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

  entries[1].binding = 1;
  entries[1].visibility = WGPUShaderStage_Compute;
  entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

  entries[2].binding = 2;
  entries[2].visibility = WGPUShaderStage_Compute;
  entries[2].buffer.type = WGPUBufferBindingType_Storage;

  entries[3].binding = 3;
  entries[3].visibility = WGPUShaderStage_Compute;
  entries[3].buffer.type = WGPUBufferBindingType_Uniform;

  entries[4].binding = 4;
  entries[4].visibility = WGPUShaderStage_Compute;
  entries[4].buffer.type = WGPUBufferBindingType_Uniform;

  entries[5].binding = 5;
  entries[5].visibility = WGPUShaderStage_Compute;
  entries[5].buffer.type = WGPUBufferBindingType_Uniform;

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 6;
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
  pipeline_desc.compute.constantCount = 2;
  pipeline_desc.compute.constants = constants;
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device, &pipeline_desc);

  WGPUBindGroupEntry bg_entries[6] = {};

  bg_entries[0].binding = 0;
  bg_entries[0].buffer = in1_tensor.buffer;
  bg_entries[0].size = in1_tensor.nbytes;

  bg_entries[1].binding = 1;
  bg_entries[1].buffer = in2_tensor.buffer;
  bg_entries[1].size = in2_tensor.nbytes;

  bg_entries[2].binding = 2;
  bg_entries[2].buffer = out_tensor.buffer;
  bg_entries[2].size = out_tensor.nbytes;

  bg_entries[3].binding = 3;
  bg_entries[3].buffer = out_meta_buf;
  bg_entries[3].size = sizeof(TensorMeta);

  bg_entries[4].binding = 4;
  bg_entries[4].buffer = in1_meta_buf;
  bg_entries[4].size = sizeof(TensorMeta);

  bg_entries[5].binding = 5;
  bg_entries[5].buffer = in2_meta_buf;
  bg_entries[5].size = sizeof(TensorMeta);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 6;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  const size_t dispatch_idx = graph.add_dispatch(
      {pipeline, bind_group, workgroup_count.x, "sub", workgroup_count.y});

  // Dynamic shapes: rebuild all 3 broadcast TensorMeta UBOs + dispatch.
  WGPUBuffer o_buf = out_meta_buf, a_buf = in1_meta_buf, b_buf = in2_meta_buf;
  auto sub_resize =
      [in1_id, in2_id, out_id, wg_size, dispatch_idx, o_buf, a_buf, b_buf](
          WebGPUGraph& g) {
        const auto& a = g.cur_dims(in1_id);
        const auto& b = g.cur_dims(in2_id);
        const size_t r = std::max(a.size(), b.size());
        std::vector<int64_t> out_d(r, 1);
        for (size_t i = 0; i < r; i++) {
          const int64_t av = (i + a.size() < r) ? 1 : a[i - (r - a.size())];
          const int64_t bv = (i + b.size() < r) ? 1 : b[i - (r - b.size())];
          if (av != bv && av != 1 && bv != 1) {
            throw std::runtime_error(
                "sub(resize): operands are not broadcast-compatible");
          }
          out_d[i] = av > bv ? av : bv;
        }
        g.set_cur_dims(out_id, out_d);
        const uint32_t out_ndim = static_cast<uint32_t>(r);
        WebGPUTensor ta, tb, to;
        ta.dims = a;
        tb.dims = b;
        to.dims = out_d;
        TensorMeta om, am, bm;
        fill_tensor_meta_broadcast(to, out_ndim, &om);
        fill_tensor_meta_broadcast(ta, out_ndim, &am);
        fill_tensor_meta_broadcast(tb, out_ndim, &bm);
        wgpuQueueWriteBuffer(g.queue(), o_buf, 0, &om, sizeof(om));
        wgpuQueueWriteBuffer(g.queue(), a_buf, 0, &am, sizeof(am));
        wgpuQueueWriteBuffer(g.queue(), b_buf, 0, &bm, sizeof(bm));
        const utils::WgCount wgc = utils::compute_2d_workgroup_count(
            g.device(), om.numel, wg_size, "sub(resize)");
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
      };
  graph.add_tensor_resize_hook(in1_id, sub_resize);
  graph.add_tensor_resize_hook(in2_id, sub_resize);

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // Graph owns them so a resize hook can rewrite them; freed in the dtor.
  graph.own_uniform_buffer(out_meta_buf);
  graph.own_uniform_buffer(in1_meta_buf);
  graph.own_uniform_buffer(in2_meta_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.sub.Tensor, sub_impl);
}

} // namespace executorch::backends::webgpu
