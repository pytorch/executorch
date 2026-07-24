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
#include <executorch/backends/webgpu/runtime/ops/logical_and/logical_and_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching the WGSL Params struct; 16-byte aligned.
struct LogicalAndParams {
  uint32_t num_words;
  uint32_t _pad[3];
};
static_assert(
    sizeof(LogicalAndParams) == 16,
    "LogicalAndParams must be 16 bytes");

// out = a & b (canonical bool); mirrors Vulkan bitwise_and (BinaryOp.cpp:161).
void logical_and_op(WebGPUGraph& graph, const std::vector<int>& args) {
  const int a_id = args.at(0);
  const int b_id = args.at(1);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(a_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(b_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("logical_and: a/b/out is not a tensor");
  }

  WGPUDevice device = graph.device();
  const auto& a_tensor = graph.get_tensor(a_id);
  const auto& b_tensor = graph.get_tensor(b_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (a_tensor.buffer == nullptr || b_tensor.buffer == nullptr ||
      out_tensor.buffer == nullptr) {
    throw std::runtime_error("logical_and: null buffer binding");
  }
  // a/b/out are all 1-byte bool tensors (int-typed, NOT int8-quantized).
  if (!a_tensor.is_int || !b_tensor.is_int || !out_tensor.is_int ||
      a_tensor.elem_size != 1 || b_tensor.elem_size != 1 ||
      out_tensor.elem_size != 1) {
    throw std::runtime_error(
        "logical_and: a/b/out must be 1-byte bool tensors");
  }
  const uint64_t numel = out_tensor.nbytes;
  // bool packed 4/word (array<u32>); numel%4==0 gates the u32 binding.
  if (numel == 0u || numel % 4u != 0u || numel > UINT32_MAX) {
    throw std::runtime_error("logical_and: numel must be a nonzero mult of 4");
  }
  if (a_tensor.nbytes != numel || b_tensor.nbytes != numel) {
    throw std::runtime_error(
        "logical_and: a/b/out numel mismatch (same-shape)");
  }

  LogicalAndParams params = {};
  params.num_words = static_cast<uint32_t>(numel / 4u);

  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kLogicalAndWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, params.num_words, wg_size, "logical_and");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(LogicalAndParams));
  graph.add_uniform_buffer_bytes(sizeof(LogicalAndParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kLogicalAndWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  // out (rw storage) + a/b (ro storage) + params (uniform).
  WGPUBindGroupLayoutEntry entries[4] = {};
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_Storage;
  entries[1].binding = 1;
  entries[1].visibility = WGPUShaderStage_Compute;
  entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[2].binding = 2;
  entries[2].visibility = WGPUShaderStage_Compute;
  entries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[3].binding = 3;
  entries[3].visibility = WGPUShaderStage_Compute;
  entries[3].buffer.type = WGPUBufferBindingType_Uniform;

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 4;
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

  WGPUBindGroupEntry bg[4] = {};
  bg[0].binding = 0;
  bg[0].buffer = out_tensor.buffer;
  bg[0].size = out_tensor.nbytes;
  bg[1].binding = 1;
  bg[1].buffer = a_tensor.buffer;
  bg[1].size = a_tensor.nbytes;
  bg[2].binding = 2;
  bg[2].buffer = b_tensor.buffer;
  bg[2].size = b_tensor.nbytes;
  bg[3].binding = 3;
  bg[3].buffer = uniform_buffer;
  bg[3].size = sizeof(LogicalAndParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 4;
  bg_desc.entries = bg;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  const size_t dispatch_idx = graph.add_dispatch(
      {pipeline,
       bind_group,
       workgroup_count.x,
       "logical_and",
       workgroup_count.y});

  // Dynamic shapes: recompute num_words/dispatch; out follows a (same-shape).
  WGPUBuffer params_buf = uniform_buffer;
  auto resize =
      [a_id, b_id, out_id, wg_size, dispatch_idx, params_buf](WebGPUGraph& g) {
        const auto& d = g.cur_dims(a_id);
        const uint64_t n = utils::numel_of(d);
        if (n == 0u || n % 4u != 0u || n > UINT32_MAX ||
            utils::numel_of(g.cur_dims(b_id)) != n) {
          throw std::runtime_error(
              "logical_and(resize): numel must be a mult of 4");
        }
        g.set_cur_dims(out_id, d);
        LogicalAndParams p = {};
        p.num_words = static_cast<uint32_t>(n / 4u);
        wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
        const utils::WgCount wgc = utils::compute_2d_workgroup_count(
            g.device(), p.num_words, wg_size, "logical_and");
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
      };
  graph.add_tensor_resize_hook(a_id, resize);
  graph.add_tensor_resize_hook(b_id, resize);

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  graph.own_uniform_buffer(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.logical_and.default, logical_and_op);
}

} // namespace executorch::backends::webgpu
