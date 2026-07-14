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
#include <executorch/backends/webgpu/runtime/ops/adamw/adamw_step_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

struct AdamwStepParams {
  uint32_t numel;
  uint32_t _pad0;
  uint32_t _pad1;
  uint32_t _pad2;
  float lr;
  float beta1;
  float beta2;
  float eps;
  float weight_decay;
  float bias_correction1;
  float bias_correction2;
  float _pad3;
};
static_assert(sizeof(AdamwStepParams) == 48, "params must be 48 bytes");

// AdamW step over an fp32 latent (elementwise, in place); mirrors torch AdamW.
void adamw_step_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int param_id = args.at(0);
  const int m_id = args.at(1);
  const int v_id = args.at(2);
  const int grad_id = args.at(3);

  WGPUDevice device = graph.device();
  const auto& param = graph.get_tensor(param_id);
  const auto& m = graph.get_tensor(m_id);
  const auto& v = graph.get_tensor(v_id);
  const auto& grad = graph.get_tensor(grad_id);

  uint64_t numel = 1;
  for (int64_t d : param.dims) {
    numel *= static_cast<uint64_t>(d);
  }
  if (param.dims.empty() || numel == 0) {
    throw std::runtime_error("adamw_step: empty param");
  }
  const uint64_t bytes = numel * sizeof(float);
  if (param.nbytes != bytes || m.nbytes != bytes || v.nbytes != bytes ||
      grad.nbytes != bytes) {
    throw std::runtime_error(
        "adamw_step: param/m/v/grad must be fp32 and same numel");
  }
  if (numel > UINT32_MAX) {
    throw std::runtime_error("adamw_step: numel exceeds u32");
  }

  auto scalar = [&](int id, const char* name) -> float {
    if (graph.get_value_type(id) != WebGPUGraph::ValueType::Double) {
      throw std::runtime_error(
          std::string("adamw_step: ") + name + " must be a float scalar");
    }
    return static_cast<float>(graph.get_double(id));
  };

  AdamwStepParams params = {};
  params.numel = static_cast<uint32_t>(numel);
  params.lr = scalar(args.at(4), "lr");
  params.beta1 = scalar(args.at(5), "beta1");
  params.beta2 = scalar(args.at(6), "beta2");
  params.eps = scalar(args.at(7), "eps");
  params.weight_decay = scalar(args.at(8), "weight_decay");
  params.bias_correction1 = scalar(args.at(9), "bias_correction1");
  params.bias_correction2 = scalar(args.at(10), "bias_correction2");
  if (params.bias_correction1 == 0.0f || params.bias_correction2 == 0.0f) {
    throw std::runtime_error("adamw_step: bias corrections must be non-zero");
  }

  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kAdamwStepWorkgroupSizeX);
  const uint32_t workgroup_count = utils::compute_1d_workgroup_count(
      device, params.numel, wg_size, "adamw_step");

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(params));
  graph.add_uniform_buffer_bytes(sizeof(params));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kAdamwStepWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry entries[5] = {};
  for (uint32_t i = 0; i <= 2; i++) {
    entries[i].binding = i;
    entries[i].visibility = WGPUShaderStage_Compute;
    entries[i].buffer.type = WGPUBufferBindingType_Storage;
  }
  entries[3].binding = 3;
  entries[3].visibility = WGPUShaderStage_Compute;
  entries[3].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
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

  WGPUBindGroupEntry bg_entries[5] = {};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer = param.buffer;
  bg_entries[0].size = param.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = m.buffer;
  bg_entries[1].size = m.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = v.buffer;
  bg_entries[2].size = v.nbytes;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = grad.buffer;
  bg_entries[3].size = grad.nbytes;
  bg_entries[4].binding = 4;
  bg_entries[4].buffer = uniform_buffer;
  bg_entries[4].size = sizeof(params);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 5;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch({pipeline, bind_group, workgroup_count, "adamw_step"});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  graph.own_uniform_buffer(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.adamw_step.default, adamw_step_impl);
}

} // namespace executorch::backends::webgpu
