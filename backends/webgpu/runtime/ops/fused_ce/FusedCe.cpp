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
#include <executorch/backends/webgpu/runtime/ops/fused_ce/fused_ce_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/reduce/reduce_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching the fused_ce.wgsl Params struct (16-byte aligned).
struct FusedCeParams {
  uint32_t vocab;
  uint32_t n_rows;
  float n_valid;
  float _pad0;
};
static_assert(sizeof(FusedCeParams) == 16, "FusedCeParams must be 16 bytes");

// Mirror reduce.wgsl Params (file-local in Reduce.cpp; re-declared here).
struct ReduceParams {
  uint32_t outer;
  uint32_t r;
  uint32_t inner;
  uint32_t is_mean;
};
static_assert(sizeof(ReduceParams) == 16, "ReduceParams must be 16 bytes");

WGPUShaderModule make_shader(WGPUDevice device, const char* wgsl) {
  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {wgsl, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  return wgpuDeviceCreateShaderModule(device, &shader_desc);
}

WGPUBuffer create_uniform(
    WebGPUGraph& graph,
    WGPUDevice device,
    const void* data,
    size_t size) {
  WGPUBufferDescriptor desc = {};
  desc.size = size;
  desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  desc.mappedAtCreation = true;
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &desc);
  std::memcpy(wgpuBufferGetMappedRange(buffer, 0, size), data, size);
  wgpuBufferUnmap(buffer);
  graph.add_uniform_buffer_bytes(size);
  return buffer;
}

// out valuelist packs the 2-tuple (loss, dlogits) as one id.
void fused_ce_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int logits_id = args.at(0);
  const int labels_id = args.at(1);
  const int n_valid_id = args.at(2);
  const std::vector<int>& outs = graph.get_value_list(args.at(3));
  if (outs.size() != 2) {
    throw std::runtime_error(
        "WebGPU fused_ce: expected 2 outputs (loss, dlogits)");
  }
  const int loss_id = outs.at(0);
  const int dlogits_id = outs.at(1);

  WGPUDevice device = graph.device();
  const auto& logits = graph.get_tensor(logits_id);
  const auto& labels = graph.get_tensor(labels_id);
  const auto& dlogits = graph.get_tensor(dlogits_id);
  const auto& loss = graph.get_tensor(loss_id);

  if (logits.dims.size() != 2) {
    throw std::runtime_error("WebGPU fused_ce: logits must be 2D [N, V]");
  }
  const uint64_t n_rows = static_cast<uint64_t>(logits.dims[0]);
  const uint64_t vocab = static_cast<uint64_t>(logits.dims[1]);
  const uint64_t numel = n_rows * vocab;

  if (dlogits.dims != logits.dims) {
    throw std::runtime_error(
        "WebGPU fused_ce: dlogits shape must match logits");
  }
  if (logits.nbytes != numel * sizeof(float) ||
      dlogits.nbytes != numel * sizeof(float)) {
    throw std::runtime_error("WebGPU fused_ce: logits/dlogits fp32-only");
  }
  if (labels.nbytes != n_rows * sizeof(int32_t)) {
    throw std::runtime_error("WebGPU fused_ce: labels must be int32 [N]");
  }
  if (loss.nbytes != sizeof(float)) {
    throw std::runtime_error("WebGPU fused_ce: loss must be a scalar [1]");
  }
  if (graph.get_value_type(n_valid_id) != WebGPUGraph::ValueType::Double) {
    throw std::runtime_error("WebGPU fused_ce: n_valid must be a float scalar");
  }
  const double n_valid = graph.get_double(n_valid_id);
  if (n_valid <= 0.0) {
    throw std::runtime_error("WebGPU fused_ce: n_valid must be positive");
  }
  if (n_rows > utils::queried_max_workgroups(device)) {
    throw std::runtime_error("WebGPU fused_ce: n_rows exceeds dispatch limit");
  }

  WGPUBuffer loss_partial = graph.create_scratch_buffer(n_rows * sizeof(float));

  // one workgroup per row
  FusedCeParams ce_params = {};
  ce_params.vocab = static_cast<uint32_t>(vocab);
  ce_params.n_rows = static_cast<uint32_t>(n_rows);
  ce_params.n_valid = static_cast<float>(n_valid);
  const uint32_t ce_wg =
      utils::clamp_workgroup_size(device, kFusedCeWorkgroupSizeX);
  WGPUBuffer ce_uniform =
      create_uniform(graph, device, &ce_params, sizeof(ce_params));
  WGPUShaderModule ce_shader = make_shader(device, kFusedCeWGSL);

  WGPUBindGroupLayoutEntry ce_entries[5] = {};
  ce_entries[0].binding = 0;
  ce_entries[0].visibility = WGPUShaderStage_Compute;
  ce_entries[0].buffer.type = WGPUBufferBindingType_Storage;
  ce_entries[1].binding = 1;
  ce_entries[1].visibility = WGPUShaderStage_Compute;
  ce_entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  ce_entries[2].binding = 2;
  ce_entries[2].visibility = WGPUShaderStage_Compute;
  ce_entries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  ce_entries[3].binding = 3;
  ce_entries[3].visibility = WGPUShaderStage_Compute;
  ce_entries[3].buffer.type = WGPUBufferBindingType_Storage;
  ce_entries[4].binding = 4;
  ce_entries[4].visibility = WGPUShaderStage_Compute;
  ce_entries[4].buffer.type = WGPUBufferBindingType_Uniform;

  WGPUBindGroupLayoutDescriptor ce_bgl_desc = {};
  ce_bgl_desc.entryCount = 5;
  ce_bgl_desc.entries = ce_entries;
  WGPUBindGroupLayout ce_bgl =
      wgpuDeviceCreateBindGroupLayout(device, &ce_bgl_desc);

  WGPUPipelineLayoutDescriptor ce_pl_desc = {};
  ce_pl_desc.bindGroupLayoutCount = 1;
  ce_pl_desc.bindGroupLayouts = &ce_bgl;
  WGPUPipelineLayout ce_pl =
      wgpuDeviceCreatePipelineLayout(device, &ce_pl_desc);

  WGPUConstantEntry ce_wg_const = {};
  ce_wg_const.key = {"wg_size", WGPU_STRLEN};
  ce_wg_const.value = static_cast<double>(ce_wg);

  WGPUComputePipelineDescriptor ce_pipe_desc = {};
  ce_pipe_desc.layout = ce_pl;
  ce_pipe_desc.compute.module = ce_shader;
  ce_pipe_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  ce_pipe_desc.compute.constantCount = 1;
  ce_pipe_desc.compute.constants = &ce_wg_const;
  WGPUComputePipeline ce_pipe =
      wgpuDeviceCreateComputePipeline(device, &ce_pipe_desc);

  WGPUBindGroupEntry ce_bg[5] = {};
  ce_bg[0].binding = 0;
  ce_bg[0].buffer = dlogits.buffer;
  ce_bg[0].size = dlogits.nbytes;
  ce_bg[1].binding = 1;
  ce_bg[1].buffer = logits.buffer;
  ce_bg[1].size = logits.nbytes;
  ce_bg[2].binding = 2;
  ce_bg[2].buffer = labels.buffer;
  ce_bg[2].size = labels.nbytes;
  ce_bg[3].binding = 3;
  ce_bg[3].buffer = loss_partial;
  ce_bg[3].size = n_rows * sizeof(float);
  ce_bg[4].binding = 4;
  ce_bg[4].buffer = ce_uniform;
  ce_bg[4].size = sizeof(ce_params);

  WGPUBindGroupDescriptor ce_bg_desc = {};
  ce_bg_desc.layout = ce_bgl;
  ce_bg_desc.entryCount = 5;
  ce_bg_desc.entries = ce_bg;
  WGPUBindGroup ce_bind_group = wgpuDeviceCreateBindGroup(device, &ce_bg_desc);

  graph.add_dispatch(
      {ce_pipe, ce_bind_group, static_cast<uint32_t>(n_rows), "fused_ce"});

  wgpuShaderModuleRelease(ce_shader);
  wgpuBindGroupLayoutRelease(ce_bgl);
  wgpuPipelineLayoutRelease(ce_pl);
  wgpuBufferRelease(ce_uniform);

  // reduce loss_partial[N] -> loss[1] (reuses reduce.wgsl)
  ReduceParams r_params = {};
  r_params.outer = 1u;
  r_params.r = static_cast<uint32_t>(n_rows);
  r_params.inner = 1u;
  r_params.is_mean = 0u; // mean already folded into loss_partial
  const uint32_t r_wg =
      utils::clamp_workgroup_size(device, kReduceWorkgroupSizeX);
  const uint32_t r_wgc =
      utils::compute_1d_workgroup_count(device, 1u, r_wg, "fused_ce_reduce");
  WGPUBuffer r_uniform =
      create_uniform(graph, device, &r_params, sizeof(r_params));
  WGPUShaderModule r_shader = make_shader(device, kReduceWGSL);

  WGPUBindGroupLayoutEntry r_entries[3] = {};
  r_entries[0].binding = 0;
  r_entries[0].visibility = WGPUShaderStage_Compute;
  r_entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  r_entries[1].binding = 1;
  r_entries[1].visibility = WGPUShaderStage_Compute;
  r_entries[1].buffer.type = WGPUBufferBindingType_Storage;
  r_entries[2].binding = 2;
  r_entries[2].visibility = WGPUShaderStage_Compute;
  r_entries[2].buffer.type = WGPUBufferBindingType_Uniform;

  WGPUBindGroupLayoutDescriptor r_bgl_desc = {};
  r_bgl_desc.entryCount = 3;
  r_bgl_desc.entries = r_entries;
  WGPUBindGroupLayout r_bgl =
      wgpuDeviceCreateBindGroupLayout(device, &r_bgl_desc);

  WGPUPipelineLayoutDescriptor r_pl_desc = {};
  r_pl_desc.bindGroupLayoutCount = 1;
  r_pl_desc.bindGroupLayouts = &r_bgl;
  WGPUPipelineLayout r_pl = wgpuDeviceCreatePipelineLayout(device, &r_pl_desc);

  WGPUConstantEntry r_wg_const = {};
  r_wg_const.key = {"wg_size", WGPU_STRLEN};
  r_wg_const.value = static_cast<double>(r_wg);

  WGPUComputePipelineDescriptor r_pipe_desc = {};
  r_pipe_desc.layout = r_pl;
  r_pipe_desc.compute.module = r_shader;
  r_pipe_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  r_pipe_desc.compute.constantCount = 1;
  r_pipe_desc.compute.constants = &r_wg_const;
  WGPUComputePipeline r_pipe =
      wgpuDeviceCreateComputePipeline(device, &r_pipe_desc);

  WGPUBindGroupEntry r_bg[3] = {};
  r_bg[0].binding = 0;
  r_bg[0].buffer = loss_partial;
  r_bg[0].size = n_rows * sizeof(float);
  r_bg[1].binding = 1;
  r_bg[1].buffer = loss.buffer;
  r_bg[1].size = loss.nbytes;
  r_bg[2].binding = 2;
  r_bg[2].buffer = r_uniform;
  r_bg[2].size = sizeof(r_params);

  WGPUBindGroupDescriptor r_bg_desc = {};
  r_bg_desc.layout = r_bgl;
  r_bg_desc.entryCount = 3;
  r_bg_desc.entries = r_bg;
  WGPUBindGroup r_bind_group = wgpuDeviceCreateBindGroup(device, &r_bg_desc);

  graph.add_dispatch({r_pipe, r_bind_group, r_wgc, "fused_ce_reduce"});

  wgpuShaderModuleRelease(r_shader);
  wgpuBindGroupLayoutRelease(r_bgl);
  wgpuPipelineLayoutRelease(r_pl);
  wgpuBufferRelease(r_uniform);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.fused_ce.default, fused_ce_impl);
}

} // namespace executorch::backends::webgpu
