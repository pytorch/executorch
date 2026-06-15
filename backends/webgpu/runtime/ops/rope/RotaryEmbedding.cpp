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
#include <executorch/backends/webgpu/runtime/ops/rope/rotary_embedding_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching the WGSL Params struct (16-byte aligned, 32 bytes).
struct RotaryParams {
  uint32_t n_heads;
  uint32_t seq;
  uint32_t head_dim;
  uint32_t half_dim;
  uint32_t num_pairs;
  uint32_t _pad0;
  uint32_t _pad1;
  uint32_t _pad2;
};
static_assert(sizeof(RotaryParams) == 32, "RotaryParams must be 32 bytes");

uint64_t numel_of(const std::vector<int64_t>& dims) {
  uint64_t n = 1;
  for (int64_t d : dims) {
    n *= static_cast<uint64_t>(d);
  }
  return n;
}

// Rotate one (x->out) with the shared shader; freqs shared between xq and xk.
void add_rope_dispatch(
    WebGPUGraph& graph,
    WGPUDevice device,
    WGPUComputePipeline pipeline,
    WGPUBindGroupLayout bgl,
    const WebGPUTensor& x,
    const WebGPUTensor& out,
    const WebGPUTensor& freqs_cos,
    const WebGPUTensor& freqs_sin,
    uint32_t n_heads,
    uint32_t seq,
    uint32_t head_dim,
    uint32_t workgroup_count) {
  const uint32_t half_dim = head_dim / 2u;
  // out.dims == in.dims (asserted in impl), so this matches the caller's wgc.
  const uint32_t num_pairs = static_cast<uint32_t>(numel_of(out.dims) / 2u);

  RotaryParams params = {};
  params.n_heads = n_heads;
  params.seq = seq;
  params.head_dim = head_dim;
  params.half_dim = half_dim;
  params.num_pairs = num_pairs;

  WGPUBufferDescriptor uniform_desc = {};
  uniform_desc.size = sizeof(RotaryParams);
  uniform_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  uniform_desc.mappedAtCreation = true;
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(device, &uniform_desc);
  void* mapped =
      wgpuBufferGetMappedRange(uniform_buffer, 0, sizeof(RotaryParams));
  std::memcpy(mapped, &params, sizeof(RotaryParams));
  wgpuBufferUnmap(uniform_buffer);
  graph.add_uniform_buffer_bytes(sizeof(RotaryParams));

  WGPUBindGroupEntry bg_entries[5] = {};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer = out.buffer;
  bg_entries[0].size = out.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = x.buffer;
  bg_entries[1].size = x.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = freqs_cos.buffer;
  bg_entries[2].size = freqs_cos.nbytes;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = freqs_sin.buffer;
  bg_entries[3].size = freqs_sin.nbytes;
  bg_entries[4].binding = 4;
  bg_entries[4].buffer = uniform_buffer;
  bg_entries[4].size = sizeof(RotaryParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 5;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch(
      {pipeline, bind_group, workgroup_count, "apply_rotary_emb"});

  wgpuBufferRelease(uniform_buffer);
}

// args: [xq, xk, freqs_cos, freqs_sin, out_list(ValueList[xq_out, xk_out])].
void apply_rotary_emb_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int xq_id = args.at(0);
  const int xk_id = args.at(1);
  const int freqs_cos_id = args.at(2);
  const int freqs_sin_id = args.at(3);

  const std::vector<int>& out_list = graph.get_value_list(args.at(4));
  if (out_list.size() != 2) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: expected an output ValueList of size 2");
  }

  WGPUDevice device = graph.device();

  const auto& xq = graph.get_tensor(xq_id);
  const auto& xk = graph.get_tensor(xk_id);
  const auto& freqs_cos = graph.get_tensor(freqs_cos_id);
  const auto& freqs_sin = graph.get_tensor(freqs_sin_id);
  const auto& xq_out = graph.get_tensor(out_list[0]);
  const auto& xk_out = graph.get_tensor(out_list[1]);

  // Vulkan shape contract: xq/xk (B,S,n_heads,head_dim), freqs (S,head_dim/2).
  if (xq.dims.size() < 3 || xk.dims.size() < 3 || freqs_cos.dims.size() < 2) {
    throw std::runtime_error("WebGPU apply_rotary_emb: malformed dims");
  }
  const uint32_t head_dim = static_cast<uint32_t>(xq.dims.back());
  const uint32_t seq = static_cast<uint32_t>(xq.dims[xq.dims.size() - 3]);
  const uint32_t n_heads_q = static_cast<uint32_t>(xq.dims[xq.dims.size() - 2]);
  const uint32_t n_heads_k = static_cast<uint32_t>(xk.dims[xk.dims.size() - 2]);
  const uint32_t seq_k = static_cast<uint32_t>(xk.dims[xk.dims.size() - 3]);
  const uint32_t half_dim = static_cast<uint32_t>(freqs_cos.dims.back());

  if (head_dim == 0 || head_dim % 2 != 0) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: head_dim must be a nonzero multiple of 2");
  }
  if (static_cast<uint32_t>(xk.dims.back()) != head_dim || seq_k != seq) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: xq/xk head_dim and seq must match");
  }
  if (half_dim * 2u != head_dim) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: head_dim != 2 * freqs_cos last dim");
  }
  if (freqs_cos.dims != freqs_sin.dims) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: freqs_cos and freqs_sin shapes differ");
  }

  if (xq.buffer == nullptr || xk.buffer == nullptr ||
      freqs_cos.buffer == nullptr || freqs_sin.buffer == nullptr ||
      xq_out.buffer == nullptr || xk_out.buffer == nullptr) {
    throw std::runtime_error("WebGPU apply_rotary_emb: null buffer binding");
  }

  // All tensors are fp32; output shapes equal their inputs.
  const uint64_t xq_numel = numel_of(xq.dims);
  const uint64_t xk_numel = numel_of(xk.dims);
  const uint64_t freqs_numel = numel_of(freqs_cos.dims);
  if (freqs_numel != static_cast<uint64_t>(seq) * half_dim ||
      xq.nbytes != xq_numel * sizeof(float) ||
      xk.nbytes != xk_numel * sizeof(float) ||
      freqs_cos.nbytes != freqs_numel * sizeof(float) ||
      freqs_sin.nbytes != freqs_numel * sizeof(float) ||
      xq_out.nbytes != xq_numel * sizeof(float) ||
      xk_out.nbytes != xk_numel * sizeof(float)) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: dtype/byte-size mismatch (all fp32) or "
        "freqs shape != [seq, head_dim/2]");
  }

  if (xq_numel / 2u > UINT32_MAX || xk_numel / 2u > UINT32_MAX) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: pair count exceeds uint32 dispatch range");
  }

  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kRotaryEmbeddingWorkgroupSizeX);
  // Validate both dispatches before any GPU-object alloc (no leak on throw).
  const uint32_t xq_wgc = utils::compute_1d_workgroup_count(
      device,
      static_cast<uint32_t>(xq_numel / 2u),
      wg_size,
      "apply_rotary_emb");
  const uint32_t xk_wgc = utils::compute_1d_workgroup_count(
      device,
      static_cast<uint32_t>(xk_numel / 2u),
      wg_size,
      "apply_rotary_emb");

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kRotaryEmbeddingWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  // Bind group: out (rw) + in/freqs_cos/freqs_sin (ro) + uniform.
  WGPUBindGroupLayoutEntry entries[5] = {};
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_Storage;
  for (uint32_t i = 1; i <= 3; i++) {
    entries[i].binding = i;
    entries[i].visibility = WGPUShaderStage_Compute;
    entries[i].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  }
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
  // One pipeline per dispatch; a shared handle would double-free.
  WGPUComputePipeline pipeline_q =
      wgpuDeviceCreateComputePipeline(device, &pipeline_desc);
  WGPUComputePipeline pipeline_k =
      wgpuDeviceCreateComputePipeline(device, &pipeline_desc);

  add_rope_dispatch(
      graph,
      device,
      pipeline_q,
      bgl,
      xq,
      xq_out,
      freqs_cos,
      freqs_sin,
      n_heads_q,
      seq,
      head_dim,
      xq_wgc);
  add_rope_dispatch(
      graph,
      device,
      pipeline_k,
      bgl,
      xk,
      xk_out,
      freqs_cos,
      freqs_sin,
      n_heads_k,
      seq,
      head_dim,
      xk_wgc);

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // pipeline_q/pipeline_k owned by their dispatches; graph dtor frees.
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.apply_rotary_emb.default, apply_rotary_emb_impl);
}

} // namespace executorch::backends::webgpu
