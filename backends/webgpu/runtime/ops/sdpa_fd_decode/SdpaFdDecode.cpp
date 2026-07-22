/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Split-KV FlashDecoding decode dispatch (split + reduce passes).

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>
#include <executorch/backends/webgpu/runtime/ops/sdpa_fd_decode/SdpaFdDecode.h>
#include <executorch/backends/webgpu/runtime/ops/sdpa_fd_decode/sdpa_fd_reduce_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/sdpa_fd_decode/sdpa_fd_split_half_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/sdpa_fd_decode/sdpa_fd_split_wgsl.h>

#include <webgpu/webgpu.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace executorch::backends::webgpu {

namespace {

// MUST match the .wgsl: MAX_SPLITS and WG_SIZE*MAX_D_PER_LANE.
constexpr uint32_t kSdpaFdSplitTile = 64; // KV positions per split
constexpr uint32_t kSdpaFdMaxSplits = 128; // == MAX_SPLITS in both .wgsl files
// Public head-dim limit (kSdpaFdMaxHeadDim) must equal the kernel's lane-owns-D
// reach; tie them so a WG_SIZE change can't silently desync the Sdpa.cpp gate.
static_assert(
    kSdpaFdMaxHeadDim == kSdpaFdSplitWorkgroupSizeX * 2u,
    "kSdpaFdMaxHeadDim must match WG_SIZE * MAX_D_PER_LANE");

struct FdSplitParams {
  uint32_t _pad0; // 16B-alignment pad (head index derived from workgroup_id)
  uint32_t Hkv;
  uint32_t D;
  uint32_t context_len;
  uint32_t g;
  uint32_t num_splits;
  uint32_t split_len;
  float scale;
};
static_assert(sizeof(FdSplitParams) == 32, "FdSplitParams must be 32B");

struct FdReduceParams {
  uint32_t D;
  uint32_t num_splits;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(FdReduceParams) == 16, "FdReduceParams must be 16B");

struct BufferBinding {
  WGPUBuffer buffer;
  uint64_t size;
};

// Mirrors Sdpa.cpp build_dispatch; n_rw leading bindings are read_write.
void build_dispatch(
    WebGPUGraph& graph,
    const char* wgsl_source,
    const BufferBinding* storage_bindings,
    uint32_t n_storage,
    uint32_t n_rw,
    WGPUBuffer uniform_buffer,
    uint64_t uniform_size,
    uint32_t workgroup_count_x,
    bool retain_uniform,
    const char* kernel_name) {
  WGPUDevice device = graph.device();

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {wgsl_source, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  constexpr uint32_t kMaxEntries = 8;
  if (n_storage + 1u > kMaxEntries) {
    throw std::runtime_error(
        "WebGPU sdpa FlashDecoding: bind group entry count exceeds kMaxEntries");
  }
  WGPUBindGroupLayoutEntry bgl_entries[kMaxEntries] = {};
  const uint32_t uniform_binding = n_storage;
  for (uint32_t i = 0; i < n_storage; i++) {
    bgl_entries[i].binding = i;
    bgl_entries[i].visibility = WGPUShaderStage_Compute;
    bgl_entries[i].buffer.type = (i < n_rw)
        ? WGPUBufferBindingType_Storage
        : WGPUBufferBindingType_ReadOnlyStorage;
  }
  bgl_entries[uniform_binding].binding = uniform_binding;
  bgl_entries[uniform_binding].visibility = WGPUShaderStage_Compute;
  bgl_entries[uniform_binding].buffer.type = WGPUBufferBindingType_Uniform;

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = n_storage + 1;
  bgl_desc.entries = bgl_entries;
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
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device, &pipeline_desc);

  WGPUBindGroupEntry bg_entries[kMaxEntries] = {};
  for (uint32_t i = 0; i < n_storage; i++) {
    bg_entries[i].binding = i;
    bg_entries[i].buffer = storage_bindings[i].buffer;
    bg_entries[i].size = storage_bindings[i].size;
  }
  bg_entries[uniform_binding].binding = uniform_binding;
  bg_entries[uniform_binding].buffer = uniform_buffer;
  bg_entries[uniform_binding].size = uniform_size;

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = n_storage + 1;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch({pipeline, bind_group, workgroup_count_x, kernel_name});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  if (retain_uniform) {
    graph.own_uniform_buffer(uniform_buffer);
  } else {
    wgpuBufferRelease(uniform_buffer);
  }
}

FdSplitParams make_split_params(const SdpaFdDecodeState& state) {
  FdSplitParams params = {};
  params.Hkv = state.Hkv;
  params.D = state.D;
  params.context_len = state.context_len;
  params.g = state.g;
  params.num_splits = state.num_splits;
  params.split_len = state.split_len;
  params.scale = state.scale;
  return params;
}

FdReduceParams make_reduce_params(const SdpaFdDecodeState& state) {
  FdReduceParams params = {};
  params.D = state.D;
  params.num_splits = state.num_splits;
  return params;
}

} // namespace

SdpaFdDecodeState make_sdpa_fd_decode_state(
    WGPUDevice device,
    int64_t Hq,
    int64_t Hkv,
    int64_t D,
    int64_t context_len,
    int64_t g,
    float scale) {
  if (Hq <= 0 || Hkv <= 0 || D <= 0 || context_len <= 0 || g <= 0) {
    throw std::runtime_error(
        "WebGPU sdpa FlashDecoding: dimensions must be positive");
  }
  if (Hq != Hkv * g) {
    throw std::runtime_error(
        "WebGPU sdpa FlashDecoding: inconsistent GQA dimensions");
  }
  if (Hq > UINT32_MAX || Hkv > UINT32_MAX || D > UINT32_MAX ||
      context_len > UINT32_MAX || g > UINT32_MAX) {
    throw std::runtime_error(
        "WebGPU sdpa FlashDecoding: parameter exceeds uint32 max");
  }
  if (D > kSdpaFdMaxHeadDim) {
    throw std::runtime_error(
        "WebGPU sdpa FlashDecoding: head dim must be <= " +
        std::to_string(kSdpaFdMaxHeadDim));
  }
  if (D % 4 != 0) {
    throw std::runtime_error(
        "WebGPU sdpa FlashDecoding: head dim must be a multiple of 4");
  }

  uint32_t num_splits = static_cast<uint32_t>(
      (context_len + kSdpaFdSplitTile - 1) / kSdpaFdSplitTile);
  num_splits = std::min(num_splits, kSdpaFdMaxSplits);
  const uint32_t split_len =
      static_cast<uint32_t>((context_len + num_splits - 1) / num_splits);

  const uint64_t split_threads = static_cast<uint64_t>(Hq) *
      static_cast<uint64_t>(num_splits) *
      static_cast<uint64_t>(kSdpaFdSplitWorkgroupSizeX);
  const uint64_t reduce_threads =
      static_cast<uint64_t>(Hq) * kSdpaFdReduceWorkgroupSizeX;
  if (split_threads > UINT32_MAX || reduce_threads > UINT32_MAX) {
    throw std::runtime_error(
        "WebGPU sdpa FlashDecoding: thread count exceeds uint32 max");
  }

  const uint32_t split_wgc = utils::compute_1d_workgroup_count(
      device,
      static_cast<uint32_t>(split_threads),
      kSdpaFdSplitWorkgroupSizeX,
      "fd_split");
  const uint32_t reduce_wgc = utils::compute_1d_workgroup_count(
      device,
      static_cast<uint32_t>(reduce_threads),
      kSdpaFdReduceWorkgroupSizeX,
      "fd_reduce");
  return {
      static_cast<uint32_t>(Hq),
      static_cast<uint32_t>(Hkv),
      static_cast<uint32_t>(D),
      static_cast<uint32_t>(context_len),
      static_cast<uint32_t>(g),
      num_splits,
      split_len,
      scale,
      {split_wgc, 1u},
      {reduce_wgc, 1u}};
}

SdpaFdDecodeResources record_sdpa_fd_decode_dispatches(
    WebGPUGraph& graph,
    const WebGPUTensor& q,
    const WebGPUTensor& k_cache,
    const WebGPUTensor& v_cache,
    const WebGPUTensor& out,
    const SdpaFdDecodeState& state) {
  const size_t dispatch_begin = graph.num_dispatches();

  // Scratch: per-(head,split) partials at kSdpaFdMaxSplits stride.
  const uint64_t po_floats = static_cast<uint64_t>(state.Hq) *
      static_cast<uint64_t>(kSdpaFdMaxSplits) * static_cast<uint64_t>(state.D);
  const uint64_t pml_floats = static_cast<uint64_t>(state.Hq) *
      static_cast<uint64_t>(kSdpaFdMaxSplits) * 2ull;
  WGPUBuffer part_o = graph.acquire_scratch(po_floats * sizeof(float));
  WebGPUGraph::ScopedScratch part_o_guard(&graph, part_o);
  WGPUBuffer part_ml = graph.acquire_scratch(pml_floats * sizeof(float));
  WebGPUGraph::ScopedScratch part_ml_guard(&graph, part_ml);

  // Pass 1: split (Hq*num_splits WGs) -> writes part_o, part_ml.
  FdSplitParams sp = make_split_params(state);
  WGPUBuffer ub_split = graph.make_uniform_buffer(&sp, sizeof(sp));
  BufferBinding split_bindings[5] = {
      {part_o, po_floats * sizeof(float)},
      {part_ml, pml_floats * sizeof(float)},
      {q.buffer, q.nbytes},
      {k_cache.buffer, k_cache.nbytes},
      {v_cache.buffer, v_cache.nbytes}};
  const char* split_shader = kSdpaFdSplitWGSL;
  if (graph.kv_f16()) {
    split_shader = kSdpaFdSplitHalfWGSL;
  }
  build_dispatch(
      graph,
      split_shader,
      split_bindings,
      5,
      2,
      ub_split,
      sizeof(sp),
      state.split_grid.x,
      true,
      "fd_split");

  // Pass 2: reduce (Hq WGs) -> reads part_o, part_ml; writes out.
  FdReduceParams rp = make_reduce_params(state);
  WGPUBuffer ub_reduce = graph.make_uniform_buffer(&rp, sizeof(rp));
  BufferBinding reduce_bindings[3] = {
      {out.buffer, out.nbytes},
      {part_o, po_floats * sizeof(float)},
      {part_ml, pml_floats * sizeof(float)}};
  build_dispatch(
      graph,
      kSdpaFdReduceWGSL,
      reduce_bindings,
      3,
      1,
      ub_reduce,
      sizeof(rp),
      state.reduce_grid.x,
      true,
      "fd_reduce");

  return {ub_split, ub_reduce, {dispatch_begin, graph.num_dispatches()}};
}

void write_sdpa_fd_decode_uniforms(
    WGPUQueue queue,
    const SdpaFdDecodeResources& resources,
    const SdpaFdDecodeState& state) {
  FdSplitParams split_params = make_split_params(state);
  FdReduceParams reduce_params = make_reduce_params(state);
  wgpuQueueWriteBuffer(
      queue, resources.split_uniform, 0, &split_params, sizeof(split_params));
  wgpuQueueWriteBuffer(
      queue,
      resources.reduce_uniform,
      0,
      &reduce_params,
      sizeof(reduce_params));
}

} // namespace executorch::backends::webgpu
